from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from .config import DataConfig, ModelConfig
from .preprocessing import RobustStateScaler, StableTabularPreprocessor


def _clear_fitted_attributes(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        if hasattr(instance, name):
            delattr(instance, name)


class EffectModel(Protocol):
    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray: ...

    def predict_uncertainty(self, frame: pd.DataFrame) -> np.ndarray: ...

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: ...


def _classifier(config: ModelConfig, seed: int, categorical_mask: np.ndarray):
    return HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        max_bins=255,
        random_state=seed,
    )


def _regressor(config: ModelConfig, seed: int, max_depth: int | None = None):
    return HistGradientBoostingRegressor(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_depth=max_depth if max_depth is not None else config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=2.0,
        random_state=seed,
    )


def _safe_probability(model, matrix: np.ndarray) -> np.ndarray:
    probability = model.predict_proba(matrix)[:, 1]
    return np.clip(probability, 1e-5, 1 - 1e-5)


def _validate_training_support(
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    minimum_joint_count: int = 1,
) -> None:
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
    if treatment.ndim != 1 or outcome.ndim != 1 or len(treatment) != len(outcome):
        raise ValueError("training treatment and outcome must be aligned vectors")
    if len(treatment) == 0:
        raise ValueError("training frame must contain at least one row")
    if np.iscomplexobj(treatment) or np.iscomplexobj(outcome):
        raise ValueError("training treatment and outcome must be real-valued")
    if (
        not np.isfinite(treatment.astype(float)).all()
        or not np.isfinite(outcome.astype(float)).all()
    ):
        raise ValueError("training treatment and outcome must be finite")
    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("training treatment must be binary")
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("both treatment arms are required")
    if not np.isin(outcome, [0, 1]).all():
        raise ValueError("binary outcome values 0 and 1 are required")
    counts = {
        (arm, label): int(np.sum((treatment == arm) & (outcome == label)))
        for arm in (0, 1)
        for label in (0, 1)
    }
    insufficient = {
        f"treatment={arm},outcome={label}": count
        for (arm, label), count in counts.items()
        if count < minimum_joint_count
    }
    if insufficient:
        raise ValueError(
            "insufficient joint treatment/outcome support: "
            f"required>={minimum_joint_count}, observed={insufficient}"
        )


@dataclass
class TwoModelUplift:
    data_config: DataConfig
    model_config: ModelConfig
    seed: int

    def fit(self, frame: pd.DataFrame) -> TwoModelUplift:
        fitted = ("preprocessor_", "model0_", "model1_", "propensity_", "state_scaler_")
        _clear_fitted_attributes(self, fitted)
        x = frame.loc[:, self.data_config.feature_columns]
        treatment = frame[self.data_config.treatment].to_numpy()
        outcome = frame[self.data_config.outcome].to_numpy()
        _validate_training_support(treatment, outcome)
        treatment = treatment.astype(int)
        outcome = outcome.astype(int)
        preprocessor = StableTabularPreprocessor(
            self.data_config.feature_columns,
            self.data_config.categorical_columns,
            self.model_config.categorical_hash_bins,
        )
        matrix = preprocessor.fit_transform(x)
        model0 = _classifier(self.model_config, self.seed, preprocessor.categorical_mask_).fit(
            matrix[treatment == 0], outcome[treatment == 0]
        )
        model1 = _classifier(self.model_config, self.seed + 1, preprocessor.categorical_mask_).fit(
            matrix[treatment == 1], outcome[treatment == 1]
        )
        propensity = float(np.mean(treatment))
        state_scaler = RobustStateScaler().fit(matrix, preprocessor.categorical_mask_)
        self.preprocessor_ = preprocessor
        self.model0_ = model0
        self.model1_ = model1
        self.propensity_ = propensity
        self.state_scaler_ = state_scaler
        return self

    def representation(self, frame: pd.DataFrame) -> np.ndarray:
        return self.preprocessor_.transform(frame.loc[:, self.data_config.feature_columns])

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.representation(frame)
        return _safe_probability(self.model0_, matrix), _safe_probability(self.model1_, matrix)

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        mu0, mu1 = self.predict_outcomes(frame)
        return mu1 - mu0

    def predict_uncertainty(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame), dtype=float)

    def density_score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.state_scaler_.score(self.representation(frame))


@dataclass
class CrossFittedDRLearner:
    data_config: DataConfig
    model_config: ModelConfig
    seed: int

    def fit(self, frame: pd.DataFrame) -> CrossFittedDRLearner:
        fitted = (
            "preprocessor_",
            "propensity_",
            "fold_propensities_",
            "fold_models_",
            "final_models_",
            "model0_",
            "model1_",
            "state_scaler_",
            "training_dr_mean_",
            "training_dr_std_",
        )
        _clear_fitted_attributes(self, fitted)
        x_frame = frame.loc[:, self.data_config.feature_columns]
        treatment = frame[self.data_config.treatment].to_numpy()
        outcome = frame[self.data_config.outcome].to_numpy()
        _validate_training_support(
            treatment,
            outcome,
            minimum_joint_count=self.model_config.dr_folds,
        )
        treatment = treatment.astype(int)
        outcome = outcome.astype(float)
        propensity = float(np.clip(np.mean(treatment), 0.02, 0.98))
        strata = treatment.astype(str) + outcome.astype(int).astype(str)
        folds = StratifiedKFold(
            n_splits=self.model_config.dr_folds, shuffle=True, random_state=self.seed
        )
        mu0_oof = np.zeros(len(frame), dtype=float)
        mu1_oof = np.zeros(len(frame), dtype=float)
        propensity_oof = np.zeros(len(frame), dtype=float)
        fold_models: list[tuple[StableTabularPreprocessor, object, object]] = []
        fold_propensities: list[float] = []
        for fold_id, (fit_index, hold_index) in enumerate(folds.split(x_frame, strata)):
            fold_preprocessor = StableTabularPreprocessor(
                self.data_config.feature_columns,
                self.data_config.categorical_columns,
                self.model_config.categorical_hash_bins,
            )
            fit_matrix = fold_preprocessor.fit_transform(x_frame.iloc[fit_index])
            hold_matrix = fold_preprocessor.transform(x_frame.iloc[hold_index])
            fit_t = treatment[fit_index]
            fit_y = outcome[fit_index]
            fold_propensity = float(np.clip(np.mean(fit_t), 0.02, 0.98))
            model0 = _classifier(
                self.model_config,
                self.seed + 10 * fold_id,
                fold_preprocessor.categorical_mask_,
            ).fit(fit_matrix[fit_t == 0], fit_y[fit_t == 0])
            model1 = _classifier(
                self.model_config,
                self.seed + 10 * fold_id + 1,
                fold_preprocessor.categorical_mask_,
            ).fit(fit_matrix[fit_t == 1], fit_y[fit_t == 1])
            mu0_oof[hold_index] = _safe_probability(model0, hold_matrix)
            mu1_oof[hold_index] = _safe_probability(model1, hold_matrix)
            propensity_oof[hold_index] = fold_propensity
            fold_models.append((fold_preprocessor, model0, model1))
            fold_propensities.append(fold_propensity)

        gamma = doubly_robust_score(outcome, treatment, mu0_oof, mu1_oof, propensity_oof)
        preprocessor = StableTabularPreprocessor(
            self.data_config.feature_columns,
            self.data_config.categorical_columns,
            self.model_config.categorical_hash_bins,
        )
        matrix = preprocessor.fit_transform(x_frame)
        final_models: list[HistGradientBoostingRegressor] = []
        rng = np.random.default_rng(self.seed + 700)
        for ensemble_id in range(self.model_config.ensemble_size):
            sample = rng.integers(0, len(frame), size=len(frame))
            final_model = _regressor(self.model_config, self.seed + 1000 + ensemble_id).fit(
                matrix[sample], gamma[sample]
            )
            final_models.append(final_model)

        model0 = _classifier(
            self.model_config, self.seed + 2001, preprocessor.categorical_mask_
        ).fit(matrix[treatment == 0], outcome[treatment == 0])
        model1 = _classifier(
            self.model_config, self.seed + 2002, preprocessor.categorical_mask_
        ).fit(matrix[treatment == 1], outcome[treatment == 1])
        state_scaler = RobustStateScaler().fit(matrix, preprocessor.categorical_mask_)
        self.preprocessor_ = preprocessor
        self.propensity_ = propensity
        self.fold_propensities_ = tuple(fold_propensities)
        self.fold_models_ = fold_models
        self.final_models_ = final_models
        self.model0_ = model0
        self.model1_ = model1
        self.state_scaler_ = state_scaler
        self.training_dr_mean_ = float(np.mean(gamma))
        self.training_dr_std_ = float(np.std(gamma))
        return self

    def representation(self, frame: pd.DataFrame) -> np.ndarray:
        return self.preprocessor_.transform(frame.loc[:, self.data_config.feature_columns])

    def predict_members(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.representation(frame)
        return np.column_stack([model.predict(matrix) for model in self.final_models_])

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        return np.mean(self.predict_members(frame), axis=1)

    def predict_uncertainty(self, frame: pd.DataFrame) -> np.ndarray:
        members = self.predict_members(frame)
        return np.maximum(np.std(members, axis=1, ddof=1), 1e-4)

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.representation(frame)
        return _safe_probability(self.model0_, matrix), _safe_probability(self.model1_, matrix)

    def density_score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.state_scaler_.score(self.representation(frame))


def doubly_robust_score(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float | np.ndarray,
) -> np.ndarray:
    arrays = [np.asarray(value) for value in (outcome, treatment, mu0, mu1)]
    if any(array.ndim != 1 for array in arrays) or not arrays[0].size:
        raise ValueError("DR inputs must be nonempty one-dimensional arrays")
    if any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("DR inputs must be aligned")
    if any(np.iscomplexobj(array) for array in arrays):
        raise ValueError("DR inputs must be real-valued")
    y, t, m0, m1 = (array.astype(float) for array in arrays)
    if not all(np.isfinite(array).all() for array in (y, t, m0, m1)):
        raise ValueError("DR inputs must be finite")
    if not np.isin(t, [0, 1]).all():
        raise ValueError("DR treatment must be binary")
    raw_propensity = np.asarray(propensity)
    if np.iscomplexobj(raw_propensity):
        raise ValueError("propensity must be real-valued")
    p = raw_propensity.astype(float)
    if p.ndim > 1 or (p.ndim == 1 and len(p) not in {1, len(y)}):
        raise ValueError("propensity must be scalar or aligned")
    if not np.isfinite(p).all() or (p <= 0).any() or (p >= 1).any():
        raise ValueError("propensity must be finite and in (0, 1)")
    p = np.clip(p, 0.02, 0.98)
    return m1 - m0 + t * (y - m1) / p - (1 - t) * (y - m0) / (1 - p)


def _validate_calibration_support(treatment: np.ndarray, outcome: np.ndarray) -> None:
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
    if treatment.ndim != 1 or outcome.ndim != 1 or len(treatment) != len(outcome):
        raise ValueError("calibration treatment and outcome must be aligned vectors")
    if np.iscomplexobj(treatment) or np.iscomplexobj(outcome):
        raise ValueError("calibration treatment and outcome must be real-valued")
    if (
        not np.isfinite(treatment.astype(float)).all()
        or not np.isfinite(outcome.astype(float)).all()
    ):
        raise ValueError("calibration treatment and outcome must be finite")
    if len(treatment) < 2 or set(np.unique(treatment)) != {0, 1}:
        raise ValueError("calibration requires both treatment arms")
    if not np.isin(outcome, [0, 1]).all():
        raise ValueError("calibration outcome must be binary")


@dataclass
class CausalIsotonicCalibrator:
    base_model: EffectModel

    def fit(
        self, frame: pd.DataFrame, treatment_column: str, outcome_column: str
    ) -> CausalIsotonicCalibrator:
        _clear_fitted_attributes(self, ("model_", "residual_scale_"))
        tau = self.base_model.predict_effect(frame)
        mu0, mu1 = self.base_model.predict_outcomes(frame)
        treatment = frame[treatment_column].to_numpy()
        outcome = frame[outcome_column].to_numpy()
        _validate_calibration_support(treatment, outcome)
        treatment = treatment.astype(int)
        outcome = outcome.astype(float)
        propensity = float(np.mean(treatment))
        gamma = doubly_robust_score(outcome, treatment, mu0, mu1, propensity)
        model = IsotonicRegression(out_of_bounds="clip").fit(tau, gamma)
        residual_scale = float(np.sqrt(np.mean((gamma - model.predict(tau)) ** 2)))
        self.model_ = model
        self.residual_scale_ = residual_scale
        return self

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.base_model.predict_effect(frame))

    def predict_uncertainty(self, frame: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict_uncertainty(frame)

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.base_model.predict_outcomes(frame)


@dataclass
class ReferenceStateEffectCalibrator:
    """PHT-inspired residual calibration in a causal reliability state space."""

    base_model: CrossFittedDRLearner
    model_config: ModelConfig
    seed: int

    def _state(self, frame: pd.DataFrame) -> np.ndarray:
        mu0, mu1 = self.base_model.predict_outcomes(frame)
        tau = self.base_model.predict_effect(frame)
        uncertainty = self.base_model.predict_uncertainty(frame)
        density = self.base_model.density_score(frame)
        overlap = np.full(
            len(frame), min(self.base_model.propensity_, 1 - self.base_model.propensity_)
        )
        return np.column_stack([mu0, mu1, tau, overlap, uncertainty, density])

    def fit(
        self, frame: pd.DataFrame, treatment_column: str, outcome_column: str
    ) -> ReferenceStateEffectCalibrator:
        _clear_fitted_attributes(
            self,
            ("correction_model_", "correction_cap_", "residual_scale_"),
        )
        treatment = frame[treatment_column].to_numpy()
        outcome = frame[outcome_column].to_numpy()
        _validate_calibration_support(treatment, outcome)
        treatment = treatment.astype(int)
        outcome = outcome.astype(float)
        mu0, mu1 = self.base_model.predict_outcomes(frame)
        tau = self.base_model.predict_effect(frame)
        gamma = doubly_robust_score(outcome, treatment, mu0, mu1, self.base_model.propensity_)
        residual = gamma - tau
        state = self._state(frame)
        correction_model = _regressor(self.model_config, self.seed, max_depth=2).fit(
            state, residual
        )
        raw_correction = correction_model.predict(state)
        correction_cap = float(np.quantile(np.abs(raw_correction), 0.95))
        residual_scale = float(np.sqrt(np.mean((residual - raw_correction) ** 2)))
        self.correction_model_ = correction_model
        self.correction_cap_ = correction_cap
        self.residual_scale_ = residual_scale
        return self

    def predict_correction(self, frame: pd.DataFrame) -> np.ndarray:
        correction = self.correction_model_.predict(self._state(frame))
        return np.clip(correction, -self.correction_cap_, self.correction_cap_)

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict_effect(frame) + self.predict_correction(frame)

    def predict_uncertainty(self, frame: pd.DataFrame) -> np.ndarray:
        base = self.base_model.predict_uncertainty(frame)
        calibration_component = 0.05 * self.residual_scale_
        return np.sqrt(base**2 + calibration_component**2)

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.base_model.predict_outcomes(frame)
