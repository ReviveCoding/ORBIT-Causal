from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import _categorical_token


@dataclass
class StableTabularPreprocessor:
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categorical_hash_bins: int = 64

    def fit(self, frame: pd.DataFrame) -> StableTabularPreprocessor:
        for name in ("continuous_columns_", "medians_", "categorical_mask_"):
            if hasattr(self, name):
                delattr(self, name)
        self._validate_contract()
        self._validate_columns(frame)
        continuous = [c for c in self.feature_columns if c not in self.categorical_columns]
        numeric_columns = {
            column: self._numeric_column(frame[column], column) for column in continuous
        }
        all_missing = [
            column for column, values in numeric_columns.items() if not values.notna().any()
        ]
        if all_missing:
            raise ValueError(f"continuous features are entirely missing: {all_missing}")
        medians = {column: float(values.median()) for column, values in numeric_columns.items()}
        categorical_mask = np.array(
            [column in self.categorical_columns for column in self.feature_columns], dtype=bool
        )
        self.continuous_columns_ = tuple(continuous)
        self.medians_ = medians
        self.categorical_mask_ = categorical_mask
        return self

    def _validate_contract(self) -> None:
        if not self.feature_columns or any(
            not isinstance(column, str) or not column for column in self.feature_columns
        ):
            raise ValueError("feature_columns must contain nonempty strings")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be unique")
        if len(set(self.categorical_columns)) != len(self.categorical_columns):
            raise ValueError("categorical_columns must be unique")
        if not set(self.categorical_columns).issubset(self.feature_columns):
            raise ValueError("categorical_columns must be a subset of feature_columns")
        if (
            not isinstance(self.categorical_hash_bins, int)
            or isinstance(self.categorical_hash_bins, bool)
            or not 2 <= self.categorical_hash_bins <= 255
        ):
            raise ValueError("categorical_hash_bins must be an integer in [2, 255]")

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
        if duplicate_columns:
            raise ValueError(f"duplicate model feature columns: {duplicate_columns}")
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"missing model features: {missing}")

    @staticmethod
    def _numeric_column(series: pd.Series, column: str) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        invalid = series.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(
                f"continuous feature {column!r} contains {int(invalid.sum())} nonnumeric values"
            )
        values = numeric.to_numpy()
        if np.iscomplexobj(values):
            raise ValueError(f"continuous feature {column!r} contains complex values")
        if np.isinf(values.astype(float)).any():
            raise ValueError(f"continuous feature {column!r} contains infinite values")
        return numeric

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "medians_"):
            raise RuntimeError("preprocessor must be fitted")
        self._validate_columns(frame)
        columns: list[np.ndarray] = []
        for column in self.feature_columns:
            if column in self.categorical_columns:
                series = frame[column]
                missing = series.isna().to_numpy()
                canonical = series.map(_categorical_token)
                encoded = (
                    pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype=np.uint64)
                    % np.uint64(self.categorical_hash_bins)
                ).astype(float)
                encoded[missing] = -1.0
                columns.append(encoded)
            else:
                numeric = self._numeric_column(frame[column], column).fillna(self.medians_[column])
                columns.append(numeric.to_numpy(dtype=float))
        matrix = np.column_stack(columns)
        if not np.isfinite(matrix[:, ~self.categorical_mask_]).all():
            raise ValueError("nonfinite continuous feature after preprocessing")
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


@dataclass
class RobustStateScaler:
    """Leakage-safe density proxy for RSEC state construction."""

    def fit(self, matrix: np.ndarray, categorical_mask: np.ndarray) -> RobustStateScaler:
        for name in ("center_", "scale_", "categorical_mask_"):
            if hasattr(self, name):
                delattr(self, name)
        raw_matrix = np.asarray(matrix)
        if np.iscomplexobj(raw_matrix):
            raise ValueError("state matrix must be real-valued")
        matrix = raw_matrix.astype(float)
        categorical_mask = np.asarray(categorical_mask)
        if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] == 0:
            raise ValueError("state matrix must be a nonempty two-dimensional array")
        if (
            categorical_mask.ndim != 1
            or len(categorical_mask) != matrix.shape[1]
            or categorical_mask.dtype != bool
        ):
            raise ValueError("categorical_mask must be an aligned boolean vector")
        if not np.isfinite(matrix).all():
            raise ValueError("state matrix must contain only finite values")
        continuous = matrix[:, ~categorical_mask]
        if continuous.shape[1] == 0:
            center = np.empty(0, dtype=float)
            scale = np.empty(0, dtype=float)
            self.categorical_mask_ = categorical_mask.copy()
            self.center_ = center
            self.scale_ = scale
            return self
        center = np.nanmedian(continuous, axis=0)
        q75 = np.nanpercentile(continuous, 75, axis=0)
        q25 = np.nanpercentile(continuous, 25, axis=0)
        scale = np.maximum(q75 - q25, 1e-6)
        self.center_ = center
        self.scale_ = scale
        self.categorical_mask_ = categorical_mask.copy()
        return self

    def score(self, matrix: np.ndarray) -> np.ndarray:
        if not hasattr(self, "categorical_mask_"):
            raise RuntimeError("state scaler must be fitted")
        raw_matrix = np.asarray(matrix)
        if np.iscomplexobj(raw_matrix):
            raise ValueError("state matrix must be real-valued")
        matrix = raw_matrix.astype(float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.categorical_mask_):
            raise ValueError("state matrix shape mismatch")
        if not np.isfinite(matrix).all():
            raise ValueError("state matrix must contain only finite values")
        continuous = matrix[:, ~self.categorical_mask_]
        if continuous.shape[1] == 0:
            return np.zeros(len(matrix), dtype=float)
        distance = np.mean(np.abs((continuous - self.center_) / self.scale_), axis=1)
        return np.clip(distance, 0.0, 20.0)
