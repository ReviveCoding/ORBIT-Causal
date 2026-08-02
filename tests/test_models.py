from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from orbit_causal.config import DataConfig, ModelConfig
from orbit_causal.data import generate_synthetic_data
from orbit_causal.models import (
    CrossFittedDRLearner,
    ReferenceStateEffectCalibrator,
    TwoModelUplift,
    doubly_robust_score,
)
from orbit_causal.preprocessing import StableTabularPreprocessor


class ModelTests(unittest.TestCase):
    frame: pd.DataFrame
    data_config: DataConfig
    model_config: ModelConfig

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = generate_synthetic_data(2000, 44)
        cls.data_config = DataConfig()
        cls.model_config = ModelConfig(
            max_iter=20, max_depth=3, min_samples_leaf=20, dr_folds=2, ensemble_size=2
        )

    def test_two_model_effect_shape(self) -> None:
        model = TwoModelUplift(self.data_config, self.model_config, 1).fit(self.frame)
        effect = model.predict_effect(self.frame.iloc[:25])
        self.assertEqual(effect.shape, (25,))
        self.assertTrue(np.isfinite(effect).all())

    def test_dr_and_rsec_are_independent_layers(self) -> None:
        train = self.frame.iloc[:1400]
        calibration = self.frame.iloc[1400:1800]
        dr = CrossFittedDRLearner(self.data_config, self.model_config, 2).fit(train)
        before = dr.predict_effect(self.frame.iloc[1800:])
        rsec = ReferenceStateEffectCalibrator(dr, self.model_config, 3).fit(
            calibration, "treatment", "visit"
        )
        after = rsec.predict_effect(self.frame.iloc[1800:])
        self.assertTrue(np.isfinite(after).all())
        self.assertGreater(float(np.std(rsec.predict_correction(self.frame.iloc[1800:]))), 0.0)
        self.assertTrue(np.allclose(before, dr.predict_effect(self.frame.iloc[1800:])))

    def test_insufficient_joint_support_has_clear_error(self) -> None:
        broken = self.frame.copy()
        broken.loc[broken["treatment"] == 0, "visit"] = 0
        with self.assertRaisesRegex(ValueError, "insufficient joint treatment/outcome support"):
            TwoModelUplift(self.data_config, self.model_config, 1).fit(broken)

    def test_direct_model_fit_rejects_fractional_binary_values_before_casting(self) -> None:
        broken = self.frame.copy()
        broken["treatment"] = broken["treatment"].astype(float)
        broken.loc[broken.index[0], "treatment"] = 0.5
        with self.assertRaisesRegex(ValueError, "treatment must be binary"):
            TwoModelUplift(self.data_config, self.model_config, 1).fit(broken)

    def test_failed_refit_invalidates_previous_fitted_state(self) -> None:
        model = TwoModelUplift(self.data_config, self.model_config, 1).fit(self.frame)
        self.assertEqual(model.predict_effect(self.frame.iloc[:2]).shape, (2,))
        broken = self.frame.copy()
        broken["treatment"] = 0
        with self.assertRaisesRegex(ValueError, "both treatment arms"):
            model.fit(broken)
        with self.assertRaises(AttributeError):
            model.predict_effect(self.frame.iloc[:2])

    def test_cross_fitted_preprocessing_is_learned_inside_each_fold(self) -> None:
        frame = self.frame.iloc[:1000]
        fit_sizes: list[int] = []
        original_fit = StableTabularPreprocessor.fit

        def tracked_fit(instance, part):
            fit_sizes.append(len(part))
            return original_fit(instance, part)

        with patch("orbit_causal.models.StableTabularPreprocessor.fit", tracked_fit):
            model = CrossFittedDRLearner(self.data_config, self.model_config, 2).fit(frame)
        self.assertEqual(len(fit_sizes), self.model_config.dr_folds + 1)
        self.assertEqual(fit_sizes.count(len(frame)), 1)
        self.assertEqual(sum(size < len(frame) for size in fit_sizes), self.model_config.dr_folds)
        self.assertEqual(len(model.fold_propensities_), self.model_config.dr_folds)
        self.assertTrue(all(0 < value < 1 for value in model.fold_propensities_))

    def test_dr_score_rejects_broadcasting_and_invalid_propensity(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            doubly_robust_score(
                np.array([0.0, 1.0]),
                np.array([0, 1]),
                np.array([0.2]),
                np.array([0.4, 0.5]),
                0.5,
            )
        with self.assertRaisesRegex(ValueError, "propensity"):
            doubly_robust_score(
                np.array([0.0, 1.0]),
                np.array([0, 1]),
                np.array([0.2, 0.3]),
                np.array([0.4, 0.5]),
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
