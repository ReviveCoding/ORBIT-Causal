from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from orbit_causal.preprocessing import RobustStateScaler, StableTabularPreprocessor


class PreprocessingTests(unittest.TestCase):
    def test_failed_refit_invalidates_preprocessor_and_scaler_state(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0], "category": ["a", "b"]})
        preprocessor = StableTabularPreprocessor(("x", "category"), ("category",)).fit(frame)
        self.assertEqual(preprocessor.transform(frame).shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "entirely missing"):
            preprocessor.fit(frame.assign(x=np.nan))
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            preprocessor.transform(frame)

        scaler = RobustStateScaler().fit(np.array([[1.0], [2.0]]), np.array([False]))
        self.assertEqual(scaler.score(np.array([[1.5]])).shape, (1,))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            scaler.fit(np.empty((0, 1)), np.array([False]))
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            scaler.score(np.array([[1.5]]))

    def test_missing_unknown_category_and_extreme_value_are_supported(self) -> None:
        train = pd.DataFrame({"continuous": [1.0, 2.0, np.nan], "category": ["a", "b", "a"]})
        processor = StableTabularPreprocessor(
            ("continuous", "category"), ("category",), categorical_hash_bins=16
        ).fit(train)
        transformed = processor.transform(
            pd.DataFrame({"continuous": [np.nan, 1e12], "category": ["unseen", None]})
        )
        self.assertEqual(transformed.shape, (2, 2))
        self.assertTrue(np.isfinite(transformed).all())

    def test_nonnumeric_continuous_value_is_not_silently_imputed(self) -> None:
        frame = pd.DataFrame({"continuous": [1.0, "bad"]})
        with self.assertRaisesRegex(ValueError, "nonnumeric"):
            StableTabularPreprocessor(("continuous",), ()).fit(frame)

    def test_all_missing_continuous_and_invalid_contract_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "entirely missing"):
            StableTabularPreprocessor(("continuous",), ()).fit(
                pd.DataFrame({"continuous": [np.nan, np.nan]})
            )
        with self.assertRaisesRegex(ValueError, "categorical_hash_bins"):
            StableTabularPreprocessor(("f0",), (), categorical_hash_bins=True).fit(
                pd.DataFrame({"f0": [1.0]})
            )

    def test_all_categorical_density_proxy_is_defined(self) -> None:
        matrix = np.array([[1.0], [2.0]])
        scaler = RobustStateScaler().fit(matrix, np.array([True]))
        self.assertTrue(np.array_equal(scaler.score(matrix), np.zeros(2)))

    def test_state_scaler_rejects_misaligned_or_nonfinite_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned boolean"):
            RobustStateScaler().fit(np.ones((2, 2)), np.array([False]))
        scaler = RobustStateScaler().fit(np.ones((2, 2)), np.array([False, False]))
        with self.assertRaisesRegex(ValueError, "finite"):
            scaler.score(np.array([[1.0, np.nan]]))


if __name__ == "__main__":
    unittest.main()
