from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from orbit_causal.reporting import _interval_errors, _validation_boundary


class ReportingTests(unittest.TestCase):
    def test_validation_boundary_matches_the_executed_data_role(self) -> None:
        fixture_markdown, _ = _validation_boundary("synthetic_fixture")
        subset_markdown, _ = _validation_boundary("public_randomized_benchmark_subset")
        full_markdown, _ = _validation_boundary("public_randomized_benchmark")
        self.assertIn("Public-data", fixture_markdown)
        self.assertIn("subset", subset_markdown)
        self.assertIn("official public benchmark", full_markdown)
        self.assertNotIn("Public-data", full_markdown)
        with self.assertRaisesRegex(ValueError, "unsupported data role"):
            _validation_boundary("forged_role")

    def test_percentile_interval_outside_point_has_nonnegative_plot_extents(self) -> None:
        errors = _interval_errors(
            pd.Series([0.10, 0.10]),
            pd.Series([0.12, 0.02]),
            pd.Series([0.20, 0.08]),
        )
        self.assertTrue(np.isfinite(errors).all())
        self.assertTrue((errors >= 0).all())
        np.testing.assert_allclose(errors[:, 0], np.array([0.0, 0.10]))
        np.testing.assert_allclose(errors[:, 1], np.array([0.08, 0.0]))


if __name__ == "__main__":
    unittest.main()
