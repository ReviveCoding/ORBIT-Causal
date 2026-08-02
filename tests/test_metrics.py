from __future__ import annotations

import unittest

import numpy as np

from orbit_causal.metrics import (
    aipw_policy_scores,
    bootstrap_mean,
    causal_calibration_table,
    policy_regret,
    paired_bootstrap_mean,
    policy_value_curve,
    qini_coefficient,
    set_valued_action_metrics,
    sqrt_pehe,
    unsupported_treatment_rate,
)


class MetricTests(unittest.TestCase):
    def test_aipw_is_finite_and_policy_sensitive(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0], dtype=float)
        t = np.array([0, 1, 0, 1, 0, 1], dtype=int)
        mu0 = np.full(6, 0.3)
        mu1 = np.full(6, 0.5)
        never = aipw_policy_scores(y, t, np.zeros(6, dtype=int), mu0, mu1, 0.5)
        all_treat = aipw_policy_scores(y, t, np.ones(6, dtype=int), mu0, mu1, 0.5)
        self.assertTrue(np.isfinite(never).all())
        self.assertNotEqual(float(never.mean()), float(all_treat.mean()))

    def test_ground_truth_metrics(self) -> None:
        truth = np.array([-0.2, 0.1, 0.3, 0.4])
        predicted = truth.copy()
        policy = np.array([0, 0, 1, 1])
        self.assertEqual(sqrt_pehe(predicted, truth), 0.0)
        self.assertEqual(policy_regret(policy, truth, 0.5), 0.0)
        self.assertEqual(unsupported_treatment_rate(policy, truth), 0.0)

    def test_bootstrap_is_deterministic(self) -> None:
        values = np.arange(10, dtype=float)
        first = bootstrap_mean(values, 50, 3)
        second = bootstrap_mean(values, 50, 3)
        self.assertEqual(first, second)

    def test_cluster_bootstrap_is_invariant_to_uniform_within_cluster_duplication(self) -> None:
        values = np.array([0.0, 1.0, 4.0])
        clusters = np.array(["a", "b", "c"])
        original = bootstrap_mean(values, 100, 7, clusters=clusters)
        duplicated = bootstrap_mean(
            np.repeat(values, 2),
            100,
            7,
            clusters=np.repeat(clusters, 2),
        )
        self.assertEqual(original, duplicated)

    def test_cluster_bootstrap_rejects_invalid_cluster_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            bootstrap_mean(np.array([1.0, 2.0]), 10, 1, clusters=np.array(["a"]))
        with self.assertRaisesRegex(ValueError, "at least two"):
            bootstrap_mean(np.array([1.0, 2.0]), 10, 1, clusters=np.array(["a", "a"]))

    def test_paired_bootstrap_preserves_row_pairing(self) -> None:
        baseline = np.array([10.0, -10.0, 5.0, -5.0])
        candidate = baseline + 0.25
        estimate = paired_bootstrap_mean(candidate, baseline, 100, 11)
        self.assertAlmostEqual(estimate.value, 0.25)
        self.assertAlmostEqual(estimate.lower, 0.25)
        self.assertAlmostEqual(estimate.upper, 0.25)

    def test_paired_bootstrap_rejects_misaligned_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            paired_bootstrap_mean(np.array([1.0, 2.0]), np.array([1.0]), 10, 2)

    def test_set_valued_action_metrics_penalize_overreach(self) -> None:
        metrics = set_valued_action_metrics(
            np.array(["{1}", "{0,1}", "{0}", "{1}"], dtype=object),
            np.array([0.2, -0.1, -0.2, -0.3]),
            0.5,
        )
        self.assertAlmostEqual(metrics["action_set_recall"], 0.75)
        self.assertAlmostEqual(metrics["action_set_precision"], 0.625)
        self.assertAlmostEqual(metrics["critical_miss_rate"], 0.25)

    def test_constant_effect_calibration_is_defined(self) -> None:
        effect = np.full(8, 0.1)
        outcome = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        treatment = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
        mu0 = np.full(8, 0.25)
        mu1 = np.full(8, 0.35)
        table, error = causal_calibration_table(effect, outcome, treatment, mu0, mu1, 0.5, bins=5)
        self.assertEqual(len(table), 1)
        self.assertTrue(np.isfinite(error))

    def test_qini_is_invariant_to_row_order_when_scores_tie(self) -> None:
        effect = np.array([0.5, 0.5, 0.5, 0.1, 0.1, -0.2])
        outcome = np.array([1, 0, 1, 0, 1, 0], dtype=float)
        treatment = np.array([1, 0, 0, 1, 0, 1], dtype=int)
        order = np.array([2, 0, 5, 4, 1, 3])
        original = qini_coefficient(effect, outcome, treatment, 0.5)
        shuffled = qini_coefficient(effect[order], outcome[order], treatment[order], 0.5)
        self.assertAlmostEqual(original, shuffled)

    def test_policy_curve_uses_the_deployed_positive_value_rule(self) -> None:
        curve = policy_value_curve(
            np.array([-0.1, -0.2, -0.3, -0.4]),
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0, 1, 0, 1]),
            np.full(4, 0.2),
            np.full(4, 0.3),
            0.5,
            (0.5,),
        )
        self.assertEqual(float(curve.iloc[0]["treat_rate"]), 0.0)

    def test_metric_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty"):
            bootstrap_mean(np.array([]), 20, 1)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            sqrt_pehe(np.array([0.1 + 1.0j]), np.array([0.1]))
        with self.assertRaisesRegex(ValueError, r"\[2, 100000\]"):
            bootstrap_mean(np.array([1.0, 2.0]), 100_001, 1)
        with self.assertRaisesRegex(ValueError, "integer"):
            causal_calibration_table(
                np.array([0.1, 0.2]),
                np.array([0.0, 1.0]),
                np.array([0, 1]),
                np.array([0.2, 0.2]),
                np.array([0.3, 0.3]),
                0.5,
                bins=2.5,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "binary"):
            aipw_policy_scores(
                np.array([0.0, 1.0]),
                np.array([0, 1]),
                np.array([0, 2]),
                np.array([0.2, 0.2]),
                np.array([0.3, 0.3]),
                0.5,
            )
        with self.assertRaisesRegex(ValueError, "binary"):
            aipw_policy_scores(
                np.array([0.0, 1.0]),
                np.array([0.0, 0.5]),
                np.array([0, 1]),
                np.array([0.2, 0.2]),
                np.array([0.3, 0.3]),
                0.5,
            )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            set_valued_action_metrics(
                np.array(["{0}", "invalid"], dtype=object),
                np.array([0.1, -0.1]),
                0.5,
            )
        with self.assertRaisesRegex(ValueError, "aligned"):
            policy_regret(np.array([1]), np.array([0.1, 0.2]), 0.5)


if __name__ == "__main__":
    unittest.main()
