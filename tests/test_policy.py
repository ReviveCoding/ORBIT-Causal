from __future__ import annotations

import unittest

import numpy as np

from orbit_causal.policy import capacity_policy


class PolicyTests(unittest.TestCase):
    def test_capacity_and_abstention(self) -> None:
        effect = np.array([0.4, 0.2, -0.5, -1.0])
        uncertainty = np.array([0.01, 0.30, 0.10, 0.01])
        result = capacity_policy(
            effect, uncertainty, capacity=0.5, risk_multiplier=1.0, guarded=True
        )
        self.assertLessEqual(int((result.decision == 1).sum()), 2)
        self.assertEqual(result.decision[1], -1)
        self.assertEqual(result.action_set[1], "{0,1}")

    def test_ties_are_deterministic(self) -> None:
        effect = np.ones(10)
        first = capacity_policy(effect, None, 0.2)
        second = capacity_policy(effect, None, 0.2)
        self.assertTrue(np.array_equal(first.decision, second.decision))
        keyed = capacity_policy(
            np.ones(4),
            None,
            0.5,
            tie_breaker=np.array(["d", "a", "c", "b"]),
        )
        self.assertEqual(keyed.decision.tolist(), [0, 1, 0, 1])

    def test_small_batch_never_exceeds_capacity_and_outside_rows_do_not_abstain(self) -> None:
        single = capacity_policy(np.array([1.0]), np.array([0.1]), capacity=0.2)
        self.assertEqual(int((single.decision == 1).sum()), 0)
        guarded = capacity_policy(
            np.array([0.5, 0.1, 0.05, 0.01]),
            np.array([0.01, 0.2, 0.2, 0.2]),
            capacity=0.25,
            risk_multiplier=1.0,
            guarded=True,
        )
        self.assertEqual(guarded.decision.tolist(), [1, 0, 0, 0])

    def test_nonfinite_policy_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            capacity_policy(np.array([0.1, np.nan]), None, 0.5)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            capacity_policy(np.array([0.1 + 1.0j]), None, 0.5)
        with self.assertRaisesRegex(ValueError, "risk_multiplier"):
            capacity_policy(np.array([0.1, 0.2]), None, 0.5, risk_multiplier=-1.0)
        with self.assertRaisesRegex(ValueError, "uncertainty must be nonnegative"):
            capacity_policy(np.array([0.1, 0.2]), np.array([0.1, -0.1]), 0.5)
        with self.assertRaisesRegex(ValueError, "capacity"):
            capacity_policy(np.array([0.1, 0.2]), None, float("nan"))
        with self.assertRaisesRegex(ValueError, "overflowed"):
            capacity_policy(
                np.array([1e308, 1.0]),
                np.array([1e308, 0.0]),
                1.0,
                risk_multiplier=1e308,
                guarded=True,
            )


if __name__ == "__main__":
    unittest.main()
