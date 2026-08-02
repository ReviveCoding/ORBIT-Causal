from __future__ import annotations

import numpy as np

from orbit_causal.locked_evaluation import (
    calibration_rmse_linearization,
    paired_cluster_bootstrap,
)


def _calibration_error(effect: np.ndarray, gamma: np.ndarray, edges: np.ndarray) -> float:
    codes = np.clip(np.searchsorted(edges, effect, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(codes, minlength=len(edges) - 1)
    gaps = (
        np.bincount(codes, weights=effect - gamma, minlength=len(edges) - 1)
        / counts
    )
    return float(np.sqrt(np.dot(counts / len(effect), gaps**2)))


def test_calibration_influence_matches_small_contamination_derivative() -> None:
    rng = np.random.default_rng(7)
    effect = rng.normal(0.02, 0.01, 4000)
    gamma = effect + rng.normal(0.0, 0.03, 4000)
    result = calibration_rmse_linearization(effect, gamma, bins=5)
    target = 120
    epsilon = 1e-5
    weights = np.full(len(effect), (1.0 - epsilon) / len(effect))
    weights[target] += epsilon
    codes = np.asarray(__import__("pandas").qcut(effect, q=5, duplicates="drop").codes)
    counts = np.bincount(codes, weights=weights, minlength=5)
    gaps = np.bincount(codes, weights=weights * (effect - gamma), minlength=5) / counts
    perturbed = float(np.sqrt(np.dot(counts, gaps**2)))
    derivative = (perturbed - result.error) / epsilon
    assert np.isclose(derivative, result.influence[target], rtol=0.03, atol=2e-4)


def test_linearized_cluster_bootstrap_tracks_full_recomputation() -> None:
    rng = np.random.default_rng(11)
    rows = 12_000
    effect_a = rng.normal(0.02, 0.01, rows)
    effect_b = effect_a + rng.normal(0.0, 0.003, rows)
    gamma = effect_a + rng.normal(0.0, 0.035, rows)
    candidate = calibration_rmse_linearization(effect_a, gamma, bins=5)
    baseline = calibration_rmse_linearization(effect_b, gamma, bins=5)
    values_a = rng.normal(0.006, 0.02, rows)
    values_b = values_a + rng.normal(0.0, 0.005, rows)
    clusters = np.arange(rows) // 3
    draws = paired_cluster_bootstrap(
        values_a,
        values_b,
        candidate.error - baseline.error,
        candidate.influence - baseline.influence,
        clusters,
        300,
        13,
    )
    assert len(draws) == 300
    assert np.isfinite(draws.to_numpy()).all()
    assert abs(
        draws["candidate_minus_baseline_calibration_rmse"].mean()
        - (candidate.error - baseline.error)
    ) < 5e-4


def test_paired_bootstrap_preserves_identical_policy_difference() -> None:
    values = np.linspace(-0.1, 0.1, 500)
    calibration_if = np.zeros(500)
    draws = paired_cluster_bootstrap(
        values,
        values,
        -0.01,
        calibration_if,
        np.arange(500),
        50,
        17,
    )
    assert np.array_equal(draws["candidate_minus_baseline"].to_numpy(), np.zeros(50))
    assert np.array_equal(
        draws["candidate_minus_baseline_calibration_rmse"].to_numpy(),
        np.full(50, -0.01),
    )
