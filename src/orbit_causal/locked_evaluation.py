"""Locked paired inference used by the ORBIT-Causal v0.3.9 evaluation.

The calibration test uses the influence function of the binned causal
calibration RMSE.  This preserves paired row/cluster dependence without
rebuilding two ten-bin tables for every bootstrap replicate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationLinearization:
    error: float
    influence: np.ndarray
    table: pd.DataFrame


def calibration_rmse_linearization(
    effect: np.ndarray,
    gamma: np.ndarray,
    bins: int = 10,
) -> CalibrationLinearization:
    """Return binned calibration RMSE and its empirical influence function."""

    effect = np.asarray(effect, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if effect.ndim != 1 or gamma.ndim != 1 or len(effect) != len(gamma) or not len(effect):
        raise ValueError("effect and gamma must be nonempty aligned vectors")
    if not np.isfinite(effect).all() or not np.isfinite(gamma).all():
        raise ValueError("effect and gamma must be finite")
    if not isinstance(bins, int) or isinstance(bins, bool) or not 2 <= bins <= 1000:
        raise ValueError("bins must be an integer in [2, 1000]")

    if np.unique(effect).size == 1:
        codes = np.zeros(len(effect), dtype=np.int64)
    else:
        codes = pd.qcut(effect, q=min(bins, np.unique(effect).size), duplicates="drop").codes
    bin_count = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=bin_count).astype(float)
    effect_sums = np.bincount(codes, weights=effect, minlength=bin_count)
    gamma_sums = np.bincount(codes, weights=gamma, minlength=bin_count)
    predicted = effect_sums / counts
    observed = gamma_sums / counts
    gaps = predicted - observed
    weights = counts / len(effect)
    squared_error = float(np.dot(weights, gaps**2))
    error = float(np.sqrt(max(squared_error, 0.0)))

    residual = effect - gamma
    influence_squared = gaps[codes] ** 2 - squared_error + 2.0 * gaps[codes] * (
        residual - gaps[codes]
    )
    if error > 0.0:
        influence = influence_squared / (2.0 * error)
    else:
        influence = np.zeros(len(effect), dtype=float)
    influence -= influence.mean()

    table = pd.DataFrame(
        {
            "bin": np.arange(bin_count, dtype=int),
            "n": counts.astype(int),
            "predicted_effect": predicted,
            "dr_observed_effect": observed,
            "calibration_gap": gaps,
        }
    )
    return CalibrationLinearization(error=error, influence=influence, table=table)


def paired_cluster_bootstrap(
    candidate_values: np.ndarray,
    baseline_values: np.ndarray,
    calibration_difference: float,
    calibration_influence_difference: np.ndarray,
    clusters: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Jointly resample paired policy values and calibration influence values."""

    candidate_values = np.asarray(candidate_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    calibration_influence_difference = np.asarray(
        calibration_influence_difference, dtype=float
    )
    clusters = np.asarray(clusters)
    length = len(candidate_values)
    if any(
        value.ndim != 1 or len(value) != length
        for value in (baseline_values, calibration_influence_difference, clusters)
    ):
        raise ValueError("bootstrap inputs must be aligned one-dimensional vectors")
    if length == 0 or not all(
        np.isfinite(value).all()
        for value in (candidate_values, baseline_values, calibration_influence_difference)
    ):
        raise ValueError("bootstrap numeric inputs must be nonempty and finite")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 2:
        raise ValueError("replicates must be an integer of at least two")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")

    codes, unique_clusters = pd.factorize(clusters, sort=True)
    cluster_count = len(unique_clusters)
    if cluster_count < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    counts = np.bincount(codes, minlength=cluster_count).astype(float)
    candidate_sums = np.bincount(codes, weights=candidate_values, minlength=cluster_count)
    baseline_sums = np.bincount(codes, weights=baseline_values, minlength=cluster_count)
    calibration_if_sums = np.bincount(
        codes, weights=calibration_influence_difference, minlength=cluster_count
    )
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 4), dtype=float)
    for replicate in range(replicates):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        denominator = counts[sampled].sum()
        candidate = candidate_sums[sampled].sum() / denominator
        baseline = baseline_sums[sampled].sum() / denominator
        calibration = calibration_difference + calibration_if_sums[sampled].sum() / denominator
        draws[replicate] = candidate, baseline, candidate - baseline, calibration
    return pd.DataFrame(
        draws,
        columns=(
            "candidate_value",
            "baseline_value",
            "candidate_minus_baseline",
            "candidate_minus_baseline_calibration_rmse",
        ),
    )
