from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from .models import doubly_robust_score
from .policy import capacity_policy, policy_treatment_indicator

MAX_BOOTSTRAP_REPLICATES = 100_000
MAX_POLICY_CURVE_CAPACITIES = 1_000


@dataclass(frozen=True)
class Estimate:
    value: float
    lower: float
    upper: float
    standard_error: float


def _aligned_finite_arrays(*values: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value) for value in values)
    if not arrays or arrays[0].size == 0:
        raise ValueError("metric inputs must be nonempty")
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("metric inputs must be aligned one-dimensional arrays")
    if any(np.iscomplexobj(array) for array in arrays):
        raise ValueError("metric inputs must be real-valued")
    length = len(arrays[0])
    if any(len(array) != length for array in arrays):
        raise ValueError("metric inputs must be aligned one-dimensional arrays")
    if any(not np.isfinite(array.astype(float)).all() for array in arrays):
        raise ValueError("metric inputs must be finite")
    return arrays


def aipw_policy_scores(
    outcome: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float,
) -> np.ndarray:
    """Compute per-observation AIPW scores for the absolute policy value."""
    observed_outcome, observed_treatment, policy_action, mu0, mu1 = (
        _aligned_finite_arrays(outcome, treatment, policy, mu0, mu1)
    )
    observed_outcome = observed_outcome.astype(float)
    if (
        not np.isin(observed_treatment, [0, 1]).all()
        or not np.isin(policy_action, [0, 1]).all()
    ):
        raise ValueError("treatment and policy must be binary")
    observed_treatment = observed_treatment.astype(int)
    policy_action = policy_action.astype(int)
    mu0 = mu0.astype(float)
    mu1 = mu1.astype(float)
    if (
        not isinstance(propensity, Real)
        or isinstance(propensity, bool)
        or not np.isfinite(propensity)
        or propensity <= 0
        or propensity >= 1
    ):
        raise ValueError("propensity must be finite and in (0, 1)")
    clipped_propensity = float(np.clip(propensity, 0.02, 0.98))
    target_action_mean = np.where(policy_action == 1, mu1, mu0)
    observed_action_mean = np.where(observed_treatment == 1, mu1, mu0)
    observed_action_probability = np.where(
        observed_treatment == 1, clipped_propensity, 1 - clipped_propensity
    )
    action_matches = (observed_treatment == policy_action).astype(float)
    return (
        target_action_mean
        + action_matches
        * (observed_outcome - observed_action_mean)
        / observed_action_probability
    )


def incremental_policy_scores(
    outcome: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float,
) -> np.ndarray:
    """Compute per-observation policy gains relative to treating nobody."""
    policy_value_scores = aipw_policy_scores(
        outcome, treatment, policy, mu0, mu1, propensity
    )
    no_treatment_scores = aipw_policy_scores(
        outcome, treatment, np.zeros_like(policy), mu0, mu1, propensity
    )
    return policy_value_scores - no_treatment_scores


def bootstrap_mean(
    values: np.ndarray,
    replicates: int,
    seed: int,
    alpha: float = 0.05,
    *,
    clusters: np.ndarray | None = None,
) -> Estimate:
    raw_values = np.asarray(values)
    if np.iscomplexobj(raw_values):
        raise ValueError("bootstrap values must be real-valued")
    values = raw_values.astype(float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a nonempty finite vector")
    if (
        not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or not 2 <= replicates <= MAX_BOOTSTRAP_REPLICATES
    ):
        raise ValueError(f"bootstrap replicates must be in [2, {MAX_BOOTSTRAP_REPLICATES}]")
    if (
        not isinstance(alpha, Real)
        or isinstance(alpha, bool)
        or not np.isfinite(alpha)
        or alpha <= 0
        or alpha >= 1
    ):
        raise ValueError("alpha must be in (0, 1)")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 2**32:
        raise ValueError("bootstrap seed must be an integer in [0, 2**32)")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    if clusters is None:
        for index in range(replicates):
            sample = rng.integers(0, len(values), size=len(values))
            means[index] = np.mean(values[sample])
    else:
        cluster_values = np.asarray(clusters)
        if cluster_values.ndim != 1 or len(cluster_values) != len(values):
            raise ValueError("bootstrap clusters must be an aligned one-dimensional vector")
        if pd.isna(cluster_values).any():
            raise ValueError("bootstrap clusters must not contain missing values")
        codes, unique_clusters = pd.factorize(cluster_values, sort=False)
        cluster_count = len(unique_clusters)
        if cluster_count < 2:
            raise ValueError("cluster bootstrap requires at least two independent clusters")
        row_counts = np.bincount(codes, minlength=cluster_count).astype(float)
        value_sums = np.bincount(codes, weights=values, minlength=cluster_count)
        for index in range(replicates):
            sampled = rng.integers(0, cluster_count, size=cluster_count)
            multiplicity = np.bincount(sampled, minlength=cluster_count).astype(float)
            means[index] = float(
                np.dot(multiplicity, value_sums) / np.dot(multiplicity, row_counts)
            )
    value = float(np.mean(values))
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(value, float(lower), float(upper), float(np.std(means, ddof=1)))


def paired_bootstrap_mean(
    candidate_values: np.ndarray,
    baseline_values: np.ndarray,
    replicates: int,
    seed: int,
    alpha: float = 0.05,
    *,
    clusters: np.ndarray | None = None,
) -> Estimate:
    """Bootstrap a paired candidate-minus-baseline mean on shared rows/clusters."""

    candidate, baseline = _aligned_finite_arrays(candidate_values, baseline_values)
    return bootstrap_mean(
        candidate.astype(float) - baseline.astype(float),
        replicates,
        seed,
        alpha,
        clusters=clusters,
    )


def policy_value_curve(
    effect: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float,
    capacities: tuple[float, ...],
    *,
    uncertainty: np.ndarray | None = None,
    risk_multiplier: float = 0.0,
    intervention_cost: float = 0.0,
    guarded: bool = False,
) -> pd.DataFrame:
    effect, outcome, treatment, mu0, mu1 = _aligned_finite_arrays(
        effect, outcome, treatment, mu0, mu1
    )
    if not 1 <= len(capacities) <= MAX_POLICY_CURVE_CAPACITIES:
        raise ValueError(f"capacities must contain 1-{MAX_POLICY_CURVE_CAPACITIES} values")
    if uncertainty is not None:
        (uncertainty,) = _aligned_finite_arrays(uncertainty)
        if len(uncertainty) != len(effect):
            raise ValueError("uncertainty must be aligned with effect")
    rows = []
    for capacity in capacities:
        result = capacity_policy(
            effect,
            uncertainty,
            capacity,
            risk_multiplier=risk_multiplier,
            intervention_cost=intervention_cost,
            guarded=guarded,
        )
        policy = policy_treatment_indicator(result)
        scores = incremental_policy_scores(outcome, treatment, policy, mu0, mu1, propensity)
        rows.append(
            {
                "capacity": capacity,
                "treat_rate": float(policy.mean()),
                "incremental_value": float(scores.mean()),
            }
        )
    return pd.DataFrame(rows)


def qini_coefficient(
    effect: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, propensity: float
) -> float:
    effect, observed_outcome, observed_treatment = _aligned_finite_arrays(
        effect, outcome, treatment
    )
    observed_outcome = observed_outcome.astype(float)
    if not np.isin(observed_treatment, [0, 1]).all():
        raise ValueError("treatment must be binary")
    observed_treatment = observed_treatment.astype(int)
    if (
        not isinstance(propensity, Real)
        or isinstance(propensity, bool)
        or not np.isfinite(propensity)
        or propensity <= 0
        or propensity >= 1
    ):
        raise ValueError("propensity must be finite and in (0, 1)")
    clipped_propensity = float(np.clip(propensity, 0.02, 0.98))
    contribution = observed_treatment * observed_outcome / clipped_propensity - (
        1 - observed_treatment
    ) * observed_outcome / (1 - clipped_propensity)
    ordered = pd.DataFrame(
        {"effect": effect, "contribution": contribution}
    ).sort_values("effect", ascending=False, kind="mergesort")
    tie_groups = ordered.groupby("effect", sort=False)["contribution"].agg(
        ["size", "sum"]
    )
    curve = np.concatenate([[0.0], tie_groups["sum"].cumsum().to_numpy() / len(effect)])
    targeted_fraction = np.concatenate(
        [[0.0], tie_groups["size"].cumsum().to_numpy() / len(effect)]
    )
    random_targeting_curve = targeted_fraction * curve[-1]
    return float(np.trapezoid(curve - random_targeting_curve, targeted_fraction))


def causal_calibration_table(
    effect: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float,
    bins: int,
) -> tuple[pd.DataFrame, float]:
    effect, outcome, treatment, mu0, mu1 = _aligned_finite_arrays(
        effect, outcome, treatment, mu0, mu1
    )
    if not isinstance(bins, int) or isinstance(bins, bool) or not 2 <= bins <= 1_000:
        raise ValueError("calibration bins must be an integer in [2, 1000]")
    gamma = doubly_robust_score(outcome, treatment, mu0, mu1, propensity)
    frame = pd.DataFrame({"effect": effect, "gamma": gamma})
    unique = frame["effect"].nunique()
    if unique == 1:
        frame["bin"] = "all"
    else:
        actual_bins = min(bins, unique)
        frame["bin"] = pd.qcut(frame["effect"], q=actual_bins, duplicates="drop")
    table = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("effect", "size"),
            predicted_effect=("effect", "mean"),
            dr_observed_effect=("gamma", "mean"),
            dr_std=("gamma", "std"),
        )
        .reset_index(drop=True)
    )
    table["calibration_gap"] = table["predicted_effect"] - table["dr_observed_effect"]
    weights = table["n"] / table["n"].sum()
    error = float(np.sqrt(np.sum(weights * table["calibration_gap"] ** 2)))
    return table, error


def sqrt_pehe(effect: np.ndarray, true_effect: np.ndarray) -> float:
    effect, true_effect = _aligned_finite_arrays(effect, true_effect)
    return float(np.sqrt(np.mean((effect - true_effect) ** 2)))


def policy_regret(policy: np.ndarray, true_effect: np.ndarray, capacity: float) -> float:
    policy, true_effect = _aligned_finite_arrays(policy, true_effect)
    if not np.isin(policy, [0, 1]).all():
        raise ValueError("policy must be binary")
    policy = policy.astype(int)
    true_effect = true_effect.astype(float)
    if (
        not isinstance(capacity, Real)
        or isinstance(capacity, bool)
        or not np.isfinite(capacity)
        or capacity <= 0
        or capacity > 1
    ):
        raise ValueError("capacity must be in (0, 1]")
    count = int(np.floor(capacity * len(true_effect)))
    if int(policy.sum()) > count:
        raise ValueError("policy exceeds the configured treatment capacity")
    oracle = np.zeros(len(true_effect), dtype=int)
    eligible = np.where(true_effect > 0)[0]
    ordered = eligible[np.argsort(-true_effect[eligible], kind="stable")]
    oracle[ordered[:count]] = 1
    return float(np.mean(oracle * true_effect) - np.mean(policy * true_effect))


def unsupported_treatment_rate(policy: np.ndarray, true_effect: np.ndarray) -> float:
    policy, true_effect = _aligned_finite_arrays(policy, true_effect)
    if not np.isin(policy, [0, 1]).all():
        raise ValueError("policy must be binary")
    policy = policy.astype(int)
    treated = policy == 1
    if not treated.any():
        return 0.0
    return float(np.mean(true_effect[treated] <= 0))


def set_valued_action_metrics(
    action_set: np.ndarray, true_effect: np.ndarray, capacity: float
) -> dict[str, float]:
    """Score a candidate action set against the synthetic oracle action.

    Recall is oracle-action coverage. Precision penalizes two-action sets as
    overreach. These diagnostics require generated individual ground truth.
    """

    actions = np.asarray(action_set, dtype=object)
    raw_true_effect = np.asarray(true_effect)
    if np.iscomplexobj(raw_true_effect):
        raise ValueError("true_effect must be real-valued")
    true_effect = raw_true_effect.astype(float)
    if actions.ndim != 1 or true_effect.ndim != 1 or len(actions) != len(true_effect):
        raise ValueError("action_set and true_effect must be aligned vectors")
    if len(actions) == 0 or not np.isfinite(true_effect).all():
        raise ValueError("action_set metrics require nonempty finite inputs")
    allowed = np.array(["{0}", "{1}", "{0,1}"], dtype=object)
    if not np.isin(actions, allowed).all():
        raise ValueError("action_set contains an unsupported value")
    if (
        not isinstance(capacity, Real)
        or isinstance(capacity, bool)
        or not np.isfinite(capacity)
        or capacity <= 0
        or capacity > 1
    ):
        raise ValueError("capacity must be in (0, 1]")
    oracle = np.zeros(len(true_effect), dtype=int)
    limit = int(np.floor(capacity * len(true_effect)))
    eligible = np.flatnonzero(true_effect > 0)
    order = eligible[np.argsort(-true_effect[eligible], kind="stable")]
    oracle[order[:limit]] = 1
    contains_zero = np.isin(actions, ["{0}", "{0,1}"])
    contains_one = np.isin(actions, ["{1}", "{0,1}"])
    covered = np.where(oracle == 1, contains_one, contains_zero).astype(float)
    set_size = np.where(actions == "{0,1}", 2.0, 1.0)
    precision = float(np.mean(covered / set_size))
    recall = float(np.mean(covered))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "action_set_recall": recall,
        "action_set_precision": precision,
        "action_set_f1": float(f1),
        "critical_miss_rate": float(np.mean(covered == 0)),
        "action_set_overreach_rate": float(np.mean(set_size > 1)),
    }
