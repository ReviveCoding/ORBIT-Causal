from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class PolicyResult:
    decision: np.ndarray
    action_set: np.ndarray
    score: np.ndarray
    rank: np.ndarray
    reason_code: np.ndarray


def capacity_policy(
    effect: np.ndarray,
    uncertainty: np.ndarray | None,
    capacity: float,
    risk_multiplier: float = 0.0,
    intervention_cost: float = 0.0,
    guarded: bool = False,
    tie_breaker: np.ndarray | None = None,
) -> PolicyResult:
    raw_effect = np.asarray(effect)
    if np.iscomplexobj(raw_effect):
        raise ValueError("effect must be real-valued")
    effect = raw_effect.astype(float)
    if effect.ndim != 1 or len(effect) == 0:
        raise ValueError("effect must be a nonempty vector")
    if not np.isfinite(effect).all():
        raise ValueError("effect must contain only finite values")
    if (
        not isinstance(capacity, Real)
        or isinstance(capacity, bool)
        or not np.isfinite(capacity)
        or capacity <= 0
        or capacity > 1
    ):
        raise ValueError("capacity must be in (0, 1]")
    if uncertainty is None:
        uncertainty = np.zeros_like(effect)
    raw_uncertainty = np.asarray(uncertainty)
    if np.iscomplexobj(raw_uncertainty):
        raise ValueError("uncertainty must be real-valued")
    uncertainty = raw_uncertainty.astype(float)
    if uncertainty.shape != effect.shape:
        raise ValueError("uncertainty shape mismatch")
    if not np.isfinite(uncertainty).all():
        raise ValueError("uncertainty must contain only finite values")
    if (uncertainty < 0).any():
        raise ValueError("uncertainty must be nonnegative")
    if (
        not isinstance(risk_multiplier, Real)
        or isinstance(risk_multiplier, bool)
        or not np.isfinite(risk_multiplier)
        or risk_multiplier < 0
    ):
        raise ValueError("risk_multiplier must be finite and nonnegative")
    if (
        not isinstance(intervention_cost, Real)
        or isinstance(intervention_cost, bool)
        or not np.isfinite(intervention_cost)
    ):
        raise ValueError("intervention_cost must be finite")

    try:
        with np.errstate(over="raise", invalid="raise"):
            risk_width = risk_multiplier * uncertainty
            lower = effect - risk_width - intervention_cost
            upper = effect + risk_width - intervention_cost
            score = lower if guarded else effect - intervention_cost
    except FloatingPointError as error:
        raise ValueError("derived policy values overflowed; rescale the inputs") from error
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.isfinite(score).all():
        raise ValueError("derived policy values must be finite")
    if tie_breaker is None:
        tie_key: np.ndarray = np.arange(len(effect))
    else:
        raw_tie_key = np.asarray(tie_breaker)
        if raw_tie_key.ndim != 1 or len(raw_tie_key) != len(effect):
            raise ValueError("tie_breaker must be an aligned one-dimensional vector")
        tie_key = raw_tie_key.astype(str)
        if len(np.unique(tie_key)) != len(tie_key):
            raise ValueError("tie_breaker values must be unique after string conversion")
    order = np.lexsort((tie_key, -score))
    rank = np.empty(len(effect), dtype=int)
    rank[order] = np.arange(1, len(effect) + 1)
    limit = int(np.floor(capacity * len(effect)))
    within_capacity = rank <= limit

    decision = np.zeros(len(effect), dtype=np.int8)
    action_set = np.full(len(effect), "{0}", dtype=object)
    reason = np.full(len(effect), "nonpositive_or_outside_capacity", dtype=object)
    if guarded:
        unresolved = (lower <= 0) & (upper > 0) & within_capacity
        action_set[unresolved] = "{0,1}"
        decision[unresolved] = -1
        reason[unresolved] = "uncertain_effect_abstain"
        treat = (lower > 0) & within_capacity
    else:
        treat = (score > 0) & within_capacity
    decision[treat] = 1
    action_set[treat] = "{1}"
    reason[treat] = "positive_value_within_capacity"
    return PolicyResult(decision, action_set, score, rank, reason)


def policy_treatment_indicator(result: PolicyResult) -> np.ndarray:
    return (result.decision == 1).astype(np.int8)
