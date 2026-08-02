from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from orbit_causal.policy import capacity_policy


@st.composite
def policy_inputs(draw):
    size = draw(st.integers(min_value=1, max_value=64))
    finite = st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )
    nonnegative = st.floats(
        min_value=0.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )
    effect = np.asarray(draw(st.lists(finite, min_size=size, max_size=size)), dtype=float)
    uncertainty = np.asarray(draw(st.lists(nonnegative, min_size=size, max_size=size)), dtype=float)
    capacity = draw(
        st.floats(
            min_value=1e-6,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
            width=64,
        )
    )
    risk = draw(nonnegative)
    cost = draw(finite)
    guarded = draw(st.booleans())
    return effect, uncertainty, capacity, risk, cost, guarded


@given(policy_inputs())
@settings(max_examples=150, deadline=None, derandomize=True)
def test_capacity_policy_invariants(inputs) -> None:
    effect, uncertainty, capacity, risk, cost, guarded = inputs
    first = capacity_policy(effect, uncertainty, capacity, risk, cost, guarded)
    second = capacity_policy(effect, uncertainty, capacity, risk, cost, guarded)

    limit = int(np.floor(capacity * len(effect)))
    assert int(np.sum(first.decision == 1)) <= limit
    assert set(first.decision.tolist()).issubset({-1, 0, 1})
    assert sorted(first.rank.tolist()) == list(range(1, len(effect) + 1))
    assert np.array_equal(first.decision, second.decision)
    assert np.array_equal(first.rank, second.rank)
    assert np.array_equal(first.action_set, second.action_set)
    assert np.all(first.rank[first.decision == 1] <= limit)
    assert np.all(first.score[first.decision == 1] > 0)
    assert np.all(first.action_set[first.decision == 1] == "{1}")
    assert np.all(first.action_set[first.decision == 0] == "{0}")
    if np.any(first.decision == -1):
        assert guarded
        assert np.all(first.rank[first.decision == -1] <= limit)
        assert np.all(first.action_set[first.decision == -1] == "{0,1}")
