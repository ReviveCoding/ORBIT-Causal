from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .policy import capacity_policy

MAX_SCORE_RECORDS = 100_000


@dataclass
class DecisionBundle:
    model: Any
    candidate_id: str
    feature_columns: tuple[str, ...]
    capacity: float
    risk_multiplier: float
    intervention_cost: float
    run_id: str
    dataset_fingerprint: str
    artifact_schema_version: str = "1.0"

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("scoring frame must be nonempty")
        if len(frame) > MAX_SCORE_RECORDS:
            raise ValueError(f"scoring frame exceeds {MAX_SCORE_RECORDS} rows")
        duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
        if duplicate_columns:
            raise ValueError(f"scoring frame contains duplicate columns: {duplicate_columns}")
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"missing scoring features: {missing}")
        if "row_key" in frame:
            raw_row_key = frame["row_key"]
            if raw_row_key.isna().any() or any(not isinstance(value, str) for value in raw_row_key):
                raise ValueError("input row_key values must be strings")
            row_key = raw_row_key.astype("string")
            if (
                (row_key.str.len() == 0).any()
                or (row_key.str.len() > 256).any()
                or row_key.str.contains(r"[\x00-\x1f\x7f]", regex=True).any()
                or row_key.duplicated().any()
            ):
                raise ValueError(
                    "input row_key values must be nonempty, at most 256 characters, "
                    "free of control characters, and unique within a batch"
                )
            row_key_values = row_key.astype(str).to_numpy()
            tie_breaker = row_key_values
        else:
            row_key_values = np.arange(len(frame)).astype(str)
            tie_breaker = None
        raw_effect = np.asarray(self.model.predict_effect(frame))
        raw_uncertainty = np.asarray(self.model.predict_uncertainty(frame))
        if np.iscomplexobj(raw_effect) or np.iscomplexobj(raw_uncertainty):
            raise ValueError("model outputs must be real-valued")
        effect = raw_effect.astype(float)
        uncertainty = raw_uncertainty.astype(float)
        if effect.shape != (len(frame),) or uncertainty.shape != (len(frame),):
            raise ValueError("model outputs must be aligned one-dimensional vectors")
        policy = capacity_policy(
            effect,
            uncertainty,
            self.capacity,
            self.risk_multiplier,
            self.intervention_cost,
            guarded=True,
            tie_breaker=tie_breaker,
        )
        result = pd.DataFrame(
            {
                "row_key": row_key_values,
                "eligible": True,
                "tau_calibrated": effect,
                "decision_uncertainty": uncertainty,
                "net_value_lower": policy.score,
                "action_set": policy.action_set,
                "decision": policy.decision,
                "rank": policy.rank,
                "reason_code": policy.reason_code,
                "candidate_id": self.candidate_id,
                "run_id": self.run_id,
                "allocation_scope": "request_batch",
                "allocation_batch_size": len(frame),
                "capacity_slots": int(np.floor(self.capacity * len(frame))),
            }
        )
        return result
