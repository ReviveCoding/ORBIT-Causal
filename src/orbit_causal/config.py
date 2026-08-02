from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

MAX_CONFIG_BYTES = 1 << 20
MAX_DATA_ROWS = 100_000_000
MAX_FEATURE_COLUMNS = 1_024
MAX_POLICY_GRID_VALUES = 1_000
MAX_MODEL_ITERATIONS = 10_000
MAX_MODEL_DEPTH = 100


@dataclass(frozen=True)
class DataConfig:
    kind: str = "synthetic"
    n_rows: int = 10_000
    outcome: str = "visit"
    treatment: str = "treatment"
    exposure: str = "exposure"
    feature_columns: tuple[str, ...] = tuple(f"f{i}" for i in range(8))
    categorical_columns: tuple[str, ...] = ("f6", "f7")


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.60
    calibration: float = 0.15
    policy: float = 0.10
    final_test: float = 0.15

    def validate(self) -> None:
        values = [self.train, self.calibration, self.policy, self.final_test]
        if any(not _is_finite_real(v) or v <= 0 for v in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split shares must be positive and sum to one")


@dataclass(frozen=True)
class ModelConfig:
    max_iter: int = 100
    max_depth: int = 4
    learning_rate: float = 0.07
    min_samples_leaf: int = 30
    dr_folds: int = 3
    ensemble_size: int = 3
    categorical_hash_bins: int = 64


@dataclass(frozen=True)
class PolicyConfig:
    primary_capacity: float = 0.20
    scenario_capacities: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
    risk_multipliers: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    intervention_cost: float = 0.0
    max_abstention: float = 0.35


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_replicates: int = 200
    calibration_bins: int = 10
    alpha: float = 0.05
    slice_noninferiority_margin: float = 0.005
    minimum_slice_rows: int = 100
    max_inference_ms_per_1000: float = 100.0
    max_bundle_size_bytes: int = 100_000_000


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: str = "1.0"
    run_name: str = "fixture_demo"
    seed: int = 20260717
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported config schema_version: {self.schema_version}")
        if not isinstance(self.run_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.run_name
        ):
            raise ValueError("run_name must be 1-64 safe ASCII characters")
        _require_integer("seed", self.seed)
        if self.seed < 0 or self.seed >= 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        self.split.validate()
        if self.data.kind not in {"synthetic", "criteo_v2"}:
            raise ValueError("data.kind must be synthetic or criteo_v2")
        if self.data.outcome not in {"visit", "conversion"}:
            raise ValueError("outcome must be visit or conversion")
        _require_integer("data.n_rows", self.data.n_rows)
        if self.data.n_rows < 1:
            raise ValueError("data.n_rows must be positive")
        if self.data.n_rows > MAX_DATA_ROWS:
            raise ValueError(f"data.n_rows must not exceed {MAX_DATA_ROWS}")
        if self.data.kind == "synthetic" and self.data.n_rows < 1_000:
            raise ValueError("synthetic n_rows must be at least 1,000")
        names = (
            *self.data.feature_columns,
            *self.data.categorical_columns,
            self.data.treatment,
            self.data.exposure,
            self.data.outcome,
        )
        if any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            for name in names
        ):
            raise ValueError("data column names must be safe nonempty identifiers")
        if not self.data.feature_columns:
            raise ValueError("feature_columns must be nonempty")
        if len(self.data.feature_columns) > MAX_FEATURE_COLUMNS:
            raise ValueError(f"feature_columns must not exceed {MAX_FEATURE_COLUMNS} entries")
        if len(set(self.data.feature_columns)) != len(self.data.feature_columns):
            raise ValueError("feature_columns must be unique")
        if len(set(self.data.categorical_columns)) != len(self.data.categorical_columns):
            raise ValueError("categorical_columns must be unique")
        categorical = set(self.data.categorical_columns)
        features = set(self.data.feature_columns)
        if not categorical.issubset(features):
            raise ValueError("categorical_columns must be a subset of feature_columns")
        prohibited = {
            self.data.treatment,
            self.data.exposure,
            "visit",
            "conversion",
        }
        leaked = sorted(
            column
            for column in self.data.feature_columns
            if column in prohibited or column.startswith("true_") or column == "split"
        )
        if leaked:
            raise ValueError(f"post-treatment or evaluation-only feature leakage: {leaked}")
        if len({self.data.treatment, self.data.exposure, "visit", "conversion"}) != 4:
            raise ValueError("treatment, exposure, visit, and conversion names must be distinct")
        if any(
            name == "split" or name.startswith("true_")
            for name in (self.data.treatment, self.data.exposure)
        ):
            raise ValueError("treatment and exposure names must not use reserved split/true fields")
        if (
            not _is_finite_real(self.policy.primary_capacity)
            or self.policy.primary_capacity <= 0
            or self.policy.primary_capacity > 1
        ):
            raise ValueError("primary_capacity must be in (0, 1]")
        if not self.policy.scenario_capacities or any(
            not _is_finite_real(value) or value <= 0 or value > 1
            for value in self.policy.scenario_capacities
        ):
            raise ValueError("scenario_capacities must contain values in (0, 1]")
        if len(set(self.policy.scenario_capacities)) != len(self.policy.scenario_capacities):
            raise ValueError("scenario_capacities must be unique")
        if len(self.policy.scenario_capacities) > MAX_POLICY_GRID_VALUES:
            raise ValueError(
                f"scenario_capacities must not exceed {MAX_POLICY_GRID_VALUES} entries"
            )
        if self.policy.primary_capacity not in self.policy.scenario_capacities:
            raise ValueError("scenario_capacities must include primary_capacity")
        if not self.policy.risk_multipliers or any(
            not _is_finite_real(value) or value < 0 for value in self.policy.risk_multipliers
        ):
            raise ValueError("risk_multipliers must be finite and nonnegative")
        if len(set(self.policy.risk_multipliers)) != len(self.policy.risk_multipliers):
            raise ValueError("risk_multipliers must be unique")
        if len(self.policy.risk_multipliers) > MAX_POLICY_GRID_VALUES:
            raise ValueError(f"risk_multipliers must not exceed {MAX_POLICY_GRID_VALUES} entries")
        if 0.0 not in self.policy.risk_multipliers:
            raise ValueError("risk_multipliers must include the zero-risk fallback")
        if not _is_finite_real(self.policy.intervention_cost):
            raise ValueError("intervention_cost must be finite")
        if (
            not _is_finite_real(self.policy.max_abstention)
            or self.policy.max_abstention < 0
            or self.policy.max_abstention > 1
        ):
            raise ValueError("max_abstention must be in [0, 1]")
        for name, value in (
            ("model.dr_folds", self.model.dr_folds),
            ("model.ensemble_size", self.model.ensemble_size),
            ("model.max_iter", self.model.max_iter),
            ("model.max_depth", self.model.max_depth),
            ("model.min_samples_leaf", self.model.min_samples_leaf),
            ("model.categorical_hash_bins", self.model.categorical_hash_bins),
            ("evaluation.bootstrap_replicates", self.evaluation.bootstrap_replicates),
            ("evaluation.calibration_bins", self.evaluation.calibration_bins),
            ("evaluation.minimum_slice_rows", self.evaluation.minimum_slice_rows),
            ("evaluation.max_bundle_size_bytes", self.evaluation.max_bundle_size_bytes),
        ):
            _require_integer(name, value)
        if self.model.dr_folds < 2 or self.model.ensemble_size < 2:
            raise ValueError("dr_folds and ensemble_size must be at least two")
        if self.model.dr_folds > 20 or self.model.ensemble_size > 100:
            raise ValueError("dr_folds must not exceed 20 and ensemble_size must not exceed 100")
        if self.model.max_iter < 1 or self.model.max_depth < 1:
            raise ValueError("max_iter and max_depth must be positive")
        if self.model.max_iter > MAX_MODEL_ITERATIONS or self.model.max_depth > MAX_MODEL_DEPTH:
            raise ValueError(
                f"max_iter must not exceed {MAX_MODEL_ITERATIONS} and "
                f"max_depth must not exceed {MAX_MODEL_DEPTH}"
            )
        if (
            not _is_finite_real(self.model.learning_rate)
            or self.model.learning_rate <= 0
            or self.model.learning_rate > 1
        ):
            raise ValueError("learning_rate must be in (0, 1]")
        if self.model.min_samples_leaf < 1 or self.model.min_samples_leaf > MAX_DATA_ROWS:
            raise ValueError(f"min_samples_leaf must be in [1, {MAX_DATA_ROWS}]")
        if self.model.categorical_hash_bins < 2 or self.model.categorical_hash_bins > 255:
            raise ValueError("categorical_hash_bins must be in [2, 255]")
        if not 2 <= self.evaluation.bootstrap_replicates <= 100_000:
            raise ValueError("bootstrap_replicates must be in [2, 100000]")
        if not 2 <= self.evaluation.calibration_bins <= 1_000:
            raise ValueError("calibration_bins must be in [2, 1000]")
        if self.evaluation.minimum_slice_rows < 1 or self.evaluation.minimum_slice_rows > MAX_DATA_ROWS:
            raise ValueError(f"minimum_slice_rows must be in [1, {MAX_DATA_ROWS}]")
        if self.evaluation.max_bundle_size_bytes < 1:
            raise ValueError("max_bundle_size_bytes must be positive")
        if (
            not _is_finite_real(self.evaluation.alpha)
            or self.evaluation.alpha <= 0
            or self.evaluation.alpha >= 1
        ):
            raise ValueError("evaluation alpha must be in (0, 1)")
        if (
            not _is_finite_real(self.evaluation.slice_noninferiority_margin)
            or self.evaluation.slice_noninferiority_margin < 0
        ):
            raise ValueError("slice_noninferiority_margin must be finite and nonnegative")
        if (
            not _is_finite_real(self.evaluation.max_inference_ms_per_1000)
            or self.evaluation.max_inference_ms_per_1000 <= 0
        ):
            raise ValueError("max_inference_ms_per_1000 must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tuple_fields(raw: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw)
    for key in ("feature_columns", "categorical_columns"):
        if key in raw:
            if not isinstance(raw[key], (list, tuple)):
                raise ValueError(f"data.{key} must be an array")
            raw[key] = tuple(raw[key])
    return raw


def _tuple_value(raw: dict[str, Any], key: str, default: tuple[Any, ...]) -> tuple[Any, ...]:
    value = raw.get(key, default)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"policy.{key} must be an array")
    return tuple(value)


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_integer(name: str, value: object) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be an object")
    return value


def _reject_unknown(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    if unknown := sorted(set(raw) - allowed):
        raise ValueError(f"unknown {context} fields: {unknown}")


def _field_names(cls: type[Any]) -> set[str]:
    return {item.name for item in fields(cls)}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"nonstandard JSON numeric constant is not allowed: {value}")


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"config path must be a regular file: {path}")
    with path.open("rb") as handle:
        payload = handle.read(MAX_CONFIG_BYTES + 1)
    if len(payload) > MAX_CONFIG_BYTES:
        raise ValueError("config exceeds the 1 MiB safety limit")
    raw = json.loads(
        payload,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    _reject_unknown(raw, _field_names(ProjectConfig), "config")
    data_raw = _section(raw, "data")
    split_raw = _section(raw, "split")
    model_raw = _section(raw, "model")
    policy_raw = _section(raw, "policy")
    evaluation_raw = _section(raw, "evaluation")
    for section, section_raw, cls in (
        ("data", data_raw, DataConfig),
        ("split", split_raw, SplitConfig),
        ("model", model_raw, ModelConfig),
        ("policy", policy_raw, PolicyConfig),
        ("evaluation", evaluation_raw, EvaluationConfig),
    ):
        _reject_unknown(section_raw, _field_names(cls), f"{section} config")
    config = ProjectConfig(
        schema_version=raw.get("schema_version", "1.0"),
        run_name=raw.get("run_name", path.stem),
        seed=raw.get("seed", 20260717),
        data=DataConfig(**_tuple_fields(data_raw)),
        split=SplitConfig(**split_raw),
        model=ModelConfig(**model_raw),
        policy=PolicyConfig(
            **{
                **policy_raw,
                "scenario_capacities": _tuple_value(
                    policy_raw, "scenario_capacities", PolicyConfig.scenario_capacities
                ),
                "risk_multipliers": _tuple_value(
                    policy_raw, "risk_multipliers", PolicyConfig.risk_multipliers
                ),
            }
        ),
        evaluation=EvaluationConfig(**evaluation_raw),
    )
    config.validate()
    return config
