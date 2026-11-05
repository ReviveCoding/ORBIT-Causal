from __future__ import annotations

import errno
import importlib
import json
import os
import shutil
import time
import tracemalloc
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    _load_authenticated_bundle,
    atomic_csv,
    atomic_json,
    environment_metadata,
    finalize_decision_bundle_manifest,
    read_strict_json,
    save_bundle,
    save_decision_bundle_manifest,
    sha256_file,
    stable_json_hash,
    utc_now,
)
from .bundle import DecisionBundle
from .config import ProjectConfig
from .data import (
    CRITEO_V21_EXPECTED_ROWS,
    _categorical_token,
    _validate_criteo_contents,
    _validate_sample_window,
    assign_splits,
    authenticated_criteo_snapshot,
    dataset_fingerprint,
    generate_synthetic_data,
    load_criteo_csv,
    profile_ids,
    sample_row_ids_sha256,
    split_summary,
    subgroup_summary,
    validate_frame,
)
from .logging_utils import JsonlLogger
from .metrics import (
    bootstrap_mean,
    causal_calibration_table,
    incremental_policy_scores,
    paired_bootstrap_mean,
    policy_regret,
    policy_value_curve,
    qini_coefficient,
    set_valued_action_metrics,
    sqrt_pehe,
    unsupported_treatment_rate,
)
from .models import (
    CausalIsotonicCalibrator,
    CrossFittedDRLearner,
    ReferenceStateEffectCalibrator,
    TwoModelUplift,
)
from .policy import capacity_policy, policy_treatment_indicator
from .protocol import (
    PRIMARY_CANDIDATE,
    SECONDARY_COMPARISONS,
    SUPERIORITY_BASELINE,
    implementation_source_fingerprint,
    release_protocol,
    validate_selection_contract,
)
from .reporting import build_figures, build_report

MIN_RELEASE_BOOTSTRAP_REPLICATES = 200
MIN_EVALUATION_CELL_COUNT = 30
EVIDENCE_EXCLUDED = frozenset({"run_manifest.json", "evidence_inventory.json", "run_seal.json"})
RUN_SEAL_SCHEMA = "1.0"


def evaluate_release_gates(
    production_row: pd.Series,
    baseline_row: pd.Series,
    *,
    schema_gate: bool,
    evaluator_gate: bool,
    integrity_gate: bool,
    slice_gate: bool,
    operational_gate: bool,
    max_abstention: float,
) -> dict[str, bool]:
    """Evaluate independent, conjunctive release gates on locked-test metrics."""

    return {
        "G0_integrity": bool(integrity_gate),
        "G1_schema": bool(schema_gate),
        "G2_evaluator": bool(evaluator_gate),
        "G3_value": bool(
            production_row["aipw_ci_lower"] > 0
            and production_row["paired_baseline_improvement_ci_lower"] > 0
        ),
        "G4_calibration": bool(
            production_row["causal_calibration_error"]
            <= 1.10 * baseline_row["causal_calibration_error"]
        ),
        "G5_slices": bool(slice_gate),
        "G6_operational": bool(operational_gate),
        "G7_abstention": bool(production_row["abstention_rate"] <= max_abstention),
    }


def _timed_fit(factory, frame: pd.DataFrame):
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        model = factory().fit(frame)
        seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        return model, seconds, peak / (1024**2)
    finally:
        if started_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()


def _predict_timed(model, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    effect = model.predict_effect(frame)
    uncertainty = model.predict_uncertainty(frame)
    elapsed = time.perf_counter() - start
    milliseconds_per_1000 = elapsed * 1000 * 1000 / max(len(frame), 1)
    return effect, uncertainty, milliseconds_per_1000


def _evaluate_policy(
    effect: np.ndarray,
    uncertainty: np.ndarray,
    model,
    evaluation_model: TwoModelUplift,
    frame: pd.DataFrame,
    config: ProjectConfig,
    risk_multiplier: float,
    guarded: bool,
    seed: int,
    bootstrap_clusters: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, Any, np.ndarray]:
    policy_result = capacity_policy(
        effect,
        uncertainty,
        config.policy.primary_capacity,
        risk_multiplier=risk_multiplier,
        intervention_cost=config.policy.intervention_cost,
        guarded=guarded,
    )
    policy = policy_treatment_indicator(policy_result)
    mu0, mu1 = evaluation_model.predict_outcomes(frame)
    outcome = frame[config.data.outcome].to_numpy(dtype=float)
    treatment = frame[config.data.treatment].to_numpy(dtype=int)
    scores = incremental_policy_scores(
        outcome, treatment, policy, mu0, mu1, evaluation_model.propensity_
    )
    estimate = bootstrap_mean(
        scores,
        config.evaluation.bootstrap_replicates,
        seed,
        config.evaluation.alpha,
        clusters=bootstrap_clusters,
    )
    calibration, calibration_error = causal_calibration_table(
        effect,
        outcome,
        treatment,
        mu0,
        mu1,
        evaluation_model.propensity_,
        config.evaluation.calibration_bins,
    )
    result: dict[str, float] = {
        "aipw_incremental_value": estimate.value,
        "aipw_ci_lower": estimate.lower,
        "aipw_ci_upper": estimate.upper,
        "aipw_standard_error": estimate.standard_error,
        "qini_coefficient": qini_coefficient(
            effect, outcome, treatment, evaluation_model.propensity_
        ),
        "causal_calibration_error": calibration_error,
        "treat_rate": float(np.mean(policy)),
        "abstention_rate": float(np.mean(policy_result.decision == -1)),
        "decision_envelope_width": float(np.mean(2.0 * risk_multiplier * np.asarray(uncertainty))),
    }
    if "true_tau" in frame:
        true_tau = frame["true_tau"].to_numpy(dtype=float)
        result.update(
            {
                "sqrt_pehe": sqrt_pehe(effect, true_tau),
                "policy_regret": policy_regret(policy, true_tau, config.policy.primary_capacity),
                "unsupported_treatment_rate": unsupported_treatment_rate(policy, true_tau),
                "envelope_coverage": float(
                    np.mean(
                        (true_tau >= effect - risk_multiplier * uncertainty)
                        & (true_tau <= effect + risk_multiplier * uncertainty)
                    )
                )
                if risk_multiplier > 0
                else float("nan"),
                **set_valued_action_metrics(
                    policy_result.action_set,
                    true_tau,
                    config.policy.primary_capacity,
                ),
            }
        )
    return result, calibration, policy, policy_result, scores


def _policy_value_on_partition(
    model,
    evaluation_model: TwoModelUplift,
    frame: pd.DataFrame,
    config: ProjectConfig,
    risk_multiplier: float = 0.0,
    guarded: bool = False,
) -> tuple[float, float, float]:
    effect = model.predict_effect(frame)
    uncertainty = model.predict_uncertainty(frame)
    result = capacity_policy(
        effect,
        uncertainty,
        config.policy.primary_capacity,
        risk_multiplier,
        config.policy.intervention_cost,
        guarded,
    )
    policy = policy_treatment_indicator(result)
    mu0, mu1 = evaluation_model.predict_outcomes(frame)
    scores = incremental_policy_scores(
        frame[config.data.outcome].to_numpy(),
        frame[config.data.treatment].to_numpy(),
        policy,
        mu0,
        mu1,
        evaluation_model.propensity_,
    )
    _, calibration_error = causal_calibration_table(
        effect,
        frame[config.data.outcome].to_numpy(),
        frame[config.data.treatment].to_numpy(),
        mu0,
        mu1,
        evaluation_model.propensity_,
        config.evaluation.calibration_bins,
    )
    return float(scores.mean()), calibration_error, float(np.mean(result.decision == -1))


def _source_fingerprint() -> str:
    return implementation_source_fingerprint()


def _slice_masks(frame: pd.DataFrame, config: ProjectConfig) -> list[tuple[str, np.ndarray]]:
    continuous = [
        column
        for column in config.data.feature_columns
        if column not in config.data.categorical_columns
    ]
    column = continuous[0] if continuous else config.data.feature_columns[0]
    series = frame[column]
    if column in config.data.categorical_columns:
        labels = series.map(_categorical_token)
    else:
        numeric = pd.to_numeric(series, errors="raise")
        if numeric.nunique(dropna=True) < 2:
            return [(f"{column}=all", np.ones(len(frame), dtype=bool))]
        labels = pd.qcut(numeric, q=5, duplicates="drop").astype("string").fillna("<MISSING>")
    return [
        (f"{column}:{value}", (labels == value).to_numpy())
        for value in sorted(labels.unique().tolist())
    ]


def _sealed_evidence_files(output_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(output_dir.rglob("*")):
        relative = path.relative_to(output_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"run evidence must not contain symbolic links: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"run evidence contains an unsupported file type: {relative}")
        if relative not in EVIDENCE_EXCLUDED:
            files.append(path)
    return files


def _validate_inventory_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("run evidence inventory contains an unsafe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("run evidence inventory contains an unsafe relative path")
    return value


def _refresh_evidence_inventory(output_dir: Path) -> tuple[str, str]:
    checksums = {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in _sealed_evidence_files(output_dir)
    }
    inventory_path = output_dir / "evidence_inventory.json"
    atomic_json(
        inventory_path,
        {
            "schema_version": "1.0",
            "excluded_mutable_or_recursive": sorted(EVIDENCE_EXCLUDED),
            "files": checksums,
        },
    )
    inventory_hash = sha256_file(inventory_path)
    manifest_path = output_dir / "run_manifest.json"
    manifest = read_strict_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must be a JSON object")
    manifest["evidence_inventory_sha256"] = inventory_hash
    atomic_json(manifest_path, manifest)
    seal_path = output_dir / "run_seal.json"
    atomic_json(
        seal_path,
        {
            "schema_version": RUN_SEAL_SCHEMA,
            "run_id": manifest.get("run_id"),
            "run_manifest_sha256": sha256_file(manifest_path),
            "evidence_inventory_sha256": inventory_hash,
        },
    )
    seal_hash = sha256_file(seal_path)
    verify_evidence_inventory(output_dir, expected_seal_sha256=seal_hash)
    return inventory_hash, seal_hash


def verify_evidence_inventory(
    output_dir: str | Path, *, expected_seal_sha256: str | None = None
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    inventory_path = output_dir / "evidence_inventory.json"
    manifest_path = output_dir / "run_manifest.json"
    seal_path = output_dir / "run_seal.json"
    if (
        not isinstance(expected_seal_sha256, str)
        or len(expected_seal_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_seal_sha256)
    ):
        raise ValueError(
            "expected_seal_sha256 must be 64 hexadecimal characters from a trusted channel"
        )
    try:
        if sha256_file(seal_path) != expected_seal_sha256.lower():
            raise ValueError("trusted run seal SHA-256 does not match run_seal.json")
        seal = read_strict_json(seal_path)
        inventory = read_strict_json(inventory_path)
        manifest = read_strict_json(manifest_path)
        if not all(isinstance(item, dict) for item in (seal, inventory, manifest)):
            raise TypeError("run seal, inventory, and manifest must be JSON objects")
        expected = inventory["files"]
        excluded = inventory["excluded_mutable_or_recursive"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("run evidence inventory is missing or malformed") from error
    if not isinstance(seal, dict) or seal.get("schema_version") != RUN_SEAL_SCHEMA:
        raise ValueError("run seal is malformed or unsupported")
    if set(seal) != {
        "schema_version",
        "run_id",
        "run_manifest_sha256",
        "evidence_inventory_sha256",
    }:
        raise ValueError("run seal contains missing or unknown fields")
    if seal.get("run_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("run seal does not bind the current run manifest")
    if seal.get("evidence_inventory_sha256") != sha256_file(inventory_path):
        raise ValueError("run seal does not bind the current evidence inventory")
    if seal.get("run_id") != manifest.get("run_id"):
        raise ValueError("run seal and manifest run IDs differ")
    if inventory.get("schema_version") != "1.0" or not isinstance(expected, dict):
        raise ValueError("unsupported or malformed evidence inventory")
    if set(inventory) != {"schema_version", "excluded_mutable_or_recursive", "files"}:
        raise ValueError("evidence inventory contains missing or unknown fields")
    if excluded != sorted(EVIDENCE_EXCLUDED):
        raise ValueError("evidence inventory exclusion contract mismatch")
    safe_expected = {
        _validate_inventory_relative_path(path): checksum for path, checksum in expected.items()
    }
    if manifest.get("evidence_inventory_sha256") != sha256_file(inventory_path):
        raise ValueError("run manifest does not bind the current evidence inventory")
    actual_paths = {
        path.relative_to(output_dir).as_posix() for path in _sealed_evidence_files(output_dir)
    }
    if actual_paths != set(safe_expected):
        missing = sorted(set(safe_expected) - actual_paths)
        extra = sorted(actual_paths - set(safe_expected))
        raise ValueError(f"run evidence file set mismatch: missing={missing}, extra={extra}")
    mismatched = sorted(
        relative
        for relative, checksum in safe_expected.items()
        if not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in checksum)
        or sha256_file(output_dir / relative) != checksum
    )
    if mismatched:
        raise ValueError(f"run evidence checksum mismatch: {mismatched}")
    return {
        "status": "verified",
        "run_id": manifest.get("run_id"),
        "file_count": len(safe_expected),
        "evidence_inventory_sha256": sha256_file(inventory_path),
        "run_seal_sha256": sha256_file(seal_path),
    }


def _run_experiment_body(
    config: ProjectConfig,
    output_dir: str | Path,
    logger: JsonlLogger,
    data_path: str | Path | None = None,
    max_rows: int | None = None,
    sample_offset: int = 0,
    selection_contract_path: str | Path | None = None,
    expected_selection_contract_sha256: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = utc_now()
    environment = environment_metadata()
    logger.event("run_started", run_name=config.run_name, config_hash=config.fingerprint())
    _validate_sample_window(max_rows, sample_offset)

    source_validation: dict[str, Any] | None = None
    if config.data.kind == "synthetic":
        if (
            data_path is not None
            or max_rows is not None
            or sample_offset != 0
            or selection_contract_path is not None
            or expected_selection_contract_sha256 is not None
        ):
            raise ValueError(
                "synthetic runs do not accept external data, sampling, or selection contracts"
            )
        frame = generate_synthetic_data(config.data.n_rows, config.seed)
        data_role = "synthetic_fixture"
    elif config.data.kind == "criteo_v2":
        if data_path is None:
            raise ValueError("--data is required for criteo_v2 and raw data is never bundled")
        if max_rows is not None and max_rows >= CRITEO_V21_EXPECTED_ROWS:
            raise ValueError(
                "--max-rows must be smaller than the official file row count; "
                "omit it for a full-benchmark run"
            )
        if max_rows is not None and sample_offset + max_rows > CRITEO_V21_EXPECTED_ROWS:
            raise ValueError("sample_offset + max_rows exceeds the official file row count")
        with authenticated_criteo_snapshot(data_path) as snapshot:
            validation = _validate_criteo_contents(
                snapshot,
                config.data,
                require_official_file=True,
            )
            source_validation = validation.to_dict()
            frame = load_criteo_csv(
                snapshot,
                max_rows=max_rows,
                sample_offset=sample_offset,
            )
        if selection_contract_path is None or expected_selection_contract_sha256 is None:
            raise ValueError(
                "Criteo runs require a pre-sealed selection contract and its trusted SHA-256"
            )
        data_role = (
            "public_randomized_benchmark_subset"
            if max_rows is not None
            else "public_randomized_benchmark"
        )
    else:
        raise ValueError(f"unsupported data kind: {config.data.kind}")

    frame = frame.copy()
    if "source_row_id" not in frame:
        frame["source_row_id"] = np.arange(len(frame), dtype=np.int64)

    source_column_map = {
        "treatment": config.data.treatment,
        "exposure": config.data.exposure,
    }
    if source_column_map != {"treatment": "treatment", "exposure": "exposure"}:
        frame = frame.rename(columns=source_column_map)

    selection_contract: dict[str, Any] | None = None
    selection_contract_hash: str | None = None
    if config.data.kind == "criteo_v2":
        if source_validation is None:
            raise RuntimeError("official source validation is missing")
        selection_contract = validate_selection_contract(
            selection_contract_path,
            expected_sha256=expected_selection_contract_sha256,
            config=config,
            source_validation=source_validation,
            source_row_ids=frame["source_row_id"].to_numpy(),
            max_rows=max_rows,
            sample_offset=sample_offset,
        )
        selection_contract_hash = expected_selection_contract_sha256.lower()
        shutil.copy2(selection_contract_path, output_dir / "selection_contract.json")
        if sha256_file(output_dir / "selection_contract.json") != selection_contract_hash:
            raise RuntimeError("copied selection contract failed authentication")
        logger.event(
            "selection_contract_admitted",
            selection_contract_sha256=selection_contract_hash,
            primary_candidate=PRIMARY_CANDIDATE,
            superiority_baseline=SUPERIORITY_BASELINE,
        )

    quality = validate_frame(frame, config.data)
    frame["split"] = assign_splits(frame, config.data, config.split, config.seed)
    fingerprint = dataset_fingerprint(frame, config.data)
    source_fingerprint = _source_fingerprint()
    run_id = (
        f"{config.run_name}-"
        f"{stable_json_hash([fingerprint, config.fingerprint(), source_fingerprint, selection_contract_hash])[:12]}"
    )
    logger.event(
        "data_admitted", run_id=run_id, dataset_fingerprint=fingerprint, **quality.to_dict()
    )

    summary = split_summary(frame, config.data)
    atomic_json(
        output_dir / "data_quality.json",
        {
            **quality.to_dict(),
            "data_role": data_role,
            "source_validation": source_validation,
            "sample_offset": sample_offset,
            "sample_row_ids_sha256": sample_row_ids_sha256(frame["source_row_id"].to_numpy()),
            "selection_contract_sha256": selection_contract_hash,
        },
    )
    atomic_csv(output_dir / "split_summary.csv", summary)
    atomic_csv(output_dir / "subgroup_distribution.csv", subgroup_summary(frame, config.data))
    parts = {name: part.reset_index(drop=True) for name, part in frame.groupby("split")}
    for required in ("train", "calibration", "policy", "final_test"):
        if required not in parts:
            raise RuntimeError(f"missing split after assignment: {required}")

    training_metrics: dict[str, dict[str, float]] = {}
    c1, seconds, memory = _timed_fit(
        lambda: TwoModelUplift(config.data, config.model, config.seed), parts["train"]
    )
    training_metrics["C1"] = {"training_seconds": seconds, "python_peak_memory_mb": memory}
    logger.event("candidate_fitted", candidate_id="C1", seconds=seconds)

    c3, seconds, memory = _timed_fit(
        lambda: CrossFittedDRLearner(config.data, config.model, config.seed + 100),
        parts["train"],
    )
    training_metrics["C3"] = {"training_seconds": seconds, "python_peak_memory_mb": memory}
    logger.event("candidate_fitted", candidate_id="C3", seconds=seconds)

    start = time.perf_counter()
    isotonic = CausalIsotonicCalibrator(c3).fit(
        parts["calibration"], config.data.treatment, config.data.outcome
    )
    training_metrics["A1"] = {
        "training_seconds": time.perf_counter() - start,
        "python_peak_memory_mb": float("nan"),
    }
    start = time.perf_counter()
    c5 = ReferenceStateEffectCalibrator(c3, config.model, config.seed + 200).fit(
        parts["calibration"], config.data.treatment, config.data.outcome
    )
    training_metrics["C5"] = {
        "training_seconds": time.perf_counter() - start,
        "python_peak_memory_mb": float("nan"),
    }

    policy_frame = parts["policy"]
    c1_policy = _policy_value_on_partition(c1, c1, policy_frame, config)
    c3_policy = _policy_value_on_partition(c3, c1, policy_frame, config)
    iso_policy = _policy_value_on_partition(isotonic, c1, policy_frame, config)
    c5_policy = _policy_value_on_partition(c5, c1, policy_frame, config)
    selection_table = pd.DataFrame(
        [
            {"candidate_id": "C1", "policy_value": c1_policy[0], "calibration_error": c1_policy[1]},
            {"candidate_id": "C3", "policy_value": c3_policy[0], "calibration_error": c3_policy[1]},
            {
                "candidate_id": "A1",
                "policy_value": iso_policy[0],
                "calibration_error": iso_policy[1],
            },
            {"candidate_id": "C5", "policy_value": c5_policy[0], "calibration_error": c5_policy[1]},
        ]
    )
    atomic_csv(output_dir / "policy_selection.csv", selection_table)

    predictive_id = str(
        selection_table.sort_values(
            ["policy_value", "candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]["candidate_id"]
    )
    relative_value_gain = (c5_policy[0] - c3_policy[0]) / max(abs(c3_policy[0]), 1e-6)
    calibration_gain = (c3_policy[1] - c5_policy[1]) / max(c3_policy[1], 1e-6)
    rsec_noninferior = c5_policy[0] >= c3_policy[0] - 0.005 * max(abs(c3_policy[0]), 1e-3)
    rsec_kept = bool((calibration_gain >= 0.10 and rsec_noninferior) or relative_value_gain >= 0.01)
    production_base = c5 if rsec_kept else c3
    production_base_id = "C5" if rsec_kept else "C3"

    risk_rows = []
    for risk_multiplier in config.policy.risk_multipliers:
        value, calibration_error, abstention = _policy_value_on_partition(
            production_base,
            c1,
            policy_frame,
            config,
            risk_multiplier=risk_multiplier,
            guarded=True,
        )
        eligible = abstention <= config.policy.max_abstention
        risk_rows.append(
            {
                "risk_multiplier": risk_multiplier,
                "policy_value": value,
                "calibration_error": calibration_error,
                "abstention_rate": abstention,
                "eligible": eligible,
            }
        )
    risk_table = pd.DataFrame(risk_rows)
    eligible_risk = risk_table.loc[risk_table["eligible"]]
    if eligible_risk.empty:
        selected_risk = 0.0
    else:
        selected_risk = float(
            eligible_risk.sort_values(
                ["policy_value", "risk_multiplier"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0]["risk_multiplier"]
        )
    atomic_csv(output_dir / "risk_selection.csv", risk_table)
    logger.event(
        "policy_frozen",
        predictive_champion=predictive_id,
        production_base=production_base_id,
        rsec_kept=rsec_kept,
        risk_multiplier=selected_risk,
    )

    final = parts["final_test"]
    candidates: list[tuple[str, str, Any, bool, float]] = [
        ("C0", "sanity_response_targeting", c1, False, 0.0),
        ("C1", "industry_two_model_hgb", c1, False, 0.0),
        ("C3", "advanced_cross_fitted_dr", c3, False, 0.0),
        ("A1", "causal_isotonic_ablation", isotonic, False, 0.0),
        ("C5", "specialty_rsec", c5, False, 0.0),
        ("C6", f"guarded_{production_base_id}", production_base, True, selected_risk),
    ]
    comparison_rows: list[dict[str, Any]] = []
    scenario_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    slice_frames: list[pd.DataFrame] = []
    candidate_scores: dict[str, np.ndarray] = {}
    predictions = pd.DataFrame({"source_row_id": final["source_row_id"].to_numpy()})
    mu0_eval, mu1_eval = c1.predict_outcomes(final)
    treatment_final = final[config.data.treatment].to_numpy()
    outcome_final = final[config.data.outcome].to_numpy()
    final_profile_ids = profile_ids(final, config.data).to_numpy()

    for candidate_id, algorithm, model, guarded, risk_multiplier in candidates:
        if candidate_id == "C0":
            mu0_candidate, _ = c1.predict_outcomes(final)
            effect = mu0_candidate
            uncertainty = np.zeros(len(final))
            latency = 0.0
        else:
            effect, uncertainty, latency = _predict_timed(model, final)
        metrics, calibration, policy, policy_result, scores = _evaluate_policy(
            effect,
            uncertainty,
            model,
            c1,
            final,
            config,
            risk_multiplier,
            guarded,
            config.seed + 300 + len(comparison_rows),
            final_profile_ids,
        )
        candidate_scores[candidate_id] = scores
        comparison_rows.append(
            {
                "candidate_id": candidate_id,
                "algorithm": algorithm,
                **metrics,
                **training_metrics.get(
                    candidate_id, {"training_seconds": 0.0, "python_peak_memory_mb": float("nan")}
                ),
                "inference_ms_per_1000": latency,
                "selected_on_policy_split": candidate_id in {predictive_id, "C6"},
                "selection_role": (
                    "prespecified_primary"
                    if candidate_id == PRIMARY_CANDIDATE
                    else "prespecified_superiority_baseline"
                    if candidate_id == SUPERIORITY_BASELINE
                    else "negative_ablation"
                    if candidate_id == "C6"
                    else "secondary_diagnostic"
                ),
                "data_role": data_role,
                "run_id": run_id,
                "dataset_fingerprint": fingerprint,
                "evaluation_split": "final_test",
                "seed": config.seed,
                "config_hash": config.fingerprint(),
                "created_at_utc": created_at,
                "hardware_platform": environment["platform"],
            }
        )
        calibration["candidate_id"] = candidate_id
        calibration_frames.append(calibration)
        curve = policy_value_curve(
            effect,
            outcome_final,
            treatment_final,
            mu0_eval,
            mu1_eval,
            c1.propensity_,
            config.policy.scenario_capacities,
            uncertainty=uncertainty,
            risk_multiplier=risk_multiplier,
            intervention_cost=config.policy.intervention_cost,
            guarded=guarded,
        )
        curve["candidate_id"] = candidate_id
        scenario_frames.append(curve)

        for slice_name, mask in _slice_masks(final, config):
            slice_scores = incremental_policy_scores(
                outcome_final[mask],
                treatment_final[mask],
                policy[mask],
                mu0_eval[mask],
                mu1_eval[mask],
                c1.propensity_,
            )
            slice_frames.append(
                pd.DataFrame(
                    [
                        {
                            "candidate_id": candidate_id,
                            "slice": slice_name,
                            "n": int(mask.sum()),
                            "incremental_value": float(slice_scores.mean()),
                            "treat_rate": float(policy[mask].mean()),
                        }
                    ]
                )
            )

        predictions[f"{candidate_id.lower()}_effect"] = effect
        predictions[f"{candidate_id.lower()}_policy"] = policy
        if candidate_id == "C6":
            predictions["c6_uncertainty"] = uncertainty
            predictions["c6_score"] = effect
            predictions["c6_decision_score"] = policy_result.score
            predictions["c6_lower"] = effect - selected_risk * uncertainty
            predictions["c6_upper"] = effect + selected_risk * uncertainty
            predictions["c6_decision"] = policy_result.decision
            predictions["c6_action_set"] = policy_result.action_set
            predictions["c6_reason_code"] = policy_result.reason_code

    predictions["rsec_correction"] = c5.predict_correction(final)
    if "true_tau" in final:
        predictions["true_tau"] = final["true_tau"].to_numpy()

    comparison = pd.DataFrame(comparison_rows)
    paired_specs = [
        ("primary", PRIMARY_CANDIDATE, SUPERIORITY_BASELINE),
        *(("secondary", PRIMARY_CANDIDATE, baseline) for baseline in SECONDARY_COMPARISONS),
        ("negative_ablation", "C6", "C3"),
    ]
    paired_rows = []
    paired_estimates = {}
    for index, (comparison_role, candidate_id, baseline_id) in enumerate(paired_specs):
        estimate = paired_bootstrap_mean(
            candidate_scores[candidate_id],
            candidate_scores[baseline_id],
            config.evaluation.bootstrap_replicates,
            config.seed + 399 + index,
            config.evaluation.alpha,
            clusters=final_profile_ids,
        )
        paired_estimates[(candidate_id, baseline_id)] = estimate
        paired_rows.append(
            {
                "comparison_role": comparison_role,
                "candidate_id": candidate_id,
                "baseline_id": baseline_id,
                "paired_improvement": estimate.value,
                "paired_improvement_ci_lower": estimate.lower,
                "paired_improvement_ci_upper": estimate.upper,
                "paired_improvement_standard_error": estimate.standard_error,
                "bootstrap_unit": "pretreatment_profile_cluster",
                "bootstrap_replicates": config.evaluation.bootstrap_replicates,
                "confidence_level": 1.0 - config.evaluation.alpha,
            }
        )
    paired_comparison = pd.DataFrame(paired_rows)
    primary_paired = paired_estimates[(PRIMARY_CANDIDATE, SUPERIORITY_BASELINE)]
    comparison["paired_baseline_id"] = None
    comparison["paired_baseline_improvement"] = float("nan")
    comparison["paired_baseline_improvement_ci_lower"] = float("nan")
    comparison["paired_baseline_improvement_ci_upper"] = float("nan")
    comparison["paired_baseline_improvement_standard_error"] = float("nan")
    primary_mask = comparison["candidate_id"] == PRIMARY_CANDIDATE
    comparison.loc[primary_mask, "paired_baseline_id"] = SUPERIORITY_BASELINE
    comparison.loc[primary_mask, "paired_baseline_improvement"] = primary_paired.value
    comparison.loc[primary_mask, "paired_baseline_improvement_ci_lower"] = primary_paired.lower
    comparison.loc[primary_mask, "paired_baseline_improvement_ci_upper"] = primary_paired.upper
    comparison.loc[primary_mask, "paired_baseline_improvement_standard_error"] = (
        primary_paired.standard_error
    )
    scenarios = pd.concat(scenario_frames, ignore_index=True)
    calibration_table = pd.concat(calibration_frames, ignore_index=True)
    slices = pd.concat(slice_frames, ignore_index=True)
    atomic_csv(output_dir / "candidate_comparison.csv", comparison)
    atomic_csv(output_dir / "paired_comparison.csv", paired_comparison)
    atomic_csv(output_dir / "decision_scenarios.csv", scenarios)
    atomic_csv(output_dir / "calibration_table.csv", calibration_table)
    atomic_csv(output_dir / "slice_metrics.csv", slices)
    atomic_csv(output_dir / "predictions.csv", predictions)

    reason_codes = (
        predictions.groupby("c6_reason_code", dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={"c6_reason_code": "reason_code"})
    )
    reason_codes["share"] = reason_codes["n"] / len(predictions)
    atomic_csv(output_dir / "reason_code_report.csv", reason_codes)

    failure_cases = predictions.copy()
    if "true_tau" in failure_cases:
        failure_cases["failure_score"] = np.abs(
            failure_cases["c6_score"] - failure_cases["true_tau"]
        )
        failure_definition = "absolute effect error against generated ground truth"
    else:
        failure_cases["failure_score"] = failure_cases["c6_uncertainty"] + np.abs(
            failure_cases["rsec_correction"]
        )
        failure_definition = "high uncertainty plus large specialty correction proxy"
    failure_cases = failure_cases.nlargest(min(50, len(failure_cases)), "failure_score")
    failure_cases["failure_definition"] = failure_definition
    atomic_csv(output_dir / "failure_cases.csv", failure_cases)

    production_row = comparison.loc[comparison["candidate_id"] == PRIMARY_CANDIDATE].iloc[0]
    baseline_row = comparison.loc[comparison["candidate_id"] == SUPERIORITY_BASELINE].iloc[0]
    primary_slices = slices.loc[slices["candidate_id"] == PRIMARY_CANDIDATE].set_index("slice")
    baseline_slices = slices.loc[slices["candidate_id"] == SUPERIORITY_BASELINE].set_index("slice")
    aligned_slices = primary_slices.join(
        baseline_slices,
        how="inner",
        lsuffix="_primary",
        rsuffix="_baseline",
    )
    slice_gate = bool(
        len(aligned_slices) == len(primary_slices) == len(baseline_slices)
        and len(aligned_slices) > 0
        and (aligned_slices["n_primary"] >= config.evaluation.minimum_slice_rows).all()
        and np.isfinite(aligned_slices["incremental_value_primary"]).all()
        and np.isfinite(aligned_slices["incremental_value_baseline"]).all()
        and (
            aligned_slices["incremental_value_primary"]
            >= aligned_slices["incremental_value_baseline"]
            - config.evaluation.slice_noninferiority_margin
        ).all()
    )
    joint_cell_counts = {
        f"treatment={arm},outcome={label}": int(
            np.sum((treatment_final == arm) & (outcome_final == label))
        )
        for arm in (0, 1)
        for label in (0, 1)
    }
    evaluator_gate = bool(
        np.isfinite(comparison["aipw_incremental_value"]).all()
        and config.evaluation.bootstrap_replicates >= MIN_RELEASE_BOOTSTRAP_REPLICATES
        and set(np.unique(treatment_final)) == {0, 1}
        and min(joint_cell_counts.values()) >= MIN_EVALUATION_CELL_COUNT
    )
    bundle = DecisionBundle(
        model=c1,
        candidate_id=PRIMARY_CANDIDATE,
        feature_columns=config.data.feature_columns,
        capacity=config.policy.primary_capacity,
        risk_multiplier=0.0,
        intervention_cost=config.policy.intervention_cost,
        run_id=run_id,
        dataset_fingerprint=fingerprint,
    )
    logger.event("artifact_publication_started")
    bundle_path = output_dir / "artifacts" / "production_bundle.joblib"
    bundle_hash = save_bundle(bundle_path, bundle)
    bundle_manifest_path = save_decision_bundle_manifest(
        bundle_path,
        bundle_hash,
        run_id=run_id,
        dataset_fingerprint=fingerprint,
        candidate_id=PRIMARY_CANDIDATE,
    )
    bundle_manifest_hash = sha256_file(bundle_manifest_path)
    portfolio_hash = save_bundle(
        output_dir / "artifacts" / "candidate_portfolio.joblib",
        {"C0": c1, "C1": c1, "C3": c3, "A1": isotonic, "C5": c5},
    )
    logger.event("artifact_publication_completed")
    bundle_size_bytes = bundle_path.stat().st_size
    operational_gate = bool(
        0 <= production_row["inference_ms_per_1000"] <= config.evaluation.max_inference_ms_per_1000
        and bundle_size_bytes <= config.evaluation.max_bundle_size_bytes
        and np.isfinite(production_row["training_seconds"])
        and production_row["training_seconds"] >= 0
    )
    integrity_gate = False
    schema_gate = False
    try:
        verified_bundle = _load_authenticated_bundle(
            bundle_path,
            expected_sha256=bundle_hash,
            expected_manifest_sha256=bundle_manifest_hash,
            manifest_path=bundle_manifest_path,
            require_release_pass=False,
        )
        probe = verified_bundle.score(final.head(min(8, len(final))))
        expected_output_columns = {
            "row_key",
            "eligible",
            "tau_calibrated",
            "decision_uncertainty",
            "net_value_lower",
            "action_set",
            "decision",
            "rank",
            "reason_code",
            "candidate_id",
            "run_id",
            "allocation_scope",
            "allocation_batch_size",
            "capacity_slots",
        }
        schema_gate = bool(
            set(probe.columns) == expected_output_columns
            and probe["decision"].isin([-1, 0, 1]).all()
            and (probe["allocation_scope"] == "request_batch").all()
        )
        integrity_gate = bool(
            sha256_file(bundle_path) == bundle_hash
            and len(probe) > 0
            and np.isfinite(probe["tau_calibrated"]).all()
        )
    except (OSError, ValueError, TypeError, AttributeError):
        integrity_gate = False
    gates = evaluate_release_gates(
        production_row,
        baseline_row,
        schema_gate=schema_gate,
        evaluator_gate=evaluator_gate,
        integrity_gate=integrity_gate,
        slice_gate=slice_gate,
        operational_gate=operational_gate,
        max_abstention=config.policy.max_abstention,
    )
    release_pass = all(gates.values())
    bundle_manifest_hash = finalize_decision_bundle_manifest(
        bundle_manifest_path,
        expected_manifest_sha256=bundle_manifest_hash,
        release_status="PASS" if release_pass else "BLOCKED",
    )
    operational = {
        "run_id": run_id,
        "candidates": training_metrics,
        "bundle_size_bytes": bundle_size_bytes,
        "bundle_sha256": bundle_hash,
        "bundle_manifest_sha256": bundle_manifest_hash,
        "candidate_portfolio_size_bytes": (output_dir / "artifacts" / "candidate_portfolio.joblib")
        .stat()
        .st_size,
        "candidate_portfolio_sha256": portfolio_hash,
        "environment": environment,
    }
    atomic_json(output_dir / "operational_metrics.json", operational)
    atomic_json(
        output_dir / "metrics.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "dataset_fingerprint": fingerprint,
            "config_hash": config.fingerprint(),
            "data_role": data_role,
            "source_validation": source_validation,
            "primary_metric": "aipw_incremental_policy_value_at_capacity",
            "primary_capacity": config.policy.primary_capacity,
            "confidence_level": 1.0 - config.evaluation.alpha,
            "bootstrap_unit": "pretreatment_profile_cluster",
            "selection_contract_sha256": selection_contract_hash,
            "release_protocol": release_protocol(config),
            "joint_treatment_outcome_counts": joint_cell_counts,
            "candidates": comparison.to_dict(orient="records"),
        },
    )
    verdict_prefix = {
        "synthetic_fixture": "FIXTURE",
        "public_randomized_benchmark": "PUBLIC_BENCHMARK",
        "public_randomized_benchmark_subset": "PUBLIC_BENCHMARK_SUBSET",
    }[data_role]
    claim_boundary = (
        "fixture-only; no public-data, GPU, financial, or production claim"
        if data_role == "synthetic_fixture"
        else "public randomized benchmark only; no GPU, financial, or production-impact claim"
        if data_role == "public_randomized_benchmark"
        else (
            "public randomized benchmark subset only; not a full-benchmark, GPU, financial, "
            "or production-impact claim"
        )
    )
    atomic_json(
        output_dir / "qualification_manifest.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "release_verdict": f"{verdict_prefix}_RELEASE_PASS"
            if release_pass
            else f"{verdict_prefix}_RELEASE_BLOCKED",
            "gates": gates,
            "predictive_champion": predictive_id,
            "production_base": production_base_id,
            "production_candidate": PRIMARY_CANDIDATE,
            "superiority_baseline": SUPERIORITY_BASELINE,
            "production_champion": PRIMARY_CANDIDATE if release_pass else None,
            "c6_role": "negative_ablation",
            "rsec_kept": rsec_kept,
            "selected_risk_multiplier": selected_risk,
            "confidence_level": 1.0 - config.evaluation.alpha,
            "minimum_release_bootstrap_replicates": MIN_RELEASE_BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "pretreatment_profile_cluster",
            "claim_boundary": claim_boundary,
            "source_validation": source_validation,
            "selection_contract_sha256": selection_contract_hash,
            "sample_offset": sample_offset,
            "sample_row_ids_sha256": sample_row_ids_sha256(frame["source_row_id"].to_numpy()),
        },
    )
    atomic_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "created_at_utc": created_at,
            "dataset_fingerprint": fingerprint,
            "config_hash": config.fingerprint(),
            "source_fingerprint": source_fingerprint,
            "report_source_fingerprint": source_fingerprint,
            "report_generated_at_utc": created_at,
            "seed": config.seed,
            "data_role": data_role,
            "source_validation": source_validation,
            "selection_contract_sha256": selection_contract_hash,
            "selection_contract": selection_contract,
            "sample_contract": {
                "max_rows": max_rows,
                "sample_offset": sample_offset,
                "sample_row_count": len(frame),
                "sample_row_ids_sha256": sample_row_ids_sha256(frame["source_row_id"].to_numpy()),
            },
            "split_contract": asdict(config.split),
            "policy_contract": asdict(config.policy),
            "evaluation_contract": {
                **asdict(config.evaluation),
                "confidence_level": 1.0 - config.evaluation.alpha,
                "minimum_release_bootstrap_replicates": MIN_RELEASE_BOOTSTRAP_REPLICATES,
                "minimum_evaluation_cell_count": MIN_EVALUATION_CELL_COUNT,
                "joint_treatment_outcome_counts": joint_cell_counts,
                "bootstrap_unit": "pretreatment_profile_cluster",
                "primary_candidate": PRIMARY_CANDIDATE,
                "superiority_baseline": SUPERIORITY_BASELINE,
                "secondary_comparisons": list(SECONDARY_COMPARISONS),
            },
            "environment": environment,
            "artifact_checksums": {
                "production_bundle.joblib": bundle_hash,
                "production_bundle.manifest.json": bundle_manifest_hash,
                "candidate_portfolio.joblib": portfolio_hash,
            },
        },
    )

    logger.event("report_generation_started")
    figure_outputs = build_figures(output_dir)
    report_outputs = build_report(output_dir)
    logger.event("report_generation_completed")

    candidate_types = {
        "C0": "sanity",
        "C1": "production",
        "C3": "advanced",
        "A1": "specialty",
        "C5": "hybrid",
        "C6": "negative_ablation",
    }
    portfolio = []
    for row in comparison.to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        specialty_components = {
            "A1": ["causal_isotonic"],
            "C5": ["RSEC"],
            "C6": ["RSEC_if_qualified", "CAS_decision_envelope", "capacity_guard"],
        }.get(candidate_id, [])
        portfolio.append(
            {
                "candidate_id": candidate_id,
                "type": candidate_types[candidate_id],
                "algorithm": row["algorithm"],
                "features_or_representation": (
                    "pretreatment features; stable categorical hashing; "
                    "median-imputed continuous values"
                ),
                "specialty_components": specialty_components,
                "config_path": (
                    "configs/demo.json"
                    if data_role == "synthetic_fixture"
                    else "configs/criteo_v2.json"
                ),
                "artifact_path": (
                    "artifacts/production_bundle.joblib"
                    if candidate_id == PRIMARY_CANDIDATE
                    else f"artifacts/candidate_portfolio.joblib#{candidate_id}"
                ),
                "metrics": {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "candidate_id",
                        "algorithm",
                        "training_seconds",
                        "python_peak_memory_mb",
                        "inference_ms_per_1000",
                        "run_id",
                        "dataset_fingerprint",
                        "config_hash",
                        "hardware_platform",
                        "created_at_utc",
                    }
                },
                "operational_metrics": {
                    "training_seconds": row.get("training_seconds"),
                    "python_peak_memory_mb": row.get("python_peak_memory_mb"),
                    "inference_ms_per_1000": row.get("inference_ms_per_1000"),
                },
                "status": (
                    ("production_selected" if release_pass else "release_blocked_candidate")
                    if candidate_id == PRIMARY_CANDIDATE
                    else "negative_ablation"
                    if candidate_id == "C6"
                    else "predictive_selected"
                    if candidate_id == predictive_id
                    else "evaluated_challenger"
                ),
            }
        )

    c3_row = comparison.loc[comparison["candidate_id"] == "C3"].iloc[0]
    c5_row = comparison.loc[comparison["candidate_id"] == "C5"].iloc[0]
    c6_row = comparison.loc[comparison["candidate_id"] == "C6"].iloc[0]
    atomic_json(
        output_dir / "project_creation_handoff.json",
        {
            "project_depth": {
                "foundation_complete": True,
                "industry_baseline_complete": True,
                "advanced_technique_complete": True,
                "specialty_module_complete": True,
                "product_layer_complete": True,
            },
            "candidate_portfolio": portfolio,
            "specialty_transfer": [
                {
                    "source_method": "PHT/reference-state alignment principle",
                    "adapted_name": "Reference-State Effect Calibration (RSEC)",
                    "adaptation_type": "inspired",
                    "project_problem_addressed": (
                        "state-conditioned systematic bias in heterogeneous effect estimates"
                    ),
                    "implementation_path": (
                        "src/orbit_causal/models.py::ReferenceStateEffectCalibrator"
                    ),
                    "test_path": "tests/test_models.py",
                    "ablation_result": {
                        "c3_policy_value": c3_row["aipw_incremental_value"],
                        "c5_policy_value": c5_row["aipw_incremental_value"],
                        "c3_calibration_error": c3_row["causal_calibration_error"],
                        "c5_calibration_error": c5_row["causal_calibration_error"],
                        "kept_by_policy_split_gate": rsec_kept,
                    },
                    "limitations": [
                        "Inspired adaptation, not the original PHT algorithm",
                        "Calibration split is required and correction is clipped",
                        "Fixture evidence does not establish external generalization",
                    ],
                },
                {
                    "source_method": "Probability/ABC-style candidate envelope principle",
                    "adapted_name": "Candidate Action Set (CAS)",
                    "adaptation_type": "derived",
                    "project_problem_addressed": (
                        "unsafe forced binary treatment decisions under effect uncertainty"
                    ),
                    "implementation_path": "src/orbit_causal/policy.py::capacity_policy",
                    "test_path": "tests/test_policy.py; tests/test_metrics.py",
                    "ablation_result": {
                        "c6_policy_value": c6_row["aipw_incremental_value"],
                        "abstention_rate": c6_row["abstention_rate"],
                        "action_set_recall_fixture_only": c6_row.get("action_set_recall"),
                        "action_set_precision_fixture_only": c6_row.get("action_set_precision"),
                    },
                    "limitations": [
                        "Ensemble dispersion is a decision heuristic, not an individual causal confidence interval",
                        "Action-set ground truth metrics exist only on generated fixtures",
                    ],
                },
            ],
            "champion_selection": {
                "predictive_champion": predictive_id,
                "production_candidate": PRIMARY_CANDIDATE,
                "superiority_baseline": SUPERIORITY_BASELINE,
                "production_champion": PRIMARY_CANDIDATE if release_pass else None,
                "selection_criteria": [
                    "pre-sealed C1 primary-candidate contract",
                    "C1 minus C3 paired cluster-bootstrap lower bound",
                    "causal calibration nondegradation",
                    "abstention capacity",
                    "artifact integrity",
                    "locked-test reporting only after freeze",
                ],
                "tradeoffs": [
                    f"RSEC production base decision: {'kept' if rsec_kept else 'rejected'}",
                    f"Selected risk multiplier: {selected_risk}",
                    "Guarded policy may sacrifice automation for safer action sets",
                ],
            },
            "outputs": {
                "numeric": [
                    "metrics.json",
                    "candidate_comparison.csv",
                    "paired_comparison.csv",
                    "decision_scenarios.csv",
                    "calibration_table.csv",
                    "slice_metrics.csv",
                    "subgroup_distribution.csv",
                    "failure_cases.csv",
                    "reason_code_report.csv",
                ],
                "visualizations": figure_outputs,
                "reports": [str(path.relative_to(output_dir)) for path in report_outputs],
                "cards": [
                    "docs/data_card.md",
                    "docs/model_card.md",
                    "docs/evaluation_card.md",
                    "docs/release_card.md",
                ],
                "api_or_cli_examples": [
                    "orbit-causal demo --config configs/demo.json --output runs/demo",
                    "orbit-causal score --bundle runs/demo/artifacts/production_bundle.joblib --expected-sha256 <trusted-bundle-sha256> --expected-manifest-sha256 <trusted-manifest-sha256> --input examples/score_input.csv --output runs/demo/batch_decisions.csv",
                ],
            },
            "experiment_lineage": {
                "run_ids": [run_id],
                "dataset_fingerprints": [fingerprint],
                "config_hashes": [config.fingerprint()],
                "artifact_checksums": [bundle_hash, portfolio_hash],
            },
            "claim_boundary": claim_boundary,
            "known_uncompleted_qualification": [
                *(
                    ["public Criteo benchmark run"]
                    if data_role == "synthetic_fixture"
                    else ["full public Criteo benchmark run"]
                    if data_role == "public_randomized_benchmark_subset"
                    else []
                ),
                "repeated-seed finalist sensitivity",
                "GPU validation",
                "temporal or campaign holdout",
                "financial-domain validation",
            ],
        },
    )
    logger.event(
        "run_completed",
        run_id=run_id,
        release_pass=release_pass,
        evidence_status="sealed immediately after this final event",
    )
    logger.flush()
    _refresh_evidence_inventory(output_dir)
    return output_dir


def _run_experiment_inplace(
    config: ProjectConfig,
    output_dir: str | Path,
    data_path: str | Path | None = None,
    max_rows: int | None = None,
    sample_offset: int = 0,
    selection_contract_path: str | Path | None = None,
    expected_selection_contract_sha256: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "events.jsonl")
    try:
        return _run_experiment_body(
            config,
            output_dir,
            logger,
            data_path=data_path,
            max_rows=max_rows,
            sample_offset=sample_offset,
            selection_contract_path=selection_contract_path,
            expected_selection_contract_sha256=expected_selection_contract_sha256,
        )
    finally:
        logger.close()


def _validate_output_target(
    output_dir: Path,
    *,
    replace_existing_run: bool,
    expected_existing_run_seal_sha256: str | None,
) -> None:
    resolved = output_dir.resolve(strict=False)
    cwd = Path.cwd().resolve()
    protected = {cwd, Path.home().resolve(), Path(resolved.anchor), *cwd.parents}
    if resolved in protected:
        raise ValueError(f"refusing protected output target: {resolved}")
    if output_dir.is_symlink():
        raise ValueError("output target must not be a symbolic link")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output target must be a directory")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        if not replace_existing_run:
            raise ValueError(
                "refusing to replace a nonempty directory without --replace-existing-run"
            )
        try:
            verify_evidence_inventory(
                output_dir,
                expected_seal_sha256=expected_existing_run_seal_sha256,
            )
        except ValueError as error:
            raise ValueError(
                "refusing to replace a nonempty directory that is not a verified ORBIT run"
            ) from error


@contextmanager
def _output_lock(output_dir: Path):
    """Hold a process-scoped advisory lock for an output path.

    The coordination file is intentionally persistent. The operating system
    releases the advisory lock when a process exits, including an abrupt exit,
    so a stale file can never permanently block a later run.
    """

    canonical_output = output_dir.resolve(strict=False)
    lock = canonical_output.parent / f".{canonical_output.name}.orbit-lock"
    if lock.is_symlink():
        raise RuntimeError(f"refusing symbolic-link output lock: {lock}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock, flags, 0o600)
    except OSError as error:
        raise RuntimeError(f"unable to open output lock: {lock}") from error
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    try:
        try:
            if os.name == "nt":
                msvcrt = importlib.import_module("msvcrt")
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"\0")
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError(f"another run owns the output lock: {lock}") from error
            raise RuntimeError(f"unable to acquire output lock: {lock}") from error

        owner = json.dumps(
            {"pid": os.getpid(), "created_at_utc": utc_now()},
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        handle.seek(0)
        handle.truncate()
        handle.write(owner)
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        if acquired:
            if os.name == "nt":
                msvcrt = importlib.import_module("msvcrt")
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def regenerate_reports(run_dir: str | Path, *, expected_seal_sha256: str) -> dict[str, Any]:
    """Regenerate all reports transactionally and return the new trusted hashes."""

    run_dir = Path(run_dir)
    _validate_output_target(
        run_dir,
        replace_existing_run=True,
        expected_existing_run_seal_sha256=expected_seal_sha256,
    )
    with _output_lock(run_dir):
        _validate_output_target(
            run_dir,
            replace_existing_run=True,
            expected_existing_run_seal_sha256=expected_seal_sha256,
        )
        staging = run_dir.parent / f".{run_dir.name}.report-staging-{uuid.uuid4().hex}"
        rollback: Path | None = None
        try:
            # A byte copy is an immutable snapshot. Hard links would let a
            # concurrent write to the verified source mutate staging after the
            # seal check and before report publication.
            shutil.copytree(run_dir, staging, copy_function=shutil.copy2)
            verify_evidence_inventory(staging, expected_seal_sha256=expected_seal_sha256)
            manifest_path = staging / "run_manifest.json"
            manifest = read_strict_json(manifest_path)
            if not isinstance(manifest, dict):
                raise ValueError("run manifest must be a JSON object")
            manifest["report_source_fingerprint"] = _source_fingerprint()
            manifest["report_generated_at_utc"] = utc_now()
            atomic_json(manifest_path, manifest)
            figures = build_figures(staging)
            markdown, html_report = build_report(staging)
            inventory_hash, seal_hash = _refresh_evidence_inventory(staging)
            rollback = run_dir.parent / f"{run_dir.name}.rollback-{uuid.uuid4().hex[:12]}"
            os.replace(run_dir, rollback)
            os.replace(staging, run_dir)
            shutil.rmtree(rollback, ignore_errors=True)
            return {
                "figures": figures,
                "report": str(run_dir / markdown.relative_to(staging)),
                "html": str(run_dir / html_report.relative_to(staging)),
                "evidence_inventory_sha256": inventory_hash,
                "run_seal_sha256": seal_hash,
            }
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if rollback is not None and rollback.exists() and not run_dir.exists():
                os.replace(rollback, run_dir)
            raise


def run_experiment(
    config: ProjectConfig,
    output_dir: str | Path,
    data_path: str | Path | None = None,
    max_rows: int | None = None,
    sample_offset: int = 0,
    selection_contract_path: str | Path | None = None,
    expected_selection_contract_sha256: str | None = None,
    replace_existing_run: bool = False,
    expected_existing_run_seal_sha256: str | None = None,
) -> Path:
    """Run into a staging directory and publish only a complete run."""

    config.validate()
    output_dir = Path(output_dir)
    if not isinstance(replace_existing_run, bool):
        raise ValueError("replace_existing_run must be a boolean")
    _validate_output_target(
        output_dir,
        replace_existing_run=replace_existing_run,
        expected_existing_run_seal_sha256=expected_existing_run_seal_sha256,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock(output_dir):
        _validate_output_target(
            output_dir,
            replace_existing_run=replace_existing_run,
            expected_existing_run_seal_sha256=expected_existing_run_seal_sha256,
        )
        staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
        rollback: Path | None = None
        try:
            _run_experiment_inplace(
                config,
                staging,
                data_path=data_path,
                max_rows=max_rows,
                sample_offset=sample_offset,
                selection_contract_path=selection_contract_path,
                expected_selection_contract_sha256=expected_selection_contract_sha256,
            )
            if output_dir.exists():
                if output_dir.is_dir() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
                else:
                    rollback = output_dir.parent / (
                        f"{output_dir.name}.rollback-{uuid.uuid4().hex[:12]}"
                    )
                    os.replace(output_dir, rollback)
            os.replace(staging, output_dir)
            if rollback is not None:
                shutil.rmtree(rollback, ignore_errors=True)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if rollback is not None and rollback.exists() and not output_dir.exists():
                os.replace(rollback, output_dir)
            raise
    return output_dir
