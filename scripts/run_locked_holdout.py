from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from orbit_causal.config import load_config
from orbit_causal.data import (
    CRITEO_V21_EXPECTED_GZIP_BYTES,
    CRITEO_V21_EXPECTED_GZIP_SHA256,
    CRITEO_V21_EXPECTED_ROWS,
    deterministic_sample_row_ids,
    load_criteo_csv,
    profile_ids,
    sample_row_ids_sha256,
)
from orbit_causal.locked_evaluation import (
    calibration_rmse_linearization,
    paired_cluster_bootstrap,
)
from orbit_causal.metrics import incremental_policy_scores
from orbit_causal.models import doubly_robust_score
from orbit_causal.policy import capacity_policy, policy_treatment_indicator


SCHEMA = "orbit-causal-v039-locked-noninferiority-v1"
PRIMARY = "E7"
BASELINE = "C3"
PRIOR_WINDOWS = ((0, 250_000), (250_000, 250_000), (500_000, 250_000))
HOLDOUT_OFFSET = 750_000
HOLDOUT_ROWS = 2_000_000
CAPACITY = 0.20
BOOTSTRAP_REPLICATES = 4_000
ONE_SIDED_ALPHA = 0.05
VALUE_NONINFERIORITY_MARGIN = -0.00025
CALIBRATION_SUPERIORITY_MARGIN = 0.0
CALIBRATION_BINS = 10
EXPECTED_ARTIFACT_SHA256 = "047dbfd1f0ae48418b0de15e6f5da299a5d6fb575ecf81790afacf7a0a5df004"
EXPECTED_PARENT_CONTRACT_SHA256 = "be16c63128760b4f24d1bfc461a9d97e760b221c02b6b9eb56fc35b57bd21412"
MIN_FREE_BYTES = 5_000_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def package_fingerprint(package_dir: Path) -> str:
    files = sorted(path for path in package_dir.rglob("*.py") if "__pycache__" not in path.parts)
    return stable_hash(
        {path.relative_to(package_dir).as_posix(): sha256_file(path) for path in files}
    )


def source_identity(path: Path) -> dict[str, object]:
    result = {
        "rows": CRITEO_V21_EXPECTED_ROWS,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if (
        result["size_bytes"] != CRITEO_V21_EXPECTED_GZIP_BYTES
        or result["sha256"] != CRITEO_V21_EXPECTED_GZIP_SHA256
    ):
        raise ValueError("official Criteo v2.1 identity mismatch")
    return result


def artifact_identity(artifact: Path, parent_contract: Path) -> dict[str, object]:
    artifact_sha = sha256_file(artifact)
    parent_sha = sha256_file(parent_contract)
    if artifact_sha != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("v0.3.8 candidate artifact identity mismatch")
    if parent_sha != EXPECTED_PARENT_CONTRACT_SHA256:
        raise ValueError("v0.3.8 parent selection contract identity mismatch")
    return {
        "path_role": "frozen_v038_E7_C3_portfolio",
        "size_bytes": artifact.stat().st_size,
        "sha256": artifact_sha,
        "parent_selection_contract_sha256": parent_sha,
    }


def row_contract() -> dict[str, object]:
    prior = []
    prior_ids = []
    for offset, rows in PRIOR_WINDOWS:
        ids = deterministic_sample_row_ids(CRITEO_V21_EXPECTED_ROWS, rows, offset)
        prior_ids.append(ids)
        prior.append(
            {
                "offset": offset,
                "rows": rows,
                "row_ids_sha256": sample_row_ids_sha256(ids),
            }
        )
    prior_all = np.concatenate(prior_ids)
    holdout = deterministic_sample_row_ids(
        CRITEO_V21_EXPECTED_ROWS, HOLDOUT_ROWS, HOLDOUT_OFFSET
    )
    intersections = [int(np.intersect1d(ids, holdout, assume_unique=True).size) for ids in prior_ids]
    if any(intersections) or np.unique(prior_all).size != len(prior_all):
        raise RuntimeError("locked holdout overlaps prior ORBIT evaluation windows")
    return {
        "prior_windows": prior,
        "prior_combined_row_ids_sha256": sample_row_ids_sha256(prior_all),
        "holdout": {
            "offset": HOLDOUT_OFFSET,
            "rows": HOLDOUT_ROWS,
            "priority_rank_window": [HOLDOUT_OFFSET, HOLDOUT_OFFSET + HOLDOUT_ROWS],
            "row_ids_sha256": sample_row_ids_sha256(holdout),
            "intersection_counts_with_prior_windows": intersections,
            "role": "single_use_locked_release_evaluation_only",
        },
    }


def contract_payload(args: argparse.Namespace) -> dict[str, object]:
    rows = row_contract()
    return {
        "schema": SCHEMA,
        "source": source_identity(args.data),
        "implementation": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "package_fingerprint": package_fingerprint(args.package_dir),
            "config_sha256": sha256_file(args.config),
        },
        "frozen_artifact": artifact_identity(args.artifact, args.parent_contract),
        "candidate": {
            "primary": PRIMARY,
            "baseline": BASELINE,
            "effect": "v0.3.8 positive-affine(0.7*C1 + 0.3*C5)",
            "artifact_retraining_permitted": False,
        },
        "data_roles": rows,
        "evaluation": {
            "capacity": CAPACITY,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "cluster": "exact_pretreatment_profile",
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "calibration_bins": CALIBRATION_BINS,
            "calibration_inference": "paired_cluster_bootstrap_of_empirical_influence_function",
            "decision_rule": "intersection_union_both_co_primary_gates_must_pass",
        },
        "co_primary_release_rules": {
            "value_noninferiority": {
                "estimand": "E7_minus_C3_AIPW_policy_value_at_20pct_capacity",
                "margin": VALUE_NONINFERIORITY_MARGIN,
                "pass": "one_sided_95pct_lower_bound_strictly_greater_than_margin",
            },
            "calibration_superiority": {
                "estimand": "RMSE_E7_minus_RMSE_C3",
                "margin": CALIBRATION_SUPERIORITY_MARGIN,
                "pass": "one_sided_95pct_upper_bound_strictly_less_than_zero",
            },
        },
        "guardrails": {
            "candidate_standalone_two_sided_95pct_ci_lower_gt": 0.0,
            "minimum_slice_difference": -0.005,
            "max_inference_ms_per_1000": 100.0,
            "max_artifact_size_bytes": 100_000_000,
        },
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def make_policy(effect: np.ndarray) -> np.ndarray:
    result = capacity_policy(effect, np.zeros(len(effect)), CAPACITY, 0.0, 0.0, False)
    return policy_treatment_indicator(result)


def interval(draws: pd.Series, alpha: float, two_sided: bool = True) -> dict[str, float]:
    if two_sided:
        lower, upper = draws.quantile([alpha / 2.0, 1.0 - alpha / 2.0])
    else:
        lower, upper = draws.quantile([alpha, 1.0 - alpha])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "standard_error": float(draws.std(ddof=1)),
    }


def preflight(output: Path) -> None:
    if output.exists():
        raise ValueError("refusing to replace an existing output directory")
    free = shutil.disk_usage(output.parent.resolve()).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"insufficient free space: required={MIN_FREE_BYTES} available={free}")


def validate_portfolio(portfolio: object) -> dict[str, object]:
    if not isinstance(portfolio, dict) or set(("C1", "C3", "C5", "E7")) - set(portfolio):
        raise ValueError("frozen artifact does not contain the sealed v0.3.8 portfolio")
    e7 = portfolio["E7"]
    if e7.get("weights") != {"C1": 0.7, "C5": 0.3}:
        raise ValueError("frozen E7 weights do not match the sealed definition")
    for name in ("affine_intercept", "affine_slope"):
        if name not in e7 or not np.isfinite(float(e7[name])):
            raise ValueError(f"frozen E7 is missing finite {name}")
    if float(e7["affine_slope"]) <= 0.0:
        raise ValueError("frozen E7 affine slope must be positive")
    return portfolio


def run(args: argparse.Namespace) -> None:
    preflight(args.output)
    if sha256_file(args.contract) != args.expected_contract_sha256.lower():
        raise ValueError("selection contract SHA-256 mismatch")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected = contract_payload(args)
    if contract != expected:
        raise ValueError("selection contract does not match implementation or inputs")

    args.output.mkdir(parents=True)
    (args.output / "selection_contract.json").write_bytes(args.contract.read_bytes())
    started = time.perf_counter()
    portfolio = validate_portfolio(joblib.load(args.artifact))
    config = load_config(args.config)

    holdout = load_criteo_csv(args.data, HOLDOUT_ROWS, HOLDOUT_OFFSET)
    holdout_ids = holdout["source_row_id"].to_numpy(dtype=np.int64)
    sealed_ids_hash = contract["data_roles"]["holdout"]["row_ids_sha256"]
    if sample_row_ids_sha256(holdout_ids) != sealed_ids_hash:
        raise RuntimeError("loaded holdout row IDs do not match the sealed contract")

    prediction_started = time.perf_counter()
    c1, c3, c5, e7 = (portfolio[name] for name in ("C1", "C3", "C5", "E7"))
    raw_effect = 0.7 * c1.predict_effect(holdout) + 0.3 * c5.predict_effect(holdout)
    candidate_effect = float(e7["affine_intercept"]) + float(e7["affine_slope"]) * raw_effect
    baseline_effect = c3.predict_effect(holdout)
    candidate_policy = make_policy(candidate_effect)
    baseline_policy = make_policy(baseline_effect)
    inference_ms_per_1000 = (time.perf_counter() - prediction_started) * 1_000_000 / len(holdout)

    mu0, mu1 = c1.predict_outcomes(holdout)
    outcome = holdout[config.data.outcome].to_numpy(float)
    treatment = holdout[config.data.treatment].to_numpy(int)
    candidate_scores = incremental_policy_scores(
        outcome, treatment, candidate_policy, mu0, mu1, c1.propensity_
    )
    baseline_scores = incremental_policy_scores(
        outcome, treatment, baseline_policy, mu0, mu1, c1.propensity_
    )
    gamma = doubly_robust_score(outcome, treatment, mu0, mu1, c1.propensity_)
    candidate_calibration = calibration_rmse_linearization(
        candidate_effect, gamma, CALIBRATION_BINS
    )
    baseline_calibration = calibration_rmse_linearization(
        baseline_effect, gamma, CALIBRATION_BINS
    )
    calibration_difference = candidate_calibration.error - baseline_calibration.error
    clusters = profile_ids(holdout, config.data).to_numpy()

    draws = paired_cluster_bootstrap(
        candidate_scores,
        baseline_scores,
        calibration_difference,
        candidate_calibration.influence - baseline_calibration.influence,
        clusters,
        BOOTSTRAP_REPLICATES,
        config.seed + 903,
    )
    value_difference = float(np.mean(candidate_scores - baseline_scores))
    value_one_sided = interval(
        draws["candidate_minus_baseline"], ONE_SIDED_ALPHA, two_sided=False
    )
    calibration_one_sided = interval(
        draws["candidate_minus_baseline_calibration_rmse"],
        ONE_SIDED_ALPHA,
        two_sided=False,
    )
    candidate_two_sided = interval(draws["candidate_value"], 0.05, two_sided=True)
    baseline_two_sided = interval(draws["baseline_value"], 0.05, two_sided=True)
    difference_two_sided = interval(
        draws["candidate_minus_baseline"], 0.05, two_sided=True
    )
    calibration_two_sided = interval(
        draws["candidate_minus_baseline_calibration_rmse"], 0.05, two_sided=True
    )

    slice_rows = []
    slice_labels = pd.qcut(holdout[config.data.feature_columns[0]], 5, duplicates="drop")
    for label in sorted(slice_labels.unique(), key=str):
        mask = (slice_labels == label).to_numpy()
        difference = candidate_scores[mask] - baseline_scores[mask]
        slice_rows.append(
            {
                "slice": str(label),
                "rows": int(mask.sum()),
                "candidate_value": float(candidate_scores[mask].mean()),
                "baseline_value": float(baseline_scores[mask].mean()),
                "difference": float(difference.mean()),
            }
        )
    slices = pd.DataFrame(slice_rows)

    gates = {
        "G0_integrity_and_disjointness": True,
        "G1_value_noninferiority": value_one_sided["lower"]
        > VALUE_NONINFERIORITY_MARGIN,
        "G2_calibration_superiority": calibration_one_sided["upper"]
        < CALIBRATION_SUPERIORITY_MARGIN,
        "G3_candidate_standalone_value": candidate_two_sided["lower"] > 0.0,
        "G4_slice_guardrail": bool((slices["difference"] >= -0.005).all()),
        "G5_operational": inference_ms_per_1000 <= 100.0
        and args.artifact.stat().st_size <= 100_000_000,
    }
    verdict = "RELEASE_ELIGIBLE" if all(gates.values()) else "RELEASE_BLOCKED"
    metrics = {
        "schema": SCHEMA,
        "candidate": PRIMARY,
        "baseline": BASELINE,
        "holdout_rows": len(holdout),
        "unique_bootstrap_clusters": int(pd.Series(clusters).nunique()),
        "candidate_value": float(candidate_scores.mean()),
        "candidate_value_two_sided_95pct": candidate_two_sided,
        "baseline_value": float(baseline_scores.mean()),
        "baseline_value_two_sided_95pct": baseline_two_sided,
        "value_difference": value_difference,
        "value_difference_two_sided_95pct": difference_two_sided,
        "value_noninferiority_margin": VALUE_NONINFERIORITY_MARGIN,
        "value_noninferiority_one_sided_95pct_lower": value_one_sided["lower"],
        "candidate_calibration_rmse": candidate_calibration.error,
        "baseline_calibration_rmse": baseline_calibration.error,
        "calibration_rmse_difference": calibration_difference,
        "calibration_difference_two_sided_95pct": calibration_two_sided,
        "calibration_superiority_one_sided_95pct_upper": calibration_one_sided["upper"],
        "policy_disagreement_rate": float(np.mean(candidate_policy != baseline_policy)),
        "inference_ms_per_1000": inference_ms_per_1000,
        "artifact_size_bytes": args.artifact.stat().st_size,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "gates": gates,
        "verdict": verdict,
        "elapsed_seconds": time.perf_counter() - started,
    }

    atomic_json(args.output / "metrics.json", metrics)
    draws.insert(0, "replicate", np.arange(BOOTSTRAP_REPLICATES))
    draws.to_csv(args.output / "bootstrap_draws.csv", index=False)
    calibration_table = pd.concat(
        [
            candidate_calibration.table.assign(candidate=PRIMARY),
            baseline_calibration.table.assign(candidate=BASELINE),
        ],
        ignore_index=True,
    )
    calibration_table.to_csv(args.output / "calibration_table.csv", index=False)
    slices.to_csv(args.output / "slice_metrics.csv", index=False)
    pd.DataFrame(
        {
            "source_row_id": holdout_ids,
            "profile_id": clusters,
            "treatment": treatment,
            "outcome": outcome,
            "mu0": mu0,
            "mu1": mu1,
            "gamma": gamma,
            "candidate_effect": candidate_effect,
            "baseline_effect": baseline_effect,
            "candidate_policy": candidate_policy,
            "baseline_policy": baseline_policy,
            "candidate_aipw_score": candidate_scores,
            "baseline_aipw_score": baseline_scores,
            "candidate_calibration_influence": candidate_calibration.influence,
            "baseline_calibration_influence": baseline_calibration.influence,
        }
    ).to_csv(args.output / "evaluation_replay.csv.gz", index=False, compression="gzip")
    shutil.copy2(args.artifact, args.output / "frozen_candidate_portfolio.joblib")

    inventory = {
        path.relative_to(args.output).as_posix(): sha256_file(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file()
    }
    atomic_json(args.output / "evidence_inventory.json", inventory)
    seal = {
        "schema": SCHEMA,
        "selection_contract_sha256": sha256_file(args.contract),
        "evidence_inventory_sha256": sha256_file(args.output / "evidence_inventory.json"),
        "frozen_artifact_sha256": sha256_file(args.output / "frozen_candidate_portfolio.joblib"),
        "verdict": verdict,
        "platform": platform.platform(),
    }
    atomic_json(args.output / "run_seal.json", seal)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "run_seal_sha256": sha256_file(args.output / "run_seal.json"),
                "metrics": metrics,
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subcommands = result.add_subparsers(dest="command", required=True)
    for name in ("seal", "run"):
        command = subcommands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--package-dir", type=Path, required=True)
        command.add_argument("--artifact", type=Path, required=True)
        command.add_argument("--parent-contract", type=Path, required=True)
        command.add_argument("--contract", type=Path, required=True)
        if name == "run":
            command.add_argument("--expected-contract-sha256", required=True)
            command.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "seal":
        if args.contract.exists():
            raise ValueError("refusing to replace an existing contract")
        atomic_json(args.contract, contract_payload(args))
        print(json.dumps({"contract": str(args.contract), "sha256": sha256_file(args.contract)}))
    else:
        run(args)


if __name__ == "__main__":
    main()
