from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


TOTAL_ROWS = 13_979_592
HOLDOUT_OFFSET = 750_000
HOLDOUT_ROWS = 2_000_000
REPLICATES = 4_000
SEED = 20260717 + 903
ARTIFACT_SHA256 = "047dbfd1f0ae48418b0de15e6f5da299a5d6fb575ecf81790afacf7a0a5df004"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sample_ids(total: int, rows: int, offset: int) -> np.ndarray:
    target = rows + offset
    selected_ids = np.empty(0, dtype=np.int64)
    selected_priorities = np.empty(0, dtype=np.uint64)
    for start in range(0, total, 250_000):
        ids = np.arange(start, min(start + 250_000, total), dtype=np.int64)
        values = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        priorities = values ^ (values >> np.uint64(31))
        selected_ids = np.concatenate((selected_ids, ids))
        selected_priorities = np.concatenate((selected_priorities, priorities))
        if len(selected_ids) > target:
            keep = np.argpartition(selected_priorities, target - 1)[:target]
            selected_ids = selected_ids[keep]
            selected_priorities = selected_priorities[keep]
    order = np.argsort(selected_priorities, kind="stable")
    return np.sort(selected_ids[order][offset:target], kind="mergesort")


def row_hash(ids: np.ndarray) -> str:
    canonical = np.sort(np.asarray(ids, dtype="<i8"), kind="mergesort")
    digest = hashlib.sha256()
    digest.update(len(canonical).to_bytes(8, "little", signed=False))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def calibration(effect: np.ndarray, gamma: np.ndarray) -> tuple[float, np.ndarray]:
    codes = pd.qcut(effect, q=10, duplicates="drop").codes
    bin_count = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=bin_count).astype(float)
    gaps = np.bincount(codes, weights=effect - gamma, minlength=bin_count) / counts
    squared = float(np.dot(counts / len(effect), gaps**2))
    error = float(np.sqrt(squared))
    residual = effect - gamma
    influence = (
        gaps[codes] ** 2 - squared + 2.0 * gaps[codes] * (residual - gaps[codes])
    ) / (2.0 * error)
    influence -= influence.mean()
    return error, influence


def incremental_scores(
    outcome: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float,
) -> np.ndarray:
    propensity = float(np.clip(propensity, 0.02, 0.98))
    observed_mu = np.where(treatment == 1, mu1, mu0)
    observed_probability = np.where(treatment == 1, propensity, 1.0 - propensity)

    def value(target: np.ndarray) -> np.ndarray:
        target_mu = np.where(target == 1, mu1, mu0)
        return target_mu + (treatment == target) * (outcome - observed_mu) / observed_probability

    return value(policy) - value(np.zeros_like(policy))


def regenerate_draws(
    candidate: np.ndarray,
    baseline: np.ndarray,
    calibration_difference: float,
    calibration_if_difference: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    codes, uniques = pd.factorize(clusters, sort=True)
    cluster_count = len(uniques)
    counts = np.bincount(codes, minlength=cluster_count).astype(float)
    candidate_sums = np.bincount(codes, weights=candidate, minlength=cluster_count)
    baseline_sums = np.bincount(codes, weights=baseline, minlength=cluster_count)
    calibration_sums = np.bincount(
        codes, weights=calibration_if_difference, minlength=cluster_count
    )
    rng = np.random.default_rng(SEED)
    result = np.empty((REPLICATES, 4), dtype=float)
    for replicate in range(REPLICATES):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        denominator = counts[sampled].sum()
        candidate_draw = candidate_sums[sampled].sum() / denominator
        baseline_draw = baseline_sums[sampled].sum() / denominator
        calibration_draw = calibration_difference + calibration_sums[sampled].sum() / denominator
        result[replicate] = (
            candidate_draw,
            baseline_draw,
            candidate_draw - baseline_draw,
            calibration_draw,
        )
    return result


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"mismatch: actual={actual} expected={expected}")


def run(output: Path, report: Path) -> None:
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    contract = json.loads((output / "selection_contract.json").read_text(encoding="utf-8"))
    inventory = json.loads((output / "evidence_inventory.json").read_text(encoding="utf-8"))
    seal = json.loads((output / "run_seal.json").read_text(encoding="utf-8"))
    if sha256_file(output / "selection_contract.json") != seal["selection_contract_sha256"]:
        raise AssertionError("selection contract seal mismatch")
    if sha256_file(output / "evidence_inventory.json") != seal["evidence_inventory_sha256"]:
        raise AssertionError("evidence inventory seal mismatch")
    for relative, expected in inventory.items():
        if sha256_file(output / relative) != expected:
            raise AssertionError(f"inventory mismatch: {relative}")
    artifact = output / "frozen_candidate_portfolio.joblib"
    if sha256_file(artifact) != ARTIFACT_SHA256 or sha256_file(artifact) != seal["frozen_artifact_sha256"]:
        raise AssertionError("frozen artifact mismatch")

    replay = pd.read_csv(output / "evaluation_replay.csv.gz")
    if len(replay) != HOLDOUT_ROWS or replay["source_row_id"].nunique() != HOLDOUT_ROWS:
        raise AssertionError("replay row count or uniqueness mismatch")
    expected_ids = sample_ids(TOTAL_ROWS, HOLDOUT_ROWS, HOLDOUT_OFFSET)
    actual_ids = np.sort(replay["source_row_id"].to_numpy(np.int64))
    if not np.array_equal(expected_ids, actual_ids):
        raise AssertionError("replay row IDs do not equal the sealed priority window")
    if row_hash(actual_ids) != contract["data_roles"]["holdout"]["row_ids_sha256"]:
        raise AssertionError("replay row ID hash mismatch")
    prior = sample_ids(TOTAL_ROWS, HOLDOUT_OFFSET, 0)
    overlap = int(np.intersect1d(prior, actual_ids, assume_unique=True).size)
    if overlap:
        raise AssertionError("replay overlaps prior windows")

    portfolio = joblib.load(artifact)
    propensity = portfolio["C1"].propensity_
    outcome = replay["outcome"].to_numpy(float)
    treatment = replay["treatment"].to_numpy(int)
    mu0 = replay["mu0"].to_numpy(float)
    mu1 = replay["mu1"].to_numpy(float)
    gamma = mu1 - mu0 + treatment * (outcome - mu1) / propensity - (
        1 - treatment
    ) * (outcome - mu0) / (1 - propensity)
    if not np.allclose(gamma, replay["gamma"].to_numpy(float), rtol=0.0, atol=2e-15):
        raise AssertionError("row-level doubly robust effect score mismatch")
    candidate_scores = incremental_scores(
        outcome,
        treatment,
        replay["candidate_policy"].to_numpy(int),
        mu0,
        mu1,
        propensity,
    )
    baseline_scores = incremental_scores(
        outcome,
        treatment,
        replay["baseline_policy"].to_numpy(int),
        mu0,
        mu1,
        propensity,
    )
    if not np.allclose(
        candidate_scores, replay["candidate_aipw_score"].to_numpy(float), rtol=0.0, atol=2e-15
    ) or not np.allclose(
        baseline_scores, replay["baseline_aipw_score"].to_numpy(float), rtol=0.0, atol=2e-15
    ):
        raise AssertionError("row-level AIPW replay mismatch")

    candidate_error, candidate_if = calibration(
        replay["candidate_effect"].to_numpy(float), gamma
    )
    baseline_error, baseline_if = calibration(
        replay["baseline_effect"].to_numpy(float), gamma
    )
    assert_close(candidate_scores.mean(), metrics["candidate_value"])
    assert_close(baseline_scores.mean(), metrics["baseline_value"])
    assert_close((candidate_scores - baseline_scores).mean(), metrics["value_difference"])
    assert_close(candidate_error, metrics["candidate_calibration_rmse"])
    assert_close(baseline_error, metrics["baseline_calibration_rmse"])
    if not np.allclose(
        candidate_if,
        replay["candidate_calibration_influence"].to_numpy(float),
        rtol=0.0,
        atol=1e-11,
    ) or not np.allclose(
        baseline_if,
        replay["baseline_calibration_influence"].to_numpy(float),
        rtol=0.0,
        atol=1e-11,
    ):
        raise AssertionError("calibration influence replay mismatch")

    regenerated = regenerate_draws(
        candidate_scores,
        baseline_scores,
        candidate_error - baseline_error,
        candidate_if - baseline_if,
        replay["profile_id"].to_numpy(),
    )
    stored = pd.read_csv(output / "bootstrap_draws.csv").iloc[:, 1:].to_numpy(float)
    max_draw_error = float(np.max(np.abs(regenerated - stored)))
    if max_draw_error > 5e-13:
        raise AssertionError(f"bootstrap draw mismatch: {max_draw_error}")

    value_lower = float(np.quantile(regenerated[:, 2], 0.05))
    calibration_upper = float(np.quantile(regenerated[:, 3], 0.95))
    assert_close(value_lower, metrics["value_noninferiority_one_sided_95pct_lower"])
    assert_close(
        calibration_upper, metrics["calibration_superiority_one_sided_95pct_upper"]
    )
    audit = {
        "schema": "orbit-causal-v039-independent-replay-audit-v1",
        "status": "PASS",
        "sealed_files_verified": len(inventory),
        "holdout_rows": len(replay),
        "unique_row_ids": int(replay["source_row_id"].nunique()),
        "prior_holdout_overlap": overlap,
        "row_ids_sha256": row_hash(actual_ids),
        "candidate_value_recomputed": float(candidate_scores.mean()),
        "baseline_value_recomputed": float(baseline_scores.mean()),
        "value_difference_recomputed": float((candidate_scores - baseline_scores).mean()),
        "candidate_calibration_rmse_recomputed": candidate_error,
        "baseline_calibration_rmse_recomputed": baseline_error,
        "value_noninferiority_lower_recomputed": value_lower,
        "calibration_superiority_upper_recomputed": calibration_upper,
        "bootstrap_draws_regenerated": REPLICATES,
        "bootstrap_max_absolute_error": max_draw_error,
        "run_seal_sha256": sha256_file(output / "run_seal.json"),
    }
    temporary = report.with_suffix(report.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, report)
    print(json.dumps(audit, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.report)


if __name__ == "__main__":
    main()
