from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_json, read_strict_json, sha256_file, stable_json_hash
from .config import ProjectConfig
from .data import (
    CriteoFileValidationReport,
    deterministic_sample_row_ids,
    sample_row_ids_sha256,
)

SELECTION_CONTRACT_SCHEMA = "1.0"
SAMPLING_ALGORITHM = "splitmix64_priority_window_v1"
PRIMARY_CANDIDATE = "C1"
SUPERIORITY_BASELINE = "C3"
SECONDARY_COMPARISONS = ("C5", "C6")


def implementation_source_fingerprint() -> str:
    package_dir = Path(__file__).resolve().parent
    hashes = {
        f"orbit_causal/{path.name}": sha256_file(path)
        for path in sorted(package_dir.glob("*.py"))
        if path.is_file()
    }
    if not hashes:
        raise RuntimeError("cannot fingerprint the installed ORBIT source")
    return stable_json_hash(hashes)


def release_protocol(config: ProjectConfig) -> dict[str, Any]:
    return {
        "primary_candidate": PRIMARY_CANDIDATE,
        "superiority_baseline": SUPERIORITY_BASELINE,
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "primary_release_rules": {
            "standalone_aipw_ci_lower_gt": 0.0,
            "paired_improvement_ci_lower_gt": 0.0,
            "calibration_ratio_max": 1.10,
            "slice_value_regression_max": config.evaluation.slice_noninferiority_margin,
            "minimum_slice_rows": config.evaluation.minimum_slice_rows,
            "max_abstention": config.policy.max_abstention,
            "max_inference_ms_per_1000": config.evaluation.max_inference_ms_per_1000,
            "max_bundle_size_bytes": config.evaluation.max_bundle_size_bytes,
        },
        "c6_role": "negative_ablation",
        "final_test_role": "locked_release_evaluation_only",
    }


def selection_contract_payload(
    config: ProjectConfig,
    source_validation: CriteoFileValidationReport,
    *,
    max_rows: int | None,
    sample_offset: int,
) -> dict[str, Any]:
    if config.data.kind != "criteo_v2":
        raise ValueError("selection contract requires data.kind='criteo_v2'")
    if not source_validation.official_file_match:
        raise ValueError("selection contract requires the pinned official Criteo file")
    source_row_ids = deterministic_sample_row_ids(
        source_validation.row_count,
        max_rows,
        sample_offset,
    )
    sample_count = len(source_row_ids)
    sampling = {
        "algorithm": SAMPLING_ALGORITHM if max_rows is not None else "complete_file_v1",
        "max_rows": max_rows,
        "sample_offset": sample_offset,
        "sample_row_count": sample_count,
        "sample_row_id_min": int(source_row_ids.min()),
        "sample_row_id_max": int(source_row_ids.max()),
        "sample_row_ids_sha256": sample_row_ids_sha256(source_row_ids),
    }
    protocol = release_protocol(config)
    return {
        "schema_version": SELECTION_CONTRACT_SCHEMA,
        "config_hash": config.fingerprint(),
        "source_fingerprint": implementation_source_fingerprint(),
        "source": {
            "row_count": source_validation.row_count,
            "size_bytes": source_validation.source_size_bytes,
            "sha256": source_validation.source_sha256,
        },
        "sampling": sampling,
        "release_protocol": protocol,
        "release_protocol_sha256": stable_json_hash(protocol),
    }


def write_selection_contract(
    path: str | Path,
    config: ProjectConfig,
    source_validation: CriteoFileValidationReport,
    *,
    max_rows: int | None,
    sample_offset: int,
) -> str:
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to replace an existing selection contract: {path}")
    atomic_json(
        path,
        selection_contract_payload(
            config,
            source_validation,
            max_rows=max_rows,
            sample_offset=sample_offset,
        ),
    )
    return sha256_file(path)


def validate_selection_contract(
    path: str | Path,
    *,
    expected_sha256: str,
    config: ProjectConfig,
    source_validation: dict[str, Any],
    source_row_ids: np.ndarray,
    max_rows: int | None,
    sample_offset: int,
) -> dict[str, Any]:
    path = Path(path)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)
    ):
        raise ValueError("expected selection-contract SHA-256 must be 64 hexadecimal characters")
    if sha256_file(path) != expected_sha256.lower():
        raise ValueError("trusted selection-contract SHA-256 does not match")
    contract = read_strict_json(path)
    if not isinstance(contract, dict):
        raise ValueError("selection contract must be a JSON object")
    expected_keys = {
        "schema_version",
        "config_hash",
        "source_fingerprint",
        "source",
        "sampling",
        "release_protocol",
        "release_protocol_sha256",
    }
    if set(contract) != expected_keys or contract.get("schema_version") != SELECTION_CONTRACT_SCHEMA:
        raise ValueError("selection contract contains missing, unknown, or unsupported fields")
    protocol = release_protocol(config)
    if contract.get("config_hash") != config.fingerprint():
        raise ValueError("selection contract config hash mismatch")
    if contract.get("source_fingerprint") != implementation_source_fingerprint():
        raise ValueError("selection contract implementation source fingerprint mismatch")
    if contract.get("release_protocol") != protocol:
        raise ValueError("selection contract release protocol mismatch")
    if contract.get("release_protocol_sha256") != stable_json_hash(protocol):
        raise ValueError("selection contract release protocol hash mismatch")
    expected_source = {
        "row_count": source_validation.get("row_count"),
        "size_bytes": source_validation.get("source_size_bytes"),
        "sha256": source_validation.get("source_sha256"),
    }
    if contract.get("source") != expected_source:
        raise ValueError("selection contract source identity mismatch")
    sampling = contract.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("selection contract sampling section is malformed")
    expected_sampling = {
        "algorithm": SAMPLING_ALGORITHM if max_rows is not None else "complete_file_v1",
        "max_rows": max_rows,
        "sample_offset": sample_offset,
        "sample_row_count": len(source_row_ids),
        "sample_row_id_min": int(np.min(source_row_ids)),
        "sample_row_id_max": int(np.max(source_row_ids)),
        "sample_row_ids_sha256": sample_row_ids_sha256(source_row_ids),
    }
    if sampling != expected_sampling:
        raise ValueError("selection contract sample-row admission mismatch")
    return contract
