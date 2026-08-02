from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import __version__


class ArtifactVerificationError(ValueError):
    """Raised before deserialization when an artifact cannot be authenticated."""


ARTIFACT_MANIFEST_SCHEMA = "1.3"
MAX_ARTIFACT_MANIFEST_BYTES = 64 * 1024
MAX_INTERNAL_JSON_BYTES = 16 * 1024 * 1024


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"nonstandard JSON numeric constant is not allowed: {value}")


def read_strict_json(path: str | Path, *, max_bytes: int = MAX_INTERNAL_JSON_BYTES) -> Any:
    """Read bounded RFC-compatible JSON without duplicate object keys."""

    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"JSON path must be a regular file: {path}")
    try:
        with path.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
    except OSError as error:
        raise ValueError(f"cannot read JSON file: {path}") from error
    if len(payload) > max_bytes:
        raise ValueError(f"JSON file exceeds the {max_bytes}-byte safety limit: {path}")
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed JSON file: {path}") from error


def _json_safe(value: Any) -> Any:
    """Convert numpy and non-finite values to strict JSON representations."""

    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort parent-directory sync after an atomic replacement."""

    if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        return


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                _json_safe(value),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_bundle(path: str | Path, bundle: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        joblib.dump(bundle, temporary, compress=3)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(path)


def atomic_text(path: str | Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def runtime_contract() -> dict[str, str]:
    return {
        "orbit_causal": __version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def save_decision_bundle_manifest(
    bundle_path: str | Path,
    checksum: str,
    *,
    run_id: str,
    dataset_fingerprint: str,
    candidate_id: str,
    release_status: str = "PENDING",
) -> Path:
    if release_status not in {"PENDING", "PASS", "BLOCKED"}:
        raise ValueError("release_status must be PENDING, PASS, or BLOCKED")
    checksum = _validate_sha256(checksum, "checksum")
    bundle_path = Path(bundle_path)
    manifest_path = bundle_path.with_suffix(".manifest.json")
    atomic_json(
        manifest_path,
        {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA,
            "artifact_name": bundle_path.name,
            "artifact_sha256": checksum,
            "artifact_schema_version": "1.0",
            "candidate_id": candidate_id,
            "run_id": run_id,
            "dataset_fingerprint": dataset_fingerprint,
            "release_status": release_status,
            "runtime_contract": runtime_contract(),
        },
    )
    return manifest_path


def _read_artifact_manifest(
    bundle_path: Path,
    manifest_path: str | Path | None,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    path = (
        Path(manifest_path)
        if manifest_path is not None
        else bundle_path.with_suffix(".manifest.json")
    )
    if not path.is_file():
        raise ArtifactVerificationError(f"artifact manifest must be a regular file: {path}")
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_ARTIFACT_MANIFEST_BYTES + 1)
        if len(payload) > MAX_ARTIFACT_MANIFEST_BYTES:
            raise ArtifactVerificationError("artifact manifest exceeds the 64 KiB safety limit")
        actual_checksum = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(actual_checksum, expected_manifest_sha256):
            raise ArtifactVerificationError("artifact manifest SHA-256 verification failed")
        manifest = json.loads(
            payload,
            object_pairs_hook=_unique_manifest_object,
            parse_constant=_reject_manifest_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactVerificationError(f"cannot read artifact manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise ArtifactVerificationError("artifact manifest must be a JSON object")
    required = {
        "schema_version",
        "artifact_name",
        "artifact_sha256",
        "artifact_schema_version",
        "candidate_id",
        "run_id",
        "dataset_fingerprint",
        "release_status",
        "runtime_contract",
    }
    if missing := sorted(required - set(manifest)):
        raise ArtifactVerificationError(f"artifact manifest is missing fields: {missing}")
    if unknown := sorted(set(manifest) - required):
        raise ArtifactVerificationError(f"artifact manifest has unknown fields: {unknown}")
    if manifest["schema_version"] != ARTIFACT_MANIFEST_SCHEMA:
        raise ArtifactVerificationError(
            f"unsupported artifact manifest schema: {manifest['schema_version']!r}"
        )
    if manifest["artifact_schema_version"] != "1.0":
        raise ArtifactVerificationError(
            f"unsupported artifact schema: {manifest['artifact_schema_version']!r}"
        )
    if manifest["artifact_name"] != bundle_path.name:
        raise ArtifactVerificationError("artifact manifest name does not match bundle path")
    if not isinstance(manifest["runtime_contract"], dict):
        raise ArtifactVerificationError("artifact runtime contract must be an object")
    if manifest["release_status"] not in {"PENDING", "PASS", "BLOCKED"}:
        raise ArtifactVerificationError("artifact manifest contains an invalid release status")
    artifact_checksum = manifest["artifact_sha256"]
    if (
        not isinstance(artifact_checksum, str)
        or len(artifact_checksum) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in artifact_checksum)
    ):
        raise ArtifactVerificationError("artifact manifest contains an invalid artifact SHA-256")
    return manifest


def _unique_manifest_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactVerificationError(f"duplicate artifact manifest key: {key!r}")
        result[key] = value
    return result


def _reject_manifest_constant(value: str) -> Any:
    raise ArtifactVerificationError(
        f"nonstandard JSON numeric constant in artifact manifest: {value}"
    )


def _validate_sha256(value: str | None, name: str) -> str:
    if value is None:
        raise ArtifactVerificationError(
            f"{name} is required and must come from an independent trusted channel"
        )
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ArtifactVerificationError(f"{name} must contain exactly 64 hexadecimal characters")
    return value.lower()


def load_verified_bundle(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    manifest_path: str | Path | None = None,
) -> Any:
    """Load only a release-authorized, independently authenticated bundle."""

    return _load_authenticated_bundle(
        path,
        expected_sha256=expected_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        manifest_path=manifest_path,
        require_release_pass=True,
    )


def _load_authenticated_bundle(
    path: str | Path,
    *,
    expected_sha256: str | None,
    expected_manifest_sha256: str | None,
    manifest_path: str | Path | None = None,
    require_release_pass: bool,
) -> Any:
    """Authenticate immutable artifact and manifest bytes before deserialization.

    The caller must supply both SHA-256 values through an independent trusted
    channel. Authenticating the sidecar is necessary because its runtime
    contract decides whether it is safe to deserialize the bundle.
    """

    path = Path(path)
    if not path.is_file():
        raise ArtifactVerificationError(f"artifact must be a regular file: {path}")
    expected_sha256 = _validate_sha256(expected_sha256, "expected_sha256")
    expected_manifest_sha256 = _validate_sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    manifest = _read_artifact_manifest(path, manifest_path, expected_manifest_sha256)
    if not isinstance(require_release_pass, bool):
        raise ArtifactVerificationError("require_release_pass must be a boolean")
    if require_release_pass and manifest["release_status"] != "PASS":
        raise ArtifactVerificationError(
            f"artifact is not release-authorized: {manifest['release_status']}"
        )
    if not hmac.compare_digest(str(manifest["artifact_sha256"]).lower(), expected_sha256):
        raise ArtifactVerificationError("trusted checksum does not match artifact manifest")
    expected_runtime = runtime_contract()
    if manifest["runtime_contract"] != expected_runtime:
        raise ArtifactVerificationError(
            "artifact runtime contract mismatch: "
            f"expected={manifest['runtime_contract']!r}, current={expected_runtime!r}"
        )

    temporary: Path | None = None
    digest = hashlib.sha256()
    try:
        with (
            path.open("rb") as source,
            tempfile.NamedTemporaryFile(
                prefix="orbit-verified-",
                suffix=".joblib",
                delete=False,
            ) as target,
        ):
            temporary = Path(target.name)
            while chunk := source.read(1 << 20):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            raise ArtifactVerificationError("artifact SHA-256 verification failed")

        try:
            bundle = joblib.load(temporary)
        except Exception as error:
            raise ArtifactVerificationError(
                "authenticated artifact could not be deserialized"
            ) from error
        from .bundle import DecisionBundle

        if not isinstance(bundle, DecisionBundle):
            raise ArtifactVerificationError("artifact is not a DecisionBundle")
        if bundle.artifact_schema_version != "1.0":
            raise ArtifactVerificationError(
                f"unsupported artifact schema: {bundle.artifact_schema_version!r}"
            )
        lineage = {
            "candidate_id": bundle.candidate_id,
            "run_id": bundle.run_id,
            "dataset_fingerprint": bundle.dataset_fingerprint,
            "artifact_schema_version": bundle.artifact_schema_version,
        }
        for key, value in lineage.items():
            if manifest[key] != value:
                raise ArtifactVerificationError(f"artifact lineage mismatch for {key}")
        return bundle
    except OSError as error:
        raise ArtifactVerificationError(f"cannot read artifact: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finalize_decision_bundle_manifest(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    release_status: str,
) -> str:
    """Atomically bind the final release-gate outcome into an artifact manifest."""

    if release_status not in {"PASS", "BLOCKED"}:
        raise ValueError("final release_status must be PASS or BLOCKED")
    expected_manifest_sha256 = _validate_sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    path = Path(manifest_path)
    if not path.is_file():
        raise ArtifactVerificationError(f"artifact manifest must be a regular file: {path}")
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_ARTIFACT_MANIFEST_BYTES + 1)
        if len(payload) > MAX_ARTIFACT_MANIFEST_BYTES:
            raise ArtifactVerificationError("artifact manifest exceeds the 64 KiB safety limit")
        if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected_manifest_sha256):
            raise ArtifactVerificationError("pending artifact manifest changed before finalization")
        manifest = json.loads(
            payload,
            object_pairs_hook=_unique_manifest_object,
            parse_constant=_reject_manifest_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactVerificationError(f"cannot finalize artifact manifest: {path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA:
        raise ArtifactVerificationError("cannot finalize malformed artifact manifest")
    if manifest.get("release_status") != "PENDING":
        raise ArtifactVerificationError("only a PENDING artifact manifest can be finalized")
    manifest["release_status"] = release_status
    atomic_json(path, manifest)
    return sha256_file(path)


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "runtime_contract": runtime_contract(),
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
