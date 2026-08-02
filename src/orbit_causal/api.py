from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .artifacts import load_verified_bundle
from .bundle import MAX_SCORE_RECORDS


class DecisionService:
    """Dependency-light product interface used by batch and optional HTTP layers."""

    def __init__(
        self,
        bundle_path: str | Path,
        *,
        expected_sha256: str,
        expected_manifest_sha256: str,
        manifest_path: str | Path | None = None,
        max_records: int = 10_000,
    ) -> None:
        if (
            not isinstance(max_records, int)
            or isinstance(max_records, bool)
            or not 1 <= max_records <= MAX_SCORE_RECORDS
        ):
            raise ValueError(f"max_records must be an integer in [1, {MAX_SCORE_RECORDS}]")
        self.bundle_path = Path(bundle_path)
        self.max_records = max_records
        self.bundle = load_verified_bundle(
            self.bundle_path,
            expected_sha256=expected_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
            manifest_path=manifest_path,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "run_id": self.bundle.run_id,
            "candidate_id": self.bundle.candidate_id,
            "feature_count": len(self.bundle.feature_columns),
            "artifact_verified": True,
            "artifact_manifest_verified": True,
            "artifact_schema_version": self.bundle.artifact_schema_version,
            "allocation_scope": "request_batch",
            "max_records": self.max_records,
        }

    def allocate_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            raise ValueError("records must be nonempty")
        if len(records) > self.max_records:
            raise ValueError(f"records exceed max_records={self.max_records}")
        if any(not isinstance(record, dict) for record in records):
            raise TypeError("every record must be an object")
        frame = pd.DataFrame.from_records(records)
        return self.bundle.score(frame).to_dict(orient="records")

    def score_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Backward-compatible alias; decisions are allocated within the request batch."""

        return self.allocate_records(records)


def create_fastapi_app(
    bundle_path: str | Path,
    *,
    expected_sha256: str,
    expected_manifest_sha256: str,
    manifest_path: str | Path | None = None,
    max_records: int = 10_000,
):
    """Create an HTTP app when the optional FastAPI dependency is installed."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as error:
        raise RuntimeError("install orbit-causal[api] to create the HTTP service") from error

    service = DecisionService(
        bundle_path,
        expected_sha256=expected_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        manifest_path=manifest_path,
        max_records=max_records,
    )
    app = FastAPI(title="ORBIT-Causal", version=__version__)

    @app.get("/health")
    def health():
        return service.health()

    def allocate(records: list[dict[str, Any]]):
        try:
            return service.allocate_records(records)
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    app.post("/allocate")(allocate)
    app.post("/score", deprecated=True)(allocate)

    return app
