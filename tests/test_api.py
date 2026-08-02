from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from orbit_causal.api import DecisionService, create_fastapi_app
from orbit_causal.artifacts import save_bundle, save_decision_bundle_manifest, sha256_file
from orbit_causal.bundle import DecisionBundle


class _StaticModel:
    def predict_effect(self, frame):
        return np.full(len(frame), 0.2)

    def predict_uncertainty(self, frame):
        return np.full(len(frame), 0.01)


class ApiTests(unittest.TestCase):
    def _fixture(self, root: Path, capacity: float = 1.0) -> tuple[Path, str, str]:
        path = root / "artifacts" / "production_bundle.joblib"
        bundle = DecisionBundle(
            model=_StaticModel(),
            candidate_id="C6",
            feature_columns=("f0",),
            capacity=capacity,
            risk_multiplier=0.25,
            intervention_cost=0.0,
            run_id="api-test",
            dataset_fingerprint="fixture",
        )
        checksum = save_bundle(path, bundle)
        manifest_path = save_decision_bundle_manifest(
            path,
            checksum,
            run_id=bundle.run_id,
            dataset_fingerprint=bundle.dataset_fingerprint,
            candidate_id=bundle.candidate_id,
            release_status="PASS",
        )
        return path, checksum, sha256_file(manifest_path)

    def test_service_reports_verified_health_and_rejects_empty_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, checksum, manifest_checksum = self._fixture(Path(directory))
            service = DecisionService(
                path,
                expected_sha256=checksum,
                expected_manifest_sha256=manifest_checksum,
                max_records=2,
            )
            self.assertTrue(service.health()["artifact_verified"])
            self.assertTrue(service.health()["artifact_manifest_verified"])
            self.assertEqual(service.health()["artifact_schema_version"], "1.0")
            self.assertEqual(service.health()["allocation_scope"], "request_batch")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                service.score_records([])
            with self.assertRaisesRegex(ValueError, "max_records"):
                service.score_records([{"f0": 0.1}] * 3)
            duplicate_columns = pd.DataFrame([[0.1, 0.2]], columns=["f0", "f0"])
            with self.assertRaisesRegex(ValueError, "duplicate columns"):
                service.bundle.score(duplicate_columns)
            with self.assertRaisesRegex(ValueError, "max_records"):
                DecisionService(
                    path,
                    expected_sha256=checksum,
                    expected_manifest_sha256=manifest_checksum,
                    max_records=100_001,
                )

    def test_fastapi_health_and_score_contract(self) -> None:
        try:
            import httpx
        except ImportError:
            self.skipTest("optional API dependencies are not installed")
        with tempfile.TemporaryDirectory() as directory:
            path, checksum, manifest_checksum = self._fixture(Path(directory))
            app = create_fastapi_app(
                path,
                expected_sha256=checksum,
                expected_manifest_sha256=manifest_checksum,
            )

            async def exercise():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    return (
                        await client.get("/health"),
                        await client.post("/allocate", json=[{"f0": 0.1}]),
                    )

            health, response = asyncio.run(exercise())
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["artifact_verified"])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()[0]["decision"], 1)
            self.assertEqual(response.json()[0]["allocation_scope"], "request_batch")

    def test_row_key_makes_tied_allocation_invariant_to_request_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, checksum, manifest_checksum = self._fixture(Path(directory), capacity=0.5)
            service = DecisionService(
                path,
                expected_sha256=checksum,
                expected_manifest_sha256=manifest_checksum,
            )
            first = service.allocate_records(
                [{"row_key": "b", "f0": 0.1}, {"row_key": "a", "f0": 0.2}]
            )
            second = service.allocate_records(
                [{"row_key": "a", "f0": 0.2}, {"row_key": "b", "f0": 0.1}]
            )
            first_decisions = {row["row_key"]: row["decision"] for row in first}
            second_decisions = {row["row_key"]: row["decision"] for row in second}
            self.assertEqual(first_decisions, second_decisions)
            self.assertEqual(first_decisions, {"a": 1, "b": 0})
            with self.assertRaisesRegex(ValueError, "nonempty"):
                service.allocate_records([{"row_key": "", "f0": 0.1}])
            with self.assertRaisesRegex(ValueError, "strings"):
                service.allocate_records([{"row_key": 1, "f0": 0.1}])
            with self.assertRaisesRegex(ValueError, "control characters"):
                service.allocate_records([{"row_key": "bad\nkey", "f0": 0.1}])


if __name__ == "__main__":
    unittest.main()
