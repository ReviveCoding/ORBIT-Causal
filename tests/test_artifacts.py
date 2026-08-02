from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from orbit_causal import __version__
from orbit_causal.artifacts import (
    ArtifactVerificationError,
    atomic_json,
    finalize_decision_bundle_manifest,
    load_verified_bundle,
    read_strict_json,
    runtime_contract,
    save_bundle,
    save_decision_bundle_manifest,
    sha256_file,
)
from orbit_causal.bundle import DecisionBundle


class _Model:
    pass


def _bundle(schema: str = "1.0") -> DecisionBundle:
    return DecisionBundle(
        model=_Model(),
        candidate_id="C6",
        feature_columns=("f0",),
        capacity=0.2,
        risk_multiplier=0.25,
        intervention_cost=0.0,
        run_id="test",
        dataset_fingerprint="abc",
        artifact_schema_version=schema,
    )


class ArtifactTests(unittest.TestCase):
    @staticmethod
    def _save(path: Path, bundle: DecisionBundle) -> tuple[str, str]:
        checksum = save_bundle(path, bundle)
        manifest_path = save_decision_bundle_manifest(
            path,
            checksum,
            run_id=bundle.run_id,
            dataset_fingerprint=bundle.dataset_fingerprint,
            candidate_id=bundle.candidate_id,
            release_status="PASS",
        )
        return checksum, sha256_file(manifest_path)

    def test_verified_loader_checks_bytes_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            checksum, manifest_checksum = self._save(path, _bundle())
            with patch("orbit_causal.artifacts.joblib.load") as loader:
                with self.assertRaisesRegex(ArtifactVerificationError, "trusted checksum"):
                    load_verified_bundle(
                        path,
                        expected_sha256="0" * 64,
                        expected_manifest_sha256=manifest_checksum,
                    )
                loader.assert_not_called()
            loaded = load_verified_bundle(
                path,
                expected_sha256=checksum,
                expected_manifest_sha256=manifest_checksum,
            )
            self.assertEqual(loaded.artifact_schema_version, "1.0")

    def test_external_checksum_and_runtime_contract_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            checksum, manifest_checksum = self._save(path, _bundle())
            with self.assertRaisesRegex(ArtifactVerificationError, "expected_sha256"):
                load_verified_bundle(path)
            with self.assertRaisesRegex(ArtifactVerificationError, "expected_manifest_sha256"):
                load_verified_bundle(path, expected_sha256=checksum)
            manifest_path = path.with_suffix(".manifest.json")
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["runtime_contract"] = {**runtime_contract(), "scikit_learn": "0.0"}
            atomic_json(manifest_path, manifest)
            with patch("orbit_causal.artifacts.joblib.load") as loader:
                with self.assertRaisesRegex(ArtifactVerificationError, "manifest SHA-256"):
                    load_verified_bundle(
                        path,
                        expected_sha256=checksum,
                        expected_manifest_sha256=manifest_checksum,
                    )
                with self.assertRaisesRegex(ArtifactVerificationError, "runtime contract"):
                    load_verified_bundle(
                        path,
                        expected_sha256=checksum,
                        expected_manifest_sha256=sha256_file(manifest_path),
                    )
                loader.assert_not_called()

    def test_runtime_contract_binds_the_orbit_release(self) -> None:
        self.assertEqual(runtime_contract()["orbit_causal"], __version__)

    def test_manifest_and_schema_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifacts" / "production_bundle.joblib"
            checksum, manifest_checksum = self._save(path, _bundle("9.9"))
            with self.assertRaisesRegex(ArtifactVerificationError, "unsupported artifact schema"):
                load_verified_bundle(
                    path,
                    expected_sha256=checksum,
                    expected_manifest_sha256=manifest_checksum,
                )

    def test_every_manifest_contract_field_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            checksum, _ = self._save(path, _bundle())
            manifest_path = path.with_suffix(".manifest.json")
            valid = read_strict_json(manifest_path)
            cases: tuple[tuple[str, object], ...] = (
                ("must be a JSON object", []),
                ("missing fields", {key: value for key, value in valid.items() if key != "run_id"}),
                ("unknown fields", {**valid, "unexpected": True}),
                ("unsupported artifact manifest schema", {**valid, "schema_version": "9.9"}),
                ("unsupported artifact schema", {**valid, "artifact_schema_version": "9.9"}),
                ("name does not match", {**valid, "artifact_name": "other.joblib"}),
                ("runtime contract must be an object", {**valid, "runtime_contract": []}),
                ("invalid release status", {**valid, "release_status": "UNKNOWN"}),
                ("invalid artifact SHA-256", {**valid, "artifact_sha256": "not-a-digest"}),
            )
            for message, malformed in cases:
                with self.subTest(message=message):
                    atomic_json(manifest_path, malformed)
                    with self.assertRaisesRegex(ArtifactVerificationError, message):
                        load_verified_bundle(
                            path,
                            expected_sha256=checksum,
                            expected_manifest_sha256=sha256_file(manifest_path),
                        )

    def test_authenticated_but_malformed_bundle_has_controlled_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            checksum, manifest_checksum = self._save(path, _bundle())
            with (
                patch(
                    "orbit_causal.artifacts.joblib.load",
                    side_effect=RuntimeError("decoder failure"),
                ),
                self.assertRaisesRegex(ArtifactVerificationError, "could not be deserialized"),
            ):
                load_verified_bundle(
                    path,
                    expected_sha256=checksum,
                    expected_manifest_sha256=manifest_checksum,
                )
            path.with_suffix(".manifest.json").unlink()
            with self.assertRaisesRegex(ArtifactVerificationError, "manifest"):
                load_verified_bundle(
                    path,
                    expected_sha256=checksum,
                    expected_manifest_sha256=manifest_checksum,
                )

    def test_pending_and_blocked_artifacts_cannot_be_served(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            bundle = _bundle()
            checksum = save_bundle(path, bundle)
            manifest_path = save_decision_bundle_manifest(
                path,
                checksum,
                run_id=bundle.run_id,
                dataset_fingerprint=bundle.dataset_fingerprint,
                candidate_id=bundle.candidate_id,
            )
            with self.assertRaisesRegex(ArtifactVerificationError, "PENDING"):
                load_verified_bundle(
                    path,
                    expected_sha256=checksum,
                    expected_manifest_sha256=sha256_file(manifest_path),
                )
            blocked_hash = finalize_decision_bundle_manifest(
                manifest_path,
                expected_manifest_sha256=sha256_file(manifest_path),
                release_status="BLOCKED",
            )
            with self.assertRaisesRegex(ArtifactVerificationError, "BLOCKED"):
                load_verified_bundle(
                    path,
                    expected_sha256=checksum,
                    expected_manifest_sha256=blocked_hash,
                )

    def test_atomic_json_writers_do_not_share_a_fixed_tempfile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            values = [{"writer": index, "payload": "x" * 1000} for index in range(12)]
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(lambda value: atomic_json(path, value), values))
            with path.open("r", encoding="utf-8") as handle:
                final = json.load(handle)
            self.assertIn(final, values)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_unsupported_directory_fsync_does_not_turn_a_committed_write_into_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            original_open = os.open

            def fail_directory_open(target, flags, *args):
                if flags & getattr(os, "O_DIRECTORY", 0):
                    raise OSError("unsupported")
                return original_open(target, flags, *args)

            with patch("orbit_causal.artifacts.os.open", side_effect=fail_directory_open):
                atomic_json(path, {"status": "committed"})
            self.assertEqual(read_strict_json(path), {"status": "committed"})

    def test_atomic_json_rejects_lossy_nonstring_object_keys(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(TypeError, "keys must be strings"),
        ):
            atomic_json(Path(directory) / "bad.json", {1: "integer key"})

    def test_strict_artifact_helpers_reject_lossy_or_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            with self.assertRaises(TypeError):
                atomic_json(path, {"unsupported": object()})
            path.write_bytes(b"payload")
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                sha256_file(path, chunk_size=0)
            with self.assertRaisesRegex(ArtifactVerificationError, "checksum"):
                save_decision_bundle_manifest(
                    path,
                    "not-a-sha256",
                    run_id="run",
                    dataset_fingerprint="data",
                    candidate_id="C6",
                )
            with self.assertRaisesRegex(ArtifactVerificationError, "64 hexadecimal"):
                load_verified_bundle(
                    path,
                    expected_sha256="a" * 62 + "  ",
                    expected_manifest_sha256="b" * 64,
                )

    def test_bounded_strict_json_rejects_duplicate_keys_and_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text('{"key": 1, "key": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                read_strict_json(path)
            path.write_text('{"key": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonstandard JSON"):
                read_strict_json(path)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safety limit"):
                read_strict_json(path, max_bytes=1)


if __name__ == "__main__":
    unittest.main()
