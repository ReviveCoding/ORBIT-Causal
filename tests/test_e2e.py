from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from orbit_causal.api import DecisionService
from orbit_causal.artifacts import ArtifactVerificationError, atomic_json, sha256_file
from orbit_causal.config import DataConfig, EvaluationConfig, ModelConfig, ProjectConfig
from orbit_causal.experiment import (
    _output_lock,
    _refresh_evidence_inventory,
    _timed_fit,
    evaluate_release_gates,
    regenerate_reports,
    run_experiment,
    verify_evidence_inventory,
)


class EndToEndTests(unittest.TestCase):
    def test_output_lock_rejects_concurrency_but_not_a_stale_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            lock = Path(directory) / ".run.orbit-lock"
            lock.write_text("stale owner metadata", encoding="utf-8")
            with (
                _output_lock(output),
                self.assertRaisesRegex(RuntimeError, "another run owns"),
                _output_lock(output),
            ):
                self.fail("concurrent lock acquisition unexpectedly succeeded")
            with _output_lock(output):
                self.assertTrue(lock.is_file())

    def test_output_lock_is_released_after_an_abrupt_process_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            command = [
                sys.executable,
                "-c",
                (
                    "import os,sys; from pathlib import Path; "
                    "from orbit_causal.experiment import _output_lock; "
                    "guard=_output_lock(Path(sys.argv[1])); guard.__enter__(); os._exit(0)"
                ),
                str(output),
            ]
            completed = subprocess.run(
                command,
                check=False,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with _output_lock(output):
                self.assertTrue((Path(directory) / ".run.orbit-lock").is_file())

    def test_failed_timed_fit_does_not_leave_tracemalloc_enabled(self) -> None:
        class FailingModel:
            def fit(self, frame):
                raise RuntimeError("fit failed")

        self.assertFalse(tracemalloc.is_tracing())
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            _timed_fit(FailingModel, pd.DataFrame({"x": [1]}))
        self.assertFalse(tracemalloc.is_tracing())

    def test_value_gate_requires_positive_paired_superiority(self) -> None:
        baseline = pd.Series({"aipw_incremental_value": 0.03, "causal_calibration_error": 0.10})
        regressed = pd.Series(
            {
                "aipw_incremental_value": 0.01,
                "aipw_ci_lower": 0.001,
                "paired_baseline_improvement_ci_lower": -0.001,
                "causal_calibration_error": 0.10,
                "abstention_rate": 0.10,
            }
        )
        gates = evaluate_release_gates(
            regressed,
            baseline,
            schema_gate=True,
            evaluator_gate=True,
            integrity_gate=True,
            slice_gate=True,
            operational_gate=True,
            max_abstention=0.35,
        )
        self.assertFalse(gates["G3_value"])

    def test_value_gate_accepts_positive_paired_superiority(self) -> None:
        baseline = pd.Series({"aipw_incremental_value": 0.03, "causal_calibration_error": 0.10})
        superior = pd.Series(
            {
                "aipw_incremental_value": 0.04,
                "aipw_ci_lower": 0.01,
                "paired_baseline_improvement_ci_lower": 0.001,
                "causal_calibration_error": 0.10,
                "abstention_rate": 0.10,
            }
        )
        gates = evaluate_release_gates(
            superior,
            baseline,
            schema_gate=True,
            evaluator_gate=True,
            integrity_gate=True,
            slice_gate=True,
            operational_gate=True,
            max_abstention=0.35,
        )
        self.assertTrue(all(gates.values()))

    def test_failed_staged_run_preserves_previous_output(self) -> None:
        config = ProjectConfig()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            atomic_json(
                output / "run_manifest.json",
                {"schema_version": "1.0", "run_id": "previous", "artifact_checksums": {}},
            )
            (output / "sentinel.txt").write_text("previous", encoding="utf-8")
            _, previous_seal = _refresh_evidence_inventory(output)
            with (
                patch(
                    "orbit_causal.experiment._run_experiment_inplace",
                    side_effect=RuntimeError("injected failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected failure"),
            ):
                run_experiment(
                    config,
                    output,
                    replace_existing_run=True,
                    expected_existing_run_seal_sha256=previous_seal,
                )
            self.assertEqual((output / "sentinel.txt").read_text(encoding="utf-8"), "previous")
            self.assertEqual(list(Path(directory).glob(".run.staging-*")), [])

    def test_unrelated_nonempty_output_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "valuable"
            output.mkdir()
            (output / "user-data.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to replace"):
                run_experiment(ProjectConfig(), output)
            self.assertEqual((output / "user-data.txt").read_text(encoding="utf-8"), "keep")

    def test_synthetic_run_rejects_silently_ignored_external_data_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "do not accept"):
                run_experiment(
                    ProjectConfig(),
                    Path(directory) / "run",
                    data_path=Path(directory) / "unused.csv",
                )
            self.assertFalse((Path(directory) / "run").exists())

    def test_criteo_run_rejects_schema_compatible_but_unofficial_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lookalike.csv"
            frame = pd.DataFrame(
                {f"f{index}": [float(index), float(index + 1)] for index in range(12)}
            ).assign(
                treatment=[0, 1],
                exposure=[0, 1],
                visit=[0, 1],
                conversion=[0, 0],
            )
            frame.to_csv(source, index=False)
            config = ProjectConfig(
                data=DataConfig(
                    kind="criteo_v2",
                    feature_columns=tuple(f"f{index}" for index in range(12)),
                    categorical_columns=(),
                )
            )
            output = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "pinned official"):
                run_experiment(config, output, data_path=source)
            self.assertFalse(output.exists())

    def test_forged_minimal_manifest_does_not_authorize_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "valuable"
            output.mkdir()
            (output / "user-data.txt").write_text("keep", encoding="utf-8")
            atomic_json(
                output / "run_manifest.json",
                {"schema_version": "1.0", "run_id": "forged", "artifact_checksums": {}},
            )
            with self.assertRaisesRegex(ValueError, "verified ORBIT run"):
                run_experiment(
                    ProjectConfig(),
                    output,
                    replace_existing_run=True,
                    expected_existing_run_seal_sha256="0" * 64,
                )
            self.assertEqual((output / "user-data.txt").read_text(encoding="utf-8"), "keep")

    def test_evidence_inventory_rejects_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            outside = Path(directory) / "outside.txt"
            outside.write_text("not run evidence", encoding="utf-8")
            (root / "linked.txt").symlink_to(outside)
            atomic_json(root / "run_manifest.json", {"schema_version": "1.0", "run_id": "x"})
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                _refresh_evidence_inventory(root)

    @pytest.mark.integration
    def test_small_pipeline_creates_product_outputs(self) -> None:
        config = ProjectConfig(
            run_name="unittest",
            seed=123,
            data=replace(ProjectConfig().data, n_rows=1000),
            model=ModelConfig(
                max_iter=5,
                max_depth=2,
                min_samples_leaf=10,
                dr_folds=2,
                ensemble_size=2,
            ),
            policy=replace(
                ProjectConfig().policy,
                primary_capacity=0.30,
                scenario_capacities=(0.10, 0.30),
            ),
            evaluation=EvaluationConfig(
                bootstrap_replicates=2,
                calibration_bins=3,
                alpha=0.10,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = run_experiment(config, directory)
            required = [
                "metrics.json",
                "candidate_comparison.csv",
                "paired_comparison.csv",
                "decision_scenarios.csv",
                "calibration_table.csv",
                "slice_metrics.csv",
                "predictions.csv",
                "operational_metrics.json",
                "run_manifest.json",
                "qualification_manifest.json",
                "project_creation_handoff.json",
                "failure_cases.csv",
                "reason_code_report.csv",
                "report.md",
                "report.html",
                "artifacts/production_bundle.joblib",
                "artifacts/candidate_portfolio.joblib",
                "figures/candidate_comparison.png",
            ]
            for relative in required:
                self.assertTrue((output / relative).is_file(), relative)
            comparison = pd.read_csv(output / "candidate_comparison.csv")
            self.assertEqual(set(comparison["candidate_id"]), {"C0", "C1", "C3", "A1", "C5", "C6"})
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertIn("source_row_id", predictions.columns)
            self.assertTrue(predictions["source_row_id"].is_unique)
            with (output / "qualification_manifest.json").open("r", encoding="utf-8") as handle:
                qualification = json.load(handle)
            self.assertIn(
                qualification["release_verdict"],
                {"FIXTURE_RELEASE_PASS", "FIXTURE_RELEASE_BLOCKED"},
            )
            self.assertFalse(qualification["gates"]["G2_evaluator"])
            self.assertEqual(qualification["production_candidate"], "C1")
            self.assertIsNone(qualification["production_champion"])
            self.assertIn(
                "90% percentile bootstrap interval",
                (output / "report.md").read_text(encoding="utf-8"),
            )
            with (output / "metrics.json").open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertEqual(metrics["primary_capacity"], 0.30)
            with (output / "operational_metrics.json").open("r", encoding="utf-8") as handle:
                operational = json.load(handle)
            bundle_path = output / "artifacts" / "production_bundle.joblib"
            if qualification["release_verdict"].endswith("_PASS"):
                service = DecisionService(
                    bundle_path,
                    expected_sha256=operational["bundle_sha256"],
                    expected_manifest_sha256=operational["bundle_manifest_sha256"],
                )
                record = {column: 0 for column in config.data.feature_columns}
                scored = service.score_records([record])
                self.assertEqual(len(scored), 1)
                self.assertIn(scored[0]["decision"], {-1, 0, 1})
                self.assertEqual(scored[0]["allocation_scope"], "request_batch")
            else:
                with self.assertRaisesRegex(ArtifactVerificationError, "BLOCKED"):
                    DecisionService(
                        bundle_path,
                        expected_sha256=operational["bundle_sha256"],
                        expected_manifest_sha256=operational["bundle_manifest_sha256"],
                    )
            with (output / "run_manifest.json").open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertIn("evidence_inventory_sha256", manifest)
            self.assertEqual(
                manifest["evaluation_contract"]["bootstrap_unit"],
                "pretreatment_profile_cluster",
            )
            seal_hash = sha256_file(output / "run_seal.json")
            verified = verify_evidence_inventory(output, expected_seal_sha256=seal_hash)
            self.assertEqual(verified["status"], "verified")
            original_report_bytes = (output / "report.md").read_bytes()
            with (
                patch(
                    "orbit_causal.experiment.build_figures",
                    side_effect=RuntimeError("injected report failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected report failure"),
            ):
                regenerate_reports(output, expected_seal_sha256=seal_hash)
            self.assertEqual((output / "report.md").read_bytes(), original_report_bytes)
            self.assertEqual(sha256_file(output / "run_seal.json"), seal_hash)
            report_path = output / "report.md"
            original_report = report_path.read_bytes()
            report_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_evidence_inventory(output, expected_seal_sha256=seal_hash)
            report_path.write_bytes(original_report)
            with self.assertRaisesRegex(ValueError, "expected_seal_sha256"):
                verify_evidence_inventory(output)
            manifest_path = output / "run_manifest.json"
            with manifest_path.open("r", encoding="utf-8") as handle:
                mutable_manifest = json.load(handle)
            mutable_manifest["run_id"] = "tampered"
            atomic_json(manifest_path, mutable_manifest)
            with self.assertRaisesRegex(ValueError, "run manifest"):
                verify_evidence_inventory(output, expected_seal_sha256=seal_hash)


if __name__ == "__main__":
    unittest.main()
