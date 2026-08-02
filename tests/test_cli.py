from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from orbit_causal.cli import _score_csv_snapshot, entrypoint, main
from orbit_causal.config import ProjectConfig


def test_packaged_demo_defaults_do_not_require_a_repository_config(capsys) -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory) / "run"
        output.mkdir()
        (output / "operational_metrics.json").write_text(
            json.dumps(
                {
                    "bundle_sha256": "1" * 64,
                    "bundle_manifest_sha256": "2" * 64,
                }
            ),
            encoding="utf-8",
        )
        (output / "run_seal.json").write_text("{}", encoding="utf-8")

        def fake_run(config, output_dir, **_kwargs):
            assert isinstance(config, ProjectConfig)
            assert Path(output_dir) == output
            return output

        with patch("orbit_causal.cli.run_experiment", side_effect=fake_run):
            assert main(["demo", "--output", str(output)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "completed"


def test_score_refuses_to_overwrite_any_input_path() -> None:
    same = "/tmp/orbit-score-input.csv"
    with pytest.raises(ValueError, match="protected input paths"):
        main(
            [
                "score",
                "--bundle",
                "/tmp/model.joblib",
                "--expected-sha256",
                "0" * 64,
                "--expected-manifest-sha256",
                "1" * 64,
                "--input",
                same,
                "--output",
                same,
            ]
        )


def test_score_rejects_duplicate_csv_headers_before_loading_bundle(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("f0,f0\n1,2\n", encoding="utf-8")
    with (
        patch("orbit_causal.cli.load_verified_bundle") as loader,
        pytest.raises(ValueError, match="duplicate columns"),
    ):
        main(
            [
                "score",
                "--bundle",
                str(tmp_path / "model.joblib"),
                "--expected-sha256",
                "0" * 64,
                "--expected-manifest-sha256",
                "1" * 64,
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "output.csv"),
            ]
        )
    loader.assert_not_called()


def test_score_snapshot_is_stable_and_bounded(tmp_path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("f0\n1\n", encoding="utf-8")
    with _score_csv_snapshot(source) as snapshot:
        assert snapshot != source
        source.write_text("f0\n999\n", encoding="utf-8")
        assert snapshot.read_text(encoding="utf-8") == "f0\n1\n"
    assert not snapshot.exists()

    source.write_bytes(b"12345")
    with (
        patch("orbit_causal.cli.MAX_SCORE_CSV_BYTES", 4),
        pytest.raises(ValueError, match="512 MiB"),
        _score_csv_snapshot(source),
    ):
        pass


def test_entrypoint_renders_operational_runtime_errors(capsys) -> None:
    with patch("orbit_causal.cli.main", side_effect=RuntimeError("output lock busy")):
        assert entrypoint() == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {"error": "output lock busy", "status": "error"}


def test_validate_rejects_a_non_criteo_config() -> None:
    config = Path(__file__).parents[1] / "configs" / "demo.json"
    with pytest.raises(ValueError, match="criteo_v2"):
        main(["validate", "--config", str(config), "--data", "/tmp/not-read.csv"])


def test_score_rejects_an_unbounded_batch_before_loading_the_bundle(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("f0\n1\n2\n", encoding="utf-8")
    with (
        patch("orbit_causal.cli.load_verified_bundle") as loader,
        pytest.raises(ValueError, match="max-records=1"),
    ):
        main(
            [
                "score",
                "--bundle",
                str(tmp_path / "model.joblib"),
                "--expected-sha256",
                "0" * 64,
                "--expected-manifest-sha256",
                "1" * 64,
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "output.csv"),
                "--max-records",
                "1",
            ]
        )
    loader.assert_not_called()
