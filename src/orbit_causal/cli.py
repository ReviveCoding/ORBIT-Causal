from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .artifacts import atomic_csv, load_verified_bundle, read_strict_json, sha256_file
from .bundle import MAX_SCORE_RECORDS
from .config import ProjectConfig, load_config
from .data import validate_criteo_file
from .experiment import regenerate_reports, run_experiment, verify_evidence_inventory
from .protocol import write_selection_contract

MAX_SCORE_CSV_BYTES = 512 * 1024 * 1024
MAX_SCORE_CSV_HEADER_BYTES = 1 * 1024 * 1024


@contextmanager
def _score_csv_snapshot(path: str | Path) -> Iterator[Path]:
    """Copy one bounded input stream so validation and parsing see identical bytes."""

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"score input must be a regular file: {source_path}")
    snapshot: Path | None = None
    size = 0
    try:
        try:
            with (
                source_path.open("rb") as source,
                tempfile.NamedTemporaryFile(
                    prefix="orbit-score-snapshot-",
                    suffix=".csv",
                    delete=False,
                ) as target,
            ):
                snapshot = Path(target.name)
                while chunk := source.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_SCORE_CSV_BYTES:
                        raise ValueError("score input exceeds the 512 MiB safety limit")
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        except OSError as error:
            raise ValueError("cannot create a stable score input snapshot") from error
        yield snapshot
    finally:
        if snapshot is not None:
            snapshot.unlink(missing_ok=True)


def _validate_score_csv_header(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"score input must be a regular file: {path}")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValueError(f"cannot inspect score input: {path}") from error
    if size > MAX_SCORE_CSV_BYTES:
        raise ValueError("score input exceeds the 512 MiB safety limit")
    try:
        with path.open("rb") as handle:
            header_bytes = handle.readline(MAX_SCORE_CSV_HEADER_BYTES + 1)
        if len(header_bytes) > MAX_SCORE_CSV_HEADER_BYTES:
            raise ValueError("score input CSV header exceeds the 1 MiB safety limit")
        header_text = header_bytes.decode("utf-8-sig")
        header = next(csv.reader([header_text]))
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise ValueError("score input must contain a valid UTF-8 CSV header") from error
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if not header or any(not name for name in header):
        raise ValueError("score input CSV header names must be nonempty")
    if duplicates:
        raise ValueError(f"score input CSV contains duplicate columns: {duplicates}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbit-causal",
        description="Orthogonal, risk-bounded intervention targeting",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the complete generated-fixture pipeline")
    demo.add_argument(
        "--config",
        help="optional demo config; packaged defaults are used when omitted",
    )
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument(
        "--replace-existing-run",
        action="store_true",
        help="explicitly replace a previously verified ORBIT run",
    )
    demo.add_argument(
        "--expected-existing-run-seal-sha256",
        help="trusted seal for the prior run when replacement is requested",
    )

    run = subparsers.add_parser("run", help="run a configured experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--data", help="external CRITEO-UPLIFTv2 CSV path")
    run.add_argument("--max-rows", type=int)
    run.add_argument("--sample-offset", type=int, default=0)
    run.add_argument("--selection-contract", required=True)
    run.add_argument("--expected-selection-contract-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument(
        "--replace-existing-run",
        action="store_true",
        help="explicitly replace a previously verified ORBIT run",
    )
    run.add_argument(
        "--expected-existing-run-seal-sha256",
        help="trusted seal for the prior run when replacement is requested",
    )

    validate = subparsers.add_parser("validate", help="validate an external Criteo file")
    validate.add_argument("--config", default="configs/criteo_v2.json")
    validate.add_argument("--data", required=True)
    validate.add_argument("--max-rows", type=int)
    validate.add_argument(
        "--require-official-file",
        action="store_true",
        help="require the pinned official v2.1 gzip row-count, size, and SHA-256",
    )

    prepare = subparsers.add_parser(
        "prepare-run",
        help="authenticate Criteo and pre-seal the sample and release protocol",
    )
    prepare.add_argument("--config", default="configs/criteo_v2.json")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--max-rows", type=int)
    prepare.add_argument("--sample-offset", type=int, default=0)
    prepare.add_argument("--output", required=True)

    score = subparsers.add_parser("score", help="score a CSV with a saved decision bundle")
    score.add_argument("--bundle", required=True)
    score.add_argument(
        "--expected-sha256",
        required=True,
        help="SHA-256 obtained through an independent trusted channel",
    )
    score.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="manifest SHA-256 obtained through an independent trusted channel",
    )
    score.add_argument("--manifest", help="artifact manifest; defaults beside the bundle")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--max-records", type=int, default=10_000)

    report = subparsers.add_parser("report", help="regenerate figures and reports")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--expected-seal-sha256", required=True)

    verify = subparsers.add_parser("verify-run", help="verify every sealed run evidence file")
    verify.add_argument("--run-dir", required=True)
    verify.add_argument("--expected-seal-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"demo", "run"}:
        config = (
            ProjectConfig()
            if args.command == "demo" and args.config is None
            else load_config(args.config)
        )
        output = run_experiment(
            config,
            args.output,
            data_path=getattr(args, "data", None),
            max_rows=getattr(args, "max_rows", None),
            sample_offset=getattr(args, "sample_offset", 0),
            selection_contract_path=getattr(args, "selection_contract", None),
            expected_selection_contract_sha256=getattr(
                args,
                "expected_selection_contract_sha256",
                None,
            ),
            replace_existing_run=args.replace_existing_run,
            expected_existing_run_seal_sha256=args.expected_existing_run_seal_sha256,
        )
        operational = read_strict_json(output / "operational_metrics.json")
        if not isinstance(operational, dict):
            raise ValueError("operational_metrics.json must contain an object")
        print(
            json.dumps(
                {
                    "status": "completed",
                    "output": str(output),
                    "production_bundle_sha256": operational["bundle_sha256"],
                    "production_manifest_sha256": operational["bundle_manifest_sha256"],
                    "run_seal_sha256": sha256_file(output / "run_seal.json"),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare-run":
        config = load_config(args.config)
        if config.data.kind != "criteo_v2":
            raise ValueError("prepare-run requires a config with data.kind='criteo_v2'")
        validation = validate_criteo_file(
            args.data,
            config.data,
            require_official_file=True,
        )
        contract_hash = write_selection_contract(
            args.output,
            config,
            validation,
            max_rows=args.max_rows,
            sample_offset=args.sample_offset,
        )
        print(
            json.dumps(
                {
                    "status": "selection_contract_sealed",
                    "output": str(Path(args.output)),
                    "selection_contract_sha256": contract_hash,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        config = load_config(args.config)
        if config.data.kind != "criteo_v2":
            raise ValueError("validate requires a config with data.kind='criteo_v2'")
        report = validate_criteo_file(
            args.data,
            config.data,
            max_rows=args.max_rows,
            require_official_file=args.require_official_file,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "score":
        if not 1 <= args.max_records <= MAX_SCORE_RECORDS:
            raise ValueError(f"--max-records must be in [1, {MAX_SCORE_RECORDS}]")
        bundle_path = Path(args.bundle).resolve(strict=False)
        manifest_path = (
            Path(args.manifest).resolve(strict=False)
            if args.manifest is not None
            else bundle_path.with_suffix(".manifest.json")
        )
        input_path = Path(args.input).resolve(strict=False)
        output_path = Path(args.output).resolve(strict=False)
        protected_inputs = {
            "bundle": bundle_path,
            "manifest": manifest_path,
            "input": input_path,
        }
        collisions = sorted(name for name, path in protected_inputs.items() if path == output_path)
        if collisions:
            raise ValueError(
                "score output must not overwrite protected input paths: " + ", ".join(collisions)
            )
        with _score_csv_snapshot(input_path) as input_snapshot:
            _validate_score_csv_header(input_snapshot)
            frame = pd.read_csv(
                input_snapshot,
                nrows=args.max_records + 1,
                dtype={"row_key": "string"},
            )
        if len(frame) > args.max_records:
            raise ValueError(f"score input exceeds --max-records={args.max_records}")
        bundle = load_verified_bundle(
            bundle_path,
            expected_sha256=args.expected_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
            manifest_path=manifest_path,
        )
        result = bundle.score(frame)
        atomic_csv(output_path, result)
        print(json.dumps({"status": "completed", "rows": len(result), "output": str(output_path)}))
        return 0
    if args.command == "report":
        outputs = regenerate_reports(
            args.run_dir,
            expected_seal_sha256=args.expected_seal_sha256,
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    **outputs,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verify-run":
        print(
            json.dumps(
                verify_evidence_inventory(
                    args.run_dir,
                    expected_seal_sha256=args.expected_seal_sha256,
                ),
                sort_keys=True,
            )
        )
        return 0
    raise RuntimeError("unreachable command")


def entrypoint() -> int:
    """Render expected user/input failures without exposing an internal traceback."""

    try:
        return main()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
