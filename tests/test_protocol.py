from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orbit_causal.artifacts import sha256_file
from orbit_causal.config import DataConfig, ProjectConfig
from orbit_causal.data import CriteoFileValidationReport, deterministic_sample_row_ids
from orbit_causal.protocol import validate_selection_contract, write_selection_contract


class SelectionProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProjectConfig(
            data=DataConfig(
                kind="criteo_v2",
                feature_columns=tuple(f"f{index}" for index in range(12)),
                categorical_columns=(),
            )
        )
        self.validation = CriteoFileValidationReport(
            schema_version="1.0",
            row_count=1_000,
            feature_count=8,
            treatment_rate=0.5,
            visit_rate=0.1,
            conversion_rate=0.01,
            source_size_bytes=123,
            source_sha256="a" * 64,
            gzip_encoded=True,
            gzip_crc_validated=True,
            full_file_validated=True,
            official_file_match=True,
        )

    def test_contract_binds_candidate_protocol_and_disjoint_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection-contract.json"
            contract_hash = write_selection_contract(
                path,
                self.config,
                self.validation,
                max_rows=250,
                sample_offset=250,
            )
            source_ids = deterministic_sample_row_ids(1_000, 250, 250)
            contract = validate_selection_contract(
                path,
                expected_sha256=contract_hash,
                config=self.config,
                source_validation=self.validation.to_dict(),
                source_row_ids=source_ids,
                max_rows=250,
                sample_offset=250,
            )
            self.assertEqual(contract["release_protocol"]["primary_candidate"], "C1")
            self.assertEqual(contract["release_protocol"]["superiority_baseline"], "C3")
            self.assertEqual(contract["release_protocol"]["c6_role"], "negative_ablation")
            self.assertEqual(sha256_file(path), contract_hash)

    def test_contract_rejects_wrong_window_and_untrusted_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection-contract.json"
            contract_hash = write_selection_contract(
                path,
                self.config,
                self.validation,
                max_rows=250,
                sample_offset=250,
            )
            with self.assertRaisesRegex(ValueError, "trusted selection-contract"):
                validate_selection_contract(
                    path,
                    expected_sha256="0" * 64,
                    config=self.config,
                    source_validation=self.validation.to_dict(),
                    source_row_ids=deterministic_sample_row_ids(1_000, 250, 250),
                    max_rows=250,
                    sample_offset=250,
                )
            with self.assertRaisesRegex(ValueError, "sample-row admission"):
                validate_selection_contract(
                    path,
                    expected_sha256=contract_hash,
                    config=self.config,
                    source_validation=self.validation.to_dict(),
                    source_row_ids=deterministic_sample_row_ids(1_000, 250, 0),
                    max_rows=250,
                    sample_offset=250,
                )

    def test_contract_creation_never_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection-contract.json"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to replace"):
                write_selection_contract(
                    path,
                    self.config,
                    self.validation,
                    max_rows=250,
                    sample_offset=250,
                )
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
