from __future__ import annotations

import gzip
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from orbit_causal.config import DataConfig, SplitConfig
from orbit_causal.data import (
    _categorical_token,
    _profile_hash,
    _validate_criteo_contents,
    assign_splits,
    authenticated_criteo_snapshot,
    dataset_fingerprint,
    deterministic_sample_row_ids,
    generate_synthetic_data,
    load_criteo_csv,
    profile_ids,
    sample_row_ids_sha256,
    subgroup_summary,
    validate_criteo_file,
    validate_frame,
)


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DataConfig()
        self.frame = generate_synthetic_data(2000, 19)

    def test_fixture_satisfies_structural_contract(self) -> None:
        report = validate_frame(self.frame, self.config)
        self.assertEqual(report.row_count, 2000)
        self.assertEqual(report.exposure_constraint_violations, 0)
        self.assertEqual(report.conversion_constraint_violations, 0)

    def test_criteo_loader_projects_schema_and_rejects_duplicate_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "criteo.csv"
            frame = pd.DataFrame(
                {f"f{index}": [float(index), float(index + 1)] for index in range(12)}
            )
            frame = frame.assign(
                treatment=[0, 1],
                exposure=[0, 1],
                visit=[0, 1],
                conversion=[0, 0],
                ignored=["x", "y"],
            )
            frame.to_csv(path, index=False)
            loaded = load_criteo_csv(path, max_rows=1)
            self.assertEqual(len(loaded), 1)
            self.assertNotIn("ignored", loaded)
            ordered = pd.DataFrame(
                {f"f{index}": np.arange(1000, dtype=float) for index in range(12)}
            ).assign(
                treatment=np.r_[np.ones(850, dtype=int), np.zeros(150, dtype=int)],
                exposure=np.r_[np.ones(850, dtype=int), np.zeros(150, dtype=int)],
                visit=np.arange(1000) % 2,
                conversion=np.zeros(1000, dtype=int),
            )
            ordered.to_csv(path, index=False)
            sampled = load_criteo_csv(path, max_rows=250)
            self.assertEqual(set(sampled["treatment"]), {0, 1})
            self.assertGreater(int(sampled["source_row_id"].max()), 850)
            path.write_text("f0,f0\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate columns"):
                load_criteo_csv(path)

    def test_criteo_gzip_streaming_validator_pins_complete_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "criteo.csv.gz"
            frame = pd.DataFrame(
                {f"f{index}": [float(index), float(index + 1)] for index in range(12)}
            ).assign(
                treatment=[0, 1],
                exposure=[0, 1],
                visit=[0, 1],
                conversion=[0, 0],
            )
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            report = validate_criteo_file(
                path,
                DataConfig(
                    kind="criteo_v2",
                    feature_columns=tuple(f"f{index}" for index in range(12)),
                    categorical_columns=(),
                ),
                chunk_rows=1,
                require_official_file=True,
                expected_rows=2,
                expected_size_bytes=path.stat().st_size,
                expected_sha256=checksum,
            )
            self.assertEqual(report.row_count, 2)
            self.assertTrue(report.gzip_crc_validated)
            self.assertTrue(report.official_file_match)
            self.assertEqual(len(load_criteo_csv(path)), 2)

    def test_official_validator_parses_the_authenticated_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "criteo.csv.gz"
            frame = pd.DataFrame(
                {f"f{index}": [float(index), float(index + 1)] for index in range(12)}
            ).assign(
                treatment=[0, 1],
                exposure=[0, 1],
                visit=[0, 1],
                conversion=[0, 0],
            )
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            config = DataConfig(
                kind="criteo_v2",
                feature_columns=tuple(f"f{index}" for index in range(12)),
                categorical_columns=(),
            )
            with patch(
                "orbit_causal.data._validate_criteo_contents",
                wraps=_validate_criteo_contents,
            ) as validator:
                report = validate_criteo_file(
                    path,
                    config,
                    require_official_file=True,
                    expected_rows=2,
                    expected_size_bytes=path.stat().st_size,
                    expected_sha256=checksum,
                )
            validated_path = Path(validator.call_args.args[0])
            self.assertNotEqual(validated_path, path)
            self.assertFalse(validated_path.exists())
            self.assertTrue(report.official_file_match)

    def test_criteo_streaming_validator_rejects_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "criteo.csv.gz"
            frame = pd.DataFrame(
                {f"f{index}": [float(index), float(index + 1)] for index in range(12)}
            ).assign(
                treatment=[0, 1],
                exposure=[0, 1],
                visit=[0, 1],
                conversion=[0, 0],
            )
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)
            with self.assertRaisesRegex(ValueError, "pinned official"):
                validate_criteo_file(
                    path,
                    DataConfig(
                        kind="criteo_v2",
                        feature_columns=tuple(f"f{index}" for index in range(12)),
                        categorical_columns=(),
                    ),
                    require_official_file=True,
                    expected_rows=2,
                    expected_size_bytes=path.stat().st_size,
                    expected_sha256="0" * 64,
                )

    def test_authenticated_snapshot_freezes_the_exact_verified_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv.gz"
            original = b"verified source bytes"
            source.write_bytes(original)
            checksum = hashlib.sha256(original).hexdigest()
            with authenticated_criteo_snapshot(
                source,
                expected_size_bytes=len(original),
                expected_sha256=checksum,
            ) as snapshot:
                self.assertEqual(snapshot.read_bytes(), original)
                source.write_bytes(b"replacement")
                self.assertEqual(snapshot.read_bytes(), original)
            self.assertFalse(snapshot.exists())

            with (
                self.assertRaisesRegex(ValueError, "pinned official"),
                authenticated_criteo_snapshot(
                    source,
                    expected_size_bytes=len(original),
                    expected_sha256=checksum,
                ),
            ):
                self.fail("mismatched source unexpectedly authenticated")

    def test_authenticated_snapshot_stops_after_the_pinned_size_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "oversized.csv.gz"
            source.write_bytes(b"placeholder")

            class GuardedBytesIO(io.BytesIO):
                calls = 0

                def read(self, size: int | None = -1) -> bytes:
                    self.calls += 1
                    if self.calls > 1:
                        raise AssertionError(
                            "oversized input was read after crossing the size bound"
                        )
                    return super().read(size)

            guarded = GuardedBytesIO(b"x" * (2 * (1 << 20)))
            original_open = Path.open

            def patched_open(path: Path, *args, **kwargs):
                if path == source:
                    return guarded
                return original_open(path, *args, **kwargs)

            with (
                patch("orbit_causal.data.Path.open", new=patched_open),
                self.assertRaisesRegex(ValueError, "exceeds the pinned"),
                authenticated_criteo_snapshot(
                    source,
                    expected_size_bytes=1,
                    expected_sha256="0" * 64,
                ),
            ):
                self.fail("oversized source unexpectedly authenticated")
            self.assertEqual(guarded.calls, 1)

    def test_invalid_post_treatment_structure_is_rejected(self) -> None:
        broken = self.frame.copy()
        broken.loc[0, "treatment"] = 0
        broken.loc[0, "exposure"] = 1
        with self.assertRaisesRegex(ValueError, "exposure_violations"):
            validate_frame(broken, self.config)

    def test_split_is_deterministic_and_profile_safe(self) -> None:
        duplicate = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        first = assign_splits(duplicate, self.config, SplitConfig(), seed=77)
        second = assign_splits(duplicate, self.config, SplitConfig(), seed=77)
        self.assertTrue(first.equals(second))
        self.assertEqual(first.iloc[0], first.iloc[-1])
        self.assertEqual(set(first.unique()), {"train", "calibration", "policy", "final_test"})
        shuffled = duplicate.sample(frac=1, random_state=9)
        shuffled_split = assign_splits(shuffled, self.config, SplitConfig(), seed=77)
        self.assertTrue(first.sort_index().equals(shuffled_split.sort_index()))
        frozen_digest = hashlib.sha256("|".join(first).encode()).hexdigest()
        self.assertEqual(
            frozen_digest,
            "4b755f625b34301df556ce8c095456310309010d08eda7e38537d3aa8996dff2",
        )

    def test_priority_windows_are_deterministic_and_disjoint(self) -> None:
        first = deterministic_sample_row_ids(1_000, 250, 0)
        second = deterministic_sample_row_ids(1_000, 250, 250)
        self.assertEqual(len(first), 250)
        self.assertEqual(len(second), 250)
        self.assertEqual(len(np.intersect1d(first, second)), 0)
        np.testing.assert_array_equal(first, deterministic_sample_row_ids(1_000, 250, 0))
        self.assertNotEqual(sample_row_ids_sha256(first), sample_row_ids_sha256(second))

    def test_csv_priority_windows_match_the_row_id_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "criteo.csv"
            frame = pd.DataFrame(
                {f"f{index}": np.arange(1_000, dtype=float) + index for index in range(12)}
            ).assign(
                treatment=np.arange(1_000) % 2,
                exposure=np.arange(1_000) % 2,
                visit=np.arange(1_000) % 2,
                conversion=np.zeros(1_000, dtype=int),
            )
            frame.to_csv(path, index=False)
            first = load_criteo_csv(path, max_rows=250, sample_offset=0)
            second = load_criteo_csv(path, max_rows=250, sample_offset=250)
            expected_first = deterministic_sample_row_ids(1_000, 250, 0)
            expected_second = deterministic_sample_row_ids(1_000, 250, 250)
            np.testing.assert_array_equal(first["source_row_id"], expected_first)
            np.testing.assert_array_equal(second["source_row_id"], expected_second)
            self.assertTrue(set(first["source_row_id"]).isdisjoint(second["source_row_id"]))

    def test_profile_serialization_has_no_delimiter_or_type_collision(self) -> None:
        self.assertNotEqual(_profile_hash(("a|b", "c")), _profile_hash(("a", "b|c")))
        self.assertNotEqual(_profile_hash((1,)), _profile_hash(("1",)))
        self.assertEqual(_profile_hash((1, -0.0)), _profile_hash((1.0, 0.0)))
        self.assertEqual(_categorical_token(1), _categorical_token(1.0))
        self.assertNotEqual(_categorical_token(1), _categorical_token("1"))

    def test_vectorized_profile_hash_collision_fails_closed(self) -> None:
        frame = self.frame.iloc[:2].copy()
        forged = pd.Series(np.array([7, 7], dtype=np.uint64), index=frame.index)
        with (
            patch("orbit_causal.data.pd.util.hash_pandas_object", return_value=forged),
            self.assertRaisesRegex(RuntimeError, "hash collision"),
        ):
            profile_ids(frame, self.config)

    def test_profile_identity_does_not_merge_large_adjacent_integers(self) -> None:
        config = DataConfig(feature_columns=("f0",), categorical_columns=())
        boundary = 2**53
        frame = pd.DataFrame({"f0": pd.Series([boundary, boundary + 1], dtype="uint64")})
        with self.assertRaisesRegex(RuntimeError, "hash collision"):
            profile_ids(frame, config)

    def test_empty_and_malformed_numeric_data_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one row"):
            validate_frame(self.frame.iloc[:0], self.config)
        malformed = self.frame.copy()
        malformed["f0"] = malformed["f0"].astype(object)
        malformed.loc[0, "f0"] = "not-numeric"
        with self.assertRaisesRegex(ValueError, "invalid_numeric=1"):
            validate_frame(malformed, self.config)
        malformed_category = self.frame.copy()
        malformed_category["f6"] = malformed_category["f6"].astype(object)
        malformed_category.at[0, "f6"] = ["nested"]
        with self.assertRaisesRegex(ValueError, "invalid_categorical=1"):
            validate_frame(malformed_category, self.config)
        malformed_binary = self.frame.copy()
        malformed_binary["treatment"] = malformed_binary["treatment"].astype(object)
        malformed_binary.at[0, "treatment"] = "bad"
        with self.assertRaisesRegex(ValueError, "invalid_binary=1"):
            validate_frame(malformed_binary, self.config)
        complex_feature = self.frame.copy()
        complex_feature["f0"] = complex_feature["f0"].astype(complex)
        with self.assertRaisesRegex(ValueError, "invalid_numeric=2000"):
            validate_frame(complex_feature, self.config)
        complex_binary = self.frame.copy()
        complex_binary["treatment"] = complex_binary["treatment"].astype(complex)
        with self.assertRaisesRegex(ValueError, "invalid_binary=2000"):
            validate_frame(complex_binary, self.config)

    def test_fingerprint_covers_quality_and_fixture_truth_but_not_split(self) -> None:
        baseline = dataset_fingerprint(self.frame, self.config)
        changed_exposure = self.frame.copy()
        treated = changed_exposure.index[changed_exposure["treatment"] == 1][0]
        changed_exposure.loc[treated, "exposure"] = 1 - changed_exposure.loc[treated, "exposure"]
        self.assertNotEqual(baseline, dataset_fingerprint(changed_exposure, self.config))
        changed_truth = self.frame.copy()
        changed_truth.loc[0, "true_tau"] += 0.01
        self.assertNotEqual(baseline, dataset_fingerprint(changed_truth, self.config))
        with_split = self.frame.assign(split="train")
        self.assertEqual(baseline, dataset_fingerprint(with_split, self.config))

    def test_fingerprint_covers_source_row_identity(self) -> None:
        baseline = self.frame.assign(source_row_id=np.arange(len(self.frame)))
        changed = baseline.copy()
        changed.loc[0, "source_row_id"] = len(changed) + 1
        self.assertNotEqual(
            dataset_fingerprint(baseline, self.config),
            dataset_fingerprint(changed, self.config),
        )

    def test_empty_categorical_summary_has_stable_schema(self) -> None:
        continuous = DataConfig(categorical_columns=())
        summary = subgroup_summary(self.frame, continuous)
        self.assertTrue(summary.empty)
        self.assertEqual(
            list(summary.columns),
            [
                "feature",
                "value",
                "n",
                "share",
                "treatment_rate",
                "outcome_rate",
                "interpretation",
            ],
        )

    def test_categorical_summary_preserves_type_and_missing_identity(self) -> None:
        frame = self.frame.iloc[:3].copy()
        frame["f6"] = frame["f6"].astype(object)
        frame.loc[:, "f6"] = [1, "1", pd.NA]
        values = set(subgroup_summary(frame, self.config).query("feature == 'f6'")["value"])
        self.assertEqual(values, {"1", "<STRING>1", "<NA>"})


if __name__ == "__main__":
    unittest.main()
