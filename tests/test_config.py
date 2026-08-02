from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from orbit_causal.config import (
    MAX_CONFIG_BYTES,
    MAX_DATA_ROWS,
    MAX_FEATURE_COLUMNS,
    MAX_MODEL_DEPTH,
    MAX_MODEL_ITERATIONS,
    MAX_POLICY_GRID_VALUES,
    load_config,
)


class ConfigTests(unittest.TestCase):
    def test_demo_config_is_valid_and_stable(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "demo.json")
        self.assertEqual(config.data.outcome, "visit")
        self.assertAlmostEqual(
            config.split.train
            + config.split.calibration
            + config.split.policy
            + config.split.final_test,
            1.0,
        )
        self.assertEqual(len(config.fingerprint()), 64)
        self.assertEqual(config.fingerprint(), config.fingerprint())

    def test_criteo_provider_schema_keeps_dense_features_continuous(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "criteo_v2.json")
        self.assertEqual(config.data.categorical_columns, ())
        self.assertEqual(len(config.data.feature_columns), 12)

    def test_post_treatment_feature_is_rejected(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "demo.json")
        leaked = replace(
            config,
            data=replace(config.data, feature_columns=(*config.data.feature_columns, "exposure")),
        )
        with self.assertRaisesRegex(ValueError, "feature leakage"):
            leaked.validate()

    def test_invalid_numeric_and_evaluation_settings_are_rejected(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "demo.json")
        with self.assertRaisesRegex(ValueError, "bootstrap_replicates"):
            replace(
                config, evaluation=replace(config.evaluation, bootstrap_replicates=1)
            ).validate()

    def test_every_bounded_configuration_family_fails_closed(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "demo.json")
        cases = (
            ("schema_version", replace(config, schema_version="2.0")),
            ("run_name", replace(config, run_name="unsafe name")),
            ("seed", replace(config, seed=2**32)),
            ("data.kind", replace(config, data=replace(config.data, kind="unknown"))),
            ("outcome", replace(config, data=replace(config.data, outcome="revenue"))),
            ("data.n_rows", replace(config, data=replace(config.data, n_rows=0))),
            (
                "data.n_rows",
                replace(config, data=replace(config.data, n_rows=MAX_DATA_ROWS + 1)),
            ),
            (
                "feature_columns must be unique",
                replace(
                    config,
                    data=replace(
                        config.data,
                        feature_columns=(
                            *config.data.feature_columns,
                            config.data.feature_columns[0],
                        ),
                    ),
                ),
            ),
            (
                "categorical_columns must be unique",
                replace(config, data=replace(config.data, categorical_columns=("f6", "f6"))),
            ),
            (
                "must be distinct",
                replace(config, data=replace(config.data, treatment=config.data.exposure)),
            ),
            (
                "scenario_capacities must be unique",
                replace(
                    config,
                    policy=replace(config.policy, scenario_capacities=(0.2, 0.2)),
                ),
            ),
            (
                "risk_multipliers must be unique",
                replace(
                    config,
                    policy=replace(config.policy, risk_multipliers=(0.0, 0.25, 0.25)),
                ),
            ),
            (
                "intervention_cost",
                replace(config, policy=replace(config.policy, intervention_cost=float("inf"))),
            ),
            (
                "max_abstention",
                replace(config, policy=replace(config.policy, max_abstention=1.01)),
            ),
            ("at least two", replace(config, model=replace(config.model, dr_folds=1))),
            ("must not exceed 20", replace(config, model=replace(config.model, dr_folds=21))),
            (
                "max_iter and max_depth must be positive",
                replace(config, model=replace(config.model, max_depth=0)),
            ),
            (
                "max_depth must not exceed",
                replace(config, model=replace(config.model, max_depth=MAX_MODEL_DEPTH + 1)),
            ),
            (
                "learning_rate",
                replace(config, model=replace(config.model, learning_rate=0.0)),
            ),
            (
                "min_samples_leaf",
                replace(config, model=replace(config.model, min_samples_leaf=0)),
            ),
            (
                "categorical_hash_bins",
                replace(config, model=replace(config.model, categorical_hash_bins=256)),
            ),
            (
                "calibration_bins",
                replace(
                    config,
                    evaluation=replace(config.evaluation, calibration_bins=1),
                ),
            ),
            (
                "evaluation alpha",
                replace(config, evaluation=replace(config.evaluation, alpha=1.0)),
            ),
            (
                "slice_noninferiority_margin",
                replace(
                    config,
                    evaluation=replace(config.evaluation, slice_noninferiority_margin=-0.1),
                ),
            ),
            (
                "minimum_slice_rows",
                replace(config, evaluation=replace(config.evaluation, minimum_slice_rows=0)),
            ),
            (
                "max_inference_ms_per_1000",
                replace(
                    config,
                    evaluation=replace(config.evaluation, max_inference_ms_per_1000=0.0),
                ),
            ),
            (
                "max_bundle_size_bytes",
                replace(config, evaluation=replace(config.evaluation, max_bundle_size_bytes=0)),
            ),
        )
        for message, invalid in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                invalid.validate()
        with self.assertRaisesRegex(ValueError, "categorical_columns"):
            replace(
                config,
                data=replace(config.data, categorical_columns=("not_a_feature",)),
            ).validate()
        with self.assertRaisesRegex(ValueError, "primary_capacity"):
            replace(
                config,
                policy=replace(config.policy, primary_capacity=float("nan")),
            ).validate()
        with self.assertRaisesRegex(ValueError, "max_iter"):
            replace(config, model=replace(config.model, max_iter=True)).validate()
        with self.assertRaisesRegex(ValueError, "primary_capacity"):
            replace(
                config,
                policy=replace(config.policy, scenario_capacities=(0.1, 0.5)),
            ).validate()
        with self.assertRaisesRegex(ValueError, "zero-risk"):
            replace(
                config,
                policy=replace(config.policy, risk_multipliers=(0.5, 1.0)),
            ).validate()
        with self.assertRaisesRegex(ValueError, "reserved"):
            replace(
                config,
                data=replace(config.data, exposure="split"),
            ).validate()
        with self.assertRaisesRegex(ValueError, "max_iter must not exceed"):
            replace(
                config,
                model=replace(config.model, max_iter=MAX_MODEL_ITERATIONS + 1),
            ).validate()
        with self.assertRaisesRegex(ValueError, "feature_columns must not exceed"):
            replace(
                config,
                data=replace(
                    config.data,
                    feature_columns=tuple(f"x{index}" for index in range(MAX_FEATURE_COLUMNS + 1)),
                    categorical_columns=(),
                ),
            ).validate()
        with self.assertRaisesRegex(ValueError, "risk_multipliers must not exceed"):
            replace(
                config,
                policy=replace(
                    config.policy,
                    risk_multipliers=tuple(
                        float(index) for index in range(MAX_POLICY_GRID_VALUES + 1)
                    ),
                ),
            ).validate()

    def test_unknown_fields_and_implicit_type_coercion_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            unknown = Path(directory) / "unknown.json"
            unknown.write_text('{"schema_verzion": "1.0"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown config fields"):
                load_config(unknown)
            coerced = Path(directory) / "coerced.json"
            coerced.write_text('{"seed": "123"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed must be an integer"):
                load_config(coerced)
            scalar_array = Path(directory) / "scalar.json"
            scalar_array.write_text('{"policy": {"scenario_capacities": 0.2}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an array"):
                load_config(scalar_array)
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_config(duplicate)
            nonstandard = Path(directory) / "nan.json"
            nonstandard.write_text('{"policy": {"primary_capacity": NaN}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonstandard JSON"):
                load_config(nonstandard)
            oversized = Path(directory) / "oversized.json"
            oversized.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "1 MiB"):
                load_config(oversized)


if __name__ == "__main__":
    unittest.main()
