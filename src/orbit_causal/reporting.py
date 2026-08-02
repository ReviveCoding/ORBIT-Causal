from __future__ import annotations

import atexit
import html
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    _MPL_CONFIG_DIR = Path(tempfile.mkdtemp(prefix="orbit-mpl-"))
    os.environ["MPLCONFIGDIR"] = str(_MPL_CONFIG_DIR)
    atexit.register(shutil.rmtree, _MPL_CONFIG_DIR, ignore_errors=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import _fsync_directory, atomic_json, atomic_text, read_strict_json


def _read_json_object(path: Path) -> dict[str, Any]:
    value = read_strict_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _interval_errors(point: pd.Series, lower: pd.Series, upper: pd.Series) -> np.ndarray:
    """Return nonnegative plotting extents even when a percentile CI excludes its point."""

    point_array = point.to_numpy(dtype=float)
    lower_array = np.minimum(lower.to_numpy(dtype=float), point_array)
    upper_array = np.maximum(upper.to_numpy(dtype=float), point_array)
    return np.vstack([point_array - lower_array, upper_array - point_array])


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: f"{value:.6f}" if pd.notna(value) else "NA"
            )
        else:
            display[column] = display[column].astype(str)
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _validation_boundary(data_role: str) -> tuple[str, str]:
    if data_role == "synthetic_fixture":
        return (
            "Public-data, GPU, temporal, campaign, and financial-domain validation require "
            "separate completed manifests.",
            "This fixture run does not establish public-data, GPU, temporal, campaign, or "
            "financial-domain validity.",
        )
    if data_role == "public_randomized_benchmark_subset":
        return (
            "The bounded public-data subset does not establish full-benchmark, GPU, temporal, "
            "campaign, or financial-domain validity.",
            "This official public-data subset run does not establish full-benchmark, GPU, "
            "temporal, campaign, or financial-domain validity.",
        )
    if data_role == "public_randomized_benchmark":
        return (
            "The official public benchmark does not establish GPU, temporal, campaign, "
            "financial-domain, or production-impact validity.",
            "This official full public-benchmark run does not establish GPU, temporal, "
            "campaign, financial-domain, or production-impact validity.",
        )
    raise ValueError(f"unsupported data role in report: {data_role!r}")


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[tuple[Path, Path]] = []
    try:
        for suffix in (".png", ".svg"):
            target = base_path.with_suffix(suffix)
            with tempfile.NamedTemporaryFile(
                dir=base_path.parent,
                prefix=f".{base_path.name}.",
                suffix=suffix,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            temporaries.append((temporary, target))
            if suffix == ".png":
                fig.savefig(temporary, dpi=160, bbox_inches="tight")
            else:
                fig.savefig(temporary, bbox_inches="tight")
            with temporary.open("rb+") as handle:
                os.fsync(handle.fileno())
        for temporary, target in temporaries:
            os.replace(temporary, target)
        _fsync_directory(base_path.parent)
    finally:
        plt.close(fig)
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)


def build_figures(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    run_manifest = _read_json_object(run_dir / "run_manifest.json")
    primary_capacity = float(run_manifest.get("policy_contract", {}).get("primary_capacity", 0.20))
    if not np.isfinite(primary_capacity) or primary_capacity <= 0 or primary_capacity > 1:
        raise ValueError("run manifest contains an invalid primary capacity")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    split = pd.read_csv(run_dir / "split_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(split["split"], split["n"], color="#4C78A8")
    axes[0].set_title("Rows by immutable split")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(split["split"], split["outcome_rate"], color="#F58518")
    axes[1].set_title("Outcome rate by split")
    axes[1].tick_params(axis="x", rotation=25)
    _save_figure(fig, figures / "data_and_split")
    outputs.extend(["figures/data_and_split.png", "figures/data_and_split.svg"])

    quality = _read_json_object(run_dir / "data_quality.json")
    quality_keys = [
        "missing_count",
        "duplicate_profiles",
        "invalid_binary_count",
        "invalid_categorical_count",
        "exposure_constraint_violations",
        "conversion_constraint_violations",
        "extreme_numeric_count",
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(quality_keys, [quality.get(key, 0) for key in quality_keys], color="#72B7B2")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Data-quality admission diagnostics")
    ax.set_ylabel("Count")
    _save_figure(fig, figures / "data_quality")
    outputs.extend(["figures/data_quality.png", "figures/data_quality.svg"])

    comparison = pd.read_csv(run_dir / "candidate_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(comparison["candidate_id"], comparison["aipw_incremental_value"], color="#54A24B")
    axes[0].errorbar(
        comparison["candidate_id"],
        comparison["aipw_incremental_value"],
        yerr=_interval_errors(
            comparison["aipw_incremental_value"],
            comparison["aipw_ci_lower"],
            comparison["aipw_ci_upper"],
        ),
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[0].set_title("Locked-test AIPW incremental value")
    axes[1].scatter(
        comparison["inference_ms_per_1000"],
        comparison["aipw_incremental_value"],
        s=70,
        c=np.arange(len(comparison)),
        cmap="viridis",
    )
    for _, row in comparison.iterrows():
        axes[1].annotate(
            row["candidate_id"], (row["inference_ms_per_1000"], row["aipw_incremental_value"])
        )
    axes[1].set_xlabel("Inference ms / 1,000 rows")
    axes[1].set_ylabel("AIPW value")
    axes[1].set_title("Quality-resource frontier")
    _save_figure(fig, figures / "candidate_comparison")
    outputs.extend(["figures/candidate_comparison.png", "figures/candidate_comparison.svg"])

    scenarios = pd.read_csv(run_dir / "decision_scenarios.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for candidate, part in scenarios.groupby("candidate_id"):
        ax.plot(part["capacity"], part["incremental_value"], marker="o", label=candidate)
    ax.axvline(
        primary_capacity,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"primary {primary_capacity:.1%}",
    )
    ax.set_xlabel("Treatment capacity")
    ax.set_ylabel("AIPW incremental value")
    ax.set_title("Capacity-value policy frontier")
    ax.legend(fontsize=8)
    _save_figure(fig, figures / "policy_frontier")
    outputs.extend(["figures/policy_frontier.png", "figures/policy_frontier.svg"])

    calibration = pd.read_csv(run_dir / "calibration_table.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    limits = [
        min(calibration["predicted_effect"].min(), calibration["dr_observed_effect"].min()),
        max(calibration["predicted_effect"].max(), calibration["dr_observed_effect"].max()),
    ]
    for candidate, part in calibration.groupby("candidate_id"):
        ax.plot(part["predicted_effect"], part["dr_observed_effect"], marker="o", label=candidate)
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted effect")
    ax.set_ylabel("DR-observed bin effect")
    ax.set_title("Causal calibration by locked bins")
    ax.legend(fontsize=8)
    _save_figure(fig, figures / "causal_calibration")
    outputs.extend(["figures/causal_calibration.png", "figures/causal_calibration.svg"])

    prediction_path = run_dir / "predictions.csv"
    prediction_columns = pd.read_csv(prediction_path, nrows=0).columns
    order_column = "c6_decision_score" if "c6_decision_score" in prediction_columns else "c6_score"
    selected_prediction_columns = list(
        dict.fromkeys(["rsec_correction", "c6_lower", "c6_upper", "c6_score", order_column])
    )
    predictions = pd.read_csv(prediction_path, usecols=selected_prediction_columns)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(predictions["rsec_correction"], bins=30, color="#B279A2")
    axes[0].set_title("RSEC correction distribution")
    axes[0].set_xlabel("Effect correction")
    envelope = predictions.sort_values(order_column, ascending=False).head(
        min(500, len(predictions))
    )
    axes[1].fill_between(
        np.arange(len(envelope)),
        envelope["c6_lower"],
        envelope["c6_upper"],
        alpha=0.3,
        color="#E45756",
        label="decision envelope",
    )
    axes[1].plot(envelope["c6_score"].to_numpy(), color="#4C78A8", linewidth=1, label="effect")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("CAS coverage-overreach envelope")
    axes[1].legend(fontsize=8)
    _save_figure(fig, figures / "specialty_diagnostics")
    outputs.extend(["figures/specialty_diagnostics.png", "figures/specialty_diagnostics.svg"])

    slices = pd.read_csv(run_dir / "slice_metrics.csv")
    pivot = slices.pivot(index="slice", columns="candidate_id", values="incremental_value")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("Worst-slice incremental policy value")
    fig.colorbar(image, ax=ax, label="AIPW incremental value")
    _save_figure(fig, figures / "worst_slice")
    outputs.extend(["figures/worst_slice.png", "figures/worst_slice.svg"])

    failures = pd.read_csv(run_dir / "failure_cases.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(failures["failure_score"], bins=min(20, len(failures)), color="#E45756")
    axes[0].set_title("Highest-priority failure scores")
    axes[0].set_xlabel("Failure score")
    axes[1].scatter(
        failures["c6_score"],
        failures.get("true_tau", failures["c6_score"]),
        c=failures["c6_uncertainty"],
        cmap="magma",
    )
    axes[1].set_xlabel("C6 estimated effect")
    axes[1].set_ylabel("Ground truth or score proxy")
    axes[1].set_title("Hard-case localization")
    _save_figure(fig, figures / "failure_cases")
    outputs.extend(["figures/failure_cases.png", "figures/failure_cases.svg"])

    fig, ax = plt.subplots(figsize=(12, 3.5))
    stages = ["Admit", "Split", "Train", "Calibrate", "Policy", "Evaluate", "Qualify"]
    xs = np.arange(len(stages))
    ax.scatter(xs, np.zeros_like(xs), s=1800, c="#4C78A8")
    for index, stage in enumerate(stages):
        ax.text(index, 0, stage, ha="center", va="center", color="white", fontsize=9)
        if index < len(stages) - 1:
            ax.annotate(
                "", xy=(index + 0.75, 0), xytext=(index + 0.25, 0), arrowprops={"arrowstyle": "->"}
            )
    ax.set_xlim(-0.7, len(stages) - 0.3)
    ax.set_ylim(-0.7, 0.7)
    ax.axis("off")
    ax.set_title("ORBIT-Causal sealed decision pipeline")
    _save_figure(fig, figures / "architecture")
    outputs.extend(["figures/architecture.png", "figures/architecture.svg"])

    specifications = {
        "data_and_split": (
            "Are immutable partitions balanced in size and outcome rate?",
            "split_summary.csv",
            "split row count and empirical outcome rate",
            "Splits are approximately sized and can be inspected for target drift.",
            "Aggregate rates do not rule out all covariate imbalance.",
        ),
        "data_quality": (
            "Did schema and structural data-admission checks find defects?",
            "data_quality.json",
            "quality defect counts",
            "Zero bars are expected for a clean generated fixture.",
            "A clean fixture does not establish public-data quality.",
        ),
        "candidate_comparison": (
            "How does the pre-specified primary compare with locked challengers?",
            "candidate_comparison.csv",
            "AIPW incremental value with pretreatment-profile cluster bootstrap interval",
            "C1 value and latency are visible against diagnostic challengers.",
            "Fixture intervals do not imply deployment impact.",
        ),
        "policy_frontier": (
            "How does intervention capacity change incremental policy value?",
            "decision_scenarios.csv",
            "AIPW incremental value by capacity",
            "The configured primary operating point can be compared with alternatives.",
            "Intervention cost is configurable and zero in the demo.",
        ),
        "causal_calibration": (
            "Do predicted effect bins align with DR-observed bin effects?",
            "calibration_table.csv",
            "weighted root mean square causal calibration gap",
            "The diagonal exposes systematic effect-scale bias.",
            "Pseudo-outcome bins are noisy and not individual effect labels.",
        ),
        "specialty_diagnostics": (
            "How large is RSEC correction and how wide is the CAS envelope?",
            "predictions.csv",
            "correction distribution and selected decision envelope",
            "Clipping and guard width expose specialty behavior.",
            "The envelope is heuristic rather than a causal confidence interval.",
        ),
        "worst_slice": (
            "Which technical slice has the weakest policy value?",
            "slice_metrics.csv",
            "AIPW incremental value by first-feature quintile",
            "Local failures hidden by the aggregate become visible.",
            "This is a technical slice, not a protected-group fairness audit.",
        ),
        "failure_cases": (
            "Where are the largest fixture effect errors or uncertainty proxies?",
            "failure_cases.csv",
            "absolute effect error or uncertainty-correction proxy",
            "Top hard cases are retained for replay and diagnosis.",
            "Real-data rows lack individual effect ground truth.",
        ),
        "architecture": (
            "Does evidence flow from admission to a qualified artifact without test leakage?",
            "repository pipeline contract",
            "ordered sealed stages",
            "Candidate selection precedes locked-test reporting.",
            "The diagram does not prove runtime correctness; tests and manifests do.",
        ),
    }
    records = []
    for name, spec in specifications.items():
        records.append(
            {
                "figure_id": name,
                "title": name.replace("_", " ").title(),
                "question": spec[0],
                "data_source": spec[1],
                "candidate_or_run": "all candidates in supplied run",
                "metric_definition": spec[2],
                "paths": [f"figures/{name}.png", f"figures/{name}.svg"],
                "generation_command": "orbit-causal report --run-dir <run-dir>",
                "interpretation": spec[3],
                "limitation": spec[4],
            }
        )
    atomic_json(
        run_dir / "visualization_manifest.json",
        {
            "canonical_command": "orbit-causal report --run-dir <run-dir>",
            "figures": records,
        },
    )
    return outputs


def build_report(run_dir: str | Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    comparison = pd.read_csv(run_dir / "candidate_comparison.csv")
    scenarios = pd.read_csv(run_dir / "decision_scenarios.csv")
    qualification = _read_json_object(run_dir / "qualification_manifest.json")
    manifest = _read_json_object(run_dir / "run_manifest.json")

    production_candidate = qualification.get(
        "production_candidate", qualification.get("production_champion")
    )
    if not isinstance(production_candidate, str):
        raise ValueError("qualification manifest is missing a production candidate")
    authorized_champion = qualification.get("production_champion")
    row = comparison.loc[comparison["candidate_id"] == production_candidate].iloc[0]
    confidence_level = float(manifest.get("evaluation_contract", {}).get("confidence_level", 0.95))
    if not np.isfinite(confidence_level) or confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("run manifest contains an invalid confidence level")
    confidence_label = f"{100.0 * confidence_level:g}%"
    markdown_boundary, html_boundary = _validation_boundary(str(manifest["data_role"]))
    markdown = f"""# ORBIT-Causal Experiment Report

## Validation status

- Run ID: `{manifest["run_id"]}`
- Data role: `{manifest["data_role"]}`
- Verdict: **{qualification["release_verdict"]}**
- Predictive champion selected on policy split: `{qualification["predictive_champion"]}`
- Production candidate pre-specified before locked-test evaluation: `{production_candidate}`
- Release-authorized production champion: `{authorized_champion or "NONE"}`
- Locked-test AIPW incremental value: `{row["aipw_incremental_value"]:.6f}`
- {confidence_label} percentile bootstrap interval: `[{row["aipw_ci_lower"]:.6f}, {row["aipw_ci_upper"]:.6f}]`

These values come from a generated causal fixture unless `data_role` says otherwise. They are not financial-client or production-impact estimates.

## Candidate comparison

{_markdown_table(comparison)}

## Decision scenarios

{_markdown_table(scenarios)}

## Interpretation

The primary candidate and C3 superiority baseline were sealed before the locked holdout was evaluated. C6 remains a negative guarded-policy ablation. Final-test outcomes report the pre-specified comparison and cannot select a replacement candidate.

## Figures

![Candidate comparison](figures/candidate_comparison.png)

![Data quality](figures/data_quality.png)

![Policy frontier](figures/policy_frontier.png)

![Causal calibration](figures/causal_calibration.png)

![Specialty diagnostics](figures/specialty_diagnostics.png)

![Worst slice](figures/worst_slice.png)

![Failure cases](figures/failure_cases.png)

## Limitations

- Real individual counterfactuals are unobservable outside the generated ground-truth track.
- Ensemble dispersion is a decision envelope, not a guaranteed individual causal confidence interval.
- Randomized grouped partitions do not establish temporal, campaign, or financial-domain generalization.
- {markdown_boundary}
"""
    markdown_path = run_dir / "report.md"
    atomic_text(markdown_path, markdown)

    html_table = comparison.to_html(index=False, float_format=lambda x: f"{x:.6f}")
    html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ORBIT-Causal Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:6px}}img{{max-width:100%}}.warn{{background:#fff3cd;padding:12px}}</style></head>
<body><h1>ORBIT-Causal Experiment Report</h1>
<div class="warn">Data role: {html.escape(str(manifest["data_role"]))}. No financial or production-impact claim is authorized.</div>
<p><b>Run:</b> {html.escape(manifest["run_id"])}<br><b>Verdict:</b> {html.escape(qualification["release_verdict"])}<br><b>Production candidate:</b> {html.escape(production_candidate)}<br><b>Release-authorized champion:</b> {html.escape(str(authorized_champion or "NONE"))}<br><b>{html.escape(confidence_label)} percentile bootstrap interval:</b> [{row["aipw_ci_lower"]:.6f}, {row["aipw_ci_upper"]:.6f}]</p>
<h2>Candidate comparison</h2>{html_table}
<h2>Decision evidence</h2><img src="figures/data_quality.png"><img src="figures/candidate_comparison.png"><img src="figures/policy_frontier.png"><img src="figures/causal_calibration.png"><img src="figures/specialty_diagnostics.png"><img src="figures/worst_slice.png"><img src="figures/failure_cases.png">
<h2>Boundary</h2><p>Decision envelopes are heuristic uncertainty guards. {html.escape(html_boundary)}</p>
</body></html>"""
    html_path = run_dir / "report.html"
    atomic_text(html_path, html_report)
    return markdown_path, html_path
