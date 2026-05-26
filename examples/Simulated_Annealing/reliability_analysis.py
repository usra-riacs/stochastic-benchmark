"""Repeat-reliability helpers for the Simulated Annealing tutorial."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


DEFAULT_QUALITY_TARGET = 0.95
DEFAULT_RELATIVE_ERROR_THRESHOLD = 0.10
DEFAULT_TARGET_CONFIDENCE = 0.99
DEFAULT_INSTANCE_IDS = (0, 1, 2)

DIAGNOSTIC_COLUMNS = [
    "instance",
    "sweeps",
    "current_resource",
    "current_repeats",
    "required_repeats",
    "additional_repeats_required",
    "p_ci_lower",
    "p_ci_upper",
    "R_c",
    "R_c_ci_lower",
    "R_c_ci_upper",
    "R99",
    "R99_ci_lower",
    "R99_ci_upper",
    "CETS",
    "CETS_ci_lower",
    "CETS_ci_upper",
    "reliability_status",
    "statistically_unresolved",
]


def quality_threshold_from_baseline(
    best_value: float,
    random_value: float,
    quality_target: float = DEFAULT_QUALITY_TARGET,
) -> float:
    """Return the energy threshold induced by the tutorial performance ratio."""

    if not 0.0 <= quality_target <= 1.0:
        raise ValueError("quality_target must be between 0 and 1")
    return float(random_value) + quality_target * (float(best_value) - float(random_value))


def attach_quality_thresholds(
    runs: pd.DataFrame,
    baselines: Mapping[int, Mapping[str, float]] | pd.DataFrame,
    quality_target: float = DEFAULT_QUALITY_TARGET,
) -> pd.DataFrame:
    """Attach per-instance reliability success thresholds to SA run rows."""

    if "instance" not in runs.columns:
        raise ValueError("runs must include an instance column")

    baseline_frame = _baseline_frame(baselines)
    missing_instances = sorted(set(runs["instance"].unique()) - set(baseline_frame["instance"]))
    if missing_instances:
        raise ValueError(f"missing baselines for instances: {missing_instances}")

    thresholds = baseline_frame.copy()
    thresholds["quality_target"] = quality_target
    thresholds["quality_threshold"] = thresholds.apply(
        lambda row: quality_threshold_from_baseline(
            row["best_value"],
            row["random_value"],
            quality_target=quality_target,
        ),
        axis=1,
    )
    return runs.merge(
        thresholds[
            [
                "instance",
                "best_value",
                "random_value",
                "quality_target",
                "quality_threshold",
            ]
        ],
        on="instance",
        how="left",
    )


def load_selected_granular_runs(
    example_dir: str | Path,
    instance_ids: Sequence[int] = DEFAULT_INSTANCE_IDS,
) -> pd.DataFrame:
    """Load real granular SA runs for a selected set of instances."""

    example_path = Path(example_dir)
    frames = []
    for instance_id in instance_ids:
        data_path = example_path / "granular_data" / f"granular_data_{instance_id}.pkl"
        if not data_path.is_file():
            raise FileNotFoundError(f"granular SA data not found: {data_path}")
        frames.append(pd.read_pickle(data_path))

    if not frames:
        raise ValueError("at least one instance_id is required")
    return pd.concat(frames, ignore_index=True)


def load_selected_raw_runs(
    example_dir: str | Path,
    instance_ids: Sequence[int] = DEFAULT_INSTANCE_IDS,
) -> pd.DataFrame:
    """Load selected SA runs from the tracked aggregate raw-runs pickle."""

    raw_runs_path = Path(example_dir) / "results" / "all_raw_runs.pkl"
    if not raw_runs_path.is_file():
        raise FileNotFoundError(f"aggregate SA raw runs not found: {raw_runs_path}")

    with raw_runs_path.open("rb") as raw_runs_file:
        all_raw_runs = pickle.load(raw_runs_file)

    frames = []
    for instance_id in instance_ids:
        if instance_id not in all_raw_runs:
            raise ValueError(f"missing raw runs for instance: {instance_id}")

        for sweeps, energies in all_raw_runs[instance_id].items():
            frame = pd.DataFrame(
                {
                    "instance": instance_id,
                    "sweeps": int(sweeps),
                    "energy": pd.Series(energies, dtype=float),
                    "resource": 1,
                }
            )
            frames.append(frame)

    if not frames:
        raise ValueError("at least one instance_id is required")
    return pd.concat(frames, ignore_index=True)


def run_reliability_analysis(
    benchmark,
    runs: pd.DataFrame,
    *,
    quality_target: float = DEFAULT_QUALITY_TARGET,
    relative_error_threshold: float = DEFAULT_RELATIVE_ERROR_THRESHOLD,
    target_confidence: float = DEFAULT_TARGET_CONFIDENCE,
) -> pd.DataFrame:
    """Run the tutorial reliability report through stochastic_benchmark."""

    report = benchmark.run_RepeatReliability(
        runs,
        group_cols=["instance", "sweeps"],
        response_col="energy",
        success_rule="min",
        threshold="quality_threshold",
        iterations="sweeps",
        effort_per_iteration=1.0,
        comparison_cols="instance",
        relative_error_threshold=relative_error_threshold,
        target_confidence=target_confidence,
    )

    report = _add_tutorial_context(report, runs, quality_target)
    benchmark.repeat_reliability = report
    return report


def select_reliability_diagnostics(
    report: pd.DataFrame,
    rows_per_instance: int = 4,
) -> pd.DataFrame:
    """Select compact diagnostics, prioritizing unresolved or under-sampled rows."""

    if rows_per_instance <= 0:
        raise ValueError("rows_per_instance must be positive")
    diagnostic_columns = _diagnostic_columns_for_report(report)
    if report.empty:
        return report.reindex(columns=diagnostic_columns)

    work = report.copy()
    work["_needs_more_trials"] = work["reliability_status"].ne("reliable")
    work["_unresolved"] = work["statistically_unresolved"].astype(bool)
    work["_additional_repeats_sort"] = pd.to_numeric(
        work["additional_repeats_required"],
        errors="coerce",
    ).fillna(0)
    selected = (
        work.sort_values(
            [
                "instance",
                "_needs_more_trials",
                "_unresolved",
                "_additional_repeats_sort",
                "current_resource",
                "sweeps",
            ],
            ascending=[True, False, False, False, True, True],
        )
        .groupby("instance", group_keys=False)
        .head(rows_per_instance)
    )
    return selected[diagnostic_columns]


def _baseline_frame(
    baselines: Mapping[int, Mapping[str, float]] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(baselines, pd.DataFrame):
        baseline_frame = baselines.copy()
    else:
        baseline_frame = pd.DataFrame.from_records(
            [
                {
                    "instance": instance,
                    "best_value": values["best_value"],
                    "random_value": values["random_value"],
                }
                for instance, values in baselines.items()
            ]
        )

    required = {"instance", "best_value", "random_value"}
    missing = required - set(baseline_frame.columns)
    if missing:
        raise ValueError(f"baselines missing required columns: {sorted(missing)}")
    return baseline_frame.loc[:, ["instance", "best_value", "random_value"]]


def _diagnostic_columns_for_report(report: pd.DataFrame) -> list[str]:
    columns = [col for col in DIAGNOSTIC_COLUMNS if col in report.columns]
    seen = set(columns)
    for column in report.columns:
        if not _is_target_repeat_column(column):
            continue
        for repeat_column in [column, f"{column}_ci_lower", f"{column}_ci_upper"]:
            if repeat_column in report.columns and repeat_column not in seen:
                columns.append(repeat_column)
                seen.add(repeat_column)
    return columns


def _is_target_repeat_column(column: str) -> bool:
    return column.startswith("R") and column[1:].isdigit()


def _add_tutorial_context(
    report: pd.DataFrame,
    runs: pd.DataFrame,
    quality_target: float,
) -> pd.DataFrame:
    thresholds = runs[
        [
            "instance",
            "best_value",
            "random_value",
            "quality_target",
            "quality_threshold",
        ]
    ].drop_duplicates("instance")
    enriched = report.merge(thresholds, on="instance", how="left")
    enriched["quality_target"] = quality_target
    enriched["current_repeats"] = enriched["trials"]
    enriched["required_repeats"] = enriched["required_trials"]
    enriched["additional_repeats_required"] = enriched["additional_trials_required"]
    enriched["current_resource"] = enriched["sweeps"] * enriched["current_repeats"]
    enriched["required_resource"] = enriched["sweeps"] * enriched["required_repeats"]
    return enriched
