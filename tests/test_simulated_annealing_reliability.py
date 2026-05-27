import importlib.util
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

import stochastic_benchmark as sb_module


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "Simulated_Annealing"
    / "reliability_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("sa_reliability_analysis", MODULE_PATH)
sa_reliability = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sa_reliability
SPEC.loader.exec_module(sa_reliability)


def _benchmark(tmp_path):
    return sb_module.stochastic_benchmark(
        parameter_names=["sweeps"],
        here=str(tmp_path),
        instance_cols=["instance"],
        response_key="energy",
        response_dir=-1,
        recover=False,
        smooth=False,
    )


def _runs():
    return pd.DataFrame(
        {
            "instance": [0] * 12 + [0] * 12 + [1] * 12 + [1] * 12,
            "sweeps": [1] * 12 + [10] * 12 + [1] * 12 + [10] * 12,
            "energy": (
                [1.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
                + [1.0] * 10
                + [8.0] * 2
                + [1.5, 1.5, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
                + [1.5] * 9
                + [8.0] * 3
            ),
        }
    )


def test_quality_threshold_uses_tutorial_best_random_convention():
    threshold = sa_reliability.quality_threshold_from_baseline(
        best_value=-10.0,
        random_value=0.0,
        quality_target=0.8,
    )

    assert threshold == pytest.approx(-8.0)

    with pytest.raises(ValueError, match="quality_target"):
        sa_reliability.quality_threshold_from_baseline(-10.0, 0.0, quality_target=1.1)


def test_load_selected_raw_runs_uses_tracked_aggregate_pickle(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    all_raw_runs = {
        0: {1: [1.0, 2.0], 5: [3.0]},
        2: {1: [4.0]},
    }
    with (results_dir / "all_raw_runs.pkl").open("wb") as raw_runs_file:
        pickle.dump(all_raw_runs, raw_runs_file)

    runs = sa_reliability.load_selected_raw_runs(tmp_path, instance_ids=[0, 2])

    assert runs.columns.tolist() == ["instance", "sweeps", "energy", "resource"]
    assert runs["instance"].tolist() == [0, 0, 0, 2]
    assert runs["sweeps"].tolist() == [1, 1, 5, 1]
    assert runs["energy"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert runs["resource"].tolist() == [1, 1, 1, 1]


def test_sa_reliability_report_surfaces_repeats_and_unreliable_cases(tmp_path):
    baselines = {
        0: {"best_value": 0.0, "random_value": 10.0},
        1: {"best_value": 0.0, "random_value": 10.0},
    }
    runs = sa_reliability.attach_quality_thresholds(
        _runs(),
        baselines,
        quality_target=0.8,
    )
    benchmark = _benchmark(tmp_path)

    report = sa_reliability.run_reliability_analysis(
        benchmark,
        runs,
        quality_target=0.8,
        relative_error_threshold=0.25,
    )
    diagnostics = sa_reliability.select_reliability_diagnostics(
        report,
        rows_per_instance=2,
    )

    assert benchmark.repeat_reliability is report
    assert set(report["instance"]) == {0, 1}
    assert set(report["sweeps"]) == {1, 10}
    assert (report["current_repeats"] == 12).all()
    assert (report["current_resource"] == report["sweeps"] * report["current_repeats"]).all()
    assert report["quality_threshold"].tolist() == pytest.approx([2.0] * len(report))
    assert {"required_repeats", "p_ci_lower", "p_ci_upper", "R99_ci_lower", "R99_ci_upper"} <= set(report.columns)
    assert report["reliability_status"].eq("needs_more_trials").any()
    assert not diagnostics.empty
    assert set(diagnostics["instance"]) == {0, 1}
    assert "current_repeats" in diagnostics.columns


@pytest.mark.parametrize(
    ("target_confidence", "repeat_column"),
    [(0.95, "R95"), (0.975, "R97_5")],
)
def test_sa_reliability_diagnostics_keep_custom_target_repeat_columns(
    tmp_path,
    target_confidence,
    repeat_column,
):
    baselines = {
        0: {"best_value": 0.0, "random_value": 10.0},
        1: {"best_value": 0.0, "random_value": 10.0},
    }
    runs = sa_reliability.attach_quality_thresholds(
        _runs(),
        baselines,
        quality_target=0.8,
    )

    report = sa_reliability.run_reliability_analysis(
        _benchmark(tmp_path),
        runs,
        quality_target=0.8,
        target_confidence=target_confidence,
    )
    diagnostics = sa_reliability.select_reliability_diagnostics(report)

    assert {"R_c", "R_c_ci_lower", "R_c_ci_upper"} <= set(diagnostics.columns)
    repeat_columns = {
        repeat_column,
        f"{repeat_column}_ci_lower",
        f"{repeat_column}_ci_upper",
    }
    assert repeat_columns <= set(diagnostics.columns)
    assert "R99" not in diagnostics.columns
    assert diagnostics[repeat_column].tolist() == pytest.approx(
        diagnostics["R_c"].tolist(),
    )


def test_attach_quality_thresholds_requires_baselines_for_all_instances():
    with pytest.raises(ValueError, match="missing baselines"):
        sa_reliability.attach_quality_thresholds(
            pd.DataFrame({"instance": [0, 1], "sweeps": [1, 1], "energy": [1.0, 1.0]}),
            {0: {"best_value": 0.0, "random_value": 10.0}},
        )
