import json
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TESTS_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(TESTS_DIR, os.pardir, "src"))
sys.path.insert(0, SRC_PATH)

import plotting
import stochastic_benchmark
from plotting import Plotting


class FakeBaseline:
    name = "VirtualBest"

    def evaluate(self):
        params_df = pd.DataFrame(
            {"alpha": [0.1, 0.2]},
            index=pd.Index([10, 20], name="resource"),
        )
        eval_df = pd.DataFrame(
            {
                "resource": [10, 20],
                "response": [0.5, 0.7],
                "response_lower": [0.4, 0.6],
                "response_upper": [0.6, 0.8],
                "count": [2, 2],
            }
        )
        return params_df, eval_df


class FakeProjectionExperiment:
    name = "Projection from TrainingStats"

    def evaluate(self):
        params_df = pd.DataFrame({"resource": [10, 20], "alpha": [0.15, 0.25]})
        eval_df = pd.DataFrame(
            {
                "resource": [10, 20],
                "response": [0.45, 0.65],
                "response_lower": [0.35, 0.55],
                "response_upper": [0.55, 0.75],
            }
        )
        preproc_params = pd.DataFrame(
            {"resource": [10, 20], "alpha": [0.12, 0.22]}
        )
        return params_df, eval_df, preproc_params

    def evaluate_monotone(self):
        params_df = pd.DataFrame({"resource": [10, 20], "alpha": [0.17, 0.27]})
        eval_df = pd.DataFrame(
            {
                "resource": [10, 20],
                "response": [0.85, 0.9],
                "response_lower": [0.8, 0.85],
                "response_upper": [0.9, 0.95],
            }
        )
        preproc_params = pd.DataFrame(
            {"resource": [10, 20], "alpha": [0.14, 0.24]}
        )
        return params_df, eval_df, preproc_params


class FakeMetaExperiment:
    name = "RandomSearch"
    meta_parameter_names = ["ExploreFrac", "tau"]
    resource = "TotalBudget"
    meta_params = pd.DataFrame(
        {
            "TotalBudget": [20, 10],
            "ExplorationBudget": [4, 1],
            "ExploreFrac": [0.2, 0.1],
            "tau": [5, 2],
        }
    )
    preproc_meta_params = pd.DataFrame(
        {
            "TotalBudget": [10, 20],
            "ExplorationBudget": [2, 6],
            "ExploreFrac": [0.2, 0.3],
            "tau": [3, 6],
        }
    )

    def evaluate(self):
        params_df = pd.DataFrame({"resource": [10, 20], "alpha": [0.3, 0.4]})
        eval_df = pd.DataFrame(
            {
                "resource": [10, 20],
                "response": [0.4, 0.6],
                "response_lower": [0.3, 0.5],
                "response_upper": [0.5, 0.7],
            }
        )
        return params_df, eval_df


def _benchmark_with_fake_results(tmp_path):
    bench = stochastic_benchmark.stochastic_benchmark(
        ["alpha"],
        here=tmp_path,
        response_key="PerfRatio",
        recover=False,
    )
    bench.baseline = FakeBaseline()
    bench.experiments = [FakeProjectionExperiment(), FakeMetaExperiment()]
    return bench


def test_export_plot_csvs_writes_resource_columns_and_manifest(tmp_path):
    bench = _benchmark_with_fake_results(tmp_path)

    manifest = bench.export_plot_csvs()

    checkpoints = bench.here.checkpoints
    baseline_params = pd.read_csv(
        os.path.join(checkpoints, "params_plotting", "baseline.csv")
    )
    random_meta = pd.read_csv(
        os.path.join(checkpoints, "meta_params_plotting", "RandomSearch.csv")
    )

    assert baseline_params.columns.tolist() == ["resource", "alpha"]
    assert "Unnamed: 0" not in baseline_params.columns
    assert random_meta.columns.tolist() == ["TotalBudget", "ExploreFrac", "tau"]
    assert random_meta["TotalBudget"].tolist() == [10, 20]
    assert manifest["baseline_name"] == "VirtualBest"
    assert manifest["parameter_names"] == ["alpha"]
    assert manifest["response_key"] == "PerfRatio"
    assert [exp["name"] for exp in manifest["experiments"]] == [
        "Projection from TrainingStats",
        "RandomSearch",
    ]

    manifest_path = os.path.join(checkpoints, "plotting_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        assert json.load(fh) == manifest


def test_init_plotting_exports_csvs_and_reads_from_checkpoint_dir(tmp_path):
    bench = _benchmark_with_fake_results(tmp_path)

    bench.initPlotting()

    assert isinstance(bench.plots, Plotting)
    assert bench.plots.parameter_names == ["alpha"]
    assert bench.plots.response_key == "PerfRatio"
    assert bench.plots.experiment_names == [
        "Projection from TrainingStats",
        "RandomSearch",
    ]

    fig, ax = bench.plots.plot_performance()
    assert ax.get_ylabel() == "PerfRatio"
    assert os.path.exists(
        os.path.join(
            bench.here.checkpoints,
            "params_plotting",
            "Projection from TrainingStats_preproc.csv",
        )
    )

    meta_figs, meta_axes = bench.plots.plot_meta_parameters()
    assert set(meta_axes["RandomSearch"]) == {"ExploreFrac", "tau"}

    together_fig, together_axes = bench.plots.plot_parameters_together()
    assert set(together_axes) == {"alpha"}

    separate_figs, separate_axes = bench.plots.plot_parameters_separate()
    assert set(separate_axes) == {"alpha"}

    distance_fig, distance_ax = bench.plots.plot_parameters_distance()
    assert distance_ax.get_ylabel() == "distance_scaled"

    plt.close(fig)
    for fig in meta_figs["RandomSearch"].values():
        plt.close(fig)
    plt.close(together_fig)
    for fig in separate_figs.values():
        plt.close(fig)
    plt.close(distance_fig)


def test_direct_plotting_constructor_exports_from_benchmark_object(tmp_path):
    bench = _benchmark_with_fake_results(tmp_path)

    plots = Plotting(bench)

    assert isinstance(plots, Plotting)
    assert plots.parameter_names == ["alpha"]
    assert os.path.exists(
        os.path.join(bench.here.checkpoints, "performance_plotting", "baseline.csv")
    )

    fig, ax = plots.plot_performance()
    assert ax.get_ylabel() == "PerfRatio"
    plt.close(fig)


def test_init_plotting_uses_legacy_monotone_switch_when_unspecified(tmp_path):
    bench = _benchmark_with_fake_results(tmp_path)
    previous_monotone = plotting.monotone

    try:
        plotting.monotone = True
        bench.initPlotting()
    finally:
        plotting.monotone = previous_monotone

    projection_perf = pd.read_csv(
        os.path.join(
            bench.here.checkpoints,
            "performance_plotting",
            "Projection from TrainingStats.csv",
        )
    )
    projection_params = pd.read_csv(
        os.path.join(
            bench.here.checkpoints,
            "params_plotting",
            "Projection from TrainingStats.csv",
        )
    )

    assert projection_perf["response"].tolist() == [0.85, 0.9]
    assert projection_params["alpha"].tolist() == [0.17, 0.27]
