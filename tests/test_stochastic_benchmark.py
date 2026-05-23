import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import names
import stats
import stochastic_benchmark as sb_module


def _make_benchmark(tmp_path, reduce_mem=False, recover=True, response_key="response"):
    return sb_module.stochastic_benchmark(
        parameter_names=["param1"],
        here=str(tmp_path),
        instance_cols=["instance"],
        response_key=response_key,
        response_dir=1,
        recover=recover,
        reduce_mem=reduce_mem,
        smooth=False,
    )


def _response_frame(response_key="response"):
    base = names.param2filename({"Key": response_key}, "")
    lower = names.param2filename({"Key": response_key, "ConfInt": "lower"}, "")
    upper = names.param2filename({"Key": response_key, "ConfInt": "upper"}, "")
    return pd.DataFrame({
        "instance": [1, 2],
        "param1": [0.1, 0.2],
        "boots": [1, 1],
        "resource": [1, 2],
        base: [0.4, 0.8],
        lower: [0.3, 0.7],
        upper: [0.5, 0.9],
    })


def test_set_bootstrap_accepts_paths_dataframes_and_lists(tmp_path):
    sb = _make_benchmark(tmp_path)
    first = pd.DataFrame({"value": [1]})
    second = pd.DataFrame({"value": [2]})
    pickle_path = tmp_path / "bootstrap.pkl"
    first.to_pickle(pickle_path)

    sb.set_Bootstrap(str(pickle_path))
    pd.testing.assert_frame_equal(sb.bs_results, first)

    sb.set_Bootstrap(second)
    pd.testing.assert_frame_equal(sb.bs_results, second)

    sb.set_Bootstrap([first, second])
    assert sb.bs_results["value"].tolist() == [1, 2]

    sb.set_Bootstrap([str(pickle_path)])
    assert sb.bs_results == [str(pickle_path)]


def test_run_bootstrap_non_reduce_uses_existing_raw_data_and_persists(tmp_path, monkeypatch):
    sb = _make_benchmark(tmp_path, reduce_mem=False, recover=False)
    sb.raw_data = pd.DataFrame({
        "instance": [1],
        "param1": [0.1],
        "resource": [1],
        "response": [0.4],
    })
    bootstrapped = pd.DataFrame({"boot": [1]})
    calls = {}

    def fake_bootstrap(df, group_on, bs_params_iter, progress_dir):
        calls["df"] = df
        calls["group_on"] = group_on
        calls["bs_params_iter"] = bs_params_iter
        calls["progress_dir"] = progress_dir
        return bootstrapped

    monkeypatch.setattr(sb_module.bootstrap, "Bootstrap", fake_bootstrap)
    bs_params_iter = object()

    sb.run_Bootstrap(bs_params_iter)

    pd.testing.assert_frame_equal(calls["df"], sb.raw_data)
    assert calls["group_on"] == ["param1", "instance"]
    assert calls["bs_params_iter"] is bs_params_iter
    assert os.path.isdir(calls["progress_dir"])
    pd.testing.assert_frame_equal(pd.read_pickle(sb.here.bootstrap), bootstrapped)


def test_run_bootstrap_non_reduce_errors_without_raw_or_checkpoint(tmp_path):
    sb = _make_benchmark(tmp_path, reduce_mem=False, recover=False)

    with pytest.raises(Exception, match="No raw data found"):
        sb.run_Bootstrap(bsParams_iter=[])


def test_run_interpolate_non_reduce_filters_missing_rows_and_clears_bootstrap(tmp_path, monkeypatch):
    sb = _make_benchmark(tmp_path, reduce_mem=False, response_key="response")
    original_bs_results = pd.DataFrame({"instance": [1], "param1": [0.1]})
    sb.bs_results = original_bs_results
    interpolated = _response_frame("response")
    interpolated.loc[1, names.param2filename({"Key": "response"}, "")] = None
    calls = {}

    def fake_interpolate(bs_results, i_params, group_on):
        calls["bs_results"] = bs_results
        calls["i_params"] = i_params
        calls["group_on"] = group_on
        return interpolated.copy()

    monkeypatch.setattr(sb_module.interpolate, "Interpolate", fake_interpolate)
    i_params = object()

    sb.run_Interpolate(i_params)

    pd.testing.assert_frame_equal(calls["bs_results"], original_bs_results)
    assert calls["i_params"] is i_params
    assert calls["group_on"] == ["param1", "instance"]
    assert len(sb.interp_results) == 1
    assert sb.bs_results is None
    pd.testing.assert_frame_equal(pd.read_pickle(sb.here.interpolate), sb.interp_results)


def test_run_stats_splits_data_computes_and_persists_train_test_stats(tmp_path, monkeypatch):
    sb = _make_benchmark(tmp_path, reduce_mem=False)
    sb.interp_results = _response_frame("response")
    stat_params = stats.StatsParameters(metrics=["response"], stats_measures=[stats.Mean()])
    stats_calls = []

    def fake_split_train_test(df, instance_cols, train_test_split):
        assert instance_cols == ["instance"]
        assert train_test_split == 0.5
        return df.assign(train=[1, 0])

    def fake_stats(df, params, group_on):
        stats_calls.append((df.copy(), params, group_on))
        return pd.DataFrame({"rows": [len(df)], "train_value": [int(df["train"].iloc[0])]})

    monkeypatch.setattr(sb_module.training, "split_train_test", fake_split_train_test)
    monkeypatch.setattr(sb_module.stats, "Stats", fake_stats)

    sb.run_Stats(stat_params, train_test_split=0.5)

    assert len(stats_calls) == 2
    assert stats_calls[0][2] == ["param1", "boots", "resource"]
    assert sb.training_stats["train_value"].tolist() == [1]
    assert sb.testing_stats["train_value"].tolist() == [0]
    assert os.path.exists(sb.here.training_stats)
    assert os.path.exists(sb.here.testing_stats)
    assert pd.read_pickle(sb.here.interpolate)["train"].tolist() == [1, 0]


def test_populate_interp_results_recovers_cached_pickle_and_adds_split(tmp_path, monkeypatch):
    sb = _make_benchmark(tmp_path, reduce_mem=False)
    cached = _response_frame("response")
    cached.to_pickle(sb.here.interpolate)

    def fake_split_train_test(df, instance_cols, train_test_split):
        assert instance_cols == ["instance"]
        return df.assign(train=[1, 0])

    monkeypatch.setattr(sb_module.training, "split_train_test", fake_split_train_test)
    sb.stat_params = stats.StatsParameters(metrics=["response"], stats_measures=[stats.Mean()])
    sb.train_test_split = 0.5

    sb.populate_interp_results()

    assert sb.interp_results["train"].tolist() == [1, 0]
    assert pd.read_pickle(sb.here.interpolate)["train"].tolist() == [1, 0]


def test_evaluate_without_bootstrap_replays_aggregated_rows(tmp_path):
    class FakeMetric:
        def __init__(self, shared_args, metric_args):
            self.shared_args = shared_args
            self.metric_args = metric_args

        def evaluate(self, bs_df, responses, resources):
            bs_df["metric"] = [responses.mean()]
            bs_df["resource_total"] = [resources.sum()]

    class FakeBootstrapParams:
        shared_args = {"resource_col": "resource", "response_col": "response"}
        metric_args = {"FakeMetric": {}}
        success_metrics = [FakeMetric]
        keep_cols = ["tag"]
        agg = "count"

        def update_rule(self, bs_params, df):
            self.updated_rows = len(df)

    sb = sb_module.stochastic_benchmark(parameter_names=["param1"], here=str(tmp_path))
    sb.bsParams_iter = iter([FakeBootstrapParams()])
    df = pd.DataFrame({
        "instance": [1, 1],
        "resource": [2, 3],
        "response": [10.0, 20.0],
        "count": [2, 1],
        "tag": ["kept", "kept"],
    })

    result = sb.evaluate_without_bootstrap(df, ["instance"])

    assert result.loc[0, "instance"] == 1
    assert result.loc[0, "metric"] == pytest.approx((10.0 + 10.0 + 20.0) / 3)
    assert result.loc[0, "resource_total"] == 7
    assert result.loc[0, "tag"] == "kept"


def test_experiment_runner_methods_append_expected_experiment_objects(tmp_path, monkeypatch):
    sb = _make_benchmark(tmp_path)
    sb.interp_results = pd.DataFrame({"resource": [1]})
    sb.training_stats = pd.DataFrame({"resource": [1]})
    sb.testing_stats = pd.DataFrame({"resource": [1]})
    sb.stat_params = stats.StatsParameters(metrics=["response"], stats_measures=[stats.Mean()])
    created = []

    class FakeBaseline:
        def __init__(self, parent_params):
            self.parent_params = parent_params
            self.name = "baseline"
            created.append(("baseline", parent_params))

    class FakeProjection:
        def __init__(self, parent_params, project_from, postprocess, postprocess_name):
            self.parent_params = parent_params
            self.name = "projection"
            created.append(("projection", project_from, postprocess, postprocess_name))

    class FakeRandom:
        def __init__(self, parent_params, rs_params, postprocess=None, postprocess_name=None):
            self.parent_params = parent_params
            self.name = "random"
            created.append(("random", rs_params, postprocess, postprocess_name))

    class FakeSequential:
        def __init__(self, parent_params, ss_params, id_name, postprocess=None, postprocess_name=None):
            self.parent_params = parent_params
            self.name = "sequential"
            created.append(("sequential", ss_params, id_name, postprocess, postprocess_name))

    class FakeStatic:
        def __init__(self, parent_params, init_from):
            self.parent_params = parent_params
            self.name = "static"
            created.append(("static", init_from))

    monkeypatch.setattr(sb_module, "VirtualBestBaseline", FakeBaseline)
    monkeypatch.setattr(sb_module, "ProjectionExperiment", FakeProjection)
    monkeypatch.setattr(sb_module, "RandomSearchExperiment", FakeRandom)
    monkeypatch.setattr(sb_module, "SequentialSearchExperiment", FakeSequential)
    monkeypatch.setattr(sb_module, "StaticRecommendationExperiment", FakeStatic)

    postprocess = lambda df: df
    sb.run_baseline()
    sb.run_ProjectionExperiment("TrainingStats", postprocess=postprocess, postprocess_name="pp")
    sb.run_RandomSearchExperiment({"tau": [1]}, postprocess=postprocess, postprocess_name="rp")
    sb.run_SequentialSearchExperiment({"tau": [2]}, id_name="seq", postprocess=postprocess, postprocess_name="sp")
    sb.run_StaticRecommendationExperiment("projection")

    assert sb.baseline.name == "baseline"
    assert [experiment.name for experiment in sb.experiments] == [
        "projection",
        "random",
        "sequential",
        "static",
    ]
    assert created[1] == ("projection", "TrainingStats", postprocess, "pp")
    assert created[2] == ("random", {"tau": [1]}, postprocess, "rp")
    assert created[3] == ("sequential", {"tau": [2]}, "seq", postprocess, "sp")
    assert created[4] == ("static", "projection")


def test_export_plot_csvs_requires_baseline(tmp_path):
    sb = _make_benchmark(tmp_path)

    with pytest.raises(AttributeError, match="run_baseline"):
        sb.export_plot_csvs()


def test_plotting_csv_frame_cleans_index_columns_and_validates_required_columns():
    dirty = pd.DataFrame({
        "resource": [1],
        "index": [99],
        "Unnamed: 0": [99],
        "level_0": [88],
        "value": [0.5],
        "extra": ["drop"],
    })

    cleaned = sb_module.stochastic_benchmark._plotting_csv_frame(
        dirty,
        required_columns=["resource", "value"],
        optional_columns=["missing"],
    )

    assert cleaned.columns.tolist() == ["resource", "value"]
    assert cleaned.loc[0, "resource"] == 1

    index_only = sb_module.stochastic_benchmark._plotting_csv_frame(
        pd.DataFrame({"index": [2], "value": [0.6]}),
        required_columns=["resource", "value"],
    )
    assert index_only.loc[0, "resource"] == 2

    with pytest.raises(ValueError, match="missing columns"):
        sb_module.stochastic_benchmark._plotting_csv_frame(
            pd.DataFrame({"resource": [1]}),
            required_columns=["response"],
        )
