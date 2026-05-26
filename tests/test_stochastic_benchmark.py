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


def test_run_repeat_reliability_uses_existing_observations_and_default_groups(tmp_path):
    sb = _make_benchmark(tmp_path)
    df = pd.DataFrame(
        {
            "instance": [1, 1, 1],
            "param1": [0.1, 0.1, 0.1],
            "response": [0.2, 0.4, 1.5],
        }
    )

    report = sb.run_RepeatReliability(
        df,
        response_col="response",
        success_rule="min",
        threshold=0.5,
    )

    assert sb.repeat_reliability is report
    assert report.loc[0, "instance"] == 1
    assert report.loc[0, "param1"] == pytest.approx(0.1)
    assert report.loc[0, "successes"] == 2
    assert report.loc[0, "trials"] == 3


@pytest.mark.parametrize("attribute", ["raw_data", "bs_results", "interp_results"])
def test_run_repeat_reliability_uses_populated_dataframe_sources(tmp_path, attribute):
    sb = _make_benchmark(tmp_path)
    df = pd.DataFrame(
        {
            "instance": [1, 1],
            "param1": [0.1, 0.1],
            "response": [0.2, 1.5],
        }
    )
    setattr(sb, attribute, df)

    report = sb.run_RepeatReliability(
        response_col="response",
        success_rule="min",
        threshold=0.5,
    )

    assert report.loc[0, "successes"] == 1
    assert report.loc[0, "trials"] == 2


def test_run_repeat_reliability_requires_existing_dataframe_source(tmp_path):
    sb = _make_benchmark(tmp_path)

    with pytest.raises(ValueError, match="df is required"):
        sb.run_RepeatReliability(response_col="response", threshold=0.5)


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


def test_reduce_mem_bootstrap_recovers_checkpoint_pickles_without_exp_raw(
    tmp_path,
):
    """Existing bootstrap pickles should make reruns independent of exp_raw."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    checkpoint_path = (
        tmp_path / "checkpoints" / "bootstrapped_results_inst=1.pkl"
    )
    pd.DataFrame({"param": [1], "instance": [1], "resource": [1]}).to_pickle(
        checkpoint_path
    )

    sb.run_Bootstrap(bsParams_iter=object())

    assert sb.bs_results == [str(checkpoint_path)]


def test_reduce_mem_bootstrap_recovers_aggregate_checkpoint_without_exp_raw(tmp_path):
    """Aggregate bootstrap pickles can be recovered when raw files are absent."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    bootstrap_path = tmp_path / "checkpoints" / "bootstrapped_results.pkl"
    pd.DataFrame({"param": [1], "instance": [1], "resource": [1]}).to_pickle(
        bootstrap_path
    )

    sb.run_Bootstrap(bsParams_iter=object())

    assert sb.bs_results == [str(bootstrap_path)]


def test_reduce_mem_bootstrap_errors_without_raw_or_checkpoints(tmp_path):
    """Reduced-memory bootstrapping still reports missing inputs clearly."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    with pytest.raises(Exception, match="No raw data found"):
        sb.run_Bootstrap(bsParams_iter=object())


def test_reduce_mem_bootstrap_recovers_expected_checkpoint_pickles_with_exp_raw(
    tmp_path, monkeypatch
):
    """Raw inputs should only recover the matching complete checkpoint set."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    raw_dir = tmp_path / "exp_raw"
    raw_dir.mkdir()
    (raw_dir / "raw_results_inst=1.pkl").write_text("not a pickle")

    checkpoint_path = tmp_path / "checkpoints" / "bootstrapped_results_inst=1.pkl"
    pd.DataFrame({"param": [1], "instance": [1], "resource": [1]}).to_pickle(
        checkpoint_path
    )

    def fail_bootstrap_from_raw(*args, **kwargs):
        pytest.fail(
            "run_Bootstrap should not use exp_raw when checkpoint pickles exist"
        )

    monkeypatch.setattr(
        sb_module.bootstrap, "Bootstrap_reduce_mem", fail_bootstrap_from_raw
    )

    sb.run_Bootstrap(bsParams_iter=object(), group_name_fcn=lambda _: "inst=1")

    assert sb.bs_results == [str(checkpoint_path)]


def test_reduce_mem_bootstrap_requires_group_name_fcn_with_exp_raw(tmp_path):
    """Raw-file reduced-memory runs need group names for checkpoint recovery."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    raw_dir = tmp_path / "exp_raw"
    raw_dir.mkdir()
    (raw_dir / "raw_results_inst=1.pkl").write_text("not a pickle")

    with pytest.raises(Exception, match="group_name_fcn should be provided"):
        sb.run_Bootstrap(bsParams_iter=object())


def test_reduce_mem_bootstrap_does_not_recover_partial_checkpoint_set(
    tmp_path, monkeypatch
):
    """Partial checkpoints should not silently drop raw-data groups."""
    sb = sb_module.stochastic_benchmark(
        parameter_names=["param"],
        here=tmp_path,
        instance_cols=["instance"],
        reduce_mem=True,
        recover=True,
    )

    raw_dir = tmp_path / "exp_raw"
    raw_dir.mkdir()
    raw_files = [
        raw_dir / "raw_results_inst=1.pkl",
        raw_dir / "raw_results_inst=2.pkl",
    ]
    for raw_file in raw_files:
        raw_file.write_text("not a pickle")

    checkpoint_path = tmp_path / "checkpoints" / "bootstrapped_results_inst=1.pkl"
    pd.DataFrame({"param": [1], "instance": [1], "resource": [1]}).to_pickle(
        checkpoint_path
    )

    expected_results = ["rebuilt-results"]

    def bootstrap_from_raw(raw_data, *args, **kwargs):
        assert sorted(raw_data) == sorted(str(raw_file) for raw_file in raw_files)
        return expected_results

    monkeypatch.setattr(sb_module.bootstrap, "Bootstrap_reduce_mem", bootstrap_from_raw)

    def group_name(raw_filename):
        if raw_filename.endswith("inst=1.pkl"):
            return "inst=1"
        return "inst=2"

    sb.run_Bootstrap(bsParams_iter=object(), group_name_fcn=group_name)

    assert sb.bs_results == expected_results
