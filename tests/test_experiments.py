"""Tests for experiments module — covers Copilot review fixes."""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import logging
import warnings
from unittest.mock import MagicMock, patch

import matplotlib.cm as mpl_cm
import matplotlib
if not hasattr(mpl_cm, 'register_cmap'):
    def register_cmap(name, cmap, **kwargs):
        matplotlib.colormaps.register(cmap, name=name)
    mpl_cm.register_cmap = register_cmap

# Add src directory to path
TESTS_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(TESTS_DIR, os.pardir, 'src'))
sys.path.insert(0, SRC_PATH)

import experiments
import names
import stats


def _make_experiment_params(
    checkpoint_path="/tmp/test_checkpoint",
    interp_results=None,
    training_stats=None,
    testing_stats=None,
    evaluate_without_bootstrap=None,
    baseline_recalibrate=None,
):
    """Create a minimal ExperimentParameters for testing."""
    return experiments.ExperimentParameters(
        parameter_names=["param1"],
        instance_cols=["instance"],
        interp_results=pd.DataFrame() if interp_results is None else interp_results,
        checkpoint_path=str(checkpoint_path),
        response_key="response",
        response_dir=1,
        smooth=False,
        stat_params=stats.StatsParameters(
            metrics=["response"],
            stats_measures=[stats.Mean()],
            lower_bounds={},
            upper_bounds={},
        ),
        training_stats=pd.DataFrame() if training_stats is None else training_stats,
        testing_stats=pd.DataFrame() if testing_stats is None else testing_stats,
        evaluate_without_bootstrap=(
            (lambda df, group_on: df)
            if evaluate_without_bootstrap is None
            else evaluate_without_bootstrap
        ),
        baseline_recalibrate=(
            (lambda df: None) if baseline_recalibrate is None else baseline_recalibrate
        ),
    )


def _make_virtual_best_baseline(response_key="response"):
    params = _make_experiment_params()
    params = experiments.ExperimentParameters(
        parameter_names=params.parameter_names,
        instance_cols=params.instance_cols,
        interp_results=params.interp_results,
        checkpoint_path=params.checkpoint_path,
        response_key=response_key,
        response_dir=params.response_dir,
        smooth=params.smooth,
        stat_params=params.stat_params,
        training_stats=params.training_stats,
        testing_stats=params.testing_stats,
        evaluate_without_bootstrap=params.evaluate_without_bootstrap,
        baseline_recalibrate=params.baseline_recalibrate,
    )
    baseline = experiments.VirtualBestBaseline.__new__(experiments.VirtualBestBaseline)
    baseline.parent_params = params
    return baseline


class TestExperimentBaseClass:
    """Tests for the Experiment base class."""

    def test_evaluate_raises_not_implemented_with_correct_spelling(self):
        """Evaluate should raise NotImplementedError with 'overridden' (not 'overriden')."""
        exp = experiments.Experiment()
        exp.parent_params = _make_experiment_params()

        with pytest.raises(NotImplementedError, match="overridden"):
            exp.evaluate()


class TestVirtualBestBaseline:
    """Tests for VirtualBestBaseline evaluation."""

    def test_evaluate_keeps_singleton_resource_levels(self):
        baseline = _make_virtual_best_baseline()
        base = names.param2filename({"Key": "response"}, "")
        lower = names.param2filename({"Key": "response", "ConfInt": "lower"}, "")
        upper = names.param2filename({"Key": "response", "ConfInt": "upper"}, "")
        baseline.rec_params = pd.DataFrame(
            {
                "resource": [1, 2, 2],
                "param1": [0.1, 0.2, 0.4],
                base: [0.2, 0.3, 0.7],
                lower: [-0.8, 0.2, 0.6],
                upper: [1.2, 0.4, 0.8],
            }
        )

        _, eval_df = baseline.evaluate()
        by_resource = eval_df.set_index("resource")

        assert set(by_resource.index) == {1, 2}
        assert by_resource.loc[1, "count"] == 1
        assert by_resource.loc[1, "response"] == pytest.approx(0.2)
        assert by_resource.loc[1, "response_lower"] == pytest.approx(-0.8)
        assert by_resource.loc[1, "response_upper"] == pytest.approx(1.2)

    def test_evaluate_uses_response_key_bounds_when_available(self):
        baseline = _make_virtual_best_baseline(response_key="PerfRatio")
        base = names.param2filename({"Key": "PerfRatio"}, "")
        lower = names.param2filename({"Key": "PerfRatio", "ConfInt": "lower"}, "")
        upper = names.param2filename({"Key": "PerfRatio", "ConfInt": "upper"}, "")
        baseline.rec_params = pd.DataFrame(
            {
                "resource": [1, 1],
                "param1": [0.1, 0.2],
                base: [0.95, 1.05],
                lower: [0.7, 0.9],
                upper: [1.2, 1.3],
            }
        )

        _, eval_df = baseline.evaluate()

        assert eval_df.loc[0, "response_upper"] == pytest.approx(1.0)


class TestStaticRecommendationExperiment:
    """Tests for StaticRecommendationExperiment preproc_rec_params initialization."""

    def test_init_from_dataframe_sets_preproc_rec_params(self):
        """When init_from is a DataFrame, preproc_rec_params should be set."""
        params = _make_experiment_params()
        df = pd.DataFrame({
            "resource": [1, 2, 3],
            "param1": [0.1, 0.2, 0.3],
        })

        exp = experiments.StaticRecommendationExperiment(params, df)

        assert hasattr(exp, "preproc_rec_params")
        assert isinstance(exp.preproc_rec_params, pd.DataFrame)
        pd.testing.assert_frame_equal(exp.preproc_rec_params, df)

    def test_init_from_projection_without_postprocess_sets_preproc_rec_params(self):
        """When init_from is a ProjectionExperiment without postprocess,
        preproc_rec_params should default to recipe.copy()."""
        params = _make_experiment_params()

        # Plain object with __class__ override to pass `type() == ProjectionExperiment`
        class _Stub:
            pass

        proj = _Stub()
        proj.__class__ = experiments.ProjectionExperiment
        proj.postprocess = None
        proj.recipe = pd.DataFrame({
            "resource": [1, 2],
            "param1": [0.5, 0.6],
        })

        exp = experiments.StaticRecommendationExperiment(params, proj)

        assert hasattr(exp, "preproc_rec_params")
        pd.testing.assert_frame_equal(exp.preproc_rec_params, proj.recipe)

    def test_init_from_projection_with_postprocess_uses_preproc_recipe(self):
        """When init_from is ProjectionExperiment with postprocess,
        preproc_rec_params should come from preproc_recipe."""
        params = _make_experiment_params()

        class _Stub:
            pass

        proj = _Stub()
        proj.__class__ = experiments.ProjectionExperiment
        proj.postprocess = lambda df: df
        proj.recipe = pd.DataFrame({
            "resource": [1, 2],
            "param1": [0.5, 0.6],
        })
        proj.preproc_recipe = pd.DataFrame({
            "resource": [1, 2],
            "param1": [0.3, 0.4],
        })

        exp = experiments.StaticRecommendationExperiment(params, proj)

        assert hasattr(exp, "preproc_rec_params")
        pd.testing.assert_frame_equal(exp.preproc_rec_params, proj.preproc_recipe)

    def test_init_from_unsupported_type_warns(self):
        """When init_from is an unsupported type, a warning should be raised."""
        params = _make_experiment_params()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exp = experiments.StaticRecommendationExperiment(params, 42)
            assert len(w) == 1
            assert "not supported" in str(w[0].message)

    def test_list_runs_attach_runs_and_evaluate_processed_dataframe(self, tmp_path):
        base = names.param2filename({"Key": "response"}, "")
        lower = names.param2filename({"Key": "response", "ConfInt": "lower"}, "")
        upper = names.param2filename({"Key": "response", "ConfInt": "upper"}, "")
        calls = {}

        def evaluate_without_bootstrap(df, group_on):
            calls["group_on"] = group_on
            calls["raw_df"] = df.copy()
            return pd.DataFrame({
                "resource": [1, 1, 2],
                base: [0.4, 0.6, 0.8],
                lower: [0.3, 0.5, 0.7],
                upper: [0.5, 0.7, 0.9],
            })

        recalibrated = []
        params = _make_experiment_params(
            checkpoint_path=tmp_path,
            evaluate_without_bootstrap=evaluate_without_bootstrap,
            baseline_recalibrate=lambda df: recalibrated.append(df.copy()),
        )
        rec_params = pd.DataFrame({
            "resource": [1, 2],
            "param1": [0.1, 0.2],
        })
        exp = experiments.StaticRecommendationExperiment(params, rec_params)

        runs = exp.list_runs()
        assert [(run.resource, run.param1) for run in runs] == [(1.0, 0.1), (2.0, 0.2)]

        raw_runs = pd.DataFrame({
            "instance": [1],
            "resource": [1],
            base: [0.4],
        })
        raw_path = tmp_path / "raw_runs.pkl"
        raw_runs.to_pickle(raw_path)
        exp.attach_runs(str(raw_path), process=True)

        params_df, eval_df, preproc_params = exp.evaluate()

        assert calls["group_on"] == ["instance", "resource"]
        assert len(recalibrated) == 1
        pd.testing.assert_frame_equal(params_df, rec_params)
        pd.testing.assert_frame_equal(preproc_params, rec_params)
        assert eval_df.loc[eval_df["resource"] == 1, "response"].iloc[0] == pytest.approx(0.5)


class TestProjectionExperiment:
    def test_training_stats_projection_populates_recipe_and_evaluates(self, tmp_path, monkeypatch):
        base = names.param2filename({"Key": "response"}, "")
        lower = names.param2filename({"Key": "response", "ConfInt": "lower"}, "")
        upper = names.param2filename({"Key": "response", "ConfInt": "upper"}, "")
        interp_results = pd.DataFrame({
            "train": [1, 0, 0],
            "instance": [1, 1, 2],
            "resource": [1, 1, 1],
            "param1": [0.1, 0.2, 0.4],
        })
        training_stats = pd.DataFrame({"resource": [1], "param1": [0.1]})
        params = _make_experiment_params(
            checkpoint_path=tmp_path,
            interp_results=interp_results,
            training_stats=training_stats,
        )

        def fake_best_parameters(df, parameter_names, response_col, response_dir, resource_col, additional_cols, smooth):
            assert df is not training_stats
            assert parameter_names == ["param1"]
            assert additional_cols == ["boots"]
            return pd.DataFrame({"resource": [1, 2], "param1": [0.2, 0.3], "boots": [1, 1]})

        def fake_evaluate(testing_results, recipe, distance, parameter_names, group_on):
            assert testing_results["train"].eq(0).all()
            pd.testing.assert_frame_equal(recipe, pd.DataFrame({
                "resource": [1, 2],
                "param1": [0.2, 0.3],
                "boots": [1, 1],
            }))
            assert parameter_names == ["param1"]
            assert group_on == ["instance"]
            return pd.DataFrame({
                "resource": [1, 1, 2],
                "param1": [0.2, 0.4, 0.3],
                base: [0.4, 0.6, 0.8],
                lower: [0.3, 0.5, 0.7],
                upper: [0.5, 0.7, 0.9],
            })

        monkeypatch.setattr(experiments.training, "best_parameters", fake_best_parameters)
        monkeypatch.setattr(experiments.training, "evaluate", fake_evaluate)

        exp = experiments.ProjectionExperiment(params, "TrainingStats")
        params_df, eval_df = exp.evaluate()

        assert (tmp_path / "BestRecommended_train.pkl").exists()
        assert (tmp_path / "Projection_from=TrainingStats.pkl").exists()
        assert params_df.loc[params_df["resource"] == 1, "param1"].iloc[0] == pytest.approx(0.3)
        assert eval_df.loc[eval_df["resource"] == 1, "response"].iloc[0] == pytest.approx(0.5)


class TestRandomSearchExperiment:
    def test_populate_computes_caches_and_evaluates(self, tmp_path, monkeypatch):
        metric_base = names.param2filename({"Key": "response", "Metric": "mean"}, "")
        metric_lower = names.param2filename(
            {"Key": "response", "Metric": "mean", "ConfInt": "lower"}, ""
        )
        metric_upper = names.param2filename(
            {"Key": "response", "Metric": "mean", "ConfInt": "upper"}, ""
        )
        params = _make_experiment_params(
            checkpoint_path=tmp_path,
            training_stats=pd.DataFrame({"resource": [1]}),
            testing_stats=pd.DataFrame({"resource": [1]}),
        )
        meta = pd.DataFrame({
            "TotalBudget": [10, 20],
            "ExplorationBudget": [2, 10],
            "tau": [0.1, 0.2],
        })
        eval_train = pd.DataFrame({"TotalBudget": [10]})
        eval_test = pd.DataFrame({
            "TotalBudget": [10, 10, 20],
            "resource": [1, 2, 1],
            "param1": [0.2, 0.4, 0.8],
            metric_base: [0.3, 0.5, 0.7],
            metric_lower: [0.2, 0.4, 0.6],
            metric_upper: [0.4, 0.6, 0.8],
        })

        monkeypatch.setattr(
            experiments.random_exploration,
            "RandomExploration",
            lambda training_stats, rs_params: (meta.copy(), eval_train.copy(), None),
        )
        monkeypatch.setattr(
            experiments.random_exploration,
            "apply_allocations",
            lambda testing_stats, rs_params, meta_params: eval_test.copy(),
        )

        exp = experiments.RandomSearchExperiment(params, rsParams={"tau": [0.1]})
        params_df, eval_df = exp.evaluate()

        assert (tmp_path / "RandomSearch_meta_params.pkl").exists()
        assert (tmp_path / "RandomSearch_evalTrain.pkl").exists()
        assert (tmp_path / "RandomSearch_evalTest.pkl").exists()
        assert exp.meta_params["ExploreFrac"].tolist() == [0.2, 0.5]
        assert params_df.loc[params_df["resource"] == 10, "param1"].iloc[0] == pytest.approx(0.3)
        assert eval_df.loc[eval_df["resource"] == 10, "response"].iloc[0] == pytest.approx(0.4)


class TestSequentialSearchExperiment:
    def test_id_postprocess_populate_and_evaluate(self, tmp_path, monkeypatch):
        base = names.param2filename({"Key": "response"}, "")
        lower = names.param2filename({"Key": "response", "ConfInt": "lower"}, "")
        upper = names.param2filename({"Key": "response", "ConfInt": "upper"}, "")
        interp_results = pd.DataFrame({
            "train": [1, 0],
            "instance": [1, 2],
            "resource": [1, 1],
        })
        params = _make_experiment_params(
            checkpoint_path=tmp_path,
            interp_results=interp_results,
        )
        meta = pd.DataFrame({
            "TotalBudget": [10],
            "ExplorationBudget": [5],
            "tau": [0.1],
        })
        eval_train = pd.DataFrame({"TotalBudget": [10]})
        eval_test = pd.DataFrame({
            "TotalBudget": [10],
            "resource": [1],
            "param1": [0.4],
            base: [0.7],
            lower: [0.6],
            upper: [0.8],
        })

        monkeypatch.setattr(
            experiments.sequential_exploration,
            "SequentialExploration",
            lambda training_results, ss_params, group_on: (meta.copy(), eval_train.copy(), None),
        )

        def fake_apply_allocations(testing_results, ss_params, meta_params, group_on):
            assert testing_results["train"].eq(0).all()
            assert meta_params["tau"].tolist() == [1.1]
            assert group_on == ["instance"]
            return eval_test.copy()

        monkeypatch.setattr(
            experiments.sequential_exploration,
            "apply_allocations",
            fake_apply_allocations,
        )

        exp = experiments.SequentialSearchExperiment(
            params,
            ssParams={"tau": [0.1]},
            id_name="trial",
            postprocess=lambda df: df.assign(tau=df["tau"] + 1),
            postprocess_name="shift",
        )
        params_df, eval_df = exp.evaluate()

        assert exp.name == "SequentialSearch_trial"
        assert (tmp_path / "SequentialSearch_meta_params_id=trial.pkl").exists()
        assert (tmp_path / "SequentialSearch_evalTrain_id=trial.pkl").exists()
        assert (tmp_path / "SequentialSearch_evalTest_id=trial_postprocess=shift.pkl").exists()
        assert params_df.loc[0, "param1"] == pytest.approx(0.4)
        assert eval_df.loc[0, "response_upper"] == pytest.approx(0.8)


class TestStochasticBenchmarkRuntimeErrors:
    """Tests for RuntimeError (not assert) on None interp_results."""

    @patch("stochastic_benchmark.interpolate.Interpolate", return_value=None)
    def test_run_Interpolate_raises_runtime_error_on_none(self, mock_interp):
        """run_Interpolate should raise RuntimeError (not assert) if Interpolate returns None."""
        import stochastic_benchmark as sb_module
        import interpolate
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            sb = sb_module.stochastic_benchmark(
                here=temp_dir,
                response_key="test_response",
                response_dir="max",
                parameter_names=["param1"],
                instance_cols=["instance"],
                reduce_mem=False,
            )
            # Give it a non-None bs_results so it doesn't short-circuit
            sb.bs_results = pd.DataFrame({
                "param1": [0.1, 0.2],
                "instance": [1, 2],
                "resource": [10, 20],
            })

            iParams = interpolate.InterpolationParameters(
                resource_fcn=lambda df: df["resource"],
            )

            with pytest.raises(RuntimeError, match="Interpolation failed"):
                sb.run_Interpolate(iParams)

    def test_run_baseline_does_not_require_existing_baseline(self):
        """run_baseline should construct the first baseline without a placeholder."""
        import stochastic_benchmark as sb_module
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            sb = sb_module.stochastic_benchmark(
                here=temp_dir,
                response_key="PerfRatio",
                response_dir=1,
                parameter_names=["iteration", "samples"],
                instance_cols=["instance"],
                reduce_mem=False,
                smooth=False,
            )
            sb.interp_results = pd.DataFrame(
                {
                    "instance": [1, 1, 2, 2],
                    "train": [0, 0, 0, 0],
                    "resource": [1, 2, 1, 2],
                    "iteration": [1, 2, 1, 2],
                    "samples": [1, 1, 1, 1],
                    "Key=PerfRatio": [0.5, 0.8, 0.6, 0.9],
                    "ConfInt=lower_Key=PerfRatio": [0.4, 0.7, 0.5, 0.8],
                    "ConfInt=upper_Key=PerfRatio": [0.6, 0.9, 0.7, 1.0],
                }
            )
            sb.training_stats = pd.DataFrame()
            sb.testing_stats = pd.DataFrame()
            sb.stat_params = stats.StatsParameters(
                metrics=["PerfRatio"],
                stats_measures=[stats.Mean()],
                lower_bounds={},
                upper_bounds={},
            )

            assert not hasattr(sb, "baseline")

            sb.run_baseline()

            assert sb.baseline.name == "VirtualBest"

    def test_experiment_parameters_recalibrate_late_bound_baseline(self):
        """Experiment parameters should recalibrate a baseline added after creation."""
        import stochastic_benchmark as sb_module
        import tempfile

        class RecordingBaseline:
            def __init__(self):
                self.calls = []

            def recalibrate(self, df):
                self.calls.append(df)

        with tempfile.TemporaryDirectory() as temp_dir:
            sb = sb_module.stochastic_benchmark(
                here=temp_dir,
                response_key="PerfRatio",
                response_dir=1,
                parameter_names=["iteration", "samples"],
                instance_cols=["instance"],
                reduce_mem=False,
                smooth=False,
            )
            sb.interp_results = pd.DataFrame()
            sb.training_stats = pd.DataFrame()
            sb.testing_stats = pd.DataFrame()
            sb.stat_params = stats.StatsParameters(
                metrics=["PerfRatio"],
                stats_measures=[stats.Mean()],
                lower_bounds={},
                upper_bounds={},
            )

            params = sb.get_experiment_parameters()
            baseline = RecordingBaseline()
            sb.baseline = baseline
            experiment = experiments.StaticRecommendationExperiment(
                params,
                pd.DataFrame({"resource": [1], "iteration": [1], "samples": [1]}),
            )
            eval_df = pd.DataFrame({"resource": [1], "response": [0.5]})

            experiment.attach_runs(eval_df, process=False)

            assert baseline.calls == [eval_df]


class TestStochasticBenchmarkUsesLogger:
    """Tests that stochastic_benchmark uses logger, not print()."""

    def test_no_print_in_stochastic_benchmark(self):
        """Source code should not contain active print() calls."""
        import inspect
        import stochastic_benchmark

        source = inspect.getsource(stochastic_benchmark.stochastic_benchmark)
        lines = source.split("\n")
        active_prints = [
            (i + 1, line.strip())
            for i, line in enumerate(lines)
            if "print(" in line and not line.strip().startswith("#")
        ]
        assert active_prints == [], (
            f"Found active print() calls in stochastic_benchmark:\n"
            + "\n".join(f"  line {n}: {l}" for n, l in active_prints)
        )


if __name__ == "__main__":
    pytest.main([__file__])
