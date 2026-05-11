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
import stats


def _make_experiment_params():
    """Create a minimal ExperimentParameters for testing."""
    return experiments.ExperimentParameters(
        parameter_names=["param1"],
        instance_cols=["instance"],
        interp_results=pd.DataFrame(),
        checkpoint_path="/tmp/test_checkpoint",
        response_key="response",
        response_dir=1,
        smooth=False,
        stat_params=stats.StatsParameters(
            metrics=[],
            stats_measures=[stats.Mean()],
            lower_bounds={},
            upper_bounds={},
        ),
        training_stats=pd.DataFrame(),
        testing_stats=pd.DataFrame(),
        evaluate_without_bootstrap=lambda df, group_on: df,
        baseline_recalibrate=lambda df: None,
    )


class TestExperimentBaseClass:
    """Tests for the Experiment base class."""

    def test_evaluate_raises_not_implemented_with_correct_spelling(self):
        """Evaluate should raise NotImplementedError with 'overridden' (not 'overriden')."""
        exp = experiments.Experiment()
        exp.parent_params = _make_experiment_params()

        with pytest.raises(NotImplementedError, match="overridden"):
            exp.evaluate()


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
