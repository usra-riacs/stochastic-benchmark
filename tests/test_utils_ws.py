import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils_ws import (
    gen_log_space,
    interp,
    interpolate_df,
    process_df_progress,
    take_closest,
)


class TestGenLogSpace:
    def test_basic_generation(self):
        vals = gen_log_space(1, 100, 5)
        assert len(vals) == 5
        assert vals[0] == 1
        assert vals[-1] <= 100
        assert np.all(np.diff(vals) > 0)


class TestTakeClosest:
    def test_various_cases(self):
        arr = [1, 3, 5, 7]
        assert take_closest(arr, 6) == 5
        assert take_closest(arr, 3) == 3
        assert take_closest(arr, 0) == 1
        assert take_closest(arr, 10) == 7


class TestInterp:
    def test_numeric_and_object_columns(self):
        df = pd.DataFrame({'num': [1, 2, 3], 'cat': ['x', 'y', 'z']}, index=[0, 5, 10])
        result = interp(df, [0, 2, 5, 7, 10])
        assert list(result.index) == [0, 2, 5, 7, 10]
        np.testing.assert_almost_equal(result['num'].tolist(), [1, 1.4, 2, 2.4, 3])
        assert result.loc[0, 'cat'] == 'x'
        assert result.loc[5, 'cat'] == 'y'
        assert pd.isna(result.loc[2, 'cat']) and pd.isna(result.loc[7, 'cat'])


class TestInterpolateDf:
    def test_none_and_empty_inputs_return_none(self, tmp_path):
        assert interpolate_df(dataframe=None, parameters_dict={}, results_path=tmp_path) is None
        assert interpolate_df(
            dataframe=pd.DataFrame(),
            parameters_dict={},
            results_path=tmp_path,
        ) is None

    def test_derived_resource_multiple_instances_and_partial_recovery(self, tmp_path):
        df = pd.DataFrame({
            "instance": [1, 1, 2, 2],
            "sweep": [2, 2, 2, 2],
            "replica": [1, 1, 1, 1],
            "boots": [1, 2, 1, 2],
            "score": [10.0, 20.0, 30.0, 50.0],
        })

        result = interpolate_df(
            dataframe=df,
            resource_column="reads",
            prefix="run.pkl",
            parameters_dict={"sweep": [2], "replica": [1]},
            results_path=tmp_path,
            resource_values=[2, 4],
        )

        assert set(result["instance"]) == {1, 2}
        assert set(result["reads"]) == {2, 4}
        assert result.loc[
            (result["instance"] == 2) & (result["reads"] == 4), "score"
        ].iloc[0] == pytest.approx(50.0)
        assert (tmp_path / "run1_partial.pkl").exists()
        assert (tmp_path / "run2_partial.pkl").exists()
        assert (tmp_path / "run_interp.pkl").exists()

        changed = df.copy()
        changed["score"] = 999.0
        recovered = interpolate_df(
            dataframe=changed,
            resource_column="reads",
            prefix="run.pkl",
            parameters_dict={"sweep": [2], "replica": [1]},
            results_path=tmp_path,
            resource_values=[2, 4],
            overwrite_pickles=False,
        )

        pd.testing.assert_frame_equal(
            result.sort_index(axis=1),
            recovered.sort_index(axis=1),
            check_dtype=False,
        )

    def test_interpolates_resource_from_boots_and_parameters(self, tmp_path):
        df = pd.DataFrame({
            'sweep': [1, 1, 1, 2, 2, 2],
            'replica': [1, 1, 1, 1, 1, 1],
            'boots': [1, 2, 4, 1, 2, 4],
            'score': [10.0, 20.0, 40.0, 20.0, 40.0, 80.0],
            'instance': ['i1', 'i1', 'i1', 'i1', 'i1', 'i1'],
        })

        result = interpolate_df(
            dataframe=df,
            parameters_dict={'sweep': [1, 2], 'replica': [1]},
            resource_values=[1, 2, 4, 8],
            prefix='results.pkl',
            results_path=tmp_path,
            save_pickle=True,
        )

        assert set(result['sweep']) == {1, 2}
        assert set(result['replica']) == {1}
        assert set(result['instance']) == {'i1'}

        sweep_one = result[result['sweep'] == 1].sort_values('reads')
        np.testing.assert_array_equal(sweep_one['reads'].to_numpy(), [1, 2, 4])
        np.testing.assert_array_equal(sweep_one['boots'].to_numpy(), [1, 2, 4])
        np.testing.assert_array_equal(sweep_one['score'].to_numpy(), [10.0, 20.0, 40.0])

        sweep_two = result[result['sweep'] == 2].sort_values('reads')
        np.testing.assert_array_equal(sweep_two['reads'].to_numpy(), [2, 4, 8])
        np.testing.assert_array_equal(sweep_two['boots'].to_numpy(), [1, 2, 4])
        np.testing.assert_array_equal(sweep_two['score'].to_numpy(), [20.0, 40.0, 80.0])

        assert (tmp_path / 'results_interp.pkl').exists()
        assert (tmp_path / 'resultsi1_partial.pkl').exists()

    def test_reads_existing_partial_when_overwrite_disabled(self, tmp_path):
        cached = pd.DataFrame({
            'reads': [3],
            'boots': [3],
            'sweep': [1],
            'replica': [1],
            'score': [30.0],
        })
        cached.to_pickle(tmp_path / 'cached0_partial.pkl')
        df = pd.DataFrame({
            'sweep': [1, 1],
            'replica': [1, 1],
            'boots': [1, 2],
            'score': [10.0, 20.0],
        })

        result = interpolate_df(
            dataframe=df,
            parameters_dict={'sweep': [1], 'replica': [1]},
            resource_values=[1, 2],
            prefix='cached.pkl',
            results_path=tmp_path,
            overwrite_pickles=False,
            save_pickle=False,
        )

        pd.testing.assert_frame_equal(result.reset_index(drop=True), cached)

    def test_all_datapoints_interpolates_inside_resource_grid(self, tmp_path):
        df = pd.DataFrame({
            "instance": [1, 1],
            "sweep": [2, 2],
            "replica": [1, 1],
            "boots": [1, 2],
            "score": [10.0, 20.0],
        })

        result = interpolate_df(
            dataframe=df,
            resource_column="reads",
            prefix="all.pkl",
            parameters_dict={"sweep": [2], "replica": [1]},
            results_path=tmp_path,
            resource_values=[2, 3, 4],
            all_datapoints=True,
            save_pickle=False,
        )

        middle = result.loc[result["reads"] == 3].iloc[0]
        assert middle["score"] == pytest.approx(15.0)
        assert middle["boots"] == pytest.approx(1.5)


class TestProcessDfProgress:
    def test_returns_none_for_missing_dataframe(self, tmp_path):
        assert process_df_progress(df_progress=None, results_path=tmp_path) is None

    def test_progress_best_and_end_frames_are_computed_and_saved(self, tmp_path):
        df = pd.DataFrame({
            "R_budget": [10, 10, 20],
            "R_explor": [2, 4, 4],
            "tau": [0.1, 0.2, 0.2],
            "experiment": [1, 1, 1],
            "perf_ratio": [0.4, 0.8, 0.7],
            "cum_reads": [5, 7, 9],
        })

        best, end = process_df_progress(
            df,
            compute_metrics=["perf_ratio"],
            stat_measures=["mean"],
            maximizing=True,
            df_progress_name="progress.pkl",
            results_path=tmp_path,
        )

        assert (tmp_path / "progress_end.pkl").exists()
        assert "f_explor" in end.columns
        best_by_budget = best.set_index("R_budget")
        assert best_by_budget.loc[10, "mean_perf_ratio"] == pytest.approx(0.8)
        assert best_by_budget.loc[10, "mean_inv_perf_ratio"] == pytest.approx(0.2)
        assert best_by_budget.loc[20, "f_explor"] == pytest.approx(0.2)

    def test_progress_can_recover_cached_end_frame_for_minimization(self, tmp_path):
        cached_end = pd.DataFrame({
            "R_budget": [10, 10],
            "R_explor": [2, 4],
            "tau": [0.1, 0.2],
            "inv_perf_ratio": [0.8, 0.2],
            "cum_reads": [5, 7],
            "f_explor": [0.2, 0.4],
        })
        cached_end.to_pickle(tmp_path / "progress_end.pkl")

        best, end = process_df_progress(
            df_progress=None,
            compute_metrics=["inv_perf_ratio"],
            stat_measures=["mean"],
            maximizing=False,
            df_progress_name="progress.pkl",
            results_path=tmp_path,
            use_raw_dataframes=False,
            save_pickle=False,
        )

        pd.testing.assert_frame_equal(end, cached_end)
        row = best.set_index("R_budget").loc[10]
        assert row["mean_inv_perf_ratio"] == pytest.approx(0.2)
        assert row["mean_perf_ratio"] == pytest.approx(0.8)

    def test_processes_inverse_metric_for_minimization(self, tmp_path):
        df = pd.DataFrame({
            'R_budget': [10, 10, 20, 20],
            'R_explor': [2, 2, 5, 5],
            'tau': [1, 1, 1, 1],
            'cum_reads': [5, 10, 10, 20],
            'inv_perf_ratio': [0.8, 0.3, 0.6, 0.1],
        })

        best, end = process_df_progress(
            df_progress=df,
            compute_metrics=['inv_perf_ratio'],
            stat_measures=['mean'],
            maximizing=False,
            df_progress_name='inverse.pkl',
            results_path=tmp_path,
            save_pickle=False,
        )

        assert 'perf_ratio' in df.columns
        assert 'mean_perf_ratio' in best.columns
        assert set(end['f_explor']) == {0.2, 0.25}
