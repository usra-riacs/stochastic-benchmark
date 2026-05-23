import pandas as pd
import numpy as np
from unittest.mock import patch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import sequential_exploration
from sequential_exploration import (
    SequentialSearchParameters,
    SequentialExploration,
    prepare_search,
    apply_allocations,
    run_experiments,
    summarize_experiments,
    SequentialExplorationSingle,
)
import names


def make_search_df():
    key = names.param2filename({"Key": "PerfRatio"}, "")
    ci_lower = names.param2filename({"Key": "PerfRatio", "ConfInt": "lower"}, "")
    ci_upper = names.param2filename({"Key": "PerfRatio", "ConfInt": "upper"}, "")
    values = np.array([0.6, 0.2, 0.4, 0.7, 0.3, 0.5])
    df = pd.DataFrame(
        {
            "resource": [1, 1, 1, 2, 2, 2],
            "sweep": [0, 1, 2, 0, 1, 2],
            "replica": [0, 0, 0, 0, 0, 0],
            "group": [0, 0, 0, 1, 1, 1],
            "orderA": [0, 1, 2, 0, 1, 2],
            "orderB": [2, 1, 0, 2, 1, 0],
            key: values,
            ci_lower: values - 0.05,
            ci_upper: values + 0.05,
        }
    )
    return df, key


def apply_grouped_in_process(grouped, func):
    frames = [func(group.copy()) for _, group in grouped]
    frames = [frame for frame in frames if len(frame) > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

class TestPrepareSearch:
    def test_calls_take_closest(self):
        df = pd.DataFrame({'resource':[1,5,10]})
        params = SequentialSearchParameters(taus=[2,7])
        with patch('sequential_exploration.take_closest', side_effect=lambda arr, v: v) as mock_tc:
            prepare_search(df, params)
            assert mock_tc.call_count == 2
            assert np.all(params.taus == np.unique([2,7]))

class TestSummarizeExperiments:
    def test_basic(self):
        data = pd.DataFrame({
            'TotalBudget':[10,10,20,20],
            'ExplorationBudget':[2,2,4,4],
            'tau':[1,1,1,1],
            'PerfRatio':[0.5,0.6,0.7,0.8]
        })
        params = SequentialSearchParameters()
        best, exp = summarize_experiments(data, params)
        assert len(best) == 2
        assert best.iloc[0]['TotalBudget'] == 20
        assert exp[exp["TotalBudget"]==20]["PerfRatio"].max() == 0.8


class TestSequentialExplorationSingle:
    def setup_method(self):
        self.key = names.param2filename({"Key": "PerfRatio"}, "")
        self.CIlower = names.param2filename({"Key": "PerfRatio", "ConfInt": "lower"}, "")
        self.CIupper = names.param2filename({"Key": "PerfRatio", "ConfInt": "upper"}, "")
        self.df = pd.DataFrame({
            "resource": [1, 1, 2, 2],
            "sweep": [0, 1, 0, 1],
            "replica": [0, 0, 0, 0],
            "orderA": [0, 1, 0, 1],
            "orderB": [1, 0, 1, 0],
            self.key: [0.1, 0.2, 0.3, 0.4],
            self.CIlower: [0.05, 0.15, 0.25, 0.35],
            self.CIupper: [0.15, 0.25, 0.35, 0.45],
        })
        self.params = SequentialSearchParameters(order_cols=["orderA", "orderB"], key=self.key)

    def test_basic_run(self):
        res = SequentialExplorationSingle(self.df, self.params, experiment=0, budget=2, explore_frac=0.5, tau=1)
        assert isinstance(res, pd.DataFrame)
        assert len(res) > 0
        assert res["tau"].eq(1).all()
        assert res["CummResource"].max() <= 2

    def test_budget_too_low_returns_none(self):
        res = SequentialExplorationSingle(self.df, self.params, experiment=0, budget=4, explore_frac=0.1, tau=5)
        assert res is None

    def test_returns_none_if_no_valid_rows(self):
        df_na = self.df.copy()
        df_na["orderA"] = np.nan
        res = SequentialExplorationSingle(df_na, self.params, experiment=0, budget=2, explore_frac=0.5, tau=1)
        assert res is None

    def test_minimization_uses_selected_order_column(self):
        df, key = make_search_df()
        params = SequentialSearchParameters(
            order_cols=["orderA", "orderB"],
            parameter_names=["sweep", "replica"],
            key=key,
            optimization_dir=-1,
        )
        result = SequentialExplorationSingle(
            df, params, experiment=1, budget=4, explore_frac=0.5, tau=1
        )
        assert isinstance(result, pd.DataFrame)
        assert result["CummResource"].max() <= 4
        assert result["tau"].eq(1).all()


class TestExperimentRunners:
    def test_run_experiments_returns_empty_frame_when_all_settings_skip(self):
        df, key = make_search_df()
        params = SequentialSearchParameters(
            budgets=[1],
            exploration_fracs=[0.1],
            taus=[10],
            order_cols=["orderA"],
            parameter_names=["sweep", "replica"],
            key=key,
        )
        result = run_experiments(df, params)
        assert result.empty
        assert {"exploit", "tau", "TotalBudget", "ExplorationBudget", "Experiment"}.issubset(result.columns)

    def test_run_experiments_and_apply_allocations_return_final_rows(self, monkeypatch):
        df, key = make_search_df()
        params = SequentialSearchParameters(
            budgets=[4],
            exploration_fracs=[0.5],
            taus=[1],
            order_cols=["orderA"],
            parameter_names=["sweep", "replica"],
            key=key,
        )
        result = run_experiments(df, params)
        assert len(result) == 1
        assert result["Experiment"].iloc[0] == 0

        monkeypatch.setattr(
            sequential_exploration.df_utils,
            "applyParallel",
            apply_grouped_in_process,
        )
        best_alloc = pd.DataFrame(
            {"TotalBudget": [4], "ExplorationBudget": [2], "tau": [1]}
        )
        applied = apply_allocations(df, params, best_alloc, ["group"])
        assert not applied.empty
        assert applied["TotalBudget"].eq(4).all()

    def test_sequential_exploration_end_to_end_uses_grouped_runner(self, monkeypatch):
        df, key = make_search_df()
        params = SequentialSearchParameters(
            budgets=[4],
            exploration_fracs=[0.5],
            taus=[1],
            order_cols=["orderA"],
            parameter_names=["sweep", "replica"],
            key=key,
        )
        monkeypatch.setattr(
            sequential_exploration.df_utils,
            "applyParallel",
            apply_grouped_in_process,
        )
        best, exp_at_best, final_values = SequentialExploration(df, params, ["group"])
        assert not best.empty
        assert not exp_at_best.empty
        assert not final_values.empty
