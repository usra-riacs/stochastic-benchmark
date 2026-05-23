import pandas as pd
import numpy as np
from unittest.mock import patch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from random_exploration import (
    RandomSearchParameters,
    RandomExploration,
    apply_allocations,
    prepare_search,
    run_experiments,
    summarize_experiments,
    single_experiment,
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
            "allowed": [True, True, False, True, True, False],
            key: values,
            ci_lower: values - 0.05,
            ci_upper: values + 0.05,
        }
    )
    return df, key

class TestPrepareSearch:
    def test_calls_take_closest(self):
        df = pd.DataFrame({'resource':[1,5,10]})
        params = RandomSearchParameters(taus=[2,7])
        with patch('random_exploration.take_closest', side_effect=lambda arr, v: v) as mock_tc:
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
        params = RandomSearchParameters()
        best, exp = summarize_experiments(data, params)
        assert len(best) == 2
        assert best.iloc[0]['TotalBudget'] == 20
        assert exp[exp["TotalBudget"]==20]["PerfRatio"].max() == 0.8

    def test_minimization_selects_lowest_response(self):
        data = pd.DataFrame({
            "TotalBudget": [10, 10, 20, 20],
            "ExplorationBudget": [2, 4, 2, 4],
            "tau": [1, 1, 1, 1],
            "PerfRatio": [0.5, 0.1, 0.7, 0.2],
        })
        params = RandomSearchParameters(optimization_dir=-1)
        best, exp = summarize_experiments(data, params)
        assert best.set_index("TotalBudget").loc[10, "PerfRatio"] == 0.1
        assert exp[exp["TotalBudget"] == 20]["PerfRatio"].iloc[0] == 0.2


class TestSingleExperiment:
    def setup_method(self):
        self.key = names.param2filename({"Key": "PerfRatio"}, "")
        self.CIlower = names.param2filename({"Key": "PerfRatio", "ConfInt": "lower"}, "")
        self.CIupper = names.param2filename({"Key": "PerfRatio", "ConfInt": "upper"}, "")
        self.df = pd.DataFrame({
            "resource": [1, 1, 2, 2],
            "sweep": [0, 1, 0, 1],
            "replica": [0, 0, 0, 0],
            self.key: [0.1, 0.2, 0.3, 0.4],
            self.CIlower: [0.05, 0.15, 0.25, 0.35],
            self.CIupper: [0.15, 0.25, 0.35, 0.45],
        })
        self.params = RandomSearchParameters(key=self.key)

    def test_basic_run(self):
        result = single_experiment(self.df, self.params, budget=2, explore_frac=0.5, tau=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert result["tau"].eq(1).all()
        assert result["CummResource"].max() <= 2

    def test_tau_alignment(self):
        result = single_experiment(self.df, self.params, budget=4, explore_frac=0.5, tau=3)
        assert isinstance(result, pd.DataFrame)
        assert result["tau"].iloc[0] == 2

    def test_budget_too_low_returns_none(self):
        res = single_experiment(self.df, self.params, budget=4, explore_frac=0.1, tau=5)
        assert res is None

    def test_zero_tau_returns_none(self):
        df = self.df.copy()
        df["resource"] = [0, 0, 1, 1]
        res = single_experiment(df, self.params, budget=4, explore_frac=0.5, tau=0)
        assert res is None

    def test_minimization_with_restrict_only_explores_allowed_rows(self):
        df, key = make_search_df()
        params = RandomSearchParameters(
            parameter_names=["sweep", "replica"],
            key=key,
            optimization_dir=-1,
            restrict="allowed",
        )
        np.random.seed(7)
        result = single_experiment(df, params, budget=4, explore_frac=0.5, tau=1)
        assert isinstance(result, pd.DataFrame)
        assert result[result["exploit"] == 0]["allowed"].all()
        assert result["CummResource"].max() <= 4


class TestExperimentRunners:
    def test_run_experiments_returns_empty_frame_when_all_settings_skip(self):
        df, key = make_search_df()
        params = RandomSearchParameters(
            budgets=[1],
            exploration_fracs=[0.1],
            Nexperiments=2,
            taus=[10],
            parameter_names=["sweep", "replica"],
            key=key,
        )
        result = run_experiments(df, params)
        assert result.empty
        assert {"exploit", "tau", "TotalBudget", "ExplorationBudget", "CummResource"}.issubset(result.columns)

    def test_run_experiments_and_apply_allocations_return_final_rows(self):
        df, key = make_search_df()
        params = RandomSearchParameters(
            budgets=[4],
            exploration_fracs=[0.5],
            Nexperiments=2,
            taus=[1],
            parameter_names=["sweep", "replica"],
            key=key,
        )
        np.random.seed(3)
        result = run_experiments(df, params)
        assert len(result) == 2
        assert set(result["Experiment"]) == {0, 1}

        best_alloc = pd.DataFrame(
            {"TotalBudget": [4], "ExplorationBudget": [2], "tau": [1]}
        )
        np.random.seed(4)
        applied = apply_allocations(df, params, best_alloc)
        assert len(applied) == 2
        assert applied["TotalBudget"].eq(4).all()

    def test_random_exploration_end_to_end_returns_best_allocation(self):
        df, key = make_search_df()
        params = RandomSearchParameters(
            budgets=[4],
            exploration_fracs=[0.5],
            Nexperiments=2,
            taus=[1],
            parameter_names=["sweep", "replica"],
            key=key,
        )
        np.random.seed(5)
        best, exp_at_best, final_values = RandomExploration(df, params)
        assert not best.empty
        assert not exp_at_best.empty
        assert len(final_values) == 2
