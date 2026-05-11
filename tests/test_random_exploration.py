import pandas as pd
import numpy as np
from unittest.mock import patch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from random_exploration import (
    RandomSearchParameters,
    prepare_search,
    summarize_experiments,
    single_experiment,
)
import names

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

