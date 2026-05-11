import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path for imports
TESTS_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(TESTS_DIR, os.pardir, 'src'))
sys.path.insert(0, SRC_PATH)

import names
from random_exploration import RandomSearchParameters, prepare_search as rs_prepare, summarize_experiments as rs_summary
from sequential_exploration import SequentialSearchParameters, prepare_search as ss_prepare, summarize_experiments as ss_summary
import random_exploration
import sequential_exploration
import stats


def create_test_df():
    """Create a simple dataframe for exploration tests."""
    key = names.param2filename({"Key": "PerfRatio"}, "")
    CIlower = names.param2filename({"Key": "PerfRatio", "ConfInt": "lower"}, "")
    CIupper = names.param2filename({"Key": "PerfRatio", "ConfInt": "upper"}, "")
    df = pd.DataFrame({
        'resource': [5, 10],
        'sweep': [0, 0],
        'replica': [0, 0],
        'order': [1, 2],
        key: [0.5, 0.8],
        CIlower: [0.4, 0.7],
        CIupper: [0.6, 0.9]
    })
    return df, key


class TestRandomSingle:
    """Tests for random_exploration.single_experiment."""

    def test_single_experiment_basic(self):
        df, key = create_test_df()
        params = random_exploration.RandomSearchParameters(
            budgets=[10],
            exploration_fracs=[0.5],
            Nexperiments=1,
            taus=[5],
            parameter_names=['sweep', 'replica'],
            key=key
        )
        result = random_exploration.single_experiment(df, params, 10, 0.5, 5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 1 in result['exploit'].values
        assert result['CummResource'].iloc[-1] <= 10

    def test_single_experiment_insufficient_budget(self):
        df, key = create_test_df()
        params = random_exploration.RandomSearchParameters(
            budgets=[10],
            exploration_fracs=[0.5],
            Nexperiments=1,
            taus=[5],
            parameter_names=['sweep', 'replica'],
            key=key
        )
        # explore_budget < tau should return None
        result = random_exploration.single_experiment(df, params, 4, 0.5, 5)
        assert result is None


class TestSequentialSingle:
    """Tests for sequential_exploration.SequentialExplorationSingle."""

    def test_sequential_single_basic(self):
        df, key = create_test_df()
        params = sequential_exploration.SequentialSearchParameters(
            budgets=[10],
            exploration_fracs=[0.5],
            taus=[5],
            order_cols=['order'],
            parameter_names=['sweep', 'replica'],
            key=key
        )
        result = sequential_exploration.SequentialExplorationSingle(df, params, 0, 10, 0.5, 5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result['exploit'].sum() >= 1

    def test_sequential_single_edge_cases(self):
        df, key = create_test_df()
        params = sequential_exploration.SequentialSearchParameters(
            budgets=[10],
            exploration_fracs=[0.5],
            taus=[5],
            order_cols=['order'],
            parameter_names=['sweep', 'replica'],
            key=key
        )
        # insufficient budget
        result = sequential_exploration.SequentialExplorationSingle(df, params, 0, 4, 0.5, 5)
        assert result is None
        # Data with NaNs should return None after dropna
        df_nan = df.copy()
        df_nan.loc[0, 'order'] = np.nan
        result = sequential_exploration.SequentialExplorationSingle(df_nan, params, 0, 10, 0.5, 5)
        assert result is None


class TestRandomExploration:
    def test_prepare_search(self):
        df = pd.DataFrame({'resource':[10,20,30]})
        params = RandomSearchParameters(taus=[5,15,25])
        rs_prepare(df, params)
        assert list(params.taus) == [10,20]

    def test_summarize_experiments(self):
        df = pd.DataFrame({
            'TotalBudget':[100,100],
            'ExplorationBudget':[50,50],
            'tau':[10,10],
            'PerfRatio':[0.8,0.9]
        })
        params = RandomSearchParameters()
        params.key = 'PerfRatio'
        params.stat_measure = stats.Mean()
        params.optimization_dir = 1
        best, exp = rs_summary(df, params)
        assert len(best) == 1
        assert len(exp) >= 1


class TestSequentialExploration:
    def test_prepare_search(self):
        df = pd.DataFrame({'resource':[10,20,30]})
        params = SequentialSearchParameters(taus=[8,18,25])
        ss_prepare(df, params)
        assert list(params.taus) == [10,20]

    def test_summarize_experiments(self):
        df = pd.DataFrame({
            'TotalBudget':[100,100],
            'ExplorationBudget':[50,50],
            'tau':[10,10],
            'PerfRatio':[0.8,0.7]
        })
        params = SequentialSearchParameters()
        params.key = 'PerfRatio'
        params.stat_measure = stats.Mean()
        params.optimization_dir = -1
        best, exp = ss_summary(df, params)
        assert len(best) == 1
        assert len(exp) >= 1
