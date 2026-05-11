import pytest
from scipy.special import erfinv
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import stats

from cross_validation import (
    baseline_evaluate,
    all_ci,
    propagate_ci,
    random_exp_evaluate,
    seq_search_evaluate,
)


class TestBaselineEvaluate:
    def test_basic(self):
        df = pd.DataFrame({
            'resource': [1, 1, 2, 2],
            'param': [10, 12, 20, 22],
            'metric': [0.1, 0.2, 0.3, 0.4]
        })
        params, eval_df = baseline_evaluate(df, ['param'], 'metric')
        assert list(params.columns) == ['resource', 'param']
        assert params.loc[0, 'param'] == 11
        assert eval_df.loc[1, 'metric'] == 0.35

class TestAllCI:
    def test_compute(self):
        df = pd.DataFrame({'resource': [1,2,3], 'p':[1.0,2.0,3.0]})
        result = all_ci(df, 'p', confidence_level=68)
        mean = df['p'].mean()
        std = np.nanstd(df["p"].values)
        fact = erfinv(68/100.0)*np.sqrt(2.0)
        np.testing.assert_allclose(result['mean'], [mean])
        np.testing.assert_allclose(result['CI_l'], [mean - fact*std])
        np.testing.assert_allclose(result['CI_u'], [mean + fact*std])

class TestPropagateCI:
    def test_mean(self):
        df = pd.DataFrame({
            'response':[10,20,30],
            'response_lower':[9,19,29],
            'response_upper':[11,21,31]
        })
        res = propagate_ci(df, 'mean')
        sm = stats.Mean()
        cent, cl, cu = sm.ConfInts(df['response'], df['response_lower'], df['response_upper'])
        np.testing.assert_allclose(res['mean'], [cent])
        np.testing.assert_allclose(res['CI_l'], [cl])
        np.testing.assert_allclose(res['CI_u'], [cu])

    def test_invalid_measure(self):
        df = pd.DataFrame({'response':[1], 'response_lower':[0], 'response_upper':[2]})
        with pytest.raises(ValueError):
            propagate_ci(df, 'invalid')

class TestPropagateCI2:
    def test_propagate_ci_mean(self):
        df = pd.DataFrame({
            'response': [1.0, 2.0, 3.0],
            'response_lower': [0.5, 1.5, 2.5],
            'response_upper': [1.5, 2.5, 3.5]
        })
        res = propagate_ci(df, 'mean')
        assert {'mean','CI_l','CI_u'} <= set(res.columns)

    def test_invalid_measure(self):
        df = pd.DataFrame({'response':[1],'response_lower':[0],'response_upper':[2]})
        with pytest.raises(ValueError):
            propagate_ci(df, 'other')


class TestEvaluateFuncs:
    def test_random_exp_evaluate(self):
        df = pd.DataFrame({
            'TotalBudget':[10,10,20,20],
            'resource':[1,2,1,2],
            'param1':[0.1,0.2,0.3,0.4],
            'Resp':[10,20,30,40],
            'ConfInt=lower_Resp':[9,18,28,38],
            'ConfInt=upper_Resp':[11,22,32,42]
        })
        params_df, eval_df = random_exp_evaluate(df, ['param1'], 'Resp')
        assert 'resource' in params_df.columns
        assert 'response' in eval_df.columns
        assert len(params_df) == 2

    def test_seq_search_evaluate(self):
        df = pd.DataFrame({
            'TotalBudget':[10,20],
            'param1':[0.1,0.2],
            'resource':[5,5],
            'Resp':[11,12],
            'ConfInt=upper_Resp':[12,13],
            'ConfInt=lower_Resp':[10,11]
        })
        params_df, eval_df = seq_search_evaluate(df, ['param1'], 'Resp')
        assert 'resource' in params_df.columns
        assert len(eval_df) == 2


