import pytest
from scipy.special import erfinv
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import stats
import cross_validation as cv

from cross_validation import (
    baseline_evaluate,
    all_ci,
    propagate_ci,
    random_exp_evaluate,
    seq_search_evaluate,
)


@pytest.fixture(autouse=True)
def clear_cross_validation_globals():
    cv.parameters_dict.clear()
    cv.performance_dict.clear()
    cv.parameters_summarized_dict.clear()
    cv.performance_summarized_dict.clear()
    yield
    cv.parameters_dict.clear()
    cv.performance_dict.clear()
    cv.parameters_summarized_dict.clear()
    cv.performance_summarized_dict.clear()


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
        assert eval_df.loc[0, "response_lower"] == 10
        assert eval_df.loc[0, "response_upper"] == 12


class TestCrossValidationFileWorkflows:
    def test_load_parameters_concatenates_splits_and_warns_for_missing_files(self, tmp_path):
        folders = [tmp_path / "split0", tmp_path / "split1"]
        for folder in folders:
            (folder / "params_plotting").mkdir(parents=True)

        pd.DataFrame({
            "Unnamed: 0": [0],
            "resource": [1],
            "param1": [0.1],
        }).to_csv(folders[0] / "params_plotting" / "baseline.csv", index=False)
        pd.DataFrame({
            "resource": [1],
            "param1": [0.2],
        }).to_csv(folders[1] / "params_plotting" / "baseline.csv", index=False)
        pd.DataFrame({
            "resource": [1],
            "param1": [0.4],
        }).to_csv(folders[0] / "params_plotting" / "experiment.csv", index=False)

        with pytest.warns(UserWarning, match="experiment.csv not found"):
            cv.load_parameters([str(folder) for folder in folders], ["experiment"])

        baseline = cv.parameters_dict["baseline"]
        experiment = cv.parameters_dict["experiment"]
        assert "Unnamed: 0" not in baseline.columns
        assert baseline["split_ind"].tolist() == [0, 1]
        assert experiment["split_ind"].tolist() == [0]

    def test_process_params_across_splits_summarizes_each_experiment(self):
        cv.parameters_dict["baseline"] = pd.DataFrame({
            "resource": [1, 1, 2, 2],
            "param1": [0.0, 2.0, 4.0, 6.0],
        })

        cv.process_params_across_splits(["param1"], confidence_level=68)

        summary = cv.parameters_summarized_dict["baseline"]["param1"]
        assert summary["resource"].tolist() == [1, 2]
        assert summary.loc[summary["resource"] == 1, "mean"].iloc[0] == pytest.approx(1.0)
        assert {"CI_l", "CI_u"} <= set(summary.columns)

    def test_load_and_process_performance_across_splits(self, tmp_path):
        folders = [tmp_path / "split0", tmp_path / "split1"]
        for idx, folder in enumerate(folders):
            (folder / "performance_plotting").mkdir(parents=True)
            pd.DataFrame({
                "Unnamed: 0": [0],
                "resource": [1],
                "response": [0.5 + idx],
            }).to_csv(folder / "performance_plotting" / "baseline.csv", index=False)
            pd.DataFrame({
                "resource": [1],
                "response": [0.7 + idx],
                "response_lower": [0.6 + idx],
                "response_upper": [0.8 + idx],
            }).to_csv(folder / "performance_plotting" / "experiment.csv", index=False)

        cv.load_performance(
            [str(folder) for folder in folders],
            ["experiment"],
            interpolate_flag=False,
        )
        assert "Unnamed: 0" not in cv.performance_dict["baseline"].columns

        cv.process_performance_across_splits(stats_measure="mean")

        baseline = cv.performance_summarized_dict["baseline"]
        experiment = cv.performance_summarized_dict["experiment"]
        assert baseline.loc[0, "mean"] == pytest.approx(1.0)
        assert {"mean", "CI_l", "CI_u"} <= set(experiment.columns)

    def test_create_eval_params_dfs_dispatches_projection_evaluator(self, tmp_path):
        response_col = "Resp"
        folders = [tmp_path / "split0", tmp_path / "split1"]
        for idx, folder in enumerate(folders):
            folder.mkdir()
            pd.DataFrame({
                "resource": [1],
                "param1": [idx + 0.5],
                response_col: [idx + 1.0],
                "ConfInt=lower_" + response_col: [idx + 0.8],
                "ConfInt=upper_" + response_col: [idx + 1.2],
            }).to_pickle(folder / "projection.pkl")

        raw, params, perf = cv.create_eval_params_dfs(
            "projection.pkl",
            [str(folder) for folder in folders],
            ["param1"],
            "proj_expt_evaluate",
            response_col,
        )

        assert raw["split"].tolist() == [1, 2]
        assert params["split"].tolist() == [1, 2]
        assert perf["response_lower"].tolist() == [0.8, 1.8]

