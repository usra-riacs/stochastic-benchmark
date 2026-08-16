import pandas as pd
import numpy as np
import pytest
from training import split_train_test, virtual_best, evaluate
import names

@pytest.fixture
def sample_df():
    # Increased the number of instances to ensure train/test splits are non-empty
    return pd.DataFrame({
        'instance': ['a', 'b', 'a', 'b', 'c', 'c', 'd', 'd', 'e', 'e', 'f', 'f'],
        'resource': [1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'param1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        'response': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    })

def test_split_train_test_runs(sample_df):
    df = sample_df.copy()
    # Set a seed that is known to produce a valid split for this data
    np.random.seed(42)
    df_split = split_train_test(df.copy(), split_on=['instance'], ptrain=0.75)
    assert 'train' in df_split.columns
    assert set(df_split['train'].unique()).issubset({0,1})
    assert len(df_split) == len(df)
    # Check that both train and test sets are non-empty
    assert df_split['train'].sum() > 0
    assert (1 - df_split['train']).sum() > 0

def test_virtual_best_runs(sample_df):
    df = sample_df.copy()
    param_names = ['param1']
    response_col = 'response'
    vb_df = virtual_best(
        df,
        parameter_names=param_names,
        response_col=response_col,
        response_dir=1,
        groupby=['instance'],
        resource_col='resource',
        additional_cols=[],
        smooth=False
    )
    assert isinstance(vb_df, pd.DataFrame)
    expected_cols = ['resource'] + param_names
    for col in expected_cols:
        assert col in vb_df.columns
    assert len(vb_df) == df.groupby(['instance', 'resource']).ngroups

def test_evaluate_runs(sample_df):
    df = sample_df.copy()
    recipes = pd.DataFrame({'resource': [1,2], 'param1': [0.15, 0.35]})
    
    # The distance function should mimic the signature of `scaled_distance`
    # and return a dataframe with a 'distance_scaled' column.
    def distance_fcn(df_eval, recipe, parameter_names):
        df_out = df_eval.copy()
        df_out['distance_scaled'] = np.zeros(len(df_eval))
        return df_out
    
    # The evaluate function expects the response column to be named with the names.param2filename format
    response_col = names.param2filename({'Key': 'response'}, '')
    df.rename(columns={'response': response_col}, inplace=True)

    df_eval = evaluate(df, recipes, distance_fcn, parameter_names=['param1'], resource_col='resource', group_on=[])
    assert isinstance(df_eval, pd.DataFrame)
    assert 'param1' in df_eval.columns
    
    df_eval_grouped = evaluate(df, recipes, distance_fcn, parameter_names=['param1'], resource_col='resource', group_on=['instance'])
    assert isinstance(df_eval_grouped, pd.DataFrame)
    assert 'param1' in df_eval_grouped.columns
    assert len(df_eval_grouped) == len(df)