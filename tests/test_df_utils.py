import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import glob
from unittest.mock import patch, MagicMock

# Import the module to test
import sys
sys.path.insert(0, '/home/runner/work/stochastic-benchmark/stochastic-benchmark/src')

from df_utils import (
    applyParallel,
    monotone_df,
    eval_cumm,
    read_exp_raw,
    parameter_set,
    get_best,
    rename_df,
    EPSILON,
    confidence_level,
    s,
    gap
)


class TestConstants:
    """Test module-level constants."""
    
    def test_constants_exist(self):
        """Test that module constants are defined."""
        assert EPSILON == 1e-10
        assert confidence_level == 68
        assert s == 0.99
        assert gap == 1.0


class TestApplyParallel:
    """Test class for applyParallel function."""
    
    def test_apply_parallel_basic(self):
        """Test basic parallel apply functionality."""
        # Create test data
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        
        # Simple aggregation function
        def sum_values(group_df):
            return pd.DataFrame({'sum': [group_df['value'].sum()]})
        
        # Group the dataframe
        grouped = df.groupby('group')
        
        # Apply parallel function
        result = applyParallel(grouped, sum_values)
        
        assert isinstance(result, pd.DataFrame)
        assert 'sum' in result.columns
        assert len(result) == 3  # Three groups
        
        # Check sums are correct
        expected_sums = [3, 7, 11]  # A: 1+2, B: 3+4, C: 5+6
        assert sorted(result['sum'].values) == expected_sums
    
    def test_apply_parallel_complex_function(self):
        """Test parallel apply with more complex function."""
        df = pd.DataFrame({
            'group': ['X', 'X', 'Y', 'Y'],
            'val1': [10, 20, 30, 40],
            'val2': [1, 2, 3, 4]
        })
        
        def compute_stats(group_df):
            return pd.DataFrame({
                'mean_val1': [group_df['val1'].mean()],
                'sum_val2': [group_df['val2'].sum()],
                'count': [len(group_df)]
            })
        
        grouped = df.groupby('group')
        result = applyParallel(grouped, compute_stats)
        
        assert len(result) == 2  # Two groups
        assert set(result.columns) == {'mean_val1', 'sum_val2', 'count'}
        
        # Check one group's results
        x_results = result[result.index == 0]  # First group result
        if len(x_results) > 0:
            assert x_results['count'].iloc[0] == 2


class TestMonotoneDf:
    """Test class for monotone_df function."""
    
    def test_monotone_df_maximization(self):
        """Test monotonic transformation for maximization (opt_sense=1)."""
        df = pd.DataFrame({
            'resource': [1, 2, 3, 4, 5],
            'response': [10, 15, 12, 20, 18],  # Non-monotonic
            'other_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = monotone_df(df.copy(), 'resource', 'response', opt_sense=1)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        
        # Response should be monotonically increasing (cummax)
        response_values = result.sort_values('resource')['response'].values
        assert np.all(np.diff(response_values) >= 0)  # Non-decreasing
    
    def test_monotone_df_minimization(self):
        """Test monotonic transformation for minimization (opt_sense=-1)."""
        df = pd.DataFrame({
            'resource': [1, 2, 3, 4, 5],
            'response': [20, 15, 18, 10, 12],  # Non-monotonic
            'other_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = monotone_df(df.copy(), 'resource', 'response', opt_sense=-1)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        
        # Response should be monotonically decreasing (cummin)
        response_values = result.sort_values('resource')['response'].values
        assert np.all(np.diff(response_values) <= 0)  # Non-increasing
    
    def test_monotone_df_with_extrapolation(self):
        """Test monotonic transformation with extrapolation dataframe."""
        df = pd.DataFrame({
            'resource': [1, 2, 3],
            'response': [10, 5, 15],
            'param': ['A', 'A', 'A']
        })
        
        extrapolate_df = pd.DataFrame({
            'resource': [1, 2, 3],
            'response': [12, 8, 16],
            'param': ['A', 'A', 'A'],
            'extra_col': ['x', 'y', 'z']
        })
        
        result = monotone_df(
            df.copy(), 
            'resource', 
            'response', 
            opt_sense=1,
            extrapolate_from=extrapolate_df,
            match_on=['param']
        )
        
        assert isinstance(result, pd.DataFrame)
        # Should use extrapolation when response is not improving
    
    def test_monotone_df_already_monotonic(self):
        """Test monotonic transformation on already monotonic data."""
        df = pd.DataFrame({
            'resource': [1, 2, 3, 4, 5],
            'response': [5, 10, 15, 20, 25],  # Already monotonic
            'other_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = monotone_df(df.copy(), 'resource', 'response', opt_sense=1)
        
        # Should be unchanged (already monotonic)
        np.testing.assert_array_equal(result['response'].values, [5, 10, 15, 20, 25])


class TestEvalCumm:
    """Test class for eval_cumm function."""
    
    def test_eval_cumm_basic(self):
        """Test basic cumulative evaluation."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'resource': [1, 2, 3, 1, 2, 3],
            'response': [10, 15, 20, 5, 10, 15]
        })
        
        result = eval_cumm(df, ['group'], 'resource', 'response', opt_sense=1)
        
        assert isinstance(result, pd.DataFrame)
        assert 'cummulativeresource' in result.columns
        
        # Check cumulative resource calculation
        group_a = result[result['group'] == 'A'].sort_values('resource')
        expected_cumm = [1, 3, 6]  # Cumulative sum of [1, 2, 3]
        np.testing.assert_array_equal(group_a['cummulativeresource'].values, expected_cumm)
    
    def test_eval_cumm_multiple_groups(self):
        """Test cumulative evaluation with multiple groups."""
        df = pd.DataFrame({
            'group1': ['X', 'X', 'Y', 'Y'],
            'group2': [1, 1, 2, 2],
            'resource': [10, 20, 15, 25],
            'response': [100, 200, 150, 250]
        })
        
        result = eval_cumm(df, ['group1', 'group2'], 'resource', 'response', opt_sense=1)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # Same number of rows
        assert 'cummulativeresource' in result.columns


class TestReadExpRaw:
    """Test class for read_exp_raw function."""
    
    def test_read_exp_raw_basic(self):
        """Test reading raw experiment files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test pickle files
            df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
            
            file1 = os.path.join(temp_dir, 'exp1.pkl')
            file2 = os.path.join(temp_dir, 'exp2.pkl')
            
            df1.to_pickle(file1)
            df2.to_pickle(file2)
            
            result = read_exp_raw(temp_dir)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 4  # Combined rows from both files
            assert set(result.columns) == {'a', 'b'}
    
    def test_read_exp_raw_with_params(self):
        """Test reading raw files with parameter extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file with parameters in name
            df1 = pd.DataFrame({'value': [1, 2]})
            file1 = os.path.join(temp_dir, 'alpha=0.5_beta=10.pkl')
            df1.to_pickle(file1)
            
            with patch('df_utils.names.filename2param') as mock_filename2param:
                mock_filename2param.return_value = {'alpha': '0.5', 'beta': '10'}
                
                result = read_exp_raw(temp_dir, name_params=['alpha', 'beta'])
                
                assert 'alpha' in result.columns
                assert 'beta' in result.columns
                assert all(result['alpha'] == '0.5')
                assert all(result['beta'] == '10')
    
    def test_read_exp_raw_no_files(self):
        """Test error when no files found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(Exception, match="No raw data found"):
                read_exp_raw(temp_dir)
    
    def test_read_exp_raw_empty_directory(self):
        """Test behavior with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a non-pickle file to ensure it's ignored
            with open(os.path.join(temp_dir, 'not_pickle.txt'), 'w') as f:
                f.write('test')
            
            with pytest.raises(Exception, match="No raw data found"):
                read_exp_raw(temp_dir)


class TestParameterSet:
    """Test class for parameter_set function."""
    
    def test_parameter_set_basic(self):
        """Test basic parameter set extraction."""
        df = pd.DataFrame({
            'param1': ['a', 'b', 'a', 'c'],
            'param2': [1, 2, 1, 3],
            'value': [10, 20, 30, 40]
        })
        
        param_set = parameter_set(df, ['param1', 'param2'])
        
        assert isinstance(param_set, np.ndarray)
        assert len(param_set) == 3  # Unique combinations: (a,1), (b,2), (c,3)
        
        # Check that params column was created
        assert 'params' in df.columns
        assert df['params'].iloc[0] == ('a', 1)
    
    def test_parameter_set_single_param(self):
        """Test parameter set with single parameter."""
        df = pd.DataFrame({
            'param1': ['x', 'y', 'x', 'z'],
            'value': [1, 2, 3, 4]
        })
        
        param_set = parameter_set(df, ['param1'])
        
        assert len(param_set) == 3  # Unique values: x, y, z
        expected_params = {('x',), ('y',), ('z',)}
        assert set(param_set) == expected_params
    
    def test_parameter_set_multiple_params(self):
        """Test parameter set with multiple parameters."""
        df = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': ['x', 'y', 'x', 'y'],
            'c': [10, 10, 20, 30],
            'value': [100, 200, 300, 400]
        })
        
        param_set = parameter_set(df, ['a', 'b', 'c'])
        
        assert len(param_set) == 4  # All combinations are unique
        
        # Check specific combinations
        params_list = list(param_set)
        assert (1, 'x', 10) in params_list
        assert (2, 'y', 30) in params_list


class TestGetBest:
    """Test class for get_best function."""
    
    def test_get_best_minimization(self):
        """Test getting best results for minimization."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'response': [10, 5, 15, 20, 8, 12],
            'param': [1, 2, 3, 1, 2, 3]
        })
        
        result = get_best(df, 'response', response_dir=-1, group_on=['group'])
        
        assert len(result) == 2  # One best per group
        
        # Check that minimum values were selected
        group_a_best = result[result['group'] == 'A']['response'].iloc[0]
        group_b_best = result[result['group'] == 'B']['response'].iloc[0]
        
        assert group_a_best == 5  # Minimum for group A
        assert group_b_best == 8  # Minimum for group B
    
    def test_get_best_maximization(self):
        """Test getting best results for maximization."""
        df = pd.DataFrame({
            'group': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'response': [10, 25, 15, 30, 18, 22],
            'param': [1, 2, 3, 1, 2, 3]
        })
        
        result = get_best(df, 'response', response_dir=1, group_on=['group'])
        
        assert len(result) == 2  # One best per group
        
        # Check that maximum values were selected
        group_x_best = result[result['group'] == 'X']['response'].iloc[0]
        group_y_best = result[result['group'] == 'Y']['response'].iloc[0]
        
        assert group_x_best == 25  # Maximum for group X
        assert group_y_best == 30  # Maximum for group Y
    
    def test_get_best_multiple_group_columns(self):
        """Test getting best with multiple grouping columns."""
        df = pd.DataFrame({
            'group1': ['A', 'A', 'B', 'B'],
            'group2': [1, 2, 1, 2],
            'response': [10, 20, 15, 25],
            'param': ['x', 'y', 'z', 'w']
        })
        
        result = get_best(df, 'response', response_dir=-1, group_on=['group1', 'group2'])
        
        assert len(result) == 4  # One best per unique group combination
        
        # All original rows should be present since each has unique group combination
        assert set(result['response'].values) == {10, 20, 15, 25}


class TestRenameDf:
    """Test class for rename_df function."""
    
    def test_rename_df_basic(self):
        """Test basic column renaming."""
        df = pd.DataFrame({
            'min_energy': [1, 2, 3],
            'mean_time': [0.1, 0.2, 0.3],
            'other_col': ['a', 'b', 'c']
        })
        
        result = rename_df(df.copy())
        
        # Check that columns were renamed
        assert 'min_energy' not in result.columns
        assert 'mean_time' not in result.columns
        assert 'other_col' in result.columns  # Should remain unchanged
        
        # Check that new column names exist (using names module format)
        renamed_cols = [col for col in result.columns if 'Key=' in col]
        assert len(renamed_cols) >= 2  # At least min_energy and mean_time renamed
    
    def test_rename_df_with_confidence_intervals(self):
        """Test renaming with confidence interval columns."""
        df = pd.DataFrame({
            'min_energy': [1, 2],
            'min_energy_conf_interval_lower': [0.8, 1.8],
            'min_energy_conf_interval_upper': [1.2, 2.2],
            'success_prob': [0.9, 0.95],
            'success_prob_conf_interval_lower': [0.85, 0.9],
            'success_prob_conf_interval_upper': [0.95, 1.0]
        })
        
        result = rename_df(df.copy())
        
        # Check that all related columns were renamed
        old_cols = [
            'min_energy', 'min_energy_conf_interval_lower', 'min_energy_conf_interval_upper',
            'success_prob', 'success_prob_conf_interval_lower', 'success_prob_conf_interval_upper'
        ]
        
        for old_col in old_cols:
            assert old_col not in result.columns
        
        # Check that new columns exist
        new_cols = [col for col in result.columns if 'Key=' in col]
        assert len(new_cols) == 6  # All six columns should be renamed
    
    def test_rename_df_partial_columns(self):
        """Test renaming when only some columns are present."""
        df = pd.DataFrame({
            'perf_ratio': [1.1, 1.2],
            'unknown_column': [10, 20],
            'rtt': [5, 6]
        })
        
        result = rename_df(df.copy())
        
        # Check that known columns were renamed
        assert 'perf_ratio' not in result.columns
        assert 'rtt' not in result.columns
        
        # Check that unknown columns remain
        assert 'unknown_column' in result.columns
        
        # Check that values are preserved
        assert list(result['unknown_column']) == [10, 20]
    
    def test_rename_df_no_matching_columns(self):
        """Test renaming when no columns match the rename dictionary."""
        df = pd.DataFrame({
            'custom_col1': [1, 2],
            'custom_col2': ['a', 'b']
        })
        
        result = rename_df(df.copy())
        
        # Should be unchanged since no columns match
        assert list(result.columns) == ['custom_col1', 'custom_col2']
        assert result.equals(df)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe_operations(self):
        """Test various operations on empty dataframes."""
        empty_df = pd.DataFrame()
        
        # parameter_set with empty df
        param_set = parameter_set(empty_df.copy(), [])
        assert len(param_set) == 0
        
        # rename_df with empty df
        renamed = rename_df(empty_df.copy())
        assert len(renamed) == 0
    
    def test_single_row_dataframe(self):
        """Test operations on single-row dataframes."""
        single_df = pd.DataFrame({
            'resource': [1],
            'response': [10],
            'group': ['A']
        })
        
        # monotone_df should work
        result = monotone_df(single_df.copy(), 'resource', 'response', opt_sense=1)
        assert len(result) == 1
        assert result['response'].iloc[0] == 10
        
        # get_best should work
        best = get_best(single_df, 'response', response_dir=1, group_on=['group'])
        assert len(best) == 1
        assert best['response'].iloc[0] == 10


if __name__ == "__main__":
    pytest.main([__file__])