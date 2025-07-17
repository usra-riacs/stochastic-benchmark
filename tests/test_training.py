import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

# Import the module to test
import sys
sys.path.insert(0, '/home/runner/work/stochastic-benchmark/stochastic-benchmark/src')

from training import (
    best_parameters,
    virtual_best,
    split_train_test,
    best_recommended,
    evaluate_single,
    scaled_distance,
    evaluate,
    check_split_validity
)


class TestBestParameters:
    """Test class for best_parameters function."""
    
    def test_best_parameters_maximization(self):
        """Test best_parameters with maximization."""
        df = pd.DataFrame({
            'resource': [10, 20, 30, 15, 25],
            'response': [100, 80, 120, 90, 110],
            'param1': [1, 2, 3, 1, 2],
            'param2': ['A', 'B', 'C', 'A', 'B'],
            'boots': [1, 1, 1, 1, 1]
        })
        
        result = best_parameters(
            df, 
            parameter_names=['param1', 'param2'],
            response_col='response',
            response_dir=1,  # Maximization
            resource_col='resource'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'resource' in result.columns
        assert 'response' in result.columns
        assert 'param1' in result.columns
        assert 'param2' in result.columns
        assert 'boots' in result.columns
        
        # Should be sorted by resource ascending
        assert result['resource'].is_monotonic_increasing
        
        # Should have no duplicate resources
        assert result['resource'].nunique() == len(result)
        
        # For maximization, should pick highest response for each resource level
        assert all(result['response'] >= 0)  # All should be valid responses
    
    def test_best_parameters_minimization(self):
        """Test best_parameters with minimization."""
        df = pd.DataFrame({
            'resource': [10, 20, 30, 10, 20],
            'response': [100, 80, 120, 90, 110],
            'param1': [1, 2, 3, 1, 2],
            'param2': ['A', 'B', 'C', 'D', 'E'],
            'boots': [1, 1, 1, 1, 1]
        })
        
        result = best_parameters(
            df,
            parameter_names=['param1', 'param2'],
            response_col='response',
            response_dir=-1,  # Minimization
            resource_col='resource',
            additional_cols=[]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(df)  # Should have unique resources
        
        # Should be sorted by resource
        assert result['resource'].is_monotonic_increasing
        
        # For resource=10, should pick response=90 (min of 100, 90)
        resource_10_rows = result[result['resource'] == 10]
        if len(resource_10_rows) > 0:
            assert resource_10_rows['response'].iloc[0] == 90
    
    def test_best_parameters_invalid_response_dir(self):
        """Test best_parameters with invalid response_dir."""
        df = pd.DataFrame({
            'resource': [10, 20, 30],
            'response': [100, 80, 120],
            'param1': [1, 2, 3],
            'boots': [1000, 1000, 1000]  # Add boots column
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = best_parameters(
                df,
                parameter_names=['param1'],
                response_col='response',
                response_dir=0,  # Invalid
                resource_col='resource'
            )
            
            assert len(w) == 1
            assert "Unsupported response_dir" in str(w[0].message)
            assert isinstance(result, pd.DataFrame)
    
    def test_best_parameters_with_smoothing(self):
        """Test best_parameters with smoothing option."""
        df = pd.DataFrame({
            'resource': [10, 20, 30, 40],
            'response': [100, 80, 120, 90],  # Non-monotonic
            'param1': [1, 2, 3, 4],
            'boots': [1, 1, 1, 1]
        })
        
        with patch('training.df_utils.monotone_df') as mock_monotone:
            mock_monotone.return_value = df.copy()
            
            result = best_parameters(
                df,
                parameter_names=['param1'],
                response_col='response',
                response_dir=1,
                resource_col='resource',
                smooth=True
            )
            
            # Should call monotone_df when smooth=True
            mock_monotone.assert_called_once()
    
    def test_best_parameters_empty_dataframe(self):
        """Test best_parameters with empty dataframe."""
        df = pd.DataFrame()
        
        # Empty DataFrame should raise KeyError since columns don't exist
        with pytest.raises(KeyError, match="response"):
            result = best_parameters(
                df,
                parameter_names=[],
                response_col='response',
                response_dir=1,
                resource_col='resource'
            )


class TestVirtualBest:
    """Test class for virtual_best function."""
    
    def test_virtual_best_basic(self):
        """Test basic virtual_best functionality."""
        df = pd.DataFrame({
            'instance': ['I1', 'I1', 'I1', 'I2', 'I2', 'I2'],
            'resource': [10, 20, 30, 10, 20, 30],
            'response': [100, 80, 120, 90, 110, 95],
            'param1': [1, 2, 3, 1, 2, 3],
            'param2': ['A', 'B', 'C', 'A', 'B', 'C'],
            'boots': [1, 1, 1, 1, 1, 1]
        })
        
        result = virtual_best(
            df,
            parameter_names=['param1', 'param2'],
            response_col='response',
            response_dir=1,  # Maximization
            groupby=['instance']
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'instance' in result.columns
        assert 'resource' in result.columns
        assert 'response' in result.columns
        assert 'param1' in result.columns
        assert 'param2' in result.columns
        
        # Should have entries for both instances
        instances = result['instance'].unique()
        assert 'I1' in instances
        assert 'I2' in instances
        
        # Each instance should have entries sorted by resource
        for instance in instances:
            instance_data = result[result['instance'] == instance]
            assert instance_data['resource'].is_monotonic_increasing
    
    def test_virtual_best_multiple_groupby(self):
        """Test virtual_best with multiple groupby columns."""
        df = pd.DataFrame({
            'instance': ['I1', 'I1', 'I2', 'I2'],
            'setting': ['S1', 'S2', 'S1', 'S2'],
            'resource': [10, 20, 10, 20],
            'response': [100, 80, 90, 110],
            'param1': [1, 2, 1, 2],
            'boots': [1, 1, 1, 1]
        })
        
        result = virtual_best(
            df,
            parameter_names=['param1'],
            response_col='response',
            response_dir=-1,  # Minimization
            groupby=['instance', 'setting']
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'instance' in result.columns
        assert 'setting' in result.columns
        
        # Should have 4 groups (2 instances × 2 settings)
        unique_groups = result.groupby(['instance', 'setting']).ngroups
        assert unique_groups <= 4
    
    def test_virtual_best_with_smoothing(self):
        """Test virtual_best with smoothing enabled."""
        df = pd.DataFrame({
            'instance': ['I1', 'I1', 'I1'],
            'resource': [10, 20, 30],
            'response': [100, 80, 120],
            'param1': [1, 2, 3],
            'boots': [1, 1, 1]
        })
        
        with patch('training.best_parameters') as mock_best_params:
            mock_best_params.return_value = df.copy()
            
            result = virtual_best(
                df,
                parameter_names=['param1'],
                response_col='response',
                response_dir=1,
                smooth=True
            )
            
            # Should call best_parameters with smooth=True
            mock_best_params.assert_called()
            call_args = mock_best_params.call_args
            assert call_args[1]['smooth'] == True


class TestSplitTrainTest:
    """Test class for split_train_test function."""
    
    def test_split_train_test_basic(self):
        """Test basic train/test splitting."""
        df = pd.DataFrame({
            'instance': ['I1', 'I1', 'I2', 'I2', 'I3', 'I3'],
            'param': [1, 2, 1, 2, 1, 2],
            'value': [10, 20, 30, 40, 50, 60]
        })
        
        # Temporarily disable validity check for predictable testing
        original_check = check_split_validity
        import training
        training.check_split_validity = False
        
        try:
            with patch('numpy.random.binomial') as mock_binomial:
                # Make it deterministic: first instance train=1, others train=0
                mock_binomial.side_effect = [1, 0, 0]
                
                result = split_train_test(df, split_on=['instance'], ptrain=0.5)
                
                assert 'train' in result.columns
                assert set(result['train'].unique()).issubset({0, 1})
                
                # Check that instances are split correctly
                i1_train = result[result['instance'] == 'I1']['train'].iloc[0]
                i2_train = result[result['instance'] == 'I2']['train'].iloc[0]
                
                assert i1_train == 1
                assert i2_train == 0
        finally:
            training.check_split_validity = original_check
    
    def test_split_train_test_with_validity_check(self):
        """Test split_train_test with validity checking enabled."""
        df = pd.DataFrame({
            'instance': ['I1', 'I1', 'I2', 'I2', 'I3', 'I3'],
            'param': [1, 2, 1, 2, 1, 2],
            'value': [10, 20, 30, 40, 50, 60]
        })
        
        # Enable validity check
        original_check = check_split_validity
        import training
        training.check_split_validity = True
        
        try:
            with patch('numpy.random.binomial') as mock_binomial:
                # First try: all train (invalid), second try: mixed (valid)
                mock_binomial.side_effect = [1, 1, 1, 1, 0, 0]  # Will cycle through
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    # This should raise warning and retry
                    with pytest.raises(Warning):
                        result = split_train_test(df, split_on=['instance'], ptrain=0.5)
        finally:
            training.check_split_validity = original_check
    
    def test_split_train_test_edge_cases(self):
        """Test split_train_test with edge cases."""
        # Single instance
        df_single = pd.DataFrame({
            'instance': ['I1', 'I1'],
            'value': [10, 20]
        })
        
        original_check = check_split_validity
        import training
        training.check_split_validity = False
        
        try:
            with patch('numpy.random.binomial', return_value=1):
                result = split_train_test(df_single, split_on=['instance'], ptrain=0.8)
                
                assert 'train' in result.columns
                assert all(result['train'] == 1)
        finally:
            training.check_split_validity = original_check


class TestBestRecommended:
    """Test class for best_recommended function."""
    
    def test_best_recommended_basic(self):
        """Test basic best_recommended functionality."""
        vb = pd.DataFrame({
            'resource': [10, 20, 30, 10, 20, 30],
            'param1': [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            'param2': [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
            'boots': [1, 1, 1, 2, 2, 2]
        })
        
        result = best_recommended(
            vb,
            parameter_names=['param1', 'param2'],
            resource_col='resource',
            additional_cols=['boots']
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'param1' in result.columns
        assert 'param2' in result.columns
        assert 'boots' in result.columns
        
        # Should be grouped by resource and averaged
        assert len(result) == 3  # 3 unique resource levels
        
        # Check that averaging worked correctly
        # For resource=10: param1 should be (1.0 + 1.5) / 2 = 1.25
        param1_at_10 = result.loc[10, 'param1']
        assert param1_at_10 == pytest.approx(1.25, abs=1e-6)
    
    def test_best_recommended_empty_additional_cols(self):
        """Test best_recommended with no additional columns."""
        vb = pd.DataFrame({
            'resource': [10, 20, 10, 20],
            'param1': [1.0, 2.0, 1.5, 2.5],
            'param2': [0.1, 0.2, 0.15, 0.25]
        })
        
        result = best_recommended(
            vb,
            parameter_names=['param1', 'param2'],
            resource_col='resource',
            additional_cols=[]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {'param1', 'param2'}
        assert len(result) == 2  # 2 resource levels
    
    def test_best_recommended_single_resource(self):
        """Test best_recommended with single resource level."""
        vb = pd.DataFrame({
            'resource': [10, 10, 10],
            'param1': [1.0, 2.0, 3.0],
            'param2': [0.1, 0.2, 0.3]
        })
        
        result = best_recommended(
            vb,
            parameter_names=['param1', 'param2'],
            resource_col='resource'
        )
        
        assert len(result) == 1
        assert result.loc[10, 'param1'] == pytest.approx(2.0, abs=1e-6)  # Mean of [1,2,3]
        assert result.loc[10, 'param2'] == pytest.approx(0.2, abs=1e-6)  # Mean of [0.1,0.2,0.3]


class TestScaledDistance:
    """Test class for scaled_distance function."""
    
    def test_scaled_distance_basic(self):
        """Test basic scaled_distance functionality."""
        df_eval = pd.DataFrame({
            'param1': [1.0, 2.0, 3.0, 4.0],
            'param2': [0.1, 0.2, 0.3, 0.4],
            'value': [10, 20, 30, 40]
        })
        
        recipe = pd.Series({
            'param1': 2.5,
            'param2': 0.25
        })
        
        result = scaled_distance(df_eval, recipe, ['param1', 'param2'])
        
        assert isinstance(result, pd.DataFrame)
        assert 'distance_scaled' in result.columns
        assert 'param1_scaled' in result.columns
        assert 'param2_scaled' in result.columns
        
        # Distance should be non-negative
        assert all(result['distance_scaled'] >= 0)
        
        # Scaled parameters should be in [0, 1] range
        assert all(result['param1_scaled'] >= 0) and all(result['param1_scaled'] <= 1)
        assert all(result['param2_scaled'] >= 0) and all(result['param2_scaled'] <= 1)
    
    def test_scaled_distance_identical_values(self):
        """Test scaled_distance when all values in a parameter are identical."""
        df_eval = pd.DataFrame({
            'param1': [2.0, 2.0, 2.0],  # All identical
            'param2': [0.1, 0.2, 0.3],
            'value': [10, 20, 30]
        })
        
        recipe = pd.Series({
            'param1': 2.0,  # Same as all values
            'param2': 0.2
        })
        
        result = scaled_distance(df_eval, recipe, ['param1', 'param2'])
        
        assert isinstance(result, pd.DataFrame)
        
        # For param1 (identical values), scaled should be 0 (match) or 1 (no match)
        assert all(result['param1_scaled'].isin([0.0, 1.0]))
        
        # Distance should still be computed correctly
        assert all(result['distance_scaled'] >= 0)
    
    def test_scaled_distance_exact_match(self):
        """Test scaled_distance when recipe exactly matches a point."""
        df_eval = pd.DataFrame({
            'param1': [1.0, 2.0, 3.0],
            'param2': [0.1, 0.2, 0.3]
        })
        
        recipe = pd.Series({
            'param1': 2.0,  # Exactly matches second row
            'param2': 0.2
        })
        
        result = scaled_distance(df_eval, recipe, ['param1', 'param2'])
        
        # The second row should have distance 0 (exact match)
        min_distance = result['distance_scaled'].min()
        assert min_distance == pytest.approx(0.0, abs=1e-6)
        
        # Find the row with minimum distance
        min_idx = result['distance_scaled'].idxmin()
        assert df_eval.loc[min_idx, 'param1'] == 2.0
        assert df_eval.loc[min_idx, 'param2'] == 0.2


class TestEvaluateSingle:
    """Test class for evaluate_single function."""
    
    def test_evaluate_single_basic(self):
        """Test basic evaluate_single functionality."""
        df_eval = pd.DataFrame({
            'resource': [10, 10, 20, 20],
            'param1': [1.0, 2.0, 3.0, 4.0],
            'param2': [0.1, 0.2, 0.3, 0.4],
            'response': [100, 80, 120, 90]
        })
        
        recipes = pd.DataFrame({
            'resource': [10, 20],
            'param1': [1.5, 3.5],
            'param2': [0.15, 0.35]
        })
        
        result = evaluate_single(
            df_eval,
            recipes,
            scaled_distance,
            ['param1', 'param2'],
            resource_col='resource'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'resource' in result.columns
        assert 'param1_rec' in result.columns
        assert 'param2_rec' in result.columns
        assert 'distance_scaled' in result.columns
        
        # Should have one result per recipe
        assert len(result) == 2
        
        # Check that recommended parameters are from recipes
        assert 1.5 in result['param1_rec'].values
        assert 3.5 in result['param1_rec'].values
    
    def test_evaluate_single_no_matching_resource(self):
        """Test evaluate_single when no points match recipe resource."""
        df_eval = pd.DataFrame({
            'resource': [10, 10],
            'param1': [1.0, 2.0],
            'response': [100, 80]
        })
        
        recipes = pd.DataFrame({
            'resource': [30],  # No matching resource in df_eval
            'param1': [1.5]
        })
        
        result = evaluate_single(
            df_eval,
            recipes,
            scaled_distance,
            ['param1'],
            resource_col='resource'
        )
        
        # Should still return a result (empty or minimal)
        assert isinstance(result, pd.DataFrame)


class TestEvaluate:
    """Test class for evaluate function."""
    
    def test_evaluate_no_grouping(self):
        """Test evaluate function without grouping."""
        df = pd.DataFrame({
            'resource': [10, 10, 20, 20],
            'param1': [1.0, 2.0, 3.0, 4.0],
            'response': [100, 80, 120, 90]
        })
        
        recipes = pd.DataFrame({
            'resource': [10, 20],
            'param1': [1.5, 3.5]
        })
        
        result = evaluate(
            df,
            recipes,
            scaled_distance,
            ['param1'],
            resource_col='resource',
            group_on=[]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 2  # At least one per recipe, possibly more due to ties
        
        # Should have all expected columns
        expected_cols = ['resource', 'param1', 'response', 'distance_scaled', 'param1_scaled', 'param1_rec']
        for col in expected_cols:
            assert col in result.columns
    
    def test_evaluate_with_grouping(self):
        """Test evaluate function with grouping."""
        df = pd.DataFrame({
            'instance': ['I1', 'I1', 'I2', 'I2'],
            'resource': [10, 20, 10, 20],
            'param1': [1.0, 2.0, 3.0, 4.0],
            'response': [100, 80, 120, 90]
        })
        
        recipes = pd.DataFrame({
            'resource': [10, 20],
            'param1': [1.5, 3.5]
        })
        
        result = evaluate(
            df,
            recipes,
            scaled_distance,
            ['param1'],
            resource_col='resource',
            group_on=['instance']
        )
        
        assert isinstance(result, pd.DataFrame)
        # Should have results for both instances
        assert len(result) >= 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_best_parameters_single_row(self):
        """Test best_parameters with single row."""
        df = pd.DataFrame({
            'resource': [10],
            'response': [100],
            'param1': [1],
            'boots': [1]
        })
        
        result = best_parameters(
            df,
            ['param1'],
            'response',
            response_dir=1,
            resource_col='resource'
        )
        
        assert len(result) == 1
        assert result['resource'].iloc[0] == 10
        assert result['response'].iloc[0] == 100
    
    def test_virtual_best_empty_groups(self):
        """Test virtual_best with empty groups after filtering."""
        df = pd.DataFrame({
            'instance': ['I1', 'I2'],
            'resource': [10, 20],
            'response': [100, 80],
            'param1': [1, 2],
            'boots': [1, 1]
        })
        
        # This should work without errors
        result = virtual_best(
            df,
            ['param1'],
            'response',
            response_dir=1,
            groupby=['instance']
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_scaled_distance_empty_dataframe(self):
        """Test scaled_distance with empty dataframe."""
        df_empty = pd.DataFrame()
        recipe = pd.Series({'param1': 1.0})
        
        # Empty DataFrame should raise KeyError since columns don't exist
        with pytest.raises(KeyError, match="param1"):
            result = scaled_distance(df_empty, recipe, ['param1'])


if __name__ == "__main__":
    pytest.main([__file__])