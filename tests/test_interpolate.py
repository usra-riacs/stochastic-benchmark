import pytest
import pandas as pd
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

# Monkey patch pandas DataFrame to add back iteritems for compatibility
if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

# Import the module to test
import sys
sys.path.insert(0, '/home/runner/work/stochastic-benchmark/stochastic-benchmark/src')

from interpolate import (
    InterpolationParameters, 
    generateResourceColumn, 
    InterpolateSingle, 
    Interpolate,
    Interpolate_reduce_mem,
    default_ninterp
)


class TestInterpolationParameters:
    """Test class for InterpolationParameters dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of InterpolationParameters."""
        def dummy_resource_fcn(df):
            return df['time']
        
        params = InterpolationParameters(resource_fcn=dummy_resource_fcn)
        
        assert params.resource_fcn == dummy_resource_fcn
        assert params.parameters == ["sweep", "replica"]
        assert params.resource_value_type == "log"
        assert params.resource_values == []
        assert params.group_on == "instance"
        assert params.min_boots == 1
        assert params.ignore_cols == []
    
    def test_custom_initialization(self):
        """Test custom initialization of InterpolationParameters."""
        def custom_resource_fcn(df):
            return df['custom_time']
        
        params = InterpolationParameters(
            resource_fcn=custom_resource_fcn,
            parameters=["param1", "param2"],
            resource_value_type="manual",
            resource_values=[1, 2, 3, 4, 5],
            group_on="custom_group",
            min_boots=5,
            ignore_cols=["col1", "col2"]
        )
        
        assert params.resource_fcn == custom_resource_fcn
        assert params.parameters == ["param1", "param2"]
        assert params.resource_value_type == "manual"
        assert params.resource_values == [1, 2, 3, 4, 5]
        assert params.group_on == "custom_group"
        assert params.min_boots == 5
        assert params.ignore_cols == ["col1", "col2"]
    
    def test_invalid_resource_value_type_warning(self):
        """Test warning for invalid resource_value_type."""
        def dummy_resource_fcn(df):
            return df['time']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = InterpolationParameters(
                resource_fcn=dummy_resource_fcn,
                resource_value_type="invalid_type"
            )
            
            # Expect 2 warnings: one for invalid type, one for removing values when switching to log
            assert len(w) == 2
            assert "Unsupported resource value type" in str(w[0].message)
            assert "does not support passing in values" in str(w[1].message)
            assert params.resource_value_type == "log"
    
    def test_manual_type_without_values_warning(self):
        """Test warning for manual type without resource values."""
        def dummy_resource_fcn(df):
            return df['time']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = InterpolationParameters(
                resource_fcn=dummy_resource_fcn,
                resource_value_type="manual",
                resource_values=[]
            )
            
            assert len(w) == 1
            assert "Manual resource value type requires resource values" in str(w[0].message)
            assert params.resource_value_type == "log"
    
    def test_data_log_type_with_values_warning(self):
        """Test warning for data/log type with provided values."""
        def dummy_resource_fcn(df):
            return df['time']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = InterpolationParameters(
                resource_fcn=dummy_resource_fcn,
                resource_value_type="data",
                resource_values=[1, 2, 3]
            )
            
            assert len(w) == 1
            assert "does not support passing in values" in str(w[0].message)


class TestGenerateResourceColumn:
    """Test class for generateResourceColumn function."""
    
    def test_generate_resource_column_data_type(self):
        """Test generating resource column with data type."""
        df = pd.DataFrame({
            'time': [1.0, 2.0, 3.0, 4.0, 5.0],
            'value': [10, 20, 30, 40, 50]
        })
        
        def resource_fcn(df):
            return df['time']
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_value_type="data"
        )
        
        generateResourceColumn(df, params)
        
        assert 'resource' in df.columns
        np.testing.assert_array_equal(df['resource'].values, [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(params.resource_values, [1.0, 2.0, 3.0, 4.0, 5.0])
    
    @patch('interpolate.gen_log_space')
    def test_generate_resource_column_log_type(self, mock_gen_log_space):
        """Test generating resource column with log type."""
        df = pd.DataFrame({
            'time': [1.0, 2.0, 3.0, 4.0, 5.0],
            'value': [10, 20, 30, 40, 50]
        })
        
        def resource_fcn(df):
            return df['time']
        
        mock_gen_log_space.return_value = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_value_type="log"
        )
        
        generateResourceColumn(df, params)
        
        assert 'resource' in df.columns
        mock_gen_log_space.assert_called_once_with(1.0, 5.0, default_ninterp)
        np.testing.assert_array_equal(params.resource_values, [1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_generate_resource_column_manual_type(self):
        """Test generating resource column with manual type."""
        df = pd.DataFrame({
            'time': [1.0, 2.0, 3.0, 4.0, 5.0],
            'value': [10, 20, 30, 40, 50]
        })
        
        def resource_fcn(df):
            return df['time']
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_value_type="manual",
            resource_values=[1.5, 2.5, 3.5]
        )
        
        generateResourceColumn(df, params)
        
        assert 'resource' in df.columns
        np.testing.assert_array_equal(df['resource'].values, [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(params.resource_values, [1.5, 2.5, 3.5])
    
    def test_resource_values_sorted_unique(self):
        """Test that resource values are sorted and unique."""
        df = pd.DataFrame({
            'time': [3.0, 1.0, 4.0, 2.0, 1.0],  # duplicates and unsorted
            'value': [10, 20, 30, 40, 50]
        })
        
        def resource_fcn(df):
            return df['time']
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_value_type="data"
        )
        
        generateResourceColumn(df, params)
        
        # Should be sorted and unique
        expected = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(params.resource_values, expected)


class TestInterpolateSingle:
    """Test class for InterpolateSingle function."""
    
    @patch('interpolate.take_closest')
    def test_interpolate_single_basic(self, mock_take_closest):
        """Test basic interpolation of a single dataframe."""
        # Mock data
        df_single = pd.DataFrame({
            'resource': [1.0, 2.0, 3.0, 4.0, 5.0],
            'value': [10.0, 20.0, 30.0, 40.0, 50.0],
            'group_col': ['A', 'A', 'A', 'A', 'A']
        })
        
        def resource_fcn(df):
            return df['resource']
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_values=np.array([1.5, 2.5, 3.5, 4.5])
        )
        
        mock_take_closest.side_effect = lambda arr, val: val  # Return the value as-is
        
        result = InterpolateSingle(df_single, params, group_on=['group_col'])
        
        assert isinstance(result, pd.DataFrame)
        assert 'value' in result.columns
        assert len(result) == 4  # interpolation points
        
        # Check that interpolation was performed
        expected_values = [15.0, 25.0, 35.0, 45.0]  # Linear interpolation
        np.testing.assert_array_almost_equal(result['value'].values, expected_values)
    
    @patch('interpolate.take_closest')
    def test_interpolate_single_with_ignore_cols(self, mock_take_closest):
        """Test interpolation with ignore columns."""
        df_single = pd.DataFrame({
            'resource': [1.0, 2.0, 3.0],
            'value': [10.0, 20.0, 30.0],
            'ignore_me': ['X', 'Y', 'Z'],
            'group_col': ['A', 'A', 'A']
        })
        
        def resource_fcn(df):
            return df['resource']
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_values=np.array([1.5, 2.5]),
            ignore_cols=['ignore_me']
        )
        
        mock_take_closest.side_effect = lambda arr, val: val
        
        result = InterpolateSingle(df_single, params, group_on=['group_col'])
        
        assert 'ignore_me' in result.columns
        # ignore_me should have the first value repeated
        assert all(result['ignore_me'] == 'X')
    
    @patch('interpolate.take_closest')
    def test_interpolate_single_duplicate_resources_warning(self, mock_take_closest):
        """Test warning when dataframe has duplicate resources."""
        df_single = pd.DataFrame({
            'resource': [1.0, 2.0, 2.0, 3.0],  # duplicate 2.0
            'value': [10.0, 20.0, 25.0, 30.0],
            'group_col': ['A', 'A', 'A', 'A']
        })
        
        def resource_fcn(df):
            return df['resource']
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_values=np.array([1.5, 2.5])
        )
        
        mock_take_closest.side_effect = lambda arr, val: val
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = InterpolateSingle(df_single, params, group_on=['group_col'])
            
            assert len(w) == 1
            assert "duplicate resources" in str(w[0].message)


class TestInterpolate:
    """Test class for Interpolate function."""
    
    @patch('interpolate.generateResourceColumn')
    def test_interpolate_basic(self, mock_generate_resource):
        """Test basic interpolation with groupby."""
        df = pd.DataFrame({
            'time': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            'value': [10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
            'instance': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        
        def resource_fcn(df):
            return df['time']
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_values=np.array([1.5, 2.5])
        )
        
        # Mock the resource column generation
        def mock_gen_resource(df, params):
            df['resource'] = df['time']
        
        mock_generate_resource.side_effect = mock_gen_resource
        
        with patch('interpolate.InterpolateSingle') as mock_interp_single:
            # Mock the single interpolation to return a simple DataFrame
            def mock_single_interp(df, params, group_on):
                return pd.DataFrame({
                    'resource': [1.5, 2.5],
                    'value': [12.5, 22.5]
                })
            
            mock_interp_single.side_effect = mock_single_interp
            
            result = Interpolate(df, params, group_on=['instance'])
            
            assert isinstance(result, pd.DataFrame)
            mock_generate_resource.assert_called_once()
            assert mock_interp_single.call_count == 2  # Two groups: A and B


class TestInterpolateReduceMem:
    """Test class for Interpolate_reduce_mem function."""
    
    @patch('pandas.read_pickle')
    @patch('interpolate.generateResourceColumn')
    def test_interpolate_reduce_mem_basic(self, mock_generate_resource, mock_read_pickle):
        """Test memory-reduced interpolation."""
        # Mock dataframes
        df1 = pd.DataFrame({
            'time': [1.0, 2.0, 3.0],
            'value': [10.0, 20.0, 30.0],
            'group_id': ['A', 'A', 'A']
        })
        
        df2 = pd.DataFrame({
            'time': [1.0, 2.0, 3.0],
            'value': [15.0, 25.0, 35.0],
            'group_id': ['B', 'B', 'B']
        })
        
        mock_read_pickle.side_effect = [df1, df2]
        
        def resource_fcn(df):
            return df['time']
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_values=np.array([1.5, 2.5])
        )
        
        def mock_gen_resource(df, params):
            df['resource'] = df['time']
        
        mock_generate_resource.side_effect = mock_gen_resource
        
        with patch('interpolate.InterpolateSingle') as mock_interp_single:
            def mock_single_interp(df, params, group_on):
                return pd.DataFrame({
                    'resource': [1.5, 2.5],
                    'value': [12.5, 22.5],
                    'group_id': [df['group_id'].iloc[0], df['group_id'].iloc[0]]
                })
            
            mock_interp_single.side_effect = mock_single_interp
            
            result = Interpolate_reduce_mem(
                ['file1.pkl', 'file2.pkl'], 
                params, 
                group_on=['group_id']  # Changed from 'instance' to 'group_id'
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 2  # At least 2 rows from concatenation
            assert mock_read_pickle.call_count == 2
            assert mock_generate_resource.call_count == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test behavior with empty dataframe."""
        df = pd.DataFrame()
        
        def resource_fcn(df):
            return pd.Series(dtype=float)
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_value_type="manual",
            resource_values=[1.0, 2.0, 3.0]
        )
        
        # Should handle empty dataframe gracefully
        generateResourceColumn(df, params)
        assert 'resource' in df.columns
        assert len(df) == 0
    
    def test_single_row_dataframe(self):
        """Test behavior with single row dataframe."""
        df = pd.DataFrame({
            'time': [1.0],
            'value': [10.0]
        })
        
        def resource_fcn(df):
            return df['time']
        
        params = InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_value_type="data"
        )
        
        generateResourceColumn(df, params)
        
        assert len(params.resource_values) == 1
        assert params.resource_values[0] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])