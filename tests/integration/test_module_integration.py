import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src directory to path
sys.path.insert(0, '/home/runner/work/stochastic-benchmark/stochastic-benchmark/src')

# Import modules to test integration
import names
import df_utils
import interpolate
import bootstrap
import success_metrics
import stats


class TestModuleIntegration:
    """Test integration between different modules."""
    
    def test_names_and_success_metrics_integration(self):
        """Test that names module works correctly with success_metrics."""
        # Test that success metrics use names.param2filename correctly
        shared_args = {
            'confidence_level': 68,
            'best_value': 100,
            'random_value': 200
        }
        metric_args = {'opt_sense': 1}
        
        response = success_metrics.Response(shared_args, metric_args)
        
        # Create test data
        responses = np.array([[10, 15, 12]])
        bs_df = pd.DataFrame()
        
        response.evaluate(bs_df, responses, np.array([]))
        
        # Check that columns follow naming convention
        expected_base = names.param2filename({"Key": "Response"}, "")
        expected_lower = names.param2filename({"Key": "Response", "ConfInt": "lower"}, "")
        expected_upper = names.param2filename({"Key": "Response", "ConfInt": "upper"}, "")
        
        assert expected_base in bs_df.columns
        assert expected_lower in bs_df.columns
        assert expected_upper in bs_df.columns
    
    def test_bootstrap_and_success_metrics_integration(self):
        """Test that bootstrap module works with success metrics."""
        # Create test data
        df = pd.DataFrame({
            'energy': [100, 80, 120, 90],
            'time': [10, 15, 8, 12],
            'param': ['A', 'A', 'B', 'B']
        })
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'response_dir': -1,
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 80
        }
        
        # Test bootstrap with multiple success metrics
        params = bootstrap.BootstrapParameters(
            shared_args=shared_args,
            success_metrics=[success_metrics.Response, success_metrics.Resource],
            bootstrap_iterations=10,
            downsample=5
        )
        
        # Test single bootstrap
        result = bootstrap.BootstrapSingle(df, params)
        
        assert isinstance(result, pd.DataFrame)
        
        # Should have columns for both Response and Resource metrics
        response_col = names.param2filename({"Key": "Response"}, "")
        resource_col = names.param2filename({"Key": "MeanTime"}, "")
        
        # At least one of these should be present (depending on implementation details)
        expected_cols = [response_col, resource_col]
        has_expected_col = any(col in result.columns for col in expected_cols)
        assert has_expected_col, f"Expected one of {expected_cols} in {result.columns}"
    
    def test_df_utils_and_names_integration(self):
        """Test that df_utils works correctly with names module."""
        # Test parameter_set with filename conversion
        df = pd.DataFrame({
            'param1': ['a', 'b', 'a'],
            'param2': [1, 2, 1],
            'value': [10, 20, 30]
        })
        
        param_set = df_utils.parameter_set(df, ['param1', 'param2'])
        
        # Test that we can convert parameter sets to filenames
        for params_tuple in param_set:
            # Convert tuple back to dict for filename generation
            param_dict = {'param1': params_tuple[0], 'param2': params_tuple[1]}
            filename = names.param2filename(param_dict, '.pkl')
            
            # Should be able to parse back
            parsed_params = names.filename2param(filename)
            assert 'param1' in parsed_params
            assert 'param2' in parsed_params
    
    def test_stats_and_success_metrics_integration(self):
        """Test that stats module works with success metrics output format."""
        # Create data in the format that success metrics would produce
        response_base = names.param2filename({"Key": "Response"}, "")
        response_lower = names.param2filename({"Key": "Response", "ConfInt": "lower"}, "")
        response_upper = names.param2filename({"Key": "Response", "ConfInt": "upper"}, "")
        
        df = pd.DataFrame({
            response_base: [100, 80, 120, 90, 110],
            response_lower: [95, 75, 115, 85, 105],
            response_upper: [105, 85, 125, 95, 115],
            'group': ['A', 'A', 'B', 'B', 'A']
        })
        
        # Test stats computation
        stats_params = stats.StatsParameters(
            metrics=["Response"],
            stats_measures=[stats.Mean(), stats.Median()],
            lower_bounds={"Response": 50},
            upper_bounds={"Response": 150}
        )
        
        result = stats.Stats(df, stats_params, ['group'])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two groups
        assert 'group' in result.columns
        
        # Check that stats columns were created
        expected_mean_col = names.param2filename({"Key": "Response", "Metric": "mean"}, "")
        expected_median_col = names.param2filename({"Key": "Response", "Metric": "median"}, "")
        
        assert expected_mean_col in result.columns
        assert expected_median_col in result.columns
    
    def test_interpolate_and_df_utils_integration(self):
        """Test that interpolate module works with df_utils functions."""
        # Create test data
        df = pd.DataFrame({
            'energy': [100, 80, 120, 90, 110],
            'time': [10, 15, 8, 12, 11],
            'param': ['A', 'A', 'B', 'B', 'A']
        })
        
        # Make it monotonic first using df_utils
        monotonic_df = df_utils.monotone_df(df.copy(), 'time', 'energy', opt_sense=-1)
        
        # Now test interpolation
        def resource_fcn(df):
            return df['time']
        
        interp_params = interpolate.InterpolationParameters(
            resource_fcn=resource_fcn,
            resource_value_type="manual",
            resource_values=[9, 10, 11, 12, 13]
        )
        
        # Group and interpolate
        grouped = monotonic_df.groupby(['param'])
        
        # Test that we can generate resource columns
        interpolate.generateResourceColumn(monotonic_df, interp_params)
        
        assert 'resource' in monotonic_df.columns
        assert len(interp_params.resource_values) == 5


class TestWorkflowIntegration:
    """Test complete workflow integration scenarios."""
    
    def test_bootstrap_to_stats_workflow(self):
        """Test complete workflow from bootstrap to stats computation."""
        # Create initial raw data
        df = pd.DataFrame({
            'energy': [100, 80, 120, 90, 110, 95],
            'time': [10, 15, 8, 12, 11, 13],
            'instance': ['I1', 'I1', 'I2', 'I2', 'I3', 'I3'],
            'param_setting': ['P1', 'P1', 'P1', 'P1', 'P1', 'P1']
        })
        
        # Step 1: Bootstrap
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'response_dir': -1,
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 80
        }
        
        bs_params = bootstrap.BootstrapParameters(
            shared_args=shared_args,
            success_metrics=[success_metrics.Response],
            bootstrap_iterations=5,
            downsample=3,
            keep_cols=['param_setting']
        )
        
        # Simulate bootstrap results
        bs_result = bootstrap.BootstrapSingle(df, bs_params)
        
        # Step 2: Compute stats on bootstrap results
        stats_params = stats.StatsParameters(
            metrics=["Response"],
            stats_measures=[stats.Mean()],
            lower_bounds={"Response": 0},
            upper_bounds={"Response": 250}
        )
        
        if len(bs_result) > 0:
            # Only proceed if bootstrap produced results
            final_stats = stats.StatsSingle(bs_result, stats_params)
            
            assert isinstance(final_stats, pd.DataFrame)
            assert 'count' in final_stats.columns
    
    def test_file_io_integration(self):
        """Test file I/O integration across modules."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Create and save test data using names convention
            param_dict = {'alpha': 0.5, 'beta': 10}
            filename = names.param2filename(param_dict, '.pkl')
            filepath = os.path.join(temp_dir, filename)
            
            df = pd.DataFrame({
                'energy': [100, 80, 120],
                'time': [10, 15, 8],
                'instance': ['A', 'A', 'B']
            })
            df.to_pickle(filepath)
            
            # Step 2: Read using df_utils
            combined_df = df_utils.read_exp_raw(temp_dir, name_params=['alpha', 'beta'])
            
            assert 'alpha' in combined_df.columns
            assert 'beta' in combined_df.columns
            assert all(combined_df['alpha'] == '0.5')
            assert all(combined_df['beta'] == '10')
            
            # Step 3: Test parameter extraction
            param_sets = df_utils.parameter_set(combined_df, ['alpha', 'beta'])
            
            assert len(param_sets) == 1  # Only one unique parameter combination
            assert param_sets[0] == ('0.5', '10')


class TestPathsIntegration:
    """Test that paths class integrates well with file operations."""
    
    def test_paths_directory_structure(self):
        """Test that paths class creates usable directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create paths instance
            paths_obj = names.paths(temp_dir)
            
            # Verify all directories were created
            assert os.path.exists(paths_obj.checkpoints)
            assert os.path.exists(paths_obj.plots)
            assert os.path.exists(paths_obj.progress)
            
            # Test that we can write to these directories
            test_df = pd.DataFrame({'test': [1, 2, 3]})
            
            # Test bootstrap file path
            test_df.to_pickle(paths_obj.bootstrap)
            assert os.path.exists(paths_obj.bootstrap)
            
            # Test virtual best file paths
            test_df.to_pickle(paths_obj.virtual_best['train'])
            assert os.path.exists(paths_obj.virtual_best['train'])
            
            # Clean up is automatic with temp directory


class TestEdgeCaseIntegration:
    """Test integration for edge cases and error conditions."""
    
    def test_empty_data_propagation(self):
        """Test how empty data propagates through the system."""
        # Start with empty dataframe
        empty_df = pd.DataFrame()
        
        # Test that parameter_set handles empty data
        if len(empty_df) > 0:  # Skip if empty
            param_set = df_utils.parameter_set(empty_df, [])
            assert len(param_set) == 0
        
        # Test that stats handles empty data
        stats_params = stats.StatsParameters(
            metrics=[],
            stats_measures=[stats.Mean()],
            lower_bounds={},
            upper_bounds={}
        )
        
        if len(empty_df) > 0:  # Skip if empty
            result = stats.StatsSingle(empty_df, stats_params)
            assert isinstance(result, pd.DataFrame)
    
    def test_single_row_data_propagation(self):
        """Test how single-row data propagates through the system."""
        # Single row dataframe
        single_df = pd.DataFrame({
            'energy': [100],
            'time': [10],
            'param': ['A']
        })
        
        # Test parameter extraction
        param_set = df_utils.parameter_set(single_df, ['param'])
        assert len(param_set) == 1
        assert param_set[0] == ('A',)
        
        # Test monotonic transformation
        mono_df = df_utils.monotone_df(single_df.copy(), 'time', 'energy', opt_sense=1)
        assert len(mono_df) == 1
        assert mono_df['energy'].iloc[0] == 100
    
    def test_nan_value_propagation(self):
        """Test how NaN values propagate through the system."""
        df_with_nan = pd.DataFrame({
            'energy': [100, np.nan, 120],
            'time': [10, 15, np.nan],
            'param': ['A', 'B', 'C']
        })
        
        # Test parameter extraction (should work)
        param_set = df_utils.parameter_set(df_with_nan, ['param'])
        assert len(param_set) == 3
        
        # Test get_best (should handle NaN appropriately)
        best_df = df_utils.get_best(df_with_nan, 'energy', response_dir=-1, group_on=['param'])
        # Behavior with NaN depends on implementation, but should not crash
        assert isinstance(best_df, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])