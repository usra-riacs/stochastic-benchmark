import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, '/home/runner/work/stochastic-benchmark/stochastic-benchmark/src')


class TestImports:
    """Test that all modules can be imported without errors."""
    
    def test_import_names(self):
        """Test importing names module."""
        import names
        assert hasattr(names, 'paths')
        assert hasattr(names, 'param2filename')
        assert hasattr(names, 'filename2param')
    
    def test_import_stats(self):
        """Test importing stats module."""
        import stats
        assert hasattr(stats, 'Stats')
        assert hasattr(stats, 'StatsParameters')
        assert hasattr(stats, 'Mean')
        assert hasattr(stats, 'Median')
    
    def test_import_df_utils(self):
        """Test importing df_utils module."""
        import df_utils
        assert hasattr(df_utils, 'applyParallel')
        assert hasattr(df_utils, 'monotone_df')
        assert hasattr(df_utils, 'parameter_set')
    
    def test_import_interpolate(self):
        """Test importing interpolate module."""
        import interpolate
        assert hasattr(interpolate, 'InterpolationParameters')
        assert hasattr(interpolate, 'Interpolate')
    
    def test_import_bootstrap(self):
        """Test importing bootstrap module."""
        import bootstrap
        assert hasattr(bootstrap, 'BootstrapParameters')
        assert hasattr(bootstrap, 'Bootstrap')
    
    def test_import_success_metrics(self):
        """Test importing success_metrics module."""
        import success_metrics
        assert hasattr(success_metrics, 'Response')
        assert hasattr(success_metrics, 'PerfRatio')
        assert hasattr(success_metrics, 'SuccessProb')
    
    def test_import_training(self):
        """Test importing training module."""
        import training
        assert hasattr(training, 'best_parameters')
        assert hasattr(training, 'virtual_best')
    
    def test_import_plotting(self):
        """Test importing plotting module."""
        try:
            import plotting
            assert hasattr(plotting, 'Plotting')
        except ImportError as e:
            pytest.skip(f"Plotting module import failed: {e}")
    
    def test_import_stochastic_benchmark(self):
        """Test importing main stochastic_benchmark module."""
        try:
            import stochastic_benchmark
            # Main module might have various attributes
            assert hasattr(stochastic_benchmark, '__name__')
        except ImportError as e:
            pytest.skip(f"stochastic_benchmark module import failed: {e}")
    
    def test_import_cross_validation(self):
        """Test importing cross_validation module."""
        try:
            import cross_validation
            # Check if it has some expected functions
            assert hasattr(cross_validation, '__name__')
        except ImportError as e:
            pytest.skip(f"cross_validation module import failed: {e}")


class TestBasicFunctionality:
    """Test basic functionality of key modules."""
    
    def test_names_basic_functionality(self):
        """Test basic names module functionality."""
        import names
        import tempfile
        
        # Test paths creation
        with tempfile.TemporaryDirectory() as temp_dir:
            paths_obj = names.paths(temp_dir)
            assert os.path.exists(paths_obj.checkpoints)
        
        # Test parameter filename conversion
        param_dict = {'alpha': 0.5, 'beta': 10}
        filename = names.param2filename(param_dict, '.pkl')
        assert '.pkl' in filename
        assert 'alpha=0.5' in filename
        
        # Test filename to parameter conversion
        parsed = names.filename2param(filename)
        assert 'alpha' in parsed
        assert 'beta' in parsed
    
    def test_stats_basic_functionality(self):
        """Test basic stats module functionality."""
        import pandas as pd
        import stats
        
        # Create test data
        df = pd.DataFrame({
            'Key=TestMetric': [1, 2, 3, 4, 5],
            'ConfInt=lower_Key=TestMetric': [0.8, 1.8, 2.8, 3.8, 4.8],
            'ConfInt=upper_Key=TestMetric': [1.2, 2.2, 3.2, 4.2, 5.2],
        })
        
        # Test stats computation
        stats_params = stats.StatsParameters(
            metrics=['TestMetric'],
            stats_measures=[stats.Mean()],
            lower_bounds={'TestMetric': 0},
            upper_bounds={'TestMetric': 10}
        )
        
        result = stats.StatsSingle(df, stats_params)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_bootstrap_basic_functionality(self):
        """Test basic bootstrap module functionality."""
        import pandas as pd
        import bootstrap
        import success_metrics
        
        # Create test data
        df = pd.DataFrame({
            'energy': [100, 80, 120],
            'time': [10, 15, 8]
        })
        
        # Test bootstrap parameters creation
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 80
        }
        
        params = bootstrap.BootstrapParameters(
            shared_args=shared_args,
            success_metrics=[success_metrics.Response],
            bootstrap_iterations=5,
            downsample=2
        )
        
        assert params.shared_args == shared_args
        assert params.bootstrap_iterations == 5
    
    def test_training_basic_functionality(self):
        """Test basic training module functionality."""
        import pandas as pd
        import training
        
        # Create test data
        df = pd.DataFrame({
            'resource': [10, 20, 30],
            'response': [100, 80, 120],
            'param1': [1, 2, 3]
        })
        
        # Test best parameters
        result = training.best_parameters(
            df,
            parameter_names=['param1'],
            response_col='response',
            response_dir=1
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'param1' in result.columns


class TestModuleCompatibility:
    """Test that modules work together correctly."""
    
    def test_names_and_stats_compatibility(self):
        """Test that names and stats modules work together."""
        import names
        import stats
        import pandas as pd
        
        # Create data using names convention
        metric_name = names.param2filename({'Key': 'TestMetric'}, '')
        ci_lower = names.param2filename({'Key': 'TestMetric', 'ConfInt': 'lower'}, '')
        ci_upper = names.param2filename({'Key': 'TestMetric', 'ConfInt': 'upper'}, '')
        
        df = pd.DataFrame({
            metric_name: [1, 2, 3],
            ci_lower: [0.8, 1.8, 2.8],
            ci_upper: [1.2, 2.2, 3.2]
        })
        
        # Use stats with this data
        stats_params = stats.StatsParameters(
            metrics=['TestMetric'],
            stats_measures=[stats.Mean()],
            lower_bounds={'TestMetric': 0},
            upper_bounds={'TestMetric': 10}
        )
        
        result = stats.StatsSingle(df, stats_params)
        assert isinstance(result, pd.DataFrame)
    
    def test_bootstrap_and_success_metrics_compatibility(self):
        """Test that bootstrap and success_metrics modules work together."""
        import pandas as pd
        import bootstrap
        import success_metrics
        
        df = pd.DataFrame({
            'energy': [100, 80],
            'time': [10, 15]
        })
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 80
        }
        
        # Test that success metrics can be used in bootstrap
        params = bootstrap.BootstrapParameters(
            shared_args=shared_args,
            success_metrics=[success_metrics.Response, success_metrics.Resource],
            bootstrap_iterations=3,
            downsample=1
        )
        
        # This should not raise an error
        assert len(params.success_metrics) == 2


class TestErrorHandling:
    """Test that modules handle errors gracefully."""
    
    def test_empty_data_handling(self):
        """Test that modules handle empty data gracefully."""
        import pandas as pd
        import df_utils
        import stats
        
        empty_df = pd.DataFrame()
        
        # Test parameter_set with empty data
        param_set = df_utils.parameter_set(empty_df, [])
        assert len(param_set) == 0
        
        # Test stats with empty data
        stats_params = stats.StatsParameters(
            metrics=[],
            stats_measures=[stats.Mean()],
            lower_bounds={},
            upper_bounds={}
        )
        
        # Should handle empty data without crashing
        try:
            result = stats.StatsSingle(empty_df, stats_params)
            assert isinstance(result, pd.DataFrame)
        except (KeyError, IndexError):
            # Some functions may legitimately fail with empty data
            pass
    
    def test_invalid_parameters_handling(self):
        """Test that modules handle invalid parameters appropriately."""
        import names
        
        # Test filename2param with invalid input
        try:
            result = names.filename2param("invalid_format")
            # If it doesn't raise an error, it should return a dict
            assert isinstance(result, dict)
        except ValueError:
            # Expected behavior for invalid input
            pass


if __name__ == "__main__":
    pytest.main([__file__])