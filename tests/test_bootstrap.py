import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from collections import defaultdict
from unittest.mock import patch, MagicMock
import copy

# Import the module to test
import sys
TESTS_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.abspath(os.path.join(TESTS_DIR, os.pardir, 'src'))
sys.path.insert(0, SRC_PATH)

from bootstrap import (
    BootstrapParameters,
    BSParams_iter,
    BSParams_range_iter,
    initBootstrap,
    BootstrapSingle,
    Bootstrap,
    Bootstrap_reduce_mem,
    EPSILON,
    confidence_level,
    gap
)
import success_metrics


def dummy_update_rule(self, df):
    """Default dummy update rule for testing.
    
    Note: update_rule functions should use signature (self, df) where
    self is the BootstrapParameters instance. The function will be called as bs_params.update_rule(bs_params, df).
    """
    pass


class TestBootstrapParameters:
    """Test class for BootstrapParameters dataclass."""
    
    def test_bootstrap_parameters_initialization(self):
        """Test BootstrapParameters initialization with defaults."""
        shared_args = {
            'resource_col': 'time',
            'response_col': 'energy',
            'response_dir': -1,
            'best_value': 100,
            'random_value': 200,
            'confidence_level': 68
        }
        
        # Create a simple update rule function
        def dummy_update_rule(self, df):
            pass
            
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        assert params.shared_args == shared_args
        assert params.agg is None
        assert params.bootstrap_iterations == 1000
        assert params.downsample == 10
        assert params.keep_cols == []
        assert callable(params.update_rule)
        assert len(params.success_metrics) == 1  # Default PerfRatio
    
    def test_bootstrap_parameters_custom_initialization(self):
        """Test BootstrapParameters with custom values."""
        shared_args = {
            'resource_col': 'time',
            'response_col': 'energy',
            'response_dir': 1,
            'confidence_level': 95
        }
        
        def custom_update(params, df):
            pass
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=custom_update,
            agg='weight',
            bootstrap_iterations=500,
            downsample=20,
            keep_cols=['param1', 'param2'],
            success_metrics=[success_metrics.Response, success_metrics.Resource]
        )
        
        assert params.shared_args == shared_args
        assert params.update_rule == custom_update
        assert params.agg == 'weight'
        assert params.bootstrap_iterations == 500
        assert params.downsample == 20
        assert params.keep_cols == ['param1', 'param2']
        assert len(params.success_metrics) == 2
    
    def test_bootstrap_parameters_post_init(self):
        """Test BootstrapParameters post-initialization behavior."""
        shared_args = {'response_col': 'energy'}
        
        def dummy_update_rule(self, df):
            pass
            
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        # Check that metric_args is a defaultdict
        assert isinstance(params.metric_args, defaultdict)
        
        # Check that accessing non-existent key returns empty dict
        assert params.metric_args['NonExistent'] == {}
        assert isinstance(params.metric_args['NonExistent'], dict)
        
        # Check that update_rule is set to default if not provided
        assert hasattr(params, 'update_rule')
        assert callable(params.update_rule)
    
    def test_default_update_minimization(self):
        """Test default update rule for minimization."""
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'response_dir': -1
        }
        
        def dummy_update_rule(self, df):
            pass
        
        # Provide metric_args with RTT structure
        metric_args = defaultdict(dict)
        metric_args['RTT'] = {}
            
        params = BootstrapParameters(
            shared_args=shared_args, 
            update_rule=dummy_update_rule,
            metric_args=metric_args
        )
        
        # Create test dataframe
        df = pd.DataFrame({
            'energy': [100, 80, 120, 90],
            'time': [10, 15, 8, 12]
        })
        
        params.default_update(df)
        
        # Should set best_value to minimum energy
        assert params.shared_args['best_value'] == 80
        
        # Should set RTT_factor
        expected_rtt_factor = 1e-6 * df['time'].sum()
        assert params.metric_args['RTT']['RTT_factor'] == expected_rtt_factor
    
    def test_default_update_maximization(self):
        """Test default update rule for maximization."""
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'response_dir': 1
        }
        
        def dummy_update_rule(self, df):
            pass
        
        # Provide metric_args with RTT structure
        metric_args = defaultdict(dict)
        metric_args['RTT'] = {}
            
        params = BootstrapParameters(
            shared_args=shared_args, 
            update_rule=dummy_update_rule,
            metric_args=metric_args
        )
        
        # Create test dataframe
        df = pd.DataFrame({
            'energy': [100, 80, 120, 90],
            'time': [10, 15, 8, 12]
        })
        
        params.default_update(df)
        
        # Should set best_value to maximum energy
        assert params.shared_args['best_value'] == 120

    def test_update_rule_called_via_initBootstrap(self):
        """Test that update_rule is called correctly through initBootstrap.
        
        This test verifies the actual code path where update_rule is called
        as bs_params.update_rule(bs_params, df) from initBootstrap().
        """
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'response_dir': -1,
            'confidence_level': 68
        }
        
        # Track whether update_rule was called correctly
        call_tracker = {'called': False, 'self_param': None, 'df_param': None}
        
        def custom_update(bs_params, df):
            call_tracker['called'] = True
            call_tracker['self_param'] = bs_params
            call_tracker['df_param'] = df
            bs_params.shared_args['best_value'] = df['energy'].min()
        
        metric_args = defaultdict(dict)
        metric_args['RTT'] = {}
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=custom_update,
            metric_args=metric_args,
            bootstrap_iterations=10,
            downsample=2
        )
        
        df = pd.DataFrame({
            'energy': [100, 80, 120, 90],
            'time': [10, 15, 8, 12]
        })
        
        # Call initBootstrap which internally calls update_rule(bs_params, df)
        responses, resources = initBootstrap(df, params)
        
        # Verify update_rule was called
        assert call_tracker['called'], "update_rule should have been called"
        assert call_tracker['self_param'] is params, "First parameter should be bs_params"
        assert isinstance(call_tracker['df_param'], pd.DataFrame), "Second parameter should be DataFrame"
        assert params.shared_args['best_value'] == 80, "best_value should be updated"

    def test_default_update_called_via_initBootstrap(self):
        """Test that default_update works when called through update_rule.
        
        This test verifies that when no custom update_rule is provided,
        the default_update method is correctly assigned and called as an
        unbound method through the update_rule attribute.
        """
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'response_dir': -1,
            'confidence_level': 68
        }
        
        metric_args = defaultdict(dict)
        metric_args['RTT'] = {}
        
        # Don't provide update_rule, should use default
        params = BootstrapParameters(
            shared_args=shared_args,
            metric_args=metric_args,
            bootstrap_iterations=10,
            downsample=2
        )
        
        df = pd.DataFrame({
            'energy': [100, 80, 120, 90],
            'time': [10, 15, 8, 12]
        })
        
        # Verify default_update was assigned
        assert params.update_rule is not None
        assert params.update_rule == BootstrapParameters.default_update
        
        # Call initBootstrap which internally calls update_rule(bs_params, df)
        responses, resources = initBootstrap(df, params)
        
        # Verify default_update was executed correctly
        assert 'best_value' in params.shared_args
        assert params.shared_args['best_value'] == 80  # min energy
        assert params.metric_args['RTT']['RTT_factor'] == 1e-6 * df['time'].sum()

    def test_update_rule_signature_with_self_and_df(self):
        """Test that update_rule functions must accept (self, df) parameters.
        
        This verifies the correct signature pattern where update_rule is an
        unbound function taking (BootstrapParameters instance, DataFrame).
        """
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'response_dir': -1,
            'confidence_level': 68
        }
        
        metric_args = defaultdict(dict)
        metric_args['Response'] = {'opt_sense': -1}
        metric_args['RTT'] = {'fail_value': np.nan, 'RTT_factor': 1.0}
        
        # Define update_rule like in wishart_ws.py example
        def update_rules(self, df):
            """Custom update rule that modifies shared_args and metric_args."""
            # Simulate getting ground truth from dataframe
            if 'ground_truth' in df.columns:
                self.shared_args['best_value'] = df['ground_truth'].iloc[0]
            self.metric_args['RTT']['RTT_factor'] = df['time'].iloc[0]
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=update_rules,
            metric_args=metric_args,
            bootstrap_iterations=10,
            downsample=2
        )
        
        df = pd.DataFrame({
            'energy': [100, 80, 120],
            'time': [10, 15, 8],
            'ground_truth': [75, 75, 75]
        })
        
        # Call update_rule the way bootstrap.py does
        if params.update_rule is not None:
            params.update_rule(params, df)
        
        # Verify the updates were applied
        assert params.shared_args['best_value'] == 75
        assert params.metric_args['RTT']['RTT_factor'] == 10

    def test_update_rule_wrong_signature_fails(self):
        """Test that update_rule with wrong signature fails at runtime.
        
        This test validates that providing an update_rule with incorrect signature
        (missing self parameter) will fail when called, helping catch common mistakes.
        """
        import inspect
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'response_dir': -1,
            'confidence_level': 68
        }
        
        # Define update_rule with WRONG signature (only takes df)
        def wrong_update(df):
            """This has wrong signature - missing self parameter."""
            pass
        
        # Create params with wrong signature function
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=wrong_update,  # type: ignore  # Intentionally wrong for testing
            metric_args=defaultdict(dict),
            bootstrap_iterations=10,
            downsample=2
        )
        
        df = pd.DataFrame({
            'energy': [100, 80, 120],
            'time': [10, 15, 8]
        })
        
        # Verify the signature is actually wrong (only 1 parameter)
        sig = inspect.signature(wrong_update)
        assert len(sig.parameters) == 1, "Test setup: function should have 1 parameter"
        
        # Calling it should fail with TypeError
        with pytest.raises(TypeError) as exc_info:
            params.update_rule(params, df)  # type: ignore  # Will fail at runtime
        
        assert "takes 1 positional argument but 2 were given" in str(exc_info.value)

    def test_update_rule_signature_validation(self):
        """Test helper to validate update_rule has correct signature.
        
        This demonstrates how to check update_rule signature before using it,
        which could be useful for providing better error messages to users.
        """
        import inspect
        
        def correct_signature(self, df):
            """Correct: takes self and df."""
            pass
        
        def wrong_signature_1(df):
            """Wrong: only takes df."""
            pass
        
        def wrong_signature_2(self, df, extra):
            """Wrong: takes too many parameters."""
            pass
        
        def wrong_signature_3():
            """Wrong: takes no parameters."""
            pass
        
        # Helper function to validate signature
        def validate_update_rule_signature(func):
            """Check if function has correct signature (2 parameters)."""
            sig = inspect.signature(func)
            param_count = len(sig.parameters)
            return param_count == 2
        
        # Test validation
        assert validate_update_rule_signature(correct_signature) is True
        assert validate_update_rule_signature(wrong_signature_1) is False
        assert validate_update_rule_signature(wrong_signature_2) is False
        assert validate_update_rule_signature(wrong_signature_3) is False
        
        # Show that correct signature works
        params = BootstrapParameters(
            shared_args={'response_col': 'energy'},
            update_rule=correct_signature,
            metric_args=defaultdict(dict)
        )
        
        df = pd.DataFrame({'energy': [100, 80, 120]})
        
        # Validate before using
        assert validate_update_rule_signature(params.update_rule) is True
        
        # Should work fine
        params.update_rule(params, df)  # type: ignore  # Already validated above


class TestBSParamsIter:
    """Test class for BSParams_iter iterator."""
    
    def test_bs_params_iter_basic(self):
        """Test basic iteration of BSParams_iter."""
        shared_args = {'response_col': 'energy'}
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        iterator = BSParams_iter()
        iteration = iterator(params, nboots=5)
        
        # Test iteration
        results = list(iteration)
        
        assert len(results) == 5
        
        # Check that downsample increases
        for i, result in enumerate(results):
            assert result.downsample == i
            # Check that it's a deep copy
            assert result is not params
    
    def test_bs_params_iter_empty(self):
        """Test BSParams_iter with zero iterations."""
        shared_args = {'response_col': 'energy'}
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        iterator = BSParams_iter()
        iteration = iterator(params, nboots=0)
        
        results = list(iteration)
        assert len(results) == 0
    
    def test_bs_params_iter_single(self):
        """Test BSParams_iter with single iteration."""
        shared_args = {'response_col': 'energy'}
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        iterator = BSParams_iter()
        iteration = iterator(params, nboots=1)
        
        results = list(iteration)
        assert len(results) == 1
        assert results[0].downsample == 0


class TestBSParamsRangeIter:
    """Test class for BSParams_range_iter iterator."""
    
    def test_bs_params_range_iter_basic(self):
        """Test basic iteration of BSParams_range_iter."""
        shared_args = {'response_col': 'energy'}
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        iterator = BSParams_range_iter()
        boots_list = [5, 10, 15, 20]
        iteration = iterator(params, boots_list)
        
        results = list(iteration)
        
        assert len(results) == 4
        
        # Check that downsample values match boots_list
        for i, result in enumerate(results):
            assert result.downsample == boots_list[i]
            assert result is not params  # Deep copy
    
    def test_bs_params_range_iter_empty(self):
        """Test BSParams_range_iter with empty list."""
        shared_args = {'response_col': 'energy'}
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        iterator = BSParams_range_iter()
        iteration = iterator(params, [])
        
        results = list(iteration)
        assert len(results) == 0
    
    def test_bs_params_range_iter_generator(self):
        """Test BSParams_range_iter with generator."""
        shared_args = {'response_col': 'energy'}
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        def boots_generator():
            for i in range(3):
                yield i * 10
        
        iterator = BSParams_range_iter()
        iteration = iterator(params, boots_generator())
        
        results = list(iteration)
        
        assert len(results) == 3
        assert results[0].downsample == 0
        assert results[1].downsample == 10
        assert results[2].downsample == 20


class TestInitBootstrap:
    """Test class for initBootstrap function."""
    
    @patch('numpy.random.choice')
    @patch('numpy.random.randint')
    def test_init_bootstrap_with_agg(self, mock_randint, mock_choice):
        """Test initBootstrap with aggregation weights."""
        # Setup
        df = pd.DataFrame({
            'energy': [100, 80, 120],
            'time': [10, 15, 8],
            'weight': [0.5, 0.3, 0.2]
        })
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time'
        }
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=dummy_update_rule,
            agg='weight',
            downsample=2,
            bootstrap_iterations=3
        )
        
        # Mock random choice
        mock_choice.return_value = np.array([[0, 1, 2], [1, 2, 0]])
        
        responses, resources = initBootstrap(df, params)
        
        # Check that np.random.choice was called with correct probabilities
        expected_probs = [0.5, 0.3, 0.2]
        mock_choice.assert_called_once()
        call_args = mock_choice.call_args[0]
        assert call_args[0] == 3  # len(df)
        assert call_args[1] == (2, 3)  # (downsample, bootstrap_iterations)
        np.testing.assert_array_almost_equal(mock_choice.call_args[1]['p'], expected_probs)
        
        # Check that randint was not called
        mock_randint.assert_not_called()
        
        # Check shapes
        assert responses.shape == (2, 3)
        assert resources.shape == (2, 3)
    
    @patch('numpy.random.randint')
    def test_init_bootstrap_without_agg(self, mock_randint):
        """Test initBootstrap without aggregation weights."""
        df = pd.DataFrame({
            'energy': [100, 80, 120, 90],
            'time': [10, 15, 8, 12]
        })
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time'
        }
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=dummy_update_rule,
            agg=None,
            downsample=3,
            bootstrap_iterations=5
        )
        
        # Mock random integers
        mock_randint.return_value = np.array([
            [0, 1, 2, 3, 0],
            [1, 2, 3, 0, 1],
            [2, 3, 0, 1, 2]
        ])
        
        responses, resources = initBootstrap(df, params)
        
        # Check that np.random.randint was called correctly
        mock_randint.assert_called_once_with(
            0, 4, size=(3, 5), dtype=np.intp
        )
        
        # Check shapes
        assert responses.shape == (3, 5)
        assert resources.shape == (3, 5)
    
    def test_init_bootstrap_update_rule_called(self):
        """Test that update_rule is called during initialization."""
        df = pd.DataFrame({
            'energy': [100, 80, 120],
            'time': [10, 15, 8]
        })
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time'
        }
        
        # Create mock update rule
        update_rule_mock = MagicMock()
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=update_rule_mock,
            downsample=1,
            bootstrap_iterations=1
        )
        
        with patch('numpy.random.randint', return_value=np.array([[0]])):
            responses, resources = initBootstrap(df, params)
        
        # Check that update_rule was called
        update_rule_mock.assert_called_once_with(params, df)


class TestBootstrapSingle:
    """Test class for BootstrapSingle function."""
    
    @patch('bootstrap.initBootstrap')
    def test_bootstrap_single_basic(self, mock_init_bootstrap):
        """Test basic BootstrapSingle functionality."""
        # Setup mock data
        mock_init_bootstrap.return_value = (
            np.array([[100, 80], [120, 90]]),  # responses
            np.array([[10, 15], [8, 12]])       # resources
        )
        
        df = pd.DataFrame({
            'energy': [100, 80, 120],
            'time': [10, 15, 8],
            'param1': ['A', 'A', 'A']
        })
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 80
        }
        
        # Mock success metric
        mock_metric_class = MagicMock()
        mock_metric_instance = MagicMock()
        mock_metric_class.return_value = mock_metric_instance
        mock_metric_class.__name__ = 'MockMetric'
        
        # Make the mock evaluate method actually modify the DataFrame
        def mock_evaluate(df, responses, resources):
            df['MockMetric'] = [1.0]  # Add some test data
        mock_metric_instance.evaluate.side_effect = mock_evaluate
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=dummy_update_rule,
            success_metrics=[mock_metric_class],
            keep_cols=['param1']
        )
        
        result = BootstrapSingle(df, params)
        
        # Check that initBootstrap was called
        mock_init_bootstrap.assert_called_once_with(df, params)
        
        # Check that metric was instantiated and evaluated
        mock_metric_class.assert_called_once_with(shared_args, params.metric_args['MockMetric'])
        mock_metric_instance.evaluate.assert_called_once()
        
        # Check that keep_cols were preserved
        assert isinstance(result, pd.DataFrame)
        assert 'param1' in result.columns
        assert result['param1'].iloc[0] == 'A'
    
    def test_bootstrap_single_multiple_metrics(self):
        """Test BootstrapSingle with multiple success metrics."""
        df = pd.DataFrame({
            'energy': [100, 80, 120],
            'time': [10, 15, 8]
        })
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 80
        }
        
        # Mock multiple metrics
        mock_metric1 = MagicMock()
        mock_metric2 = MagicMock()
        mock_metric1.__name__ = 'Metric1'
        mock_metric2.__name__ = 'Metric2'
        
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_metric1.return_value = mock_instance1
        mock_metric2.return_value = mock_instance2
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=dummy_update_rule,
            success_metrics=[mock_metric1, mock_metric2]
        )
        
        with patch('bootstrap.initBootstrap') as mock_init:
            mock_init.return_value = (np.array([[100]]), np.array([[10]]))
            
            result = BootstrapSingle(df, params)
            
            # Check that both metrics were called
            mock_metric1.assert_called_once()
            mock_metric2.assert_called_once()
            mock_instance1.evaluate.assert_called_once()
            mock_instance2.evaluate.assert_called_once()


class TestBootstrap:
    """Test class for Bootstrap function."""
    
    @patch('bootstrap.BootstrapSingle')
    def test_bootstrap_with_dataframe(self, mock_bootstrap_single):
        """Test Bootstrap function with DataFrame input."""
        # Setup mock return value
        mock_result = pd.DataFrame({
            'result_col': [1, 2],
            'group': ['A', 'B'],
            'boots': [1, 1]
        })
        mock_bootstrap_single.return_value = mock_result
        
        df = pd.DataFrame({
            'energy': [100, 80, 120, 90],
            'time': [10, 15, 8, 12],
            'group': ['A', 'A', 'B', 'B']
        })
        
        shared_args = {'response_col': 'energy', 'resource_col': 'time'}
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule, downsample=1)
        
        with patch('bootstrap.Pool') as mock_pool:
            mock_pool.return_value.__enter__.return_value.map.return_value = [mock_result]
            
            result = Bootstrap(df, ['group'], [params])
        
        assert isinstance(result, pd.DataFrame)
        # BootstrapSingle should be called via groupby apply
        # The exact call count depends on groupby behavior
    
    def test_bootstrap_with_string_input(self):
        """Test Bootstrap function with string (file path) input."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            # Create test dataframe and save as pickle
            df = pd.DataFrame({
                'energy': [100, 80],
                'time': [10, 15],
                'group': ['A', 'A']
            })
            df.to_pickle(tmp_file.name)
            
            shared_args = {'response_col': 'energy', 'resource_col': 'time'}
            params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule, downsample=1)
            
            with patch('bootstrap.BootstrapSingle') as mock_bootstrap_single:
                mock_result = pd.DataFrame({'result': [1], 'group': ['A'], 'boots': [1]})
                mock_bootstrap_single.return_value = mock_result
                
                with patch('bootstrap.Pool') as mock_pool:
                    mock_pool.return_value.__enter__.return_value.map.return_value = [mock_result]
                    
                    result = Bootstrap(tmp_file.name, ['group'], [params])
            
            assert isinstance(result, pd.DataFrame)
            
            # Clean up
            os.unlink(tmp_file.name)
    
    def test_bootstrap_with_list_input(self):
        """Test Bootstrap function with list of DataFrames."""
        df1 = pd.DataFrame({
            'energy': [100, 80],
            'time': [10, 15],
            'group': ['A', 'A']
        })
        
        df2 = pd.DataFrame({
            'energy': [120, 90],
            'time': [8, 12],
            'group': ['B', 'B']
        })
        
        shared_args = {'response_col': 'energy', 'resource_col': 'time'}
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule, downsample=1)
        
        with patch('bootstrap.BootstrapSingle') as mock_bootstrap_single:
            mock_result = pd.DataFrame({'result': [1, 2], 'group': ['A', 'B'], 'boots': [1, 1]})
            mock_bootstrap_single.return_value = mock_result
            
            with patch('bootstrap.Pool') as mock_pool:
                mock_pool.return_value.__enter__.return_value.map.return_value = [mock_result]
                
                result = Bootstrap([df1, df2], ['group'], [params])
        
        assert isinstance(result, pd.DataFrame)
    
    def test_bootstrap_with_progress_dir(self):
        """Test Bootstrap function with progress directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            df = pd.DataFrame({
                'energy': [100, 80],
                'time': [10, 15],
                'group': ['A', 'A']
            })
            
            shared_args = {'response_col': 'energy', 'resource_col': 'time'}
            params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule, downsample=1)
            
            # Create a progress file that should be loaded
            progress_file = os.path.join(temp_dir, 'bootstrapped_results_boots=1.pkl')
            existing_result = pd.DataFrame({'loaded': [True], 'group': ['A'], 'boots': [1]})
            existing_result.to_pickle(progress_file)
            
            with patch('bootstrap.Pool') as mock_pool:
                mock_pool.return_value.__enter__.return_value.map.return_value = [existing_result]
                
                result = Bootstrap(df, ['group'], [params], progress_dir=temp_dir)
            
            assert isinstance(result, pd.DataFrame)


class TestBootstrapReduceMem:
    """Test class for Bootstrap_reduce_mem function."""
    
    def test_bootstrap_reduce_mem_basic(self):
        """Test basic Bootstrap_reduce_mem functionality."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            # Create test dataframe
            df = pd.DataFrame({
                'energy': [100, 80, 120, 90],
                'time': [10, 15, 8, 12],
                'upper_group': ['X', 'X', 'Y', 'Y'],
                'lower_group': ['A', 'B', 'A', 'B']
            })
            df.to_pickle(tmp_file.name)
            
            shared_args = {'response_col': 'energy', 'resource_col': 'time'}
            params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule, downsample=1)
            
            def name_function(group_data):
                return f"group_{group_data}"
            
            with tempfile.TemporaryDirectory() as bootstrap_dir:
                with patch('bootstrap.BootstrapSingle') as mock_bootstrap_single:
                    mock_result = pd.DataFrame({'result': [1], 'boots': [1]})
                    mock_bootstrap_single.return_value = mock_result
                    
                    with patch('bootstrap.Pool') as mock_pool:
                        mock_pool.return_value.__enter__.return_value.map.return_value = [mock_result]
                        
                        result = Bootstrap_reduce_mem(
                            tmp_file.name,
                            [['upper_group'], ['lower_group']],
                            [params],
                            bootstrap_dir,
                            name_function
                        )
            
            # Clean up
            os.unlink(tmp_file.name)
    
    def test_bootstrap_reduce_mem_requires_name_fcn(self):
        """Test that Bootstrap_reduce_mem fails fast when name_fcn is None."""
        df = pd.DataFrame({
            'energy': [100, 80],
            'time': [10, 15],
            'group': ['A', 'B']
        })
        
        shared_args = {'response_col': 'energy', 'resource_col': 'time'}
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        with tempfile.TemporaryDirectory() as bootstrap_dir:
            # Test with None name_fcn - should fail fast at function start
            with pytest.raises(ValueError, match="name_fcn is required for Bootstrap_reduce_mem"):
                Bootstrap_reduce_mem(
                    df,
                    [['group']],
                    [params],
                    bootstrap_dir,
                    name_fcn=None  # This should cause immediate failure
                )


class TestConstants:
    """Test module constants."""
    
    def test_constants_exist(self):
        """Test that module constants are defined."""
        assert EPSILON == 1e-10
        assert confidence_level == 68
        assert gap == 1.0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_bootstrap_single_empty_dataframe(self):
        """Test BootstrapSingle with empty dataframe."""
        df = pd.DataFrame()
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time',
            'confidence_level': 68
        }
        
        params = BootstrapParameters(shared_args=shared_args, update_rule=dummy_update_rule)
        
        # Should handle empty dataframe gracefully or raise appropriate error
        with pytest.raises((IndexError, KeyError, ValueError)):
            BootstrapSingle(df, params)
    
    def test_bootstrap_parameters_with_empty_success_metrics(self):
        """Test BootstrapParameters with empty success metrics list."""
        shared_args = {'response_col': 'energy'}
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=dummy_update_rule,
            success_metrics=[]
        )
        
        assert params.success_metrics == []
    
    def test_init_bootstrap_single_row(self):
        """Test initBootstrap with single row dataframe."""
        df = pd.DataFrame({
            'energy': [100],
            'time': [10]
        })
        
        shared_args = {
            'response_col': 'energy',
            'resource_col': 'time'
        }
        
        params = BootstrapParameters(
            shared_args=shared_args,
            update_rule=dummy_update_rule,
            downsample=1,
            bootstrap_iterations=1
        )
        
        with patch('numpy.random.randint', return_value=np.array([[0]])):
            responses, resources = initBootstrap(df, params)
        
        assert responses.shape == (1, 1)
        assert resources.shape == (1, 1)
        assert responses[0, 0] == 100
        assert resources[0, 0] == 10


if __name__ == "__main__":
    pytest.main([__file__])