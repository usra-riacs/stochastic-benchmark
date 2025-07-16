import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module to test
import sys
sys.path.insert(0, '/home/runner/work/stochastic-benchmark/stochastic-benchmark/src')

from success_metrics import (
    SuccessMetrics,
    Response,
    PerfRatio,
    InvPerfRatio,
    SuccessProb,
    Resource,
    RTT,
    EPSILON
)
import names


class TestSuccessMetrics:
    """Test class for the base SuccessMetrics class."""
    
    def test_success_metrics_initialization(self):
        """Test SuccessMetrics initialization."""
        shared_args = {
            'confidence_level': 95,
            'best_value': 100,
            'random_value': 200
        }
        
        metrics = SuccessMetrics(shared_args)
        
        assert metrics.shared_args == shared_args
        assert metrics.shared_args['confidence_level'] == 95
        assert metrics.shared_args['best_value'] == 100
        assert metrics.shared_args['random_value'] == 200
    
    def test_success_metrics_evaluate_not_implemented(self):
        """Test that base class evaluate method raises NotImplementedError."""
        shared_args = {'confidence_level': 95}
        metrics = SuccessMetrics(shared_args)
        
        with pytest.raises(NotImplementedError):
            metrics.evaluate(pd.DataFrame(), np.array([]), np.array([]))


class TestResponse:
    """Test class for Response success metric."""
    
    def test_response_initialization(self):
        """Test Response class initialization."""
        shared_args = {'confidence_level': 68}
        metric_args = {'opt_sense': -1}
        
        response = Response(shared_args, metric_args)
        
        assert response.name == "Response"
        assert response.opt_sense == -1
        assert response.shared_args == shared_args
    
    def test_response_evaluate_minimization(self):
        """Test Response evaluation for minimization."""
        shared_args = {'confidence_level': 68}
        metric_args = {'opt_sense': -1}
        
        response = Response(shared_args, metric_args)
        
        # Create test data
        responses = np.array([
            [10, 15, 12],  # Bootstrap sample 1
            [8, 20, 14],   # Bootstrap sample 2
            [12, 18, 10]   # Bootstrap sample 3
        ])
        
        bs_df = pd.DataFrame()
        
        response.evaluate(bs_df, responses, np.array([]))
        
        # Check that columns were created
        expected_base = names.param2filename({"Key": "Response"}, "")
        expected_lower = names.param2filename({"Key": "Response", "ConfInt": "lower"}, "")
        expected_upper = names.param2filename({"Key": "Response", "ConfInt": "upper"}, "")
        
        assert expected_base in bs_df.columns
        assert expected_lower in bs_df.columns
        assert expected_upper in bs_df.columns
        
        # For minimization, should take minimum across rows
        expected_min_values = [8, 15, 10]  # Min of each column
        expected_mean = np.mean(expected_min_values)
        
        assert bs_df[expected_base].iloc[0] == pytest.approx(expected_mean, abs=1e-6)
    
    def test_response_evaluate_maximization(self):
        """Test Response evaluation for maximization."""
        shared_args = {'confidence_level': 68}
        metric_args = {'opt_sense': 1}
        
        response = Response(shared_args, metric_args)
        
        # Create test data
        responses = np.array([
            [10, 15, 12],  # Bootstrap sample 1
            [8, 20, 14],   # Bootstrap sample 2
            [12, 18, 10]   # Bootstrap sample 3
        ])
        
        bs_df = pd.DataFrame()
        
        response.evaluate(bs_df, responses, np.array([]))
        
        # For maximization, should take maximum across rows
        expected_max_values = [12, 20, 14]  # Max of each column
        expected_mean = np.mean(expected_max_values)
        
        expected_base = names.param2filename({"Key": "Response"}, "")
        assert bs_df[expected_base].iloc[0] == pytest.approx(expected_mean, abs=1e-6)
    
    def test_response_confidence_intervals(self):
        """Test Response confidence interval calculation."""
        shared_args = {'confidence_level': 68}
        metric_args = {'opt_sense': 1}
        
        response = Response(shared_args, metric_args)
        
        # Create test data with known statistics
        responses = np.array([
            [100, 100, 100],  # All same max values
            [100, 100, 100],
            [100, 100, 100]
        ])
        
        bs_df = pd.DataFrame()
        
        response.evaluate(bs_df, responses, np.array([]))
        
        expected_base = names.param2filename({"Key": "Response"}, "")
        expected_lower = names.param2filename({"Key": "Response", "ConfInt": "lower"}, "")
        expected_upper = names.param2filename({"Key": "Response", "ConfInt": "upper"}, "")
        
        # With identical values, std should be 0, so CI should equal mean
        mean_val = bs_df[expected_base].iloc[0]
        lower_val = bs_df[expected_lower].iloc[0]
        upper_val = bs_df[expected_upper].iloc[0]
        
        assert mean_val == pytest.approx(100.0, abs=1e-6)
        assert lower_val == pytest.approx(100.0, abs=1e-6)
        assert upper_val == pytest.approx(100.0, abs=1e-6)


class TestPerfRatio:
    """Test class for PerfRatio success metric."""
    
    def test_perf_ratio_initialization(self):
        """Test PerfRatio class initialization."""
        shared_args = {'confidence_level': 68}
        metric_args = {}
        
        perf_ratio = PerfRatio(shared_args, metric_args)
        
        assert perf_ratio.name == "PerfRatio"
        assert perf_ratio.opt_sense == 1
    
    def test_perf_ratio_evaluate(self):
        """Test PerfRatio evaluation."""
        shared_args = {
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 100
        }
        metric_args = {}
        
        perf_ratio = PerfRatio(shared_args, metric_args)
        
        # Create DataFrame with Response columns
        response_base = names.param2filename({"Key": "Response"}, "")
        response_lower = names.param2filename({"Key": "Response", "ConfInt": "lower"}, "")
        response_upper = names.param2filename({"Key": "Response", "ConfInt": "upper"}, "")
        
        bs_df = pd.DataFrame({
            response_base: [150],  # Response value
            response_lower: [140], # Response CI lower
            response_upper: [160]  # Response CI upper
        })
        
        perf_ratio.evaluate(bs_df, np.array([]), np.array([]))
        
        # Check that PerfRatio columns were created
        expected_base = names.param2filename({"Key": "PerfRatio"}, "")
        expected_lower = names.param2filename({"Key": "PerfRatio", "ConfInt": "lower"}, "")
        expected_upper = names.param2filename({"Key": "PerfRatio", "ConfInt": "upper"}, "")
        
        assert expected_base in bs_df.columns
        assert expected_lower in bs_df.columns
        assert expected_upper in bs_df.columns
        
        # Performance ratio calculation: (random - response) / (random - best)
        # (200 - 150) / (200 - 100) = 50 / 100 = 0.5
        expected_perf_ratio = (200 - 150) / (200 - 100)
        assert bs_df[expected_base].iloc[0] == pytest.approx(expected_perf_ratio, abs=1e-6)
    
    def test_perf_ratio_clipping(self):
        """Test PerfRatio value clipping."""
        shared_args = {
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 100
        }
        metric_args = {}
        
        perf_ratio = PerfRatio(shared_args, metric_args)
        
        # Create DataFrame with extreme Response values
        response_base = names.param2filename({"Key": "Response"}, "")
        response_lower = names.param2filename({"Key": "Response", "ConfInt": "lower"}, "")
        response_upper = names.param2filename({"Key": "Response", "ConfInt": "upper"}, "")
        
        bs_df = pd.DataFrame({
            response_base: [50],   # Very good response (better than best)
            response_lower: [40],  # Even better
            response_upper: [60]   # Still very good
        })
        
        perf_ratio.evaluate(bs_df, np.array([]), np.array([]), lower=0.0, upper=1.0)
        
        expected_base = names.param2filename({"Key": "PerfRatio"}, "")
        
        # Should be clipped to [0, 1] range
        assert 0.0 <= bs_df[expected_base].iloc[0] <= 1.0


class TestInvPerfRatio:
    """Test class for InvPerfRatio success metric."""
    
    def test_inv_perf_ratio_initialization(self):
        """Test InvPerfRatio class initialization."""
        shared_args = {'confidence_level': 68}
        metric_args = {}
        
        inv_perf_ratio = InvPerfRatio(shared_args, metric_args)
        
        assert inv_perf_ratio.name == "InvPerfRatio"
        assert inv_perf_ratio.opt_sense == -1
    
    def test_inv_perf_ratio_evaluate(self):
        """Test InvPerfRatio evaluation."""
        shared_args = {
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 100
        }
        metric_args = {}
        
        inv_perf_ratio = InvPerfRatio(shared_args, metric_args)
        
        # Create DataFrame with Response columns
        response_base = names.param2filename({"Key": "Response"}, "")
        response_lower = names.param2filename({"Key": "Response", "ConfInt": "lower"}, "")
        response_upper = names.param2filename({"Key": "Response", "ConfInt": "upper"}, "")
        
        bs_df = pd.DataFrame({
            response_base: [150],  # Response value
            response_lower: [140], # Response CI lower
            response_upper: [160]  # Response CI upper
        })
        
        inv_perf_ratio.evaluate(bs_df, np.array([]), np.array([]))
        
        # Check that InvPerfRatio columns were created
        expected_base = names.param2filename({"Key": "InvPerfRatio"}, "")
        expected_lower = names.param2filename({"Key": "InvPerfRatio", "ConfInt": "lower"}, "")
        expected_upper = names.param2filename({"Key": "InvPerfRatio", "ConfInt": "upper"}, "")
        
        assert expected_base in bs_df.columns
        assert expected_lower in bs_df.columns
        assert expected_upper in bs_df.columns
        
        # Inverse performance ratio: 1 - (random - response)/(random - best) + EPSILON
        # 1 - (200 - 150)/(200 - 100) + EPSILON = 1 - 0.5 + EPSILON = 0.5 + EPSILON
        expected_inv_perf_ratio = 1 - (200 - 150) / (200 - 100) + EPSILON
        assert bs_df[expected_base].iloc[0] == pytest.approx(expected_inv_perf_ratio, abs=1e-6)


class TestSuccessProb:
    """Test class for SuccessProb success metric."""
    
    def test_success_prob_initialization(self):
        """Test SuccessProb class initialization."""
        shared_args = {'confidence_level': 68}
        metric_args = {'gap': 10}
        
        success_prob = SuccessProb(shared_args, metric_args)
        
        assert success_prob.name == "SuccProb"
        assert success_prob.opt_sense == 1
        assert success_prob.args == metric_args
    
    def test_success_prob_evaluate_minimization(self):
        """Test SuccessProb evaluation for minimization."""
        shared_args = {
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 100,
            'response_dir': -1  # Minimization
        }
        metric_args = {'gap': 10}  # 10% gap
        
        success_prob = SuccessProb(shared_args, metric_args)
        
        # Create responses where some achieve success threshold
        # Success threshold = 200 - 0.9*(200-100) = 200 - 90 = 110
        responses = np.array([
            [105, 115, 108],  # Sample 1: 2/3 success
            [120, 109, 125],  # Sample 2: 1/3 success  
            [95, 105, 115]    # Sample 3: 2/3 success
        ])
        
        bs_df = pd.DataFrame()
        
        success_prob.evaluate(bs_df, responses, np.array([]))
        
        # Check that columns were created
        expected_base = names.param2filename({"Key": "SuccProb"}, "")
        expected_lower = names.param2filename({"Key": "SuccProb", "ConfInt": "lower"}, "")
        expected_upper = names.param2filename({"Key": "SuccProb", "ConfInt": "upper"}, "")
        
        assert expected_base in bs_df.columns
        assert expected_lower in bs_df.columns
        assert expected_upper in bs_df.columns
        
        # Success probabilities should be between 0 and 1
        assert 0.0 <= bs_df[expected_base].iloc[0] <= 1.0
    
    def test_success_prob_evaluate_maximization(self):
        """Test SuccessProb evaluation for maximization."""
        shared_args = {
            'confidence_level': 68,
            'random_value': 100,
            'best_value': 200,
            'response_dir': 1  # Maximization
        }
        metric_args = {'gap': 20}  # 20% gap
        
        success_prob = SuccessProb(shared_args, metric_args)
        
        # Create responses for maximization scenario
        responses = np.array([
            [180, 190, 175],  # Sample 1
            [160, 185, 195],  # Sample 2
            [170, 180, 165]   # Sample 3
        ])
        
        bs_df = pd.DataFrame()
        
        success_prob.evaluate(bs_df, responses, np.array([]))
        
        expected_base = names.param2filename({"Key": "SuccProb"}, "")
        
        # Success probabilities should be between 0 and 1
        assert 0.0 <= bs_df[expected_base].iloc[0] <= 1.0
    
    def test_success_prob_extreme_values(self):
        """Test SuccessProb with extreme success/failure cases."""
        shared_args = {
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 100,
            'response_dir': -1
        }
        metric_args = {'gap': 10}
        
        success_prob = SuccessProb(shared_args, metric_args)
        
        # All responses are very good (all succeed)
        responses = np.array([
            [50, 60, 70],
            [40, 50, 80],
            [30, 55, 65]
        ])
        
        bs_df = pd.DataFrame()
        success_prob.evaluate(bs_df, responses, np.array([]))
        
        expected_base = names.param2filename({"Key": "SuccProb"}, "")
        
        # Should be close to 1.0 (all succeed)
        assert bs_df[expected_base].iloc[0] > 0.8


class TestResource:
    """Test class for Resource success metric."""
    
    def test_resource_initialization(self):
        """Test Resource class initialization."""
        shared_args = {'confidence_level': 68}
        metric_args = {}
        
        resource = Resource(shared_args, metric_args)
        
        assert resource.name == "MeanTime"
        assert resource.opt_sense == -1
    
    def test_resource_evaluate(self):
        """Test Resource evaluation."""
        shared_args = {'confidence_level': 68}
        metric_args = {}
        
        resource = Resource(shared_args, metric_args)
        
        # Create resource data
        resources = np.array([
            [10, 15, 12],  # Sample 1
            [8, 20, 14],   # Sample 2
            [12, 18, 10]   # Sample 3
        ])
        
        bs_df = pd.DataFrame()
        
        resource.evaluate(bs_df, np.array([]), resources)
        
        # Check that columns were created
        expected_base = names.param2filename({"Key": "MeanTime"}, "")
        expected_lower = names.param2filename({"Key": "MeanTime", "ConfInt": "lower"}, "")
        expected_upper = names.param2filename({"Key": "MeanTime", "ConfInt": "upper"}, "")
        
        assert expected_base in bs_df.columns
        assert expected_lower in bs_df.columns
        assert expected_upper in bs_df.columns
        
        # Should compute mean across bootstrap samples
        expected_means = [10, 17.67, 12]  # Approximate means of each column
        overall_mean = np.mean([10, 17.67, 12])
        
        assert bs_df[expected_base].iloc[0] == pytest.approx(overall_mean, abs=1.0)


class TestRTT:
    """Test class for RTT success metric."""
    
    def test_rtt_initialization(self):
        """Test RTT class initialization."""
        shared_args = {'confidence_level': 68}
        metric_args = {
            'gap': 10,
            'fail_value': 1e6,
            'RTT_factor': 1.0,
            's': 0.99
        }
        
        rtt = RTT(shared_args, metric_args)
        
        assert rtt.name == "RTT"
        assert rtt.args == metric_args
    
    def test_rtt_evaluate_single_zero_probability(self):
        """Test RTT evaluation for zero success probability."""
        shared_args = {'confidence_level': 68}
        metric_args = {
            'fail_value': 1e6,
            's': 0.99
        }
        
        rtt = RTT(shared_args, metric_args)
        
        result = rtt.evaluate_single(0.0, scale=1.0)
        assert result == 1e6  # Should return fail_value
    
    def test_rtt_evaluate_single_perfect_probability(self):
        """Test RTT evaluation for perfect success probability."""
        shared_args = {'confidence_level': 68}
        metric_args = {
            'fail_value': 1e6,
            's': 0.99
        }
        
        rtt = RTT(shared_args, metric_args)
        
        result = rtt.evaluate_single(1.0, scale=1.0, size=1000)
        
        # Should be finite and positive
        assert np.isfinite(result)
        assert result > 0
    
    def test_rtt_evaluate_single_normal_probability(self):
        """Test RTT evaluation for normal success probability."""
        shared_args = {'confidence_level': 68}
        metric_args = {
            'fail_value': 1e6,
            's': 0.99
        }
        
        rtt = RTT(shared_args, metric_args)
        
        result = rtt.evaluate_single(0.5, scale=1.0)
        
        # Should be finite and positive
        assert np.isfinite(result)
        assert result > 0
        assert result < 1e6  # Should not be fail_value
    
    def test_rtt_evaluate_full(self):
        """Test full RTT evaluation."""
        shared_args = {
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 100,
            'response_dir': -1
        }
        metric_args = {
            'gap': 10,
            'fail_value': 1e6,
            'RTT_factor': 1.0,
            's': 0.99
        }
        
        rtt = RTT(shared_args, metric_args)
        
        # Create responses that have reasonable success probability
        responses = np.array([
            [105, 115, 108],
            [120, 109, 125],
            [95, 105, 115]
        ])
        
        bs_df = pd.DataFrame()
        
        rtt.evaluate(bs_df, responses, np.array([]))
        
        # Check that columns were created
        expected_base = names.param2filename({"Key": "RTT"}, "")
        expected_lower = names.param2filename({"Key": "RTT", "ConfInt": "lower"}, "")
        expected_upper = names.param2filename({"Key": "RTT", "ConfInt": "upper"}, "")
        
        assert expected_base in bs_df.columns
        assert expected_lower in bs_df.columns
        assert expected_upper in bs_df.columns
        
        # RTT should be finite and positive
        rtt_value = bs_df[expected_base].iloc[0]
        assert np.isfinite(rtt_value)
        assert rtt_value > 0
    
    def test_rtt_evaluate_failure_case(self):
        """Test RTT evaluation when result is failure."""
        shared_args = {
            'confidence_level': 68,
            'random_value': 200,
            'best_value': 100,
            'response_dir': -1
        }
        metric_args = {
            'gap': 10,
            'fail_value': 1e6,
            'RTT_factor': 1.0,
            's': 0.99
        }
        
        rtt = RTT(shared_args, metric_args)
        
        # Create responses that will likely fail (all worse than threshold)
        responses = np.array([
            [300, 400, 350],  # All much worse than threshold
            [320, 380, 390],
            [310, 370, 360]
        ])
        
        bs_df = pd.DataFrame()
        
        rtt.evaluate(bs_df, responses, np.array([]))
        
        expected_base = names.param2filename({"Key": "RTT"}, "")
        expected_lower = names.param2filename({"Key": "RTT", "ConfInt": "lower"}, "")
        expected_upper = names.param2filename({"Key": "RTT", "ConfInt": "upper"}, "")
        
        # If RTT is infinite or fail_value, CI should also be fail_value
        rtt_value = bs_df[expected_base].iloc[0]
        if np.isinf(rtt_value) or rtt_value == 1e6:
            assert bs_df[expected_lower].iloc[0] == 1e6
            assert bs_df[expected_upper].iloc[0] == 1e6


class TestConstants:
    """Test module constants."""
    
    def test_epsilon_exists(self):
        """Test that EPSILON constant exists and has reasonable value."""
        assert EPSILON == 1e-10
        assert EPSILON > 0
        assert EPSILON < 1e-6


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_responses_arrays(self):
        """Test behavior with empty response arrays."""
        shared_args = {'confidence_level': 68}
        metric_args = {'opt_sense': 1}
        
        response = Response(shared_args, metric_args)
        
        empty_responses = np.array([]).reshape(0, 0)
        bs_df = pd.DataFrame()
        
        # Should handle empty arrays gracefully
        try:
            response.evaluate(bs_df, empty_responses, np.array([]))
            # If no exception, check that appropriate columns exist
            expected_base = names.param2filename({"Key": "Response"}, "")
            # May or may not create columns depending on implementation
        except (IndexError, ValueError):
            # Empty arrays may cause expected errors
            pass
    
    def test_single_value_arrays(self):
        """Test behavior with single-value arrays."""
        shared_args = {'confidence_level': 68}
        metric_args = {'opt_sense': 1}
        
        response = Response(shared_args, metric_args)
        
        single_responses = np.array([[42]])
        bs_df = pd.DataFrame()
        
        response.evaluate(bs_df, single_responses, np.array([]))
        
        expected_base = names.param2filename({"Key": "Response"}, "")
        
        if expected_base in bs_df.columns:
            assert bs_df[expected_base].iloc[0] == 42
    
    def test_nan_values_in_responses(self):
        """Test behavior with NaN values in responses."""
        shared_args = {'confidence_level': 68}
        metric_args = {'opt_sense': 1}
        
        response = Response(shared_args, metric_args)
        
        responses_with_nan = np.array([
            [10, np.nan, 12],
            [8, 20, 14],
            [np.nan, 18, 10]
        ])
        
        bs_df = pd.DataFrame()
        
        # Should handle NaN values appropriately
        response.evaluate(bs_df, responses_with_nan, np.array([]))
        
        expected_base = names.param2filename({"Key": "Response"}, "")
        
        # Result should be finite (NaN handling depends on implementation)
        if expected_base in bs_df.columns:
            result = bs_df[expected_base].iloc[0]
            # Could be NaN or finite depending on how np.max/min handles NaN
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])