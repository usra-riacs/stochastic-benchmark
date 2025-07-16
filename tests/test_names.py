import pytest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock

# Import the module to test
import sys
sys.path.insert(0, '/home/runner/work/stochastic-benchmark/stochastic-benchmark/src')

from names import paths, param2filename, filename2param


class TestPaths:
    """Test class for the paths utility class."""

    def test_paths_initialization(self):
        """Test paths class initialization and directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            p = paths(temp_dir)
            
            # Test basic attributes
            assert p.cwd == temp_dir
            assert p.raw_data == os.path.join(temp_dir, "exp_raw")
            assert p.checkpoints == os.path.join(temp_dir, "checkpoints")
            assert p.plots == os.path.join(temp_dir, "plots")
            assert p.progress == os.path.join(temp_dir, "progress")
            
            # Test directory creation
            assert os.path.exists(p.checkpoints)
            assert os.path.exists(p.plots)
            assert os.path.exists(p.progress)
    
    def test_paths_checkpoint_files(self):
        """Test checkpoint file paths are correctly set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            p = paths(temp_dir)
            
            expected_checkpoints = os.path.join(temp_dir, "checkpoints")
            assert p.bootstrap == os.path.join(expected_checkpoints, "bootstrapped_results.pkl")
            assert p.interpolate == os.path.join(expected_checkpoints, "interpolated_results.pkl")
            assert p.training_stats == os.path.join(expected_checkpoints, "training_stats.pkl")
            assert p.testing_stats == os.path.join(expected_checkpoints, "testing_stats.pkl")
    
    def test_paths_virtual_best_files(self):
        """Test virtual best file paths are correctly set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            p = paths(temp_dir)
            
            expected_checkpoints = os.path.join(temp_dir, "checkpoints")
            assert p.virtual_best["train"] == os.path.join(expected_checkpoints, "vb_train.pkl")
            assert p.virtual_best["test"] == os.path.join(expected_checkpoints, "vb_test.pkl")
    
    def test_paths_best_rec_files(self):
        """Test best recommendation file paths are correctly set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            p = paths(temp_dir)
            
            expected_checkpoints = os.path.join(temp_dir, "checkpoints")
            assert p.best_rec["stats"] == os.path.join(expected_checkpoints, "br_stats.pkl")
            assert p.best_rec["results"] == os.path.join(expected_checkpoints, "br_results.pkl")
    
    def test_paths_projections_files(self):
        """Test projection file paths are correctly set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            p = paths(temp_dir)
            
            expected_checkpoints = os.path.join(temp_dir, "checkpoints")
            assert p.projections["stats"] == os.path.join(expected_checkpoints, "proj_stats.pkl")
            assert p.projections["results"] == os.path.join(expected_checkpoints, "proj_results.pkl")
    
    def test_paths_additional_files(self):
        """Test additional file paths are correctly set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            p = paths(temp_dir)
            
            expected_checkpoints = os.path.join(temp_dir, "checkpoints")
            assert p.best_agg_alloc == os.path.join(expected_checkpoints, "best_agg_alloc.pkl")
            assert p.train_exp_at_best == os.path.join(expected_checkpoints, "train_exp_at_best.pkl")
            assert p.final_values == os.path.join(expected_checkpoints, "final_values.pkl")
            assert p.test_exp_at_best == os.path.join(expected_checkpoints, "test_exp_at_best.pkl")
            assert p.seq_exp_values == os.path.join(expected_checkpoints, "seq_exp_values.pkl")

    def test_paths_with_existing_directories(self):
        """Test paths class when directories already exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Pre-create directories
            checkpoints_dir = os.path.join(temp_dir, "checkpoints")
            plots_dir = os.path.join(temp_dir, "plots")
            progress_dir = os.path.join(temp_dir, "progress")
            
            os.makedirs(checkpoints_dir)
            os.makedirs(plots_dir)
            os.makedirs(progress_dir)
            
            # Test that paths class doesn't fail when directories exist
            p = paths(temp_dir)
            
            assert os.path.exists(p.checkpoints)
            assert os.path.exists(p.plots)
            assert os.path.exists(p.progress)


class TestParam2Filename:
    """Test class for param2filename function."""
    
    def test_basic_param_dict(self):
        """Test basic parameter dictionary to filename conversion."""
        param_dict = {"alpha": 0.5, "beta": 10, "gamma": "test"}
        result = param2filename(param_dict, ".pkl")
        
        # Keys should be sorted, so alpha comes first
        expected = "alpha=0.5_beta=10_gamma=test.pkl"
        assert result == expected
    
    def test_empty_param_dict(self):
        """Test empty parameter dictionary."""
        param_dict = {}
        result = param2filename(param_dict, ".txt")
        
        expected = ".txt"
        assert result == expected
    
    def test_single_param(self):
        """Test single parameter."""
        param_dict = {"learning_rate": 0.01}
        result = param2filename(param_dict, ".json")
        
        expected = "learning_rate=0.01.json"
        assert result == expected
    
    def test_ignore_params(self):
        """Test ignoring specific parameters."""
        param_dict = {"alpha": 0.5, "beta": 10, "gamma": "test"}
        result = param2filename(param_dict, ".pkl", ignore=["beta"])
        
        expected = "alpha=0.5_gamma=test.pkl"
        assert result == expected
    
    def test_ignore_all_params(self):
        """Test ignoring all parameters."""
        param_dict = {"alpha": 0.5, "beta": 10}
        result = param2filename(param_dict, ".pkl", ignore=["alpha", "beta"])
        
        expected = ".pkl"
        assert result == expected
    
    def test_numeric_params(self):
        """Test various numeric parameter types."""
        param_dict = {"int_val": 42, "float_val": 3.14159, "zero": 0}
        result = param2filename(param_dict, ".dat")
        
        expected = "float_val=3.14159_int_val=42_zero=0.dat"
        assert result == expected
    
    def test_special_characters_in_values(self):
        """Test parameter values with special characters."""
        param_dict = {"method": "sgd", "config": "test_config"}
        result = param2filename(param_dict, ".cfg")
        
        expected = "config=test_config_method=sgd.cfg"
        assert result == expected
    
    def test_boolean_params(self):
        """Test boolean parameter values."""
        param_dict = {"verbose": True, "debug": False}
        result = param2filename(param_dict, ".log")
        
        expected = "debug=False_verbose=True.log"
        assert result == expected


class TestFilename2Param:
    """Test class for filename2param function."""
    
    def test_basic_filename(self):
        """Test basic filename to parameter dictionary conversion."""
        filename = "alpha=0.5_beta=10_gamma=test.pkl"
        result = filename2param(filename)
        
        expected = {"alpha": "0.5", "beta": "10", "gamma": "test"}
        assert result == expected
    
    def test_filename_without_extension(self):
        """Test filename without extension."""
        filename = "alpha=0.5_beta=10"
        result = filename2param(filename)
        
        expected = {"alpha": "0.5", "beta": "10"}
        assert result == expected
    
    def test_single_param_filename(self):
        """Test single parameter filename."""
        filename = "learning_rate=0.01.json"
        result = filename2param(filename)
        
        expected = {"learning_rate": "0.01"}
        assert result == expected
    
    def test_complex_filename(self):
        """Test complex filename with multiple extensions."""
        filename = "method=sgd_lr=0.01_epochs=100.model.pkl"
        result = filename2param(filename)
        
        expected = {"method": "sgd", "lr": "0.01", "epochs": "100"}
        assert result == expected
    
    def test_filename_with_special_values(self):
        """Test filename with special values."""
        filename = "debug=False_verbose=True_method=adam.cfg"
        result = filename2param(filename)
        
        expected = {"debug": "False", "verbose": "True", "method": "adam"}
        assert result == expected
    
    def test_empty_filename_components(self):
        """Test that function handles malformed filenames gracefully."""
        # This should raise an error for malformed filenames
        with pytest.raises(ValueError):
            filename2param("invalid_filename_format.pkl")
    
    def test_roundtrip_conversion(self):
        """Test that param2filename and filename2param are inverses (with string conversion)."""
        original_dict = {"alpha": 0.5, "beta": 10, "method": "test"}
        
        # Convert to filename and back
        filename = param2filename(original_dict, ".pkl")
        reconstructed_dict = filename2param(filename)
        
        # Note: values become strings after reconstruction
        expected_dict = {"alpha": "0.5", "beta": "10", "method": "test"}
        assert reconstructed_dict == expected_dict
    
    def test_filename_with_equals_in_values(self):
        """Test filename with equals signs in values (should fail gracefully)."""
        # This tests edge case handling
        filename = "param1=value=with=equals_param2=normal.txt"
        with pytest.raises(ValueError):
            filename2param(filename)


if __name__ == "__main__":
    pytest.main([__file__])