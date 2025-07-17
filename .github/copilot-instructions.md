# GitHub Copilot Instructions for stochastic-benchmark

## Repository Overview

The `stochastic-benchmark` repository is a Python package for benchmarking and analyzing stochastic optimization algorithms. It provides comprehensive tools for bootstrap sampling, statistical analysis, and result visualization.

## Directory Structure

```
stochastic-benchmark/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                 # GitHub Actions CI/CD pipeline
│   ├── copilot-instructions.md    # This file
│   └── copilot-setup-steps.yml    # Copilot setup configuration
├── src/                           # Main source code directory
│   ├── __init__.py
│   ├── bootstrap.py               # Bootstrap sampling and resampling methods
│   ├── cross_validation.py        # Cross-validation utilities
│   ├── df_utils.py               # DataFrame manipulation utilities
│   ├── interpolate.py            # Data interpolation and resource generation
│   ├── names.py                  # Path management and filename utilities
│   ├── plotting.py               # Visualization and plotting functions
│   ├── random_exploration.py     # Random exploration algorithms
│   ├── sequential_exploration.py # Sequential exploration algorithms
│   ├── stats.py                  # Statistical analysis and metrics
│   ├── stochastic_benchmark.py   # Main benchmarking framework
│   ├── success_metrics.py        # Success metric calculations
│   ├── training.py               # Training algorithms and optimization
│   └── utils_ws.py               # Workspace utilities
├── tests/                        # Comprehensive test suite
│   ├── integration/              # Integration tests
│   │   └── test_module_integration.py
│   ├── test_bootstrap.py         # Bootstrap module tests
│   ├── test_df_utils.py          # DataFrame utilities tests
│   ├── test_interpolate.py       # Interpolation tests
│   ├── test_names.py             # Names/path utilities tests
│   ├── test_stats.py             # Statistics tests
│   ├── test_success_metrics.py   # Success metrics tests
│   └── test_training.py          # Training algorithms tests
├── examples/                     # Example usage and tutorials
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration and build settings
├── run_tests.py                 # Test runner script
├── TESTING.md                   # Testing guidelines and documentation
└── README.md                    # Project documentation
```

## Code Standards and Guidelines

### Python Version Compatibility
- **Minimum Python version**: 3.9
- **Tested versions**: 3.9, 3.10, 3.11, 3.12
- **Type hints**: Use `from typing import List, Dict, DefaultDict, ...` for compatibility
- **Avoid**: Modern syntax like `list[str]` or `dict[str, int]` (use `List[str]`, `Dict[str, int]`)

### Code Quality Standards
- **Linting**: Code is linted with flake8 (max line length: 120 characters)
- **Type annotations**: All function parameters and return types should be annotated
- **Docstrings**: Use NumPy-style docstrings for all classes and functions
- **Error handling**: Implement proper exception handling with meaningful error messages

### Testing Requirements

#### Test Coverage Expectations
- **Unit tests**: Cover all public methods and functions
- **Integration tests**: Test cross-module functionality
- **Edge cases**: Handle empty inputs, boundary conditions, and error states
- **Mocking**: Use `unittest.mock` for external dependencies and multiprocessing

#### Test Patterns
```python
# Standard test class structure
class TestModuleName:
    """Test class for module functionality."""
    
    def test_function_basic(self):
        """Test basic functionality."""
        # Setup
        input_data = create_test_data()
        expected = expected_result()
        
        # Execute
        result = module_function(input_data)
        
        # Assert
        assert result == expected
        assert isinstance(result, expected_type)
```

#### Mocking Guidelines
```python
# For multiprocessing functions
@patch('module.Pool')
def test_multiprocess_function(self, mock_pool):
    mock_pool.return_value.__enter__.return_value.map.return_value = [expected_result]
    result = function_using_pool()
    assert result == expected_output

# For success metrics evaluation
@patch.object(SuccessMetric, 'evaluate')
def test_bootstrap_with_metrics(self, mock_evaluate):
    def mock_evaluate_func(df, responses, resources):
        df['Key=Metric'] = [test_value]
        df['ConfInt=lower_Key=Metric'] = [lower_value]
        df['ConfInt=upper_Key=Metric'] = [upper_value]
    mock_evaluate.side_effect = mock_evaluate_func
```

### Module-Specific Guidelines

#### Bootstrap Module (`bootstrap.py`)
- **Key classes**: `BootstrapParameters`, `BSParams_iter`, `BSParams_range_iter`
- **Main functions**: `BootstrapSingle`, `Bootstrap`, `Bootstrap_reduce_mem`
- **Testing notes**: Mock `initBootstrap` and success metrics; use proper column naming
- **Multiprocessing**: Functions use local functions that require careful mocking

#### Statistics Module (`stats.py`)
- **Key classes**: `StatsParameters`, `Mean`, `Median`
- **Main functions**: `StatsSingle`, `Stats`, `applyBounds`
- **Testing notes**: Requires multiple rows of data (single row returns empty DataFrame)
- **Column naming**: Uses `names.param2filename` convention

#### Success Metrics (`success_metrics.py`)
- **Key classes**: `Response`, `PerfRatio`, `SuccessProb`, `Resource`
- **Testing notes**: Mock `evaluate` methods to populate DataFrames with correct column names
- **Column format**: `Key=MetricName`, `ConfInt=lower_Key=MetricName`, `ConfInt=upper_Key=MetricName`

#### Names Module (`names.py`)
- **Main functions**: `param2filename`, `filename2param`, `parseDir`
- **Testing notes**: Test parameter-to-filename conversion and directory parsing
- **File paths**: Handle both relative and absolute paths correctly

## Common Development Tasks

### Adding New Tests
1. **Create test file**: Follow naming convention `test_module_name.py`
2. **Test structure**: Use class-based organization with descriptive method names
3. **Setup/teardown**: Use pytest fixtures for complex setup
4. **Assertions**: Prefer specific assertions over generic `assert True`

### Working with DataFrames
- **Empty checks**: Always check `len(df) > 0` before processing
- **Column validation**: Verify expected columns exist before accessing
- **Mock data**: Create realistic test DataFrames with appropriate dtypes

### Multiprocessing Code
- **Testing**: Always mock `Pool` at the module level (`@patch('module.Pool')`)
- **Local functions**: Avoid local functions in multiprocessing contexts (pickle issues)
- **Error handling**: Implement proper exception handling for process failures

### Performance Considerations
- **Large datasets**: Use sampling or mocking for performance-critical tests
- **Memory usage**: Monitor memory consumption in tests with large data
- **Timeout**: Set appropriate timeouts for long-running operations

## Debugging Common Issues

### Test Failures
1. **Import errors**: Check PYTHONPATH and module imports
2. **Mock issues**: Verify mock targets use correct module paths
3. **Empty DataFrames**: Ensure test data has multiple rows for statistics
4. **Column name errors**: Use `names.param2filename` for consistent naming

### Type Annotation Issues
- **Import error**: Add missing imports from `typing` module
- **Compatibility**: Use `List`, `Dict`, `DefaultDict` instead of built-in generics
- **Union types**: Use `Union[type1, type2]` for multiple possible types

### CI/CD Issues
- **Dependency conflicts**: Update `requirements.txt` and CI configuration
- **Platform differences**: Test on Ubuntu environment matching CI
- **Coverage failures**: Ensure tests cover all code paths

## Best Practices Summary

1. **Follow existing code patterns** in the repository
2. **Write comprehensive tests** before implementing features
3. **Use proper type annotations** for all new code
4. **Mock external dependencies** appropriately in tests
5. **Handle edge cases** and error conditions
6. **Maintain backward compatibility** with existing APIs
7. **Document complex algorithms** with clear comments
8. **Test on multiple Python versions** (3.9-3.12)
9. **Keep functions focused** with single responsibilities
10. **Use descriptive variable names** and function signatures

## Getting Help

- **Test execution**: Use `python run_tests.py` for comprehensive testing
- **Documentation**: Refer to `TESTING.md` for detailed testing guidelines
- **Examples**: Check `examples/` directory for usage patterns
- **CI logs**: Review GitHub Actions output for build failures

This repository maintains high standards for code quality, test coverage, and documentation. When contributing, ensure all tests pass and follow the established patterns for consistency and maintainability.