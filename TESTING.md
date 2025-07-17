# Testing Infrastructure for Stochastic Benchmark

## Overview

This repository now includes comprehensive testing infrastructure to ensure code quality and reliability.

## Test Structure

### Unit Tests
- `tests/test_names.py` - Tests for naming utilities and file path management
- `tests/test_interpolate.py` - Tests for interpolation functionality
- `tests/test_df_utils.py` - Tests for DataFrame utilities and processing functions
- `tests/test_success_metrics.py` - Tests for success metric calculations
- `tests/test_bootstrap.py` - Tests for bootstrap sampling and resampling
- `tests/test_training.py` - Tests for training and parameter optimization
- `tests/test_smoke.py` - Basic smoke tests for all modules

### Integration Tests
- `tests/integration/test_module_integration.py` - Tests for cross-module functionality

## Running Tests

### Quick Start
```bash
# Run all tests
python run_tests.py all

# Run only unit tests
python run_tests.py unit

# Run with coverage report
python run_tests.py coverage

# Run smoke tests (basic functionality)
python run_tests.py smoke
```

### Using pytest directly
```bash
# Set PYTHONPATH and run tests
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_names.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## GitHub Actions CI/CD

The repository includes automated testing via GitHub Actions:

- **Matrix Testing**: Tests across Python versions 3.8, 3.9, 3.10, 3.11, 3.12
- **Linting**: Code quality checks with flake8
- **Coverage**: Automated coverage reporting via Codecov
- **Integration Tests**: Cross-module functionality verification

## Test Coverage

### Current Coverage Status
- ✅ **names.py** (103 lines) - Path management, parameter/filename conversion
- ✅ **interpolate.py** (203 lines) - Data interpolation and resource generation
- ✅ **df_utils.py** (284 lines) - DataFrame utilities and processing
- ✅ **success_metrics.py** (353 lines) - Success metric calculations (Response, PerfRatio, etc.)
- ✅ **bootstrap.py** (441 lines) - Bootstrap sampling and statistical methods
- ✅ **training.py** (328 lines) - Training algorithms and parameter optimization

### Modules Needing Tests
- 🔄 **plotting.py** (611 lines) - Plotting and visualization
- 🔄 **stochastic_benchmark.py** (1796 lines) - Main benchmark class
- 🔄 **utils_ws.py** (533 lines) - Utility functions
- 🔄 **cross_validation.py** (534 lines) - Cross-validation methods
- 🔄 **sequential_exploration.py** (388 lines) - Sequential exploration strategies
- 🔄 **random_exploration.py** (315 lines) - Random exploration methods

## Test Quality Standards

All tests follow these principles:

1. **Comprehensive Coverage**: Test initialization, core functionality, edge cases, and error conditions
2. **Isolation**: Tests are independent and can run in any order
3. **Mocking**: External dependencies are mocked appropriately
4. **Documentation**: Clear test names and docstrings explain what is being tested
5. **Assertions**: Meaningful assertions with appropriate tolerances for numerical tests

### Test Categories

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test interaction between modules
- **Smoke Tests**: Basic functionality and import tests
- **Edge Case Tests**: Boundary conditions and error handling

## Contributing Tests

When adding new functionality:

1. Add corresponding unit tests in the appropriate `test_*.py` file
2. Include edge cases and error conditions
3. Update integration tests if the change affects module interactions
4. Ensure tests pass locally before submitting PR
5. Maintain test coverage above 80%

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`  
- Test methods: `test_<functionality>_<scenario>`

### Example Test Structure
```python
class TestModuleName:
    """Test class for ModuleName functionality."""
    
    def test_function_basic(self):
        """Test basic functionality of function."""
        # Test implementation
        
    def test_function_edge_case(self):
        """Test edge case handling."""
        # Edge case test
        
    def test_function_error_conditions(self):
        """Test error condition handling."""
        # Error handling test
```

## Performance Testing

For performance-critical functions, consider adding performance benchmarks:

```python
@pytest.mark.slow
def test_bootstrap_performance(self):
    """Test bootstrap performance with large datasets."""
    # Performance test implementation
```

## Dependencies

Testing infrastructure requires:
- `pytest` - Main testing framework
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- Standard scientific Python stack (pandas, numpy, scipy)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the `src` directory
2. **Missing Dependencies**: Install required packages or skip tests with unavailable dependencies
3. **Slow Tests**: Use `-m "not slow"` to skip performance tests during development
4. **Path Issues**: Use absolute paths in tests when dealing with file I/O

### Running Specific Tests
```bash
# Run tests for a specific module
pytest tests/test_names.py

# Run a specific test class
pytest tests/test_names.py::TestPaths

# Run a specific test method
pytest tests/test_names.py::TestPaths::test_paths_initialization
```