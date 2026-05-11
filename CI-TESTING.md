# Local CI Testing Setup

This directory contains files to replicate the GitHub Actions CI environment locally for testing the `stochastic-benchmark` package.

## Files Created

1. **`environment-ci.yml`** - Conda environment specification that matches CI dependencies
2. **`setup-ci-env.sh`** - Script to create and configure the conda environment
3. **`run-ci-tests.sh`** - Script to run tests exactly as the CI does

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Create environment with Python 3.10 (default)
./setup-ci-env.sh

# Or create environment with a specific Python version
./setup-ci-env.sh 3.11  # Python 3.11
./setup-ci-env.sh 3.12  # Python 3.12
```

### Option 2: Manual Setup

```bash
# Create the conda environment
conda env create -f environment-ci.yml

# Activate the environment
conda activate stochastic-benchmark-ci

# Install the package in development mode
pip install -e .
```

## Running Tests

Once the environment is set up and activated:

```bash
# Activate the environment (if not already active)
conda activate stochastic-benchmark-ci-py310  # or py311, py312

# Run all tests exactly as CI does
./run-ci-tests.sh
```

This script will:
1. ✅ Lint code with flake8
2. ✅ Run unit tests with pytest and coverage
3. ✅ Generate coverage reports (XML, HTML, terminal)
4. ✅ Run integration tests
5. ✅ Run smoke tests to verify module imports

### Manual Test Commands

If you prefer to run tests manually:

```bash
# Set PYTHONPATH (important!)
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=xml --cov-report=html --cov-report=term

# Run specific test files
pytest tests/test_bootstrap.py -v

# Run with parallel execution (faster)
pytest tests/ -n auto

# Run only integration tests
pytest tests/integration/ -v

# Lint the code
flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 src --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics
```

## Testing Multiple Python Versions (Like CI)

The CI tests against Python 3.10, 3.11, and 3.12. To replicate this locally:

```bash
# Create environments for each Python version
./setup-ci-env.sh 3.10
./setup-ci-env.sh 3.11
./setup-ci-env.sh 3.12

# Test each version
conda activate stochastic-benchmark-ci-py310
./run-ci-tests.sh

conda activate stochastic-benchmark-ci-py311
./run-ci-tests.sh

conda activate stochastic-benchmark-ci-py312
./run-ci-tests.sh
```

## Viewing Coverage Reports

After running tests, coverage reports are generated:

- **Terminal**: Displayed automatically after test run
- **HTML**: Open `htmlcov/index.html` in your browser
  ```bash
  firefox htmlcov/index.html  # or chrome, etc.
  ```
- **XML**: `coverage.xml` (for tools like codecov)

## Troubleshooting

### Import Errors

Make sure `PYTHONPATH` is set correctly:
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### Environment Already Exists

If you need to recreate an environment:
```bash
conda env remove -n stochastic-benchmark-ci-py310
./setup-ci-env.sh 3.10
```

### Dependency Issues

Update the environment with new dependencies:
```bash
conda env update -f environment-ci.yml -n stochastic-benchmark-ci-py310
```

## CI Workflow Comparison

| CI Step | Local Equivalent |
|---------|------------------|
| Install system dependencies | Handled by conda |
| Install Python dependencies | `conda env create -f environment-ci.yml` |
| Install package | `pip install -e .` |
| Lint with flake8 | Included in `run-ci-tests.sh` |
| Run tests with pytest | Included in `run-ci-tests.sh` |
| Generate coverage | Included in `run-ci-tests.sh` |
| Run integration tests | Included in `run-ci-tests.sh` |
| Smoke tests | Included in `run-ci-tests.sh` |

## Environment Details

The CI environment includes:

**Core Dependencies:**
- numpy >= 2.0
- pandas >= 2.3
- scipy >= 1.11
- matplotlib >= 3.7
- seaborn >= 0.13.2
- networkx >= 3.0
- tqdm >= 4.66

**Additional Dependencies:**
- cloudpickle >= 2.2
- dill >= 0.3.5
- hyperopt >= 0.2.7
- multiprocess >= 0.70.18
- munkres >= 1.1.4

**Testing Tools:**
- pytest
- pytest-cov
- pytest-xdist
- flake8
- coverage

## Cleaning Up

To remove the conda environment:
```bash
conda deactivate
conda env remove -n stochastic-benchmark-ci-py310
```

To remove coverage reports and test artifacts:
```bash
rm -rf htmlcov/ .coverage coverage.xml .pytest_cache/
```
