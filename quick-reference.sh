#!/bin/bash
# Quick Reference Card for CI Testing
# Print this for easy reference

cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║                    STOCHASTIC-BENCHMARK CI TESTING                        ║
║                           Quick Reference Card                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

📋 INITIAL SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Create CI environment (Python 3.10 default)
  ./setup-ci-env.sh
  
  # Or specify Python version
  ./setup-ci-env.sh 3.11  # Python 3.11
  ./setup-ci-env.sh 3.12  # Python 3.12

🧪 RUNNING TESTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Activate environment
  conda activate stochastic-benchmark-ci-py310
  
  # Run full CI test suite
  ./run-ci-tests.sh
  
  # Run specific tests
  export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
  pytest tests/test_bootstrap.py -v
  
  # Run with parallel execution
  pytest tests/ -n auto

🔄 TEST ALL PYTHON VERSIONS (3.10, 3.11, 3.12)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Test all versions automatically
  ./test-all-python-versions.sh

📊 VIEW COVERAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # HTML report (open in browser)
  firefox htmlcov/index.html
  
  # Terminal report
  coverage report

🔍 LINTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Check for critical errors
  flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
  
  # Full style check
  flake8 src --max-line-length=120 --statistics

🧹 CLEANUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Remove environment
  conda deactivate
  conda env remove -n stochastic-benchmark-ci-py310
  
  # Remove coverage reports
  rm -rf htmlcov/ .coverage coverage.xml .pytest_cache/

📚 ENVIRONMENT NAMES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Python 3.10 → stochastic-benchmark-ci-py310
  Python 3.11 → stochastic-benchmark-ci-py311
  Python 3.12 → stochastic-benchmark-ci-py312

💡 TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Always set PYTHONPATH before running manual pytest commands
  • Use run-ci-tests.sh to replicate CI exactly
  • Create all Python version environments to match CI matrix
  • Check CI-TESTING.md for detailed documentation

📄 FILES CREATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • environment-ci.yml          → Conda environment specification
  • setup-ci-env.sh             → Environment setup script
  • run-ci-tests.sh             → Run full CI test suite
  • test-all-python-versions.sh → Test all Python versions
  • CI-TESTING.md               → Complete documentation
  • quick-reference.sh          → This reference card

═══════════════════════════════════════════════════════════════════════════
EOF
