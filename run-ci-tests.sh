#!/bin/bash
# Script to run tests exactly as the CI does
# This replicates the test steps from .github/workflows/ci.yml

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running CI Tests Locally${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Set PYTHONPATH to include src directory (matches CI)
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
print_step "PYTHONPATH set: ${PYTHONPATH}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src" ] || [ ! -d "tests" ]; then
    print_error "Please run this script from the repository root directory"
    exit 1
fi

# Step 1: Lint with flake8 (optional, matches CI)
print_step "Step 1: Linting with flake8"
echo ""
echo "Checking for syntax errors and undefined names..."
flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics || {
    print_error "Critical flake8 errors found!"
}

echo ""
echo "Checking code style (warnings only)..."
flake8 src --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics

echo ""

# Step 2: Run tests with pytest and coverage (matches CI)
print_step "Step 2: Running tests with pytest and coverage"
echo ""
pytest tests/ -v --cov=src --cov-report=xml --cov-report=html --cov-report=term

echo ""

# Step 3: Generate coverage report (matches CI)
print_step "Step 3: Coverage Summary"
echo ""
coverage report

echo ""

# Step 4: Run integration tests if they exist (matches CI)
print_step "Step 4: Running integration tests"
echo ""
if [ -d "tests/integration" ]; then
    pytest tests/integration/ -v
else
    echo "No integration tests found (this is OK)"
fi

echo ""

# Step 5: Smoke tests - ensure main modules import (matches CI)
print_step "Step 5: Running smoke tests"
echo ""
python -c "
import sys
sys.path.insert(0, 'src')
try:
    import stochastic_benchmark
    import bootstrap
    import plotting
    import stats
    import names
    print('✓ All main modules import successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All CI Tests Passed! ✓${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Coverage reports generated:"
echo -e "  - Terminal output (above)"
echo -e "  - XML: ${BLUE}coverage.xml${NC}"
echo -e "  - HTML: ${BLUE}htmlcov/index.html${NC}"
echo ""
echo -e "To view the HTML coverage report:"
echo -e "  ${BLUE}firefox htmlcov/index.html${NC}  # or your preferred browser"
echo ""
