#!/bin/bash
# Script to set up and test the stochastic-benchmark package locally
# This replicates the CI environment for local testing

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Stochastic Benchmark CI Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print colored messages
print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Get Python version from argument or use default
PYTHON_VERSION="${1:-3.10}"
ENV_NAME="stochastic-benchmark-ci-py${PYTHON_VERSION//./}"

echo -e "Python version: ${BLUE}${PYTHON_VERSION}${NC}"
echo -e "Environment name: ${BLUE}${ENV_NAME}${NC}"
echo ""

# Create conda environment
print_step "Creating conda environment: ${ENV_NAME}"
conda env create -f environment-ci.yml -n "${ENV_NAME}" python="${PYTHON_VERSION}" || {
    print_warning "Environment already exists. Updating instead..."
    conda env update -f environment-ci.yml -n "${ENV_NAME}"
}

# Activate environment (note: this won't persist outside the script)
print_step "Activating environment"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Verify Python version
ACTUAL_PYTHON=$(python --version)
print_step "Python version: ${ACTUAL_PYTHON}"

# Install package in development mode
print_step "Installing package in development mode"
pip install -e .

# Set PYTHONPATH (matches CI setup)
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
print_step "PYTHONPATH set to include src directory"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Environment Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "To activate the environment, run:"
echo -e "  ${BLUE}conda activate ${ENV_NAME}${NC}"
echo ""
echo -e "To run tests (like CI does), use:"
echo -e "  ${BLUE}./run-ci-tests.sh${NC}"
echo ""
echo -e "Or run tests manually:"
echo -e "  ${BLUE}export PYTHONPATH=\"\${PYTHONPATH}:\${PWD}/src\"${NC}"
echo -e "  ${BLUE}pytest tests/ -v --cov=src --cov-report=xml --cov-report=html --cov-report=term${NC}"
echo ""
