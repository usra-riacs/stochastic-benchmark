#!/bin/bash
# Script to test all Python versions that CI uses
# This helps verify compatibility across Python 3.10, 3.11, and 3.12

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PYTHON_VERSIONS=("3.10" "3.11" "3.12")
FAILED_VERSIONS=()
PASSED_VERSIONS=()

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing All Python Versions (Like CI)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for PY_VERSION in "${PYTHON_VERSIONS[@]}"; do
    ENV_NAME="stochastic-benchmark-ci-py${PY_VERSION//./}"
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing Python ${PY_VERSION}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    
    # Check if environment exists
    if ! conda env list | grep -q "^${ENV_NAME} "; then
        echo -e "${YELLOW}Environment ${ENV_NAME} not found. Creating it...${NC}"
        ./setup-ci-env.sh "${PY_VERSION}"
    fi
    
    # Activate and test
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    
    echo -e "Testing with environment: ${BLUE}${ENV_NAME}${NC}"
    echo ""
    
    if ./run-ci-tests.sh; then
        echo -e "${GREEN}✓ Python ${PY_VERSION} tests PASSED${NC}"
        PASSED_VERSIONS+=("${PY_VERSION}")
    else
        echo -e "${RED}✗ Python ${PY_VERSION} tests FAILED${NC}"
        FAILED_VERSIONS+=("${PY_VERSION}")
    fi
    
    conda deactivate
    echo ""
    echo ""
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${GREEN}Passed (${#PASSED_VERSIONS[@]}/${#PYTHON_VERSIONS[@]}):${NC}"
for version in "${PASSED_VERSIONS[@]}"; do
    echo -e "  ✓ Python ${version}"
done

if [ ${#FAILED_VERSIONS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed (${#FAILED_VERSIONS[@]}/${#PYTHON_VERSIONS[@]}):${NC}"
    for version in "${FAILED_VERSIONS[@]}"; do
        echo -e "  ✗ Python ${version}"
    done
    echo ""
    exit 1
else
    echo ""
    echo -e "${GREEN}All Python versions passed! 🎉${NC}"
    echo ""
fi
