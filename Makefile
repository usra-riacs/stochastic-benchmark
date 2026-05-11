# Makefile for stochastic-benchmark CI testing
# Provides convenient commands for setting up and running CI tests locally

.PHONY: help setup-ci test lint coverage clean clean-all test-all quick-ref

# Default Python version for CI environment
PYTHON_VERSION ?= 3.10
ENV_NAME = stochastic-benchmark-ci-py$(subst .,,$(PYTHON_VERSION))

help:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Stochastic Benchmark - CI Testing Makefile"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  make setup-ci         Create CI conda environment (Python $(PYTHON_VERSION))"
	@echo "  make test             Run full CI test suite"
	@echo "  make test-all         Test all Python versions (3.10, 3.11, 3.12)"
	@echo "  make lint             Run flake8 linting"
	@echo "  make coverage         Generate coverage report"
	@echo "  make clean            Remove coverage reports and cache"
	@echo "  make clean-all        Remove environments and all generated files"
	@echo "  make quick-ref        Display quick reference card"
	@echo ""
	@echo "Environment management:"
	@echo "  make setup-ci PYTHON_VERSION=3.11   Setup with Python 3.11"
	@echo "  make setup-ci PYTHON_VERSION=3.12   Setup with Python 3.12"
	@echo ""
	@echo "Current settings:"
	@echo "  Python version: $(PYTHON_VERSION)"
	@echo "  Environment name: $(ENV_NAME)"
	@echo ""

setup-ci:
	@echo "Creating CI environment with Python $(PYTHON_VERSION)..."
	./setup-ci-env.sh $(PYTHON_VERSION)

test:
	@echo "Running CI tests..."
	@if ! conda env list | grep -q "$(ENV_NAME)"; then \
		echo "Environment $(ENV_NAME) not found. Run 'make setup-ci' first."; \
		exit 1; \
	fi
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(ENV_NAME) && ./run-ci-tests.sh"

test-all:
	@echo "Testing all Python versions..."
	./test-all-python-versions.sh

lint:
	@echo "Running flake8 linting..."
	@which flake8 > /dev/null || (echo "flake8 not found. Run 'make setup-ci' first." && exit 1)
	flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics

coverage:
	@echo "Generating coverage report..."
	@export PYTHONPATH="$${PYTHONPATH}:$${PWD}/src" && \
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo ""
	@echo "HTML coverage report: htmlcov/index.html"

clean:
	@echo "Cleaning coverage reports and cache..."
	rm -rf htmlcov/ .coverage coverage.xml .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done!"

clean-all: clean
	@echo "Removing all CI environments..."
	@for py_ver in 310 311 312; do \
		env_name="stochastic-benchmark-ci-py$$py_ver"; \
		if conda env list | grep -q "$$env_name"; then \
			echo "Removing $$env_name..."; \
			conda env remove -n "$$env_name" -y; \
		fi; \
	done
	@echo "All environments removed!"

quick-ref:
	@./quick-reference.sh

# Shorthand aliases
test-py310:
	@$(MAKE) setup-ci PYTHON_VERSION=3.10
	@$(MAKE) test PYTHON_VERSION=3.10

test-py311:
	@$(MAKE) setup-ci PYTHON_VERSION=3.11
	@$(MAKE) test PYTHON_VERSION=3.11

test-py312:
	@$(MAKE) setup-ci PYTHON_VERSION=3.12
	@$(MAKE) test PYTHON_VERSION=3.12

.DEFAULT_GOAL := help
