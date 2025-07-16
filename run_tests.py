#!/usr/bin/env python3
"""
Test runner script for stochastic-benchmark package.

This script provides convenient commands to run different types of tests.
"""

import argparse
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run tests for stochastic-benchmark")
    parser.add_argument(
        "test_type", 
        choices=["unit", "integration", "smoke", "all", "coverage"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true", 
        help="Skip slow tests"
    )
    
    args = parser.parse_args()
    
    # Change to repository root
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    
    # Add src to PYTHONPATH
    env = os.environ.copy()
    src_path = os.path.join(repo_root, 'src')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = src_path
    
    success = True
    
    if args.test_type == "unit" or args.test_type == "all":
        cmd = ["python", "-m", "pytest", "tests/test_*.py"]
        if args.verbose:
            cmd.append("-v")
        if args.fast:
            cmd.extend(["-m", "not slow"])
        success &= run_command(cmd, "Unit tests")
    
    if args.test_type == "integration" or args.test_type == "all":
        cmd = ["python", "-m", "pytest", "tests/integration/"]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "Integration tests")
    
    if args.test_type == "smoke" or args.test_type == "all":
        cmd = ["python", "-m", "pytest", "tests/test_smoke.py"]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "Smoke tests")
    
    if args.test_type == "coverage":
        # Run tests with coverage
        cmd = [
            "python", "-m", "pytest", 
            "--cov=src", 
            "--cov-report=html", 
            "--cov-report=term",
            "--cov-report=xml",
            "tests/"
        ]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "Coverage tests")
        
        if success:
            print("\n📊 Coverage report generated in htmlcov/index.html")
    
    if success:
        print(f"\n🎉 All {args.test_type} tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Some {args.test_type} tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()