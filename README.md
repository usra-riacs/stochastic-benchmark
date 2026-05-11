# Window Sticker - Stochastic Benchmark

[![CI](https://github.com/bernalde/stochastic-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/bernalde/stochastic-benchmark/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bernalde/stochastic-benchmark/branch/main/graph/badge.svg)](https://codecov.io/gh/bernalde/stochastic-benchmark)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/github/license/bernalde/stochastic-benchmark)](LICENSE)

Repository for Stochastic Optimization Solvers Benchmark implementation of the Window Sticker framework.

The benchmarking approach is described in this [preprint](https://arxiv.org/abs/2402.10255) titled: *Benchmarking the Operation of Quantum Heuristics and Ising Machines: Scoring Parameter Setting Strategies on Optimization Applications*.

Details of the implementation and an illustrative example for Wishart instances found [here](examples/wishart_n_50_alpha_0.5/wishart_n_50_alpha_0.50.ipynb) are given in this [document](stochastic-benchmarking-notes.pdf).

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)
- [Testing](#testing)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Background
This code has been created in order to produce a set of plots that inform the performance of parameterized stochastic optimization solvers when addressing a well-established family of optimization problems.
These plots are produced based on experimental data from the execution of such solvers in seen instances of the problem family and evaluated further in an unseen subset of problems.
More details of the methodology have been presented in the [APS March meeting](https://meetings.aps.org/Meeting/MAR22/Session/F38.5) and [INFORMS Annual meeting](https://www.abstractsonline.com/pp8/#!/10693/presentation/8455) conferences.
A manuscript explaining the methodology is in preparation.
The performance plot, or as we like to call it *Window Sticker*, is a graphical representation of the expected performance of a solution method or parameter setting strategy with an unseen instance from the same problem family that it is generated aiming to answer the question With X% confidence, will we find a solution with Y quality after using R resource?
Consider that the quality metric and the resource values can be arbitrary functions of the parameters and performance of the given solver, providing a flexible analysis tool for its performance.

The current package implements the following functionality:
- Parsing results from files from parameterized stochastic solvers such as PySA and D-Wave ocean tools.
- Through bootstrapping and downsampling, simulate the lower data performance for such solvers.
- Compute best-recommended parameters based on aggregated statistics and individual results for each parameter setting.
- Compute optimistic bound performance, known as virtual best performance, based on the provided experiments.
- Perform an exploration-exploitation parameter setting strategy, where the fraction of the allocated resources used in the exploration round is optimized. The exploration procedure is implemented as a random search in the seen parameter settings or a Bayesian-based method known as the tree of parzen and implemented in the package [Hyperopt](https://hyperopt.github.io/hyperopt/).
- Plot the Window sticker, comparing the performance curves corresponding to the virtual best, recommended parameters, and exploration-exploitation parameter setting strategies.
- Plots the values of the parameters and their best values with respect to the resource considered, a plot we call the Strategy plot. These plots can show the actual solver parameter values or the meta-parameters associated with parameter-setting strategies.

## Installation

### Method 1: Cloning the Repository

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/usra-riacs/stochastic-benchmark.git
    cd stochastic-benchmark
    ```

2. **Set up a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Optional: Install Example Dependencies** (needed for some example notebooks):
    ```bash
    pip install -r requirements-examples.txt
    ```

### Method 2: Downloading as a Zip Archive

1. **Download the Repository**:
    - Navigate to the [stochastic-benchmark GitHub page](https://github.com/usra-riacs/stochastic-benchmark).
    - Click on the `Code` button.
    - Choose `Download ZIP`.
    - Once downloaded, extract the ZIP archive and navigate to the extracted folder in your terminal or command prompt.

2. **Set up a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Optional: Install Example Dependencies** (needed for some example notebooks):
    ```bash
    pip install -r requirements-examples.txt
    ```

<!-- the following `pip` command can install this package -->

<!-- ``pip install -i https://test.pypi.org/simple/ stochastic-benchmark==0.1.0`` -->

## Examples

For a full demonstration of the stochastic-benchmark analysis in action, refer to the example notebooks located in the `examples` folder of this repository.

## Documentation

Use the root README as the entry point, then follow the focused documents for the part of the workflow you need:

- [examples/general_workflow.md](examples/general_workflow.md) for the end-to-end benchmark flow
- [CI-TESTING.md](CI-TESTING.md) for local CI reproduction and environment setup
- [TESTING.md](TESTING.md) for the test suite overview
- [examples/wishart_n_50_alpha_0.5/README.md](examples/wishart_n_50_alpha_0.5/README.md) for the Wishart example details

## Testing

Tests can be executed using the helper script `run_tests.py`. Specify the type of
tests to run along with any optional flags:

```bash
python run_tests.py [unit|integration|smoke|all|coverage] [--verbose] [--fast]
```

Example commands:

- Run the unit test suite:

  ```bash
  python run_tests.py unit
  ```

- Generate a coverage report:

  ```bash
  python run_tests.py coverage
  ```

For additional details see [TESTING.md](TESTING.md).

## Contributors
- [@robinabrown](https://github.com/robinabrown) Robin Brown
- [@PratikSathe](https://github.com/PratikSathe) Pratik Sathe
- [@bernalde](https://github.com/bernalde) David Bernal Neira

## Acknowledgements

This code was developed under the NSF Expeditions Program NSF award CCF-1918549 on [Coherent Ising Machines](https://cohesing.org)


## License

[Apache 2.0](LICENSE)
