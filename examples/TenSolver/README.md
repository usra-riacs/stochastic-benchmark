# TenSolver Benchmark Example

Benchmarking [TenSolver](https://github.com/SECQUOIA/TenSolver.jl) on 23 QUBO instances from the QpLib subset of QUBOLib (instances 308–330).

## What's included

- **Energy distribution visualizations** — heatmap showing how the sampled energy distribution evolves across DMRG iterations for a representative instance.
- **Performance ratio** — computed against Gurobi's best-known objectives as the reference minimum and random bitstring sampling as the baseline.
- **Bootstrap analysis** — performance ratio inferred for 1, 10, 100, and 1000 samples per iteration using the stochastic-benchmark framework.
- **Resource analysis** — `iterations × samples` used as a proxy for compute; interpolation, train/test split, virtual best baseline, and projection experiments.

## Data files

| File | Description |
|---|---|
| `results/energy_history.json` | 1000 energy samples per DMRG iteration for all 23 instances |
| `results/random_baseline.json` | Mean and min energy from 1000 random bitstrings per instance |
| `results/gurobi_best.json` | Best objective found by Gurobi (900 s limit) per instance |

## Dependencies

Install the repository's core dependencies and the shared example notebook dependencies from the repository root:

```bash
pip install -r requirements.txt
pip install -r requirements-examples.txt
```

The example requirements include the notebook execution tooling used for command-line checks:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace examples/TenSolver/TenSolver.ipynb
```
