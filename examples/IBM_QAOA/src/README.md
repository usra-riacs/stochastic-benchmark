# IBM_QAOA `src`

This folder contains utilities used by the IBM QAOA example notebooks and campaign scripts, including the hardware-analysis notebook and the simulation-validation/Window Sticker workflow.

## Files

- `__init__.py`
  - Package marker for the IBM_QAOA example modules.

- `Processing.py`
  - Loads and parses IBM Quantum hardware result data and QAOA parameter-training data.
  - Provides helpers to locate data/instance files and normalize them into Python objects used for downstream analysis.

- `approx_ratio_calc.py`
  - MaxCut-specific helpers for approximation ratio calculations.
  - Locates per-instance min/max cut reference data, extracts needed parameters, and computes approximation ratios.

- `utils.py`
  - General helper utilities used by the notebooks:
    - DataFrame helpers (e.g., expanding IBM counts into sample rows)
    - Plotting helpers used in the analysis and simulation-validation notebooks
    - Small statistical/label helpers used by plots (e.g., `sem`, `title_from_instance_names`)
    - Factory helper `make_asof_per_file` to build the groupby-apply function used when merging cumulative training duration.
    - Window Sticker notebook helpers for shared approximation-ratio axes, monotone curves, multi-strategy summaries, and plot saving.

- `simulation_validation.py`
  - Campaign and analysis utilities for simulation-method validation and Window Sticker experiments.
  - Builds generated train/test instance sets, exact FA/PT PSS points, budget frontiers, stochastic-benchmark campaign tables, and Window Sticker summaries.
  - Uses a shared generated-instance cache by default at `examples/IBM_QAOA/data/generated_instances`.
  - The cache key is derived from graph type, node count, graph parameters, starting train index, and train count, so repeated campaigns can reuse the same graph JSON and min/max cut files.
  - Requires local checkouts of:
    - QAOA-Parameter-Setting: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting
    - qaoa_training_pipeline: https://github.com/qiskit-community/qaoa_training_pipeline
  - By default these are resolved as sibling directories of this repository,
    but they can be overridden with `QAOA_PARAMETER_SETTING_ROOT` and
    `QAOA_TRAINING_PIPELINE_ROOT`.

## Campaign scripts

- `../run_prepare_pss_campaign.py`
  - Script equivalent of the setup/precompute path for `Simulation_Method_Validation_and_WS.ipynb`.
  - Accepts `--instance-cache-root` to override the shared generated-instance cache.
  - Accepts `--main-repo` and `--pipeline-repo` to point to non-default
    checkouts of the required external repositories.
  - Continues to write campaign-specific summaries and frontier outputs under `../results/pss_window_sticker/<result_tag>`.

## External Repository Layout

The default local layout is:

```text
<workspace>/stochastic-benchmark
<workspace>/QAOA-Parameter-Setting
<workspace>/qaoa_training_pipeline
```

Equivalent environment-variable configuration:

```bash
export QAOA_PARAMETER_SETTING_ROOT=/path/to/QAOA-Parameter-Setting
export QAOA_TRAINING_PIPELINE_ROOT=/path/to/qaoa_training_pipeline
export IBM_QAOA_INSTANCE_CACHE_ROOT=/path/to/generated_instances
```
