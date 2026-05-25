# IBM_QAOA Notebooks

This folder contains notebooks used to analyze IBM Quantum QAOA experiments (hardware runs + parameter-training runs) and to integrate the resulting measurements into the `stochastic-benchmark` style performance/resource analysis.

Primary upstream data/method repository: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting

## Files

- `Analysis.ipynb`
  - End-to-end analysis notebook:
    - Loads IBM hardware data and training data (via `../src/Processing.py`)
    - Computes MaxCut approximation ratios (via `../src/approx_ratio_calc.py`)
    - Several helper functions used by the notebook live in `../src/utils.py`.
    - Builds derived resource-cost columns (e.g., total duration = QPU time + estimated training time)
    - Produces performance plots and training-cost breakdown plots

- `Simulation_Method_Validation_and_WS.ipynb`
  - Simulation-method validation and Window Sticker notebook for the FA/PT-style PSS campaign.
  - Loads campaign frontier data from `../results/pss_window_sticker/<result_tag>`.
  - Runs the stochastic-benchmark Window Sticker workflow through `../src/simulation_validation.py`.
  - Uses plotting and reporting helpers from `../src/utils.py`; this notebook should remain orchestration/display only and should not define local helper functions.
  - Keeps the proxy-vs-exact calibration as the power-law/through-origin diagnostic plot used by the current experiment.

## Notes

- The notebook expects local data directories/instance directories to be available (paths are currently set inside the notebook).
- Generated training instances for the simulation-validation campaigns are cached under `../data/generated_instances` by default. New campaign outputs should reference that shared cache instead of regenerating the same graph/min-max-cut files under each result directory.

## External Repositories For Simulation Validation

`Simulation_Method_Validation_and_WS.ipynb` and the matching campaign script
`../run_prepare_pss_campaign.py` require two external repositories in addition
to this `stochastic-benchmark` checkout:

- QAOA-Parameter-Setting: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting
  - Provides the IBM/QAOA data layout used by the notebooks, including
    training data, hardware data, evaluation-time tables, generated instances,
    and min/max-cut reference files.
- qaoa_training_pipeline: https://github.com/qiskit-community/qaoa_training_pipeline
  - Provides the trainer/evaluator implementations used when preparing
    simulation-validation campaigns.

By default, `../src/simulation_validation.py` looks for sibling checkouts next
to this repository:

```text
<workspace>/stochastic-benchmark
<workspace>/QAOA-Parameter-Setting
<workspace>/qaoa_training_pipeline
```

If your checkouts live elsewhere, set these environment variables before
running the notebook or script:

```bash
export QAOA_PARAMETER_SETTING_ROOT=/path/to/QAOA-Parameter-Setting
export QAOA_TRAINING_PIPELINE_ROOT=/path/to/qaoa_training_pipeline
export IBM_QAOA_INSTANCE_CACHE_ROOT=/path/to/generated_instances
```

For script runs, the same paths can be passed directly with
`--main-repo`, `--pipeline-repo`, and `--instance-cache-root`.
