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
- The heavy-hex n=144 instance set under `../data/generated_instances/heavy_hex__n=144__.../` is committed deliberately as a reproducibility snapshot for the p=7/p=9 campaigns in this PR, not as regenerable scratch output. The code (`generate_training_instances_like_qps`, `_compute_and_write_minmax`) can regenerate it from the same deterministic cache key (graph type, node count, start index, count), but the committed files pin the exact min/max-cut values the committed campaign results were computed against.

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

### Python dependency setup

In addition to `pip install -r requirements.txt -r requirements-examples.txt`
from the repo root (which now includes `qiskit-aer`), install the two sibling
checkouts as editable packages:

```bash
python -m pip install -e /path/to/qaoa_training_pipeline "qiskit-aer==0.17.2"
python -m pip install -e /path/to/QAOA-Parameter-Setting
```

This mirrors the setup used by `nautilus/run_simulation_validation.sh`, which
is the authoritative recipe for a from-scratch environment.
