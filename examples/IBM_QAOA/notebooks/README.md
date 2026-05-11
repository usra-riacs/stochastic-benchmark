# IBM_QAOA Notebooks

This folder contains notebooks used to analyze IBM Quantum QAOA experiments (hardware runs + parameter-training runs) and to integrate the resulting measurements into the `stochastic-benchmark` style performance/resource analysis.

Upstream repository: https://github.com/Quantum-Working-Groups/QAOA-Parameter-Setting

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
