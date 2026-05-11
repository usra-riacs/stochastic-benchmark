# Wishart n=50 α=0.50 Example

This example demonstrates the stochastic benchmarking framework applied to Wishart problem instances.

## Prerequisites

### Additional Dependencies

This example has different dependencies depending on what you want to do:

#### For Running Analysis Only (using pre-generated data):
```bash
pip install -r ../../requirements-examples.txt
```

This installs:
- **scikit-learn** - Required for polynomial regression models in parameter recommendation

#### For Generating New Experimental Data:
If you want to generate new data (not just analyze existing results), you also need:
- **pysa** - For running simulated annealing experiments
- Note: `wishart_runs.py` handles missing pysa gracefully with a try/except

The analysis notebook (`wishart_n_50_alpha_0.50.ipynb`) only requires scikit-learn and works with pre-generated data files.

### Data Files

The example expects the following directory structure relative to this folder:

```
wishart_n_50_alpha_0.5/
├── wishart_ws.py                          # Main analysis functions
├── wishart_runs.py                        # Experimental run functions
├── wishart_n_50_alpha_0.50.ipynb         # Main analysis notebook
├── rerun_data/                            # Experimental results (pickled data)
│   ├── hpoTrials_warmstart=*_trial=*_inst=*.pkl
│   └── ...
└── wishart_planting_N_50_alpha_0.50/     # Problem instances
    ├── wishart_planting_N_50_alpha_0.50_inst_*.txt
    └── gs_energies.txt                    # Ground state energies
```

## Running the Example

1. **Ensure you have the data files** in the appropriate directories (see structure above)

2. **Install dependencies**:
   ```bash
   pip install -r ../../requirements-examples.txt
   ```

3. **Open and run the Jupyter notebook**:
   ```bash
   jupyter notebook wishart_n_50_alpha_0.50.ipynb
   ```

## What This Example Demonstrates

- Loading experimental data from multiple parameter configurations
- Bootstrap resampling for statistical analysis
- Interpolation across resource levels
- Computing virtual best (oracle) performance
- Comparing different parameter recommendation strategies
- Generating performance plots (Window Stickers)

## Key Files

- **wishart_ws.py**: Contains `stoch_bench_setup()` which initializes the benchmarking framework with Wishart-specific configuration
- **wishart_runs.py**: Functions for running QAOA/simulated annealing experiments on Wishart instances
- **wishart_n_50_alpha_0.50.ipynb**: Main analysis notebook with visualization

## Notes

- The paths in `wishart_ws.py` and `wishart_runs.py` have been configured to use relative paths for portability
- If you encounter permission errors, ensure you're running the notebook from the correct directory
- The example uses polynomial regression models (via scikit-learn) for parameter recommendation strategies
