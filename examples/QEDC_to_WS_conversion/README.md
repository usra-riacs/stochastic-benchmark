# QED-C to Window Sticker conversion

This directory contains an external conversion utility for QED-C MaxCut output.
It is not a self-contained stochastic-benchmark tutorial: `conversion.py`
imports QED-C's `maxcut_benchmark` and `qedclib.metrics` modules and expects
QED-C result JSON files under a local `__results` directory.

`maxcut_benchmark` and `qedclib` are supplied by the QED-C repository. They are
not provided by this repository, and `maxcut_benchmark` is not available as a
PyPI package named `maxcut-benchmark`.

## External setup

Clone and install QED-C separately:

```bash
git clone https://github.com/SRI-International/QC-App-Oriented-Benchmarks.git
cd QC-App-Oriented-Benchmarks
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install this repository's runtime dependencies into the same environment so the
converter has `pandas` and `scipy` available:

```bash
python -m pip install -r /path/to/stochastic-benchmark/requirements.txt
```

Copy this conversion utility into QED-C's MaxCut Qiskit directory:

```bash
cp /path/to/stochastic-benchmark/examples/QEDC_to_WS_conversion/conversion.py qedcbench/maxcut/qiskit/
cp /path/to/stochastic-benchmark/examples/QEDC_to_WS_conversion/conversion.ipynb qedcbench/maxcut/qiskit/
```

Place or copy the QED-C MaxCut result data under:

```text
qedcbench/maxcut/qiskit/__results/
```

The notebook assumes folders like:

```text
__results/instance=0/approx_ratio/rounds-2_shots-100/
```

If your QED-C output uses a different layout, update `get_folder_names` in the
notebook before executing it.

From the QED-C checkout, run:

```bash
cd qedcbench/maxcut/qiskit
jupyter notebook conversion.ipynb
```

The converted pickle files are written under `__results_pkl/`, and the
bootstrapped Window Sticker inputs are written under `checkpoints/`.
