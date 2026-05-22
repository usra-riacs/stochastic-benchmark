# QED-C to Window Sticker conversion

This directory converts QED-C MaxCut result JSON into the Window Sticker pickle
format. It includes a tiny QED-C-compatible fixture under `__results/`, so the
notebook can be executed from this repository without installing QED-C or
rerunning a MaxCut experiment. This makes the tutorial self-contained.

The bundled fixture is sample data for exercising the conversion path. It is not
a benchmark-quality dataset and should not be used for performance conclusions.

`conversion.py` also supports full QED-C output. If the QED-C modules
`maxcut_benchmark` and `qedclib.metrics` are importable, the converter uses
QED-C's loader. Otherwise, it reads the small subset of the QED-C JSON schema
needed by this tutorial directly.

## Run the included fixture

Install this repository's runtime dependencies, then execute the notebook from
this directory:

```bash
python -m pip install -r ../../requirements.txt
python -m pip install -r ../../requirements-examples.txt
python -m jupyter nbconvert --to notebook --execute --inplace conversion.ipynb
```

The notebook reads folders like:

```text
__results/instance=0/approx_ratio/rounds-2_shots-100/
```

The converted pickle files are written under `__results_pkl/`, and the
bootstrapped Window Sticker inputs are written under `checkpoints/`.

## Use full QED-C output

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
python -m jupyter nbconvert --to notebook --execute --inplace conversion.ipynb
```

The generated pickle files are written next to the notebook in the same output
directories used by the fixture run.
