"""Smoke checks for the TenSolver example notebook."""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "examples" / "TenSolver" / "TenSolver.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text())


def _code_cell_sources():
    return [
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    ]


def test_first_tensolver_code_cell_runs_from_repo_root(monkeypatch):
    """The notebook should locate its data when launched from the repo root."""
    monkeypatch.chdir(REPO_ROOT)
    os.environ.setdefault("MPLBACKEND", "Agg")

    namespace = {"__name__": "__main__"}
    exec(compile(_code_cell_sources()[0], str(NOTEBOOK), "exec"), namespace)

    assert namespace["HERE"] == REPO_ROOT / "examples" / "TenSolver"
    assert "323" in namespace["all_data"]


def test_tensolver_notebook_seeds_stochastic_steps():
    """Bootstrap and train/test split output should be reproducible."""
    source = "\n".join(_code_cell_sources())

    assert "RANDOM_SEED = " in source
    assert source.count("np.random.seed(RANDOM_SEED)") >= 2


def test_tensolver_notebook_uses_example_dir_for_outputs():
    """Checkpoint and plot output should stay in the example directory."""
    source = "\n".join(_code_cell_sources())

    assert "here=str(HERE)" in source
    assert "fig.savefig(HERE / \"performance.png\")" in source
