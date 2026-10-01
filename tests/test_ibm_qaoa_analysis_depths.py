"""Keep Analysis's full depth cohort independent of its duration-plot subset."""

import ast
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from pandas.testing import assert_frame_equal

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
IBM_QAOA_ROOT = REPO_ROOT / "examples" / "IBM_QAOA"
for path in (REPO_ROOT / "src", IBM_QAOA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.utils import prepare_training_bricks_data, plot_ibm_qaoa_training_bricks  # noqa: E402


def _cell_source(cell_id: str) -> str:
    notebook = json.loads((IBM_QAOA_ROOT / "notebooks" / "Analysis.ipynb").read_text())
    return "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == cell_id))


def test_analysis_loads_all_hardware_and_training_depths() -> None:
    """The shared depth list [-] includes every depth in the comparison."""
    config = ast.parse(_cell_source("e1eb45ea"))
    assignment = next(
        node for node in config.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "p_list" for target in node.targets)
    )
    namespace = {}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "analysis config", "exec"), namespace)
    assert namespace["p_list"] == list(range(2, 11))


def test_duration_cell_filters_only_its_plot_data(tmp_path: Path) -> None:
    """The actual cell plots p=5/10 [-] without pruning shared duration data [s]."""
    hardware = []
    training = []
    for depth in (5, 7, 10):
        for run in (0, 1):
            file_name = f"run_{depth}_{run}"
            hardware.append({"file_name": file_name, "job_p": depth, "training_method": f"FA_PP_opt_{depth}"})
            training.extend([
                {"file_name": file_name, "level": "outer", "iteration": 0,
                 "depth_step": 0, "duration": 1.0 + run},
                {"file_name": file_name, "level": "inner", "iteration": 1,
                 "depth_step": 1, "duration": float(depth * 10 + run)},
            ])
    df_flat = pd.DataFrame(training)
    df_hardware = pd.DataFrame(hardware)
    originals = df_flat.copy(deep=True), df_hardware.copy(deep=True)
    namespace = {
        "df_flat": df_flat, "df_hardware_new": df_hardware,
        "prepare_training_bricks_data": prepare_training_bricks_data,
        "plot_ibm_qaoa_training_bricks": plot_ibm_qaoa_training_bricks,
        "color_map": {}, "label_map": {}, "_graph_id": "depth-regression", "SAVE_DIR": str(tmp_path),
    }
    try:
        exec(compile(_cell_source("1cd2236b"), "analysis duration cell", "exec"), namespace)
        assert set(namespace["agg"]["job_p"]) == {5, 10}
        assert [tick.get_text() for tick in plt.gcf().axes[0].get_xticklabels()] == ["5", "10"]
        assert_frame_equal(namespace["df_flat"], originals[0])
        assert_frame_equal(namespace["df_hardware_new"], originals[1])
        assert (tmp_path / "depth-regression_training_bricks.pdf").is_file()
    finally:
        plt.close("all")
