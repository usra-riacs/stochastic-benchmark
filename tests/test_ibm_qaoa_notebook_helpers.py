"""Characterization tests for helper functions still living inline inside
Simulation_Method_Validation_and_WS.ipynb, ahead of moving them into
examples/IBM_QAOA/src/utils.py.

These tests lock CURRENT behavior (Protect phase of the sefop-training-hub
refactoring guide) so the move in the next step can be verified byte-for-byte
against them rather than against what the function is assumed to do.

Since the functions don't live in an importable module yet, each one is
extracted verbatim from the notebook's own JSON at test-collection time and
exec'd into a private namespace -- this is throwaway scaffolding for this one
test file and goes away once Step 2 moves the functions into src/utils.py and
this file is repointed to import them normally.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IBM_QAOA_ROOT = REPO_ROOT / "examples" / "IBM_QAOA"
NOTEBOOK_PATH = IBM_QAOA_ROOT / "notebooks" / "Simulation_Method_Validation_and_WS.ipynb"
for path in (REPO_ROOT / "src", IBM_QAOA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.utils import curve_from_response_summary, _method_label_from_training_method  # noqa: E402


def _extract_cell_source(cell_id: str) -> str:
    nb = json.loads(NOTEBOOK_PATH.read_text())
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell.get("source", []))
    raise KeyError(f"No cell with id={cell_id!r} in {NOTEBOOK_PATH}")


def _extract_function(cell_id: str, func_name: str, namespace: dict) -> None:
    """Exec just the `def <func_name>(...): ...` block from a notebook cell.

    Slices from the `def` line to the next top-level (unindented, non-blank)
    line so cell code before/after the function (which may reference names
    that don't exist in this test's namespace) is never executed.
    """
    src = _extract_cell_source(cell_id)
    lines = src.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"def {func_name}("))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith((" ", "\t")):
            end = i
            break
    func_src = "\n".join(lines[start:end])
    exec(compile(func_src, f"<notebook cell {cell_id}:{func_name}>", "exec"), namespace)


# ---------------------------------------------------------------------------
# _relativize_warning_filename / _scrubbed_formatwarning  (cell 914a9b2d)
# ---------------------------------------------------------------------------

@pytest.fixture
def warning_scrub_ns(tmp_path):
    ns = {"Path": Path, "WORKSPACE_ROOT": tmp_path}
    _extract_function("914a9b2d", "_relativize_warning_filename", ns)
    _extract_function("914a9b2d", "_scrubbed_formatwarning", ns)
    return ns


class TestRelativizeWarningFilename:
    def test__relativize_warning_filename__given_path_under_workspace_root__strips_prefix(self, warning_scrub_ns, tmp_path):
        # ARRANGE
        target = tmp_path / "repo" / "src" / "interpolate.py"
        target.parent.mkdir(parents=True)
        target.touch()

        # ACT
        result = warning_scrub_ns["_relativize_warning_filename"](str(target))

        # ASSERT
        assert result == "repo/src/interpolate.py"

    def test__relativize_warning_filename__given_workspace_container_prefix__strips_it(self, warning_scrub_ns):
        # ARRANGE
        filename = "/workspace/repo/src/interpolate.py"

        # ACT
        result = warning_scrub_ns["_relativize_warning_filename"](filename)

        # ASSERT
        assert result == "repo/src/interpolate.py"

    def test__relativize_warning_filename__given_unrelated_path__returns_unchanged(self, warning_scrub_ns):
        # ARRANGE
        filename = "/some/other/machine/path/module.py"

        # ACT
        result = warning_scrub_ns["_relativize_warning_filename"](filename)

        # ASSERT
        assert result == filename


class TestScrubbedFormatwarning:
    def test__scrubbed_formatwarning__given_message_and_location__formats_without_absolute_path(self, warning_scrub_ns, tmp_path):
        # ARRANGE
        target = tmp_path / "src" / "interpolate.py"
        target.parent.mkdir(parents=True)
        target.touch()

        # ACT
        result = warning_scrub_ns["_scrubbed_formatwarning"](
            "Dataframe has duplicate resources.", UserWarning, str(target), 121
        )

        # ASSERT
        assert result == "src/interpolate.py:121: UserWarning: Dataframe has duplicate resources.\n"

    def test__scrubbed_formatwarning__end_to_end_via_warnings_warn__scrubs_the_path(self, warning_scrub_ns, tmp_path, monkeypatch):
        # ARRANGE
        monkeypatch.setattr(warnings, "formatwarning", warning_scrub_ns["_scrubbed_formatwarning"])
        target = tmp_path / "src" / "interpolate.py"
        target.parent.mkdir(parents=True)
        target.touch()

        # ACT
        with warnings.catch_warnings(record=False):
            formatted = warnings.formatwarning("test message", UserWarning, str(target), 5)

        # ASSERT
        assert str(tmp_path) not in formatted
        assert formatted == "src/interpolate.py:5: UserWarning: test message\n"


# ---------------------------------------------------------------------------
# _relativize_paths  (cell d78417eb)
# ---------------------------------------------------------------------------

@pytest.fixture
def relativize_paths_ns():
    ns = {"pd": pd, "Path": Path}
    _extract_function("d78417eb", "_relativize_paths", ns)
    return ns


class TestRelativizePaths:
    def test__relativize_paths__given_path_under_base__resolves_relative(self, relativize_paths_ns, tmp_path):
        # ARRANGE
        target = tmp_path / "instances" / "graph_000.json"
        target.parent.mkdir(parents=True)
        target.touch()
        df = pd.DataFrame({"graph_path": [str(target)], "other_col": [1]})

        # ACT
        result = relativize_paths_ns["_relativize_paths"](df, ["graph_path"], tmp_path)

        # ASSERT
        assert result["graph_path"].iloc[0] == "instances/graph_000.json"

    def test__relativize_paths__given_workspace_container_prefix__strips_it(self, relativize_paths_ns, tmp_path):
        # ARRANGE
        df = pd.DataFrame({"graph_path": ["/workspace/repo/instances/graph_000.json"]})

        # ACT
        result = relativize_paths_ns["_relativize_paths"](df, ["graph_path"], tmp_path)

        # ASSERT
        assert result["graph_path"].iloc[0] == "repo/instances/graph_000.json"

    def test__relativize_paths__given_nan_value__passes_through_unchanged(self, relativize_paths_ns, tmp_path):
        # ARRANGE
        df = pd.DataFrame({"graph_path": [np.nan]})

        # ACT
        result = relativize_paths_ns["_relativize_paths"](df, ["graph_path"], tmp_path)

        # ASSERT
        assert pd.isna(result["graph_path"].iloc[0])

    def test__relativize_paths__given_column_not_in_df__is_a_no_op_for_that_column(self, relativize_paths_ns, tmp_path):
        # ARRANGE
        df = pd.DataFrame({"other_col": [1, 2]})

        # ACT
        result = relativize_paths_ns["_relativize_paths"](df, ["graph_path"], tmp_path)

        # ASSERT
        assert list(result.columns) == ["other_col"]

    def test__relativize_paths__does_not_mutate_the_input_dataframe(self, relativize_paths_ns, tmp_path):
        # ARRANGE
        target = tmp_path / "graph_000.json"
        target.touch()
        df = pd.DataFrame({"graph_path": [str(target)]})

        # ACT
        relativize_paths_ns["_relativize_paths"](df, ["graph_path"], tmp_path)

        # ASSERT
        assert df["graph_path"].iloc[0] == str(target)


# ---------------------------------------------------------------------------
# _rescale_resource  (cell 7f87d664)
# ---------------------------------------------------------------------------

@pytest.fixture
def rescale_resource_ns():
    ns = {"pd": pd}
    _extract_function("7f87d664", "_rescale_resource", ns)
    return ns


class TestRescaleResource:
    def test__rescale_resource__given_matching_method_label__multiplies_resource_by_factor(self, rescale_resource_ns):
        # ARRANGE
        df = pd.DataFrame({"method_label": ["A", "B"], "resource": [10.0, 20.0]})

        # ACT
        result = rescale_resource_ns["_rescale_resource"](df, {"A": 2.0})

        # ASSERT
        assert result.loc[result["method_label"] == "A", "resource"].iloc[0] == pytest.approx(20.0)
        assert result.loc[result["method_label"] == "B", "resource"].iloc[0] == pytest.approx(20.0)  # unscaled

    def test__rescale_resource__given_none_dataframe__returns_empty_dataframe(self, rescale_resource_ns):
        # ARRANGE / ACT
        result = rescale_resource_ns["_rescale_resource"](None, {"A": 2.0})

        # ASSERT
        assert result.empty

    def test__rescale_resource__given_empty_dataframe__returns_copy_unchanged(self, rescale_resource_ns):
        # ARRANGE
        df = pd.DataFrame({"resource": []})

        # ACT
        result = rescale_resource_ns["_rescale_resource"](df, {"A": 2.0})

        # ASSERT
        assert result.empty

    def test__rescale_resource__given_no_resource_column__returns_copy_unchanged(self, rescale_resource_ns):
        # ARRANGE
        df = pd.DataFrame({"other_col": [1, 2]})

        # ACT
        result = rescale_resource_ns["_rescale_resource"](df, {"A": 2.0})

        # ASSERT
        pd.testing.assert_frame_equal(result, df)

    def test__rescale_resource__does_not_mutate_the_input_dataframe(self, rescale_resource_ns):
        # ARRANGE
        df = pd.DataFrame({"method_label": ["A"], "resource": [10.0]})

        # ACT
        rescale_resource_ns["_rescale_resource"](df, {"A": 2.0})

        # ASSERT
        assert df["resource"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _sim_entries  (cell 612c9536)
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_entries_ns():
    ns = {
        "pd": pd,
        "np": np,
        "_curve_from_response_summary": curve_from_response_summary,
    }
    _extract_function("612c9536", "_sim_entries", ns)
    return ns


class TestSimEntries:
    def test__sim_entries__given_empty_dataframe__returns_empty_list(self, sim_entries_ns):
        # ARRANGE
        df = pd.DataFrame()

        # ACT
        result = sim_entries_ns["_sim_entries"](df)

        # ASSERT
        assert result == []

    def test__sim_entries__given_two_resource_points_no_ci__returns_one_entry_with_nan_bounds(self, sim_entries_ns):
        # ARRANGE
        df = pd.DataFrame({
            "method_label": ["A", "A"],
            "resource": [1.0, 2.0],
            "response": [0.5, 0.6],
        })

        # ACT
        result = sim_entries_ns["_sim_entries"](df)

        # ASSERT
        assert len(result) == 1
        label, xs, ys, ci_lower, ci_upper = result[0]
        assert label == "A"
        np.testing.assert_allclose(xs, [1.0, 2.0])
        np.testing.assert_allclose(ys, [50.0, 60.0])
        assert np.all(np.isnan(ci_lower))
        assert np.all(np.isnan(ci_upper))

    def test__sim_entries__given_single_resource_point__is_dropped(self, sim_entries_ns):
        # ARRANGE -- a method_label needs >= 2 points to form a curve
        df = pd.DataFrame({
            "method_label": ["A"],
            "resource": [1.0],
            "response": [0.5],
        })

        # ACT
        result = sim_entries_ns["_sim_entries"](df)

        # ASSERT
        assert result == []

    def test__sim_entries__given_ci_columns__rescales_95pct_ci_down_to_1_sem(self, sim_entries_ns):
        # ARRANGE -- response_lower/upper represent response +/- 1.96*SEM;
        # entries should carry response +/- 1*SEM instead.
        df = pd.DataFrame({
            "method_label": ["A", "A"],
            "resource": [1.0, 2.0],
            "response": [0.50, 0.60],
            "response_lower": [0.50 - 1.96 * 0.02, 0.60 - 1.96 * 0.02],
            "response_upper": [0.50 + 1.96 * 0.02, 0.60 + 1.96 * 0.02],
        })

        # ACT
        result = sim_entries_ns["_sim_entries"](df)

        # ASSERT
        _, _, ys, ci_lower, ci_upper = result[0]
        np.testing.assert_allclose(ci_lower, ys - 2.0, atol=1e-6)
        np.testing.assert_allclose(ci_upper, ys + 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# _label_hw_frontier  (cell 612c9536)
# ---------------------------------------------------------------------------

@pytest.fixture
def label_hw_frontier_ns():
    ns = {"_method_label_from_training_method": _method_label_from_training_method}
    _extract_function("612c9536", "_label_hw_frontier", ns)
    return ns


class TestLabelHwFrontier:
    def test__label_hw_frontier__builds_method_label_from_color_label_and_job_p(self, label_hw_frontier_ns):
        # ARRANGE
        frontier = pd.DataFrame({"color_label": ["FA_PP_opt"], "job_p": [6]})

        # ACT
        result = label_hw_frontier_ns["_label_hw_frontier"](frontier)

        # ASSERT
        assert "(p=6)" in result["method_label"].iloc[0]

    def test__label_hw_frontier__does_not_mutate_the_input_dataframe(self, label_hw_frontier_ns):
        # ARRANGE
        frontier = pd.DataFrame({"color_label": ["FA_PP_opt"], "job_p": [6]})

        # ACT
        label_hw_frontier_ns["_label_hw_frontier"](frontier)

        # ASSERT
        assert "method_label" not in frontier.columns
