"""Regressions for examples/IBM_QAOA/run_latency_recost.py, the script that
re-prices the simulated campaigns under the circuit-preparation charge and
writes the variant roots the cost-model figure reads.

Three things reviewers found wrong at 485eaa4 are pinned here so they cannot
come back: the charged variant billed N*M requested shots while the stated
equation bills the n_evals*M actually recorded; the default campaign list
lagged the notebook's and silently skipped the depth-7 roots; and the
hardware frontier, whose QPU term is a shot-rate recalibration rather than a
measurement, was reused unchanged in the charged panel as if it already
carried a submission cost.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IBM_QAOA_ROOT = REPO_ROOT / "examples" / "IBM_QAOA"
for path in (REPO_ROOT / "src", IBM_QAOA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.utils import _build_hw_frontier, plot_cost_model_comparison_panels  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("run_latency_recost", IBM_QAOA_ROOT / "run_latency_recost.py")
recost = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recost)

T_SHOT = 1.0 / 2470.0
T_PREP = 13.87
NOTEBOOK = IBM_QAOA_ROOT / "notebooks" / "Simulation_Method_Validation_and_WS.ipynb"


def _reviewer_row() -> pd.DataFrame:
    """N=10, M=1000, Q=100, 15 objective evaluations, 15,000 recorded shots."""
    return pd.DataFrame({
        "N": [10.0], "M": [1000.0], "Q": [100.0],
        "num_objective_evaluations": [15], "total_training_shots": [15000],
        "training_cost_proxy": [10 * 1000 * T_SHOT],
        "sampling_cost_proxy": [100 * T_SHOT],
        "classical_setup_cost_proxy": [0.0],
        "T_exact_proxy": [10 * 1000 * T_SHOT + 100 * T_SHOT],
        "T_proxy": [10 * 1000 * T_SHOT + 100 * T_SHOT],
        "p": [5], "strategy": ["LR_PP_opt"], "instance": ["000"], "split": ["train"],
    })


# ---------------------------------------------------------------------------
# price_variant / build_variant: the charged equation bills recorded shots
# ---------------------------------------------------------------------------
class TestPriceVariant:
    def test__price_variant__charged__bills_recorded_training_shots_not_n_times_m(self):
        # ACT
        priced = recost.price_variant(_reviewer_row(), T_PREP)

        # ASSERT -- t_pre + n_evals*(t_prep + M*t_shot) + (t_prep + Q*t_shot)
        expected = 15 * (T_PREP + 1000 * T_SHOT) + (T_PREP + 100 * T_SHOT)
        assert priced["T_proxy"].iloc[0] == pytest.approx(expected)
        assert priced["T_proxy"].iloc[0] == pytest.approx(228.03336, abs=1e-5)
        # and specifically NOT the N*M accounting the figure used to be built on
        assert priced["T_proxy"].iloc[0] != pytest.approx(226.00907, abs=1e-3)

    def test__price_variant__no_charge__leaves_the_published_pricing_untouched(self):
        row = _reviewer_row()
        priced = recost.price_variant(row, 0.0)
        assert priced is row
        assert priced["T_proxy"].iloc[0] == pytest.approx(10 * 1000 * T_SHOT + 100 * T_SHOT)

    def test__build_variant__feeds_the_recorded_shot_pricing_to_the_frontier_builder(self, monkeypatch, tmp_path):
        # ARRANGE -- stub the stochastic stages and capture what reaches them
        seen: dict = {}

        def fake_frontier(priced, **kwargs):
            seen["priced"] = priced.copy()
            return pd.DataFrame({"T": priced["T_proxy"], "BestApproximationRatio": [0.9] * len(priced),
                                 "split": priced["split"], "strategy": priced["strategy"]})

        monkeypatch.setattr(recost, "build_resource_frontier_from_exact_points", fake_frontier)
        monkeypatch.setattr(recost, "run_stochastic_benchmark_pss", lambda *a, **k: {})
        monkeypatch.setattr(recost, "build_strategy_budget_summary", lambda *a, **k: pd.DataFrame())

        # ACT
        recost.build_variant(_reviewer_row(), tmp_path / "v", {}, circuit_prep_time=T_PREP,
                             num_bins=10, bootstrap_range=range(1, 2), train_test_split=0.5)

        # ASSERT -- the figure generator, not just the helper's default path, uses recorded shots
        assert seen["priced"]["T_proxy"].iloc[0] == pytest.approx(228.03336, abs=1e-5)


# ---------------------------------------------------------------------------
# DEFAULT_TAGS: the documented command must regenerate every campaign the
# notebook's cost-model figure reads
# ---------------------------------------------------------------------------
def _notebook_cost_panel_tags() -> set[str]:
    code = "\n".join(
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code"
    )
    tags = set(re.findall(r"'(heavy_hex_144_[A-Za-z0-9_]+_expanded)'", code))
    excluded_block = re.search(r"COST_PANEL_EXCLUDED_TAGS = \[([^\]]*)\]", code).group(1)
    excluded = set(re.findall(r"'([^']+)'", excluded_block))
    return tags - excluded


class TestDefaultTags:
    def test__default_tags__cover_every_campaign_the_notebook_figure_reads(self):
        missing = _notebook_cost_panel_tags() - set(recost.DEFAULT_TAGS)
        assert not missing, f"notebook reads these but the script's defaults never generate them: {sorted(missing)}"

    def test__default_tags__include_both_depth_7_campaigns(self):
        assert "heavy_hex_144_LR_opt_p7_expanded" in recost.DEFAULT_TAGS
        assert "heavy_hex_144_FA_no_opt_p7_expanded" in recost.DEFAULT_TAGS


# ---------------------------------------------------------------------------
# Hardware frontier: its QPU term is num_shots * t_shot, not the measured QPU
# time, so it carries no submission overhead and the charged panel must add
# one t_prep for the single sampling job each run submitted.
# ---------------------------------------------------------------------------
def _hardware_inputs():
    hw_new = pd.DataFrame({
        "file_name": ["000_MC_A.json"], "job_p": [6], "training_method": ["FA_PP_opt_6"],
        "approximation_ratio": [0.9],
        "QPU_time (s)": [20.0],                      # the measured value ...
        "QPU_time_noiseless (s)": [10000 * T_SHOT],  # ... and the recalibration that replaces it
        "num_shots": [10000], "total_train_cost": [0.0],
    })
    hw = pd.DataFrame({"file_name": ["000_MC_A.json"], "instance_name": ["000"]})
    return hw_new, hw


class TestHardwareResource:
    def test__build_hw_frontier__prices_shots_at_the_calibrated_rate_and_ignores_measured_qpu_time(self):
        hw_new, hw = _hardware_inputs()
        frontier = _build_hw_frontier("QPU_time_noiseless (s)", hw_new, hw, 144)
        assert frontier["dur_mean"].iloc[0] == pytest.approx(10000 * T_SHOT)      # 4.04858 s
        assert frontier["dur_mean"].iloc[0] != pytest.approx(20.0)                # the 20 s is not used

    def test__cost_model_panels__charged_panel_shifts_the_hardware_frontier_by_one_t_prep(self, tmp_path):
        # ARRANGE -- identical simulated curve in both panels; hardware charged in the second only
        hw_new, hw = _hardware_inputs()
        hw_frontier = _build_hw_frontier("QPU_time_noiseless (s)", hw_new, hw, 144)
        hw_frontier["method_label"] = "Fixed Angles$^\\star$ (p=6)"
        prescription = pd.DataFrame({
            "resource": [0.1, 1.0, 10.0], "response": [0.80, 0.85, 0.90],
            "response_lower": [0.79, 0.84, 0.89], "response_upper": [0.81, 0.86, 0.91],
            "method_label": ["Fixed Angles$^\\star$ (p=5)"] * 3,
        })
        overlay = {"label": "hw", "frontier_df": hw_frontier, "linestyle": ":", "marker": "s"}
        panels = [
            {"title": "a", "calibrations": [{"label": "nl", "prescription_df": prescription}],
             "hardware": dict(overlay)},
            {"title": "b", "calibrations": [{"label": "nl", "prescription_df": prescription}],
             "hardware": dict(overlay, extra_cost=T_PREP)},
        ]

        # ACT
        import matplotlib.pyplot as plt
        plot_cost_model_comparison_panels(panels=panels, plot_dir=str(tmp_path), filename="hw",
                                          show_error_bars=False, cluster_inset=False)
        fig = plt.gcf()
        axes = [ax for ax in fig.axes if ax.get_xlabel()][:2]

        def hardware_x(ax):
            xs = [line.get_xydata()[:, 0] for line in ax.lines if line.get_linestyle() == ":"]
            return min(x.min() for x in xs if len(x))

        # ASSERT -- the one submitted sampling job is billed once, and only under the charge
        assert hardware_x(axes[1]) - hardware_x(axes[0]) == pytest.approx(T_PREP)
        assert hardware_x(axes[0]) == pytest.approx(10000 * T_SHOT)
        plt.close("all")
