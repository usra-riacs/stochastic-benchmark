"""Tests for the remaining untested pure/IO-light helpers in
examples/IBM_QAOA/src/utils.py: the window-sticker label/color helpers
(pure regex and numeric logic, deterministic in this test environment since
the optional qaoa_parameter_setting.utils.labels import isn't on sys.path
here and falls back to the module's own QPS_METHOD_COLORS/_METHOD_NAMES
tables), a couple of small pure-data plot-prep helpers, and the result-root
summary-CSV loaders. Matplotlib-heavy plot_* functions remain out of scope
(verified visually all session).

Second and final chunk of Step 5 ("Extend") of the IBM_QAOA cleanup plan;
see test_ibm_qaoa_utils_data_prep.py for the first chunk. Intentionally
skipped in both chunks: the QPS method-label/color resolution chain
(_compact_method_label, _plain_method_label_from_training_method,
_method_color_from_training_method, _window_sticker_method_color,
_marker_from_training_method, _style_plot_kwargs,
_normalise_training_method_to_config, _method_config_to_method,
_evaluation_label_from_training_method, _optimization_size_maps,
_optimization_level, _optimization_alpha, _evaluator_edge_width) -- lower-
value style plumbing already implicitly exercised via
_method_label_from_training_method's existing coverage in
test_ibm_qaoa_simulation_validation.py, and the six functions dropped from
every import list in the Step 4 dead-code pass (title_from_instance_names,
make_asof_per_file, plot_training_bricks, plot_method_curves,
plot_multi_method_window_sticker_components, build_binned_budget_dataset),
since testing code nothing calls isn't a good use of this pass.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IBM_QAOA_ROOT = REPO_ROOT / "examples" / "IBM_QAOA"
for path in (REPO_ROOT / "src", IBM_QAOA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.utils import (  # noqa: E402
    _display_cross_strategy_envelope,
    _ensure_save_dir,
    _lighten_color,
    _prepare_parameter_curve,
    _shade_color,
    _window_sticker_label_base,
    _window_sticker_label_depth,
    _ws_display_method_label,
    draw_hardware_frontier_steps,
    load_cost_model_panels,
    load_multi_strategy_summaries,
    read_first_summary_csv,
    read_summary_csv,
    rebuild_strategy_budget_summary,
    resolve_result_root,
    window_sticker_method_color,
    window_sticker_method_color_map,
)


# ---------------------------------------------------------------------------
# _window_sticker_label_base / _window_sticker_label_depth
# ---------------------------------------------------------------------------

class TestWindowStickerLabelBase:
    def test__window_sticker_label_base__strips_depth_optimization_markers_and_case(self):
        assert _window_sticker_label_base("Fixed Angles$^\\star$ (p=5)") == "fixed angles"

    def test__window_sticker_label_base__given_no_markers__still_lowercases(self):
        assert _window_sticker_label_base("Linear Ramp") == "linear ramp"


class TestWindowStickerLabelDepth:
    def test__window_sticker_label_depth__given_depth_suffix__extracts_it(self):
        assert _window_sticker_label_depth("Fixed Angles* (p=7)") == 7

    def test__window_sticker_label_depth__given_no_suffix__returns_none(self):
        assert _window_sticker_label_depth("Fixed Angles*") is None


# ---------------------------------------------------------------------------
# _shade_color / _lighten_color
# ---------------------------------------------------------------------------

class TestShadeColor:
    def test__shade_color__given_positive_amount__lightens_toward_white(self):
        r, g, b = _shade_color("#000000", 0.5)
        assert r == pytest.approx(0.5)
        assert g == pytest.approx(0.5)
        assert b == pytest.approx(0.5)

    def test__shade_color__given_negative_amount__darkens_toward_black(self):
        r, g, b = _shade_color("#FFFFFF", -0.5)
        assert r == pytest.approx(0.5)


class TestLightenColor:
    def test__lighten_color__given_default_amount__blends_halfway_to_white(self):
        r, g, b = _lighten_color("#000000")
        assert r == pytest.approx(0.5)
        assert g == pytest.approx(0.5)
        assert b == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# window_sticker_method_color / window_sticker_method_color_map
# ---------------------------------------------------------------------------

class TestWindowStickerMethodColor:
    @pytest.mark.parametrize("label,expected_color", [
        ("Fixed Angles$^\\star$ (p=5)", "#4477AA"),
        ("Linear Ramp (p=5)", "#CCBB44"),
        ("Param. Transfer (p=5)", "#BBBBBB"),
        ("Interp.$^\\star$ (p=5)", "#228833"),
        ("Fourier* (p=5)", "#EE6677"),
        ("Recursive TS* (p=5)", "#66CCEE"),
        ("TQA* (p=5)", "#AA3377"),
    ])
    def test__window_sticker_method_color__given_a_known_family__returns_its_paper_color(self, label, expected_color):
        assert window_sticker_method_color(label) == expected_color


class TestWindowStickerMethodColorMap:
    def test__window_sticker_method_color_map__given_one_label_per_family__uses_canonical_colors(self):
        labels = ["Fixed Angles* (p=5)", "Linear Ramp (p=5)"]
        color_map = window_sticker_method_color_map(labels)
        assert color_map["Fixed Angles* (p=5)"] == "#4477AA"
        assert color_map["Linear Ramp (p=5)"] == "#CCBB44"

    def test__window_sticker_method_color_map__given_several_depths_of_one_family__shades_them_apart(self):
        # ARRANGE -- same family (Fixed Angles), three different depths
        labels = ["Fixed Angles* (p=5)", "Fixed Angles* (p=6)", "Fixed Angles* (p=7)"]

        # ACT
        color_map = window_sticker_method_color_map(labels)

        # ASSERT -- all three get distinct colors (shaded apart by depth)
        assert len({color_map[lbl] for lbl in labels}) == 3


# ---------------------------------------------------------------------------
# _ws_display_method_label
# ---------------------------------------------------------------------------

class TestWsDisplayMethodLabel:
    def test__ws_display_method_label__converts_latex_star_to_unicode_asterisk(self):
        assert _ws_display_method_label("Fixed Angles$^\\star$ (p=5)") == "Fixed Angles* (p=5)"

    def test__ws_display_method_label__converts_latex_dagger_to_unicode_dagger(self):
        assert _ws_display_method_label("Fixed Angles$^\\dagger$ (p=5)") == "Fixed Angles† (p=5)"

    def test__ws_display_method_label__given_no_marker__leaves_label_unchanged(self):
        assert _ws_display_method_label("Linear Ramp (p=5)") == "Linear Ramp (p=5)"


# ---------------------------------------------------------------------------
# _ensure_save_dir
# ---------------------------------------------------------------------------

class TestEnsureSaveDir:
    def test__ensure_save_dir__given_a_path__creates_it(self, tmp_path):
        target = tmp_path / "nested" / "plots"
        _ensure_save_dir(str(target))
        assert target.is_dir()

    def test__ensure_save_dir__given_none__is_a_no_op(self):
        _ensure_save_dir(None)  # must not raise


# ---------------------------------------------------------------------------
# _display_cross_strategy_envelope
# ---------------------------------------------------------------------------

class TestDisplayCrossStrategyEnvelope:
    def test__display_cross_strategy_envelope__envelope_is_the_running_max_across_methods(self):
        # ARRANGE -- "A" is flat at 0.5; "B" rises from 0.2 to 0.9 across the
        # same resource range, so the envelope should start near A's level
        # and end at B's higher level, non-decreasing throughout.
        curve_df = pd.DataFrame([
            {"method_label": "A", "resource": 1.0, "response_monotone": 0.5},
            {"method_label": "A", "resource": 10.0, "response_monotone": 0.5},
            {"method_label": "B", "resource": 1.0, "response_monotone": 0.2},
            {"method_label": "B", "resource": 10.0, "response_monotone": 0.9},
        ])

        # ACT
        envelope = _display_cross_strategy_envelope(curve_df, "resource", num_points=50)

        # ASSERT
        values = envelope["response_monotone"].to_numpy()
        assert values[0] == pytest.approx(0.5, abs=1e-6)
        assert values[-1] == pytest.approx(0.9, abs=1e-6)
        assert np.all(np.diff(values) >= -1e-9)  # non-decreasing (running max)

    def test__display_cross_strategy_envelope__given_empty_dataframe__returns_empty_with_expected_columns(self):
        result = _display_cross_strategy_envelope(pd.DataFrame(), "resource")
        assert result.empty
        assert list(result.columns) == ["resource", "response_monotone"]

    def test__display_cross_strategy_envelope__given_missing_required_column__returns_empty(self):
        result = _display_cross_strategy_envelope(pd.DataFrame({"resource": [1.0]}), "resource")
        assert result.empty


# ---------------------------------------------------------------------------
# _prepare_parameter_curve
# ---------------------------------------------------------------------------

class TestPrepareParameterCurve:
    def test__prepare_parameter_curve__collapses_duplicate_resources_with_median(self):
        df = pd.DataFrame([
            {"resource": 1.0, "N": 10.0, "M": 20.0},
            {"resource": 1.0, "N": 30.0, "M": 40.0},
            {"resource": 2.0, "N": 5.0, "M": 6.0},
        ])
        curve = _prepare_parameter_curve(df, resource_col="resource", parameter_cols=("N", "M"))
        curve = curve.set_index("resource")
        assert curve.loc[1.0, "N"] == pytest.approx(20.0)  # median of 10, 30

    def test__prepare_parameter_curve__drops_non_positive_resources(self):
        df = pd.DataFrame([
            {"resource": 0.0, "N": 10.0},
            {"resource": -1.0, "N": 10.0},
            {"resource": 2.0, "N": 5.0},
        ])
        curve = _prepare_parameter_curve(df, resource_col="resource", parameter_cols=("N",))
        assert list(curve["resource"]) == [2.0]

    def test__prepare_parameter_curve__given_missing_required_column__returns_empty(self):
        df = pd.DataFrame({"resource": [1.0]})  # no "N" column
        curve = _prepare_parameter_curve(df, resource_col="resource", parameter_cols=("N",))
        assert curve.empty


# ---------------------------------------------------------------------------
# resolve_result_root / read_summary_csv / read_first_summary_csv
# ---------------------------------------------------------------------------

class TestResolveResultRoot:
    def test__resolve_result_root__given_a_relative_tag__joins_it_under_results_base(self, tmp_path):
        result = resolve_result_root("my_tag", tmp_path)
        assert result == (tmp_path / "my_tag").resolve()

    def test__resolve_result_root__given_an_absolute_path__ignores_results_base(self, tmp_path):
        abs_path = tmp_path / "elsewhere"
        result = resolve_result_root(str(abs_path), tmp_path / "unrelated_base")
        assert result == abs_path.resolve()


class TestReadSummaryCsv:
    def test__read_summary_csv__given_an_existing_file__reads_it(self, tmp_path):
        (tmp_path / "summary.csv").write_text("a,b\n1,2\n")
        result = read_summary_csv(tmp_path, "summary.csv")
        assert list(result.columns) == ["a", "b"]
        assert result.iloc[0]["a"] == 1

    def test__read_summary_csv__given_a_missing_file__returns_empty_dataframe(self, tmp_path):
        result = read_summary_csv(tmp_path, "does_not_exist.csv")
        assert result.empty


class TestReadFirstSummaryCsv:
    def test__read_first_summary_csv__returns_the_first_present_file(self, tmp_path):
        (tmp_path / "second.csv").write_text("a\n1\n")
        result = read_first_summary_csv(tmp_path, ["first.csv", "second.csv"])
        assert list(result.columns) == ["a"]

    def test__read_first_summary_csv__given_none_present__returns_empty_dataframe(self, tmp_path):
        result = read_first_summary_csv(tmp_path, ["first.csv", "second.csv"])
        assert result.empty


# ---------------------------------------------------------------------------
# rebuild_strategy_budget_summary
# ---------------------------------------------------------------------------

class TestRebuildStrategyBudgetSummary:
    def test__rebuild_strategy_budget_summary__given_no_frontier_file__returns_empty(self, tmp_path):
        result = rebuild_strategy_budget_summary(tmp_path)
        assert result.empty


# ---------------------------------------------------------------------------
# load_multi_strategy_summaries
# ---------------------------------------------------------------------------

class TestLoadMultiStrategySummaries:
    def test__load_multi_strategy_summaries__given_a_missing_root__skips_it(self, tmp_path, capsys):
        # ARRANGE / ACT
        result = load_multi_strategy_summaries(["does_not_exist"], tmp_path)

        # ASSERT -- skipped, and reported as missing rather than raising
        assert result == []
        assert "does_not_exist" in capsys.readouterr().out

    def test__load_multi_strategy_summaries__given_a_valid_root__attaches_metadata_to_every_table(self, tmp_path):
        # ARRANGE
        root = tmp_path / "my_strategy"
        root.mkdir()
        (root / "strategy_budget_summary_train.csv").write_text(
            "method_label,T,response_mean,strategy,p\nFA_PP_opt,10,0.5,FA_PP_opt,5\n"
        )

        # ACT
        result = load_multi_strategy_summaries(["my_strategy"], tmp_path)

        # ASSERT
        assert len(result) == 1
        entry = result[0]
        assert entry["result_tag"] == "my_strategy"
        assert "p=5" in entry["method_label"]
        assert entry["strategy_budget"]["result_tag"].iloc[0] == "my_strategy"
        assert entry["actionable_lookup"].empty  # no matching CSV was written


# ---------------------------------------------------------------------------
# load_cost_model_panels
#
# Assembles the two panels of the resource-cost comparison figure from the
# re-costed campaign roots that run_latency_recost.py writes.
# ---------------------------------------------------------------------------

def _write_campaign_root(base, tag, suffix, resources=(1.0, 2.0), responses=(0.5, 0.6)):
    root = base / f"{tag}__{suffix}"
    root.mkdir(parents=True)
    rows = "\n".join(
        f"{r},{resp},FA_PP_opt,5" for r, resp in zip(resources, responses)
    )
    (root / "fitted_actionable_projection_test.csv").write_text(
        "resource,response,strategy,p\n" + rows + "\n"
    )
    return root


class TestLoadCostModelPanels:
    def _spec(self, title, calibrations):
        return {"title": title, "calibrations": calibrations}

    def test__load_cost_model_panels__fills_prescription_data_per_calibration(self, tmp_path):
        # ARRANGE -- two calibrations in one panel, each its own root suffix
        for suffix in ("prep0", "nc_prep0"):
            _write_campaign_root(tmp_path, "camp_a", suffix)

        # ACT
        panels = load_cost_model_panels(
            [self._spec("baseline", [
                {"label": "Noiseless", "suffix": "prep0"},
                {"label": "Noise-Corrected", "suffix": "nc_prep0"},
            ])],
            ["camp_a"], tmp_path, verbose=False,
        )

        # ASSERT
        calibrations = panels[0]["calibrations"]
        assert [c["label"] for c in calibrations] == ["Noiseless", "Noise-Corrected"]
        assert all(not c["prescription_df"].empty for c in calibrations)

    def test__load_cost_model_panels__builds_one_panel_per_spec(self, tmp_path):
        for suffix in ("prep0", "prep13p87s"):
            _write_campaign_root(tmp_path, "camp_a", suffix)
        panels = load_cost_model_panels(
            [self._spec("baseline", [{"label": "nl", "suffix": "prep0"}]),
             self._spec("charged", [{"label": "nl", "suffix": "prep13p87s"}])],
            ["camp_a"], tmp_path, verbose=False,
        )
        assert [p["title"] for p in panels] == ["baseline", "charged"]

    def test__load_cost_model_panels__given_a_suffix_with_no_roots__raises(self, tmp_path):
        _write_campaign_root(tmp_path, "camp_a", "prep0")
        with pytest.raises(FileNotFoundError):
            load_cost_model_panels(
                [self._spec("charged", [{"label": "nl", "suffix": "prep13p87s"}])],
                ["camp_a"], tmp_path, verbose=False,
            )

    def test__load_cost_model_panels__given_a_partial_set__continues_and_reports(self, tmp_path, capsys):
        _write_campaign_root(tmp_path, "camp_a", "prep0")
        panels = load_cost_model_panels(
            [self._spec("baseline", [{"label": "nl", "suffix": "prep0"}])],
            ["camp_a", "camp_b"], tmp_path,
        )
        assert len(panels) == 1
        assert "1/2 roots" in capsys.readouterr().out

    def test__load_cost_model_panels__excluded_tags_are_dropped(self, tmp_path, capsys):
        _write_campaign_root(tmp_path, "camp_a", "prep0")
        _write_campaign_root(tmp_path, "camp_interp", "prep0")
        load_cost_model_panels(
            [self._spec("baseline", [{"label": "nl", "suffix": "prep0"}])],
            ["camp_a", "camp_interp"], tmp_path, exclude_tags=["camp_interp"],
        )
        out = capsys.readouterr().out
        assert "1 roots" in out and "camp_interp" not in out

    def test__load_cost_model_panels__preserves_style_and_panel_extras(self, tmp_path):
        _write_campaign_root(tmp_path, "camp_a", "prep0")
        spec = self._spec("baseline", [
            {"label": "Noiseless", "suffix": "prep0", "linestyle": "--", "marker": "D"}])
        spec["hardware"] = {"label": "hw", "frontier_df": None}
        panels = load_cost_model_panels([spec], ["camp_a"], tmp_path, verbose=False)
        calibration = panels[0]["calibrations"][0]
        assert calibration["linestyle"] == "--" and calibration["marker"] == "D"
        assert panels[0]["hardware"]["label"] == "hw"


class TestDrawHardwareFrontierSteps:
    def _axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt.subplots()

    def _frontier(self):
        return pd.DataFrame({
            "dur_mean": [10.0, 100.0],
            "ar_mean": [0.85, 0.90],
            "dur_sem": [1.0, 5.0],
            "ar_sem": [0.01, 0.01],
            "method_label": ["FA (p=5)", "FA (p=6)"],
        })

    def test__draw_hardware_frontier_steps__returns_the_plotted_resources(self):
        fig, ax = self._axes()
        x = draw_hardware_frontier_steps(ax, self._frontier(), {"FA (p=5)": "r", "FA (p=6)": "b"})
        np.testing.assert_allclose(x, [10.0, 100.0])

    def test__draw_hardware_frontier_steps__extra_cost_shifts_every_point(self):
        # ARRANGE / ACT -- the charged panel bills the one submitted circuit
        fig, ax = self._axes()
        x = draw_hardware_frontier_steps(
            ax, self._frontier(), {"FA (p=5)": "r", "FA (p=6)": "b"}, extra_cost=13.87
        )

        # ASSERT
        np.testing.assert_allclose(x, [23.87, 113.87])

    def test__draw_hardware_frontier_steps__given_empty_frontier__draws_nothing(self):
        fig, ax = self._axes()
        x = draw_hardware_frontier_steps(ax, pd.DataFrame(), {})
        assert x.size == 0
        assert len(ax.lines) == 0

    def test__draw_hardware_frontier_steps__unknown_label_falls_back_to_a_default_colour(self):
        fig, ax = self._axes()
        x = draw_hardware_frontier_steps(ax, self._frontier(), {})  # empty colour map
        assert x.size == 2
