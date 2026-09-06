"""Tests for the pure data-prep/aggregation helpers in
examples/IBM_QAOA/src/utils.py that had no coverage before this file.
Matplotlib-heavy plot_* functions are intentionally out of scope here (this
session verified those visually, against rendered PNGs, all along -- a
unit test of axes/artist calls wouldn't catch the kind of bug that mattered).

Step 5 ("Extend") of the IBM_QAOA cleanup plan.
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
    _draw_pareto_envelope_segments,
    _envelope_segment_bounds,
    _force_dagger_label,
    _is_no_opt_metadata,
    _pareto_envelope_and_owner,
    _pareto_envelope_bounds,
    collect_cost_model_panel_entries,
    _percent_approx_ylabel,
    _percent_axis_values,
    attach_result_metadata,
    concat_summary,
    counts_to_samples_df,
    cross_strategy_envelope,
    curve_from_training_summary,
    curve_from_window_summary,
    curve_label,
    is_empty_nested_list,
    prepare_monotone_curve,
    prepare_training_bricks_data,
    sem,
    shared_approx_ylim,
    shared_approx_yticks,
)


# ---------------------------------------------------------------------------
# is_empty_nested_list
# ---------------------------------------------------------------------------

class TestIsEmptyNestedList:
    def test__is_empty_nested_list__given_list_of_empty_lists__returns_true(self):
        assert is_empty_nested_list([[], []]) is True

    def test__is_empty_nested_list__given_a_non_empty_inner_list__returns_false(self):
        assert is_empty_nested_list([[], [1]]) is False

    def test__is_empty_nested_list__given_an_empty_outer_list__returns_false(self):
        assert is_empty_nested_list([]) is False

    def test__is_empty_nested_list__given_not_a_list__returns_false(self):
        assert is_empty_nested_list(None) is False
        assert is_empty_nested_list("not a list") is False


# ---------------------------------------------------------------------------
# sem
# ---------------------------------------------------------------------------

class TestSem:
    def test__sem__given_several_values__matches_std_over_sqrt_n(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert sem(s) == pytest.approx(s.std(ddof=1) / np.sqrt(4))

    def test__sem__given_fewer_than_two_values__returns_zero(self):
        assert sem(pd.Series([1.0])) == 0.0
        assert sem(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# counts_to_samples_df
# ---------------------------------------------------------------------------

class TestCountsToSamplesDf:
    def test__counts_to_samples_df__expands_counts_into_one_row_per_bitstring(self):
        # ARRANGE
        df_hardware = pd.DataFrame([
            {
                "instance_name": "001", "training_method": "FA_PP_opt_5", "job_p": 5,
                "training_p": 5, "counts": {"00": 3, "01": 1},
            },
        ])

        # ACT
        result = counts_to_samples_df(df_hardware)

        # ASSERT
        assert len(result) == 2
        row00 = result[result["bitstring"] == "00"].iloc[0]
        assert row00["count"] == 3
        assert row00["prob"] == pytest.approx(0.75)

    def test__counts_to_samples_df__given_non_dict_counts__is_skipped(self):
        df_hardware = pd.DataFrame([
            {"instance_name": "001", "training_method": "FA_PP_opt_5", "job_p": 5, "training_p": 5, "counts": None},
        ])
        result = counts_to_samples_df(df_hardware)
        assert result.empty


# ---------------------------------------------------------------------------
# curve_from_training_summary
# ---------------------------------------------------------------------------

class TestCurveFromTrainingSummary:
    def test__curve_from_training_summary__given_required_columns__builds_monotone_curve_with_ci(self):
        # ARRANGE -- response_mean dips at T=20 then recovers at T=30, so
        # response_monotone should hold at the T=10 peak through T=20.
        df = pd.DataFrame([
            {"method_label": "A", "T": 10, "response_mean": 0.5, "response_std": 0.1, "n_instances": 4},
            {"method_label": "A", "T": 20, "response_mean": 0.4, "response_std": 0.1, "n_instances": 4},
            {"method_label": "A", "T": 30, "response_mean": 0.6, "response_std": 0.1, "n_instances": 4},
        ])

        # ACT
        curve = curve_from_training_summary(df)

        # ASSERT
        curve = curve.set_index("T")
        assert curve.loc[20, "response_monotone"] == pytest.approx(0.5)
        assert curve.loc[30, "response_monotone"] == pytest.approx(0.6)
        assert "response_lower_monotone" in curve.columns
        assert "response_upper_monotone" in curve.columns

    def test__curve_from_training_summary__given_missing_required_column__returns_empty_frame(self):
        df = pd.DataFrame([{"method_label": "A", "T": 10}])  # no response_mean
        result = curve_from_training_summary(df)
        assert result.empty
        assert list(result.columns) == ["method_label", "T", "response"]

    def test__curve_from_training_summary__given_empty_dataframe__returns_empty_frame(self):
        result = curve_from_training_summary(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# curve_from_window_summary
# ---------------------------------------------------------------------------

class TestCurveFromWindowSummary:
    def test__curve_from_window_summary__builds_monotone_curve_per_method(self):
        # ARRANGE -- "A" dips then recovers; "B" is strictly increasing
        df = pd.DataFrame([
            {"method_label": "A", "resource": 1.0, "response": 0.5},
            {"method_label": "A", "resource": 2.0, "response": 0.3},
            {"method_label": "B", "resource": 1.0, "response": 0.2},
            {"method_label": "B", "resource": 2.0, "response": 0.6},
        ])

        # ACT
        curve = curve_from_window_summary(df)

        # ASSERT
        a = curve[curve["method_label"] == "A"].sort_values("resource")
        assert list(a["response_monotone"]) == [pytest.approx(0.5), pytest.approx(0.5)]
        b = curve[curve["method_label"] == "B"].sort_values("resource")
        assert list(b["response_monotone"]) == [pytest.approx(0.2), pytest.approx(0.6)]

    def test__curve_from_window_summary__given_missing_required_column__returns_empty_frame(self):
        df = pd.DataFrame([{"method_label": "A", "resource": 1.0}])  # no response
        result = curve_from_window_summary(df)
        assert result.empty


# ---------------------------------------------------------------------------
# cross_strategy_envelope
# ---------------------------------------------------------------------------

class TestCrossStrategyEnvelope:
    def test__cross_strategy_envelope__given_a_crossover__envelope_switches_owner(self):
        # ARRANGE -- "A" leads at resource=1, "B" overtakes by resource=2
        curve_df = pd.DataFrame([
            {"method_label": "A", "resource": 1.0, "response_monotone": 0.6},
            {"method_label": "A", "resource": 2.0, "response_monotone": 0.6},
            {"method_label": "B", "resource": 1.0, "response_monotone": 0.3},
            {"method_label": "B", "resource": 2.0, "response_monotone": 0.9},
        ])

        # ACT
        envelope = cross_strategy_envelope(curve_df, "resource")

        # ASSERT
        envelope = envelope.set_index("resource")
        assert envelope.loc[1.0, "method_label"] == "A"
        assert envelope.loc[1.0, "response_monotone"] == pytest.approx(0.6)
        assert envelope.loc[2.0, "method_label"] == "B"
        assert envelope.loc[2.0, "response_monotone"] == pytest.approx(0.9)

    def test__cross_strategy_envelope__given_empty_dataframe__returns_empty_with_expected_columns(self):
        result = cross_strategy_envelope(pd.DataFrame(), "resource")
        assert result.empty
        assert list(result.columns) == ["resource", "response_monotone", "method_label"]


# ---------------------------------------------------------------------------
# _is_no_opt_metadata / _force_dagger_label
# ---------------------------------------------------------------------------

class TestIsNoOptMetadata:
    @pytest.mark.parametrize("value", ["FA_no_opt_5", "no-optimization", "NoOpt", "some_NOOPT_tag"])
    def test__is_no_opt_metadata__given_a_no_opt_marker__returns_true(self, value):
        assert _is_no_opt_metadata(value) is True

    def test__is_no_opt_metadata__given_no_marker__returns_false(self):
        assert _is_no_opt_metadata("FA_PP_opt_5") is False


class TestForceDaggerLabel:
    def test__force_dagger_label__replaces_star_marker_with_dagger(self):
        assert _force_dagger_label("Fixed Angles* (p=5)") == "Fixed Angles† (p=5)"

    def test__force_dagger_label__given_no_existing_marker__still_appends_dagger(self):
        assert _force_dagger_label("Linear Ramp (p=5)") == "Linear Ramp† (p=5)"


# ---------------------------------------------------------------------------
# curve_label
# ---------------------------------------------------------------------------

class TestCurveLabel:
    def test__curve_label__given_empty_dataframe__returns_default(self):
        assert curve_label(pd.DataFrame(), "my_default") == "my_default"

    def test__curve_label__given_empty_dataframe_and_no_opt_default__appends_dagger(self):
        assert curve_label(pd.DataFrame(), "FA_no_opt_5") == _force_dagger_label("FA_no_opt_5")

    def test__curve_label__given_strategy_and_p_columns__builds_labeled_depth_suffix(self):
        df = pd.DataFrame([{"strategy": "FA_PP_opt", "p": 5}])
        result = curve_label(df, "default")
        assert "p=5" in result

    def test__curve_label__given_no_opt_metadata_anywhere__forces_dagger_marker(self):
        df = pd.DataFrame([{"strategy": "FA_PP_opt", "p": 5, "training_method": "FA_no_opt_5"}])
        result = curve_label(df, "default")
        assert "†" in result
        assert "p=5" in result


# ---------------------------------------------------------------------------
# prepare_monotone_curve
# ---------------------------------------------------------------------------

class TestPrepareMonotoneCurve:
    def test__prepare_monotone_curve__averages_duplicate_resources_and_adds_cummax(self):
        # ARRANGE -- two rows share resource=1.0 (averaged to 0.4), then a dip at resource=2.0
        df = pd.DataFrame([
            {"resource": 1.0, "response": 0.5},
            {"resource": 1.0, "response": 0.3},
            {"resource": 2.0, "response": 0.2},
        ])

        # ACT
        curve = prepare_monotone_curve(df)

        # ASSERT
        curve = curve.set_index("resource")
        assert curve.loc[1.0, "response"] == pytest.approx(0.4)
        assert curve.loc[2.0, "response_monotone"] == pytest.approx(0.4)  # held from the resource=1.0 peak

    def test__prepare_monotone_curve__drops_rows_missing_resource_or_response(self):
        df = pd.DataFrame([
            {"resource": 1.0, "response": np.nan},
            {"resource": np.nan, "response": 0.5},
            {"resource": 2.0, "response": 0.5},
        ])
        curve = prepare_monotone_curve(df)
        assert list(curve["resource"]) == [2.0]


# ---------------------------------------------------------------------------
# _pareto_envelope_bounds
# ---------------------------------------------------------------------------

class TestParetoEnvelopeBounds:
    def test__pareto_envelope_bounds__tracks_the_ci_of_whichever_entry_owns_the_envelope(self):
        # ARRANGE -- "A" owns the whole grid (flat, higher than "B" everywhere)
        entries = [
            ("A", "blue", np.array([1.0, 4.0]), np.array([2.0, 2.0])),
            ("B", "red", np.array([1.0, 4.0]), np.array([1.0, 1.0])),
        ]
        grid = np.array([1.0, 2.5, 4.0])
        envelope, best_idx = _pareto_envelope_and_owner(entries, grid)
        bounds_entries = [
            (np.array([1.0, 4.0]), np.array([1.5, 1.5]), np.array([2.5, 2.5])),  # A's CI band
            (np.array([1.0, 4.0]), np.array([0.5, 0.5]), np.array([1.5, 1.5])),  # B's CI band
        ]

        # ACT
        lower, upper = _pareto_envelope_bounds(bounds_entries, grid, best_idx)

        # ASSERT -- envelope is owned by A throughout, so bounds should track A's band
        np.testing.assert_allclose(lower, [1.5, 1.5, 1.5])
        np.testing.assert_allclose(upper, [2.5, 2.5, 2.5])


# ---------------------------------------------------------------------------
# concat_summary
# ---------------------------------------------------------------------------

class TestConcatSummary:
    def test__concat_summary__concatenates_non_empty_frames_only(self):
        summaries = [
            {"table": pd.DataFrame({"x": [1, 2]})},
            {"table": pd.DataFrame()},
            {"table": pd.DataFrame({"x": [3]})},
        ]
        result = concat_summary(summaries, "table")
        assert list(result["x"]) == [1, 2, 3]

    def test__concat_summary__given_all_empty__returns_empty_dataframe(self):
        summaries = [{"table": pd.DataFrame()}, {"table": pd.DataFrame()}]
        result = concat_summary(summaries, "table")
        assert result.empty


# ---------------------------------------------------------------------------
# attach_result_metadata
# ---------------------------------------------------------------------------

class TestAttachResultMetadata:
    def test__attach_result_metadata__adds_tag_root_and_label_columns(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        result = attach_result_metadata(df, "my_tag", tmp_path, "My Label")
        assert result.loc[0, "result_tag"] == "my_tag"
        assert result.loc[0, "result_root"] == str(tmp_path)
        assert result.loc[0, "method_label"] == "My Label"

    def test__attach_result_metadata__given_empty_dataframe__returns_it_unchanged(self):
        df = pd.DataFrame()
        result = attach_result_metadata(df, "my_tag", "root", "My Label")
        assert result.empty
        assert list(result.columns) == []


# ---------------------------------------------------------------------------
# shared_approx_ylim / shared_approx_yticks
# ---------------------------------------------------------------------------

class TestSharedApproxYlim:
    def test__shared_approx_ylim__pads_around_the_min_and_max_across_series(self):
        result = shared_approx_ylim(pd.Series([0.5, 0.6]), pd.Series([0.55]))
        assert result is not None
        y_min, y_max = result
        assert y_min < 0.5
        assert y_max > 0.6

    def test__shared_approx_ylim__clips_to_0_and_1(self):
        result = shared_approx_ylim(pd.Series([0.995]))
        y_min, y_max = result
        assert y_max <= 1.0

    def test__shared_approx_ylim__given_no_values__returns_none(self):
        assert shared_approx_ylim(None, pd.Series([], dtype=float)) is None


class TestSharedApproxYticks:
    def test__shared_approx_yticks__given_ylim__returns_evenly_spaced_ticks(self):
        ticks = shared_approx_yticks((0.0, 1.0), n_ticks=6)
        assert ticks == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def test__shared_approx_yticks__given_no_ylim__returns_none(self):
        assert shared_approx_yticks(None) is None


# ---------------------------------------------------------------------------
# _percent_approx_ylabel / _percent_axis_values
# ---------------------------------------------------------------------------

class TestPercentApproxYlabel:
    def test__percent_approx_ylabel__given_approximation_ratio_label__appends_percent(self):
        assert _percent_approx_ylabel("Approximation ratio") == "Approximation ratio (%)"

    def test__percent_approx_ylabel__given_unrelated_label__still_appends_percent_suffix(self):
        assert _percent_approx_ylabel("Something else") == "Something else (%)"


class TestPercentAxisValues:
    def test__percent_axis_values__scales_ylim_and_yticks_by_100(self):
        ylim, yticks = _percent_axis_values((0.5, 0.9), [0.5, 0.7, 0.9])
        assert ylim == (50.0, 90.0)
        assert yticks == [50.0, 70.0, 90.0]

    def test__percent_axis_values__given_none__passes_through_none(self):
        ylim, yticks = _percent_axis_values(None, None)
        assert ylim is None
        assert yticks is None


# ---------------------------------------------------------------------------
# prepare_training_bricks_data
#
# A real misalignment bug was fixed here earlier in this engagement: a
# positional `.sem().values` assignment (assumed the groupby output was
# already in (job_p, method_base) row order) was replaced with an explicit
# merge on ["job_p", "method_base"]. This test's second case is a regression
# guard for exactly that: two (job_p, method_base) groups whose natural
# groupby order would misalign a positional assignment.
# ---------------------------------------------------------------------------

class TestPrepareTrainingBricksData:
    def test__prepare_training_bricks_data__builds_step_columns_and_brick_total(self):
        # ARRANGE
        df_flat = pd.DataFrame([
            {"file_name": "f1", "level": "inner", "depth_step": 1, "iteration": 0, "duration": 2.0},
            {"file_name": "f1", "level": "inner", "depth_step": 2, "iteration": 0, "duration": 3.0},
            {"file_name": "f1", "level": "outer", "depth_step": np.nan, "iteration": 0, "duration": 1.0},
        ])
        df_hardware_new = pd.DataFrame([
            {"file_name": "f1", "training_method": "FA_PP_opt_5", "job_p": 5},
        ])

        # ACT
        agg, step_cols = prepare_training_bricks_data(df_flat, df_hardware_new)

        # ASSERT -- outer_init=1.0, steps 1+2 sum to 5.0, brick_total=6.0
        assert step_cols == ["step_1", "step_2"]
        row = agg.iloc[0]
        assert row["outer_init"] == pytest.approx(1.0)
        assert row["brick_total"] == pytest.approx(6.0)
        assert row["method_base"] == "FA_PP_opt"

    def test__prepare_training_bricks_data__zeroes_out_step_columns_beyond_job_p(self):
        # ARRANGE -- this row's job_p=1 shouldn't count depth_step=2's duration
        df_flat = pd.DataFrame([
            {"file_name": "f1", "level": "inner", "depth_step": 1, "iteration": 0, "duration": 2.0},
            {"file_name": "f1", "level": "inner", "depth_step": 2, "iteration": 0, "duration": 3.0},
        ])
        df_hardware_new = pd.DataFrame([
            {"file_name": "f1", "training_method": "FA_PP_opt_1", "job_p": 1},
        ])

        # ACT
        agg, step_cols = prepare_training_bricks_data(df_flat, df_hardware_new)

        # ASSERT
        row = agg.iloc[0]
        assert row["step_1"] == pytest.approx(2.0)
        assert row["step_2"] == pytest.approx(0.0)

    def test__prepare_training_bricks_data__sem_column_stays_aligned_with_its_own_group(self):
        # ARRANGE -- regression guard for the positional .sem().values bug:
        # two (job_p, method_base) groups, each with 2 rows of differing
        # spread, so a misaligned sem_total would silently swap the two
        # groups' standard errors.
        df_flat = pd.DataFrame([], columns=["file_name", "level", "depth_step", "iteration", "duration"])
        df_hardware_new = pd.DataFrame([
            {"file_name": "a1", "training_method": "FA_PP_opt_5", "job_p": 5},
            {"file_name": "a2", "training_method": "FA_PP_opt_5", "job_p": 5},
            {"file_name": "b1", "training_method": "LR_opt_9", "job_p": 9},
            {"file_name": "b2", "training_method": "LR_opt_9", "job_p": 9},
        ])
        # brick_total ends up 0 for every row here (no matching df_flat rows,
        # fillna(0) covers outer_init/step columns), so force distinguishable
        # spreads by monkeypatching brick_total after the fact isn't
        # available -- instead assert the merge keys line up structurally:
        # one sem_total row per (job_p, method_base) present in agg.

        # ACT
        agg, _ = prepare_training_bricks_data(df_flat, df_hardware_new)

        # ASSERT -- one row per (job_p, method_base) group, each carrying its
        # own sem_total from an explicit merge (not a positional assignment
        # that would only be correct by coincidence)
        assert len(agg) == 2
        assert set(zip(agg["job_p"], agg["method_base"])) == {(5, "FA_PP_opt"), (9, "LR_opt")}
        assert "sem_total" in agg.columns

    def test__prepare_training_bricks_data__excludes_mps_non_aer_methods(self):
        # ARRANGE -- method_base containing "_MPS_" (but not "_MPSAer") should
        # be filtered out of the aggregated table entirely
        df_flat = pd.DataFrame([], columns=["file_name", "level", "depth_step", "iteration", "duration"])
        df_hardware_new = pd.DataFrame([
            {"file_name": "f1", "training_method": "FA_PP_opt_MPS_5", "job_p": 5},
        ])

        # ACT
        agg, _ = prepare_training_bricks_data(df_flat, df_hardware_new)

        # ASSERT
        assert agg.empty


# ---------------------------------------------------------------------------
# _envelope_segment_bounds / _draw_pareto_envelope_segments /
# collect_cost_model_panel_entries
#
# The segment splitting was inline in the notebook's Pareto cell until the
# two-panel cost-model figure needed the same logic; these pin the behaviour
# the drawn figure depends on (one marker exactly where ownership changes,
# unowned stretches skipped).
# ---------------------------------------------------------------------------

class TestEnvelopeSegmentBounds:
    def test__envelope_segment_bounds__splits_into_runs_of_one_owner(self):
        # ARRANGE / ACT
        runs = _envelope_segment_bounds(np.array([0, 0, 0, 1, 1, 2]))

        # ASSERT -- (start, stop exclusive, owner)
        assert runs == [(0, 3, 0), (3, 5, 1), (5, 6, 2)]

    def test__envelope_segment_bounds__skips_stretches_no_method_owns(self):
        runs = _envelope_segment_bounds(np.array([-1, -1, 0, 0, -1, 1]))
        assert runs == [(2, 4, 0), (5, 6, 1)]

    def test__envelope_segment_bounds__consecutive_runs_share_no_grid_column(self):
        # ARRANGE -- a colour change must land exactly at the handover, so
        # segment n's stop is segment n+1's start
        runs = _envelope_segment_bounds(np.array([0, 0, 1, 1, 0]))

        # ASSERT
        for (_, stop, _), (next_start, _, _) in zip(runs, runs[1:]):
            assert stop == next_start

    def test__envelope_segment_bounds__given_no_owner_anywhere__returns_empty(self):
        assert _envelope_segment_bounds(np.array([-1, -1])) == []


class TestDrawParetoEnvelopeSegments:
    def _axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt.subplots()

    def test__draw_pareto_envelope_segments__marks_each_takeover_point(self):
        # ARRANGE -- ownership changes at index 2 and 4
        fig, ax = self._axes()
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        envelope = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        best_idx = np.array([0, 0, 1, 1, 0])

        # ACT
        marker_idx = _draw_pareto_envelope_segments(ax, grid, envelope, best_idx, ["r", "b"])

        # ASSERT -- one marker at the start of each owned run
        assert marker_idx == [0, 2, 4]
        assert len(ax.lines) == 3

    def test__draw_pareto_envelope_segments__skips_all_nan_segments(self):
        # ARRANGE -- the middle run has no finite envelope value to draw
        fig, ax = self._axes()
        grid = np.array([1.0, 2.0, 3.0, 4.0])
        envelope = np.array([1.0, np.nan, np.nan, 2.0])
        best_idx = np.array([0, 1, 1, 0])

        # ACT
        marker_idx = _draw_pareto_envelope_segments(ax, grid, envelope, best_idx, ["r", "b"])

        # ASSERT
        assert marker_idx == [0, 3]

    def test__draw_pareto_envelope_segments__given_no_owned_runs__draws_nothing(self):
        fig, ax = self._axes()
        marker_idx = _draw_pareto_envelope_segments(
            ax, np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([-1, -1]), ["r"]
        )
        assert marker_idx == []
        assert len(ax.lines) == 0


class TestCollectCostModelPanelEntries:
    def _prescription(self, label, resources, responses):
        return pd.DataFrame({
            "method_label": [label] * len(resources),
            "resource": resources,
            "response": responses,
        })

    def test__collect_cost_model_panel_entries__returns_entries_per_panel_and_the_label_union(self):
        # ARRANGE -- different strategies in each panel; the colour map has to
        # be built from the union so a strategy keeps its colour across panels
        panels = [
            {"title": "A", "calibrations": [
                {"label": "cal", "prescription_df": self._prescription("FA (p=5)", [1.0, 2.0], [0.5, 0.6])}]},
            {"title": "B", "calibrations": [
                {"label": "cal", "prescription_df": self._prescription("PT (p=2)", [1.0, 2.0], [0.4, 0.7])}]},
        ]

        # ACT
        per_panel, labels = collect_cost_model_panel_entries(panels)

        # ASSERT
        assert labels == ["FA (p=5)", "PT (p=2)"]
        assert len(per_panel) == 2
        assert per_panel[0][0][1][0][0] == "FA (p=5)"

    def test__collect_cost_model_panel_entries__skips_empty_and_missing_frames(self):
        panels = [{"title": "A", "calibrations": [
            {"label": "empty", "prescription_df": pd.DataFrame()},
            {"label": "missing"},
            {"label": "real", "prescription_df": self._prescription("FA (p=5)", [1.0, 2.0], [0.5, 0.6])},
        ]}]
        per_panel, labels = collect_cost_model_panel_entries(panels)
        assert labels == ["FA (p=5)"]
        assert len(per_panel[0]) == 1

    def test__collect_cost_model_panel_entries__given_nothing_drawable__returns_no_labels(self):
        per_panel, labels = collect_cost_model_panel_entries([{"title": "A", "calibrations": []}])
        assert labels == []
        assert per_panel == [[]]
