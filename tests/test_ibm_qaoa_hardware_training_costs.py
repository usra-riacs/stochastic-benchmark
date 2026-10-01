"""Tests for examples/IBM_QAOA/src/Processing.py's QAOAHardware class and the
resolve_hardware_training_costs pipeline (its 8 private helpers plus the
public entry point). None of this had any test coverage before this file:
it's the code behind the notebook's _build_hw_frontier / _best_bitstring_ar
pipeline that the whole hardware-overlay Pareto plot this session's work
revolved around depends on.

Step 5 ("Extend") of the IBM_QAOA cleanup plan.
"""
import json
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

from src.Processing import (  # noqa: E402
    QAOAHardware,
    _as_finite_float,
    _build_inner_duration_tables,
    _build_stage_manifest,
    _choose_stage_for_method,
    _max_abs_angle_diff,
    _normalize_angle_list,
    _resolve_inner_duration,
    _resolve_training_stage,
    resolve_hardware_training_costs,
)


# ---------------------------------------------------------------------------
# QAOAHardware.locate_hardware_instance
# ---------------------------------------------------------------------------

class TestLocateHardwareInstance:
    def test__locate_hardware_instance__given_heavy_hex__matches_by_instance_nodes_and_depth(self, tmp_path):
        # ARRANGE
        expected = tmp_path / "001N144HH_seed3_5_job.json"
        expected.touch()
        (tmp_path / "001N144HH_seed3_6_job.json").touch()  # different depth, shouldn't match

        # ACT
        result = QAOAHardware.locate_hardware_instance(str(tmp_path), "heavy_hex", "1", "144", "5")

        # ASSERT
        assert result == [expected]

    def test__locate_hardware_instance__given_unknown_graph_type__returns_empty_list(self, tmp_path):
        # ARRANGE / ACT
        result = QAOAHardware.locate_hardware_instance(str(tmp_path), "not_a_graph_type", "1", "144", "5")

        # ASSERT
        assert result == []


# ---------------------------------------------------------------------------
# QAOAHardware.load_hardware_instance
# ---------------------------------------------------------------------------

def _objective(bitstring, _context):
    return {"00": 0.0, "01": 1.0}.get(bitstring, 0.0)


@pytest.fixture
def hardware_json_path(tmp_path):
    def _write(records, filename="001N144HH_seed3_5_job.json"):
        path = tmp_path / filename
        path.write_text(json.dumps(records))
        return path
    return _write


class TestLoadHardwareInstance:
    def test__load_hardware_instance__given_one_valid_record__parses_it_and_extracts_job_p(self, hardware_json_path):
        # ARRANGE
        records = [
            {
                "total_time": 10.0,
                "num_shots": 100,
                "metadata": {
                    "circuit_metadata": {
                        "eval_energy": True,
                        "short_name": "001HH",
                        "problem_class": "MaxCut",
                        "method": "FA_PP_opt_5",
                        "result_file": "001_FA_PP_opt_5.json",
                        "params": [0.1, 0.2],
                        "trainer": "FixedAngleConjecture",
                    },
                },
                "counts": {"00": 50, "01": 50},
            },
        ]
        path = hardware_json_path(records)

        # ACT
        result = QAOAHardware.load_hardware_instance(path, objective_from_bitstring=_objective)

        # ASSERT -- job_p comes from the "_5_" in the filename
        assert len(result) == 1
        run = result[0]
        assert run.job_p == 5
        assert run.training_p == 5  # trailing digit of "FA_PP_opt_5"
        assert run.instance_name == "001HH"
        assert run.QPU_time == pytest.approx(10.0)  # only one valid record, so no splitting
        assert run.expected_energy == pytest.approx(0.5)  # mean of 0.0 and 1.0

    def test__load_hardware_instance__given_several_valid_records__splits_qpu_time_evenly(self, hardware_json_path):
        # ARRANGE -- job-level total_time is on record 0; two valid circuit records
        records = [
            {
                "total_time": 10.0,
                "num_shots": 100,
                "metadata": {
                    "circuit_metadata": {
                        "eval_energy": True,
                        "short_name": "001HH",
                        "problem_class": "MaxCut",
                        "method": "FA_PP_opt_5",
                        "result_file": "001_FA_PP_opt_5.json",
                    },
                },
                "counts": {"00": 1},
            },
            {
                "metadata": {
                    "circuit_metadata": {
                        "eval_energy": True,
                        "short_name": "001HH",
                        "problem_class": "MaxCut",
                        "method": "FA_PP_opt_6",
                        "result_file": "001_FA_PP_opt_6.json",
                    },
                },
                "counts": {"01": 1},
            },
        ]
        path = hardware_json_path(records)

        # ACT
        result = QAOAHardware.load_hardware_instance(path, objective_from_bitstring=_objective)

        # ASSERT
        assert len(result) == 2
        assert all(run.QPU_time == pytest.approx(5.0) for run in result)

    def test__load_hardware_instance__given_record_missing_eval_energy__is_skipped(self, hardware_json_path):
        # ARRANGE -- circuit_metadata present but no "eval_energy" key
        records = [
            {
                "total_time": 10.0,
                "metadata": {
                    "circuit_metadata": {
                        "short_name": "001HH",
                        "problem_class": "MaxCut",
                        "method": "FA_PP_opt_5",
                        "result_file": "001_FA_PP_opt_5.json",
                    },
                },
                "counts": {"00": 1},
            },
        ]
        path = hardware_json_path(records)

        # ACT
        result = QAOAHardware.load_hardware_instance(path, objective_from_bitstring=_objective)

        # ASSERT
        assert result == []

    def test__load_hardware_instance__given_record_with_empty_counts__is_skipped(self, hardware_json_path):
        # ARRANGE
        records = [
            {
                "total_time": 10.0,
                "metadata": {
                    "circuit_metadata": {
                        "eval_energy": True,
                        "short_name": "001HH",
                        "problem_class": "MaxCut",
                        "method": "FA_PP_opt_5",
                        "result_file": "001_FA_PP_opt_5.json",
                    },
                },
                "counts": {},
            },
        ]
        path = hardware_json_path(records)

        # ACT
        result = QAOAHardware.load_hardware_instance(path, objective_from_bitstring=_objective)

        # ASSERT
        assert result == []

    def test__load_hardware_instance__given_method_without_trailing_digit__falls_back_to_result_file_suffix(self, hardware_json_path):
        # ARRANGE -- method has no trailing digit, so training_p must come
        # from the "_<p>.json" suffix on result_file instead.
        records = [
            {
                "total_time": 10.0,
                "metadata": {
                    "circuit_metadata": {
                        "eval_energy": True,
                        "short_name": "001HH",
                        "problem_class": "MaxCut",
                        "method": "FA_PP_opt",
                        "result_file": "001_FA_PP_opt_7.json",
                    },
                },
                "counts": {"00": 1},
            },
        ]
        path = hardware_json_path(records)

        # ACT
        result = QAOAHardware.load_hardware_instance(path, objective_from_bitstring=_objective)

        # ASSERT
        assert result[0].training_p == 7


# ---------------------------------------------------------------------------
# _normalize_angle_list
# ---------------------------------------------------------------------------

class TestNormalizeAngleList:
    def test__normalize_angle_list__given_numeric_list__returns_float_list(self):
        # ARRANGE / ACT
        result = _normalize_angle_list([0, "1.5", 2])

        # ASSERT
        assert result == [0.0, 1.5, 2.0]

    def test__normalize_angle_list__given_non_numeric_entry__returns_none(self):
        # ARRANGE / ACT
        result = _normalize_angle_list([0.1, "not-a-number"])

        # ASSERT
        assert result is None

    def test__normalize_angle_list__given_not_a_list__returns_none(self):
        # ARRANGE / ACT / ASSERT
        assert _normalize_angle_list(None) is None
        assert _normalize_angle_list("0.1,0.2") is None


# ---------------------------------------------------------------------------
# _max_abs_angle_diff
# ---------------------------------------------------------------------------

class TestMaxAbsAngleDiff:
    def test__max_abs_angle_diff__given_two_lists__returns_largest_elementwise_diff(self):
        # ARRANGE / ACT
        result = _max_abs_angle_diff([0.1, 0.5], [0.1, 0.3])

        # ASSERT
        assert result == pytest.approx(0.2)

    def test__max_abs_angle_diff__given_either_side_none_or_empty__returns_nan(self):
        # ARRANGE / ACT / ASSERT
        assert np.isnan(_max_abs_angle_diff(None, [0.1]))
        assert np.isnan(_max_abs_angle_diff([0.1], None))
        assert np.isnan(_max_abs_angle_diff([], [0.1]))


# ---------------------------------------------------------------------------
# _as_finite_float
# ---------------------------------------------------------------------------

class TestAsFiniteFloat:
    def test__as_finite_float__given_numeric_string__returns_float(self):
        assert _as_finite_float("3.5") == pytest.approx(3.5)

    def test__as_finite_float__given_none_or_non_numeric__returns_nan(self):
        assert np.isnan(_as_finite_float(None))
        assert np.isnan(_as_finite_float("not-a-number"))

    def test__as_finite_float__given_infinite_value__returns_nan(self):
        assert np.isnan(_as_finite_float(float("inf")))


# ---------------------------------------------------------------------------
# _choose_stage_for_method
# ---------------------------------------------------------------------------

def _stage_candidate(stage, trainer_name="FixedAngleConjecture", physical_training_file="a.json"):
    return pd.Series({
        "stage": stage,
        "trainer_name": trainer_name,
        "physical_training_file": physical_training_file,
    })


class TestChooseStageForMethod:
    def test__choose_stage_for_method__given_no_candidates__returns_none_and_no_notes(self):
        chosen, notes = _choose_stage_for_method([], "FA_PP_opt_5")
        assert chosen is None
        assert notes == []

    def test__choose_stage_for_method__given_one_candidate__returns_it_unconditionally(self):
        candidate = _stage_candidate(0)
        chosen, notes = _choose_stage_for_method([candidate], "anything")
        assert chosen is candidate
        assert notes == []

    def test__choose_stage_for_method__given_duplicate_stage_and_trainer__resolves_by_filename(self):
        # ARRANGE -- same (stage, trainer_name), differing only by physical file
        candidates = [_stage_candidate(0, physical_training_file="b.json"), _stage_candidate(0, physical_training_file="a.json")]

        # ACT
        chosen, notes = _choose_stage_for_method(candidates, "anything")

        # ASSERT -- lexicographically smallest physical_training_file wins
        assert chosen["physical_training_file"] == "a.json"
        assert notes == ["resolved_duplicate_physical_files"]

    def test__choose_stage_for_method__given_no_opt_method__picks_the_smallest_stage(self):
        candidates = [_stage_candidate(2), _stage_candidate(0), _stage_candidate(1)]
        chosen, notes = _choose_stage_for_method(candidates, "FA_no_opt_5")
        assert chosen["stage"] == 0
        assert notes == ["resolved_by_method_suffix"]

    @pytest.mark.parametrize("method", ["FA_PP_angleOpt_5", "LR_opt_p5", "FA_PP_opt"])
    def test__choose_stage_for_method__given_opt_method__picks_the_largest_stage(self, method):
        candidates = [_stage_candidate(2), _stage_candidate(0), _stage_candidate(1)]
        chosen, notes = _choose_stage_for_method(candidates, method)
        assert chosen["stage"] == 2
        assert notes == ["resolved_by_method_suffix"]

    def test__choose_stage_for_method__given_ambiguous_method_and_stages__returns_none(self):
        # ARRANGE -- distinct stages, no _opt/_no_opt/angleOpt marker in the method
        candidates = [_stage_candidate(0), _stage_candidate(1)]

        # ACT
        chosen, notes = _choose_stage_for_method(candidates, "Interp")

        # ASSERT
        assert chosen is None
        assert notes == []


# ---------------------------------------------------------------------------
# _build_stage_manifest
# ---------------------------------------------------------------------------

class TestBuildStageManifest:
    def test__build_stage_manifest__given_ordered_stages__accumulates_outer_duration(self):
        # ARRANGE -- stage_summaries out of order on purpose to confirm sorting
        df_training = pd.DataFrame([
            {
                "file_name": "f1",
                "physical_training_file": "f1_phys.json",
                "pre_processing_time": 1.0,
                "stage_summaries": [
                    {"stage": 1, "train_duration": 2.0, "trainer_name": "T1", "optimized_qaoa_angles": [0.2]},
                    {"stage": 0, "train_duration": 3.0, "trainer_name": "T0", "optimized_qaoa_angles": [0.1]},
                ],
            },
        ])

        # ACT
        manifest = _build_stage_manifest(df_training)

        # ASSERT -- stage 0 processed first: running_outer = 1.0 + 3.0 = 4.0;
        # stage 1 next: running_outer = 4.0 + 2.0 = 6.0
        manifest = manifest.set_index("stage")
        assert manifest.loc[0, "outer_init_duration_sum"] == pytest.approx(4.0)
        assert manifest.loc[1, "outer_init_duration_sum"] == pytest.approx(6.0)

    def test__build_stage_manifest__given_missing_pre_processing_time__defaults_to_zero(self):
        # ARRANGE
        df_training = pd.DataFrame([
            {
                "file_name": "f1",
                "physical_training_file": "f1_phys.json",
                "pre_processing_time": np.nan,
                "stage_summaries": [
                    {"stage": 0, "train_duration": 5.0, "trainer_name": "T0", "optimized_qaoa_angles": [0.1]},
                ],
            },
        ])

        # ACT
        manifest = _build_stage_manifest(df_training)

        # ASSERT
        assert manifest.loc[0, "outer_init_duration_sum"] == pytest.approx(5.0)

    def test__build_stage_manifest__given_non_list_stage_summaries__contributes_no_rows(self):
        # ARRANGE
        df_training = pd.DataFrame([
            {"file_name": "f1", "physical_training_file": "f1.json", "pre_processing_time": 0.0, "stage_summaries": np.nan},
        ])

        # ACT
        manifest = _build_stage_manifest(df_training)

        # ASSERT
        assert manifest.empty

    def test__build_stage_manifest__given_stage_missing_train_duration__is_skipped(self):
        # ARRANGE
        df_training = pd.DataFrame([
            {
                "file_name": "f1",
                "physical_training_file": "f1.json",
                "pre_processing_time": 0.0,
                "stage_summaries": [
                    {"stage": 0, "trainer_name": "T0", "optimized_qaoa_angles": [0.1]},  # no train_duration
                    {"stage": 1, "train_duration": 4.0, "trainer_name": "T1", "optimized_qaoa_angles": [0.2]},
                ],
            },
        ])

        # ACT
        manifest = _build_stage_manifest(df_training)

        # ASSERT -- only stage 1 makes it in, and running_outer wasn't bumped by the skipped stage
        assert list(manifest["stage"]) == [1]
        assert manifest.iloc[0]["outer_init_duration_sum"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# _build_inner_duration_tables
# ---------------------------------------------------------------------------

class TestBuildInnerDurationTables:
    def test__build_inner_duration_tables__given_inner_rows__cumsums_within_iteration(self):
        # ARRANGE -- two depth steps within iteration 0, one within iteration 1
        df_flat = pd.DataFrame([
            {"file_name": "f1", "physical_training_file": "p1", "level": "inner", "iteration": 0, "stage": 0, "depth_step": 2, "duration": 1.0},
            {"file_name": "f1", "physical_training_file": "p1", "level": "inner", "iteration": 0, "stage": 0, "depth_step": 3, "duration": 2.0},
            {"file_name": "f1", "physical_training_file": "p1", "level": "inner", "iteration": 1, "stage": 1, "depth_step": 2, "duration": 5.0},
            {"file_name": "f1", "physical_training_file": "p1", "level": "outer", "iteration": 0, "stage": 0, "depth_step": np.nan, "duration": 9.0},
        ])

        # ACT
        inner, inner_totals = _build_inner_duration_tables(df_flat)

        # ASSERT -- depth_step 3 cumulates on top of depth_step 2 within iteration 0
        def _cum(iteration, depth_step):
            row = inner[(inner["iteration"] == iteration) & (inner["depth_step"] == depth_step)]
            return row["inner_cum_within_stage"].iloc[0]

        assert _cum(0, 2) == pytest.approx(1.0)
        assert _cum(0, 3) == pytest.approx(3.0)

        totals = inner_totals.set_index("iteration")
        assert totals.loc[0, "inner_stage_total"] == pytest.approx(3.0)
        assert totals.loc[1, "inner_stage_total"] == pytest.approx(5.0)

    def test__build_inner_duration_tables__given_no_inner_rows__returns_empty_tables(self):
        # ARRANGE
        df_flat = pd.DataFrame([
            {"file_name": "f1", "physical_training_file": "p1", "level": "outer", "iteration": 0, "stage": 0, "depth_step": np.nan, "duration": 9.0},
        ])

        # ACT
        inner, inner_totals = _build_inner_duration_tables(df_flat)

        # ASSERT
        assert inner.empty
        assert inner_totals.empty


# ---------------------------------------------------------------------------
# _resolve_training_stage
# ---------------------------------------------------------------------------

@pytest.fixture
def stage_manifest():
    return pd.DataFrame([
        {
            "file_name": "f1", "physical_training_file": "p1.json", "stage": 0,
            "trainer_name": "T0", "optimized_qaoa_angles": [0.1, 0.2],
            "outer_duration": 1.0, "outer_init_duration_sum": 1.0,
        },
        {
            "file_name": "f1", "physical_training_file": "p1.json", "stage": 1,
            "trainer_name": "T1", "optimized_qaoa_angles": [0.1, 0.2, 0.3, 0.4],
            "outer_duration": 2.0, "outer_init_duration_sum": 3.0,
        },
    ])


class TestResolveTrainingStage:
    def test__resolve_training_stage__given_no_candidates__returns_missing_training_file(self, stage_manifest):
        hw_row = pd.Series({"file_name": "no_such_file", "training_method": "FA_PP_opt_5", "params": [0.1]})
        result = _resolve_training_stage(hw_row, stage_manifest)
        assert result["training_match_status"] == "missing_training_file"

    def test__resolve_training_stage__given_missing_params__returns_no_stage_match(self, stage_manifest):
        hw_row = pd.Series({"file_name": "f1", "training_method": "FA_PP_opt_5", "params": None})
        result = _resolve_training_stage(hw_row, stage_manifest)
        assert result["training_match_status"] == "no_stage_match"
        assert result["training_match_note"] == "hardware params missing"

    def test__resolve_training_stage__given_exact_angle_match__matches_that_stage(self, stage_manifest):
        # ARRANGE -- params match stage 0's angles exactly
        hw_row = pd.Series({"file_name": "f1", "training_method": "FA_PP_opt_5", "params": [0.1, 0.2], "job_p": 2, "training_p": 2})

        # ACT
        result = _resolve_training_stage(hw_row, stage_manifest)

        # ASSERT
        assert result["training_match_status"] == "matched"
        assert result["matched_stage"] == 0
        assert result["outer_init_duration_sum"] == pytest.approx(1.0)

    def test__resolve_training_stage__given_no_angle_match_and_not_depth_prefix__returns_no_stage_match(self, stage_manifest):
        # ARRANGE -- params don't match any stage's angles, and training_p <= job_p
        # so the depth-prefix fallback doesn't apply
        hw_row = pd.Series({"file_name": "f1", "training_method": "FA_PP_opt_5", "params": [9.9, 9.9], "job_p": 5, "training_p": 5})

        # ACT
        result = _resolve_training_stage(hw_row, stage_manifest)

        # ASSERT
        assert result["training_match_status"] == "no_stage_match"
        assert result["training_match_note"] == "no parameter match"

    def test__resolve_training_stage__given_depth_prefix_row__falls_back_to_prefix_match(self, stage_manifest):
        # ARRANGE -- 2-angle params don't equal any stage's angles, but
        # training_p > job_p marks this as a depth-prefix row, and stage 1's
        # 4-angle list is at least as long as params so it qualifies as a
        # prefix candidate. Method ends in "_opt" so the max-stage branch of
        # _choose_stage_for_method picks stage 1 (the only candidate anyway).
        hw_row = pd.Series({"file_name": "f1", "training_method": "FA_PP_opt", "params": [9.9, 9.9], "job_p": 2, "training_p": 10})

        # ACT
        result = _resolve_training_stage(hw_row, stage_manifest)

        # ASSERT
        assert result["training_match_status"] == "matched"
        assert result["matched_stage"] == 1
        assert "depth_prefix_match_without_angle_equality" in result["training_match_note"]

    def test__resolve_training_stage__given_duplicate_physical_files__notes_it(self, stage_manifest):
        # ARRANGE -- add a second physical file for the same logical file_name
        manifest = pd.concat([
            stage_manifest,
            pd.DataFrame([{
                "file_name": "f1", "physical_training_file": "p2.json", "stage": 0,
                "trainer_name": "T0", "optimized_qaoa_angles": [0.9, 0.9],
                "outer_duration": 1.0, "outer_init_duration_sum": 1.0,
            }]),
        ], ignore_index=True)
        hw_row = pd.Series({"file_name": "f1", "training_method": "FA_PP_opt_5", "params": [0.1, 0.2], "job_p": 2, "training_p": 2})

        # ACT
        result = _resolve_training_stage(hw_row, manifest)

        # ASSERT
        assert result["training_match_status"] == "matched"
        assert "duplicate_logical_file" in result["training_match_note"]


# ---------------------------------------------------------------------------
# _resolve_inner_duration
# ---------------------------------------------------------------------------

class TestResolveInnerDuration:
    def test__resolve_inner_duration__given_unmatched_row__returns_nan(self):
        hw_row = {"training_match_status": "no_stage_match"}
        result = _resolve_inner_duration(hw_row, pd.DataFrame(), pd.DataFrame())
        assert np.isnan(result)

    def test__resolve_inner_duration__given_matched_row__sums_previous_stages_plus_current_stage_up_to_job_p(self):
        # ARRANGE -- iteration 0 fully precedes the matched stage (iteration 1)
        # and should be added in full via inner_totals; within the matched
        # stage, only depth_step <= job_p should count.
        inner = pd.DataFrame([
            {"file_name": "f1", "physical_training_file": "p1", "iteration": 1, "depth_step": 2, "inner_cum_within_stage": 1.0},
            {"file_name": "f1", "physical_training_file": "p1", "iteration": 1, "depth_step": 5, "inner_cum_within_stage": 4.0},
        ])
        inner_totals = pd.DataFrame([
            {"file_name": "f1", "physical_training_file": "p1", "iteration": 0, "inner_stage_total": 10.0},
        ])
        hw_row = {
            "training_match_status": "matched",
            "file_name": "f1",
            "physical_training_file": "p1",
            "matched_stage": 1,
            "job_p": 2,
        }

        # ACT
        result = _resolve_inner_duration(hw_row, inner, inner_totals)

        # ASSERT -- previous_total=10.0 (iteration 0) + current_total=1.0
        # (depth_step=2 is <= job_p=2; depth_step=5 is excluded)
        assert result == pytest.approx(11.0)

    def test__resolve_inner_duration__given_no_inner_rows_within_the_matched_stage__uses_zero_for_current(self):
        # ARRANGE -- matched stage has no inner rows with depth_step <= job_p
        inner = pd.DataFrame(columns=["file_name", "physical_training_file", "iteration", "stage", "depth_step", "inner_cum_within_stage"])
        inner_totals = pd.DataFrame([
            {"file_name": "f1", "physical_training_file": "p1", "iteration": 0, "inner_stage_total": 6.0},
        ])
        hw_row = {
            "training_match_status": "matched",
            "file_name": "f1",
            "physical_training_file": "p1",
            "matched_stage": 1,
            "job_p": 2,
        }

        # ACT
        result = _resolve_inner_duration(hw_row, inner, inner_totals)

        # ASSERT
        assert result == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# resolve_hardware_training_costs (orchestrator)
# ---------------------------------------------------------------------------

class TestResolveHardwareTrainingCosts:
    def test__resolve_hardware_training_costs__given_a_matched_row__sums_outer_and_inner_duration(self):
        # ARRANGE
        df_training = pd.DataFrame([
            {
                "file_name": "f1", "physical_training_file": "p1.json", "pre_processing_time": 0.0,
                "stage_summaries": [
                    {"stage": 0, "train_duration": 3.0, "trainer_name": "T0", "optimized_qaoa_angles": [0.1, 0.2]},
                ],
            },
        ])
        df_flat = pd.DataFrame([
            {"file_name": "f1", "physical_training_file": "p1.json", "level": "inner", "iteration": 0, "stage": 0, "depth_step": 2, "duration": 2.0},
        ])
        df_hardware = pd.DataFrame([
            {"file_name": "f1", "training_method": "FA_PP_opt_5", "params": [0.1, 0.2], "job_p": 2},
        ])

        # ACT
        result = resolve_hardware_training_costs(df_training, df_flat, df_hardware)

        # ASSERT -- outer_init_duration_sum=3.0, inner_duration_sum=2.0
        assert result.loc[0, "training_match_status"] == "matched"
        assert result.loc[0, "total_train_cost"] == pytest.approx(5.0)

    def test__resolve_hardware_training_costs__given_an_unmatched_row__total_cost_is_nan(self):
        # ARRANGE -- training data exists, but not for this hardware row's file_name
        df_training = pd.DataFrame([
            {
                "file_name": "some_other_file", "physical_training_file": "p1.json", "pre_processing_time": 0.0,
                "stage_summaries": [
                    {"stage": 0, "train_duration": 1.0, "trainer_name": "T0", "optimized_qaoa_angles": [0.1]},
                ],
            },
        ])
        df_flat = pd.DataFrame(columns=["file_name", "physical_training_file", "level", "iteration", "stage", "depth_step", "duration"])
        df_hardware = pd.DataFrame([
            {"file_name": "missing", "training_method": "FA_PP_opt_5", "params": [0.1, 0.2], "job_p": 2},
        ])

        # ACT
        result = resolve_hardware_training_costs(df_training, df_flat, df_hardware)

        # ASSERT
        assert result.loc[0, "training_match_status"] == "missing_training_file"
        assert np.isnan(result.loc[0, "total_train_cost"])
