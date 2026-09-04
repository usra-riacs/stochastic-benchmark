"""Tests for the pure/deterministic data-logic functions in
examples/IBM_QAOA/src/simulation_validation.py that had no coverage before
this file. tests/test_ibm_qaoa_simulation_validation.py already covers
_cached_fa_grid_point, _fit_log_power_law, _reset_cumulative_cost_on_cold_start,
build_bound_circuit_simulator, and fit_recommended_recipe_curves;
tests/test_ibm_qaoa_processing.py covers InstanceSpec's basic construction,
build_cumulative_budget_frontier, and filter_pss_exact_points.

Out of scope here, matching the exclusions used for utils.py and
Processing.py: the qiskit/Aer circuit-building and sampling functions
(build_bound_qaoa_circuit, MPSAerSampleEvaluator, sample_bound_circuit_counts,
sample_fixed_angles, ...), the external-pipeline-repo-dependent functions
(generate_linear_ramp_angles, run_method_from_config,
register_local_pipeline_evaluators, ensure_pipeline_imports,
ensure_qiskit_imports), the heavy multi-hundred-line orchestration functions
(run_pt_pss_exact_points, run_fa_pss_exact_points, generate_pss_exact_points,
run_stochastic_benchmark_pss, setup_stochastic_benchmark_campaign,
build_test_instance_set_from_repo, generate_training_instances_like_qps,
build_train_test_instance_sets), and build_binned_budget_dataset (dead code,
its only caller's import was dropped in the Step 4 cleanup commit).

Step 5 ("Extend") of the IBM_QAOA cleanup plan, simulation_validation.py pass.
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

from src.simulation_validation import (  # noqa: E402
    DEFAULT_INSTANCE_CACHE_ROOT,
    DEFAULT_MAIN_REPO,
    DEFAULT_PIPELINE_REPO,
    InstanceSpec,
    _build_response_summary_from_rec_params,
    _centers_from_edges,
    _deduplicate_exact_points,
    _edges_from_centers,
    _encode_categories,
    _exact_group_filter,
    _exact_group_is_complete,
    _first_optimizer_stage_index,
    _mpsaer_simulator_options,
    _predict_log_power_law,
    _resolve_time_per_shot,
    _resource_distance,
    align_projection_resources,
    assign_deterministic_train_split,
    attach_ws_codebook_columns,
    build_budget_bin_edges,
    build_budget_frontier,
    build_dense_budget_grid,
    build_sampled_training_config,
    build_strategy_budget_summary,
    decode_ws_params,
    instance_cache_root_path,
    instance_specs_to_dataframe,
    latest_stage_data,
    load_instance_context,
    locate_instance_file,
    main_repo_path,
    normalize_sb_compatible_dtypes,
    persist_campaign_exp_raw,
    pipeline_repo_path,
    raw_results_group_name,
    resource_metric_column,
    snap_actionable_fit_to_feasible_grid,
    snap_resources_to_grid,
    stage_records_from_bundle,
    strategy_family,
    strategy_runtime_label,
    summarize_frontier_instance_coverage,
    total_objective_evaluations_from_bundle,
    total_train_duration_from_bundle,
    total_training_shots_from_bundle,
)


# ---------------------------------------------------------------------------
# InstanceSpec.instance_id
# ---------------------------------------------------------------------------

class TestInstanceSpecInstanceId:
    def test__instance_id__zero_pads_to_three_digits(self):
        spec = InstanceSpec("heavy_hex", 144, 7)
        assert spec.instance_id == "007"

    def test__instance_id__given_an_already_wide_instance__is_unchanged(self):
        spec = InstanceSpec("heavy_hex", 144, 1234)
        assert spec.instance_id == "1234"


# ---------------------------------------------------------------------------
# main_repo_path / pipeline_repo_path / instance_cache_root_path
# ---------------------------------------------------------------------------

class TestRepoPathResolution:
    def test__main_repo_path__given_explicit_arg__uses_it(self):
        assert main_repo_path("/some/repo") == Path("/some/repo").expanduser()

    def test__main_repo_path__given_env_var__uses_it_over_the_default(self, monkeypatch):
        monkeypatch.setenv("QAOA_PARAMETER_SETTING_ROOT", "/from/env")
        assert main_repo_path() == Path("/from/env").expanduser()

    def test__main_repo_path__given_neither__falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("QAOA_PARAMETER_SETTING_ROOT", raising=False)
        assert main_repo_path() == DEFAULT_MAIN_REPO.expanduser()

    def test__pipeline_repo_path__given_env_var__uses_it(self, monkeypatch):
        monkeypatch.setenv("QAOA_TRAINING_PIPELINE_ROOT", "/from/env")
        assert pipeline_repo_path() == Path("/from/env").expanduser()

    def test__pipeline_repo_path__given_neither__falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("QAOA_TRAINING_PIPELINE_ROOT", raising=False)
        assert pipeline_repo_path() == DEFAULT_PIPELINE_REPO.expanduser()

    def test__instance_cache_root_path__given_neither__falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("IBM_QAOA_INSTANCE_CACHE_ROOT", raising=False)
        assert instance_cache_root_path() == DEFAULT_INSTANCE_CACHE_ROOT.expanduser()


# ---------------------------------------------------------------------------
# latest_stage_data / stage_records_from_bundle / bundle totals
# ---------------------------------------------------------------------------

class TestLatestStageData:
    def test__latest_stage_data__returns_the_highest_integer_stage_key(self):
        bundle = {0: {"x": "first"}, 1: {"x": "second"}, "meta": {"y": 1}}
        assert latest_stage_data(bundle) == {"x": "second"}

    def test__latest_stage_data__given_no_integer_keys__returns_the_whole_bundle(self):
        bundle = {"meta": {"y": 1}}
        assert latest_stage_data(bundle) == bundle


class TestStageRecordsFromBundle:
    def test__stage_records_from_bundle__returns_stages_in_key_order(self):
        bundle = {1: {"stage": 1}, 0: {"stage": 0}, "meta": {}}
        assert stage_records_from_bundle(bundle) == [{"stage": 0}, {"stage": 1}]


class TestBundleTotals:
    def test__total_training_shots_from_bundle__sums_across_stages(self):
        bundle = {0: {"training_shots_used": 100}, 1: {"training_shots_used": 50}}
        assert total_training_shots_from_bundle(bundle) == 150

    def test__total_training_shots_from_bundle__treats_missing_value_as_zero(self):
        bundle = {0: {}, 1: {"training_shots_used": 50}}
        assert total_training_shots_from_bundle(bundle) == 50

    def test__total_objective_evaluations_from_bundle__sums_across_stages(self):
        bundle = {0: {"num_objective_evaluations": 10}, 1: {"num_objective_evaluations": 5}}
        assert total_objective_evaluations_from_bundle(bundle) == 15

    def test__total_train_duration_from_bundle__sums_across_stages(self):
        bundle = {0: {"train_duration": 1.5}, 1: {"train_duration": 2.5}}
        assert total_train_duration_from_bundle(bundle) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# strategy_family / strategy_runtime_label
# ---------------------------------------------------------------------------

class TestStrategyFamily:
    @pytest.mark.parametrize("method_name,expected", [
        ("PT_PP_AAA", "PT"),
        ("linear_ramp_no_opt", "LR_no_opt"),
        ("FA_MPSAer_opt_5", "FA_MPSAer_native"),
        ("LR_MPSAer_opt_5", "LR_MPSAer_native"),
        ("FA_PP_opt_5", "FA"),
        ("LR_PP_opt_5", "LR"),
        ("Interp_5", "Interp_5"),  # unrecognized prefix passes through unchanged
    ])
    def test__strategy_family__maps_known_prefixes(self, method_name, expected):
        assert strategy_family(method_name) == expected


class TestStrategyRuntimeLabel:
    def test__strategy_runtime_label__combines_method_and_reps(self):
        assert strategy_runtime_label("FA_PP_opt", 5) == "FA_PP_opt_5"


# ---------------------------------------------------------------------------
# build_sampled_training_config
# ---------------------------------------------------------------------------

class TestBuildSampledTrainingConfig:
    def test__build_sampled_training_config__sets_shots_and_reps_for_fixed_angle_conjecture(self):
        config = {"trainer_chain": [{"trainer": "FixedAngleConjecture", "train_kwargs": {}}]}
        result = build_sampled_training_config(
            "FA_PP_opt", config, reps=5, cobyla_maxiter=None, shots_per_evaluation=200,
        )
        assert result["trainer_chain"][0]["train_kwargs"]["reps"] == 5

    def test__build_sampled_training_config__transfer_trainer_uses_qaoa_depth_not_reps(self):
        config = {"trainer_chain": [{"trainer": "TransferTrainer", "train_kwargs": {"reps": 99}}]}
        result = build_sampled_training_config(
            "PT_PP_AAA", config, reps=7, cobyla_maxiter=None, shots_per_evaluation=200,
        )
        stage = result["trainer_chain"][0]
        assert stage["train_kwargs"]["qaoa_depth"] == 7
        assert "reps" not in stage["train_kwargs"]

    def test__build_sampled_training_config__injects_evaluator_and_shots_when_evaluator_present(self):
        config = {"trainer_chain": [{"trainer": "ScipyTrainer", "trainer_init": {"evaluator": "SomeOtherEvaluator"}}]}
        result = build_sampled_training_config(
            "FA_PP_opt", config, reps=5, cobyla_maxiter=None, shots_per_evaluation=300,
        )
        trainer_init = result["trainer_chain"][0]["trainer_init"]
        assert trainer_init["evaluator"] == "MPSAerSampleEvaluator"
        assert trainer_init["evaluator_init"]["shots"] == 300

    def test__build_sampled_training_config__applies_cobyla_maxiter_to_scipy_trainer(self):
        config = {"trainer_chain": [{"trainer": "ScipyTrainer", "trainer_init": {}}]}
        result = build_sampled_training_config(
            "FA_PP_opt", config, reps=5, cobyla_maxiter=250, shots_per_evaluation=300,
        )
        minimize_args = result["trainer_chain"][0]["trainer_init"]["minimize_args"]
        assert minimize_args["options"]["maxiter"] == 250
        assert minimize_args["method"] == "COBYLA"

    def test__build_sampled_training_config__recursion_trainer_applies_to_nested_trainer_init(self):
        config = {
            "trainer_chain": [{
                "trainer": "RecursionTrainer",
                "trainer_init": {"trainer_init": {"evaluator": "SomeEvaluator"}},
            }]
        }
        result = build_sampled_training_config(
            "I_full", config, reps=7, cobyla_maxiter=100, shots_per_evaluation=300,
        )
        nested = result["trainer_chain"][0]["trainer_init"]["trainer_init"]
        assert nested["evaluator"] == "MPSAerSampleEvaluator"
        assert nested["evaluator_init"]["shots"] == 300
        assert nested["minimize_args"]["options"]["maxiter"] == 100

    def test__build_sampled_training_config__does_not_mutate_the_input_config(self):
        config = {"trainer_chain": [{"trainer": "FixedAngleConjecture", "train_kwargs": {}}]}
        build_sampled_training_config("FA_PP_opt", config, reps=5, cobyla_maxiter=None, shots_per_evaluation=200)
        assert "reps" not in config["trainer_chain"][0]["train_kwargs"]


# ---------------------------------------------------------------------------
# _mpsaer_simulator_options / _first_optimizer_stage_index
# ---------------------------------------------------------------------------

class TestMpsaerSimulatorOptions:
    def test__mpsaer_simulator_options__maps_chi_and_threads(self):
        result = _mpsaer_simulator_options({"chi": 64, "max_parallel_threads": 4})
        assert result == {"matrix_product_state_max_bond_dimension": 64, "max_parallel_threads": 4}

    def test__mpsaer_simulator_options__given_none__returns_empty_dict(self):
        assert _mpsaer_simulator_options(None) == {}


class TestFirstOptimizerStageIndex:
    def test__first_optimizer_stage_index__finds_the_first_scipy_or_tqa_stage(self):
        config = {"trainer_chain": [{"trainer": "FixedAngleConjecture"}, {"trainer": "ScipyTrainer"}]}
        assert _first_optimizer_stage_index(config) == 1

    def test__first_optimizer_stage_index__given_no_optimizer_stage__returns_none(self):
        config = {"trainer_chain": [{"trainer": "FixedAngleConjecture"}]}
        assert _first_optimizer_stage_index(config) is None


# ---------------------------------------------------------------------------
# _resolve_time_per_shot
# ---------------------------------------------------------------------------

class TestResolveTimePerShot:
    def test__resolve_time_per_shot__given_a_float__returns_it_unchanged(self):
        assert _resolve_time_per_shot(0.002, reps=5) == pytest.approx(0.002)

    def test__resolve_time_per_shot__given_a_mapping_with_the_exact_depth__uses_it(self):
        assert _resolve_time_per_shot({5: 0.002, 6: 0.003}, reps=5) == pytest.approx(0.002)

    def test__resolve_time_per_shot__given_a_mapping_missing_the_depth__falls_back_to_the_mean(self):
        assert _resolve_time_per_shot({5: 0.002, 6: 0.004}, reps=7) == pytest.approx(0.003)

    def test__resolve_time_per_shot__given_an_empty_mapping__falls_back_to_one(self):
        assert _resolve_time_per_shot({}, reps=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_dense_budget_grid / build_budget_bin_edges / _centers_from_edges / _edges_from_centers
# ---------------------------------------------------------------------------

class TestBuildDenseBudgetGrid:
    def test__build_dense_budget_grid__spans_from_min_to_max_on_a_log_scale(self):
        df = pd.DataFrame({"T_exact_proxy": [1.0, 10.0, 100.0]})
        grid = build_dense_budget_grid(df, num_points=5)
        assert grid[0] == pytest.approx(1.0)
        assert grid[-1] == pytest.approx(100.0)
        assert len(grid) == 5

    def test__build_dense_budget_grid__given_empty_dataframe__returns_empty_list(self):
        assert build_dense_budget_grid(pd.DataFrame({"T_exact_proxy": []})) == []

    def test__build_dense_budget_grid__given_a_single_distinct_value__returns_that_one_value(self):
        df = pd.DataFrame({"T_exact_proxy": [5.0, 5.0]})
        assert build_dense_budget_grid(df) == [5.0]


class TestBuildBudgetBinEdges:
    def test__build_budget_bin_edges__returns_num_bins_plus_one_edges(self):
        df = pd.DataFrame({"T_exact_proxy": [1.0, 100.0]})
        edges = build_budget_bin_edges(df, num_bins=4)
        assert len(edges) == 5
        assert edges[0] == pytest.approx(1.0)
        assert edges[-1] == pytest.approx(100.0)


class TestCentersAndEdgesRoundtrip:
    def test__centers_from_edges__log_scale_uses_geometric_midpoint(self):
        edges = np.array([1.0, 4.0, 16.0])
        centers = _centers_from_edges(edges, scale="log")
        np.testing.assert_allclose(centers, [2.0, 8.0])

    def test__edges_from_centers__given_a_single_center__brackets_it_symmetrically_in_log_space(self):
        edges = _edges_from_centers(np.array([4.0]), scale="log")
        np.testing.assert_allclose(edges, [4.0 / np.sqrt(2.0), 4.0 * np.sqrt(2.0)])

    def test__edges_from_centers__given_no_centers__returns_empty_array(self):
        assert _edges_from_centers(np.array([])).size == 0


# ---------------------------------------------------------------------------
# _resource_distance / build_budget_frontier
# ---------------------------------------------------------------------------

class TestResourceDistance:
    def test__resource_distance__log_scale_measures_distance_in_log_space(self):
        values = pd.Series([1.0, 10.0, 100.0])
        distances = _resource_distance(values, target=10.0, scale="log")
        np.testing.assert_allclose(distances, [np.log(10.0), 0.0, np.log(10.0)])

    def test__resource_distance__given_non_finite_values__returns_inf_for_those_entries(self):
        values = pd.Series([10.0, np.nan])
        distances = _resource_distance(values, target=10.0, scale="log")
        assert distances[0] == pytest.approx(0.0)
        assert np.isinf(distances[1])


class TestBuildBudgetFrontier:
    def test__build_budget_frontier__delegates_to_build_cumulative_budget_frontier(self):
        # ARRANGE -- same fixture shape as
        # test_ibm_qaoa_processing.py's build_cumulative_budget_frontier test
        exact_df = pd.DataFrame([
            {
                "graph_type": "heavy_hex", "num_nodes": 144, "instance": "0", "split": "train",
                "strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5,
                "N": 10, "M": 10, "Q": 100, "T_exact_proxy": 100.0, "T_exact": 110.0,
                "BestApproximationRatio": 0.80,
            },
        ])

        # ACT
        frontier = build_budget_frontier(exact_df, t_grid=[100.0])

        # ASSERT
        assert not frontier.empty
        assert frontier.iloc[0]["BestApproximationRatio"] == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# attach_ws_codebook_columns / summarize_frontier_instance_coverage
# ---------------------------------------------------------------------------

class TestAttachWsCodebookColumns:
    def test__attach_ws_codebook_columns__maps_strategy_and_simulation_to_codes(self):
        frontier_df = pd.DataFrame([
            {"strategy": "FA_PP_opt", "simulation_method": "MPSAer", "split": "train"},
            {"strategy": "PT_PP_AAA", "simulation_method": "SV", "split": "test"},
        ])
        codebooks = {
            "strategy": {0: "FA_PP_opt", 1: "PT_PP_AAA"},
            "simulation": {0: "MPSAer", 1: "SV"},
        }
        result = attach_ws_codebook_columns(frontier_df, codebooks)
        assert list(result["strategy_code"]) == [0.0, 1.0]
        assert list(result["simulation_code"]) == [0.0, 1.0]
        assert list(result["train"]) == [1, 0]

    def test__attach_ws_codebook_columns__given_empty_dataframe__returns_it_unchanged(self):
        result = attach_ws_codebook_columns(pd.DataFrame(), {})
        assert result.empty


class TestSummarizeFrontierInstanceCoverage:
    def test__summarize_frontier_instance_coverage__flags_budget_points_missing_instances(self):
        # ARRANGE -- at T=100 both instances are present (full coverage);
        # at T=200 only instance "0" is present (partial coverage)
        frontier_df = pd.DataFrame([
            {"split": "train", "strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T": 100.0, "instance": "0"},
            {"split": "train", "strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T": 100.0, "instance": "1"},
            {"split": "train", "strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T": 200.0, "instance": "0"},
        ])
        result = summarize_frontier_instance_coverage(frontier_df)
        row = result.iloc[0]
        assert row["expected_instances"] == 2
        assert row["min_instances"] == 1
        assert row["n_partial_budget_points"] == 1

    def test__summarize_frontier_instance_coverage__given_empty_dataframe__returns_empty(self):
        assert summarize_frontier_instance_coverage(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# _exact_group_filter / _exact_group_is_complete / _deduplicate_exact_points
# ---------------------------------------------------------------------------

class TestExactGroupFilter:
    def test__exact_group_filter__matches_on_every_field(self):
        df = pd.DataFrame([
            {"graph_type": "heavy_hex", "num_nodes": 144, "instance": "005", "split": "train", "p": 5, "strategy": "FA_PP_opt"},
            {"graph_type": "heavy_hex", "num_nodes": 144, "instance": "006", "split": "train", "p": 5, "strategy": "FA_PP_opt"},
        ])
        spec = InstanceSpec("heavy_hex", 144, 5, split="train")
        mask = _exact_group_filter(df, spec=spec, reps=5, strategy="FA_PP_opt")
        assert list(mask) == [True, False]

    def test__exact_group_filter__defaults_split_to_train_when_spec_split_is_none(self):
        df = pd.DataFrame([{"graph_type": "heavy_hex", "num_nodes": 144, "instance": "005", "split": "train", "p": 5, "strategy": "FA_PP_opt"}])
        spec = InstanceSpec("heavy_hex", 144, 5, split=None)
        mask = _exact_group_filter(df, spec=spec, reps=5, strategy="FA_PP_opt")
        assert list(mask) == [True]


class TestExactGroupIsComplete:
    def test__exact_group_is_complete__fa_strategy_requires_full_n_m_q_grid(self):
        df = pd.DataFrame([
            {"N": 10, "M": 10, "Q": 100}, {"N": 10, "M": 10, "Q": 200},
            {"N": 20, "M": 10, "Q": 100}, {"N": 20, "M": 10, "Q": 200},
        ])
        complete = _exact_group_is_complete(
            df, strategy="FA_PP_opt", fa_method_name="FA_PP_opt", pt_method_name="PT_PP_AAA",
            fa_n_values=[10, 20], fa_m_values=[10], q_values=[100, 200],
        )
        assert complete is True

    def test__exact_group_is_complete__fa_strategy_missing_one_grid_point_is_incomplete(self):
        df = pd.DataFrame([{"N": 10, "M": 10, "Q": 100}])
        complete = _exact_group_is_complete(
            df, strategy="FA_PP_opt", fa_method_name="FA_PP_opt", pt_method_name="PT_PP_AAA",
            fa_n_values=[10, 20], fa_m_values=[10], q_values=[100, 200],
        )
        assert complete is False

    def test__exact_group_is_complete__pt_strategy_only_needs_all_q_values(self):
        df = pd.DataFrame([{"Q": 100}, {"Q": 200}])
        complete = _exact_group_is_complete(
            df, strategy="PT_PP_AAA", fa_method_name="FA_PP_opt", pt_method_name="PT_PP_AAA",
            fa_n_values=[10], fa_m_values=[10], q_values=[100, 200],
        )
        assert complete is True

    def test__exact_group_is_complete__given_empty_dataframe__returns_false(self):
        assert _exact_group_is_complete(
            pd.DataFrame(), strategy="FA_PP_opt", fa_method_name="FA_PP_opt", pt_method_name="PT_PP_AAA",
            fa_n_values=[10], fa_m_values=[10], q_values=[100],
        ) is False


class TestDeduplicateExactPoints:
    def test__deduplicate_exact_points__keeps_the_last_row_per_grid_point(self):
        df = pd.DataFrame([
            {"graph_type": "heavy_hex", "num_nodes": 144, "instance": "0", "split": "train", "strategy": "FA_PP_opt", "p": 5, "N": 10, "M": 10, "Q": 100, "BestApproximationRatio": 0.5},
            {"graph_type": "heavy_hex", "num_nodes": 144, "instance": "0", "split": "train", "strategy": "FA_PP_opt", "p": 5, "N": 10, "M": 10, "Q": 100, "BestApproximationRatio": 0.9},
        ])
        result = _deduplicate_exact_points(df)
        assert len(result) == 1
        assert result.iloc[0]["BestApproximationRatio"] == pytest.approx(0.9)

    def test__deduplicate_exact_points__given_empty_dataframe__returns_it_unchanged(self):
        df = pd.DataFrame()
        assert _deduplicate_exact_points(df).empty


# ---------------------------------------------------------------------------
# _predict_log_power_law
# ---------------------------------------------------------------------------

class TestPredictLogPowerLaw:
    def test__predict_log_power_law__interpolates_between_knots_in_log_space(self):
        log_x_knots = np.log(np.array([1.0, 100.0]))
        log_y_knots = np.log(np.array([10.0, 1000.0]))  # y = 10 * x
        pred = _predict_log_power_law([10.0], log_x_knots, log_y_knots)
        assert pred[0] == pytest.approx(100.0, rel=1e-6)

    def test__predict_log_power_law__flat_extrapolates_beyond_the_knot_range(self):
        log_x_knots = np.log(np.array([1.0, 10.0]))
        log_y_knots = np.log(np.array([1.0, 10.0]))
        pred = _predict_log_power_law([1000.0], log_x_knots, log_y_knots)
        assert pred[0] == pytest.approx(10.0)  # held at the right-edge knot value

    def test__predict_log_power_law__clips_to_y_min_and_y_max(self):
        log_x_knots = np.log(np.array([1.0, 10.0]))
        log_y_knots = np.log(np.array([1.0, 100.0]))
        pred = _predict_log_power_law([10.0], log_x_knots, log_y_knots, y_max=50.0)
        assert pred[0] == pytest.approx(50.0)

    def test__predict_log_power_law__given_nonpositive_t__returns_nan(self):
        log_x_knots = np.log(np.array([1.0, 10.0]))
        log_y_knots = np.log(np.array([1.0, 10.0]))
        pred = _predict_log_power_law([0.0, -1.0], log_x_knots, log_y_knots)
        assert np.all(np.isnan(pred))


# ---------------------------------------------------------------------------
# snap_actionable_fit_to_feasible_grid
# ---------------------------------------------------------------------------

class TestSnapActionableFitToFeasibleGrid:
    def test__snap_actionable_fit_to_feasible_grid__picks_the_closest_budget_then_closest_parameters(self):
        # ARRANGE -- two candidates at the same (strategy, simulation_method, p);
        # the fit targets T=105 (closer to the T_exact_proxy=100 row) and
        # N_fit=12 (closer to the N=10 row, once budget distance is tied... here
        # budget alone already disambiguates: 100 is closer to 105 than 200 is).
        exact_df = pd.DataFrame([
            {"strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T_exact_proxy": 100.0, "N": 10, "M": 10, "Q": 100, "BestApproximationRatio": 0.5},
            {"strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T_exact_proxy": 200.0, "N": 20, "M": 20, "Q": 200, "BestApproximationRatio": 0.6},
        ])
        actionable_fit_df = pd.DataFrame([
            {"strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T": 105.0, "N_fit": 12.0, "M_fit": 12.0, "Q_fit": 100.0},
        ])

        # ACT
        result = snap_actionable_fit_to_feasible_grid(exact_df, actionable_fit_df)

        # ASSERT
        assert len(result) == 1
        assert result.iloc[0]["N"] == 10
        assert result.iloc[0]["T"] == pytest.approx(105.0)  # target T is preserved on the output row

    def test__snap_actionable_fit_to_feasible_grid__given_no_matching_group__is_skipped(self):
        exact_df = pd.DataFrame([
            {"strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T_exact_proxy": 100.0, "N": 10, "M": 10, "Q": 100, "BestApproximationRatio": 0.5},
        ])
        actionable_fit_df = pd.DataFrame([
            {"strategy": "PT_PP_AAA", "simulation_method": "MPSAer", "p": 5, "T": 105.0, "N_fit": 12.0, "M_fit": 12.0, "Q_fit": 100.0},
        ])
        result = snap_actionable_fit_to_feasible_grid(exact_df, actionable_fit_df)
        assert result.empty

    def test__snap_actionable_fit_to_feasible_grid__given_empty_inputs__returns_empty(self):
        assert snap_actionable_fit_to_feasible_grid(pd.DataFrame(), pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# resource_metric_column
# ---------------------------------------------------------------------------

class TestResourceMetricColumn:
    def test__resource_metric_column__prefers_the_key_mean_column(self):
        df = pd.DataFrame({"Key=MeanT_exact": [1.0], "T_exact": [2.0]})
        result = resource_metric_column(df, "T_exact")
        assert list(result) == [1.0]

    def test__resource_metric_column__falls_back_to_key_meantime(self):
        df = pd.DataFrame({"Key=MeanTime": [3.0]})
        result = resource_metric_column(df, "T_exact")
        assert list(result) == [3.0]

    def test__resource_metric_column__falls_back_to_the_raw_column_name(self):
        df = pd.DataFrame({"T_exact": [4.0]})
        result = resource_metric_column(df, "T_exact")
        assert list(result) == [4.0]

    def test__resource_metric_column__given_no_match__raises_keyerror(self):
        df = pd.DataFrame({"other": [1.0]})
        with pytest.raises(KeyError):
            resource_metric_column(df, "T_exact")


# ---------------------------------------------------------------------------
# assign_deterministic_train_split
# ---------------------------------------------------------------------------

class TestAssignDeterministicTrainSplit:
    def test__assign_deterministic_train_split__joins_split_by_instance(self):
        # ARRANGE -- instance ids as strings, matching this codebase's usual
        # convention (e.g. InstanceSpec.instance_id); the function itself
        # only str-casts interp_results' side of the join key, not
        # exp_raw_df's, so a numeric exp_raw_df instance column would fail
        # the merge with a dtype mismatch.
        interp_results = pd.DataFrame({"instance": ["0", "1"], "resource": [10.0, 20.0]})
        exp_raw_df = pd.DataFrame({"instance": ["0", "1"], "train": [1, 0], "split": ["train", "test"]})
        result = assign_deterministic_train_split(interp_results, exp_raw_df)
        assert list(result["split"]) == ["train", "test"]

    def test__assign_deterministic_train_split__overwrites_any_existing_split_column(self):
        interp_results = pd.DataFrame({"instance": ["0"], "split": ["stale"]})
        exp_raw_df = pd.DataFrame({"instance": ["0"], "train": [1], "split": ["train"]})
        result = assign_deterministic_train_split(interp_results, exp_raw_df)
        assert result.iloc[0]["split"] == "train"


# ---------------------------------------------------------------------------
# snap_resources_to_grid / align_projection_resources
# ---------------------------------------------------------------------------

class TestSnapResourcesToGrid:
    def test__snap_resources_to_grid__snaps_to_the_nearest_grid_value(self):
        df = pd.DataFrame({"resource": [9.0, 14.0]})
        result = snap_resources_to_grid(df, [10.0, 20.0])
        assert list(result["resource"]) == [10.0, 10.0]

    def test__snap_resources_to_grid__given_empty_dataframe__returns_it_unchanged(self):
        df = pd.DataFrame({"resource": []})
        assert snap_resources_to_grid(df, [10.0]).empty


class TestAlignProjectionResources:
    def test__align_projection_resources__snaps_training_and_testing_stats_in_place(self):
        class _Sb:
            pass
        sb = _Sb()
        sb.training_stats = pd.DataFrame({"resource": [9.0]})
        sb.testing_stats = pd.DataFrame({"resource": [21.0]})

        align_projection_resources(sb, shared_grid=[10.0, 20.0])

        assert sb.training_stats.iloc[0]["resource"] == pytest.approx(10.0)
        assert sb.testing_stats.iloc[0]["resource"] == pytest.approx(20.0)

    def test__align_projection_resources__given_none_stats__does_not_raise(self):
        class _Sb:
            training_stats = None
            testing_stats = None
        align_projection_resources(_Sb(), shared_grid=[10.0])


# ---------------------------------------------------------------------------
# _encode_categories / normalize_sb_compatible_dtypes / raw_results_group_name
# ---------------------------------------------------------------------------

class TestEncodeCategories:
    def test__encode_categories__assigns_sorted_integer_codes(self):
        # ARRANGE -- categories come from series.dropna(), but the returned
        # codes are computed from the ORIGINAL (non-dropna'd) series passed
        # through .astype(str); a None entry becomes the literal string
        # "None" there, which isn't a real category, so it maps to NaN.
        series = pd.Series(["b", "a", "b", None])

        # ACT
        codes, codebook = _encode_categories(series)

        # ASSERT
        assert codebook == {0: "a", 1: "b"}
        assert list(codes.iloc[:3]) == [1, 0, 1]
        assert np.isnan(codes.iloc[3])


class TestNormalizeSbCompatibleDtypes:
    def test__normalize_sb_compatible_dtypes__converts_string_dtype_to_object(self):
        df = pd.DataFrame({"a": pd.array(["x", "y"], dtype="string")})
        result = normalize_sb_compatible_dtypes(df)
        assert result["a"].dtype == object


class TestRawResultsGroupName:
    def test__raw_results_group_name__strips_the_raw_results_prefix(self):
        assert raw_results_group_name("raw_results_inst=0_depth=5.pkl") == "inst=0_depth=5"


class TestPersistCampaignExpRaw:
    def test__persist_campaign_exp_raw__writes_one_pickle_per_instance_depth_group(self, tmp_path):
        campaign_df = pd.DataFrame({
            "instance": [0, 0, 1],
            "p": [5, 5, 5],
            "value": [1, 2, 3],
        })
        written = persist_campaign_exp_raw(campaign_df, tmp_path)
        assert len(written) == 2
        assert all(path.exists() for path in written)
        assert (tmp_path / "exp_raw").is_dir()


# ---------------------------------------------------------------------------
# decode_ws_params / _build_response_summary_from_rec_params
# ---------------------------------------------------------------------------

class TestDecodeWsParams:
    def test__decode_ws_params__maps_codes_back_to_strategy_and_simulation_names(self):
        df = pd.DataFrame({"strategy_code": [0.0], "simulation_code": [1.0]})
        codebooks = {"strategy": {0: "FA_PP_opt"}, "simulation": {0: "MPSAer", 1: "SV"}}
        result = decode_ws_params(df, codebooks)
        assert result.iloc[0]["strategy"] == "FA_PP_opt"
        assert result.iloc[0]["simulation_method"] == "SV"


class TestBuildResponseSummaryFromRecParams:
    def test__build_response_summary_from_rec_params__computes_a_95pct_ci_from_sem_by_default(self):
        # ARRANGE -- two candidates at the same resource, response 0.5 and 0.7
        rec_params = pd.DataFrame([
            {"resource": 10.0, "N": 5.0, "Key=BestApproximationRatio": 0.5},
            {"resource": 10.0, "N": 7.0, "Key=BestApproximationRatio": 0.7},
        ])

        # ACT
        summary = _build_response_summary_from_rec_params(
            rec_params, ["N"], {}, response_key="BestApproximationRatio"
        )

        # ASSERT -- mean=0.6, std=sqrt(0.02)~=0.1414, sem=std/sqrt(2)~=0.1,
        # ci_half_width=1.96*0.1~=0.196
        row = summary.iloc[0]
        assert row["response"] == pytest.approx(0.6)
        assert row["response_lower"] == pytest.approx(0.6 - 0.196, abs=1e-3)
        assert row["response_upper"] == pytest.approx(0.6 + 0.196, abs=1e-3)
        assert row["N"] == pytest.approx(6.0)

    def test__build_response_summary_from_rec_params__native_ci_columns_override_the_computed_sem_ci(self):
        # ARRANGE -- native CI columns present and non-zero-width; the response
        # values differ (0.4, 0.6) so the SEM-based CI would differ from the
        # native CI, proving the override actually took effect.
        rec_params = pd.DataFrame([
            {
                "resource": 10.0, "N": 5.0, "Key=BestApproximationRatio": 0.4,
                "ConfInt=lower_Key=BestApproximationRatio": 0.3, "ConfInt=upper_Key=BestApproximationRatio": 0.6,
            },
            {
                "resource": 10.0, "N": 7.0, "Key=BestApproximationRatio": 0.6,
                "ConfInt=lower_Key=BestApproximationRatio": 0.3, "ConfInt=upper_Key=BestApproximationRatio": 0.6,
            },
        ])

        # ACT
        summary = _build_response_summary_from_rec_params(
            rec_params, ["N"], {}, response_key="BestApproximationRatio"
        )

        # ASSERT
        row = summary.iloc[0]
        assert row["response_lower"] == pytest.approx(0.3)
        assert row["response_upper"] == pytest.approx(0.6)

    def test__build_response_summary_from_rec_params__given_empty_input__returns_empty(self):
        result = _build_response_summary_from_rec_params(pd.DataFrame(), ["N"], {}, response_key="BestApproximationRatio")
        assert result.empty

    def test__build_response_summary_from_rec_params__given_missing_response_column__raises_keyerror(self):
        rec_params = pd.DataFrame([{"resource": 10.0, "N": 5.0}])
        with pytest.raises(KeyError):
            _build_response_summary_from_rec_params(rec_params, ["N"], {}, response_key="BestApproximationRatio")


# ---------------------------------------------------------------------------
# build_strategy_budget_summary
# ---------------------------------------------------------------------------

class TestBuildStrategyBudgetSummary:
    def test__build_strategy_budget_summary__aggregates_by_strategy_simulation_p_and_budget(self):
        frontier_df = pd.DataFrame([
            {
                "split": "train", "strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T": 100.0,
                "BestApproximationRatio": 0.5, "N": 10, "M": 10, "Q": 100,
                "training_cost": 1.0, "sampling_cost": 2.0, "training_cost_proxy": 1.0, "sampling_cost_proxy": 2.0,
                "selected_exact_T": 100.0, "selected_exact_T_proxy": 100.0, "instance": "0",
            },
            {
                "split": "train", "strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T": 100.0,
                "BestApproximationRatio": 0.7, "N": 10, "M": 10, "Q": 100,
                "training_cost": 1.0, "sampling_cost": 2.0, "training_cost_proxy": 1.0, "sampling_cost_proxy": 2.0,
                "selected_exact_T": 100.0, "selected_exact_T_proxy": 100.0, "instance": "1",
            },
        ])
        result = build_strategy_budget_summary(frontier_df)
        row = result.iloc[0]
        assert row["response_mean"] == pytest.approx(0.6)
        assert row["n_instances"] == 2

    def test__build_strategy_budget_summary__given_no_rows_for_the_requested_split__returns_empty(self):
        frontier_df = pd.DataFrame([{"split": "test", "strategy": "FA_PP_opt", "simulation_method": "MPSAer", "p": 5, "T": 100.0}])
        assert build_strategy_budget_summary(frontier_df, split="train").empty

    def test__build_strategy_budget_summary__given_empty_dataframe__returns_empty(self):
        assert build_strategy_budget_summary(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# locate_instance_file / load_instance_context / instance_specs_to_dataframe
# ---------------------------------------------------------------------------

class TestLocateInstanceFile:
    def test__locate_instance_file__given_an_explicit_graph_path__returns_it_directly(self):
        spec = InstanceSpec("heavy_hex", 144, 0, graph_path="/explicit/path.json")
        assert locate_instance_file(spec) == Path("/explicit/path.json")

    def test__locate_instance_file__given_no_graph_path__globs_for_it_under_main_repo(self, tmp_path):
        instances_dir = tmp_path / "instances" / "heavy_hex"
        instances_dir.mkdir(parents=True)
        expected = instances_dir / "000_seed1_heavyhex_144nodes.json"
        expected.touch()
        spec = InstanceSpec("heavy_hex", 144, 0)
        result = locate_instance_file(spec, main_repo=tmp_path)
        assert result == expected

    def test__locate_instance_file__given_no_match__raises_filenotfounderror(self, tmp_path):
        (tmp_path / "instances" / "heavy_hex").mkdir(parents=True)
        spec = InstanceSpec("heavy_hex", 144, 0)
        with pytest.raises(FileNotFoundError):
            locate_instance_file(spec, main_repo=tmp_path)


class TestLoadInstanceContext:
    def test__load_instance_context__given_explicit_graph_and_minmax_paths__loads_both(self, tmp_path):
        graph_path = tmp_path / "graph.json"
        graph_path.write_text(
            '{"edge list": [{"nodes": [0, 1], "weight": 1.0}, {"nodes": [1, 2], "weight": 1.0}]}'
        )
        minmax_path = tmp_path / "minmax.json"
        minmax_path.write_text('{"min_cut": 0.0, "max_cut": 2.0, "sum_of_weights": 2.0}')
        spec = InstanceSpec("heavy_hex", 3, 0, graph_path=str(graph_path), minmax_path=str(minmax_path))

        result = load_instance_context(spec)

        assert result["min_cut"] == pytest.approx(0.0)
        assert result["max_cut"] == pytest.approx(2.0)
        assert result["short_name"] == "graph"
        assert result["instance_context"]["sum_weights"] == pytest.approx(2.0)


class TestInstanceSpecsToDataframe:
    def test__instance_specs_to_dataframe__builds_one_row_per_spec(self):
        specs = [
            InstanceSpec("heavy_hex", 144, 0, split="train"),
            InstanceSpec("heavy_hex", 144, 1, split="test"),
        ]
        df = instance_specs_to_dataframe(specs)
        assert list(df["instance"]) == ["000", "001"]
        assert list(df["split"]) == ["train", "test"]
