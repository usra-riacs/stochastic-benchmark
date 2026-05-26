import math

import numpy as np
import pandas as pd
import pytest

from repeat_reliability import (
    IntervalEstimate,
    ProportionInterval,
    RepeatCountInterval,
    agresti_coull_interval,
    agresti_coull_interval_from_estimate,
    cets_from_repeat_count,
    maximum_relative_error,
    normal_critical_value,
    annotate_reliability_comparisons,
    propagate_success_probability_interval,
    relative_repeats_error,
    repeat_count,
    repeat_count_interval,
    repeat_reliability_report,
    repeat_reliability_metrics,
    repeats_to_solution,
    required_repeats_exact,
    required_repeats_for_probability_error,
    required_repeats_lower_bound,
    required_trials_for_relative_error,
    rtt_from_repeat_count,
    scaled_repeat_count_interval,
    success_probability_margin,
)


REFERENCE_GRID = [
    (0.0, 100, 0.018496749103, 0.0, 0.044412051134, 246.662118617232, 101.371886661212, math.inf, math.inf),
    (0.0, 1000, 0.001913379243, 0.0, 0.004616716146, 2404.522301848311, 995.194735863067, math.inf, math.inf),
    (0.0, 10000, 0.000191999185, 0.0, 0.000463500969, 23983.06049718036, 9933.318713707136, math.inf, math.inf),
    (0.01, 100, 0.028126814121, 0.0, 0.059926861882, 161.415289821065, 74.520210629335, math.inf, math.inf),
    (0.01, 1000, 0.011875111658, 0.005174101181, 0.018576122135, 385.492989139588, 245.598267487361, 887.737969708359, 1.30286411094),
    (0.01, 10000, 0.010188159202, 0.008220323323, 0.01215599508, 449.705481961264, 376.532146256283, 557.911893764758, 0.240616172459),
    (0.1, 100, 0.114797399283, 0.053484752289, 0.176110046277, 37.766262448893, 23.772508663081, 83.778808093604, 1.218350524015),
    (0.1, 1000, 0.101530703394, 0.082846876078, 0.12021453071, 43.013750285732, 35.956209885714, 53.25075866433, 0.237993858024),
    (0.1, 10000, 0.100153599348, 0.094270825358, 0.106036373339, 43.637998557021, 41.084506657875, 46.509850931411, 0.06581081785),
    (0.5, 100, 0.5, 0.403831530366, 0.596168469634, 6.643856189775, 5.078723061497, 8.903490739773, 0.340108889394),
    (0.5, 1000, 0.5, 0.469069600368, 0.530930399632, 6.643856189775, 6.083414957563, 7.273721592451, 0.094804189718),
    (0.5, 10000, 0.5, 0.490202061815, 0.509797938185, 6.643856189775, 6.459429512745, 6.835225017241, 0.028803878651),
    (0.9, 100, 0.885202600717, 0.823889953723, 0.946515247711, 2.127505781374, 1.572611385934, 2.651760699727, 0.260819218588),
    (0.9, 1000, 0.898469296606, 0.87978546929, 0.917153123922, 2.013282419249, 1.84890069314, 2.173811359636, 0.081648617471),
    (0.9, 10000, 0.899846400652, 0.893963626661, 0.905729174642, 2.001334012617, 1.950034860166, 2.052239480047, 0.0256324792),
    (0.99, 100, 0.971873185879, 0.940073138118, 1.0, 1.289590877697, 1.0, 1.636154480046, 0.268739185693),
    (0.99, 1000, 0.988124888342, 0.981423877865, 0.994825898819, 1.038765536191, 1.0, 1.155371504963, 0.112254368006),
    (0.99, 10000, 0.989811840798, 0.98784400492, 0.991779676677, 1.004064313122, 1.0, 1.044272188673, 0.040045119645),
    (1.0, 100, 0.981503250897, 0.955587948866, 1.0, 1.154131627644, 1.0, 1.478743977145, 0.281261115913),
    (1.0, 1000, 0.998086620757, 0.995383283854, 1.0, 1.0, 1.0, 1.0, 0.0),
    (1.0, 10000, 0.999808000815, 0.999536499031, 1.0, 1.0, 1.0, 1.0, 0.0),
]


@pytest.mark.parametrize(
    "p_hat,n,expected_p,expected_lower,expected_upper,"
    "expected_r,expected_r_lower,expected_r_upper,expected_error",
    REFERENCE_GRID,
)
def test_noori_reference_grid_matches_deterministic_fixtures(
    p_hat,
    n,
    expected_p,
    expected_lower,
    expected_upper,
    expected_r,
    expected_r_lower,
    expected_r_upper,
    expected_error,
):
    interval = agresti_coull_interval_from_estimate(p_hat, n)
    repeat_estimate = repeat_count(interval.estimate)
    repeat_interval = repeat_count_interval(interval.lower, interval.upper)
    max_error = maximum_relative_error(
        repeat_estimate,
        repeat_interval.lower,
        repeat_interval.upper,
    )

    assert interval.estimate == pytest.approx(expected_p, abs=5e-13)
    assert interval.lower == pytest.approx(expected_lower, abs=5e-13)
    assert interval.upper == pytest.approx(expected_upper, abs=5e-13)
    assert repeat_estimate == pytest.approx(expected_r, abs=5e-12)
    assert repeat_interval.lower == pytest.approx(expected_r_lower, abs=5e-12)

    if math.isinf(expected_r_upper):
        assert math.isinf(repeat_interval.upper)
        assert math.isinf(max_error)
    else:
        assert repeat_interval.upper == pytest.approx(expected_r_upper, abs=5e-12)
        assert max_error == pytest.approx(expected_error, abs=5e-12)


def test_public_agresti_coull_zero_trials_is_defined():
    interval = agresti_coull_interval(0, 0)

    assert interval.estimate == pytest.approx(0.5)
    assert interval.lower == pytest.approx(0.0)
    assert interval.upper == pytest.approx(1.0)
    assert math.isnan(interval.raw_estimate)


def test_rtt_and_cets_scaling_preserve_relative_error():
    repeat_estimate = repeat_count(0.25, target_confidence=0.99)
    repeat_interval = repeat_count_interval(0.22, 0.28, target_confidence=0.99)
    repeat_error = maximum_relative_error(
        repeat_estimate,
        repeat_interval.lower,
        repeat_interval.upper,
    )

    runtime_scale = 0.25
    rtt_interval = scaled_repeat_count_interval(repeat_interval, runtime_scale)
    rtt_estimate = rtt_from_repeat_count(repeat_estimate, runtime_scale)

    cets_scale = 500 * 0.02
    cets_interval = scaled_repeat_count_interval(repeat_interval, cets_scale)
    cets_estimate = cets_from_repeat_count(repeat_estimate, iterations=500, effort_per_iteration=0.02)

    assert maximum_relative_error(rtt_estimate, rtt_interval.lower, rtt_interval.upper) == pytest.approx(repeat_error)
    assert maximum_relative_error(cets_estimate, cets_interval.lower, cets_interval.upper) == pytest.approx(repeat_error)


def test_interval_width_properties_cover_zero_estimate():
    interval = ProportionInterval(estimate=0.25, lower=0.2, upper=0.3, half_width=0.05)
    zero_estimate_interval = ProportionInterval(estimate=0.0, lower=0.0, upper=0.1, half_width=0.05)

    assert interval.width == pytest.approx(0.1)
    assert interval.relative_width == pytest.approx(0.4)
    assert math.isinf(zero_estimate_interval.relative_width)
    assert interval.to_interval().estimate == pytest.approx(0.25)
    assert interval.to_dict("")["estimate"] == pytest.approx(0.25)
    assert IntervalEstimate(estimate=0.25, lower=0.2, upper=0.3).to_dict()["upper"] == pytest.approx(0.3)


def test_required_repeats_probability_error_bound_matches_paper_equation():
    assert required_repeats_for_probability_error(0.01) == 9600
    assert required_repeats_for_probability_error(0.03) == 1064


def test_required_repeats_lower_bound_and_exact_search_are_deterministic():
    assert required_repeats_lower_bound(0.1, 0.1) == 4180
    assert required_repeats_exact(0.1, 0.1) == 4605
    assert required_repeats_exact(0.01, 0.1) > required_repeats_exact(0.1, 0.1)
    assert required_repeats_exact(0.1, 0.1) > required_repeats_exact(0.5, 0.1)
    assert math.isinf(required_repeats_lower_bound(0.0, 0.1))
    assert math.isinf(required_repeats_exact(0.0, 0.1))
    assert math.isinf(required_repeats_exact(0.01, 0.001, max_repeats=2))
    assert required_repeats_exact(1.0, 0.1) == 1


def test_issue_73_dataframe_friendly_metrics_and_aliases():
    metrics = repeat_reliability_metrics(
        25,
        100,
        rtt_factor=3.0,
        iterations=4,
        effort_per_iteration=2.0,
    )
    df = pd.DataFrame([metrics.to_dict()])

    assert repeats_to_solution(0.5) == pytest.approx(repeat_count(0.5))
    assert "success_probability_estimate" in df.columns
    assert "success_probability_successes" in df.columns
    assert "r_c_relative_error" in df.columns
    assert "rtt_estimate" in df.columns
    assert "cets_estimate" in df.columns
    assert df.loc[0, "success_probability_successes"] == 25
    assert df.loc[0, "success_probability_trials"] == 100


def test_issue_73_interval_propagation_and_relative_error():
    metrics = propagate_success_probability_interval(
        0.5,
        0.4,
        0.6,
        rtt_factor=2.0,
        iterations=10,
        effort_per_iteration=0.5,
    )

    assert metrics.r_c.estimate == pytest.approx(repeat_count(0.5))
    assert metrics.r_c.lower == pytest.approx(repeat_count(0.6))
    assert metrics.r_c.upper == pytest.approx(repeat_count(0.4))
    assert metrics.rtt.estimate == pytest.approx(metrics.r_c.estimate * 2.0)
    assert metrics.cets.estimate == pytest.approx(metrics.r_c.estimate * 5.0)
    assert relative_repeats_error(10.0, 8.0, 13.0) == pytest.approx(0.3)

    with pytest.raises(ValueError):
        relative_repeats_error(10.0, 11.0, 13.0)


def test_issue_73_required_trials_wrapper_boundaries_and_methods():
    exact = required_trials_for_relative_error(0.9, method="exact")
    bound = required_trials_for_relative_error(0.9, method="bound")

    assert exact > bound
    assert required_trials_for_relative_error(0.1) > required_trials_for_relative_error(0.5)
    assert math.isinf(required_trials_for_relative_error(0.0))
    assert required_trials_for_relative_error(1.0) == 0

    with pytest.raises(ValueError):
        required_trials_for_relative_error(0.5, relative_error_threshold=0)
    with pytest.raises(ValueError):
        required_trials_for_relative_error(0.5, method="unsupported")
    with pytest.raises(ValueError):
        required_trials_for_relative_error(1.0, relative_error_threshold=0)
    with pytest.raises(ValueError):
        required_trials_for_relative_error(1.0, target_confidence=1.0)
    with pytest.raises(ValueError):
        required_trials_for_relative_error(0.5, method="bound", target_confidence=1.0)


def test_issue_74_report_supports_grouped_run_level_min_thresholds():
    df = pd.DataFrame(
        {
            "solver": ["alpha"] * 10 + ["beta"] * 10,
            "energy": [0.5] * 8 + [2.0] * 2 + [0.5] * 5 + [2.0] * 5,
            "runtime": [2.0] * 20,
            "iterations": [50] * 20,
        }
    )

    report = repeat_reliability_report(
        df,
        group_cols="solver",
        response_col="energy",
        success_rule="min",
        threshold=1.0,
        rtt_factor="runtime",
        iterations="iterations",
        effort_per_iteration=0.5,
    ).set_index("solver")

    assert report.loc["alpha", "successes"] == 8
    assert report.loc["alpha", "trials"] == 10
    assert report.loc["alpha", "success_rate"] == pytest.approx(0.8)
    assert report.loc["alpha", "p_hat"] > report.loc["beta", "p_hat"]
    assert report.loc["alpha", "R99"] < report.loc["beta", "R99"]
    assert report.loc["alpha", "RTT"] == pytest.approx(report.loc["alpha", "R99"] * 2.0)
    assert report.loc["alpha", "CETS"] == pytest.approx(report.loc["alpha", "R99"] * 25.0)
    assert {"best_point_estimate", "ci_overlaps_best", "statistically_unresolved"} <= set(report.columns)


def test_issue_74_report_supports_count_column_perf_ratio_data():
    df = pd.DataFrame(
        {
            "solver": ["alpha", "alpha", "beta", "beta"],
            "PerfRatio": [0.95, 0.50, 0.90, 0.10],
            "n": [90, 10, 70, 30],
        }
    )

    report = repeat_reliability_report(
        df,
        group_cols="solver",
        response_col="PerfRatio",
        success_rule="PerfRatio",
        threshold=0.8,
        count_col="n",
    ).set_index("solver")

    assert report.loc["alpha", "successes"] == 90
    assert report.loc["alpha", "trials"] == 100
    assert report.loc["beta", "successes"] == 70
    assert report.loc["beta", "trials"] == 100
    assert report.loc["alpha", "R99"] < report.loc["beta", "R99"]


@pytest.mark.parametrize(
    "kwargs,expected_successes",
    [
        ({"response_col": "value", "success_rule": "min", "threshold": 1.0}, 2),
        ({"response_col": "value", "success_rule": "max", "threshold": 1.0}, 2),
        (
            {
                "response_col": "value",
                "success_rule": "absolute",
                "threshold": 0.11,
                "target_value": 1.0,
            },
            2,
        ),
        (
            {
                "response_col": "value",
                "success_rule": "gap_min",
                "best_value": 1.0,
                "gap": 0.10,
            },
            2,
        ),
        (
            {
                "response_col": "value",
                "success_rule": "gap_max",
                "best_value": 1.0,
                "gap": 0.05,
            },
            2,
        ),
    ],
)
def test_issue_74_success_rule_variants(kwargs, expected_successes):
    df = pd.DataFrame({"value": [0.9, 1.0, 1.2]})

    report = repeat_reliability_report(df, **kwargs)

    assert report.loc[0, "successes"] == expected_successes
    assert report.loc[0, "trials"] == 3


def test_issue_74_comparison_flags_mark_unresolved_intervals():
    df = pd.DataFrame(
        {
            "budget": [10, 10, 10],
            "candidate": ["alpha", "beta", "gamma"],
            "R99": [10.0, 11.0, 20.0],
            "R99_ci_lower": [8.0, 9.0, 18.0],
            "R99_ci_upper": [12.0, 13.0, 22.0],
        }
    )

    flagged = annotate_reliability_comparisons(df, comparison_cols="budget").set_index("candidate")

    assert flagged.loc["alpha", "best_point_estimate"]
    assert flagged.loc["alpha", "statistically_unresolved"]
    assert flagged.loc["beta", "ci_overlaps_best"]
    assert not flagged.loc["gamma", "statistically_unresolved"]


def test_issue_74_empty_report_keeps_joinable_columns_and_flags():
    df = pd.DataFrame({"solver": [], "value": []})

    report = repeat_reliability_report(
        df,
        group_cols="solver",
        response_col="value",
        success_rule="min",
        threshold=1.0,
        comparison_cols="budget",
    )

    assert report.empty
    assert {"solver", "budget", "successes", "R99", "best_point_estimate"} <= set(report.columns)


def test_issue_74_report_accepts_success_column_and_rejects_null_successes():
    df = pd.DataFrame({"solver": ["alpha", "alpha"], "ok": [True, False], "n": [2, 3]})

    report = repeat_reliability_report(
        df,
        group_cols="solver",
        success_col="ok",
        count_col="n",
        add_comparison_flags=False,
    )

    assert report.loc[0, "successes"] == 2
    assert report.loc[0, "trials"] == 5

    with pytest.raises(ValueError, match="success_col"):
        repeat_reliability_report(
            pd.DataFrame({"ok": [True, None]}),
            success_col="ok",
        )


@pytest.mark.parametrize(
    "counts,match",
    [
        (["bad"], "numeric"),
        ([-1], "non-negative"),
        ([1.5], "integer"),
    ],
)
def test_issue_74_report_rejects_invalid_count_columns(counts, match):
    df = pd.DataFrame({"value": [1.0], "n": counts})

    with pytest.raises(ValueError, match=match):
        repeat_reliability_report(
            df,
            response_col="value",
            success_rule="min",
            threshold=1.0,
            count_col="n",
        )


def test_issue_74_gap_rule_supports_percent_random_reference_and_response_dir():
    df = pd.DataFrame(
        {
            "response": [10.0, 11.0, 12.0],
            "best": [10.0, 10.0, 10.0],
            "random": [20.0, 20.0, 20.0],
        }
    )

    report = repeat_reliability_report(
        df,
        response_col="response",
        success_rule="gap",
        response_dir=-1,
        best_value="best",
        random_value="random",
        gap=10.0,
        gap_is_percent=True,
    )

    assert report.loc[0, "successes"] == 2


def test_issue_74_report_validation_errors_cover_public_api_edges():
    df = pd.DataFrame({"value": [1.0]})

    with pytest.raises(TypeError, match="pandas DataFrame"):
        repeat_reliability_report(object())
    with pytest.raises(ValueError, match="missing required columns"):
        repeat_reliability_report(df, group_cols="missing", response_col="value", threshold=1.0)
    with pytest.raises(ValueError, match="success_rule"):
        repeat_reliability_report(df, response_col="value", success_rule=object(), threshold=1.0)
    with pytest.raises(ValueError, match="unsupported success_rule"):
        repeat_reliability_report(df, response_col="value", success_rule="unknown", threshold=1.0)
    with pytest.raises(ValueError, match="response_col"):
        repeat_reliability_report(df, success_rule="min", threshold=1.0)
    with pytest.raises(ValueError, match="threshold"):
        repeat_reliability_report(df, response_col="value", success_rule="min")
    with pytest.raises(ValueError, match="threshold"):
        repeat_reliability_report(df, response_col="value", success_rule="min", threshold="not-a-number")
    with pytest.raises(ValueError, match="threshold"):
        repeat_reliability_report(
            pd.DataFrame({"value": [1.0], "threshold": [np.nan]}),
            response_col="value",
            success_rule="min",
            threshold="threshold",
        )
    with pytest.raises(ValueError, match="gap"):
        repeat_reliability_report(
            df,
            response_col="value",
            success_rule="gap_min",
            best_value=1.0,
            gap=-0.1,
        )
    with pytest.raises(ValueError, match="rtt_factor"):
        repeat_reliability_report(
            df,
            response_col="value",
            success_rule="min",
            threshold=1.0,
            rtt_factor=-1.0,
        )


def test_issue_74_zero_count_rows_produce_empty_trial_report():
    df = pd.DataFrame({"value": [1.0], "n": [0], "runtime": [3.0]})

    report = repeat_reliability_report(
        df,
        response_col="value",
        success_rule="min",
        threshold=1.0,
        count_col="n",
        rtt_factor="runtime",
        add_comparison_flags=False,
    )

    assert report.loc[0, "successes"] == 0
    assert report.loc[0, "trials"] == 0
    assert math.isnan(report.loc[0, "success_rate"])
    assert report.loc[0, "rtt_factor"] == pytest.approx(3.0)


def test_issue_74_comparison_flags_handle_max_objective_and_invalid_intervals():
    df = pd.DataFrame(
        {
            "candidate": ["alpha", "beta", "gamma"],
            "p_hat": [0.8, 0.7, np.nan],
            "p_ci_lower": [0.7, 0.6, np.nan],
            "p_ci_upper": [0.9, 0.69, np.nan],
        }
    )

    flagged = annotate_reliability_comparisons(
        df,
        estimate_col="p_hat",
        lower_col="p_ci_lower",
        upper_col="p_ci_upper",
        objective="max",
    ).set_index("candidate")

    assert flagged.loc["alpha", "best_point_estimate"]
    assert not flagged.loc["beta", "statistically_unresolved"]
    assert not flagged.loc["gamma", "ci_overlaps_best"]

    empty = annotate_reliability_comparisons(
        df.iloc[0:0],
        estimate_col="p_hat",
        lower_col="p_ci_lower",
        upper_col="p_ci_upper",
        objective="max",
    )
    assert empty.empty

    with pytest.raises(ValueError, match="interval lower"):
        annotate_reliability_comparisons(
            pd.DataFrame(
                {
                    "R99": [1.0, 2.0],
                    "R99_ci_lower": [2.0, 1.5],
                    "R99_ci_upper": [1.0, 2.5],
                }
            )
        )
    with pytest.raises(ValueError, match="objective"):
        annotate_reliability_comparisons(
            pd.DataFrame({"R99": [1.0], "R99_ci_lower": [0.9], "R99_ci_upper": [1.1]}),
            objective="better",
        )
    with pytest.raises(TypeError, match="pandas DataFrame"):
        annotate_reliability_comparisons(object())


def test_issue_74_comparison_flags_skip_all_nan_estimates():
    df = pd.DataFrame({"R99": [np.nan], "R99_ci_lower": [0.0], "R99_ci_upper": [1.0]})

    flagged = annotate_reliability_comparisons(df, objective=1)

    assert not flagged.loc[0, "best_point_estimate"]
    assert not flagged.loc[0, "statistically_unresolved"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: normal_critical_value(1.0),
        lambda: normal_critical_value("bad"),
        lambda: agresti_coull_interval(0),
        lambda: agresti_coull_interval(0, 10, trials=11),
        lambda: agresti_coull_interval("bad", 10),
        lambda: agresti_coull_interval(-1, 10),
        lambda: agresti_coull_interval(1.5, 10),
        lambda: agresti_coull_interval(0, True),
        lambda: agresti_coull_interval(0, "bad"),
        lambda: agresti_coull_interval(0, 1.5),
        lambda: agresti_coull_interval(0, -1),
        lambda: agresti_coull_interval(0, 10, confidence_level="bad"),
        lambda: agresti_coull_interval(0, 10, confidence_fraction="bad"),
        lambda: agresti_coull_interval_from_estimate(-0.1, 10),
        lambda: success_probability_margin(0.5, 0),
        lambda: repeat_count("bad"),
        lambda: repeat_count(0.5, target_confidence=0.0),
        lambda: repeat_count(0.5, target_confidence=1.0),
        lambda: repeat_count_interval(0.8, 0.2),
        lambda: repeat_count_interval(0.1, 0.2, target_confidence=0.0),
        lambda: propagate_success_probability_interval(0.5),
        lambda: propagate_success_probability_interval(0.5, 0.6, 0.4),
        lambda: propagate_success_probability_interval(
            ProportionInterval(estimate=0.5, lower=0.4, upper=0.6, half_width=0.1),
            0.4,
            0.6,
        ),
        lambda: propagate_success_probability_interval(0.5, 0.4, 0.6, rtt_factor=-1),
        lambda: propagate_success_probability_interval(0.5, 0.4, 0.6, iterations=-1),
        lambda: propagate_success_probability_interval(0.5, 0.4, 0.6, effort_per_iteration=-1),
        lambda: cets_from_repeat_count(1.0, iterations=-1),
        lambda: cets_from_repeat_count(1.0, iterations=1, effort_per_iteration=-1),
        lambda: cets_from_repeat_count(math.inf, iterations=0),
        lambda: rtt_from_repeat_count(1.0, runtime_per_repeat=-1),
        lambda: rtt_from_repeat_count(math.inf, runtime_per_repeat=0),
        lambda: scaled_repeat_count_interval(RepeatCountInterval(1.0, 2.0), scale=-1),
        lambda: scaled_repeat_count_interval(RepeatCountInterval(math.inf, math.inf), scale=0),
        lambda: required_repeats_for_probability_error(0.0),
        lambda: required_repeats_for_probability_error(0.1, min_repeats=-1),
        lambda: required_repeats_lower_bound(0.5, 0.1, min_repeats=-1),
        lambda: required_repeats_lower_bound(0.5, 1.0),
        lambda: required_repeats_lower_bound(0.5, "bad"),
        lambda: required_repeats_exact(0.5, 0.1, target_confidence=0.0),
        lambda: required_repeats_exact(0.5, 0.1, target_confidence=1.0),
        lambda: required_repeats_exact(0.5, 0.1, min_repeats=0),
        lambda: required_repeats_exact(0.5, 0.1, min_repeats=10, max_repeats=1),
    ],
)
def test_validation_rejects_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_maximum_relative_error_rejects_nonfinite_estimates():
    assert math.isinf(maximum_relative_error(0.0, 0.0, 1.0))
    assert math.isinf(maximum_relative_error(1.0, 0.0, math.inf))


def test_fixed_seed_bernoulli_rc_interval_is_asymmetric():
    rng = np.random.default_rng(202603)
    successes = int(rng.binomial(100, 0.1))

    interval = agresti_coull_interval(successes, 100)
    repeat_estimate = repeat_count(interval.estimate)
    repeat_interval = repeat_count_interval(interval.lower, interval.upper)

    assert successes == 13
    assert repeat_interval.upper - repeat_estimate > repeat_estimate - repeat_interval.lower


def test_fixed_seed_low_success_probability_requires_more_repeats():
    rng = np.random.default_rng(202604)
    low_successes = int(rng.binomial(1000, 0.1))
    medium_successes = int(rng.binomial(1000, 0.5))

    low_estimate = agresti_coull_interval(low_successes, 1000).estimate
    medium_estimate = agresti_coull_interval(medium_successes, 1000).estimate

    assert low_estimate < medium_estimate
    assert required_repeats_lower_bound(low_estimate, 0.1) > required_repeats_lower_bound(medium_estimate, 0.1)


def test_fixed_seed_noisy_point_estimates_can_invert_rankings():
    rng = np.random.default_rng(1)
    better_true_successes = int(rng.binomial(100, 0.25))
    worse_true_successes = int(rng.binomial(100, 0.2))

    better_estimate = agresti_coull_interval(better_true_successes, 100).estimate
    worse_estimate = agresti_coull_interval(worse_true_successes, 100).estimate

    assert better_true_successes == 25
    assert worse_true_successes == 27
    assert better_estimate < worse_estimate
    assert repeat_count(better_estimate) > repeat_count(worse_estimate)
