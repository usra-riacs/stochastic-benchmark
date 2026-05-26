import math

import numpy as np
import pytest

from repeat_reliability import (
    agresti_coull_interval,
    agresti_coull_interval_from_estimate,
    cets_from_repeat_count,
    maximum_relative_error,
    repeat_count,
    repeat_count_interval,
    required_repeats_exact,
    required_repeats_for_probability_error,
    required_repeats_lower_bound,
    rtt_from_repeat_count,
    scaled_repeat_count_interval,
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


def test_rtt_and_cets_scaling_preserve_relative_error():
    repeat_estimate = repeat_count(0.25, confidence=0.99)
    repeat_interval = repeat_count_interval(0.22, 0.28, confidence=0.99)
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


def test_required_repeats_probability_error_bound_matches_paper_equation():
    assert required_repeats_for_probability_error(0.01) == 9600
    assert required_repeats_for_probability_error(0.03) == 1064


def test_required_repeats_lower_bound_and_exact_search_are_deterministic():
    assert required_repeats_lower_bound(0.1, 0.1) == 4180
    assert required_repeats_exact(0.1, 0.1) == 4605
    assert required_repeats_exact(0.01, 0.1) > required_repeats_exact(0.1, 0.1)
    assert required_repeats_exact(0.1, 0.1) > required_repeats_exact(0.5, 0.1)


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
