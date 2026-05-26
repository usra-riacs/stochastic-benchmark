"""Reference formulas for repeat-reliability validation.

The functions in this module are small, deterministic implementations of the
equations from Noori et al., "A Statistical Analysis for Per-Instance
Evaluation of Stochastic Optimizers: Avoiding Unreliable Conclusions."
They are intended as reviewable references for tests and later integrations.
"""

from dataclasses import dataclass
import math

from scipy import stats as scipy_stats


@dataclass(frozen=True)
class ProportionInterval:
    """Agresti-Coull interval for a binomial success probability."""

    estimate: float
    lower: float
    upper: float
    half_width: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def relative_width(self) -> float:
        if self.estimate == 0:
            return math.inf
        return self.width / self.estimate


@dataclass(frozen=True)
class RepeatCountInterval:
    """Confidence interval induced on R_c by a probability interval."""

    lower: float
    upper: float


def normal_critical_value(confidence_fraction: float = 0.95) -> float:
    """Return z_alpha for a two-sided confidence interval."""

    _validate_open_probability(confidence_fraction, "confidence_fraction")
    return float(scipy_stats.norm.ppf(0.5 + confidence_fraction / 2.0))


def agresti_coull_interval(
    successes: float,
    repeats: int,
    confidence_fraction: float = 0.95,
) -> ProportionInterval:
    """Compute the Agresti-Coull interval from paper Eq. (8).

    Eq. (8) defines n_hat = n + z_alpha^2, n_s_hat = n_s + z_alpha^2 / 2,
    p_hat = n_s_hat / n_hat, and the interval
    p_hat +/- z_alpha * sqrt(p_hat * (1 - p_hat) / n_hat).
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if successes < 0 or successes > repeats:
        raise ValueError("successes must be between 0 and repeats")

    z = normal_critical_value(confidence_fraction)
    adjusted_repeats = repeats + z**2
    estimate = (successes + z**2 / 2.0) / adjusted_repeats
    half_width = z * math.sqrt(estimate * (1.0 - estimate) / adjusted_repeats)
    return ProportionInterval(
        estimate=estimate,
        lower=max(0.0, estimate - half_width),
        upper=min(1.0, estimate + half_width),
        half_width=half_width,
    )


def agresti_coull_interval_from_estimate(
    success_probability: float,
    repeats: int,
    confidence_fraction: float = 0.95,
) -> ProportionInterval:
    """Compute Eq. (8) from an observed success fraction n_s / n."""

    _validate_probability(success_probability, "success_probability")
    return agresti_coull_interval(
        successes=success_probability * repeats,
        repeats=repeats,
        confidence_fraction=confidence_fraction,
    )


def success_probability_margin(
    adjusted_success_probability: float,
    repeats: int,
    confidence_fraction: float = 0.95,
) -> float:
    """Return epsilon_p from paper Eq. (11) for an adjusted p_hat."""

    _validate_probability(adjusted_success_probability, "adjusted_success_probability")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    z = normal_critical_value(confidence_fraction)
    adjusted_repeats = repeats + z**2
    return z * math.sqrt(
        adjusted_success_probability * (1.0 - adjusted_success_probability)
        / adjusted_repeats
    )


def repeat_count(success_probability: float, target_confidence: float = 0.99) -> float:
    """Compute R_c from paper Eqs. (1) and (2)."""

    _validate_probability(success_probability, "success_probability")
    _validate_open_probability(target_confidence, "target_confidence")

    if success_probability == 0:
        return math.inf
    if success_probability >= target_confidence:
        return 1.0
    return max(
        math.log1p(-target_confidence) / math.log1p(-success_probability),
        1.0,
    )


def repeat_count_interval(
    success_probability_lower: float,
    success_probability_upper: float,
    target_confidence: float = 0.99,
) -> RepeatCountInterval:
    """Induce an R_c interval from a probability interval.

    This implements the monotonic transformation described in the confidence
    interval section: R_c^- uses the upper probability bound and R_c^+ uses the
    lower probability bound.
    """

    _validate_probability(success_probability_lower, "success_probability_lower")
    _validate_probability(success_probability_upper, "success_probability_upper")
    _validate_open_probability(target_confidence, "target_confidence")
    if success_probability_lower > success_probability_upper:
        raise ValueError("success_probability_lower must be <= success_probability_upper")

    return RepeatCountInterval(
        lower=repeat_count(success_probability_upper, target_confidence=target_confidence),
        upper=repeat_count(success_probability_lower, target_confidence=target_confidence),
    )


def cets_from_repeat_count(
    repeats_to_confidence: float,
    iterations: float,
    effort_per_iteration: float = 1.0,
) -> float:
    """Scale R_c to CETS from paper Eq. (3)."""

    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if effort_per_iteration < 0:
        raise ValueError("effort_per_iteration must be non-negative")
    return _scale_repeat_count_value(
        repeats_to_confidence,
        iterations * effort_per_iteration,
    )


def rtt_from_repeat_count(
    repeats_to_confidence: float,
    runtime_per_repeat: float = 1.0,
) -> float:
    """Scale R_c to runtime-to-target/time-to-solution."""

    if runtime_per_repeat < 0:
        raise ValueError("runtime_per_repeat must be non-negative")
    return _scale_repeat_count_value(repeats_to_confidence, runtime_per_repeat)


def scaled_repeat_count_interval(
    interval: RepeatCountInterval,
    scale: float,
) -> RepeatCountInterval:
    """Apply deterministic RTT/CETS scaling to an R_c interval."""

    if scale < 0:
        raise ValueError("scale must be non-negative")
    return RepeatCountInterval(
        lower=_scale_repeat_count_value(interval.lower, scale),
        upper=_scale_repeat_count_value(interval.upper, scale),
    )


def maximum_relative_error(
    estimate: float,
    lower: float,
    upper: float,
) -> float:
    """Compute the maximum relative R_c error from paper Eq. (13)."""

    if estimate <= 0 or not math.isfinite(estimate):
        return math.inf
    if not math.isfinite(lower) or not math.isfinite(upper):
        return math.inf
    return max(estimate - lower, upper - estimate, 0.0) / estimate


def required_repeats_for_probability_error(
    error_tolerance: float,
    confidence_fraction: float = 0.95,
    min_repeats: int = 1,
) -> int:
    """Return the worst-case n bound from paper Eq. (10)."""

    if error_tolerance <= 0:
        raise ValueError("error_tolerance must be positive")
    if min_repeats < 0:
        raise ValueError("min_repeats must be non-negative")

    z = normal_critical_value(confidence_fraction)
    bound = (z / (2.0 * error_tolerance)) ** 2 - z**2
    return max(min_repeats, math.ceil(bound))


def required_repeats_lower_bound(
    adjusted_success_probability: float,
    relative_error_threshold: float,
    confidence_fraction: float = 0.95,
    min_repeats: int = 1,
) -> float:
    """Return the conservative lower-bound repeat count from paper Eq. (19)."""

    _validate_probability(adjusted_success_probability, "adjusted_success_probability")
    _validate_relative_error_threshold(relative_error_threshold)
    if min_repeats < 0:
        raise ValueError("min_repeats must be non-negative")

    if adjusted_success_probability == 0:
        return math.inf

    z = normal_critical_value(confidence_fraction)
    threshold_factor = z * (1.0 + relative_error_threshold) / relative_error_threshold
    bound = (
        threshold_factor**2
        * (1.0 - adjusted_success_probability)
        / adjusted_success_probability
        - z**2
    )
    return max(min_repeats, math.ceil(bound))


def required_repeats_exact(
    adjusted_success_probability: float,
    relative_error_threshold: float,
    target_confidence: float = 0.99,
    confidence_fraction: float = 0.95,
    min_repeats: int = 1,
    max_repeats: int = 100_000_000,
) -> float:
    """Find the smallest n whose induced R_c error satisfies Eq. (14).

    This is the exact numerical counterpart to the Eq. (19) lower bound. It
    keeps p_hat fixed, uses Eq. (11) to compute epsilon_p(n), clips the induced
    probability interval to [0, 1], and searches for the first n where the
    induced Eq. (13) relative error is within the requested threshold.
    """

    _validate_probability(adjusted_success_probability, "adjusted_success_probability")
    _validate_relative_error_threshold(relative_error_threshold)
    _validate_open_probability(target_confidence, "target_confidence")
    if min_repeats <= 0:
        raise ValueError("min_repeats must be positive")
    if max_repeats < min_repeats:
        raise ValueError("max_repeats must be >= min_repeats")
    if adjusted_success_probability == 0:
        return math.inf

    estimate = repeat_count(adjusted_success_probability, target_confidence=target_confidence)

    if _repeat_count_error_satisfies_threshold(
        adjusted_success_probability,
        estimate,
        min_repeats,
        relative_error_threshold,
        target_confidence,
        confidence_fraction,
    ):
        return min_repeats

    low = min_repeats
    high = min_repeats
    while high < max_repeats:
        high = min(high * 2, max_repeats)
        if _repeat_count_error_satisfies_threshold(
            adjusted_success_probability,
            estimate,
            high,
            relative_error_threshold,
            target_confidence,
            confidence_fraction,
        ):
            break
        low = high + 1
    else:
        return math.inf

    while low < high:
        midpoint = (low + high) // 2
        if _repeat_count_error_satisfies_threshold(
            adjusted_success_probability,
            estimate,
            midpoint,
            relative_error_threshold,
            target_confidence,
            confidence_fraction,
        ):
            high = midpoint
        else:
            low = midpoint + 1
    return low


def _repeat_count_error_satisfies_threshold(
    adjusted_success_probability: float,
    repeat_estimate: float,
    repeats: int,
    relative_error_threshold: float,
    target_confidence: float,
    confidence_fraction: float,
) -> bool:
    margin = success_probability_margin(
        adjusted_success_probability,
        repeats,
        confidence_fraction=confidence_fraction,
    )
    lower_probability = max(0.0, adjusted_success_probability - margin)
    upper_probability = min(1.0, adjusted_success_probability + margin)
    interval = repeat_count_interval(
        lower_probability,
        upper_probability,
        target_confidence=target_confidence,
    )
    error = maximum_relative_error(repeat_estimate, interval.lower, interval.upper)
    return error <= relative_error_threshold


def _scale_repeat_count_value(repeats_to_confidence: float, scale: float) -> float:
    if scale == 0 and not math.isfinite(repeats_to_confidence):
        raise ValueError("zero scale is undefined for non-finite repeat counts")
    return scale * repeats_to_confidence


def _validate_probability(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_open_probability(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1, exclusive")


def _validate_relative_error_threshold(value: float) -> None:
    if not 0 < value < 1:
        raise ValueError("relative_error_threshold must be between 0 and 1")
