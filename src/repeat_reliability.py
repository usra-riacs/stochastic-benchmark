"""Repeat reliability primitives for stochastic optimizer evaluations.

This module contains deterministic implementations of the repeat-reliability
formulas used in Noori et al., "Statistical analysis for per-instance
evaluation of stochastic optimizers: Avoiding unreliable conclusions."
The helpers are independent from the existing bootstrap pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from scipy import stats as scipy_stats


DEFAULT_CONFIDENCE_LEVEL = 95.0
DEFAULT_CONFIDENCE_FRACTION = 0.95
DEFAULT_TARGET_CONFIDENCE = 0.99
DEFAULT_RELATIVE_ERROR_THRESHOLD = 0.10


@dataclass(frozen=True)
class IntervalEstimate:
    """Structured point estimate and confidence interval."""

    estimate: float
    lower: float
    upper: float

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return _prefixed_dict(
            prefix,
            {
                "estimate": self.estimate,
                "lower": self.lower,
                "upper": self.upper,
            },
        )


@dataclass(frozen=True)
class ProportionInterval:
    """Agresti-Coull interval for a binomial success probability."""

    estimate: float
    lower: float
    upper: float
    half_width: float
    successes: float | None = None
    trials: int | None = None
    confidence_level: float | None = None
    confidence_fraction: float | None = None
    z_value: float | None = None
    adjusted_successes: float | None = None
    adjusted_trials: float | None = None
    raw_estimate: float | None = None

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def relative_width(self) -> float:
        if self.estimate == 0:
            return math.inf
        return self.width / self.estimate

    def to_interval(self) -> IntervalEstimate:
        return IntervalEstimate(self.estimate, self.lower, self.upper)

    def to_dict(self, prefix: str = "success_probability") -> dict[str, float | int]:
        values: dict[str, float | int] = {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "half_width": self.half_width,
        }
        optional_values = {
            "successes": self.successes,
            "trials": self.trials,
            "confidence_level": self.confidence_level,
            "confidence_fraction": self.confidence_fraction,
            "z_value": self.z_value,
            "adjusted_successes": self.adjusted_successes,
            "adjusted_trials": self.adjusted_trials,
            "raw_estimate": self.raw_estimate,
        }
        values.update(
            {key: value for key, value in optional_values.items() if value is not None}
        )
        return _prefixed_dict(prefix, values)


AgrestiCoullInterval = ProportionInterval


@dataclass(frozen=True)
class RepeatCountInterval:
    """Confidence interval induced on R_c by a probability interval."""

    lower: float
    upper: float


@dataclass(frozen=True)
class MetricInterval:
    """Structured interval for propagated repeat-derived metrics."""

    estimate: float
    lower: float
    upper: float
    relative_error: float

    def to_dict(self, prefix: str) -> dict[str, float]:
        return _prefixed_dict(
            prefix,
            {
                "estimate": self.estimate,
                "lower": self.lower,
                "upper": self.upper,
                "relative_error": self.relative_error,
            },
        )


@dataclass(frozen=True)
class RepeatReliabilityMetrics:
    """Probability, R_c, RTT, and CETS estimates with propagated intervals."""

    success_probability: ProportionInterval | IntervalEstimate
    r_c: MetricInterval
    rtt: MetricInterval
    cets: MetricInterval
    target_confidence: float
    rtt_factor: float
    iterations: float
    effort_per_iteration: float

    def to_dict(self) -> dict[str, float | int]:
        data: dict[str, float | int] = {}
        data.update(self.success_probability.to_dict("success_probability"))
        data.update(self.r_c.to_dict("r_c"))
        data.update(self.rtt.to_dict("rtt"))
        data.update(self.cets.to_dict("cets"))
        data.update(
            {
                "target_confidence": self.target_confidence,
                "rtt_factor": self.rtt_factor,
                "iterations": self.iterations,
                "effort_per_iteration": self.effort_per_iteration,
            }
        )
        return data


def normal_critical_value(confidence_fraction: float = DEFAULT_CONFIDENCE_FRACTION) -> float:
    """Return z_alpha for a two-sided confidence interval."""

    confidence = _validate_open_probability(confidence_fraction, "confidence_fraction")
    return float(scipy_stats.norm.ppf(0.5 + confidence / 2.0))


def agresti_coull_interval(
    successes: float,
    repeats: int | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    *,
    trials: int | None = None,
    confidence_fraction: float | None = None,
) -> ProportionInterval:
    """Compute the Agresti-Coull interval for a Bernoulli success probability.

    ``confidence_level`` may be provided as a percentage (95) or a fraction
    (0.95). ``confidence_fraction`` is retained for compatibility with the
    validation helpers introduced for issue #72. Zero trials are defined by the
    adjusted Agresti-Coull formula and return an adjusted estimate of 0.5 with a
    clipped interval of [0, 1].
    """

    repeat_count_value = _coerce_repeats_argument(repeats, trials)
    return _compute_agresti_coull_interval(
        successes,
        repeat_count_value,
        _confidence_fraction_from_level(confidence_level, confidence_fraction),
        require_integer_successes=True,
        allow_zero_repeats=True,
    )


def agresti_coull_interval_from_estimate(
    success_probability: float,
    repeats: int,
    confidence_fraction: float = DEFAULT_CONFIDENCE_FRACTION,
) -> ProportionInterval:
    """Compute the Agresti-Coull interval from an observed success fraction."""

    probability = _validate_probability(success_probability, "success_probability")
    repeat_count_value = _validate_repeats(repeats, "repeats", allow_zero=False)
    return _compute_agresti_coull_interval(
        probability * repeat_count_value,
        repeat_count_value,
        confidence_fraction,
        require_integer_successes=False,
        allow_zero_repeats=False,
    )


def success_probability_margin(
    adjusted_success_probability: float,
    repeats: int,
    confidence_fraction: float = DEFAULT_CONFIDENCE_FRACTION,
) -> float:
    """Return epsilon_p from the paper for an adjusted p_hat."""

    probability = _validate_probability(
        adjusted_success_probability,
        "adjusted_success_probability",
    )
    repeat_count_value = _validate_repeats(repeats, "repeats", allow_zero=False)

    z = normal_critical_value(confidence_fraction)
    adjusted_repeats = repeat_count_value + z**2
    return z * math.sqrt(probability * (1.0 - probability) / adjusted_repeats)


def repeat_count(success_probability: float, target_confidence: float = 0.99) -> float:
    """Compute R_c from the success probability and target confidence."""

    probability = _validate_probability(success_probability, "success_probability")
    target = _validate_open_probability(target_confidence, "target_confidence")

    if probability == 0:
        return math.inf
    if probability >= target:
        return 1.0
    return max(
        math.log1p(-target) / math.log1p(-probability),
        1.0,
    )


def repeats_to_solution(
    p: float,
    target_confidence: float = DEFAULT_TARGET_CONFIDENCE,
) -> float:
    """Return the continuous R_c repeats needed to succeed with confidence c."""

    return repeat_count(p, target_confidence=target_confidence)


def repeat_count_interval(
    success_probability_lower: float,
    success_probability_upper: float,
    target_confidence: float = 0.99,
) -> RepeatCountInterval:
    """Induce an R_c interval from a probability interval."""

    lower_probability = _validate_probability(
        success_probability_lower,
        "success_probability_lower",
    )
    upper_probability = _validate_probability(
        success_probability_upper,
        "success_probability_upper",
    )
    _validate_open_probability(target_confidence, "target_confidence")
    if lower_probability > upper_probability:
        raise ValueError("success_probability_lower must be <= success_probability_upper")

    return RepeatCountInterval(
        lower=repeat_count(upper_probability, target_confidence=target_confidence),
        upper=repeat_count(lower_probability, target_confidence=target_confidence),
    )


def cets_from_repeat_count(
    repeats_to_confidence: float,
    iterations: float,
    effort_per_iteration: float = 1.0,
) -> float:
    """Scale R_c to CETS."""

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
    """Compute the maximum relative R_c error."""

    if estimate <= 0 or not math.isfinite(estimate):
        return math.inf
    if not math.isfinite(lower) or not math.isfinite(upper):
        return math.inf
    return max(estimate - lower, upper - estimate, 0.0) / estimate


def relative_repeats_error(r_c: float, r_c_lower: float, r_c_upper: float) -> float:
    """Return max relative error for an R_c point estimate and interval."""

    if r_c_lower > r_c or r_c_upper < r_c:
        raise ValueError("repeat interval must satisfy lower <= estimate <= upper")
    return maximum_relative_error(r_c, r_c_lower, r_c_upper)


def propagate_success_probability_interval(
    success_probability: float | IntervalEstimate | ProportionInterval,
    probability_lower: float | None = None,
    probability_upper: float | None = None,
    *,
    target_confidence: float = DEFAULT_TARGET_CONFIDENCE,
    rtt_factor: float = 1.0,
    iterations: float = 1.0,
    effort_per_iteration: float = 1.0,
) -> RepeatReliabilityMetrics:
    """Propagate a success-probability interval to R_c, RTT, and CETS."""

    probability_interval = _coerce_probability_interval(
        success_probability,
        probability_lower,
        probability_upper,
    )
    target = _validate_open_probability(target_confidence, "target_confidence")
    if rtt_factor < 0:
        raise ValueError("rtt_factor must be non-negative")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if effort_per_iteration < 0:
        raise ValueError("effort_per_iteration must be non-negative")

    r_c_estimate = repeat_count(probability_interval.estimate, target)
    r_c_bounds = repeat_count_interval(
        probability_interval.lower,
        probability_interval.upper,
        target_confidence=target,
    )
    r_c = MetricInterval(
        estimate=r_c_estimate,
        lower=r_c_bounds.lower,
        upper=r_c_bounds.upper,
        relative_error=maximum_relative_error(
            r_c_estimate,
            r_c_bounds.lower,
            r_c_bounds.upper,
        ),
    )

    rtt = _scale_metric_interval(r_c, rtt_factor)
    cets = _scale_metric_interval(r_c, iterations * effort_per_iteration)
    return RepeatReliabilityMetrics(
        success_probability=probability_interval,
        r_c=r_c,
        rtt=rtt,
        cets=cets,
        target_confidence=target,
        rtt_factor=rtt_factor,
        iterations=iterations,
        effort_per_iteration=effort_per_iteration,
    )


def repeat_reliability_metrics(
    successes: float,
    trials: int,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    target_confidence: float = DEFAULT_TARGET_CONFIDENCE,
    rtt_factor: float = 1.0,
    iterations: float = 1.0,
    effort_per_iteration: float = 1.0,
) -> RepeatReliabilityMetrics:
    """Compute Agresti-Coull probability and propagated repeat metrics."""

    probability_interval = agresti_coull_interval(
        successes,
        trials=trials,
        confidence_level=confidence_level,
    )
    return propagate_success_probability_interval(
        probability_interval,
        target_confidence=target_confidence,
        rtt_factor=rtt_factor,
        iterations=iterations,
        effort_per_iteration=effort_per_iteration,
    )


def required_repeats_for_probability_error(
    error_tolerance: float,
    confidence_fraction: float = DEFAULT_CONFIDENCE_FRACTION,
    min_repeats: int = 1,
) -> int:
    """Return the worst-case n bound for an absolute probability error."""

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
    confidence_fraction: float = DEFAULT_CONFIDENCE_FRACTION,
    min_repeats: int = 1,
) -> float:
    """Return the closed-form lower-bound repeat count approximation."""

    probability = _validate_probability(
        adjusted_success_probability,
        "adjusted_success_probability",
    )
    _validate_relative_error_threshold(relative_error_threshold)
    if min_repeats < 0:
        raise ValueError("min_repeats must be non-negative")

    if probability == 0:
        return math.inf

    z = normal_critical_value(confidence_fraction)
    threshold_factor = z * (1.0 + relative_error_threshold) / relative_error_threshold
    bound = threshold_factor**2 * (1.0 - probability) / probability - z**2
    return max(min_repeats, math.ceil(bound))


def required_repeats_exact(
    adjusted_success_probability: float,
    relative_error_threshold: float,
    target_confidence: float = 0.99,
    confidence_fraction: float = DEFAULT_CONFIDENCE_FRACTION,
    min_repeats: int = 1,
    max_repeats: int = 100_000_000,
) -> float:
    """Find the smallest n whose induced R_c error satisfies the threshold."""

    probability = _validate_probability(
        adjusted_success_probability,
        "adjusted_success_probability",
    )
    _validate_relative_error_threshold(relative_error_threshold)
    _validate_open_probability(target_confidence, "target_confidence")
    if min_repeats <= 0:
        raise ValueError("min_repeats must be positive")
    if max_repeats < min_repeats:
        raise ValueError("max_repeats must be >= min_repeats")
    if probability == 0:
        return math.inf

    estimate = repeat_count(probability, target_confidence=target_confidence)

    if _repeat_count_error_satisfies_threshold(
        probability,
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
            probability,
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
            probability,
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


def required_trials_for_relative_error(
    p: float,
    *,
    relative_error_threshold: float = DEFAULT_RELATIVE_ERROR_THRESHOLD,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    target_confidence: float = DEFAULT_TARGET_CONFIDENCE,
    method: Literal["exact", "bound"] = "exact",
) -> int | float:
    """Return trials needed to keep R_c relative error under a threshold."""

    probability = _validate_probability(p, "p")
    confidence_fraction = _confidence_fraction_from_level(confidence_level, None)
    if method not in {"exact", "bound"}:
        raise ValueError("method must be either 'exact' or 'bound'")
    if probability == 1.0:
        return 0

    if method == "bound":
        return required_repeats_lower_bound(
            probability,
            relative_error_threshold,
            confidence_fraction=confidence_fraction,
            min_repeats=0,
        )
    return required_repeats_exact(
        probability,
        relative_error_threshold,
        target_confidence=target_confidence,
        confidence_fraction=confidence_fraction,
    )


def _compute_agresti_coull_interval(
    successes: float,
    repeats: int,
    confidence_fraction: float,
    *,
    require_integer_successes: bool,
    allow_zero_repeats: bool,
) -> ProportionInterval:
    repeat_count_value = _validate_repeats(
        repeats,
        "repeats",
        allow_zero=allow_zero_repeats,
    )
    success_count = _validate_successes(
        successes,
        repeat_count_value,
        require_integer=require_integer_successes,
    )

    z = normal_critical_value(confidence_fraction)
    adjusted_repeats = repeat_count_value + z**2
    adjusted_successes = success_count + z**2 / 2.0
    estimate = adjusted_successes / adjusted_repeats
    half_width = z * math.sqrt(estimate * (1.0 - estimate) / adjusted_repeats)
    return ProportionInterval(
        estimate=estimate,
        lower=max(0.0, estimate - half_width),
        upper=min(1.0, estimate + half_width),
        half_width=half_width,
        successes=success_count,
        trials=repeat_count_value,
        confidence_level=confidence_fraction * 100.0,
        confidence_fraction=confidence_fraction,
        z_value=z,
        adjusted_successes=adjusted_successes,
        adjusted_trials=adjusted_repeats,
        raw_estimate=success_count / repeat_count_value if repeat_count_value else math.nan,
    )


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


def _scale_metric_interval(interval: MetricInterval, scale: float) -> MetricInterval:
    return MetricInterval(
        estimate=_scale_repeat_count_value(interval.estimate, scale),
        lower=_scale_repeat_count_value(interval.lower, scale),
        upper=_scale_repeat_count_value(interval.upper, scale),
        relative_error=interval.relative_error,
    )


def _coerce_probability_interval(
    success_probability: float | IntervalEstimate | ProportionInterval,
    probability_lower: float | None,
    probability_upper: float | None,
) -> IntervalEstimate | ProportionInterval:
    if isinstance(success_probability, (IntervalEstimate, ProportionInterval)):
        if probability_lower is not None or probability_upper is not None:
            raise ValueError("do not provide probability bounds with an interval object")
        interval = success_probability
    else:
        if probability_lower is None or probability_upper is None:
            raise ValueError("probability_lower and probability_upper are required")
        interval = IntervalEstimate(
            estimate=_validate_probability(success_probability, "success_probability"),
            lower=_validate_probability(probability_lower, "probability_lower"),
            upper=_validate_probability(probability_upper, "probability_upper"),
        )

    if interval.lower > interval.estimate or interval.upper < interval.estimate:
        raise ValueError("probability interval must satisfy lower <= estimate <= upper")
    return interval


def _coerce_repeats_argument(repeats: int | None, trials: int | None) -> int:
    if repeats is None and trials is None:
        raise ValueError("trials must be provided")
    if repeats is not None and trials is not None and repeats != trials:
        raise ValueError("repeats and trials must match when both are provided")
    return repeats if repeats is not None else trials


def _confidence_fraction_from_level(
    confidence_level: float,
    confidence_fraction: float | None,
) -> float:
    if confidence_fraction is not None:
        return _validate_open_probability(confidence_fraction, "confidence_fraction")

    try:
        level = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be between 0 and 100") from exc
    if level > 1.0:
        level = level / 100.0
    return _validate_open_probability(level, "confidence_level")


def _prefixed_dict(prefix: str, values: dict[str, float | int]) -> dict[str, float | int]:
    if not prefix:
        return values
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _validate_successes(
    successes: float,
    repeats: int,
    *,
    require_integer: bool,
) -> float:
    try:
        success_count = float(successes)
    except (TypeError, ValueError) as exc:
        raise ValueError("successes must be numeric") from exc
    if not math.isfinite(success_count) or success_count < 0 or success_count > repeats:
        raise ValueError("successes must be between 0 and repeats")
    if require_integer and not success_count.is_integer():
        raise ValueError("successes must be an integer count")
    return int(success_count) if require_integer else success_count


def _validate_repeats(value: int, name: str, *, allow_zero: bool) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        repeats = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(repeats) or not repeats.is_integer():
        raise ValueError(f"{name} must be an integer")
    if allow_zero:
        if repeats < 0:
            raise ValueError(f"{name} must be non-negative")
    elif repeats <= 0:
        raise ValueError(f"{name} must be positive")
    return int(repeats)


def _validate_probability(value: float, name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0 and 1") from exc
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return probability


def _validate_open_probability(value: float, name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between 0 and 1, exclusive") from exc
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError(f"{name} must be between 0 and 1, exclusive")
    return probability


def _validate_relative_error_threshold(value: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("relative_error_threshold must be between 0 and 1") from exc
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("relative_error_threshold must be between 0 and 1")
    return threshold
