# Noori et al. Repeat-Reliability Validation

This note defines how this repository validates the Noori et al. repeat-reliability
formulas without an author reference implementation, executable code, or raw
numerical data. The formulas are implemented as deterministic reference helpers
in `src/repeat_reliability.py`; downstream production integrations should cite
the same equation labels and reuse these tests when behavior is changed.

Paper reference:

Noori, Moslem, Elisabetta Valiante, Ignacio Rozada, Thomas Van Vaerenbergh, and
Masoud Mohseni. "Statistical analysis for per-instance evaluation of stochastic
optimizers: Avoiding unreliable conclusions." Physical Review Applied 25, no. 3
(2026): 034081.

## Equation Map

| Paper item | Repository function | Validation coverage |
| --- | --- | --- |
| Eq. (8), Agresti-Coull confidence interval for success probability | `agresti_coull_interval`, `agresti_coull_interval_from_estimate` | Deterministic fixture grid in `tests/test_repeat_reliability.py` for `p_hat in {0, 0.01, 0.1, 0.5, 0.9, 0.99, 1}` and `n in {100, 1000, 10000}`. |
| Eq. (9), confidence-interval width and relative width | `ProportionInterval.width`, `ProportionInterval.relative_width` | Covered by the same fixture grid because lower, upper, and adjusted estimate are asserted numerically. |
| Eq. (10), worst-case repeats needed for an absolute success-probability error | `required_repeats_for_probability_error` | Exact numeric checks for error tolerances `0.01` and `0.03` at 95 percent confidence. |
| Eqs. (1)-(2), `R_c` from success probability | `repeat_count` | Deterministic fixture grid checks the induced `R_c` point estimate from each Agresti-Coull estimate. |
| Confidence interval of `R_c` induced by the success-probability interval | `repeat_count_interval` | Deterministic fixture grid checks lower and upper `R_c` bounds, including boundary cases with infinite upper bounds. |
| Eq. (3), CETS scaling from `R_c` | `cets_from_repeat_count`, `scaled_repeat_count_interval` | Unit test verifies deterministic scaling preserves relative error. |
| RTT/time-to-solution scaling from `R_c` | `rtt_from_repeat_count`, `scaled_repeat_count_interval` | Unit test verifies deterministic scaling preserves relative error. |
| Eq. (13), maximum relative error in `R_c` | `maximum_relative_error` | Deterministic fixture grid checks exact relative-error values for finite intervals. |
| Eq. (19), lower-bound repeats for target `R_c` relative error | `required_repeats_lower_bound` | Numeric checks and qualitative fixed-seed Bernoulli tests cover monotonic behavior for low vs. moderate success probabilities. |
| Exact numerical repeats for target `R_c` relative error | `required_repeats_exact` | Binary-search numerical checks assert stable required-repeat values and compare them with lower-bound behavior. |

## Validation Strategy

The validation intentionally uses two classes of checks.

Exact deterministic checks:

- Use fixed `p_hat` and `n` fixture inputs.
- Assert numerical Agresti-Coull estimates and intervals.
- Assert induced `R_c` intervals and maximum relative errors.
- Assert lower-bound and exact repeat-count outputs for selected parameters.
- Do not compare against paper plots or author data that are not available.

Qualitative fixed-seed Bernoulli checks:

- Show that an `R_c` interval induced by a success-probability interval is
  asymmetric.
- Show that low estimated success probability requires more repeats than a
  moderate estimated success probability under the Eq. (19) lower bound.
- Show that noisy small-sample estimates can invert optimizer rankings even
  when the first optimizer has the larger true success probability.

These tests use fixed RNG seeds and only assert qualitative claims from the
paper. They are not treated as reproduction of the paper's Monte Carlo tables or
figures.

## Boundary Policy

The implementation clips Agresti-Coull probability intervals to `[0, 1]`, as
discussed in the paper for boundary cases. `R_c(0)` is infinite and `R_c(p)` is
clamped to `1` when `p >= c`, matching Eq. (1)'s `max(..., 1)` form.

The paper source text contains an apparent mismatch around the `epsilon_p = 0.03`
repeat-count example. Eq. (10) with a two-sided 95 percent normal critical value
gives `n = 1064`, and one later sentence in the source also refers to `1064`;
the tests follow the equation rather than the inconsistent prose value.
