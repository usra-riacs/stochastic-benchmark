# Passive Repeat-Reliability Reports

Passive repeat-reliability reports summarize existing solver observations. They do not run a solver or resample new data. Use them when each row already records a stochastic run, or when rows are pre-aggregated with a count column.

## Input Shapes

Run-level data should contain one row per observed run:

```python
from repeat_reliability import repeat_reliability_report

report = repeat_reliability_report(
    runs,
    group_cols=["solver", "budget", "instance"],
    response_col="energy",
    success_rule="min",
    threshold=-42.0,
    rtt_factor="runtime_seconds",
    iterations="num_reads",
    effort_per_iteration="sweeps_per_read",
)
```

Pre-aggregated data should use `count_col` for the number of identical observations represented by each row. Counts weight successes, trials, and resource-factor averages:

```python
report = repeat_reliability_report(
    summarized_runs,
    group_cols=["solver", "budget"],
    success_col="solved",
    count_col="n_runs",
    rtt_factor="runtime_seconds",
    iterations="num_reads",
    effort_per_iteration="sweeps_per_read",
)
```

`success_col` accepts booleans, `0`/`1`, and common boolean strings such as `"true"`, `"false"`, `"yes"`, `"no"`, `"1"`, and `"0"`. Null or ambiguous values are rejected so string values such as `"False"` are not accidentally treated as truthy.

## Success Rules

Use `success_col` when the input already contains a pass/fail indicator. Otherwise choose a rule and response columns:

- `success_rule="min"`: success means `response_col <= threshold`.
- `success_rule="max"`: success means `response_col >= threshold`.
- `success_rule="absolute"`: success means `abs(response_col - target_value) <= threshold`.
- `success_rule="gap"`, `"gap_min"`, or `"gap_max"`: success means the observed optimality gap is within `gap`.
- `success_rule="PerfRatio"`: success means an existing performance-ratio column, selected with `perf_ratio_col` or `response_col`, is greater than or equal to `threshold`.

All success rules are evaluated row by row before aggregation.

## Output Columns

The report preserves `group_cols` and any `comparison_cols`, followed by reliability metrics:

- `successes`, `trials`, and `success_rate`.
- `p_hat`: raw empirical success rate, `successes / trials`.
- `adjusted_p_hat`, `p_ci_lower`, and `p_ci_upper`: Agresti-Coull adjusted estimate and interval.
- `R_c`, `R_c_ci_lower`, and `R_c_ci_upper`: target-neutral repeat count for `target_confidence`.
- A target-specific repeat-count alias, such as `R99` for `target_confidence=0.99` or `R95` for `target_confidence=0.95`.
- `RTT` and `CETS` intervals. `CETS` uses `cets_factor`, the count-weighted per-row product of `iterations * effort_per_iteration`.
- Reliability status columns, including `required_trials`, `additional_trials_required`, `reliable`, and `reliability_status`.
- Comparison flags, when enabled: `best_point_estimate`, `ci_overlaps_best`, and `statistically_unresolved`.

Lower `R_c`, `RTT`, and `CETS` values are better. For probability-like comparison columns, call `annotate_reliability_comparisons` with `objective="max"`.

## Benchmark Wrapper

`stochastic_benchmark.run_RepeatReliability` forwards keyword arguments to `repeat_reliability_report`:

```python
report = benchmark.run_RepeatReliability(
    response_col="energy",
    success_rule="min",
    threshold=-42.0,
    rtt_factor="runtime_seconds",
    iterations="num_reads",
    effort_per_iteration="sweeps_per_read",
)
```

If `df` is omitted, the wrapper uses the first populated DataFrame among `raw_data`, `bs_results`, and `interp_results`. If `group_cols` is omitted, it groups by `parameter_names + instance_cols`.

Reports are flat DataFrames and can be joined to benchmark outputs on the group columns:

```python
joined = recommendations.merge(
    report,
    on=["solver", "budget", "instance"],
    how="left",
)
```
