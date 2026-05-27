# General Benchmark Workflow

Given a benchmark task, the workflow generally follows these steps:

1. Define the metrics
2. Perform bootstrapping
3. Perform interpolation
4. Train/Test split and Statistical Aggregation
5. Virtual Best Baseline
6. Evaluating Parameter Recommendation Strategies
7. Repeat Reliability
8. Visualization

Each step is explored in greater detail along with code implementation at `examples/QAOA_iterative/qaoa_demo.ipynb`.
## Define the Metrics
In this first step, we configure the central object for the benchmark task. We define:

- The algorithm parameters being tuned
- Columns used to group problem instances
- The column containing the optimization objective
- Optimization direction (Minimization / Maximization)
## Bootstrapping
Suppose we have a distribution of data based on a numerical parameter. In the case of an optimization task, a more concrete example would be to have the execution performance of an algorithm based on input values. It is often useful to understand how this distribution would look if the numerical parameters were different.

Bootstrapping is a statistical process that allows us to build distributions of the **expected values** of the results we are interested in for different input values.

Example: For algorithm `A` with an input parameter `i`, we have a distribution of `n` performances `p`.
$$
P=[p1​,...,pn​]
$$
The goal of `A` is to minimize `p`. We are interested in knowing what this performance would look like, had the algorithm been run with input `j > i`. Although we are not able to know in a concrete way what the minimum performance would be in that case, bootstrapping allows us to infer a **distribution for the minimum performance**. From that, we get the expected value and confidence interval for the minimum performance given different inputs.
## Interpolation
When comparing different experimental runs, they often have data at different resource levels. This makes comparison difficult. Interpolation allows us to compute, for each run/method, a common comparison metric.

Example: Algorithm `A` and `B` might solve the same problem with different sets of parameters, which is a problem at first. But, the **total energy** spent by `A` could be determined by some combination of its input parameters, while the same is true for `B`. Interpolation allows us to compare them through the lens of this common metric.
## Train/Test split and Statistical Aggregation
One of the goals of the framework is to provide recommendations for the input parameters. For that, different data splits serve different purposes. The train set is used to learn the best parameter strategies, while the test set is used to verify them against unseen data.
## Virtual Best Baseline
When dealing with different problem instances, it is likely that the input parameters that work well for one instance are not optimal for another. Even if that's the case, it is useful to study how good of a performance we would get if it were possible to select the best input parameters for every unknown instance. This is what we are calling Virtual Best Baseline. It gives us an upper bound that generates insights into how good the current performance is, even though the Virtual Best Baseline itself is unachievable.
## Evaluating Parameter Recommendation Strategies
After having devised parameter recommendations from the training set, we need a way to evaluate how well these recommended strategies perform on new instances (test set). For this, the framework applies two different strategies:

1. We can aggregate statistics across all training instances to learn a parameter recipe. We refer to this as the Aggregate-then-Recommend projection strategy.
2. It is also possible to look at what parameters work best for each instance individually. After this, we can average those recommendations.

## Repeat Reliability

Noori et al. 2026 show that per-instance stochastic optimizer conclusions can
be unreliable when they are based on too few repeats:

Noori, Moslem, Elisabetta Valiante, Ignacio Rozada, Thomas Van Vaerenbergh, and
Masoud Mohseni. "Statistical analysis for per-instance evaluation of stochastic
optimizers: Avoiding unreliable conclusions." Physical Review Applied 25, no. 3
(2026): 034081.

This check is separate from the cross-instance Window Sticker uncertainty used
elsewhere in the workflow. Window Sticker plots summarize expected performance
over unseen instances from a problem family. Repeat reliability checks whether a
specific instance, solver configuration, and resource level has enough repeated
runs to estimate its success probability and the derived repeat counts.

Analytic repeat-reliability guarantees apply when each run can be treated as a
Bernoulli success event. In this package that covers:

- `R_c`, the repeats required to reach a target success confidence
- RTT/TTS values derived from `R_c` and runtime-per-repeat scaling
- CETS values derived from `R_c`, iterations, and effort-per-iteration scaling
- Thresholded continuous metrics, such as Response or PerfRatio after a
  success threshold converts each run into success or failure

Continuous Response curves and continuous PerfRatio curves remain
bootstrap-based unless they are converted to thresholded success events. Their
bootstrap intervals describe empirical cross-instance variation and resampling
uncertainty, not the analytic repeat-count guarantees from the paper.

Use `repeat_reliability_report` or `stochastic_benchmark.run_RepeatReliability`
before treating a benchmark conclusion as final. The resulting
`required_trials`, `additional_trials_required`, `reliable`, and
`reliability_status` columns tell users whether the existing data are reliable
enough or whether the solver should be rerun.

## Methodology Criticism Checklist

| Criticism | Workflow response |
| --- | --- |
| Repeat-count sufficiency | Check required and additional trials before trusting a per-instance success estimate. |
| Bootstrap-only uncertainty | Use analytic Bernoulli intervals for success events and keep bootstrap intervals for continuous cross-instance curves. |
| Noisy HPO choices | Treat parameter recommendations as provisional when repeat reliability is low or intervals are wide. |
| CI-overlap ambiguity | Mark overlapping comparisons as statistically unresolved instead of selecting a winner from point estimates alone. |
| Virtual-best optimism | Interpret virtual best as an unattainable optimistic reference, not as a deployable strategy. |

## Visualization

The framework's results can be visualized, mainly, through two lenses:

1. We can look at how our defined metric of interest compares against the defined resource. Here we can compare the Virtual Best, the Projection from the Training Set, and the Performance from the Training Set. Essentially, this tells us how close we can expect to be from the Virtual Best when new problem instances come in.
2. We can look at how we should distribute our input parameters for different resource amounts. Getting back to the previous **energy** example, if we have a given amount we are willing to spend, this can be achieved via different combinations of inputs. This analysis tells us what inputs to pick in order to achieve the performance metrics computed by the framework.
