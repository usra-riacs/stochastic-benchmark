# General Benchmark Workflow

Given a benchmark task, the workflow generally follows these steps:

1. Define the metrics
2. Perform bootstrapping
3. Perform interpolation
4. Train/Test split and Statistical Aggregation
5. Virtual Best Baseline
6. Evaluating Parameter Recommendation Strategies
7. Visualization

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
## Visualization

The framework's results can be visualized, mainly, through two lenses:

1. We can look at how our defined metric of interest compares against the defined resource. Here we can compare the Virtual Best, the Projection from the Training Set, and the Performance from the Training Set. Essentially, this tells us how close we can expect to be from the Virtual Best when new problem instances come in.
2. We can look at how we should distribute our input parameters for different resource amounts. Getting back to the previous **energy** example, if we have a given amount we are willing to spend, this can be achieved via different combinations of inputs. This analysis tells us what inputs to pick in order to achieve the performance metrics computed by the framework.