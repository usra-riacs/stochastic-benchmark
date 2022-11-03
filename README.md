# Stochastic Benchmark - Windows Sticker

Repository for Stochastic Optimization Solvers Benchmark code

This code has been created in order to produce a set of plots that inform the performance of parameterized stochastic optimization solvers when addressing a well-established family of optimization problems.
These plots are produced based on experimental data from the execution of such solvers in seen instances of the problem family and evaluated further in an unseen subset of problems.
More details of the methodology have been presented in the [APS March meeting](https://meetings.aps.org/Meeting/MAR22/Session/F38.5) and [INFORMS Annual meeting](https://www.abstractsonline.com/pp8/#!/10693/presentation/8455) conferences.
A manuscript explaining the methodology is in preparation.
The performance plot, or as we like to call it *Windows Sticker*, is a graphical representation of the expected performance of a solution method or parameter setting strategy with an unseen instance from the same problem family that it is generated aiming to answer the question With X% confidence, will we find a solution with Y quality after using R resource?
Consider that the quality metric and the resource values can be arbitrary functions of the parameters and performance of the given solver, providing a flexible analysis tool for its performance.

The current package implements the following functionality:
- Parsing results from files from parameterized stochastic solvers such as PySA and D-Wave ocean tools.
- Through bootstrapping and downsampling, simulate the lower data performance for such solvers.
- Compute best-recommended parameters based on aggregated statistics and individual results for each parameter setting.
- Compute optimistic bound performance, known as virtual best performance, based on the provided experiments.
- Perform an exploration-exploitation parameter setting strategy, where the fraction of the allocated resources used in the exploration round is optimized. The exploration procedure is implemented as a random search in the seen parameter settings or a Bayesian-based method known as the tree of parzen and implemented in the package [Hyperopt](https://hyperopt.github.io/hyperopt/).
- Plot the Windows sticker, comparing the performance curves corresponding to the virtual best, recommended parameters, and exploration-exploitation parameter setting strategies.
- Plots the values of the parameters and their best values with respect to the resource considered, a plot we call the Strategy plot. These plots can show the actual solver parameter values or the meta-parameters associated with parameter-setting strategies.

<!-- the following `pip` command can install this package -->

<!-- ``pip install -i https://test.pypi.org/simple/ stochastic-benchmark==0.0.1`` -->
