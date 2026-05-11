from collections import defaultdict
import glob
import numpy as np
import os
import pandas as pd
from typing import Optional, Union, List
import warnings
import logging

import bootstrap
import df_utils
import interpolate
from plotting import *
import stats
import success_metrics
import training

logger = logging.getLogger(__name__)
import names

median = False

from experiments import ( 
    RandomSearchExperiment, 
    ProjectionExperiment, 
    StaticRecommendationExperiment, 
    SequentialSearchExperiment,
    VirtualBestBaseline,
    ExperimentParameters
)


def default_bootstrap(
    nboots=1000,
    response_col=names.param2filename({"Key": "MinEnergy"}, ""),
    resource_col=names.param2filename({"Key": "MeanTime"}, ""),
):
    """
    Default bootstrapping parameters

    Parameters
    ----------
    nboots : int, optional
        Number of bootstrap iterations, by default 1000
    response_col : str, optional
        Column name of response, by default names.param2filename({"Key": "MinEnergy"}, "")
    resource_col : str, optional
        Column name of resource, by default names.param2filename({"Key": "MeanTime"}, "")

    Returns
    -------
    bsparams_iter : bootstrap.BSParams_iter
        Iterator of bootstrap parameters
    """
    shared_args = {
        "response_col": response_col,
        "resource_col": resource_col,
        "response_dir": -1,
        "confidence_level": 68,
        "random_value": 0.0,
    }

    metric_args = defaultdict(dict)
    metric_args["Response"] = {"opt_sense": -1}
    metric_args["SuccessProb"] = {"gap": 1.0, "response_dir": -1}
    metric_args["RTT"] = {
        "fail_value": np.nan,
        "RTT_factor": 1.0,
        "gap": 1.0,
        "s": 0.99,
    }
    sms = [
        success_metrics.Response,
        success_metrics.PerfRatio,
        success_metrics.InvPerfRatio,
        success_metrics.SuccessProb,
        success_metrics.Resource,
        success_metrics.RTT,
    ]
    bsParams = bootstrap.BootstrapParameters(
        shared_args=shared_args,
        metric_args=metric_args,
        success_metrics=sms
    )

    bs_iter_class = bootstrap.BSParams_iter()
    bsparams_iter = bs_iter_class(bsParams, nboots)
    return bsparams_iter


def sweep_boots_resource(df):
    """
    Default resource computation - resource = sweeps * boots

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to compute resource for

    Returns
    -------
    pd.Series
        Series of resource values
    """
    return df["sweep"] * df["boots"]


class stochastic_benchmark:
    """
    Attributes
    ----------
    parameter_names : list[str]
        list of parameter names
    here : str
        path to parent directory of data
    instance_cols : list[str]
        Columns that define an instance. i.e., datapoints that match on all of these cols are the same instance
    bsParams_iter :
        Iterator that yields bootstrap parameters
    iParams :
        Interpolation parameters
    stat_params : stats.StatsParameters
        Parameters for computing stats dataframes
    resource_fcn : callable(pd.DataFrame)
        Function that writes a 'resource' function depending on dataframe parameters
    response_key : str
        Column that we want to optimize
    train_test_split : float
        Fraction of instances that should be training data
    recover : bool
        Whether dataframes should be recovered where available or generated from scratch
    reduce_mem : bool
        Whether to reduce memory usage by converting columns to appropriate types
    smooth : bool
        Whether to smooth the response values
    bs_results : pd.DataFrame
        Dataframe of bootstrap results
    interp_results : pd.DataFrame
        Dataframe of interpolation results
    training_stats : pd.DataFrame
        Dataframe of training stats
    testing_stats : pd.DataFrame
        Dataframe of testing stats
    experiments : list[Experiment]
        List of experiments

    Methods
    -------
    initAll(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Initialize all dataframes
    run_Bootstrap(bsParams_iter, group_name_fcn)
        Run bootstrap
    set_Bootstrap(bs_results)
        Set bootstrap results
    run_Interpolate(iParams)
        Run interpolation
    run_Stats(stat_params, train_test_split)
        Run stats
    populate_training_stats()
        Populate training stats
    populate_testing_stats()
        Populate testing stats
    populate_interp_results()
        Populate interpolation results
    evaluate_without_bootstrap(df, group_on)
        Evaluate a dataframe without bootstrap
    run_Baseline(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run baseline
    run_ProjectionExperiment(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run projection experiment
    run_RandomSearchExperiment(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run random search experiment
    run_SequentialSearchExperiment(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run sequential search experiment
    run_StaticRecommendationExperiment(bsParams_iter, iParams, stat_params, resource_fcn, train_test_split)
        Run static recommendation experiment
    init_Plotting()
        Initialize plotting
    """

    def __init__(
        self,
        parameter_names,
        here=os.getcwd(),
        instance_cols=["instance"],
        response_key="PerfRatio",
        response_dir=1,
        recover=True,
        reduce_mem=True,
        smooth=True,
    ):
        # Needed at initialization (for everything)
        self.here = names.paths(here)
        self.parameter_names = parameter_names
        self.instance_cols = instance_cols

        self.recover = recover
        self.reduce_mem = reduce_mem
        self.smooth = smooth

        self.response_key = response_key
        self.response_dir = response_dir

        ## Dataframes needed for experiments and baselines
        self.bs_results: Optional[Union[pd.DataFrame, List[str]]] = None
        self.interp_results: Optional[pd.DataFrame] = None
        self.training_stats: Optional[pd.DataFrame] = None
        self.testing_stats: Optional[pd.DataFrame] = None

        self.experiments = []

    def initAll(
        self,
        bsParams_iter=default_bootstrap(),
        iParams=None,
        stat_params=stats.StatsParameters(stats_measures=[stats.Median()]),
        resource_fcn=sweep_boots_resource,
        train_test_split=0.5,
        group_name_fcn=None,
    ):
        # Bootstrapping
        self.bsParams_iter = bsParams_iter
        self.group_name_fcn = group_name_fcn  # and stats

        # Interpolation
        self.resource_fcn = resource_fcn

        if iParams is None:
            self.iParams = interpolate.InterpolationParameters(
                self.resource_fcn, parameters=self.parameter_names
            )
        else:
            self.iParams = iParams

        # Stats
        self.stat_params = stat_params
        self.train_test_split = train_test_split

        # Recursive file recovery
        # Note: Bootstrap results are NOT auto-populated here. They must be explicitly
        # created via run_Bootstrap() method to give users control over when bootstrapping
        # occurs. This design was established in November 2022.
        while any(
            [
                v is None
                for v in [self.interp_results, self.training_stats, self.testing_stats]
            ]
        ):
            self.populate_training_stats()
            self.populate_testing_stats()
            self.populate_interp_results()

    def get_experiment_parameters(self) -> ExperimentParameters:
        return ExperimentParameters(
            parameter_names=self.parameter_names,
            instance_cols=self.instance_cols,
            interp_results=self.interp_results,
            checkpoint_path=self.here.checkpoints,
            response_key=self.response_key,
            response_dir=self.response_dir,
            smooth=self.smooth,
            stat_params=self.stat_params,
            training_stats=self.training_stats,
            testing_stats=self.testing_stats,
            evaluate_without_bootstrap=self.evaluate_without_bootstrap,
            baseline_recalibrate=self.baseline.recalibrate,
        )

    def run_Bootstrap(self, bsParams_iter, group_name_fcn=None):
        if self.bs_results is not None:
            logger.info("Bootstrapped results is already populated: doing nothing.")
            return

        logger.info("Loading and bootstrapping experimental data...")
        if self.reduce_mem:
            self.raw_data = glob.glob(os.path.join(self.here.raw_data, "*.pkl"))

            if len(self.raw_data) == 0:
                found_bs_results = glob.glob(
                    os.path.join(self.here.checkpoints, "bootstrapped_results*.pkl")
                )
                if len(found_bs_results) >= 1:
                    logger.info(
                        "Found %s bootstrapped results files and no raw data: reading results.",
                        len(found_bs_results),
                    )
                    self.bs_results = found_bs_results
                else:
                    raise Exception(
                        "No raw data found at: {} \n No bootstrapped results found at: {}".format(
                            self.here.raw_data, self.here.checkpoints
                        )
                    )
            else:
                if group_name_fcn is None:
                    raise Exception(
                        "group_name_fcn should be provided for reduced memory version."
                    )

                def raw2bs_names(raw_filename):
                    group_name = group_name_fcn(raw_filename)
                    bs_filename = os.path.join(
                        self.here.checkpoints,
                        "bootstrapped_results_{}.pkl".format(group_name),
                    )
                    return bs_filename

                bs_names = [raw2bs_names(raw_file) for raw_file in self.raw_data]

                if (
                    all([os.path.exists(bs_name) for bs_name in bs_names])
                    and len(bs_names) > 1
                    and self.recover
                ):
                    logger.info(
                        "All bootstrapped results are already found in checkpoints: reading results."
                    )
                    self.bs_results = bs_names
                    return

                group_on = self.parameter_names + self.instance_cols
                self.bs_results = bootstrap.Bootstrap_reduce_mem(
                    self.raw_data,
                    group_on,
                    bsParams_iter,
                    self.here.checkpoints,
                    group_name_fcn,
                )
        else:
            if os.path.exists(self.here.bootstrap) and self.recover:
                logger.info(
                    "All bootstrapped results are already found in checkpoints: reading results."
                )
                self.bs_results = pd.read_pickle(self.here.bootstrap)
                return

            logger.info("Running bootstrapped results")
            group_on = self.parameter_names + self.instance_cols
            if not hasattr(self, "raw_data"):
                if os.path.exists(self.here.raw_data) and glob.glob(
                    os.path.join(self.here.raw_data, "*.pkl")
                ):
                    self.raw_data = df_utils.read_exp_raw(self.here.raw_data)
                else:
                    if os.path.exists(self.here.bootstrap):
                        logger.info(
                            "Raw data missing but bootstrap pickle found: reading results."
                        )
                        self.bs_results = pd.read_pickle(self.here.bootstrap)
                        return
                    raise Exception(
                        "No raw data found at {} and no bootstrap pickle present".format(
                            self.here.raw_data
                        )
                    )

            progress_dir = os.path.join(self.here.progress, "bootstrap/")
            if not os.path.exists(progress_dir):
                os.makedirs(progress_dir)

            self.bs_results = bootstrap.Bootstrap(
                self.raw_data, group_on, bsParams_iter, progress_dir
            )
            if isinstance(self.bs_results, pd.DataFrame):
                self.bs_results.to_pickle(self.here.bootstrap)

    def set_Bootstrap(self, bs_results):
        """
        Sets bootstrap results without doing anything
        """
        if isinstance(bs_results, str):
            self.bs_results = pd.read_pickle(bs_results)
        elif isinstance(bs_results, pd.DataFrame):
            self.bs_results = bs_results
        elif isinstance(bs_results, list):
            if isinstance(bs_results[0], pd.DataFrame):
                self.bs_results = pd.concat(bs_results, ignore_index=True)
            elif isinstance(bs_results[0], str):
                self.bs_results = bs_results

    def run_Interpolate(self, iParams):
        if self.interp_results is not None:
            logger.info("Interpolated results is already populated: doing nothing.")
            return

        if os.path.exists(self.here.interpolate) and self.recover:
            logger.info("Interpolated results are found in checkpoints: reading results.")
            self.interp_results = pd.read_pickle(self.here.interpolate)
            return

        if self.bs_results is None:
            raise ValueError(
                "bs_results is None - bootstrapped results must be populated before interpolation. "
                "Ensure Bootstrap() or initBootstrap() has been called successfully."
            )

        logger.info("Interpolating results across resource budgets...")
        if self.reduce_mem:
            logger.info("Interpolating results with parameters: %s", iParams)
            if not isinstance(self.bs_results, list):
                raise TypeError(
                    f"Expected list for reduce_mem mode but got {type(self.bs_results).__name__}"
                )
            self.interp_results = interpolate.Interpolate_reduce_mem(
                self.bs_results, iParams, self.parameter_names + self.instance_cols
            )
        else:
            logger.info("Interpolating results with parameters: %s", iParams)
            if not isinstance(self.bs_results, pd.DataFrame):
                raise TypeError(
                    f"Expected DataFrame for non-reduce_mem mode but got {type(self.bs_results).__name__}"
                )
            self.interp_results = interpolate.Interpolate(
                self.bs_results, iParams, self.parameter_names + self.instance_cols
            )
        
        if self.interp_results is None:
            raise RuntimeError("Interpolation failed to produce results")

        base = names.param2filename({"Key": self.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.response_key, "ConfInt": "upper"}, ""
        )
        self.interp_results.dropna(subset=[base, CIlower, CIupper], inplace=True)

        # self.interp_results = training.split_train_test(self.interp_results, self.instance_cols, self.train_test_split)
        self.interp_results.to_pickle(self.here.interpolate)
        self.bs_results = None

    def run_Stats(self, stat_params, train_test_split=0.5):
        self.stat_params = stat_params
        if self.interp_results is None:
            raise Exception(
                "Interpolated results needs to be populated before computing stats."
            )

        logger.info("Computing training/testing statistics...")
        if "train" not in self.interp_results.columns:
            self.interp_results = training.split_train_test(
                self.interp_results, self.instance_cols, train_test_split
            )
            self.interp_results.to_pickle(self.here.interpolate)
        
        if self.interp_results is None:
            raise RuntimeError("interp_results was unexpectedly None")

        if self.training_stats is None:
            if os.path.exists(self.here.training_stats) and self.recover:
                logger.info("Training stats found in checkpoints: reading results.")
                self.training_stats = pd.read_pickle(self.here.training_stats)

            else:
                training_results = self.interp_results[
                    self.interp_results["train"] == 1
                ]
                logger.info("Computing training stats")
                self.training_stats = stats.Stats(
                    training_results,
                    stat_params,
                    self.parameter_names + ["boots", "resource"],
                )
        
        if self.training_stats is not None:
            self.training_stats.to_pickle(self.here.training_stats)

        if self.testing_stats is None:
            if os.path.exists(self.here.testing_stats) and self.recover:
                logger.info("Testing stats found in checkpoints: reading results.")
                self.testing_stats = pd.read_pickle(self.here.testing_stats)

            else:
                testing_results = self.interp_results[self.interp_results["train"] == 0]
                if len(testing_results) == 0:
                    warnings.warn("There are no testing sets")
                    # raise Exception('No instances assigned to test set. Reassign train/test split')

                else:
                    self.testing_stats = stats.Stats(
                        testing_results,
                        stat_params,
                        self.parameter_names + ["boots", "resource"],
                    )
        
        if self.testing_stats is not None:
            self.testing_stats.to_pickle(self.here.testing_stats)

    def populate_training_stats(self):
        """
        Tries to recover or computes training stats
        """
        if self.training_stats is None:
            if os.path.exists(self.here.training_stats) and self.recover:
                self.training_stats = pd.read_pickle(self.here.training_stats)
            elif self.interp_results is not None:
                training_results = self.interp_results[
                    self.interp_results["train"] == 1
                ]
                logger.info("Computing training stats")
                self.training_stats = stats.Stats(
                    training_results,
                    self.stat_params,
                    self.parameter_names + ["boots", "resource"],
                )
                if self.training_stats is not None:
                    self.training_stats.to_pickle(self.here.training_stats)

    def populate_testing_stats(self):
        """
        Tries to recover or computes testing stats
        """
        if self.testing_stats is None:
            if os.path.exists(self.here.testing_stats) and self.recover:
                self.testing_stats = pd.read_pickle(self.here.testing_stats)

            elif self.interp_results is not None:
                testing_results = self.interp_results[self.interp_results["train"] == 0]
                logger.info("Computing testing stats")
                if len(testing_results) == 0:
                    self.testing_stats = pd.DataFrame()

                else:
                    self.testing_stats = stats.Stats(
                        testing_results,
                        self.stat_params,
                        self.parameter_names + ["boots", "resource"],
                    )
                    if self.testing_stats is not None:
                        self.testing_stats.to_pickle(self.here.testing_stats)

    def populate_interp_results(self):
        """
        Tries to recover or computes interpolated results
        """
        if self.interp_results is None:
            if os.path.exists(self.here.interpolate) and self.recover:
                self.interp_results = pd.read_pickle(self.here.interpolate)
                if self.interp_results is not None and "train" not in self.interp_results.columns:
                    self.interp_results = training.split_train_test(
                        self.interp_results, self.instance_cols, self.train_test_split
                    )
                    self.interp_results.to_pickle(self.here.interpolate)

            elif self.bs_results is not None:
                # print(self.bs_results)
                if self.reduce_mem:
                    logger.info("Interpolating results with parameters: %s", self.iParams)
                    if not isinstance(self.bs_results, list):
                        raise TypeError(
                            f"Expected list for reduce_mem mode but got {type(self.bs_results).__name__}"
                        )
                    self.interp_results = interpolate.Interpolate_reduce_mem(
                        self.bs_results,
                        self.iParams,
                        self.parameter_names + self.instance_cols,
                    )
                else:
                    logger.info("Interpolating results with parameters: %s", self.iParams)
                    if not isinstance(self.bs_results, pd.DataFrame):
                        raise TypeError(
                            f"Expected DataFrame for non-reduce_mem mode but got {type(self.bs_results).__name__}"
                        )
                    self.interp_results = interpolate.Interpolate(
                        self.bs_results,
                        self.iParams,
                        self.parameter_names + self.instance_cols,
                    )
                
                if self.interp_results is None:
                    raise RuntimeError("Interpolation failed to produce results")

                base = names.param2filename({"Key": self.response_key}, "")
                CIlower = names.param2filename(
                    {"Key": self.response_key, "ConfInt": "lower"}, ""
                )
                CIupper = names.param2filename(
                    {"Key": self.response_key, "ConfInt": "upper"}, ""
                )
                self.interp_results.dropna(
                    subset=[base, CIlower, CIupper], inplace=True
                )

                self.interp_results = training.split_train_test(
                    self.interp_results, self.instance_cols, self.train_test_split
                )
                self.interp_results.to_pickle(self.here.interpolate)
                self.bs_results = None
            # Note: If neither interp_results nor bs_results exist, they must be explicitly
            # created via run_Bootstrap(). Auto-population was removed in commit 857574f (Nov 2022)
            # to give users explicit control. For reference implementation of auto-population
            # with reduce_mem support, see git history.

    def evaluate_without_bootstrap(self, df, group_on):
        """ "
        Runs same computations evaluations as bootstrap without bootstrapping

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe of results
        group_on : list[str]
            Columns to group on

        Returns
        -------
        pd.DataFrame
            Dataframe of results
        """
        bs_params = next(self.bsParams_iter)
        resource_col = bs_params.shared_args["resource_col"]
        response_col = bs_params.shared_args["response_col"]
        agg = bs_params.agg

        def evaluate_single(df_single):
            bs_params.update_rule(bs_params, df_single)
            resources = df_single[resource_col].values
            responses = df_single[response_col].values
            resources = np.repeat(resources, df_single[agg])
            responses = np.repeat(responses, df_single[agg])

            bs_df = pd.DataFrame()
            for metric_ref in bs_params.success_metrics:
                metric = metric_ref(
                    bs_params.shared_args, bs_params.metric_args[metric_ref.__name__]
                )
                metric.evaluate(bs_df, responses, resources)
            for col in bs_params.keep_cols:
                if col in df_single.columns:
                    val = df_single[col].iloc[0]
                    bs_df[col] = val

            return bs_df

        full_eval = (
            df.groupby(group_on).apply(lambda df: evaluate_single(df), include_groups=False).reset_index()
        )
        full_eval.drop(columns=["level_{}".format(len(group_on))], inplace=True)
        return full_eval

    def run_baseline(self):
        """
        Adds virtual best baseline
        """
        logger.info("Computing virtual best baseline...")
        self.baseline = VirtualBestBaseline(self.get_experiment_parameters())

    def run_ProjectionExperiment(
        self, project_from, postprocess=None, postprocess_name=None
    ):
        """
        Runs projections experiments

        Parameters
        ----------
        project_from : str
            Name of experiment to project from
        postprocess : callable(pd.DataFrame)
            Function to postprocess results
        postprocess_name : str
            Name of postprocess

        Returns
        -------
        ProjectionExperiment
            Experiment object
        """
        logger.info("Running ProjectionExperiment from %s...", project_from)
        self.experiments.append(
            ProjectionExperiment(self.get_experiment_parameters(), project_from, postprocess, postprocess_name)
        )

    def run_RandomSearchExperiment(
        self, rsParams, postprocess=None, postprocess_name=None
    ):
        """
        Runs random search experiments
        """
        logger.info("Running RandomSearchExperiment...")
        self.experiments.append(
            RandomSearchExperiment(
                self.get_experiment_parameters(),
                rsParams,
                postprocess=postprocess,
                postprocess_name=postprocess_name,
            )
        )

    def run_SequentialSearchExperiment(
        self, ssParams, id_name=None, postprocess=None, postprocess_name=None
    ):
        """
        Runs sequential search experiments

        Parameters
        ----------
        ssParams : SequentialSearchParameters
            Parameters for sequential search
        id_name : str
            Name of experiment
        postprocess : callable(pd.DataFrame)
            Function to postprocess results
        postprocess_name : str
            Name of postprocess

        Returns
        -------
        SequentialSearchExperiment
            Experiment object
        """
        id_label = f" ({id_name})" if id_name else ""
        logger.info("Running SequentialSearchExperiment%s...", id_label)
        self.experiments.append(
            SequentialSearchExperiment(
                self.get_experiment_parameters(),
                ssParams,
                id_name,
                postprocess=postprocess,
                postprocess_name=postprocess_name,
            )
        )

    def run_StaticRecommendationExperiment(self, init_from):
        """
        Runs static recommendation experiments

        Parameters
        ----------
        init_from : str
            Name of experiment to initialize from

        Returns
        -------
        StaticRecommendationExperiment
            Experiment object
        """
        logger.info("Running static recommendation experiment")
        self.experiments.append(StaticRecommendationExperiment(self.get_experiment_parameters(), init_from))

    def initPlotting(self):
        """
        Sets up plotting - this should be run after all experiments are run
        """
        self.plots = Plotting(self)
