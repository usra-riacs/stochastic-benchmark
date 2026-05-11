from collections import namedtuple
import os
import pandas as pd
from typing import List, Callable
import warnings
import logging
from dataclasses import dataclass

import df_utils
from plotting import *
import random_exploration
import sequential_exploration
import training
import stats

logger = logging.getLogger(__name__)
import names

median = False

@dataclass(frozen=True)
class ExperimentParameters:
    parameter_names: List[str]
    instance_cols: List[str]
    interp_results: pd.DataFrame
    checkpoint_path: str
    response_key: str
    response_dir: int
    smooth: bool
    stat_params: stats.StatsParameters
    training_stats: pd.DataFrame
    testing_stats: pd.DataFrame
    evaluate_without_bootstrap: Callable[[pd.DataFrame, List[str]], pd.DataFrame]
    baseline_recalibrate: Callable[[pd.DataFrame], None]


class Experiment:
    """
    Base class for experiments

    Attributes
    ----------
    parent_params: ExperimentParameters
        Parent stochastic_benchmark instance. This is a forward reference to the
        stochastic_benchmark class defined later in this module.
    name : str
        Name of experiment

    Methods
    -------
    __init__(parent_params, name)
        Initializes experiment
    evaluate()
        Evaluates experiment
    evaluate_monotone()
        Monotonizes the response and parameters from evaluate
    """
    
    parent_params: ExperimentParameters

    def evaluate(self):
        raise NotImplementedError(
            "Evaluate should be overridden by a subclass of Experiment"
        )

    def evaluate_monotone(self):
        """
        Monotonizes the response and parameters from evaluate

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        res = self.evaluate()
        params_df = None
        eval_df = None
        preproc_params = None
        
        if len(res) == 2:
            params_df, eval_df = res
        elif len(res) == 3:
            params_df, eval_df, preproc_params = res
        
        if params_df is None or eval_df is None:
            raise ValueError("evaluate() returned invalid result")
        
        joint = params_df.merge(eval_df, on="resource")
        joint = df_utils.monotone_df(joint, "resource", "response", 1)
        
        params_df = joint.loc[:, ["resource"] + self.parent_params.parameter_names]
        eval_df = joint.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]

        if len(res) == 3 and preproc_params is not None:
            return params_df, eval_df, preproc_params
        else:
            return params_df, eval_df


class ProjectionExperiment(Experiment):
    """
    Holds information needed for projection experiments.
    Used for evaluating performance of a recipe on the test set if the user cannot re-run experiments.
    Recipes can be post-processed by a user-defined function (e.g., smoothed fit) and queried for running
    evaluatation experiments.

    Attributes
    ----------
    parent_params: ExperimentParameters
        Parent experiment parameters
    name : str
        name for pretty printing
    project_from : str
        'TrainingStats' or 'TrainingResults'
    recipe : pd.DataFrame
        Recommended parameters for each resource (can be postprocessed). This is not projected
    rec_params : pd.DataFrame
        Projected recommended parameters for each resource
    rec_path : str
        Path to recipe
    postprocess : function
        Function to postprocess recipe
    postprocess_name : str
        Name of postprocessing function

    Methods
    -------
    __init__(parent_params, project_from, postprocess=None, postprocess_name=None)
        Initializes projection experiment
    set_rec_path()
        Sets rec_path
    get_TrainingStats_recipe()
        Gets recipe from TrainingStats
    get_TrainingResults_recipe()
        Gets recipe from TrainingResults
    evaluate()
        Evaluates experiment
    """

    def __init__(self, parent_params, project_from, postprocess=None, postprocess_name=None):
        self.parent_params = parent_params
        self.name = "Projection from {}".format(project_from)
        self.project_from = project_from
        self.postprocess = postprocess
        self.postprocess_name = postprocess_name
        self.populate()

    def populate(self):
        """
        Adds recipe depending on source. Currently only projection from the best recommended from the training stats or results are available.
        Any addition recipe specifications should be implemented here
        """
        # Set rec_path, i.e. the path where the recipe is/will be stored
        self.set_rec_path()

        # Prepare the recipes
        if self.project_from == "TrainingStats":
            self.get_TrainingStats_recipe()
        elif self.project_from == "TrainingResults":
            self.get_TrainingResults_recipe()
        else:
            raise NotImplementedError(
                "Projection from {} has not been implemented".format(self.project_from)
            )

        # Run the projections
        if os.path.exists(self.rec_path):
            self.rec_params = pd.read_pickle(self.rec_path)
        else:
            logger.info("Evaluating recommended parameters on testing results")
            testing_results = self.parent_params.interp_results[
                self.parent_params.interp_results["train"] == 0
            ].copy()
            self.rec_params = training.evaluate(
                testing_results,
                self.recipe,
                training.scaled_distance,
                parameter_names=self.parent_params.parameter_names,
                group_on=self.parent_params.instance_cols,
            )
            self.rec_params.to_pickle(self.rec_path)

    def get_TrainingResults_recipe(self):
        """
        If TrainingResults recipe is already stored in a pkl file, load it. Otherwise, create and store it by obtaining the best parameters from training_stats (and post_processing, if requested)
        """
        vb_train_path = os.path.join(
            self.parent_params.checkpoint_path, "VirtualBest_train.pkl"
        )

        if os.path.exists(vb_train_path):
            self.vb_train = pd.read_pickle(vb_train_path)
        else:
            response_col = names.param2filename({"Key": self.parent_params.response_key}, "")
            training_results = self.parent_params.interp_results[
                self.parent_params.interp_results["train"] == 1
            ].copy()
            self.vb_train = training.virtual_best(
                training_results,
                parameter_names=self.parent_params.parameter_names,
                response_col=response_col,
                response_dir=1,
                groupby=self.parent_params.instance_cols,
                resource_col="resource",
                smooth=self.parent_params.smooth,
            )
            self.vb_train.to_pickle(vb_train_path)

        self.recipe = training.best_recommended(
            self.vb_train.copy(),
            parameter_names=self.parent_params.parameter_names,
            resource_col="resource",
            additional_cols=["boots"],
        ).reset_index()

        if self.postprocess is not None:
            self.preproc_recipe = self.recipe.copy()
            self.recipe = self.postprocess(self.recipe)

    def get_TrainingStats_recipe(self):
        """
        If TrainingStats recipe is already stored in a pkl file, load it. Otherwise, create and store it by obtaining the best parameters from training_stats (and post_processing, if requested)
        """

        best_rec_train_path = os.path.join(
            self.parent_params.checkpoint_path, "BestRecommended_train.pkl"
        )
        if os.path.exists(best_rec_train_path):
            # If the recipe was already stored in a pkl file, simply load it
            self.recipe = pd.read_pickle(best_rec_train_path)
        else:
            # If not, create the recipe dataframe, and store it in a pkl file
            # Get the name of the response column
            response_col = names.param2filename(
                {
                    "Key": self.parent_params.response_key,
                    "Metric": self.parent_params.stat_params.stats_measures[0].name,
                },
                "",
            )

            # Obtain the recipe, before the postprocessing step
            self.recipe = training.best_parameters(
                self.parent_params.training_stats.copy(),
                parameter_names=self.parent_params.parameter_names,
                response_col=response_col,
                response_dir=1,
                resource_col="resource",
                additional_cols=["boots"],
                smooth=self.parent_params.smooth,
            )

            self.recipe.to_pickle(best_rec_train_path)

        if self.postprocess is not None:
            best_rec_train_path_post = os.path.join(
                self.parent_params.checkpoint_path,
                "BestRecommended_train_postprocess={}.pkl".format(
                    self.postprocess_name
                ),
            )
            # Copy the recipe to preproc_recipe before postprocessing
            self.preproc_recipe = self.recipe.copy()
            # Implement post-processing
            self.recipe = self.postprocess(self.recipe)
            self.recipe.to_pickle(best_rec_train_path_post)

    def evaluate(self, monotone=False):
        """
        Evaluates the recommended parameters on the testing results, and returns the recommended parameters and the responses

        Parameters
        ----------
        monotone : bool, optional
            If True, the recommended parameters are evaluated on the monotone testing results, by default False
        
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        # params_df = self.rec_params.loc[:, ['resource'] + self.parent.parameter_names].copy()

        # base = names.param2filename({'Key': self.parent_params.response_key}, '')
        # CIlower = names.param2filename({'Key': self.parent_params.response_key,
        #                                 'ConfInt':'lower'}, '')
        # CIupper = names.param2filename({'Key': self.parent_params.response_key,
        #                                 'ConfInt':'upper'}, '')
        # eval_df = self.rec_params.copy()
        # eval_df.rename(columns = {
        #     base :'response',
        #     CIlower :'response_lower',
        #     CIupper :'response_upper',
        # }, inplace=True
        # )

        # joint = self.rec_params.copy()
        # base = names.param2filename({'Key': self.parent_params.response_key}, '')
        # CIlower = names.param2filename({'Key': self.parent_params.response_key,
        #                                 'ConfInt':'lower'}, '')
        # CIupper = names.param2filename({'Key': self.parent_params.response_key,
        #                                 'ConfInt':'upper'}, '')
        # joint.rename(columns = {
        #     base :'response',
        #     CIlower :'response_lower',
        #     CIupper :'response_upper',
        # }, inplace=True
        # )
        # joint = joint.loc[:, ['resource'] + self.parent.parameter_names +
        # ['response', 'response_lower', 'response_upper'] + self.parent.instance_cols]

        # extrapolate_from = self.parent.interp_results.loc[self.parent.interp_results['train'] == 0].copy()
        # extrapolate_from.rename(columns = {
        #     base :'response',
        #     CIlower :'response_lower',
        #     CIupper :'response_upper',
        # }, inplace=True
        # )

        # def mono(df):
        #     # res = df_utils.monotone_df(joint, 'resource', 'response', 1,
        #     # extrapolate_from=extrapolate_from, match_on = self.parent.parameter_names + self.parent.instance_cols)
        #     res = df_utils.monotone_df(joint, 'resource', 'response', 1)
        #     return res

        # joint = joint.groupby(self.parent.instance_cols, include_groups=False).apply(mono)

        # params_df = joint.loc[:, ['resource'] + self.parent.parameter_names]
        # eval_df = joint.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        # params_df = params_df.groupby('resource').mean()
        # params_df.reset_index(inplace=True)

        # eval_df = eval_df.groupby('resource').median()
        # eval_df = eval_df.groupby('resource').median()
        # eval_df.reset_index(inplace=True)

        params_df = self.rec_params.loc[
            :, ["resource"] + self.parent_params.parameter_names
        ].copy()
        params_df = params_df.groupby("resource").mean()
        params_df.reset_index(inplace=True)

        base = names.param2filename({"Key": self.parent_params.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.parent_params.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.parent_params.response_key, "ConfInt": "upper"}, ""
        )
        eval_df = self.rec_params.copy()
        eval_df.rename(
            columns={
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )
        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        if median:
            eval_df = eval_df.groupby("resource").median()
        else:
            eval_df = eval_df.groupby("resource").mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df

    def evaluate_monotone(self):
        """
        Monotonizes the response and parameters from evaluate

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df, eval_df = self.evaluate()

        joint = params_df.merge(eval_df, on="resource")
        extrapolate_from = self.parent_params.testing_stats.copy()
        base = names.param2filename(
            {
                "Key": self.parent_params.response_key,
                "Metric": self.parent_params.stat_params.stats_measures[0].name,
            },
            "",
        )
        CIlower = names.param2filename(
            {
                "Key": self.parent_params.response_key,
                "ConfInt": "lower",
                "Metric": self.parent_params.stat_params.stats_measures[0].name,
            },
            "",
        )
        CIupper = names.param2filename(
            {
                "Key": self.parent_params.response_key,
                "ConfInt": "upper",
                "Metric": self.parent_params.stat_params.stats_measures[0].name,
            },
            "",
        )
        extrapolate_from.rename(
            columns={
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )

        # joint = df_utils.monotone_df(joint, 'resource', 'response', 1,
        #     extrapolate_from=extrapolate_from, match_on = self.parent.parameter_names)
        joint = df_utils.monotone_df(joint, "resource", "response", 1)
        params_df = joint.loc[:, ["resource"] + self.parent_params.parameter_names]
        eval_df = joint.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        return params_df, eval_df

    def set_rec_path(self):
        """
        Define the path where the recipe is to be stored
        """
        if self.postprocess is not None:
            self.rec_path = os.path.join(
                self.parent_params.checkpoint_path,
                "Projection_from={}_postprocess={}.pkl".format(
                    self.project_from, self.postprocess_name
                ),
            )
        else:
            self.rec_path = os.path.join(
                self.parent_params.checkpoint_path,
                "Projection_from={}.pkl".format(self.project_from),
            )


class StaticRecommendationExperiment(Experiment):
    """
    Holds parameters for fixed recommendation experiments

    Attributes
    ----------
    parent_params: ExperimentParameters
        Parent experiment parameters
    name : str
        name for pretty printing
    rec_params : pd.DataFrame
        Recommended parameters for evaluation
    preproc_rec_params : pd.DataFrame
        Recommended parameters before processing

    Methods
    -------
    __init__(parent_params, init_from)
        Initialize the class
    list_runs()
        Returns a list of experiments evaluate
    evaluate()
        Returns the recommended parameters and responses
    evaluate_monotone()
        Monotonizes the response and parameters from evaluate
    set_rec_path()
        Define the path where the recipe is to be stored
    """

    def __init__(self, parent_params, init_from):
        self.parent_params = parent_params
        self.name = "FixedRecommendation"

        if type(init_from) == ProjectionExperiment:
            self.rec_params = init_from.recipe
            if init_from.postprocess is not None:
                self.preproc_rec_params = init_from.preproc_recipe.copy()
            else:
                self.preproc_rec_params = init_from.recipe.copy()

        elif type(init_from) == pd.DataFrame:
            self.rec_params = init_from
            self.preproc_rec_params = init_from.copy()
        else:
            warn_str = (
                "init_from type is not supported. No recommended parameters are set."
            )
            warnings.warn(warn_str)

    def list_runs(self):
        """
        Returns a list of experiments evaluate.

        Returns
        -------
        runs : list
            List of named tuples of parameters
        """
        parameter_names = "resource " + " ".join(self.parent_params.parameter_names)
        Parameter = namedtuple("Parameter", parameter_names)
        runs = []
        for _, row in self.rec_params.iterrows():
            runs.append(
                Parameter(
                    row["resource"], *[row[k] for k in self.parent_params.parameter_names]
                )
            )
        return runs

    def attach_runs(self, df, process=True):
        """
        Attaches reruns of experiment to the experiment object

        Parameters
        ----------
        df : pd.DataFrame or str
            Dataframe of responses
        process : bool
            Whether to process the dataframe

        Returns
        -------
        None
        """
        if type(df) == str:
            df = pd.read_pickle(df)
        if process:
            group_on = self.parent_params.instance_cols + ["resource"]
            self.eval_df = self.parent_params.evaluate_without_bootstrap(df, group_on)
        else:
            self.eval_df = df
        self.parent_params.baseline_recalibrate(self.eval_df)

    def evaluate(self):
        """
        Evaluates the recommended parameters

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df = self.rec_params.loc[
            :, ["resource"] + self.parent_params.parameter_names
        ].copy()
        preproc_params = self.preproc_rec_params.loc[
            :, ["resource"] + self.parent_params.parameter_names
        ].copy()
        # params_df = params_df.groupby('resource').mean()
        # params_df.reset_index(inplace=True)

        base = names.param2filename({"Key": self.parent_params.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.parent_params.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.parent_params.response_key, "ConfInt": "upper"}, ""
        )
        eval_df = self.eval_df.copy()
        eval_df.rename(
            columns={
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )
        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        eval_df = eval_df.groupby("resource").mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df, preproc_params


class RandomSearchExperiment(Experiment):
    """
    Holds parameters needed for random search experiment

    Attributes
    ----------
    parent_params: ExperimentParameters
        Parent experiment parameters
    name : str
        name for pretty printing
    meta_params : pd.DataFrame
        Best metaparameters (Exploration budget and Tau)
    eval_train : pd.DataFrame
        Resulting parameters of meta_params on training set
    eval_test : pd.DataFrame
        Resulting parameters of meta_params on testing set
    rsParams : dict
        Dictionary of parameters for random search
    postprocess : function
        Function to postprocess the results
    postprocess_name : str
        Name of the postprocessing function

    Methods
    -------
    __init__(parent, rsParams, postprocess=None, postprocess_name=None)
        Initialize the class
    populate()
        Populates meta_params, eval_train, eval_test
    """

    def __init__(self, parent_params, rsParams, postprocess=None, postprocess_name=None):
        self.parent_params = parent_params
        self.name = "RandomSearch"
        self.rsParams = rsParams
        self.meta_parameter_names = ["ExploreFrac", "tau"]
        self.resource = "TotalBudget"
        self.postprocess = postprocess
        self.postprocess_name = postprocess_name
        self.populate()

    def populate(self):
        """
        Populates meta_params, eval_train, eval_test
        """
        meta_params_path = os.path.join(
            self.parent_params.checkpoint_path, "RandomSearch_meta_params.pkl"
        )
        eval_train_path = os.path.join(
            self.parent_params.checkpoint_path, "RandomSearch_evalTrain.pkl"
        )

        if self.postprocess is None:
            eval_test_path = os.path.join(
                self.parent_params.checkpoint_path, "RandomSearch_evalTest.pkl"
            )
        else:
            eval_test_path = os.path.join(
                self.parent_params.checkpoint_path,
                "RandomSearch_evalTest_postprocess={}.pkl".format(
                    self.postprocess_name
                ),
            )

        if os.path.exists(meta_params_path):
            self.meta_params = pd.read_pickle(meta_params_path)
        else:
            self.meta_params, self.eval_train, _ = random_exploration.RandomExploration(
                self.parent_params.training_stats, self.rsParams
            )
            self.meta_params.to_pickle(meta_params_path)
            self.eval_train.to_pickle(eval_train_path)
        self.meta_params["ExploreFrac"] = (
            self.meta_params["ExplorationBudget"] / self.meta_params["TotalBudget"]
        )

        if self.postprocess is not None:
            self.preproc_meta_params = self.meta_params.copy()
            self.meta_params = self.postprocess(self.meta_params)

        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
            logger.info("\t Evaluating random search on test")
            self.eval_test = random_exploration.apply_allocations(
                self.parent_params.testing_stats.copy(), self.rsParams, self.meta_params
            )
            self.eval_test.to_pickle(eval_test_path)

    def evaluate(self):
        """
        Evaluates the random search

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df = self.eval_test.loc[:, ["TotalBudget"] + self.parent_params.parameter_names]
        params_df = params_df.groupby("TotalBudget").mean()
        params_df.reset_index(inplace=True)
        params_df.rename(columns={"TotalBudget": "resource"}, inplace=True)

        base = names.param2filename(
            {
                "Key": self.parent_params.response_key,
                "Metric": self.parent_params.stat_params.stats_measures[0].name,
            },
            "",
        )
        CIlower = names.param2filename(
            {
                "Key": self.parent_params.response_key,
                "ConfInt": "lower",
                "Metric": self.parent_params.stat_params.stats_measures[0].name,
            },
            "",
        )
        CIupper = names.param2filename(
            {
                "Key": self.parent_params.response_key,
                "ConfInt": "upper",
                "Metric": self.parent_params.stat_params.stats_measures[0].name,
            },
            "",
        )
        eval_df = self.eval_test.copy()
        eval_df.drop("resource", axis=1, inplace=True)
        eval_df.rename(
            columns={
                "TotalBudget": "resource",
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )

        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        if median:
            eval_df = eval_df.groupby("resource").median()
        else:
            eval_df = eval_df.groupby("resource").mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df


class SequentialSearchExperiment(Experiment):
    """
    Holds parameters needed for sequential search experiment

    Attributes
    ----------
    parent_params: ExperimentParameters
        Parent experiment parameters
    name : str
        name for pretty printing
    meta_params : pd.DataFrame
        Best metaparameters (Exploration budget and Tau)
    eval_train : pd.DataFrame4
        Resulting parameters of meta_params on training set
    eval_test : pd.DataFrame
        Resulting parameters of meta_params on testing set
    ssParams : SequentialSearchParameters
        Parameters for sequential search
    id_name : str
        Name of experiment
    postprocess : function
        Function to postprocess meta_params
    postprocess_name : str
        Name of postprocess function

    Methods
    -------
    __init__(parent_params, ssParams, id_name=None, postprocess=None, postprocess_name=None)
        Initialize the class
    populate()
        Populates meta_params, eval_train, eval_test
    evaluate()
        Evaluates the sequential search
    """

    def __init__(
        self, parent_params, ssParams, id_name=None, postprocess=None, postprocess_name=None
    ):
        self.parent_params = parent_params
        if id_name is None:
            self.name = "SequentialSearch"
        else:
            self.name = "SequentialSearch_{}".format(id_name)
        self.ssParams = ssParams
        self.id_name = id_name
        self.meta_parameter_names = ["ExploreFrac", "tau"]
        self.resource = "TotalBudget"
        self.postprocess = postprocess
        self.postprocess_name = postprocess_name
        self.populate()

    def populate(self):
        if self.id_name is None:
            meta_params_path = os.path.join(
                self.parent_params.checkpoint_path, "SequentialSearch_meta_params.pkl"
            )
            eval_train_path = os.path.join(
                self.parent_params.checkpoint_path, "SequentialSearch_evalTrain.pkl"
            )
            if self.postprocess is None:
                eval_test_path = os.path.join(
                    self.parent_params.checkpoint_path, "SequentialSearch_evalTest.pkl"
                )
            else:
                eval_test_path = os.path.join(
                    self.parent_params.checkpoint_path,
                    "SequentialSearch_evalTest_postprocess={}.pkl".format(
                        self.postprocess_name
                    ),
                )
        else:
            meta_params_path = os.path.join(
                self.parent_params.checkpoint_path,
                "SequentialSearch_meta_params_id={}.pkl".format(self.id_name),
            )
            eval_train_path = os.path.join(
                self.parent_params.checkpoint_path,
                "SequentialSearch_evalTrain_id={}.pkl".format(self.id_name),
            )
            if self.postprocess is None:
                eval_test_path = os.path.join(
                    self.parent_params.checkpoint_path,
                    "SequentialSearch_evalTest_id={}.pkl".format(self.id_name),
                )
            else:
                eval_test_path = os.path.join(
                    self.parent_params.checkpoint_path,
                    "SequentialSearch_evalTest_id={}_postprocess={}.pkl".format(
                        self.id_name, self.postprocess_name
                    ),
                )

        if os.path.exists(meta_params_path):
            self.meta_params = pd.read_pickle(meta_params_path)
            self.eval_train = pd.read_pickle(eval_train_path)
        else:
            training_results = self.parent_params.interp_results[
                self.parent_params.interp_results["train"] == 1
            ].copy()
            (
                self.meta_params,
                self.eval_train,
                _,
            ) = sequential_exploration.SequentialExploration(
                training_results, self.ssParams, group_on=self.parent_params.instance_cols
            )
            self.meta_params.to_pickle(meta_params_path)
            self.eval_train.to_pickle(eval_train_path)
        self.meta_params["ExploreFrac"] = (
            self.meta_params["ExplorationBudget"] / self.meta_params["TotalBudget"]
        )
        if self.postprocess is not None:
            self.preproc_meta_params = self.meta_params.copy()
            self.meta_params = self.postprocess(self.meta_params)
        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
            # try:
            logger.info("\t Evaluating sequential search on test")
            testing_results = self.parent_params.interp_results[
                self.parent_params.interp_results["train"] == 0
            ].copy()
            self.eval_test = sequential_exploration.apply_allocations(
                testing_results,
                self.ssParams,
                self.meta_params,
                self.parent_params.instance_cols,
            )
            self.eval_test.to_pickle(eval_test_path)
            # except:
            #     print('Not enough test data for sequential search. Evaluating on train.')

    def evaluate(self):
        """
        Evaluates the sequential search

        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        if hasattr(self, "eval_test"):
            params_df = self.eval_test.loc[
                :, ["TotalBudget"] + self.parent_params.parameter_names
            ]
            eval_df = self.eval_test.copy()
        else:
            params_df = self.eval_train.loc[
                :, ["TotalBudget"] + self.parent_params.parameter_names
            ]
            eval_df = self.eval_train.copy()

        for col in params_df.columns:
            if params_df[col].dtype == "object":
                params_df.loc[:, col] = params_df.loc[:, col].astype(float)

        temp = params_df.groupby("TotalBudget").mean()
        params_df.reset_index(inplace=True)
        params_df.rename(columns={"TotalBudget": "resource"}, inplace=True)
        base = names.param2filename({"Key": self.parent_params.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.parent_params.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.parent_params.response_key, "ConfInt": "upper"}, ""
        )

        eval_df.drop("resource", axis=1, inplace=True)
        eval_df.rename(
            columns={
                "TotalBudget": "resource",
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )

        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]
        if median:
            eval_df = eval_df.groupby("resource").median()
        else:
            eval_df = eval_df.groupby("resource").mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df


class VirtualBestBaseline:
    """
    Calculates virtual best on an instance by instance basis

    Attributes
    ----------
    parent_params: ExperimentParameters
        Parent experiment parameters
    name : str
        name for pretty printing
    rec_params : pd.DataFrame
        Dataframe of best paremeters per instance and resource level

    Methods
    -------
    savename()
        Returns the path to save the results
    populate()
        Calculates the virtual best
    evaluate()
        Evaluates the virtual best
    recalibrate()
        Recalibrates the response of virtual best based on best found value
    """

    parent_params: ExperimentParameters

    def __init__(self, parent_params):
        self.parent_params = parent_params
        self.name = "VirtualBest"
        self.populate()

    def savename(self):
        return os.path.join(self.parent_params.checkpoint_path, "VirtualBest_test.pkl")

    def populate(self):
        if os.path.exists(self.savename()):
            self.rec_params = pd.read_pickle(self.savename())
        else:
            response_col = names.param2filename({"Key": self.parent_params.response_key}, "")
            testing_results = self.parent_params.interp_results[
                self.parent_params.interp_results["train"] == 0
            ].copy()
            self.rec_params = training.virtual_best(
                testing_results,
                parameter_names=self.parent_params.parameter_names,
                response_col=response_col,
                response_dir=self.parent_params.response_dir,
                groupby=self.parent_params.instance_cols,
                resource_col="resource",
                smooth=self.parent_params.smooth,
                additional_cols=[
                    "ConfInt=lower_" + response_col,
                    "ConfInt=upper_" + response_col,
                ],
            )
            self.rec_params.to_pickle(self.savename())

    def recalibrate(self, new_df):
        """
        Parameters
        ----------
        new_df : pd.DataFrame
            pandas dataframe with the new data. Should only have columns
            ['resource'. *(parameters_names), response, response_lower, response_upper]
            response cols should match name of results columns
        Updates params and evaluation to take in new data
        """
        base = names.param2filename({"Key": self.parent_params.response_key}, "")
        joint_cols = (
            ["resource", base] + self.parent_params.parameter_names + self.parent_params.instance_cols
        )
        new_df = new_df.loc[:, joint_cols]
        joint = pd.concat(
            [self.rec_params.loc[:, joint_cols], new_df], ignore_index=True
        )

        self.rec_params = training.virtual_best(
            joint,
            parameter_names=self.parent_params.parameter_names,
            response_col=base,
            response_dir=self.parent_params.response_dir,
            groupby=self.parent_params.instance_cols,
            resource_col="resource",
            additional_cols=[],
            smooth=self.parent_params.smooth,
        )

    def evaluate(self):
        """
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df = self.rec_params.loc[:, ["resource"] + self.parent_params.parameter_names]
        params_df = params_df.groupby("resource").mean()

        base = names.param2filename({"Key": self.parent_params.response_key}, "")
        CIlower = names.param2filename(
            {"Key": self.parent_params.response_key, "ConfInt": "lower"}, ""
        )
        CIupper = names.param2filename(
            {"Key": self.parent_params.response_key, "ConfInt": "upper"}, ""
        )
        eval_df = self.rec_params.copy()
        eval_df.rename(
            columns={
                base: "response",
                CIlower: "response_lower",
                CIupper: "response_upper",
            },
            inplace=True,
        )

        eval_df = eval_df.loc[
            :, ["resource", "response", "response_lower", "response_upper"]
        ]

        def StatsSingle(df_single: pd.DataFrame, stat_params: stats.StatsParameters):
            """
            Function for computing the stat (such as mean) and confidence intervals of the response

            Parameters
            ----------
            df_single : pd.DataFrame
                Dataframe of a single resource level
            stat_params : stats.StatsParameters
                Only one stats_measure will be used.

            Returns
            -------
            pd.Dataframe
            """
            df_dict = {}
            sm = stat_params.stats_measures[0]  # Ignore the rest if they exist
            base, CIlower, CIupper = sm.ConfInts(
                df_single["response"],
                df_single["response_lower"],
                df_single["response_upper"],
            )

            df_dict["response"] = [base]
            df_dict["response_lower"] = [CIlower]
            df_dict["response_upper"] = [CIupper]
            df_dict["count"] = len(df_single["response"])

            df_stats_single = pd.DataFrame.from_dict(df_dict)
            return df_stats_single

        def applyBounds(df: pd.DataFrame, stat_params: stats.StatsParameters):
            """
            Trim the response values obtained from statsSingle to be between 0 and 1

            Parameters
            ----------
            df : pd.DataFrame
                Dataframe of a single resource level
            stat_params : stats.StatsParameters
                Only one stats_measure will be used.
            """
            df_copy = df.loc[:, ("response_lower")].copy()
            df_copy.clip(lower=0.0, inplace=True)
            df.loc[:, ("response_lower")] = df_copy

            df_copy = df.loc[:, ("response_upper")].copy()
            df_copy.clip(upper=1.0, inplace=True)
            df.loc[:, ("response_upper")] = df_copy
            return

        def Stats(
            df: pd.DataFrame, stats_params: stats.StatsParameters, group_on=["resource"]
        ):
            """
            Compute a stat(eg. mean) of the response along with CIs for it, for each value of resource for the virtual best

            Parameters
            ----------
            df : pd.DataFrame
                Dataframe of a single resource level with columns 'resource', 'response', 'response_lower' and 'response_upper'
            stats_params : stats.StatsParameters
                Only one statsMeasure will be used.
            group_on : list[str]
                Confidence interval propagation will be done for all rows of dataframe having the same values for groupon
            
            Returns
            -------
            pd.DataFrame
                Dataframe with columns 'resource', 'response', 'response_lower' and 'response_upper'
            """

            def dfSS(df):
                return StatsSingle(df, stats_params)

            df_stats = df.groupby(group_on).progress_apply(dfSS, include_groups=False).reset_index()
            df_stats.drop("level_{}".format(len(group_on)), axis=1, inplace=True)
            applyBounds(df_stats, stats_params)

            return df_stats

        if median:
            stParams = stats.StatsParameters(
                metrics=["response"], stats_measures=[stats.Median()]
            )
            eval_df = Stats(eval_df, stParams, ["resource"])
        else:
            stParams = stats.StatsParameters(
                metrics=["response"], stats_measures=[stats.Mean()]
            )
            eval_df = Stats(eval_df, stParams, ["resource"])
        eval_df.reset_index(inplace=True)
        return params_df, eval_df
