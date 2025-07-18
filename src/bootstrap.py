from collections import defaultdict
import copy
from dataclasses import dataclass, field
import df_utils
from itertools import product
from multiprocess import Process, Pool, Manager
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Callable, List, DefaultDict

import names
import success_metrics

EPSILON = 1e-10
confidence_level = 68
gap = 1.0

tqdm.pandas()


@dataclass
class BootstrapParameters:
    """
    Parameters for the bootstrap method.

    Attributes
    ----------
    shared_args : dict
        Shared arguments for the bootstrap method.
        We usually have 'resource_col, response_col, response_dir, best_value, random_value, confidence_level'
    update_rule : Callable[[pd.DataFrame], None]
        Function to update the dataframe with the bootstrap results.
    agg : str
        Aggregation function to use for the bootstrap.
    metric_args : DefaultDict[str, dict]
        Dictionary of metric arguments to pass to the success_metrics functions.
    success_metrics : dict
        Dictionary of success_metrics functions to use.
    bootstrap_iterations : int
        Number of bootstrap iterations to perform.
    downsample : int
        Number of bootstrap iterations to perform.
    keep_cols : list
        List of columns to keep in the dataframe.

    Methods
    -------
    __post_init__()
        Post-initialization function.
    default_update(df)
        Default update rule for the bootstrap method.
    """

    shared_args: dict  #'resource_col, response_col, response_dir, best_value, random_value, confidence_level'
    update_rule: Callable[[pd.DataFrame], None] = field()
    agg: str = field(default_factory=lambda: None)
    metric_args: DefaultDict[str, dict] = field(
        default_factory=lambda: defaultdict(lambda: None)
    )
    success_metrics: dict = field(default_factory=lambda: [success_metrics.PerfRatio])
    bootstrap_iterations: int = 1000
    downsample: int = 10
    keep_cols: List = field(default_factory=lambda: [])

    def __post_init__(self):
        temp_metric_args = defaultdict(lambda: None)
        temp_metric_args.update(self.metric_args)
        self.metric_args = temp_metric_args

        if not hasattr(self, "update_rule"):
            self.update_rule = self.default_update

    def default_update(self, df):
        if self.shared_args["response_dir"] == -1:  # Minimization
            self.shared_args["best_value"] = df[self.shared_args["response_col"]].min()
        else:  # Maximization
            self.shared_args["best_value"] = df[self.shared_args["response_col"]].max()
        self.metric_args["RTT"]["RTT_factor"] = (
            1e-6 * df[self.shared_args["resource_col"]].sum()
        )


class BSParams_iter:
    """
    Iterator for bootstrap parameters

    Attributes
    ----------
    nboots : int
        Number of bootstrap iterations to perform.
    bs_params : BootstrapParameters
        Bootstrap parameters.

    Methods
    -------
    __call__(bs_params, nboots)
        Initialize the iterator.

    __iter__()
        Return the iterator.
    """

    def __iter__(self):
        return self

    def __next__(self):
        if self.bs_params.downsample <= self.nboots - 1:
            # Make a deep copy with current downsample value
            result = copy.deepcopy(self.bs_params)
            result.downsample = self.bs_params.downsample
            # Then increment for next iteration
            self.bs_params.downsample += 1
            return result
        else:
            raise StopIteration

    def __call__(self, bs_params, nboots):
        self.nboots = nboots
        self.bs_params = bs_params
        self.bs_params.downsample = 0
        return self


class BSParams_range_iter:
    """
    Iterator for bootstrap parameters

    Attributes
    ----------
    bs_params : BootstrapParameters
        Bootstrap parameters.
    boots_iter : iter
        Iterator for the bootstrap iterations.

    Methods
    -------
    __call__(bs_params, boots_iter)
        Initialize the iterator.
    __iter__()
        Return the iterator.
    __next__()
        Return the next bootstrap parameters.
    """

    def __iter__(self):
        return self

    def __next__(self):
        self.bs_params.downsample = next(self.boots_iter)
        if self.bs_params is not None:
            return copy.deepcopy(self.bs_params)
        else:
            raise StopIteration

    def __call__(self, bs_params, boots_iter):
        self.boots_iter = iter(boots_iter)
        self.bs_params = bs_params

        return self


def initBootstrap(df, bs_params):
    """
    Initialize the bootstrap method.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.

    Returns
    -------
    resamples : list
        List of resamples.
    responses : numpy.ndarray
        Array of responses.
    times : numpy.ndarray
        Array of times.
    """
    if bs_params.agg is not None:
        p = list(df[bs_params.agg] / df[bs_params.agg].sum())
        resamples = np.random.choice(
            len(df), (bs_params.downsample, bs_params.bootstrap_iterations), p=p
        )
    else:
        resamples = np.random.randint(
            0,
            len(df),
            size=(bs_params.downsample, bs_params.bootstrap_iterations),
            dtype=np.intp,
        )
    responses = df[bs_params.shared_args["response_col"]].values[resamples]
    resources = df[bs_params.shared_args["resource_col"]].values[resamples]

    bs_params.update_rule(bs_params, df)
    return responses, resources


def BootstrapSingle(df, bs_params):
    """
    Bootstrap single function.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.

    Returns
    -------
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    """
    responses, resources = initBootstrap(df, bs_params)
    bs_params.update_rule(bs_params, df)
    bs_df = pd.DataFrame()
    for metric_ref in bs_params.success_metrics:
        metric = metric_ref(
            bs_params.shared_args, bs_params.metric_args[metric_ref.__name__]
        )
        metric.evaluate(bs_df, responses, resources)

    for col in bs_params.keep_cols:
        val = df[col].iloc[0]
        bs_df[col] = val

    return bs_df


def Bootstrap(df, group_on, bs_params_list, progress_dir=None):
    """
    Bootstrap function.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.

    If Split=False
        group_on : List[str]
            Column name to group on.
    If split=True
        group_on = List[List[str]]
            group_on = [upper_group_on, lower_group_on]
            upper_group_on splits will be written to different files
            lower_group_on
    bs_params_list: list or iterator of bootstrap parameters
        Number of bootstraps.
    progress_dir : str, optional
        Directory to write progress to. The default is None.

    Returns
    -------
    If split=False:
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    If split=True:
    df_list : List[str]
        List of strings pointing to files with portions of bootstrapped_results
    """
    if type(df) == list:
        if type(df)[0] == pd.DataFrame:
            df = pd.concat(df, ignore_index=True)
        elif type(df)[0] == str:
            df = pd.concat([pd.read_pickle(df_str) for df_str in df], ignore_index=True)
    elif type(df) == str:
        df = pd.read_pickle(df)

    if type(df) != pd.DataFrame:
        print("Unsupport type as bootstrap input")

    def f(bs_params):
        if progress_dir is not None:
            filename = os.path.join(
                progress_dir,
                "bootstrapped_results_boots={}.pkl".format(bs_params.downsample),
            )
            if os.path.exists(filename):
                temp_df = pd.read_pickle(filename)
                return temp_df

        temp_df = (
            df.groupby(group_on)
            .progress_apply(lambda df: BootstrapSingle(df, bs_params))
            .reset_index()
        )
        temp_df.drop("level_{}".format(len(group_on)), axis=1, inplace=True)
        temp_df["boots"] = bs_params.downsample
        return temp_df

    with Pool() as p:
        df_list = p.map(f, bs_params_list)
    return pd.concat(df_list)


def Bootstrap_reduce_mem(df, group_on, bs_params_list, bootstrap_dir, name_fcn=None):
    """
    Bootstrap function with reduced memory usage.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    group_on : List[str]
        Column name to group on.
    bs_params_list: list or iterator of bootstrap parameters
        Number of bootstraps.
    bootstrap_dir : str
        Directory to write progress to.
    name_fcn : function, optional
        Function to generate the name of the group. The default is None.

    Returns
    -------
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    """
    bs_params_list = list(bs_params_list)

    if (type(df) == pd.DataFrame) or (type(df) == str):
        if type(df) == str:
            df = pd.read_pickle(df)
        lower_group_on = group_on[1]
        group_on = group_on[0]

        def upper_f(df_upper_group, bs_params_list):
            group_name = name_fcn(df_upper_group[0])
            df_group = df_upper_group[1]
            filename = os.path.join(
                bootstrap_dir, "bootstrapped_results_{}.pkl".format(group_name)
            )  # TODO fix filename

            if not os.path.exists(filename):

                def bs_all_par(par_group):
                    bs_params = par_group[0]
                    lower_group_df = par_group[1]
                    return BootstrapSingle(lower_group_df[1], bs_params)

                def bs_params_eval(bs_params):
                    temp_df = (
                        df_group.groupby(group_on)
                        .progress_apply(lambda df: BootstrapSingle(df, bs_params))
                        .reset_index()
                    )
                    temp_df.drop("level_{}".format(len(group_on)), axis=1, inplace=True)
                    temp_df["boots"] = bs_params.downsample
                    return temp_df

                # par_group = product(bs_params_list, df_group.groupby(group_on))
                # with Pool() as p:
                #     df_list = p.imap(bs_all_par, par_group)

                with Pool() as p:
                    df_list = p.map(bs_params_eval, bs_params_list)

                res = pd.concat(df_list, ignore_index=True)
                res.to_pickle(filename)
            return filename

    elif type(df) == list:
        if type(df[0]) == str:
            print("calling list of names method")

            def upper_f(upper_group_filename, bs_params_list):
                df_group = pd.read_pickle(upper_group_filename)
                group_name = name_fcn(upper_group_filename)
                print("evaluation bs for {}".format(group_name))
                filename = os.path.join(
                    bootstrap_dir, "bootstrapped_results_{}.pkl".format(group_name)
                )  # TODO fix filename
                if not os.path.exists(filename):

                    def bs_all_par(par_group):
                        bs_params = par_group[0]
                        lower_group_df = par_group[1]
                        return BootstrapSingle(lower_group_df[1], bs_params)

                    def bs_params_eval(bs_params):
                        temp_df = (
                            df_group.groupby(group_on)
                            .progress_apply(lambda df: BootstrapSingle(df, bs_params))
                            .reset_index()
                        )
                        temp_df.drop(
                            "level_{}".format(len(group_on)), axis=1, inplace=True
                        )
                        temp_df["boots"] = bs_params.downsample
                        return temp_df

                    # par_group = product(bs_params_list, df_group.groupby(group_on))
                    # with Pool() as p:
                    #     df_list = p.imap(bs_all_par, par_group)

                    with Pool() as p:
                        df_list = p.map(bs_params_eval, bs_params_list)

                    res = pd.concat(df_list, ignore_index=True)
                    res.to_pickle(filename)
                return filename

        elif type(df[0]) == pd.DataFrame:

            def upper_f(df_group, bs_params_list):
                group_name = name_fcn(df_group)
                filename = os.path.join(
                    bootstrap_dir, "bootstrapped_results_{}.pkl".format(group_name)
                )  # TODO fix filename
                if not os.path.exists(filename):

                    def bs_all_par(par_group):
                        bs_params = par_group[0]
                        lower_group_df = par_group[1]
                        return BootstrapSingle(lower_group_df[1], bs_params)

                    def bs_params_eval(bs_params):
                        temp_df = (
                            df_group.groupby(group_on)
                            .progress_apply(lambda df: BootstrapSingle(df, bs_params))
                            .reset_index()
                        )
                        temp_df.drop(
                            "level_{}".format(len(group_on)), axis=1, inplace=True
                        )
                        temp_df["boots"] = bs_params.downsample
                        return temp_df

                    # par_group = product(bs_params_list, df_group.groupby(group_on))
                    # with Pool() as p:
                    #     df_list = p.imap(bs_all_par, par_group)

                    with Pool() as p:
                        df_list = p.map(bs_params_eval, bs_params_list)

                    res = pd.concat(df_list, ignore_index=True)
                    res.to_pickle(filename)
                return filename

    bs_filenames = [upper_f(df_group, bs_params_list) for df_group in df]
    return bs_filenames
