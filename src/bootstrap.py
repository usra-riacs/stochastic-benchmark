from collections import defaultdict
import copy
from dataclasses import dataclass, field
from multiprocess import Process, Pool, Manager
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Callable

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
    """
    shared_args: dict #'resource_col, response_col, response_dir, best_value, random_value, confidence_level'
    update_rule: Callable[[pd.DataFrame], None]= field()
    agg: str = field(default_factory = lambda : None)
    metric_args: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(lambda: None))
    success_metrics: dict = field(default_factory = lambda:[success_metrics.PerfRatio])
    bootstrap_iterations: int = 1000
    downsample: int = 10
    keep_cols: list = field(default_factory=lambda: [])
    
    def __post_init__(self):
        temp_metric_args = defaultdict(lambda : None)
        temp_metric_args.update(self.metric_args)
        self.metric_args = temp_metric_args
        
        if not hasattr(self, 'update_rule'):
            self.update_rule = self.default_update
    
    def default_update(self, df):
        if self.shared_args['response_dir'] == -1:  # Minimization
            self.shared_args['best_value'] = df[self.shared_args['response_col']].min()
        else:  # Maximization
            self.shared_args['best_value'] = df[self.shared_args['response_col']].max()
        self.metric_args['RTT']['RTT_factor'] = 1e-6*df[self.shared_args['resource_col']].sum()

class BSParams_iter:
    """
    Iterator for bootstrap parameters
    """
    def __iter__(self):
        return self

    def __next__(self):
        if self.bs_params.downsample <= self.nboots - 1:
            boots = self.bs_params.downsample
            self.bs_params.downsample += 1
            return copy.deepcopy(self.bs_params)
        else:
            raise StopIteration

    def __call__(self, bs_params, nboots):
        self.nboots = nboots
        self.bs_params = bs_params
        self.bs_params.downsample = 0
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
        p =  list(df[bs_params.agg] / df[bs_params.agg].sum())
        resamples = np.random.choice(len(df), (bs_params.downsample, bs_params.bootstrap_iterations), p)
    else:
        resamples = np.random.randint(0, len(df), size=(
            bs_params.downsample, bs_params.bootstrap_iterations), dtype=np.intp)
    responses = df[bs_params.shared_args['response_col']].values[resamples]
    resources = df[bs_params.shared_args['resource_col']].values[resamples]
    
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
        metric = metric_ref(bs_params.shared_args,bs_params.metric_args[metric_ref.__name__])
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
    group_on : str
        Column name to group on.
    bs_params_list: list or iterator of bootstrap parameters
        Number of bootstraps.

    Returns
    -------
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    """       
    def f(bs_params):

        if progress_dir is not None:
            filename = os.path.join(progress_dir, 'bootstrapped_results_boots={}.pkl'.format(bs_params.downsample))
            if os.path.exists(filename):
                temp_df = pd.read_pickle(filename)
                return temp_df

        temp_df = df.groupby(group_on).progress_apply(lambda df : BootstrapSingle(df, bs_params)).reset_index()
        temp_df.drop('level_{}'.format(len(group_on)), axis=1, inplace=True)
        temp_df['boots'] = bs_params.downsample
        
        if progress_dir is not None:
            temp_df.to_pickle(filename)

        return temp_df

    with Pool() as p:
        df_list = p.map(f, bs_params_list)
    return pd.concat(df_list)
