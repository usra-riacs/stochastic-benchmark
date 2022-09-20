from collections import defaultdict
import copy
from dataclasses import dataclass, field
from multiprocess import Process, Pool, Manager
import numpy as np
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
    agg: str = field()
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

# Iterator for bootstrap parameters
class BSParams_iter:
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
    if hasattr(bs_params, 'agg'):
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


def Bootstrap(df, group_on, bs_params_list):
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
        temp_df = df.groupby(group_on).progress_apply(lambda df : BootstrapSingle(df, bs_params)).reset_index()
        temp_df.drop('level_{}'.format(len(group_on)), axis=1, inplace=True)
        temp_df['boots'] = bs_params.downsample
        return temp_df

    with Pool() as p:
        df_list = p.map(f, bs_params_list)
    return pd.concat(df_list)


# def computeResponse(df, bs_df, bs_params, resamples, responses):
#     """
#     Compute the response of each bootstrap samples and its corresponding confidence interval based on the resamples.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing the data.
#     bs_df : pandas.DataFrame
#         DataFrame containing the bootstrap results.
#     bs_params : BootstrapParameters
#         Parameters for the bootstrap method.
#     resamples : numpy.ndarray
#         Array of resamples.
#     responses : numpy.ndarray
#         Array of responses.

#     Returns
#     -------
#     bs_df : pandas.DataFrame
#         DataFrame containing the bootstrap results.
#     """
#     # Compute the minimum value of each bootstrap samples and its corresponding confidence interval based on the resamples
#     if bs_params.response_dir == -1:  # Minimization
#         response_dist = np.apply_along_axis(
#             func1d=np.min, axis=0, arr=responses[resamples])
#     else:  # Maximization
#         response_dist = np.apply_along_axis(
#             func1d=np.max, axis=0, arr=responses[resamples])
#         # TODO This could be generalized as the X best samples
    
#     key = 'Response'
#     basename = names.param2filename({'Key': key}, '')
#     CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
#     CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')

#     bs_df[basename] = [np.mean(response_dist)]
#     bs_df[CIlower] = np.nanpercentile(
#         response_dist, 50-confidence_level/2)
#     bs_df[CIupper] = np.nanpercentile(
#         response_dist, 50+confidence_level/2)
#     # TODO PyLance is crying here because confidence_level is not defined


# def computePerfRatio(df, bs_df, bs_params):
#     """
#     Compute the performance ratio of each bootstrap samples and its corresponding confidence interval based on the resamples.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing the data.
#     bs_df : pandas.DataFrame
#         DataFrame containing the bootstrap results.
#     bs_params : BootstrapParameters
#         Parameters for the bootstrap method.
#     """
#     # Compute the success metric (performance ratio) of each bootstrap samples and its corresponding confidence interval based on the resamples
#     if bs_params.success_metric == 'PerfRatio':
#         key = 'PerfRatio'
#         basename = names.param2filename({'Key': key}, '')
#         CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
#         CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')

#         bs_df[basename] = (bs_params.random_value - bs_df[names.param2filename({'Key': 'Response'}, '')])\
#             / (bs_params.random_value - bs_params.best_value)
#         bs_df[CIlower] = (bs_params.random_value - bs_df[names.param2filename({'Key': 'Response', 'ConfInt': 'upper'}, '')]) \
#             / (bs_params.random_value - bs_params.best_value)
#         bs_df[CIupper] = (bs_params.random_value - bs_df[names.param2filename({'Key': 'Response', 'ConfInt': 'lower'}, '')])\
#             / (bs_params.random_value - bs_params.best_value)
#     else:
#         print("Success metric not implemented yet")
#         # TODO here the input could be a function


# def computeInvPerfRatio(df, bs_df, bs_params):
#     """
#     Compute the inverse performance ratio of each bootstrap samples and its corresponding confidence interval based on the resamples.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing the data.
#     bs_df : pandas.DataFrame
#         DataFrame containing the bootstrap results.
#     bs_params : BootstrapParameters
#         Parameters for the bootstrap method.
#     """
#     # Compute the inverse success metric (performance ratio) of each bootstrap samples and its corresponding confidence interval based on the resamples
#     if bs_params.success_metric == 'PerfRatio':
#         key = 'InvPerfRatio'
#         basename = names.param2filename({'Key': key}, '')
#         CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
#         CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')

#         bs_df[basename] = 1 - (bs_params.random_value - bs_df[names.param2filename({'Key': 'Response'}, '')]) / \
#             (bs_params.random_value - bs_params.best_value) + EPSILON
#         bs_df[CIlower] = 1 - (bs_params.random_value - bs_df[names.param2filename({'Key': 'Response', 'ConfInt': 'lower'}, '')])\
#             / (bs_params.random_value - bs_params.best_value) + EPSILON
#         bs_df[CIupper] = 1 - (bs_params.random_value - bs_df[names.param2filename({'Key': 'Response', 'ConfInt': 'upper'}, '')])\
#             / (bs_params.random_value - bs_params.best_value) + EPSILON
#         # TODO PyLance is crying here because EPSILON is not defined
#     else:
#         print("Success metric not implemented yet")
#         # TODO here the input could be a function


# def computeSuccessProb(df, bs_df, bs_params, resamples, responses):
#     """
#     Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing the data.
#     bs_df : pandas.DataFrame
#         DataFrame containing the bootstrap results.
#     bs_params : BootstrapParameters
#         Parameters for the bootstrap method.
#     resamples : numpy.ndarray
#         Array of resamples.
#     responses : numpy.ndarray
#         Array of responses.
#     """
#     aggregated_df_flag = False
#     if bs_params.response_dir == - 1:  # Minimization
#         success_val = bs_params.random_value - \
#             (1.0 - gap/100.0)*(bs_params.random_value - bs_params.best_value)
#     else:  # Maximization
#         success_val = (1.0 - gap/100.0) * \
#             (bs_params.best_value - bs_params.random_value) - bs_params.random_value
#     # TODO Here we only include relative performance ratio. Consider other objectives as in benchopt
#     # TODO PyLance is crying here because gap is not defined

#     # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
#     if aggregated_df_flag:
#         print('Aggregated dataframe')
#         return []
#         # TODO: One can think about desaggregating the dataframe here. Maybe check Dwave's code for this.
#     else:
#         if bs_params.response_dir == -1:  # Minimization
#             success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
#                 x < success_val)/bs_params.downsample, axis=0, arr=responses[resamples])
#         else:  # Maximization
#             success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
#                 x > success_val)/bs_params.downsample, axis=0, arr=responses[resamples])
#     # Consider np.percentile instead to reduce package dependency. We need to benchmark and test alternative

#     key = 'SuccProb'
#     basename = names.param2filename({'Key': key}, '')
#     CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
#     CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')

#     bs_df[basename] = np.mean(success_prob_dist)
#     bs_df[CIlower] = np.nanpercentile(
#         success_prob_dist, 50 - bs_params.confidence_level/2)
#     bs_df[CIupper] = np.nanpercentile(
#         success_prob_dist, 50 + bs_params.confidence_level/2)


# def computeResource(df, bs_df, bs_params, resamples, times):
#     """
#     Compute the resource of each bootstrap samples and its corresponding confidence interval based on the resamples.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing the data.
#     bs_df : pandas.DataFrame
#         DataFrame containing the bootstrap results.
#     bs_params : BootstrapParameters
#         Parameters for the bootstrap method.
#     resamples : numpy.ndarray
#         Array of resamples.
#     times : numpy.ndarray
#         Array of times.
#     """
#     # Compute the resource (time) of each bootstrap samples and its corresponding confidence interval based on the resamples
#     resource_dist = np.apply_along_axis(
#         func1d=np.mean, axis=0, arr=times[resamples])

#     key = 'MeanTime'
#     basename = names.param2filename({'Key': key}, '')
#     CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
#     CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')

#     bs_df[basename] = np.mean(resource_dist)
#     bs_df[CIlower] = np.nanpercentile(
#         resource_dist, 50 - bs_params.confidence_level/2)
#     bs_df[CIupper] = np.nanpercentile(
#         resource_dist, 50 + bs_params.confidence_level/2)


# def computeRTT(df, bs_df, bs_params, resamples, responses):
#     """
#     Compute the RTT of each bootstrap samples and its corresponding confidence interval based on the resamples.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing the data.
#     bs_df : pandas.DataFrame
#         DataFrame containing the bootstrap results.
#     bs_params : BootstrapParameters
#         Parameters for the bootstrap method.
#     resamples : numpy.ndarray
#         Array of resamples.
#     responses : numpy.ndarray
#         Array of responses.
#     """
#     if bs_params.response_dir == - 1:  # Minimization
#         success_val = bs_params.random_value - \
#             (1.0 - gap/100.0)*(bs_params.random_value - bs_params.best_value)
#     else:  # Maximization
#         success_val = (1.0 - gap/100.0) * \
#             (bs_params.best_value - bs_params.random_value) - bs_params.random_value
#     aggregated_df_flag = False
#     # Compute the resource to target (RTT) within certain threshold of each bootstrap
#     # samples and its corresponding confidence interval based on the resamples
#     rtt_factor = 1e-6*df[bs_params.resource_col].sum()

#     # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
#     if aggregated_df_flag:
#         print('Aggregated dataframe')
#         return []
#         # TODO: One can think about desaggregating the dataframe here. Maybe check Dwave's code for this.
#     else:
#         if bs_params.response_dir == -1:  # Minimization
#             success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
#                 x < success_val)/bs_params.downsample, axis=0, arr=responses[resamples])
#         else:  # Maximization
#             success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
#                 x > success_val)/bs_params.downsample, axis=0, arr=responses[resamples])

#     rtt_dist = computeRTT_vectorized(
#         success_prob_dist, bs_params, scale=rtt_factor)
#     # Question: should we scale the RTT with the number of bootstrapping we do, intuition says we don't need to
#     key = 'RTT'
#     basename = names.param2filename({'Key': key}, '')
#     CIupper = names.param2filename({'Key': key, 'ConfInt': 'upper'}, '')
#     CIlower = names.param2filename({'Key': key, 'ConfInt': 'lower'}, '')

#     rtt = np.mean(rtt_dist)

#     bs_df[basename] = rtt
#     if np.isinf(rtt) or np.isnan(rtt) or rtt == bs_params.fail_value:
#         bs_df[CIlower] = bs_params.fail_value
#         bs_df[CIupper] = bs_params.fail_value
#     else:
#         # rtt_conf_interval = computeRTT_vectorized(
#         #     success_prob_conf_interval, s=0.99, scale=1e-6*df_default_samples['runtime (us)'].sum())
#         bs_df[CIlower] = np.nanpercentile(
#             rtt_dist, 50-confidence_level/2)
#         bs_df[CIupper] = np.nanpercentile(
#             rtt_dist, 50+confidence_level/2)
#         # TODO: Pylance is crying here because confidence interval is not defined.
#     # Question: How should we compute the confidence interval of the RTT? Should we compute the function on the confidence interval of the probability or compute the confidence interval over the RTT distribution?


# def computeRTTSingle(success_probability: float, bs_params, scale: float = 1.0, size: int = 1000):
#     '''
#     Computes the resource to target metric given some success probabilty of getting to that target and a scale factor.

#     Parameters
#     ----------
#     success_probability : float
#         The success probability of getting to the target.
#     bs_params : BootstrapParameters
#         Parameters for the bootstrap method.
#     scale : float
#         The scale factor.
#     size : int
#         The number of bootstrap samples to use.

#     Returns
#     -------
#     rtt : float
#         The resource to target metric.
#     '''
#     # Defaults to np.inf but then overwrites (if None to nan)
#     if bs_params.fail_value is None:
#         bs_params.fail_value = np.nan
#     if success_probability == 0:
#         return bs_params.fail_value
#     elif success_probability == 1:
#         # Consider continuous RTT and RTT scaled by assuming success_probability=1 as success_probability=1/size*(1-1/10)
#         return scale*np.log(1.0 - bs_params.s) / np.log(1 - (1 - 1/10)/size)
#     else:
#         return scale*np.log(1.0 - bs_params.s) / np.log(1 - success_probability)


# computeRTT_vectorized = np.vectorize(computeRTTSingle, excluded=(1, 2, 3))


