from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import List, Tuple, Union
import warnings

import names

EPSILON = 1e-10

tqdm.pandas()

@dataclass
class StatsParameters:
    """
    Parameters for stats computation
    """
    metrics: list = field(default_factory=lambda: ['MinEnergy', 'RTT',
                                                   'PerfRatio', 'SuccProb', 'MeanTime', 'InvPerfRatio'])
    lower_bounds: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(lambda: None))
    upper_bounds: defaultdict[dict] = field(
        default_factory=lambda: defaultdict(lambda: None))
    # PyLance cries as follows: Subscript for class "defaultdict" will generate runtime exception; enclose type annotation in quotes
    stats_measures: list = field(default_factory=lambda: [Median()])

    def __post_init__(self):
        self.lower_bounds['SuccProb'] = 0.0
        self.lower_bounds['MeanTime'] = 0.0
        self.lower_bounds['InvPerfRatio'] = EPSILON

        self.upper_bounds['SuccProb'] = 1.0
        self.upper_bounds['PerfRatio'] = 1.0


class StatsMeasure:
    """
    General class for stats measures with a center and confidence intervals
    """
    def __init__(self):
        self.name = None
    def __call__(self, base, lower, upper):
        raise NotImplementedError(
            "Call should be overriden by a subclass of StatsMeasure")
    def center(self, base, lower, upper):
        raise NotImplementedError(
            "Center should be overriden by a subclass of StatsMeasure")

    def ConfIntlower(self, base, lower, upper):
        raise NotImplementedError(
            "ConfIntlower should be overriden by a subclass of StatsMeasure")

    def ConfIntupper(self, base, lower, upper):
        raise NotImplementedError(
            "ConfIntupper should be overriden by a subclass of StatsMeasure")

    def ConfInts(self, base, lower, upper):
        raise NotImplementedError(
            "ConfInts should be overriden by a subclass of StatsMeasure")


class Mean(StatsMeasure):
    """
    Mean stat measure
    """
    def __init__(self):
        self.name = 'mean'
    
    def __call__(self, base: pd.DataFrame):
        return base.mean()
    
    def center(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        return base.mean()

    def ConfIntlower(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        mean_deviation = np.sqrt(
            sum((upper-lower)*(upper-lower))/(4*len(base)))
        CIlower = self.center(base, lower, upper) - mean_deviation
        return CIlower

    def ConfIntupper(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        mean_deviation = np.sqrt(
            sum((upper-lower)*(upper-lower))/(4*len(base)))
        CIupper = self.center(base, lower, upper) + mean_deviation
        return CIupper

    def ConfInts(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        cent = self.center(base, lower, upper)
        mean_deviation = np.sqrt(
            sum((upper-lower)*(upper-lower))/(4*len(base)))
        CIlower = cent - mean_deviation
        CIupper = cent + mean_deviation

        return cent, CIlower, CIupper


class Median(StatsMeasure):
    """
    Median stat measure
    """
    def __init__(self):
        self.name = 'median'
    
    def __call__(self, base: pd.DataFrame):
        return base.median()
    
    def center(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        return base.median()

    def ConfIntlower(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        mean_deviation = np.sqrt(
            sum((upper-lower)*(upper-lower))/(4*len(base)))
        median_deviation = mean_deviation * \
            np.sqrt(np.pi*len(base)/(2*len(base)-1))
        CIlower = self.center(base, lower, upper) - median_deviation
        return CIlower

    def ConfIntupper(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        mean_deviation = np.sqrt(
            sum((upper-lower)*(upper-lower))/(4*len(base)))
        median_deviation = mean_deviation * \
            np.sqrt(np.pi*len(base)/(2.*len(base)-1))
        CIupper = self.center(base, lower, upper) + median_deviation
        return CIupper

    def ConfInts(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        cent = self.center(base, lower, upper)
        mean_deviation = np.sqrt(sum((upper-lower)*(upper-lower))/(4*len(base)))
        if len(base) == 2:
            median_deviation = mean_deviation * np.sqrt(np.pi*len(base)/(4*(3./ 2.- 1.)))
        else:
            median_deviation = mean_deviation * np.sqrt(np.pi*len(base)/(4*(len(base)/ 2.- 1.)))
        CIlower = cent - median_deviation
        CIupper = cent + median_deviation

        return cent, CIlower, CIupper


class Percentile(StatsMeasure):
    """
    Percentile stat measure
    """
    def __init__(self, q, nboots, confidence_level: float = 68):
        self.q = q
        self.name = '{}Percentile'.format(q)
        self.nboots = int(nboots)
        self.confidence_level = confidence_level

    def __call__(self, base: pd.DataFrame):
        return base.quantile(self.q / 100.)
    
    def center(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        return base.quantile(self.q / 100.)

    def ConfInts(self, base: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame):
        cent = base.quantile(self.q / 100.)
        boot_dist = []
        for i in range(self.nboots):
            resampler = np.random.randint(0, len(base), len(
                base), dtype=np.intp)  # intp is indexing dtype
            # Check the following, in original code sample_ci_lower = x[key_string + '_conf_interval_upper'].values.take(resampler, axis=0)
            # but that doesn't make sense
            sample = base.values.take(resampler, axis=0)
            sample_ci_upper = upper.values.take(resampler, axis=0)
            sample_ci_lower = lower.values.take(resampler, axis=0) 
            sample_std = (sample_ci_upper-sample_ci_lower)/2.
            sample_error = np.random.normal(0, sample_std, len(sample))
            # Check the following: previously q=stat_measure/100 but percentile is defined on range 0 - 100
            # print(sample)
            # print(sample_ci_upper)
            # print(sample_ci_lower)
            # print(sample_std)
            # print(sample_error)
            # print(sample + sample_error)
            # print(np.percentile(sample + sample_error, q=self.q))
            # print(np.max(sample + sample_error))
            # print('------')
            boot_dist.append(pd.Series(sample + sample_error).quantile(self.q / 100.))            
        p =  .50 - self.confidence_level / (2 * 100.), .50 + self.confidence_level / (2. * 100.)
        (CIlower, CIupper) = pd.Series(boot_dist).quantile(p)
        return cent, CIlower, CIupper


def StatsSingle(df_single: pd.DataFrame, stat_params: StatsParameters):
    """
    Compute statistics for a single column

    Parameters
    ----------
    df_single: pd.DataFrame
        Dataframe with the metric to be analyzed
    stat_params: StatsParameters
        Parameters for the statistics

    Returns
    -------
    pd.DataFrame
        Dataframe with the statistics    
    """
    if len(df_single) == 1:
        return pd.DataFrame()
    df_dict = {}
    for sm in stat_params.stats_measures:
        for key in stat_params.metrics:
            pre_base = names.param2filename({'Key': key}, '')
            pre_CIlower = names.param2filename(
                {'Key': key, 'ConfInt': 'lower'}, '')
            pre_CIupper = names.param2filename(
                {'Key': key, 'ConfInt': 'upper'}, '')
            
            metric_basename = names.param2filename(
                {'Key': key, 'Metric': sm.name}, '')
            metric_CIlower_name = names.param2filename(
                {'Key': key, 'Metric': sm.name, 'ConfInt': 'lower'}, '')
            metric_CIupper_name = names.param2filename(
                {'Key': key, 'Metric': sm.name, 'ConfInt': 'upper'}, '')
            
            base, CIlower, CIupper = sm.ConfInts(
                df_single[pre_base], df_single[pre_CIlower], df_single[pre_CIupper])

            df_dict[metric_basename] = [base]
            df_dict[metric_CIlower_name] = [CIlower]
            df_dict[metric_CIupper_name] = [CIupper]
            df_dict['count'] = len(df_single[pre_base])

    df_stats_single = pd.DataFrame.from_dict(df_dict)
    return df_stats_single


def applyBounds(df: pd.DataFrame, stat_params: StatsParameters):
    """
    Apply the bounds to the dataframe

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the metric to be analyzed
    stat_params: StatsParameters
        Parameters for the statistics

    Returns
    -------
    pd.DataFrame
        Dataframe with the statistics   
    """
    for sm in stat_params.stats_measures:
        for key, value in stat_params.lower_bounds.items():
            lower_name = names.param2filename(
                {'Key': key, 'Metric': sm.name, 'ConfInt': 'lower'}, '')
            if lower_name in df.columns:
                df_copy = df.loc[:, (lower_name)].copy()
                df_copy.clip(lower=value, inplace=True)
                df.loc[:, (lower_name)] = df_copy

        for key, value in stat_params.upper_bounds.items():
            upper_name = names.param2filename(
                {'Key': key, 'Metric': sm.name, 'ConfInt': 'upper'}, '')
            if upper_name in df.columns:
                df_copy = df.loc[:, (upper_name)].copy()
                df_copy.clip(upper=value, inplace=True)
                df.loc[:, (upper_name)] = df_copy
    return


def Stats(df: pd.DataFrame, stats_params: StatsParameters, group_on):
    """
    Compute statistics for a dataframe

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the metric to be analyzed
    stats_params: StatsParameters
        Parameters for the statistics
    group_on: list
        List of columns to group on
    
    Returns
    -------
    pd.DataFrame
        Dataframe with the statistics
    """
    def dfSS(df): return StatsSingle(df, stats_params)
    df_stats = df.groupby(group_on).progress_apply(dfSS).reset_index()
    df_stats.drop('level_{}'.format(len(group_on)), axis=1, inplace=True)
    applyBounds(df_stats, stats_params)
    
    return df_stats
