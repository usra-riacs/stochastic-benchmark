from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Union
import warnings

import names

EPSILON = 1e-10


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

            base, CIlower, CIupper = sm.ConfInts(
                df_single[pre_base], df_single[pre_CIlower], df_single[pre_CIupper])
            metric_basename = names.param2filename(
                {'Key': key, 'Metric': sm.name}, '')
            metric_CIlower_name = names.param2filename(
                {'Key': key, 'Metric': sm.name, 'ConfInt': 'lower'}, '')
            metric_CIupper_name = names.param2filename(
                {'Key': key, 'Metric': sm.name, 'ConfInt': 'upper'}, '')

            df_dict[metric_basename] = [base]
            df_dict[metric_CIlower_name] = [CIlower]
            df_dict[metric_CIupper_name] = [CIupper]

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
    df_stats = df.groupby(group_on).apply(dfSS).reset_index()
    df_stats.drop('level_{}'.format(len(group_on)), axis=1, inplace=True)
    applyBounds(df_stats, stats_params)

    # if 'replica' in df_stats.columns:
    #     df_stats['reads'] = df_stats['sweep'] * df_stats['replica'] * df_stats['boots']
    # elif 'rep' in df_stats.columns:
    #     df_stats['reads'] = df_stats['swe'] * df_stats['rep'] * df_stats['boots']
    # else:
    #     df_stats['reads'] = df_stats['sweep'] * df_stats['boots']

    return df_stats

# kept below for testing. can delete after testing


def conf_interval(
    x: pd.Series,
    key_string: str,
    stat_measure='median',
    confidence_level: float = 68,
    bootstrap_iterations: int = 1000,
):
    '''
    Compute the mean or median and confidence interval of a series (see http://mathworld.wolfram.com/StatisticalMedian.html for uncertainty propagation)

    Args:
        x (pd.Series): Series to compute the median and confidence interval
        key_string (str): String to use as key for the output dataframe
        stat_measure (str): String to use as key for the output dataframe

    Returns:
        pd.Series: Series with median and confidence interval
    '''

    key_estimator_string = str(stat_measure) + '_' + key_string
    mean_deviation = np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(
        x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string])))

    if isinstance(stat_measure, str):

        # Allow named numpy functions
        f = getattr(np, stat_measure)

        # Try to use nan-aware version of function if necessary
        missing_data = np.isnan(np.sum(np.column_stack(x[key_string])))

        if missing_data and not stat_measure.startswith("nan"):
            nanf = getattr(np, f"nan{stat_measure}", None)
            if nanf is None:
                print(
                    "Data contain nans but no nan-aware version of `{func}` found")
            else:
                f = nanf
    elif isinstance(stat_measure, int) or isinstance(stat_measure, float):
        f = getattr(np, 'nanpercentile')
        # TODO I need to see how to instantiate this
    else:
        f = stat_measure

    if isinstance(stat_measure, int) or isinstance(stat_measure, float):
        center = np.nanpercentile(x[key_string], stat_measure)
        # TODO Fix this
        lower_interval = center
        upper_interval = center
        bootstrap_confidence_interval = False
        if bootstrap_confidence_interval:
            boot_dist = []
            # Rationale here is that we perform bootstrapping over the entire data set but considering original confidence intervals, which we assume resemble standard deviation from a normally distributed error population. This is in line with the data generation but we might want to fix it
            for i in range(int(bootstrap_iterations)):
                resampler = np.random.randint(0, len(x[key_string]), len(
                    x[key_string]), dtype=np.intp)  # intp is indexing dtype
                sample = x[key_string].values.take(resampler, axis=0)
                sample_ci_upper = x[key_string +
                                    '_conf_interval_upper'].values.take(resampler, axis=0)
                sample_ci_lower = x[key_string +
                                    '_conf_interval_upper'].values.take(resampler, axis=0)
                sample_std = (sample_ci_upper-sample_ci_lower)/2
                sample_error = np.random.normal(0, sample_std, len(sample))
                boot_dist.append(np.percentile(
                    sample + sample_error, q=stat_measure/100))
            np.array(boot_dist)
            p = 50 - confidence_level / 2, 50 + confidence_level / 2
            (lower_interval, upper_interval) = np.nanpercentile(
                boot_dist, p, axis=0)
    else:
        center = f(x[key_string])
    if stat_measure == 'mean':
        # center = np.mean(x[key_string])
        lower_interval = center - mean_deviation
        upper_interval = center + mean_deviation
    elif stat_measure == 'median':
        # center = np.median(x[key_string])
        median_deviation = mean_deviation * \
            np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1))
        lower_interval = center - median_deviation
        upper_interval = center + median_deviation

    result = {
        key_estimator_string: center,
        key_estimator_string + '_conf_interval_lower': lower_interval,
        key_estimator_string + '_conf_interval_upper': upper_interval}
    results = pd.Series(result)
    return results

# %%


def generateStatsDataframe(
    df_all: List[dict] = None,
    stat_measures: List[str] = ['mean', 'median'],
    instance_list: List[str] = None,
    parameters_names: List[str] = None,
    metrics: List[str] = ['min_energy', 'perf_ratio',
                          'success_prob', 'rtt', 'mean_time', 'inv_perf_ratio'],
    results_path: str = None,
    use_raw_dataframes: bool = False,
    confidence_level: float = 68,
    bootstrap_iterations: int = 1000,
    save_pickle: bool = True,
    lower_bounds: List[float] = None,
    upper_bounds: List[float] = None,
    prefix: str = '',
) -> pd.DataFrame:
    '''
    Function to generate statistics from the aggregated dataframe

    Args:
    TODO review parameters list
        df_all: List of dictionaries containing the aggregated dataframe
        stat_measures: List of statistics to be calculated
        instance_list: List of instances to be considered
        parameters_names: List of parameters' names
        resource_list: List of resources to be considered
        results_path: Path to the directory containing the results
        use_raw_dataframes: Boolean indicating whether to use the raw data for generating the aggregated dataframe
        confidence_level: Confidence level for the confidence intervals in bootstrapping
        bootstrap_iterations: Number of bootstrap iterations
        save_pickle: Boolean indicating whether to save the aggregated pickle
        ocean_df_flag: Boolean indicating whether to use the ocean dataframe
        suffix: Suffix to be added to the pickle name
    '''
    if df_all is None:
        print("Please provide the aggregated dataframe")
        return None
    # Split large dataframe such that we can compute the statistics and confidence interval for each metric across the instances
    # TODO This can be improved by the lambda function version of the approach defining an input parameter for the function as a dictionary. Currently this is too slow
    # Not succesful reimplementation of the operation above
    # df_stats_all = df_results_all.set_index(
    #     'instance').groupby(parameters + ['boots']
    #     ).apply(lambda s: pd.Series({
    #         stat_measure + '_' + metric : conf_interval(s,metric, stat_measure) for metric in metrics_list for stat_measure in stat_measures})
    #     ).reset_index()

    # Create filename
    df_name = prefix + 'df_stats.pkl'
    df_path = os.path.join(results_path, df_name)
    if os.path.exists(df_path):
        df_all_stats = pd.read_pickle(df_path)
    else:
        df_all_stats = pd.DataFrame()

    resources = ['boots']

    if all([str(stat_measure) + '_' + metric + '_conf_interval_' + limit in df_all_stats.columns for stat_measure in stat_measures for metric in metrics for limit in ['lower', 'upper']]) and not use_raw_dataframes:
        pass
    else:
        df_all_groups = df_all[
            df_all['instance'].isin(instance_list)
        ].set_index(
            'instance'
        ).groupby(
            parameters_names + resources
        )
        # Remove all groups with fewer than a single instance
        df_filtered = df_all_groups.filter(lambda x: len(x) > 1)
        df_groups = df_filtered.groupby(
            parameters_names + resources
        )
        dataframes = []
        # This function could resemble what is done inside of seaborn to bootstrap everything https://github.com/mwaskom/seaborn/blob/77e3b6b03763d24cc99a8134ee9a6f43b32b8e7b/seaborn/regression.py#L159
        counter = 0
        for metric in metrics:
            for stat_measure in stat_measures:

                df_all_estimator = df_groups.apply(
                    conf_interval,
                    key_string=metric,
                    stat_measure=stat_measure,
                    confidence_level=confidence_level,
                    bootstrap_iterations=bootstrap_iterations,
                )
                dataframes.append(df_all_estimator)

                # Save intermediate file after 10 metric x stat_measures computations
                if counter % 10 == 0:
                    with open(df_path + '.pickle', 'wb') as f:
                        pickle.dump(dataframes, f)
                counter += 1
        if all([len(i) == 0 for i in dataframes]):
            print('No dataframes to merge')
            return None

        df_all_stats = pd.concat(dataframes, axis=1).reset_index()

    df_stats = df_all_stats.copy()

    for stat_measure in stat_measures:
        for key, value in lower_bounds.items():
            if str(stat_measure) + '_' + key + '_conf_interval_lower' in df_stats.columns:
                df_copy = df_stats.loc[:, (str(
                    stat_measure) + '_' + key + '_conf_interval_lower')].copy()
                df_copy.clip(lower=value, inplace=True)
                df_stats.loc[:, (str(stat_measure) + '_' +
                                 key + '_conf_interval_lower')] = df_copy
        for key, value in upper_bounds.items():
            if str(stat_measure) + '_' + key + '_conf_interval_upper' in df_stats.columns:
                df_copy = df_stats.loc[:, (str(
                    stat_measure) + '_' + key + '_conf_interval_upper')].copy()
                df_copy.clip(upper=value, inplace=True)
                df_stats.loc[:, (str(stat_measure) + '_' +
                                 key + '_conf_interval_upper')] = df_copy

    # TODO Implement resource_factor as in generate_ws_dataframe.py
    if 'replica' in parameters_names:
        df_stats['reads'] = df_stats['sweep'] * \
            df_stats['replica'] * df_stats['boots']
    elif 'rep' in parameters_names:
        df_stats['reads'] = df_stats['swe'] * \
            df_stats['rep'] * df_stats['boots']
    else:
        df_stats['reads'] = df_stats['sweep'] * df_stats['boots']

    if save_pickle:
        # df_stats = cleanup_df(df_stats)
        print(df_path)
        df_stats.to_pickle(df_path)

    return df_stats

# %%
# Join all the results in a single dataframe
