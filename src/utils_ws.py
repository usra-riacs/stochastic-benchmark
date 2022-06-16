# %%
import datetime
import itertools
import os
import random
from bisect import bisect_left
from math import hypot
from typing import List, Tuple, Union
import sys
from matplotlib.pyplot import xlabel, xscale
import numpy as np
import pandas as pd

idx = pd.IndexSlice
EPSILON = 1e-10

# %%
# Function to generate logarithmically spaced integers
def gen_log_space(
    lower: int,
    upper: int,
    n: int,
    )->List[int]:
    '''
    Function to generate logarithmically spaced integers without repeats.

    Args:
        lower (int): lower bound of the range
        upper (int): upper bound of the range
        n (int): number of integers to generate

    Returns:
        list: list of integers

    '''
    result = [lower]
    if n > 1:  # just a check to avoid ZeroDivisionError
        ratio = ((float(upper) - float(lower)) /
                 result[-1]) ** (1.0/(n-len(result)))
    while len(result) < n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = ((float(upper) - float(lower)) /
                     result[-1]) ** (1.0/(n-len(result)))
    # round and return np.uint64 array
    return np.array(list(map(lambda x: round(x), result)), dtype=np.uint64)



# %%
# Function to interpolate dataframes into new index
def interp(
    df: pd.DataFrame,
    new_index: list,
    )->pd.DataFrame:
    '''
    Return a new DataFrame with all columns values interpolated
    to the new_index values.

    Args:
        df (pd.DataFrame): DataFrame to interpolate
        new_index (list): list of new index values

    Returns:
        df_out (pd.DataFrame): new DataFrame with interpolated values
    '''
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        col = pd.to_numeric(col, errors='ignore')
        if np.issubdtype(col, int) or np.issubdtype(col, float):
            df_out[colname] = np.interp(new_index, df.index, col)
        else:
            print(colname)
            df_out[colname] = col

    return df_out


# %%
# Function to take closest element in a list to a given value

def take_closest(
    myList: list, 
    myNumber,
    ):
    '''
    Function to take closest element in a list to a given value
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.

    Args:
        myList (list): list of values
        myNumber: value to find closest to

    Returns:
        closest value in myList to myNumber

    '''
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


# %%
# Function to interpolate dataframes across a resource column


def interpolate_df(
    dataframe: pd.DataFrame = None,
    resource_column: str = 'reads',
    prefix: str = '',
    parameters_dict: dict = None,
    default_boots: int = 1000,
    minimum_boots: int = 1,
    resource_proportional_parameters: list = ['sweep', 'replica'],
    idx=pd.IndexSlice,
    results_path: str = None,
    save_pickle: bool = True,
    overwrite_pickles: bool = True,
    all_datapoints: bool = False,
    resource_values: list = None,
)->pd.DataFrame:
    '''
    Function to interpolate dataframes across a resource column.

    Args:
        dataframe (pd.DataFrame): DataFrame to interpolate
        resource_column (str): column to interpolate across
        prefix (str): prefix to add to the interpolated dataframe names
        parameters_dict (dict): dictionary of parameters to interpolate across
        default_boots (int): number of bootstrap iterations to use if no
            bootstrap iterations are specified in the dataframe
        minimum_boots (int): minimum number of bootstrap iterations to use
            if no bootstrap iterations are specified in the dataframe
        resource_proportional_parameters (list): list of parameters to
            interpolate across the resource_column if resource_proportional_parameters
            is not None. Otherwise, the resource_column will be interpolated
            across the resource_column.
        idx (pd.IndexSlice): index to use for the interpolated dataframe
        results_path (str): path to save the interpolated dataframes to
        save_pickle (bool): if True, the interpolated dataframes will be saved
            to the results_path.
        overwrite_pickles (bool): if True, the interpolated dataframes will be
            saved to the results_path. If False, the interpolated dataframes
            will be saved to the results_path with a timestamp.
        all_datapoints (bool): if True, all datapoints will be used to
            interpolate. If False, only the datapoints with the lowest
            resource value will be used.
        resource_values (list): list of resource values to interpolate across.
            If None, the resource_column will be interpolated across the
            resource_column.

    Returns:
        df_out (pd.DataFrame): interpolated dataframe
    '''

    if dataframe is None:
        print('Error: Dataframe is None')
        return None
    if len(dataframe) == 0:
        print('Error: Dataframe is empty')
        return None
    df = dataframe.copy()
    parameter_names = list(parameters_dict.keys())
    parameter_sets = itertools.product(
        *(parameters_dict[Name] for Name in parameters_dict))
    parameter_sets = list(parameter_sets)
    r_indices = []
    if resource_column not in df.columns:
        df[resource_column] = df['boots']
        for r_parameters in resource_proportional_parameters:
            if r_parameters in parameter_names:
                df[resource_column] *= df[r_parameters]
    if resource_values is None:
        if all_datapoints:
            resource_values = df[resource_column].values
        else:
            resource_values = gen_log_space(min(df[resource_column].values), max(
                df[resource_column].values), default_boots // 10)
    resource_values = np.sort(np.unique(resource_values))
    instances = [0]
    if 'instance' in df.columns:
        instances = df['instance'].unique().tolist()
    df_index = df.set_index(parameter_names).sort_index().copy()
    for r_parameters in resource_proportional_parameters:
        if r_parameters in parameter_names:
            r_indices.append(parameter_names.index(r_parameters))

    dataframes = []
    for instance in instances:
        df_name_partial = prefix.rsplit(
            '.', 1)[0] + str(instance) + '_partial.pkl'
        df_path_partial = os.path.join(results_path, df_name_partial)
        if os.path.exists(df_path_partial) and not overwrite_pickles:
            print('Loaded partial dataframe from file')
            df_interpolate = pd.read_pickle(df_path_partial)
            dataframes_instance = [df_interpolate]
        else:
            dataframes_instance = []
            # for parameter_set in parameter_sets:
            #     if parameter_set not in df_index.index.to_list():
            #         print('Parameter set', parameter_set, 'not found')
            #         continue  # For each parameter setting remove repeated reads
            # df_values = df_index.loc[idx[parameter_set]].copy()
            if 'instance' in df.columns:
                df_values = df_index.loc[df_index['instance'] == instance].copy(
                )
            else:
                df_values = df_index.copy()
            for parameter_set in set(df_values.index.to_list()):
                df_original = df_values.loc[idx[parameter_set]].copy()
                # Reading the parameter columns
                for key, value in zip(parameter_names, parameter_set):
                    df_original[key] = value
                if 'params' in df_original.columns:
                    df_original.drop(columns=['params'], inplace=True)
                if len(df_original) == 0:
                    print('No data for parameter set',
                          parameter_set, 'with instance', instance)
                    continue
                resource_factor = 1
                for r_index in r_indices:
                    resource_factor *= parameter_set[r_index]
                    # resource_factor *= index[r_index]
                # Set interpolation points for the responses at all the relevant reads values
                maximum_boots = df_original['boots'].max()
                interpolate_resource = resource_values[
                    np.where(
                        (resource_values <= take_closest(resource_values, maximum_boots*resource_factor)) &
                        (resource_values >= take_closest(
                            resource_values, minimum_boots*resource_factor))
                    )
                ]
                if all_datapoints:
                    # Create a dataframe with the interesting reads as index and all the columns
                    dummy_df = pd.DataFrame(
                        np.NaN,
                        index=interpolate_resource,
                        columns=df_index.columns
                    )
                    dummy_df.drop(columns=resource_column, inplace=True)
                    # Fill out the values that we have certain
                    dummy_df.update(df_original.set_index(resource_column))
                    df_interpolate = dummy_df.copy()
                    # Interpolate for all the other values (without extrapolating)
                    df_interpolate = df_interpolate.interpolate(
                        method='linear', limit_area='inside'
                    ).dropna(how='all').reset_index().rename(
                        columns={'index': resource_column})
                else:
                    df_interpolate = interp(df_original.set_index(
                        resource_column).sort_index(), interpolate_resource)
                    df_interpolate = df_interpolate.reset_index().rename(
                        columns={'index': resource_column})
                # Computing the boots column
                df_interpolate['boots'] = df_interpolate[resource_column] / \
                    resource_factor
                if 'instance' in df.columns:
                    df_interpolate['instance'] = instance
                dataframes_instance.append(df_interpolate)
            df_interpolate = pd.concat(
                dataframes_instance).reset_index(drop=True)
            df_interpolate.to_pickle(df_path_partial)

    if len(instances) == 1:
        dataframes = dataframes_instance
    else:
        for instance in instances:
            df_name_partial = prefix.rsplit(
                '.', 1)[0] + str(instance) + '_partial.pkl'
            df_path_partial = os.path.join(results_path, df_name_partial)
            df_interpolate = pd.read_pickle(df_path_partial)
            dataframes.append(df_interpolate)

    if all([len(i) == 0 for i in dataframes]):
        print('No dataframes to merge')
        return None
    df_interpolated = pd.concat(dataframes).reset_index(drop=True)
    if save_pickle:
        df_name_interpolated = prefix.rsplit('.', 1)[0] + '_interp.pkl'
        df_path_interpolated = os.path.join(results_path, df_name_interpolated)
        df_interpolated.to_pickle(df_path_interpolated)
    return df_interpolated


# %%
# Function to perform percentile aggregation


def percentile(n):
    def percentile_(x):
        return np.nanpercentile(x, n)
    percentile_.__name__ = '%s' % n
    return percentile_



# %%
# Function to perform alternative processing of progress dataframes


def process_df_progress(
    df_progress: pd.DataFrame = None,
    compute_metrics: list = ['perf_ratio'],
    stat_measures: list = ['mean'],
    maximizing: bool = True,
    df_progress_name: str = 'df_progress.pkl',
    results_path: str = None,
    use_raw_dataframes: bool = True,
    save_pickle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Function to process progress dataframes, computing statistics across experiments and reporting best behavior for each budget.

    Args:
        df_progress: Dataframe containing progress data
        compute_metrics: List of metrics to compute
        stat_measures: List of statistics to compute
        maximizing: Boolean indicating whether to maximize or minimize the metric

    Returns:
        Tuple of dictionaries of which there are
            df_progress_processed: Processed dataframe
            df_progress_end: Processed dataframe


    '''

    for stat_measure in stat_measures:
        if type(stat_measure) == int:
            stat_measure = 'percentile('+stat_measure+')'

    experiment_setting = ['R_budget', 'R_explor', 'tau']
    individual_run = experiment_setting.copy()

    if maximizing:
        opt_stats = ['max', 'idxmax']
        del_str = 'max_'
    else:
        opt_stats = ['min', 'idxmin']
        del_str = 'min_'

    df_progress_name_end = df_progress_name.rsplit('.', 1)[0] + '_end.pkl'
    df_progress_path_end = os.path.join(results_path, df_progress_name_end)
    if not use_raw_dataframes and os.path.exists(df_progress_path_end):
        df_progress_end = pd.read_pickle(df_progress_path_end)
    else:
        if df_progress is None:
            return None
        if 'experiment' in df_progress.columns:
            individual_run += ['experiment']
        df_progress_end = df_progress[
            individual_run + compute_metrics + ['cum_reads']
        ].loc[
            df_progress.sort_values(
                ['cum_reads'],
                ascending=False,
            ).groupby(
                individual_run
            )['cum_reads'].idxmax()
        ].copy()

        for compute_metric in compute_metrics:
            if 'inv_' in compute_metric:
                df_progress[compute_metric.replace(
                    'inv_', '')] = 1 - df_progress[compute_metric] + EPSILON
                expanding_metric = compute_metric
                df_progress_end[compute_metric.replace(
                    'inv_', '')] = 1 - df_progress_end[compute_metric] + EPSILON
            else:
                position = compute_metric.index('perf_ratio')
                df_progress[compute_metric[:position] + 'inv_' +
                            compute_metric[position:]] = df_progress[compute_metric] + EPSILON
                df_progress_end[compute_metric[:position] + 'inv_' +
                                compute_metric[position:]] = df_progress_end[compute_metric] + EPSILON
                expanding_metric = compute_metric[:position] + \
                    'inv_' + compute_metric[position:]
        df_progress['best_inv_perf_ratio'] = df_progress.sort_values(
            ['cum_reads', 'R_budget']
        ).expanding(min_periods=1).min()[expanding_metric]
        df_progress['best_perf_ratio'] = 1 - \
            df_progress['best_inv_perf_ratio'] + EPSILON

        if 'f_explor' not in df_progress:
            df_progress['f_explor'] = df_progress['R_explor'] / \
                df_progress['R_budget']

        # df_progress = cleanup_df(df_progress)

        if 'f_explor' not in df_progress_end:
            df_progress_end['f_explor'] = df_progress_end['R_explor'] / \
                df_progress_end['R_budget']
        # df_progress_end = cleanup_df(df_progress_end)

    if save_pickle:
        df_progress_end.to_pickle(df_progress_path_end)

    df_progress_best = df_progress_end.groupby(
        experiment_setting
    )[
        compute_metrics
    ].agg(stat_measures).groupby(
        ['R_budget']
    ).agg(
        opt_stats
    ).copy()

    df_progress_best.columns = ["_".join(reversed(pair)).replace(
        del_str, '') for pair in df_progress_best.columns]
    df_progress_best.reset_index(inplace=True)

    for stat_measure in stat_measures:
        for compute_metric in compute_metrics:
            if 'inv_' in compute_metric:
                df_progress_best[stat_measure + '_' + compute_metric.replace('inv_', '')] = 1 - \
                    df_progress_best[stat_measure +
                                     '_' + compute_metric] + EPSILON
            else:
                position = compute_metric.index('perf_ratio')
                df_progress_best[stat_measure + '_' + compute_metric[:position] + 'inv_' +
                                 compute_metric[position:]] = 1 - df_progress_best[stat_measure + '_' + compute_metric] + EPSILON

    # df_progress_best = cleanup_df(df_progress_best)

    for column in df_progress_best.columns:
        if 'idx' in column:
            df_progress_best = df_progress_best.merge(df_progress_best[
                column
            ].apply(pd.Series).rename(
                columns={i: experiment_setting[i]
                         for i in range(len(experiment_setting))}
            ), on='R_budget', how='left')

    if 'f_explor' not in df_progress_best.columns:
        df_progress_best['f_explor'] = df_progress_best['R_explor'] / \
            df_progress_best['R_budget']

    return df_progress_best, df_progress_end

