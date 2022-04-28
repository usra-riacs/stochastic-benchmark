# %%
# Import the Dwave packages dimod and neal
import functools
import itertools
import os
import pickle
import time
import random
from typing import List, Union

import dimod
# Import Matplotlib to edit plots
import matplotlib.pyplot as plt
import neal
import networkx as nx
# Import numpy and scipy for certain numerical calculations below
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from pysa.sa import Solver
from scipy import sparse, stats

from plotting import *
from retrieve_data import *
from do_dneal import *
from util_benchmark import *

idx = pd.IndexSlice

# %% Specify problem to be solved

instance_class = 'sk'
N = 100  # Number of variables
prefix = "random_n_" + str(N) + "_inst_"
# suffix = '_' + str(N)
suffix = ''

# Specify single instance
instance = 42

# Specify all instances
# instance_list = [i for i in range(20)] + [42]
# training_instance_list = [i for i in range(20)]
instance_list = [i for i in range(20)] + [42]
training_instance_list = [i for i in range(20)]


# %%
# Specify default parameters

# Default parameters
default_sweeps = 1000
total_reads = 1000
float_type = 'float32'
default_reads = 1000
total_reads = 1000
# TODO rename this total_reads parameter, remove redundancy with above
default_Tfactor = 1.0
default_schedule = 'geometric'
default_replicas = 1
default_p_hot = 50.0
default_p_cold = 1.0
default_dict = {
    'schedule': default_schedule,
    'sweeps': default_sweeps,
    'Tfactor': default_Tfactor,
    'boots': default_boots,
}

# Bootstrapping parameters
default_boots = default_reads
boots_list = [int(i*10**j) for j in [1, 2]
              for i in [1, 1.5, 2, 3, 5, 7]] + [int(1e3)]
bootstrap_iterations = 1000
confidence_level = 68


# %%
# Define experiment setting
parameters_list = ['schedule', 'sweeps', 'Tfactor']
ocean_df_flag = True
sweeps_list = [i for i in range(1, 21, 1)] + [
    i for i in range(20, 201, 10)] + [
    i for i in range(200, 501, 20)] + [
    i for i in range(500, 1001, 25)]
Tfactor_list = [default_Tfactor]
# schedules_list = ['geometric', 'linear']
schedules_list = ['geometric']

# TODO can this be generated automatically?
parameters_dict = {
    'schedule': schedules_list,
    'sweeps': sweeps_list,
    'Tfactor': Tfactor_list,
}


# %%
# Create dictionaries for upper and lower bounds of confidence intervals
metrics_list = ['min_energy', 'tts',
                'perf_ratio', 'success_prob', 'mean_time', 'inv_perf_ratio']
lower_bounds = {key: None for key in metrics_list}
lower_bounds['success_prob'] = 0.0
lower_bounds['mean_time'] = 0.0
lower_bounds['inv_perf_ratio'] = EPSILON
upper_bounds = {key: None for key in metrics_list}
upper_bounds['success_prob'] = 1.0
upper_bounds['perf_ratio'] = 1.0


# %%
# Function to update the dataframes
# TODO Remove all the list_* variables and name them as plurals instead
# TODO: Prefix is assumed given directly to the file

def createDnealResultsDataframes(
    df: pd.DataFrame = None,
    instance_list: List[int] = [0],
    parameters_dict: dict = None,
    boots_list: List[int] = [1000],
    data_path: str = None,
    results_path: str = None,
    pickle_path: str = None,
    use_raw_dataframes: bool = False,
    confidence_level: float = 68,
    gap: float = 1.0,
    bootstrap_iterations: int = 1000,
    s: float = 0.99,
    fail_value: float = np.inf,
    save_pickle: bool = True,
    ocean_df_flag: bool = True,
    suffix: str = '',
) -> pd.DataFrame:
    '''
    Function to create the dataframes for the experiments

    Args:
        df: The dataframe to be updated
        instance: The instance number
        boots: The number of bootstraps
        parameters_dict: The parameters dictionary with values as lists
        data_path: The path to the instances information (best found and random solutions)
        results_path: The path to the results
        pickle_path: The path to the pickle files
        use_raw_dataframes: If we want to use the raw data
        use_raw_sample_pickles: If we want to use the raw sample pickles
        overwrite_pickles: If we want to overwrite the pickle files
        confidence_level: The confidence level
        gap: The gap
        bootstrap_iterations: The number of bootstrap iterations
        s: The success probability
        fail_value: The fail value
        save_pickle: If we want to save the pickle files
        ocean_df_flag: If we want to use the ocean dataframe
        suffix: The suffix to add to the dataframe name

    Returns:
        The results dataframe

    '''
    # Remove repeated elements in the parameters_dict values (sets)
    if parameters_dict is not None:
        for i, j in parameters_dict.items():
            parameters_dict[i] = set(j)
    # Create list of parameters
    params = list(parameters_dict.keys())
    # Sort it alphabetically (ignoring uppercase)
    # params.sort(key=str.lower)
    # Check that the parameters are columns in the dataframe
    if df is not None:
        assert all([i in df.columns for i in params])
        # In case that the data is already in the dataframe, return it
        if all([k in df[i].values for (i, j) in parameters_dict.items() for k in j]):
            print('The dataframe has some data for the parameters')
            # The parameters dictionary has lists as values as the loop below makes the concatenation faster than running the loop for each parameter
            cond = [df[k].isin(v).astype(bool)
                    for k, v in parameters_dict.items()]
            cond_total = functools.reduce(lambda x, y: x & y, cond)
            if all(boots in df[cond_total]['boots'].values for boots in boots_list):
                print('The dataframe already has all the data')
                return df

    # Create filename
    # TODO modify filenames inteligently to make it easier to work with
    if len(instance_list) > 1:
        df_name = 'df_results' + suffix + '.pkl'
    else:
        df_name = 'df_results_' + str(instance_list[0]) + suffix + '.pkl'
    df_path = os.path.join(results_path, df_name)

    # If use_raw_dataframes compute the row
    if use_raw_dataframes or not os.path.exists(df_path):
        # TODO Remove all the list_* variables and name them as plurals instead
        list_results = []
        for instance in instance_list:
            random_energy = loadEnergyFromFile(os.path.join(
                data_path, 'random_energies.txt'), prefix + str(instance))
            min_energy = loadEnergyFromFile(os.path.join(
                data_path, 'gs_energies.txt'), prefix + str(instance))
            # We will assume that the insertion order in the keys is preserved (hence Python3.7+ only) and is sorted alphabetically
            parameter_sets = itertools.product(
                *(parameters_dict[Name] for Name in parameters_dict))
            parameter_sets = list(parameter_sets)
            for parameter_set in parameter_sets:
                list_inputs = [instance] + [i for i in parameter_set]
                parameters = dict(zip(params, parameter_set))

                # TODO This can be further generalized to use arbitrary parameter dictionaries
                # A proposal is to change all the input data to something along the way of '_'.join(str(keys) + '_' + str(vals) for keys,vals in parameters.items())
                # 'schedule_geometric_sweep_1000_Tfactor_1'
                if parameters is None:
                    parameters = {
                        'schedule': 'geometric',
                        'sweep': 1000,
                        'Tfactor': 1.0,
                    }

                # Gather instance names
                # TODO: We need to adress renaming problems, one proposal is to be very judicious about the keys order in parameters and be consistent with naming, another idea is sorting them alphabetically before joining them
                dict_pickle_name = prefix + str(instance) + "_" + \
                    '_'.join(str(vals)
                             for vals in parameters.values()) + suffix + ".p"
                df_samples_name = 'df_' + dict_pickle_name + 'kl'
                df_path = os.path.join(pickle_path, df_samples_name)
                if os.path.exists(df_path):
                    # TODO: This loop is wrong as it never forces to rerun the solver
                    try:
                        df_samples = pd.read_pickle(df_path)
                    except (pickle.UnpicklingError, EOFError):
                        print('Error in reading the pickle file' + df_samples_name)
                        os.replace(df_path, df_path + '.bak')

                for boots in boots_list:

                    # TODO Good place to replace with mask and isin1d()
                    # This generated the undersampling using bootstrapping, filtering by all the parameters values
                    if (df is not None) and \
                            (boots in df.loc[(df[list(parameters)] == pd.Series(parameters)).all(axis=1)]['boots'].values):
                        continue
                    else:
                        # print("Generating results for instance:", instance,
                        #   parameters, "boots:", boots)
                        list_outputs = computeResultsList(
                            df=df_samples,
                            random_energy=random_energy,
                            min_energy=min_energy,
                            downsample=boots,
                            bootstrap_iterations=bootstrap_iterations,
                            confidence_level=confidence_level,
                            gap=gap,
                            s=s,
                            fail_value=fail_value,
                            ocean_df_flag=ocean_df_flag,
                        )
                    list_results.append(
                        list_inputs + list_outputs)
        # TODO: Organize these column names to be created automatically from metric list
        df_results = pd.DataFrame(list_results,
                                  columns=[
                                      'instance'] + params + ['boots',
                                                              'min_energy', 'min_energy_conf_interval_lower', 'min_energy_conf_interval_upper',
                                                              'perf_ratio', 'perf_ratio_conf_interval_lower', 'perf_ratio_conf_interval_upper',
                                                              'success_prob', 'success_prob_conf_interval_lower', 'success_prob_conf_interval_upper',
                                                              'tts', 'tts_conf_interval_lower', 'tts_conf_interval_upper',
                                                              'mean_time', 'mean_time_conf_interval_lower', 'mean_time_conf_interval_upper',
                                                              'inv_perf_ratio', 'inv_perf_ratio_conf_interval_lower', 'inv_perf_ratio_conf_interval_upper',
                                                              ])
        if df is not None:
            df_new = pd.concat(
                [df, df_results], axis=0, ignore_index=True)
        else:
            df_new = df_results.copy()

    else:
        print("Loading the raw data dataframe")
        df_new = pd.read_pickle(df_path)

    if save_pickle:
        df_new = cleanup_df(df_new)
        df_new.to_pickle(df_path)
    return df_new


# %%
# Function to generate stats aggregated dataframe
# TODO: this can be generalized by acknowledging that the boots are the resource R
# TODO: This function does not receive the metrics_list, lower_bounds, or upper_bounds or the pickle path. Consider clipping values as well as confidence intervals

def generateStatsDataframe(
    df_all: List[dict] = None,
    stat_measures: List[str] = ['mean', 'median'],
    instance_list: List[str] = None,
    parameters_dict: dict = None,
    resource_list: List[int] = [default_boots],
    data_path: str = None,
    results_path: str = None,
    pickle_path: str = None,
    use_raw_full_dataframe: bool = False,
    use_raw_dataframes: bool = False,
    s: float = 0.99,
    confidence_level: float = 68,
    bootstrap_iterations: int = 1000,
    gap: float = 1.0,
    fail_value: float = None,
    save_pickle: bool = True,
    ocean_df_flag: bool = False,
    suffix: str = '',
) -> pd.DataFrame:
    '''
    Function to generate statistics from the aggregated dataframe

    Args:
        df_all: List of dictionaries containing the aggregated dataframe
        stat_measures: List of statistics to be calculated
        instance_list: List of instances to be considered
        parameters_dict: Dictionary of parameters to be considered, with list as values
        resource_list: List of resources to be considered
        data_path: Path to the instance data (ground state, random, etc)
        results_path: Path to the directory containing the results
        use_raw_full_dataframe: If True, the full dataframe is used
        use_raw_dataframes: Boolean indicating whether to use the raw data for generating the aggregated dataframe
        use_raw_samples_pickles: Boolean indicating whether to use the raw pickles for generating the aggregated pickles
        overwrite_pickles: Boolean indicating whether to overwrite the pickles
        s: The success factor (usually said as RTT within s% probability).
        confidence_level: Confidence level for the confidence intervals in bootstrapping
        bootstrap_iterations: Number of bootstrap iterations
        gap: The threshold for the considering a read successful w.r.t the performance ratio [%].
        fail_value: Value to be used for failed runs
        save_pickle: Boolean indicating whether to save the aggregated pickle
        ocean_df_flag: Boolean indicating whether to use the ocean dataframe
        suffix: Suffix to be added to the pickle name
    '''
    df_all = createDnealResultsDataframes(
        df=df_all,
        instance_list=instance_list,
        parameters_dict=parameters_dict,
        boots_list=resource_list,
        data_path=data_path,
        results_path=results_path,
        pickle_path=pickle_path,
        use_raw_dataframes=use_raw_dataframes,
        s=s,
        confidence_level=confidence_level,
        bootstrap_iterations=bootstrap_iterations,
        gap=gap,
        fail_value=fail_value,
        save_pickle=True,
        ocean_df_flag=ocean_df_flag,
    )
    # Split large dataframe such that we can compute the statistics and confidence interval for each metric across the instances
    # TODO This can be improved by the lambda function version of the approach defining an input parameter for the function as a dictionary. Currently this is too slow
    # Not succesful reimplementation of the operation above
    # df_stats_all = df_results_all.set_index(
    #     'instance').groupby(parameters + ['boots']
    #     ).apply(lambda s: pd.Series({
    #         stat_measure + '_' + metric : conf_interval(s,metric, stat_measure) for metric in metrics_list for stat_measure in stat_measures})
    #     ).reset_index()

    parameters = list(parameters_dict.keys())
    resources = ['boots']
    df_name = 'df_results_stats' + suffix
    df_path = os.path.join(results_path, df_name + suffix + '.pkl')
    if os.path.exists(df_path) and not use_raw_full_dataframe:
        print('Loading the stats dataframe')
        df_all_stats = pd.read_pickle(df_path)
    else:
        df_all_stats = pd.DataFrame()
    if all([stat_measure + '_' + metric + '_conf_interval_' + limit in df_all_stats.columns for stat_measure in stat_measures for metric in metrics_list for limit in ['lower', 'upper']]) and not use_raw_full_dataframe:
        pass
    else:
        df_all_groups = df_all[df_all['instance'].isin(instance_list)].set_index(
            'instance').groupby(parameters + resources)
        dataframes = []
        for metric in metrics_list:
            for stat_measure in stat_measures:
                df_all_stat = df_all_groups.apply(
                    conf_interval, key_string=metric, stat_measure=stat_measure)
                dataframes.append(df_all_stat)

            # if 'mean' in stat_measures:
            #     df_all_mean = df_all_groups.apply(
            #         conf_interval, key_string=metric)
            #     dataframes.append(df_all_mean)
            # if 'median' in stat_measures:
            #     df_all_median = df_all_groups.apply(
            #         conf_interval, key_string=metric)
            #     dataframes.append(df_all_median)

        df_all_stats = pd.concat(dataframes, axis=1).reset_index()

    df_stats = df_all_stats.copy()

    for stat_measure in stat_measures:
        for key, value in lower_bounds.items():
            df_stats[stat_measure + '_' + key + '_conf_interval_lower'].clip(
                lower=value, inplace=True)
        for key, value in upper_bounds.items():
            df_stats[stat_measure + '_' + key + '_conf_interval_upper'].clip(
                upper=value, inplace=True)
    if save_pickle:
        df_stats = cleanup_df(df_stats)
        df_stats.to_pickle(df_path)

    return df_stats


# %%
# Specify and if non-existing, create directories for results

current_path = os.getcwd()
data_path = os.path.join(current_path, '../data/' + instance_class + '/')
if not(os.path.exists(data_path)):
    print('Data directory ' + data_path +
          ' does not exist. We will create it.')
    os.makedirs(data_path)

dneal_results_path = os.path.join(data_path, 'dneal/')
if not(os.path.exists(dneal_results_path)):
    print('Dwave-neal results directory ' + dneal_results_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_results_path)

dneal_pickle_path = os.path.join(dneal_results_path, 'pickles/')
if not(os.path.exists(dneal_pickle_path)):
    print('Dwave-neal pickles directory' + dneal_pickle_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_pickle_path)

pysa_results_path = os.path.join(data_path, 'pysa/')
if not(os.path.exists(pysa_results_path)):
    print('PySA results directory ' + pysa_results_path +
          ' does not exist. We will create it.')
    os.makedirs(pysa_results_path)

pysa_pickle_path = os.path.join(pysa_results_path, 'pickles/')
if not(os.path.exists(pysa_pickle_path)):
    print('PySA pickles directory' + pysa_pickle_path +
          ' does not exist. We will create it.')
    os.makedirs(pysa_pickle_path)

instance_path = os.path.join(data_path, 'instances/')
if not(os.path.exists(instance_path)):
    print('Instances directory ' + instance_path +
          ' does not exist. We will create it.')
    os.makedirs(instance_path)

plots_path = os.path.join(data_path, 'plots/')
if not(os.path.exists(plots_path)):
    print('Plots directory ' + plots_path +
          ' does not exist. We will create it.')
    os.makedirs(plots_path)

if ocean_df_flag:
    results_path = dneal_results_path
else:
    results_path = pysa_results_path

# %%
# Import single instance datafile

df_name_single_instance = "df_results_" + str(instance) + suffix + ".pkl"
df_path_single_instance = os.path.join(results_path, df_name_single_instance)
if os.path.exists(df_path_single_instance):
    df_single_instance = pd.read_pickle(df_path_single_instance)
else:
    df_single_instance = None


# %%
# Import all instances datafile

df_name = "df_results" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
if os.path.exists(df_path):
    df_results_all = pd.read_pickle(df_path)
else:
    df_results_all = None

# %%
# Generate stats results
use_raw_full_dataframe = True
use_raw_dataframes = True
# TODO: this function assume we want all the combinatios of the parameters, while in reality, we might want to use those in a list
stat_measures = ['mean', 'median']
df_results_all_stats = generateStatsDataframe(
    df_all=None,
    stat_measures=stat_measures,
    instance_list=training_instance_list,
    parameters_dict=parameters_dict,
    resource_list=boots_list,
    data_path=data_path,
    results_path=results_path,
    pickle_path=dneal_pickle_path,
    use_raw_full_dataframe=use_raw_full_dataframe,
    use_raw_dataframes=use_raw_dataframes,
    s=s,
    confidence_level=confidence_level,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
    ocean_df_flag=ocean_df_flag,
)


# %%
# Gather all the data for the best tts of the ensemble training set for each instance
best_ensemble_sweeps = []
df_list = []
use_raw_dataframes = False
for stat_measure in stat_measures:
    best_sweep = df_results_all_stats[df_results_all_stats['boots'] == default_boots].nsmallest(
        1, stat_measure + '_tts')['sweeps'].values[0]
    best_ensemble_sweeps.append(best_sweep)
parameters_best_ensemble_dict = {
    'schedule': schedules_list,
    'sweeps': best_ensemble_sweeps,
    'Tfactor': [default_Tfactor],
}
# %%
# Possible presentation plot!
# Generate plots for TTS of ensemble together with single instance (42)
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_single_instance,
    x_axis='sweeps',
    y_axis='tts',
    ax=ax,
    dict_fixed={'schedule': 'geometric'},
    list_dicts=[{'instance': 42, 'boots': j}
                for j in [100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=True,
    colormap=plt.cm.Dark2,
)
plot_1d_singleinstance_list(
    df=df_results_all_stats,
    x_axis='sweeps',
    y_axis='median_tts',
    ax=ax,
    dict_fixed={'schedule': 'geometric'},
    list_dicts=[{'boots': j}
                for j in [100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=True,
    colors=['colormap']
)
# %%
# Generate plots for performance ratio of ensemble vs sweeps
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_results_all_stats,
    x_axis='sweeps',
    y_axis='median_perf_ratio',
    ax=ax,
    dict_fixed={'schedule': 'geometric'},
    list_dicts=[{'boots': j}
                for j in boots_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    ylim=[0.975, 1.0025],
    colors=['colormap'],
)
# %%
# Generate plots for performance ratio of ensemble vs reads
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_results_all_stats,
    x_axis='reads',
    y_axis='median_perf_ratio',
    ax=ax,
    dict_fixed={'schedule': 'geometric', 'sweeps': 500},
    list_dicts=[{'boots': j}
                for j in boots_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    marker='*',
    use_colorbar=False,
    ylim=[0.975, 1.0025],
    colors=['colormap'],
)
# %%
# Generate plots for performance ratio of ensemble vs reads
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_results_all_stats,
    x_axis='reads',
    y_axis='median_perf_ratio',
    ax=ax,
    dict_fixed={'schedule': 'geometric'},
    list_dicts=[{'sweeps': j}
                for j in best_ensemble_sweeps],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    ylim=[0.975, 1.0025],
    colors=['colormap'],
)
# %%
# Generate plots for performance ratio of ensemble vs reads
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_results_all_stats,
    x_axis='reads',
    y_axis='mean_success_prob',
    ax=ax,
    dict_fixed={'schedule': 'geometric'},
    list_dicts=[{'boots': j}
                for j in boots_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    # ylim=[0.9, 1.005],
    # xlim=[1e2, 5e4],
    colors=['colormap'],
)
# %%
# Create virtual best and virtual worst columns
# TODO This can be generalized as using as groups the parameters that are not dependent of the metric (e.g., schedule) or that signify different solvers
# TODO This needs to be functionalized
stale_parameters = ['schedule', 'Tfactor']

df_name = "df_results_virt" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
use_raw_dataframes = True
if use_raw_dataframes or os.path.exists(df_path) is False:
    df_virtual_all = df_results_all.groupby(
        ['reads']
    ).apply(lambda s: pd.Series({
            'virt_best_tts': np.nanmin(s['tts']),
            'virt_best_perf_ratio': np.nanmax(s['perf_ratio']),
            'virt_best_inv_perf_ratio': np.nanmin(s['inv_perf_ratio']),
            'virt_best_success_prob': np.nanmax(s['success_prob']),
            'virt_best_mean_time': np.nanmin(s['mean_time']),
            'virt_worst_perf_ratio': np.nanmin(s['perf_ratio']),
            'virt_worst_inv_perf_ratio': np.nanmax(s['inv_perf_ratio'])
            })
            ).reset_index()
    df_virtual_best_max = df_virtual_all[
        ['reads',
         'virt_best_perf_ratio',
         'virt_best_success_prob']
    ].sort_values('reads'
                  ).expanding(min_periods=1).max()

    # This is done as the virtual worst counts the worst case, computed as the minimum from last read to first
    df_virtual_worst_max = df_virtual_all[
        ['reads',
         'virt_worst_perf_ratio']
    ].sort_values('reads', ascending=False
                  ).expanding(min_periods=1).min()

    df_virtual_best_min = df_virtual_all[
        ['reads',
         'virt_best_tts',
         'virt_best_mean_time',
         'virt_best_inv_perf_ratio']
    ].sort_values('reads'
                  ).expanding(min_periods=1).agg({
                      'reads': np.max,
                      'virt_best_tts': np.min,
                      'virt_best_mean_time': np.min,
                      'virt_best_inv_perf_ratio': np.min}
    )
    # df_virtual_best_min['reads'] = df_virtual_all[
    #     ['reads', 'schedule']
    # ].sort_values(['schedule','reads'])['reads']

    df_virtual_worst_min = df_virtual_all[
        ['reads',
         'virt_worst_inv_perf_ratio']
    ].sort_values('reads', ascending=False
                  ).expanding(min_periods=1).agg({
                      'reads': np.min,
                      'virt_worst_inv_perf_ratio': np.max}
    )
    df_virtual_best_min['reads'] = df_virtual_all[
        ['reads']
    ].sort_values('reads', ascending=False)['reads']

    df_virtual_best = df_virtual_best_max.merge(
        df_virtual_worst_max,
        on='reads',
        how='left')
    df_virtual_best = df_virtual_best.merge(
        df_virtual_best_min,
        on='reads',
        how='left')
    df_virtual_best = df_virtual_best.merge(
        df_virtual_worst_min,
        on='reads',
        how='left')

    # Dirty workaround to compute virtual best perf_ratio as commended by Davide, several points: 1) the perf_ratio is computed as the maximum (we are assuming we care about the max) of for each instance for each read, 2) the median of this idealized solver (that has the best parameters for each case) across the instances is computed
    params = ['schedule', 'sweeps', 'Tfactor']

    df_virtual_best = df_virtual_best.merge(
        df_results_all.set_index(
            params
        ).groupby(['instance', 'reads']
                  )['perf_ratio'].max().reset_index().set_index(
            ['instance']
        ).groupby(['reads']
                  ).median().reset_index().sort_values(
            ['reads']
        ).expanding(min_periods=1).max(),
        on=['reads'],
        how='left')

    recipe_lazy = df_results_all_stats[params + ['median_perf_ratio', 'reads']].set_index(
        params
    ).groupby(['reads']
              ).idxmax()

    df_virtual_best = df_virtual_best.merge(
        df_results_all_stats[params + ['median_perf_ratio', 'reads']].set_index(
            params
        ).groupby(['reads']
                  ).max().reset_index().rename(columns={'median_perf_ratio': 'lazy_perf_ratio'}))

    recipe_mean_best_params = df_results_all.set_index(
        params
    ).groupby(['instance', 'reads']
              )['perf_ratio'].idxmax().apply(pd.Series).reset_index().set_index(
        ['instance']
    ).groupby(['reads']
              ).mean().rename(columns={1: 'sweeps', 2: 'Tfactor'})

    recipe_mean_best_params['sweeps'] = recipe_mean_best_params['sweeps'].apply(
        lambda x: take_closest(sweeps_list, x))

    recipe_mean_best_params['Tfactor'] = recipe_mean_best_params['Tfactor'].apply(
        lambda x: take_closest(Tfactor_list, x))

    # Project the reads to the closest value in boots_list*sweeps
    recipe_mean_best_params['boots'] = recipe_mean_best_params.index / \
        recipe_mean_best_params['sweeps']
    recipe_mean_best_params['boots'] = recipe_mean_best_params['boots'].apply(
        lambda x: take_closest(boots_list, x))
    recipe_mean_best_params.index = recipe_mean_best_params['boots'] * \
        recipe_mean_best_params['sweeps']
    recipe_mean_best_params.index.rename('reads', inplace=True)

    recipe_mean_best_params['params'] = list(zip(
        ['geometric']*len(recipe_mean_best_params),
        recipe_mean_best_params['sweeps'],
        recipe_mean_best_params['Tfactor'],
        recipe_mean_best_params.index))

    df_virtual_best = df_virtual_best.merge(df_results_all_stats.set_index(
        params + ['reads']
    ).loc[pd.MultiIndex.from_tuples(recipe_mean_best_params['params']
                                    )]['median_perf_ratio'].reset_index().rename(columns={'level_3': 'reads', 'median_perf_ratio': 'mean_param_perf_ratio'}
                                                                                 ).drop(columns=['level_0', 'level_1', 'level_2']),
        on=['reads'],
        how='left')

    recipe_median_best_params = df_results_all.set_index(
        params
    ).groupby(['instance', 'reads']
              )['perf_ratio'].idxmax().apply(pd.Series).reset_index().set_index(
        ['instance']
    ).groupby(['reads']
              ).median().rename(columns={1: 'sweeps', 2: 'Tfactor'})

    recipe_median_best_params['sweeps'] = recipe_median_best_params['sweeps'].apply(
        lambda x: take_closest(sweeps_list, x))

    recipe_median_best_params['Tfactor'] = recipe_median_best_params['Tfactor'].apply(
        lambda x: take_closest(Tfactor_list, x))

    # Project the reads to the closes value in boots_list*sweeps
    recipe_median_best_params['boots'] = recipe_median_best_params.index / \
        recipe_median_best_params['sweeps']
    recipe_median_best_params['boots'] = recipe_median_best_params['boots'].apply(
        lambda x: take_closest(boots_list, x))
    recipe_median_best_params.index = recipe_median_best_params['boots'] * \
        recipe_median_best_params['sweeps']
    recipe_median_best_params.index.rename('reads', inplace=True)

    recipe_median_best_params['params'] = list(zip(
        ['geometric']*len(recipe_median_best_params),
        recipe_median_best_params['sweeps'],
        recipe_median_best_params['Tfactor'],
        recipe_median_best_params.index))

    df_virtual_best = df_virtual_best.merge(df_results_all_stats.set_index(
        params + ['reads']
    ).loc[pd.MultiIndex.from_tuples(recipe_median_best_params['params']
                                    )]['median_perf_ratio'].reset_index().rename(columns={'level_3': 'reads', 'median_perf_ratio': 'median_param_perf_ratio'}
                                                                                 ).drop(columns=['level_0', 'level_1', 'level_2']),
        on=['reads'],
        how='left')

    df_virtual_best['inv_lazy_perf_ratio'] = 1 - \
        df_virtual_best['lazy_perf_ratio'] + EPSILON

    df_virtual_best['inv_perf_ratio'] = 1 - \
        df_virtual_best['perf_ratio'] + EPSILON

    df_virtual_best = cleanup_df(df_virtual_best)
    df_virtual_best.to_pickle(df_path)
else:
    df_virtual_best = pd.read_pickle(df_path)
    df_virtual_best = cleanup_df(df_virtual_best)

window_average = 60
# df_rolled = df_virtual_best.sort_index(ascending=True).rolling(window=window_average, min_periods=0).mean()
df_rolled = df_virtual_best.sort_index(ascending=True).ewm(alpha=0.9).mean()
df_expand = df_virtual_best.sort_index(
    ascending=True).expanding(min_periods=1).max()
df_virtual_best['soft_lazy_perf_ratio'] = df_rolled['lazy_perf_ratio']
df_virtual_best['soft_median_param_perf_ratio'] = df_rolled['median_param_perf_ratio']
df_virtual_best['soft_mean_param_perf_ratio'] = df_expand['mean_param_perf_ratio']
