# %%
# Import the Dwave packages dimod and neal
import functools
import itertools
import os
import pickle
import random
import time
from ctypes.wintypes import DWORD
from gc import collect
from typing import List, Union
from unicodedata import category

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
from util_benchmark import *

# from do_pysa import *

idx = pd.IndexSlice

EPSILON = 1e-10

# %%
# Specify instance 42
N = 100  # Number of variables
instance = 42
np.random.seed(instance)  # Fixing the random seed to get the same result
J = np.random.rand(N, N)
# We only consider upper triangular matrix ignoring the diagonal
J = np.triu(J, 1)
h = np.random.rand(N)
# %%
# Specify and if non-existing, create directories for results
current_path = os.getcwd()
data_path = os.path.join(current_path, '../data/sk/')
if not(os.path.exists(data_path)):
    print('Data directory ' + data_path +
          ' does not exist. We will create it.')
    os.makedirs(data_path)

dneal_results_path = os.path.join(data_path, 'dneal/')
if not(os.path.exists(dneal_results_path)):
    print('Dwave-neal results directory ' + dneal_results_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_results_path)

dneal_pickles_path = os.path.join(dneal_results_path, 'pickles/')
if not(os.path.exists(dneal_pickles_path)):
    print('Dwave-neal pickles directory' + dneal_pickles_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_pickles_path)

pysa_results_path = os.path.join(data_path, 'pysa/')
if not(os.path.exists(pysa_results_path)):
    print('PySA results directory ' + pysa_results_path +
          ' does not exist. We will create it.')
    os.makedirs(pysa_results_path)

pysa_pickles_path = os.path.join(pysa_results_path, 'pickles/')
if not(os.path.exists(pysa_pickles_path)):
    print('PySA pickles directory' + pysa_pickles_path +
          ' does not exist. We will create it.')
    os.makedirs(pysa_pickles_path)

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

# %%
# Define default values

default_sweeps = 1000
total_reads = 1000
default_boots = default_reads
float_type = 'float32'
default_reads = 1000
total_reads = 1000
# TODO rename this total_reads parameter, remove redundancy with above
default_Tfactor = 1.0
default_schedule = 'geometric'
default_replicas = 1
default_p_hot = 50.0
default_p_cold = 1.0
parameters_list = ['swe', 'rep', 'pcold', 'phot']
suffix = 'P'
ocean_df_flag = False
results_path = pysa_results_path
pickles_path = pysa_pickles_path

# %%
# Create instance 42
model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

prefix = "random_n_" + str(N) + "_inst_"
# TODO: this is prefix for file but we should get a nicer description for the plots
instance_name = prefix + str(instance)

# %%
# Load zipped results if using raw data
overwrite_pickles = False
use_raw_dataframes = False

# %%
# Function to generate samples dataframes or load them otherwise


def createPySASamplesDataframe(
    instance: int = 42,
    parameters: dict = None,
    total_reads: int = 1000,
    pickles_path: str = None,
    use_raw_sample_pickles: bool = False,
    overwrite_pickles: bool = False,
) -> pd.DataFrame:
    '''
    Creates a dataframe with the samples for the pysa algorithm.

    Args:
        instance: The instance to load/create the samples for.
        parameters: The parameters to use for PySA.
        total_reads: The total number of reads to use in PySA.
        pickles_path: The path to the pickle files.
        use_raw_sample_pickles: Whether to use the raw pickles or not.
        overwrite_pickles: Whether to overwrite the pickles or not.

    Returns:
        The dataframe with the samples for the pysa algorithm.
    '''
    # TODO This can be further generalized to use arbitrary parameter dictionaries
    # A proposal is to change all the input data to something along the way of '_'.join(str(keys) + '_' + str(vals) for keys,vals in parameters.items())
    # 'schedule_geometric_sweep_1000_Tfactor_1'
    if parameters is None:
        parameters = {
            'swe': 1000,
            'rep': 1,
            'pcold': 1.0,
            'phot': 50.0,
        }

    # Gather instance names
    # TODO: We need to adress renaming problems, one proposal is to be very judicious about the keys order in parameters and be consistent with naming, another idea is sorting them alphabetically before joining them
    instance_name = prefix + str(instance)
    df_samples_name = instance_name + "_" + \
        '_'.join(str(keys) + '_' + str(vals)
                 for keys, vals in parameters.items()) + ".pkl"
    df_path = os.path.join(pickles_path, df_samples_name)
    if os.path.exists(df_path) and not use_raw_sample_pickles:
        try:
            df_samples = pd.read_pickle(df_path)
        except (pickle.UnpicklingError, EOFError):
            print('Pickle file ' + df_path +
                  ' is corrupted. We will create a new one.')
            os.replace(df_path, df_path + '.bak')
            # TODO: How to jump to other branch of conditional?
    else:
        file_path = os.path.join(instance_path, instance_name + ".txt")

        data = np.loadtxt(file_path, dtype=float)
        M = sparse.coo_matrix(
            (data[:, 2], (data[:, 0], data[:, 1])), shape=(N, N))
        problem = M.A
        problem = problem+problem.T-np.diag(np.diag(problem))

        # Get solver
        solver = Solver(
            problem=problem,
            problem_type='ising',
            float_type=float_type,
        )

        min_temp = 2 * \
            np.min(np.abs(problem[np.nonzero(problem)])
                   ) / np.log(100/parameters['pcold'])
        min_temp_cal = 2*min(sum(abs(i)
                                 for i in problem)) / np.log(100/parameters['pcold'])
        max_temp = 2*max(sum(abs(i)
                             for i in problem)) / np.log(100/parameters['phot'])

        df_samples = solver.metropolis_update(
            num_sweeps=parameters['swe'],
            num_reads=total_reads,
            num_replicas=parameters['rep'],
            update_strategy='random',
            min_temp=min_temp,
            max_temp=max_temp,
            initialize_strategy='random',
            recompute_energy=True,
            sort_output_temps=True,
            parallel=True,  # True by default
            use_pt=True,
            verbose=False,
        )
        df_samples.to_pickle(df_path)

    return df_samples


# %%
# Function to update the dataframes
# TODO Remove all the list_* variables and name them as plurals instead
def createPySAResultsDataframes(
    df: pd.DataFrame = None,
    instance_list: List[int] = [0],
    parameters_dict: dict = None,
    boots_list: List[int] = [1000],
    data_path: str = None,
    results_path: str = None,
    pickles_path: str = None,
    use_raw_dataframes: bool = False,
    use_raw_sample_pickles: bool = False,
    overwrite_pickles: bool = False,
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
        results_path: The path to the results
        pickles_path: The path to the pickle files
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
                df_samples = createPySASamplesDataframe(
                    instance=instance,
                    parameters=parameters,
                    total_reads=total_reads,
                    pickles_path=pickles_path,
                    use_raw_sample_pickles=use_raw_sample_pickles,
                    overwrite_pickles=overwrite_pickles,
                )

                for boots in boots_list:

                    # TODO Good place to replace with mask and isin1d()
                    # This generated the undersampling using bootstrapping, filtering by all the parameters values
                    if (df is not None) and \
                            (boots in df.loc[(df[list(parameters)] == pd.Series(parameters)).all(axis=1)]['boots'].values):
                        continue
                    else:
                        # print("Generating results for instance:", instance,
                        #   "schedule:", schedule, "sweep:", sweep, "boots:", boots)
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
        print("Loading the dataframe")
        df_new = pd.read_pickle(df_path)

    if save_pickle:
        df_new = cleanup_df(df_new)
        df_new.to_pickle(df_path)
    return df_new


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
# Function to generate stats aggregated dataframe
# TODO: this can be generalized by acknowledging that the boots are the resource R


def generateStatsDataframe(
    df_all: List[dict] = None,
    stat_measures: List[str] = ['mean', 'median'],
    instance_list: List[str] = None,
    parameters_dict: dict = None,
    resource_list: List[int] = [default_boots],
    data_path: str = None,
    results_path: str = None,
    pickles_path: str = None,
    use_raw_full_dataframe: bool = False,
    use_raw_dataframes: bool = False,
    use_raw_sample_pickles: bool = False,
    overwrite_pickles: bool = False,
    s: float = 0.99,
    confidence_level: float = 0.68,
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
    df_all = createPySAResultsDataframes(
        df=df_all,
        instance_list=instance_list,
        parameters_dict=parameters_dict,
        boots_list=resource_list,
        data_path=data_path,
        results_path=results_path,
        pickles_path=pickles_path,
        use_raw_dataframes=use_raw_dataframes,
        use_raw_sample_pickles=use_raw_sample_pickles,
        overwrite_pickles=overwrite_pickles,
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
    df_name = 'df_results_stats'
    df_path = os.path.join(results_path, df_name + suffix + '.pkl')
    if os.path.exists(df_path):
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
            df_all_mean = df_all_groups.apply(
                mean_conf_interval, key_string=metric)
            df_all_median = df_all_groups.apply(
                median_conf_interval, key_string=metric)
            dataframes.append(df_all_mean)
            dataframes.append(df_all_median)

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
# Compute results for instance 42 using PySA
instance = 42
metrics_list = ['min_energy', 'tts',
                'perf_ratio', 'success_prob', 'mean_time', 'inv_perf_ratio']
sweeps_list = [i for i in range(1, 21, 1)] + [
    i for i in range(20, 201, 10)] + [
    i for i in range(200, 501, 20)] + [
    i for i in range(500, 1001, 100)]
n_replicas_list = [1, 2, 4, 8]
# sweeps = [i for i in range(
#     1, 21, 1)] + [i for i in range(
#         21, 101, 10)]
p_hot_list = [50.0]
p_cold_list = [1.0]
bootstrap_iterations = 1000
s = 0.99  # This is the success probability for the TTS calculation
gap = 1.0  # This is a percentual treshold of what the minimum energy should be
conf_int = 68  #
fail_value = np.inf
# Confidence interval for bootstrapping, value used to get standard deviation
confidence_level = 68
boots_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
# TODO there should be an attribute to the parameters and if they vary logarithmically, have a function that generates the list of values "equally" spaced in logarithmic space

df_name = "df_results_" + str(instance) + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
if os.path.exists(df_path):
    df_42 = pd.read_pickle(df_path)
else:
    df_42 = None

parameters_dict = {
    'swe': sweeps_list,
    'rep': n_replicas_list,
    'pcold': p_cold_list,
    'phot': p_hot_list,
}
use_raw_dataframes = False
use_raw_sample_pickles = False
overwrite_pickles = False

df_42 = createPySAResultsDataframes(
    df=df_42,
    instance_list=[instance],
    parameters_dict=parameters_dict,
    boots_list=boots_list,
    data_path=data_path,
    results_path=results_path,
    pickles_path=pickles_path,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_sample_pickles=use_raw_sample_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
    ocean_df_flag=ocean_df_flag,
    suffix=suffix,
)

# %%
# Define plot longer labels
labels = {
    'N': 'Number of variables',
    'instance': 'Random instance',
    'replicas': 'Number of replicas',
    'sweeps': 'Number of sweeps',
    'rep': 'Number of replicas',
    'swe': 'Number of sweeps',
    'swe': 'Number of sweeps',
    'pcold': 'Probability of dEmin flip at cold temperature',
    'phot': 'Probability of dEmax flip at hot temperature',
    'mean_time': 'Mean time [us]',
    'success_prob': 'Success probability \n (within ' + str(gap) + '% of best found)',
    'median_success_prob': 'Success probability \n (within ' + str(gap) + '% of best found)',
    'mean_success_prob': 'Success probability \n (within ' + str(gap) + '% of best found)',
    'perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'median_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'mean_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'median_mean_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'mean_mean_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'median_median_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'mean_median_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'tts': 'TTS ' + str(100*s) + '% confidence  \n (within ' + str(gap) + '% of best found) [s]',
    'median_tts': 'TTS ' + str(100*s) + '% confidence  \n (within ' + str(gap) + '% of best found) [s]',
    'mean_tts': 'TTS ' + str(100*s) + '% confidence  \n (within ' + str(gap) + '% of best found) [s]',
    'boots': 'Number of downsamples during bootrapping',
    'reads': 'Total number of reads (proportional to time)',
    'cum_reads': 'Total number of reads (proportional to time)',
    'mean_cum_reads': 'Total number of reads (proportional to time)',
    'min_energy': 'Minimum energy found',
    'mean_time': 'Mean time [us]',
    'Tfactor': 'Factor to multiply lower temperature by',
    'experiment': 'Experiment',
    'inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'median_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'mean_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'median_mean_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'median_median_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'mean_mean_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'mean_median_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    # 'tts': 'TTS to GS with 99% confidence \n [s * replica] ~ [MVM]',
}

# %%
# Performance ratio vs sweeps for different bootstrap downsamples
default_dict = {
    'swe': default_sweeps,
    'rep': default_replicas,
    'pcold': default_p_cold,
    'phot': default_p_hot,
    'boots': default_boots
}
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_42,
    x_axis='swe',
    y_axis='perf_ratio',
    dict_fixed={'instance': 42},
    ax=ax,
    list_dicts=[{'rep': j, 'boots': i}
                for i in [1, 10, 100, 1000] for j in n_replicas_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict.update(
        {'instance': 42, 'reads': default_sweeps*default_boots}),
    use_colorbar=False,
    ylim=[0.6, 1.01],
    colors=['colormap'],
)
# %%
# Inverse performance ratio vs sweeps for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_42,
    x_axis='swe',
    y_axis='inv_perf_ratio',
    dict_fixed={'instance': 42},
    ax=ax,
    list_dicts=[{'rep': j, 'boots': i}
                for i in [1, 10, 100, 1000] for j in n_replicas_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    colors=['colormap'],
)
# %%
# Performance ratio vs runs for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_42,
    x_axis='reads',
    y_axis='perf_ratio',
    dict_fixed={'instance': 42},
    ax=ax,
    list_dicts=[{'rep': j, 'boots': i}
                for i in [1, 10, 100, 1000] for j in n_replicas_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    ylim=[0.9, 1.005],
    colors=['colormap'],
)
# %%
# Inverse performance ratio vs runs for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_42,
    x_axis='reads',
    y_axis='inv_perf_ratio',
    dict_fixed={'instance': 42},
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    colors=['colormap'],
    # ylim=[0.95, 1.005]
)
# %%
# Mean time plot of some fixed parameter setting
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_42,
    x_axis='swe',
    y_axis='mean_time',
    dict_fixed={'instance': 42, 'boots': 1000},
    ax=ax,
    list_dicts=[{'rep': i} for i in n_replicas_list],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
)
# %%
# Success probability of some fixed parameter setting
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_42,
    x_axis='swe',
    y_axis='success_prob',
    dict_fixed={'instance': 42, 'boots': 1000},
    ax=ax,
    list_dicts=[{'rep': i} for i in n_replicas_list],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
)
# %%
# TTS Plot for all bootstrapping downsamples in both schedules
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_42,
    x_axis='swe',
    y_axis='tts',
    ax=ax,
    dict_fixed={'instance': 42},
    list_dicts=[{'rep': i, 'boots': j}
                for i in n_replicas_list for j in [10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    colors=['colormap'],
)
# %%
# TTS Plot for all bootstrapping downsamples in both schedules
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_42,
    x_axis='reads',
    y_axis='tts',
    ax=ax,
    dict_fixed={'instance': 42},
    list_dicts=[{'rep': i, 'boots': j}
                for i in n_replicas_list for j in [10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    colors=['colormap'],
)
# %%
# Loop over the dataframes with 4 values of sweeps and a sweep in boots then compute the results, and complete by creating main Dataframe
# TODO: Generalize this description
interesting_sweeps = [
    df_42[df_42['boots'] == default_boots].nsmallest(1, 'tts')[
        'swe'].values[0],
    1,
    10,
    100,
    default_sweeps // 2,
    default_sweeps,
]

interesting_replicas = [
    df_42[df_42['boots'] == default_boots].nsmallest(1, 'tts')[
        'rep'].values[0],
    default_replicas,
    n_replicas_list[-1],
]

# Iterating for all values of bootstrapping downsampling proves to be very expensive, rather use steps of 10
# all_boots_list = list(range(1, 1001, 1))
all_boots_list = [i*10**j for j in range(0, 3) for i in range(1, 10)]
parameters_detailed_dict = {
    'swe': interesting_sweeps,
    'rep': interesting_replicas,
    'pcold': p_cold_list,
    'phot': p_hot_list,
}

use_raw_dataframes = False
use_raw_sample_pickles = False
df_42 = createPySAResultsDataframes(
    df=df_42,
    instance_list=[instance],
    parameters_dict=parameters_detailed_dict,
    boots_list=all_boots_list,
    data_path=data_path,
    results_path=results_path,
    pickles_path=pickles_path,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_sample_pickles=use_raw_sample_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
    ocean_df_flag=ocean_df_flag,
    suffix=suffix,
)

# %%
# Plot with performance ratio vs reads for interesting sweeps
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_42,
    x_axis='reads',
    y_axis='perf_ratio',
    # instance=42,
    dict_fixed={
        'instance': 42,
        # 'schedule':'geometric'
    },
    ax=ax,
    list_dicts=[{'swe': i, 'rep': j}
                for j in interesting_replicas for i in interesting_sweeps + [20]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    colors=['colormap'],
    ylim=[0.95, 1.005],
    # xlim=[1e2, 5e5],
)
# %%
# Plot with performance ratio vs reads for interesting sweeps
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_42,
    x_axis='reads',
    y_axis='perf_ratio',
    # instance=42,
    dict_fixed={
        'instance': 42,
        'rep': interesting_replicas[0]
    },
    ax=ax,
    list_dicts=[{'swe': i} for i in interesting_sweeps + [20]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    colors=['colormap'],
    ylim=[0.95, 1.005],
    xlim=[1e2, 5e5],
)
# %%
# Plot with inverse performance ratio vs reads for interesting sweeps
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_42,
    x_axis='reads',
    y_axis='inv_perf_ratio',
    # instance=42,
    dict_fixed={
        'instance': 42,
        'rep': interesting_replicas[0]
    },
    ax=ax,
    list_dicts=[{'swe': i} for i in interesting_sweeps + [20]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    colors=['colormap'],
    # ylim=[0.95, 1.005],
    xlim=[1e2, default_sweeps*default_boots*1.1],
)
# %%
# Compute all instances with solver
instance_list = [i for i in range(20)] + [42]
training_instance_list = [i for i in range(20)]
# %%
# Merge all results dataframes in a single one
df_list = []
use_raw_dataframes = False
use_raw_sample_pickles = False
# all_boots_list = list(range(1, 1001, 1))
for instance in instance_list:
    df_name = "df_results_" + str(instance) + suffix + ".pkl"
    df_path = os.path.join(results_path, df_name)
    if os.path.exists(df_path):
        df_results_instance = pd.read_pickle(df_path)
    else:
        df_results_instance = None
    parameters_dict = {
        'swe': sweeps_list,
        'rep': n_replicas_list,
        'pcold': p_cold_list,
        'phot': p_hot_list,
    }
    df_results_instance = createPySAResultsDataframes(
        df=df_results_instance,
        instance_list=[instance],
        parameters_dict=parameters_dict,
        boots_list=boots_list,
        data_path=data_path,
        results_path=results_path,
        pickles_path=pickles_path,
        use_raw_dataframes=use_raw_dataframes,
        use_raw_sample_pickles=use_raw_sample_pickles,
        overwrite_pickles=overwrite_pickles,
        s=s,
        confidence_level=conf_int,
        bootstrap_iterations=bootstrap_iterations,
        gap=gap,
        fail_value=fail_value,
        save_pickle=True,
        ocean_df_flag=ocean_df_flag,
        suffix=suffix,
    )

    # Loop over the dataframes with 4 values of sweeps and a sweep in boots then compute the results, and complete by creating main Dataframe
    interesting_sweeps = [
        df_results_instance[df_results_instance['boots'] == default_boots].nsmallest(1, 'tts')[
            'swe'].values[0],
        1,
        10,
        100,
        default_sweeps // 2,
        default_sweeps,
    ]
    interesting_replicas = [
        df_results_instance[df_results_instance['boots'] == default_boots].nsmallest(1, 'tts')[
            'rep'].values[0],
        default_replicas,
        n_replicas_list[-1],
    ]

    parameters_detailed_dict = {
        'swe': interesting_sweeps,
        'rep': interesting_replicas,
        'pcold': p_cold_list,
        'phot': p_hot_list,
    }

    df_results_instance = createPySAResultsDataframes(
        df=df_results_instance,
        instance_list=[instance],
        parameters_dict=parameters_detailed_dict,
        boots_list=all_boots_list,
        data_path=data_path,
        results_path=results_path,
        pickles_path=pickles_path,
        use_raw_dataframes=use_raw_dataframes,
        use_raw_sample_pickles=use_raw_sample_pickles,
        overwrite_pickles=overwrite_pickles,
        s=s,
        confidence_level=conf_int,
        bootstrap_iterations=bootstrap_iterations,
        gap=gap,
        fail_value=fail_value,
        save_pickle=True,
        ocean_df_flag=ocean_df_flag,
        suffix=suffix,
    )

    df_list.append(df_results_instance)

df_results_all = pd.concat(df_list, ignore_index=True)
df_name = "df_results" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
df_results_all = cleanup_df(df_results_all)
df_results_all.to_pickle(df_path)

# %%
# Run all the instances with solver
overwrite_pickles = False
use_raw_dataframes = False
use_raw_sample_pickles = False

df_results_all = createPySAResultsDataframes(
    df=df_results_all,
    instance_list=instance_list,
    parameters_dict=parameters_dict,
    boots_list=boots_list,
    data_path=data_path,
    results_path=results_path,
    pickles_path=pickles_path,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_sample_pickles=use_raw_sample_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
    ocean_df_flag=ocean_df_flag,
    suffix=suffix,
)

# %%
# Generate stats results
use_raw_full_dataframe = False
use_raw_dataframes = False
use_raw_sample_pickles = False
df_results_all_stats = generateStatsDataframe(
    df_all=df_results_all,
    stat_measures=['mean', 'median'],
    instance_list=training_instance_list,
    parameters_dict=parameters_dict,
    resource_list=boots_list,
    data_path=data_path,
    results_path=results_path,
    pickles_path=pickles_path,
    use_raw_full_dataframe=use_raw_full_dataframe,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_sample_pickles=use_raw_sample_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
    ocean_df_flag=ocean_df_flag,
)
# %%
# Generate plots for TTS of ensemble together with single instance (42)
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_42,
    x_axis='swe',
    y_axis='tts',
    ax=ax,
    dict_fixed={'pcold': 1, 'phot': 50},
    list_dicts=[{'instance': 42, 'boots': j, 'rep': i}
                for j in [1000] for i in n_replicas_list],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=True,
    colormap=plt.cm.Dark2,
)
plot_1d_singleinstance_list(
    df=df_results_all_stats,
    x_axis='swe',
    y_axis='median_tts',
    ax=ax,
    dict_fixed={'pcold': 1, 'phot': 50},
    list_dicts=[{'boots': j, 'rep': i}
                for j in [1000] for i in n_replicas_list],
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
    x_axis='swe',
    y_axis='median_perf_ratio',
    ax=ax,
    dict_fixed={'pcold': 1, 'phot': 50},
    list_dicts=[{'boots': j, 'rep': i}
                for j in [1000] for i in n_replicas_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    ylim=[0.95, 1.005],
)
# %%
# Generate plots for performance ratio of ensemble vs reads
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_results_all_stats,
    x_axis='reads',
    y_axis='median_perf_ratio',
    ax=ax,
    dict_fixed={'pcold': 1, 'phot': 50, 'swe': 500, 'rep': 8},
    list_dicts=[{'boots': j}
                for j in all_boots_list[::1]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    marker='*',
    use_colorbar=True,
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
    dict_fixed={'pcold': 1, 'phot': 50},
    list_dicts=[{'rep': i, 'swe': j}
                for j in interesting_sweeps for i in n_replicas_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    ylim=[0.95, 1.005],
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
    dict_fixed={'pcold': 1, 'phot': 50},
    list_dicts=[{'boots': j}
                for j in [1, 10, 100, 1000]],
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
# Gather all the data for the best tts of the ensemble training set for each instance
best_ensemble_sweeps = []
best_ensemble_replicas = []
df_list = []
stat_measures = ['mean', 'median']
use_raw_dataframes = False
for stat_measure in stat_measures:
    best_ensemble_sweeps.append(df_results_all_stats[df_results_all_stats['boots'] == default_boots].nsmallest(
        1, stat_measure + '_tts')['swe'].values[0])
    best_ensemble_replicas.append(df_results_all_stats
                                  [(df_results_all_stats['boots'] == default_boots)].nsmallest(
                                      1, stat_measure + '_tts')['rep'].values[0])
parameters_best_ensemble_dict = {
    'swe': best_ensemble_sweeps,
    'rep': best_ensemble_replicas,
    'pcold': p_cold_list,
    'phot': p_hot_list,
}
for instance in instance_list:
    df_name = "df_results_" + str(instance) + suffix + ".pkl"
    df_path = os.path.join(results_path, df_name)
    df_results_instance = pd.read_pickle(df_path)
    df_results_instance = createPySAResultsDataframes(
        df=df_results_instance,
        instance_list=[instance],
        parameters_dict=parameters_best_ensemble_dict,
        boots_list=all_boots_list,
        data_path=data_path,
        results_path=results_path,
        pickles_path=pickles_path,
        use_raw_dataframes=use_raw_dataframes,
        use_raw_sample_pickles=use_raw_sample_pickles,
        overwrite_pickles=overwrite_pickles,
        s=s,
        confidence_level=conf_int,
        bootstrap_iterations=bootstrap_iterations,
        gap=gap,
        fail_value=fail_value,
        save_pickle=True,
        ocean_df_flag=ocean_df_flag,
        suffix=suffix,
    )
    df_list.append(df_results_instance)

df_results_all = pd.concat(df_list, ignore_index=True)
df_results_all = cleanup_df(df_results_all)
df_name = "df_results" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
df_results_all.to_pickle(df_path)

# %%
# Reload all results with the best tts of the ensemble for each instance
df_results_all = createPySAResultsDataframes(
    df=df_results_all,
    instance_list=instance_list,
    parameters_dict=parameters_best_ensemble_dict,
    boots_list=all_boots_list,
    data_path=data_path,
    results_path=results_path,
    pickles_path=pickles_path,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_sample_pickles=use_raw_sample_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
    ocean_df_flag=ocean_df_flag,
    suffix=suffix,
)
# %%
# Compute inverse of performance ratio for all instances
if 'inv_perf_ratio' not in df_results_all.columns:
    df_results_all['inv_perf_ratio'] = 1 - \
        df_results_all['perf_ratio'] + EPSILON
    df_results_all['inv_perf_ratio_conf_interval_lower'] = 1 - \
        df_results_all['perf_ratio_conf_interval_upper'] + EPSILON
    df_results_all['inv_perf_ratio_conf_interval_upper'] = 1 - \
        df_results_all['perf_ratio_conf_interval_lower'] + EPSILON
df_results_all = cleanup_df(df_results_all)
df_name = "df_results" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
df_results_all.to_pickle(df_path)

if 'inv_perf_ratio' not in df_42.columns:
    df_42['inv_perf_ratio'] = 1 - df_42['perf_ratio'] + EPSILON
    df_42['inv_perf_ratio_conf_interval_lower'] = 1 - \
        df_42['perf_ratio_conf_interval_upper'] + EPSILON
    df_42['inv_perf_ratio_conf_interval_upper'] = 1 - \
        df_42['perf_ratio_conf_interval_lower'] + EPSILON
df_42 = cleanup_df(df_42)
df_name = "df_results_42" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
df_42.to_pickle(df_path)
# %%
# Obtain the tts for each instance in the median and the mean of the ensemble accross the sweeps
# TODO generalize this code. In general, one parameter (or several) are fixed in certain interesting values and then for all instances with the all other values of remaining parameters we report the metric output, everything at 1000 bootstraps


for metric in ['perf_ratio', 'success_prob', 'tts', 'inv_perf_ratio']:

    df_results_all[metric + '_lower'] = df_results_all[metric] - \
        df_results_all[metric + '_conf_interval_lower']
    df_results_all[metric + '_upper'] = df_results_all[metric] - \
        df_results_all[metric + '_conf_interval_upper']

# These following lines can be extracted from the loop
df_default = df_results_all[
    (df_results_all['boots'] == default_boots) &
    (df_results_all['swe'] == default_sweeps) &
    (df_results_all['rep'] == default_replicas) &
    (df_results_all['pcold'] == default_p_cold) &
    (df_results_all['phot'] == default_p_hot)
    ].set_index(['instance'])
df_default.fillna(fail_value, inplace=True)
keys_list = ['default']
df_ensemble_best = []
# TODO: How to generalize this to zips of all parameters that change?
for i, swe_rep in enumerate(zip(best_ensemble_sweeps, best_ensemble_replicas)):
    df_metric_best = df_results_all[
        (df_results_all['boots'] == default_boots) &
        (df_results_all['swe'] == swe_rep[0]) &
        (df_results_all['rep'] == swe_rep[1]) &
        (df_results_all['pcold'] == default_p_cold) &
        (df_results_all['phot'] == default_p_hot)
        ].set_index(['instance'])
    df_metric_best.fillna(fail_value, inplace=True)
    df_ensemble_best.append(df_metric_best)
    keys_list.append(stat_measures[i] + '_ensemble')
# Until here can be done off-loop
keys_list.append('best')

for metric in ['perf_ratio', 'success_prob', 'tts', 'inv_perf_ratio']:

    df_list = [df_default] + df_ensemble_best

    # Metrics that you want to minimize
    ascent = metric in ['tts', 'mean_time', 'inv_perf_ratio']
    df_best = df_results_all[
        (df_results_all['boots'] == default_boots)
    ].sort_values(metric, ascending=ascent).groupby(
        ['instance']).apply(
            pd.DataFrame.head, n=1
    ).droplevel(-1).drop(columns=['instance']
                         )

    df_list.append(df_best)

    df_merged = pd.concat(df_list, axis=1, keys=keys_list,
                          names=['stat_metric', 'measure'])

    fig, ax = plt.subplots()
    df_merged.loc[
        slice(None),
        :
    ].plot.bar(
        y=[(stat_metric, metric) for stat_metric in keys_list],
        yerr=df_merged.loc[
            slice(None),
            (slice(None), [metric + '_lower', metric + '_upper'])
        ].to_numpy().T,
        ax=ax,
    )
    # TODO: leverage use of index
    ax.set(title='Different performance of ' + metric +
           ' in instances ' + prefix + '\n' +
           'evaluated individually and with the ensemble')
    ax.set(ylabel=labels[metric])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if metric in ['tts', 'inv_perf_ratio']:
        ax.set(yscale='log')

df_results_all = cleanup_df(df_results_all)
df_name = "df_results" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
df_results_all.to_pickle(df_path)
# %%
# Plot with performance ratio vs reads for interesting sweeps
for instance in [3, 0, 7, 42]:
    interesting_sweeps = [
        df_results_all[
            (df_results_all['boots'] == default_boots) &
            (df_results_all['instance'] == instance)].nsmallest(1, 'tts')[
            'swe'].values[0],
        default_sweeps,
    ] + best_ensemble_sweeps
    interesting_replicas = [
        df_results_all[
            (df_results_all['boots'] == default_boots) &
            (df_results_all['instance'] == instance)].nsmallest(1, 'tts')['rep'].values[0],
        default_replicas,
    ] + best_ensemble_replicas

    f, ax = plt.subplots()
    ax = plot_1d_singleinstance_list(
        df=df_results_all,
        x_axis='reads',
        y_axis='perf_ratio',
        dict_fixed={
            'instance': instance,
        },
        ax=ax,
        list_dicts=[{'swe': i, 'rep': j}
                    for (i, j) in zip(interesting_sweeps, interesting_replicas)],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        default_dict=default_dict.update({'instance': instance}),
        use_colorbar=False,
        ylim=[0.975, 1.0025],
    )

# %%
# Plot with inverse performance ratio vs reads for interesting sweeps
for instance in [3, 0, 7, 42]:
    interesting_sweeps = [
        df_results_all[
            (df_results_all['boots'] == default_boots) &
            (df_results_all['instance'] == instance)].nsmallest(1, 'tts')[
            'swe'].values[0],
        default_sweeps,
    ] + best_ensemble_sweeps
    interesting_replicas = [
        df_results_all[
            (df_results_all['boots'] == default_boots) &
            (df_results_all['instance'] == instance)].nsmallest(1, 'tts')['rep'].values[0],
        default_replicas,
    ] + best_ensemble_replicas

    f, ax = plt.subplots()
    ax = plot_1d_singleinstance_list(
        df=df_results_all,
        x_axis='reads',
        y_axis='inv_perf_ratio',
        dict_fixed={
            'instance': instance,
        },
        ax=ax,
        list_dicts=[{'swe': i, 'rep': j}
                    for (i, j) in zip(interesting_sweeps, interesting_replicas)],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        save_fig=False,
        default_dict=default_dict.update({'instance': instance}),
        use_colorbar=False,
    )

# %%
# Regenerate the dataframe with the statistics to get the complete performance plot
use_raw_full_dataframe = True
df_results_all_stats = generateStatsDataframe(
    df_all=df_results_all,
    stat_measures=['mean', 'median'],
    instance_list=training_instance_list,
    parameters_dict=parameters_dict,
    resource_list=boots_list,
    data_path=data_path,
    results_path=results_path,
    use_raw_full_dataframe=use_raw_full_dataframe,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_sample_pickles=use_raw_sample_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
    ocean_df_flag=ocean_df_flag,
)
# %%
# Generate plots for performance ratio of ensemble vs reads
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    for instance in instance_list:
        plot_1d_singleinstance_list(
            df=df_results_all,
            x_axis='reads',
            y_axis='perf_ratio',
            ax=ax,
            dict_fixed={'pcold': 1, 'phot': 50},
            list_dicts=[{
                'swe': i,
                'instance': instance,
                'rep': j}
                for (i, j) in zip([default_sweeps] + best_ensemble_sweeps, [default_replicas] + best_ensemble_replicas)],
            labels=labels,
            prefix=prefix,
            log_x=True,
            log_y=False,
            save_fig=False,
            use_colorbar=False,
            marker=None,
            alpha=0.15,
            colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                    u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
        )
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        list_dicts=[{
            'swe': i,
            'rep': j}
            for (i, j) in zip([default_sweeps] + best_ensemble_sweeps, [default_replicas] + best_ensemble_replicas)],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        ylim=[0.95, 1.0025],
        xlim=[5e2, 1e6],
        use_colorbar=False,
        linewidth=2.5,
        colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
    )

# %%
# Generate plots for inverse performance ratio of ensemble vs reads
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_inv_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        list_dicts=[{
            'swe': i,
            'rep': j}
            for (i, j) in zip([default_sweeps] + best_ensemble_sweeps, [default_replicas] + best_ensemble_replicas)],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        save_fig=False,
        xlim=[5e2, 1e6],
        use_colorbar=False,
        linewidth=2.5,
        colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
    )
# %%
# Create virtual best and virtual worst columns
# TODO This can be generalized as using as groups the parameters that are not dependent of the metric (e.g., schedule) or that signify different solvers
# TODO This needs to be functionalized
params = ['swe','rep','pcold', 'phot']
stale_parameters = ['pcold', 'phot']

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
        on=['reads'],
        how='left')
    df_virtual_best = df_virtual_best.merge(
        df_virtual_best_min,
        on=['reads'],
        how='left')
    df_virtual_best = df_virtual_best.merge(
        df_virtual_worst_min,
        on=['reads'],
        how='left')


    

    recipe_lazy = df_results_all_stats[params + ['median_perf_ratio','reads']].set_index(
                params
                ).groupby(['reads']
                ).idxmax()

    df_virtual_best = df_virtual_best.merge(
        df_results_all_stats[params + ['median_perf_ratio','reads']].set_index(
                params
                ).groupby(['reads']
                ).max().reset_index().rename(columns={'median_perf_ratio':'lazy_perf_ratio'}))
    
    # Dirty workaround to compute virtual best perf_ratio as commended by Davide, several points: 1) the perf_ratio is computed as the maximum (we are assuming we care about the max) of for each instance for each read, 2) the median of this idealized solver (that has the best parameters for each case) across the instances is computed
    

    df_virtual_best = df_virtual_best.merge(
        df_results_all.set_index(
            params
        ).groupby(
            ['instance','reads']
            )['perf_ratio'].max().reset_index().set_index(
            ['instance']
            ).groupby(
                ['reads']
            ).median().reset_index().sort_values(
                ['reads']
                ).expanding(min_periods=1).max(),
            on=['reads'],
            how='left')

    df_virtual_best['inv_lazy_perf_ratio'] = 1 - df_virtual_best['lazy_perf_ratio'] + EPSILON

    df_virtual_best['inv_perf_ratio'] = 1 - df_virtual_best['perf_ratio'] + EPSILON
    df_virtual_best = cleanup_df(df_virtual_best)
    df_virtual_best.to_pickle(df_path)


    df_virtual_best = cleanup_df(df_virtual_best)
    df_virtual_best.to_pickle(df_path)
else:
    df_virtual_best = pd.read_pickle(df_path)
    df_virtual_best = cleanup_df(df_virtual_best)



# Here I'm filtering for low noise data
window_average = 20
df_rolled = df_virtual_best.sort_index().rolling(window=window_average, min_periods=0).mean()
df_virtual_best['soft_lazy_perf_ratio'] = df_rolled['lazy_perf_ratio']


# %%
# Generate plots for performance ratio of ensemble vs reads with best and worst performance
soft_flag = True
soft_str = ''
if soft_flag:
    soft_str += 'soft_'
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed=None,
        labels=labels,
        prefix=prefix,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['k'],
    )
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis=soft_str+'lazy_perf_ratio',
        ax=ax,
        label_plot='Suggested fixed parameters',
        dict_fixed=None,
        labels=labels,
        prefix=prefix,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['r'],
    )
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        list_dicts=[{
            'swe': i,
            'rep': j}
            for (i, j) in zip([default_sweeps] + best_ensemble_sweeps, [default_replicas] + best_ensemble_replicas)],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        ylim=[0.975, 1.0025],
        xlim=[5e2, 1e6],
        use_colorbar=False,
        linewidth=1.5,
        markersize=1,
        colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
    )
# %%
# Generate plots for inverse performance ratio of ensemble vs reads with best and worst performance
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='inv_perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed=None,
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['k'],
    )
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='inv_lazy_perf_ratio',
        ax=ax,
        label_plot='Suggested fixed parameters',
        dict_fixed=None,
        labels=labels,
        prefix=prefix,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['r'],
    )
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_inv_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        list_dicts=[{
            'swe': i,
            'rep': j}
            for (i, j) in zip([default_sweeps] + best_ensemble_sweeps, [default_replicas] + best_ensemble_replicas)],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        save_fig=False,
        use_colorbar=False,
        xlim=[5e2, 1e6],
        linewidth=1.5,
        markersize=1,
        colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
    )
# %%
# Random search for the ensemble
repetitions = 10  # Times to run the algorithm
# rs = [1, 5, 10]  # resources per parameter setting (runs)
rs = [1, 2, 5, 10, 20, 50, 100]  # resources per parameter setting (runs)
# frac_r_exploration = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
frac_r_exploration = [0.05, 0.1, 0.2, 0.5, 0.75]
# R_budgets = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
# R_budgets = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
R_budgets = [i*10**j for i in [1, 1.5, 2, 3, 5, 7] for j in [1, 2, 3, 4, 5]] + [1e6]
experiments = rs * repetitions
df_name = "df_progress_total" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
search_metric = 'median_perf_ratio'
compute_metric = 'median_perf_ratio'
df_search = df_results_all_stats[
    parameters_list + ['boots',
                  'median_perf_ratio', 'reads']
].set_index(
    parameters_list + ['boots'])
parameter_sets = itertools.product(
    *(parameters_dict[Name] for Name in parameters_dict))
parameter_sets = list(parameter_sets)
use_raw_dataframes = False
if use_raw_dataframes or os.path.exists(df_path) is False:
    progress_list = []
    for R_budget in R_budgets:
        for frac_expl_total in frac_r_exploration:
            # R_exploration = 50  # budget for exploration (runs)
            # budget for exploration (runs)
            R_exploration = int(R_budget*frac_expl_total)
            # budget for exploitation (runs)
            R_exploitation = R_budget - R_exploration
            for r in rs:
                if r > R_exploration:
                    print(
                        "R_exploration must be larger than single exploration step")
                    continue
                for experiment in range(repetitions):
                    random_parameter_sets = random.choices(
                        parameter_sets, k=int(R_exploration / r))
                    # Conservative estimate of very unlikely scenario that we choose all sweeps=1, replicas=1
                    # % Question: Should we replace these samples?
                    if r*random_parameter_sets[0][0]*random_parameter_sets[0][1] > R_exploration:
                        # TODO: There should be a better way of having parameters that affect runtime making an appear. An idea, having a function f(params) = runs that we can call
                        print(
                            "R_exploration must be larger than single exploration step")
                        continue
                        # We allow it to run at least once assuming that
                    series_list = []
                    total_reads = 0
                    for random_parameter_set in random_parameter_sets:
                        total_reads += r*random_parameter_set[0]*random_parameter_set[1]
                        if total_reads > R_exploration:
                            break
                        series_list.append(
                            df_search.loc[
                                idx[random_parameter_set + (r,)]
                            ]
                        )
                    exploration_step = pd.concat(series_list, axis=1).T.rename_axis(
                        parameters_list + ['boots'])
                    exploration_step[compute_metric] = exploration_step[compute_metric].expanding(
                        min_periods=1).max()
                    exploration_step.reset_index('boots', inplace=True)
                    exploration_step['experiment'] = experiment
                    exploration_step['run_per_solve'] = r
                    exploration_step['R_explor'] = R_exploration
                    exploration_step['R_exploit'] = R_exploitation
                    exploration_step['R_budget'] = R_budget
                    exploration_step['cum_reads'] = exploration_step.groupby('experiment').expanding(
                        min_periods=1)['reads'].sum().reset_index(drop=True).values
                    progress_list.append(exploration_step)

                    exploitation_step = df_search.reset_index().set_index(
                        parameters_list).loc[exploration_step.nlargest(1, compute_metric).index]
                    exploitation_step['cum_reads'] = exploitation_step['reads'] + \
                        exploration_step['cum_reads'].max()
                    exploitation_step.sort_values(['cum_reads'], inplace=True)
                    exploitation_step = exploitation_step[exploitation_step['cum_reads'] <= R_budget]
                    exploitation_step[compute_metric].fillna(
                        0, inplace=True)
                    exploitation_step[compute_metric].clip(
                        lower=exploration_step[compute_metric].max(), inplace=True)
                    exploitation_step[compute_metric] = exploitation_step[compute_metric].expanding(
                        min_periods=1).max()
                    exploitation_step['experiment'] = experiment
                    exploitation_step['run_per_solve'] = r
                    exploitation_step['R_explor'] = R_exploration
                    exploitation_step['R_exploit'] = R_exploitation
                    exploitation_step['R_budget'] = R_budget
                    progress_list.append(exploitation_step)
    df_progress_total = pd.concat(progress_list, axis=0)
    df_progress_total.reset_index(inplace=True)
    df_progress_total.to_pickle(df_path)
else:
    df_progress_total = pd.read_pickle(df_path)

if 'R_budget' not in df_progress_total.columns:
    df_progress_total['R_budget'] = df_progress_total['R_explor'] + \
        df_progress_total['R_exploit']
df_progress_total = cleanup_df(df_progress_total)

for stat_measure in ['median']:
    if 'best_' + stat_measure + '_perf_ratio' not in df_progress_total.columns:
        df_progress_total[stat_measure + '_inv_perf_ratio'] = 1 - \
            df_progress_total[stat_measure + '_perf_ratio'] + EPSILON
        df_progress_total['best_' + stat_measure + '_inv_perf_ratio'] = df_progress_total.sort_values(
            ['cum_reads', 'R_budget']
        ).expanding(min_periods=1).min()[stat_measure + '_inv_perf_ratio']
        df_progress_total['best_' + stat_measure + '_perf_ratio'] = 1 - \
            df_progress_total['best_' + stat_measure +
                              '_inv_perf_ratio'] + EPSILON
df_progress_total = cleanup_df(df_progress_total)

# %%
# Plot for all the experiments trajectories
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_progress_total,
    x_axis='cum_reads',
    y_axis='median_perf_ratio',
    ax=ax,
    dict_fixed={
        'R_budget': R_budgets[-1],
        'R_explor': R_budgets[-1]*frac_r_exploration[0],
        'run_per_solve': rs[0],
    },
    # label_plot='Ordered exploration',
    list_dicts=[{'experiment': i}
                for i in range(repetitions)],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    colors=['colormap'],
    colormap=plt.cm.get_cmap('rainbow'),
    use_colorbar=False,
    use_conf_interval=False,
    save_fig=False,
    ylim=[0.90, 1.0025],
    xlim=[1e2, 1e5],
    linewidth=1.5,
    marker=None,
)

# %%
# Alternative implementation of the best random exploration - exploitation
# Gather the performance at the last period of the experiments
# Processing random search for instance ensemble
df_progress_best, df_progress_end = process_df_progress(
    df_progress=df_progress_total,
    compute_metrics=['median_perf_ratio'],
    stat_measures=['mean', 'median'],
    maximizing=True,
)

# %%
# Generate plots for performance ratio of ensemble vs reads with best and worst performance
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed=None,
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['k'],
    )
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='lazy_perf_ratio',
        ax=ax,
        label_plot='Suggested fixed parameters',
        dict_fixed=None,
        labels=labels,
        prefix=prefix,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['r'],
    )
    # plot_1d_singleinstance(
    #     df=df_virtual_best,
    #     x_axis='reads',
    #     y_axis='virt_worst_perf_ratio',
    #     ax=ax,
    #     label_plot='Virtual worst',
    #     dict_fixed={'pcold': 1, 'phot': 50},
    #     labels=labels,
    #     prefix=prefix,
    #     log_x=True,
    #     log_y=False,
    #     use_conf_interval=False,
    #     save_fig=False,
    #     linewidth=2.5,
    #     marker=None,
    #     color=['r'],
    # )
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        list_dicts=[{
            'swe': i,
            'rep': j}
            for (i, j) in zip([default_sweeps] + best_ensemble_sweeps, [default_replicas] + best_ensemble_replicas)],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        use_conf_interval=False,
        save_fig=False,
        ylim=[0.975, 1.0025],
        # xlim=[5e2, 5e4],
        use_colorbar=False,
        linewidth=1.5,
        markersize=1,
        colormap=plt.cm.get_cmap('viridis'),
        colors=['colormap'],
        style='--',
    )
    plot_1d_singleinstance(
        df=df_progress_best,
        x_axis='R_budget',
        # y_axis='mean_' + stat_measure + '_perf_ratio',
        y_axis=stat_measure + '_median_perf_ratio',
        ax=ax,
        dict_fixed=None,
        # label_plot='Ordered exploration',
        # list_dicts=[{'R_budget': i}
        #             for i in R_budgets],
        labels=labels,
        label_plot = 'Best random exploration exploitation',
        prefix=prefix,
        log_x=True,
        log_y=False,
        use_conf_interval=False,
        save_fig=False,
        ylim=[0.975, 1.0025],
        xlim=[1e2, R_budgets[-1]],
        linewidth=2.5,
        marker=None,
        color=['m'],
    )
# %%
# Generate plots for inverse performance ratio of ensemble vs reads with best and worst performance
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='inv_perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed=None,
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['k'],
    )
    # plot_1d_singleinstance(
    #     df=df_virtual_best,
    #     x_axis='reads',
    #     y_axis='virt_worst_inv_perf_ratio',
    #     ax=ax,
    #     label_plot='Virtual worst',
    #     dict_fixed={'pcold': 1, 'phot': 50},
    #     labels=labels,
    #     prefix=prefix,
    #     log_x=True,
    #     log_y=False,
    #     use_conf_interval=False,
    #     save_fig=False,
    #     linewidth=2.5,
    #     marker=None,
    #     color=['r'],
    # )
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_inv_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        list_dicts=[{
            'swe': i,
            'rep': j}
            for (i, j) in zip([default_sweeps] + best_ensemble_sweeps, [default_replicas] + best_ensemble_replicas)],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        use_conf_interval=False,
        save_fig=False,
        use_colorbar=False,
        linewidth=1.5,
        markersize=1,
        colormap=plt.cm.get_cmap('viridis'),
        colors=['colormap'],
        style='--',
    )
    plot_1d_singleinstance(
        df=df_progress_best,
        x_axis='R_budget',
        # y_axis='mean_' + stat_measure + '_inv_perf_ratio',
        y_axis=stat_measure + '_median_inv_perf_ratio',
        ax=ax,
        dict_fixed=None,
        # label_plot='Ordered exploration',
        # list_dicts=[{'R_budget': i}
        #             for i in R_budgets],
        labels=labels,
        label_plot = 'Best random exploration exploitation',
        prefix=prefix,
        log_x=True,
        log_y=True,
        # colors=['colormap'],
        # colormap=plt.cm.get_cmap('tab10'),
        # use_colorbar=False,
        use_conf_interval=False,
        save_fig=False,
        # ylim=[0.975, 1.0025],
        ylim=[1e-10, 1e0],
        xlim=[1e2, R_budgets[-1]],
        linewidth=1.5,
        markersize=1,
    )

# %%
# Until here I've checked, above we still need to use virt_best and virt_worst logic to get the performance ratio on the binary search
# Computing up ternary search across parameter
# We assume that the performance of the parameter is unimodal (in decreases and the increases)
# r = 1  # resource per parameter setting (runs)
rs = [1, 5, 10]
# R_budget = 550  # budget for exploitation (runs)
df_name = "df_progress_ternary" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
df_search = df_results_all_stats[
    ['schedule', 'swe', 'boots',
     'median_perf_ratio', 'mean_perf_ratio', 'reads']
].set_index(
    ['schedule', 'swe', 'boots']
)
use_raw_dataframes = False
if use_raw_dataframes or os.path.exists(df_path) is False:
    progress_list = []
    for r in rs:
        series_list = []
        lo = 0
        val_lo = sweeps_list[lo]
        up = len(sweeps_list) - 1
        val_up = sweeps_list[up]
        perf_lo = df_search.loc[
            idx['geometric', val_lo, r]]['median_perf_ratio']
        perf_up = df_search.loc[
            idx['geometric', val_up, r]]['median_perf_ratio']
        series_list.append(df_search.loc[
            idx['geometric', val_lo, r]])
        series_list.append(df_search.loc[
            idx['geometric', val_up, r]])
        while lo <= up:
            x1 = int(lo + (up - lo) / 3)
            x2 = int(up - (up - lo) / 3)
            val_x1 = sweeps_list[x1]
            perf_x1 = df_search.loc[
                idx['geometric', val_x1, r]]['median_perf_ratio']
            series_list.append(df_search.loc[
                idx['geometric', val_x1, r]])
            val_x2 = sweeps_list[x2]
            perf_x2 = df_search.loc[
                idx['geometric', val_x2, r]]['median_perf_ratio']
            series_list.append(df_search.loc[
                idx['geometric', val_x2, r]])
            if perf_x2 == perf_up:
                up -= 1
                val_up = sweeps_list[up]
                perf_up = df_search.loc[
                    idx['geometric', val_up, r]]['median_perf_ratio']
                series_list.append(df_search.loc[
                    idx['geometric', val_up, r]])
            elif perf_x1 == perf_lo:
                lo += 1
                val_lo = sweeps_list[lo]
                perf_lo = df_search.loc[
                    idx['geometric', val_lo, r]]['median_perf_ratio']
                series_list.append(df_search.loc[
                    idx['geometric', val_lo, r]])
            elif perf_x1 > perf_x2:
                up = x2
                val_up = sweeps_list[up]
                perf_up = df_search.loc[
                    idx['geometric', val_up, r]]['median_perf_ratio']
            else:
                lo = x1
                val_lo = sweeps_list[lo]
                perf_lo = df_search.loc[
                    idx['geometric', val_lo, r]]['median_perf_ratio']

        exploration_step = pd.concat(series_list, axis=1).T.rename_axis(
            ['schedule', 'Tfactor', 'boots'])
        exploration_step['median_perf_ratio'] = exploration_step['median_perf_ratio'].expanding(
            min_periods=1).max()
        exploration_step['mean_perf_ratio'] = exploration_step['mean_perf_ratio'].expanding(
            min_periods=1).max()
        exploration_step.reset_index('boots', inplace=True)
        exploration_step['run_per_solve'] = r
        exploration_step['cum_reads'] = exploration_step.expanding(
            min_periods=1)['reads'].sum().reset_index(drop=True).values
        progress_list.append(exploration_step)

        exploitation_step = df_search.reset_index().set_index(
            ['schedule', 'swe']).loc[exploration_step.nlargest(1, 'median_perf_ratio').index]
        exploitation_step['cum_reads'] = exploitation_step['reads'] + \
            exploration_step['cum_reads'].max()
        exploitation_step = exploitation_step[exploitation_step['cum_reads']
                                              <= default_reads*exploration_step.nlargest(1, 'median_perf_ratio').index.values[0][-1]
                                              ]
        # TODO this is not a very stable way to get the sweeps values
        exploitation_step.sort_values(['cum_reads'], inplace=True)
        exploitation_step['median_perf_ratio'].fillna(
            0, inplace=True)
        exploitation_step['median_perf_ratio'].clip(
            lower=exploration_step['median_perf_ratio'].max(), inplace=True)
        exploitation_step['median_perf_ratio'] = exploitation_step['median_perf_ratio'].expanding(
            min_periods=1).max()
        exploitation_step['median_perf_ratio'].fillna(
            0, inplace=True)
        exploitation_step['mean_perf_ratio'].clip(
            lower=exploration_step['mean_perf_ratio'].max(), inplace=True)
        exploitation_step['mean_perf_ratio'] = exploitation_step['mean_perf_ratio'].expanding(
            min_periods=1).max()
        exploitation_step['run_per_solve'] = r
        progress_list.append(exploitation_step)
    df_progress_ternary = pd.concat(progress_list, axis=0)
    df_progress_ternary.reset_index(inplace=True)
    df_progress_ternary = cleanup_df(df_progress_ternary)
    df_progress_ternary.to_pickle(df_path)
else:
    df_progress_ternary = pd.read_pickle(df_path)

if 'median_inv_perf_ratio' not in df_progress_ternary.columns:
    df_progress_ternary['median_inv_perf_ratio'] = 1 - \
        df_progress_ternary['median_perf_ratio'] + EPSILON
if 'mean_inv_perf_ratio' not in df_progress_ternary.columns:
    df_progress_ternary['mean_inv_perf_ratio'] = 1 - \
        df_progress_ternary['mean_perf_ratio'] + EPSILON
# %%
# Presentation slides!
# Plots of ternary search together with the best performing schedule
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed=None,
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['k'],
    )
    # plot_1d_singleinstance(
    #     df=df_virtual_best,
    #     x_axis='reads',
    #     y_axis='virt_worst_perf_ratio',
    #     ax=ax,
    #     label_plot='Virtual worst',
    #     dict_fixed={'pcold': 1, 'phot': 50},
    #     labels=labels,
    #     prefix=prefix,
    #     log_x=True,
    #     log_y=False,
    #     use_conf_interval=False,
    #     save_fig=False,
    #     linewidth=2.5,
    #     marker=None,
    #     color=['r'],
    # )
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        list_dicts=[{'swe': i, 'rep': j}
                    for (i, j) in zip(best_ensemble_sweeps + [default_sweeps], best_ensemble_replicas + [default_replicas])],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        use_conf_interval=True,
        save_fig=False,
        ylim=[0.975, 1.0025],
        xlim=[5e2, 1e6],
        use_colorbar=False,
        linewidth=1.5,
        markersize=1,
        style='--',
        colormap=plt.cm.get_cmap('viridis'),
        colors=['colormap'],
    )
    plot_1d_singleinstance(
        df=df_progress_best,
        x_axis='R_budget',
        # y_axis='mean_' + stat_measure + '_perf_ratio',
        y_axis=stat_measure + '_median_perf_ratio',
        ax=ax,
        dict_fixed=None,
        # label_plot='Ordered exploration',
        # list_dicts=[{'R_budget': i}
        #             for i in R_budgets],
        labels=labels,
        prefix=prefix,
        label_plot='Random search',
        log_x=True,
        log_y=False,
        # colors=['colormap'],
        # colormap=plt.cm.get_cmap('tab10'),
        # use_colorbar=False,
        use_conf_interval=False,
        save_fig=False,
        linewidth=1.5,
        markersize=1,
        color=['m'],
    )
    # plot_1d_singleinstance_list(
    #     df=df_progress_ternary,
    #     x_axis='R_budget',
    #     y_axis=stat_measure + '_perf_ratio',
    #     ax=ax,
    #     dict_fixed={'pcold': 1, 'phot': 50},
    #     # label_plot='Ordered exploration',
    #     list_dicts=[{'run_per_solve': i}
    #                 for i in rs],
    #     labels=labels,
    #     prefix=prefix,
    #     log_x=True,
    #     log_y=False,
    #     colors=['colormap'],
    #     colormap=plt.cm.get_cmap('tab10'),
    #     use_colorbar=False,
    #     use_conf_interval=False,
    #     save_fig=False,
    #     ylim=[0.975, 1.0025],
    #     xlim=[1e2, 1e6],
    #     linewidth=1.5,
    #     markersize=10,
    #     style='.-',
    # )
    # ax.legend(loc='upper left', fontsize=10)
# %%
# Generate plots for inverse performance ratio of ensemble vs reads with best and worst performance
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='virt_best_inv_perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed={'pcold': 1, 'phot': 50},
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['k'],
    )
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='virt_worst_inv_perf_ratio',
        ax=ax,
        label_plot='Virtual worst',
        dict_fixed={'pcold': 1, 'phot': 50},
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        use_conf_interval=False,
        save_fig=False,
        linewidth=2.5,
        marker=None,
        color=['r'],
    )
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_inv_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        list_dicts=[{'swe': i}
                    for i in list(set(best_ensemble_sweeps)) + [default_sweeps]],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        use_conf_interval=False,
        save_fig=False,
        use_colorbar=False,
        linewidth=1.5,
        markersize=1,
        style='--',
        colormap=plt.cm.get_cmap('viridis'),
        colors=['colormap'],
    )
    plot_1d_singleinstance_list(
        df=df_best_random,
        x_axis='cum_reads',
        # y_axis='mean_' + stat_measure + '_inv_perf_ratio',
        y_axis=stat_measure + '_median_inv_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        # label_plot='Ordered exploration',
        list_dicts=[{'R_budget': i}
                    for i in R_budgets],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        colors=['colormap'],
        colormap=plt.cm.get_cmap('tab10'),
        use_colorbar=False,
        use_conf_interval=False,
        save_fig=False,
        linewidth=1.5,
        markersize=1,
    )
    plot_1d_singleinstance_list(
        df=df_progress_ternary,
        x_axis='cum_reads',
        y_axis=stat_measure + '_inv_perf_ratio',
        ax=ax,
        dict_fixed={'pcold': 1, 'phot': 50},
        # label_plot='Ordered exploration',
        list_dicts=[{'run_per_solve': i}
                    for i in rs],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        colors=['colormap'],
        colormap=plt.cm.get_cmap('tab10'),
        use_colorbar=False,
        use_conf_interval=False,
        save_fig=False,
        linewidth=1.5,
        markersize=10,
        style='.-',
        ylim=[9e-11, 1e0],
        xlim=[1e2, 1e6],
    )
# %%
# ECDF plot
# f,ax = plt.subplots()
# sns.ecdfplot(data=df_results_all[(df_results_all['schedule'] == 'geometric') & (df_results_all['swe'] == default_sweeps)][['reads','inv_perf_ratio', 'perf_ratio', 'instance']], y='perf_ratio', hue='instance', color='gray', ax=ax)
# sns.ecdfplot(data=df_results_all_stats[(df_results_all_stats['swe'] == default_sweeps)], y='mean_perf_ratio', ax=ax)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.set(ylim=[0.9,1.0025])

# %%
# Presentation figure!
# Plot to visualize best found strategy in random search
median_median_perf_ratio = df_progress_best['idxmedian_median_perf_ratio'].apply(pd.Series)
median_median_perf_ratio.columns=['R_budget','R_explor','tau']
median_median_perf_ratio['f_explor'] = median_median_perf_ratio['R_explor'] / median_median_perf_ratio['R_budget']
median_median_perf_ratio['median_median_perf_ratio'] = df_progress_best['median_median_perf_ratio']

f,ax = plt.subplots()
sns.scatterplot(
    data=median_median_perf_ratio,
    x='R_budget',
    size='f_explor',
    hue='f_explor',
    style='tau',
    y='median_median_perf_ratio',
    ax=ax,
    palette='magma',
    hue_norm=(0, 2),
    sizes=(20, 200),
    legend='brief')
# sns.lineplot(
#     data=df_progress_end,
#     x='R_budget',
#     style='run_per_solve',
#     hue='f_explor',
#     y='median_perf_ratio',
#     ax=ax,palette='magma',
#     legend='brief',
#     # units='experiment',
#     hue_norm=(0, 2),
#     estimator=np.median,
#     ci=None)
sns.lineplot(data=df_progress_best,x='R_budget',y='median_median_perf_ratio',ax=ax,estimator=None,ci=None,color='m', linewidth=2, label='Med best random search expl-expl')
sns.lineplot(data=df_virtual_best,x='reads',y='perf_ratio',ax=ax,estimator=None,ci=None,color='k', linewidth=2, label='Med Virtual best')
ax.set(xlim=[5e2,1e6])
ax.set(ylim=[0.975,1.0025])
ax.set(xscale='log')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
# Computing up ternary search across parameter for instance 42
# We assume that the performance of the parameter is unimodal (in decreases and the increases)
rs = [1, 5, 10]
df_name = "df_progress_ternary_42" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
# TODO: check that 'geometric' is replaced accross the code with default_schedule
default_schedule = 'geometric'
search_metric = 'perf_ratio'
compute_metric = 'perf_ratio'
if search_metric == 'tts':
    search_direction = -1  # -1 for decreasing, 1 for increasing
else:
    search_direction = 1
df_search = df_42[
    ['schedule', 'swe', 'boots', 'reads'] +
    list(set([compute_metric, search_metric]))
].sort_values([search_metric]).set_index(
    ['schedule', 'swe', 'boots']
)
use_raw_dataframes = False
if use_raw_dataframes or os.path.exists(df_path) is False:
    progress_list = []
    for r in rs:
        series_list = []
        lo = 0
        val_lo = sweeps_list[lo]
        up = len(sweeps_list) - 1
        val_up = sweeps_list[up]
        perf_lo = df_search.loc[
            idx[default_schedule, val_lo, r]][search_metric]
        perf_up = df_search.loc[
            idx[default_schedule, val_up, r]][search_metric]
        series_list.append(df_search.loc[
            idx[default_schedule, val_lo, r]])
        series_list.append(df_search.loc[
            idx[default_schedule, val_up, r]])
        while lo <= up:
            x1 = int(lo + (up - lo) / 3)
            x2 = int(up - (up - lo) / 3)
            val_x1 = sweeps_list[x1]
            perf_x1 = df_search.loc[
                idx[default_schedule, val_x1, r]][search_metric]
            series_list.append(df_search.loc[
                idx[default_schedule, val_x1, r]])
            val_x2 = sweeps_list[x2]
            perf_x2 = df_search.loc[
                idx[default_schedule, val_x2, r]][search_metric]
            series_list.append(df_search.loc[
                idx[default_schedule, val_x2, r]])
            if perf_x2 == perf_up:
                up -= 1
                val_up = sweeps_list[up]
                perf_up = df_search.loc[
                    idx[default_schedule, val_up, r]][search_metric]
                series_list.append(df_search.loc[
                    idx[default_schedule, val_up, r]])
            elif perf_x1 == perf_lo:
                lo += 1
                val_lo = sweeps_list[lo]
                perf_lo = df_search.loc[
                    idx[default_schedule, val_lo, r]][search_metric]
                series_list.append(df_search.loc[
                    idx[default_schedule, val_lo, r]])
            elif search_direction*perf_x1 > search_direction*perf_x2:
                up = x2
                val_up = sweeps_list[up]
                perf_up = df_search.loc[
                    idx[default_schedule, val_up, r]][search_metric]
            else:
                lo = x1
                val_lo = sweeps_list[lo]
                perf_lo = df_search.loc[
                    idx[default_schedule, val_lo, r]][search_metric]

        exploration_step = pd.concat(series_list, axis=1).T.rename_axis(
            ['schedule', 'swe', 'boots'])
        exploration_step[compute_metric] = exploration_step[compute_metric].expanding(
            min_periods=1).max()
        exploration_step.reset_index('boots', inplace=True)
        exploration_step['run_per_solve'] = r
        exploration_step['cum_reads'] = exploration_step.expanding(
            min_periods=1)['reads'].sum().reset_index(drop=True).values
        progress_list.append(exploration_step)

        # TODO: This can be further generalized, seach over df_search with indices now being the parameters
        if search_direction == 1:
            exploitation_step = df_search.reset_index().set_index(
                ['schedule', 'swe']).loc[exploration_step.nlargest(1, search_metric).index]
        else:
            exploitation_step = df_search.reset_index().set_index(
                ['schedule', 'swe']).loc[exploration_step.nsmallest(1, search_metric).index]
        exploitation_step['cum_reads'] = exploitation_step['reads'] + \
            exploration_step['cum_reads'].max()
        exploitation_step = exploitation_step[exploitation_step['cum_reads']
                                              <= default_reads*exploration_step.nlargest(1, 'perf_ratio').index.values[0][-1]
                                              ]
        # TODO this is not a very stable way to get the sweeps values
        exploitation_step.sort_values(['cum_reads'], inplace=True)
        exploitation_step[compute_metric].fillna(
            0, inplace=True)
        exploitation_step[compute_metric].clip(
            lower=exploration_step[compute_metric].max(), inplace=True)
        exploitation_step[compute_metric] = exploitation_step.expanding(
            min_periods=1).max()[compute_metric]
        exploitation_step['run_per_solve'] = r
        progress_list.append(exploitation_step)
    df_progress_ternary_42 = pd.concat(progress_list, axis=0)
    df_progress_ternary_42.reset_index(inplace=True)
    df_progress_ternary_42 = cleanup_df(df_progress_ternary_42)
    df_progress_ternary_42.to_pickle(df_path)
else:
    df_progress_ternary_42 = pd.read_pickle(df_path)

if 'inv_perf_ratio' not in df_progress_ternary_42.columns:
    df_progress_ternary_42['inv_perf_ratio'] = 1 - \
        df_progress_ternary_42['perf_ratio'] + EPSILON


# %%
# Exploration-exploitation with best parameters from ensemble
# TODO: We should compute full exploration-expliotation for this single instance and compare to what the ensemble recommmends
repetitions = 10  # Times to run the algorithm
# rs = [1, 5, 10]  # resources per parameter setting (runs)
# frac_r_exploration = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# R_budgets = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
df_name = "df_progress_42" + suffix + ".pkl"
df_path = os.path.join(results_path, df_name)
compute_metric = 'perf_ratio'
parameters = ['schedule', 'swe']
df_search = df_42[
    parameters + ['boots', 'reads'] + [compute_metric]
].set_index(
    parameters + ['boots']
)
use_raw_dataframes = False
if use_raw_dataframes or os.path.exists(df_path) is False:
    progress_list = []
    for index, row in best_random_search_idx.to_frame(index=False).iterrows():
        R_budget = row['R_budget']
        frac_r_exploration = row['frac_r_exploration']
        # TODO: run and read used interchangeably
        r = row['run_per_solve']
    # for R_budget in R_budgets:
    #     for frac_expl_total in frac_r_exploration:
        R_exploration = int(R_budget*frac_expl_total)
        # budget for exploitation (runs)
        R_exploitation = R_budget - R_exploration
        # for r in rs:
        for experiment in range(repetitions):
            random_sweeps = np.random.choice(
                sweeps_list, size=int(R_exploration / (r*default_sweeps)), replace=True)
            # % Question: Should we replace these samples?
            if r*default_sweeps > R_exploration:
                print(
                    "R_exploration must be larger than single exploration step")
                continue
            series_list = []
            total_reads = 0
            for sweeps in random_sweeps:
                series_list.append(df_search.loc[
                    idx[default_schedule, sweeps, r]]
                )
                total_reads += r
                if total_reads > R_exploration:
                    converged = True
                    break
            exploration_step = pd.concat(series_list, axis=1).T.rename_axis(
                parameters + ['boots'])
            exploration_step[compute_metric] = exploration_step[compute_metric].expanding(
                min_periods=1).max()
            exploration_step.reset_index('boots', inplace=True)
            exploration_step['experiment'] = experiment
            exploration_step['run_per_solve'] = r
            exploration_step['R_explor'] = R_exploration
            exploration_step['R_exploit'] = R_exploitation
            exploration_step['cum_reads'] = exploration_step.groupby('experiment').expanding(
                min_periods=1)['reads'].sum().reset_index(drop=True).values
            progress_list.append(exploration_step)

            exploitation_step = df_search.reset_index().set_index(
                parameters).loc[exploration_step.nlargest(1, compute_metric).index]
            exploitation_step['cum_reads'] = exploitation_step['reads'] + \
                exploration_step['cum_reads'].max()
            exploitation_step.sort_values(['cum_reads'], inplace=True)
            exploitation_step = exploitation_step[exploitation_step['cum_reads'] <= R_budget]
            exploitation_step[compute_metric].fillna(
                0, inplace=True)
            exploitation_step[compute_metric].clip(
                lower=exploration_step[compute_metric].max(), inplace=True)
            exploitation_step[compute_metric] = exploitation_step[compute_metric].expanding(
                min_periods=1).max()
            exploitation_step['experiment'] = experiment
            exploitation_step['run_per_solve'] = r
            exploitation_step['R_explor'] = R_exploration
            exploitation_step['R_exploit'] = R_exploitation
            progress_list.append(exploitation_step)
    df_progress_total_42 = pd.concat(progress_list, axis=0)
    df_progress_total_42.reset_index(inplace=True)
    df_progress_total_42.to_pickle(df_path)
else:
    df_progress_total_42 = pd.read_pickle(df_path)

if 'R_budget' not in df_progress_total_42.columns:
    df_progress_total_42['R_budget'] = df_progress_total_42['R_explor'] + \
        df_progress_total_42['R_exploit']

if 'best_perf_ratio' not in df_progress_total_42.columns:
    df_progress_total_42['inv_perf_ratio'] = 1 - \
        df_progress_total_42['perf_ratio'] + EPSILON
    df_progress_total_42['best_inv_perf_ratio'] = df_progress_total_42.sort_values(
        ['cum_reads', 'R_budget']
    ).expanding(min_periods=1).min()['inv_perf_ratio']
    df_progress_total_42['best_perf_ratio'] = 1 - \
        df_progress_total_42['best_inv_perf_ratio'] + EPSILON
df_progress_total_42 = cleanup_df(df_progress_total_42)
df_progress_total_42.to_pickle(df_path)

# %%
# Evaluate instance 42 with strategies learned from ensemble anaylsis
# Plot with performance ratio vs reads for interesting sweeps
instance = 42
interesting_sweeps = list(set([
    int(df_results_all[(df_results_all['boots'] == default_boots) & (df_results_all['instance'] == instance)].nsmallest(1, 'tts')[
        'swe'].values[0]),
    default_sweeps,
] + best_ensemble_sweeps))
f, ax = plt.subplots()
# random_plot = sns.lineplot(
#     data=df_progress_total_42,
#     x='cum_reads',
#     y='perf_ratio',
#     hue='R_budget',
#     estimator='median',
#     ci=None,
#     ax=ax,
#     # palette=sns.color_palette('rainbow', len(R_budgets)),
#     legend=[str(i) for i in R_budgets],
#     linewidth=1.5,
# )
# random_plot.legend(labels=['R_bu'+str(i) for i in R_budgets])
plot_1d_singleinstance(
    df=df_progress_total_42,
    x_axis='cum_reads',
    y_axis='best_perf_ratio',
    dict_fixed={
        'schedule': 'geometric'
    },
    ax=ax,
    label_plot='Best random exploration-exploitation',
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict.update({'instance': instance}),
    ylim=[0.975, 1.0025],
    xlim=[1e3, 1e6*1.1],
    linewidth=2.5,
)
plot_1d_singleinstance_list(
    df=df_results_all,
    x_axis='reads',
    y_axis='perf_ratio',
    dict_fixed={
        'instance': instance,
        'schedule': 'geometric'
    },
    ax=ax,
    list_dicts=[{'swe': i}
                for i in interesting_sweeps],
    labels=labels,
    prefix=prefix,
    save_fig=False,
    log_x=True,
    log_y=False,
    use_conf_interval=False,
    default_dict=default_dict.update({'instance': instance}),
    use_colorbar=False,
    ylim=[0.975, 1.0025],
    xlim=[1e3, 5e4],
)
# plot_1d_singleinstance_list(
#     df=df_progress_ternary_42,
#     x_axis='cum_reads',
#     y_axis='perf_ratio',
#     ax=ax,
#     dict_fixed={'schedule': default_schedule},
#     # label_plot='Ordered exploration',
#     list_dicts=[{'run_per_solve': i}
#                 for i in rs],
#     labels=labels,
#     prefix=prefix,
#     log_x=True,
#     log_y=False,
#     colors=['colormap'],
#     colormap=plt.cm.get_cmap('tab10'),
#     use_colorbar=False,
#     use_conf_interval=False,
#     save_fig=False,
#     linewidth=1.5,
#     markersize=10,
#     style='.-',
#     ylim=[0.975, 1.0025],
#     xlim=[1e3, 1e6*1.1],
# )
# %%
# Evaluate instance 42 with strategies learned from ensemble anaylsis
# Plot with inverse performance ratio vs reads for interesting sweeps
f, ax = plt.subplots()
# random_plot = sns.lineplot(
#     data=df_progress_total_42,
#     x='cum_reads',
#     y='inv_perf_ratio',
#     hue='R_budget',
#     estimator='median',
#     ci='sd',
#     ax=ax,
#     # palette=sns.color_palette('rainbow', len(R_budgets)),
#     # legend=None,
#     linewidth=2,
# )
# random_plot.legend(labels=['R_bu'+str(i) for i in R_budgets])
plot_1d_singleinstance(
    df=df_progress_total_42,
    x_axis='cum_reads',
    y_axis='best_inv_perf_ratio',
    dict_fixed={
        'schedule': 'geometric'
    },
    ax=ax,
    label_plot='Best random exploration-exploitation',
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict.update({'instance': instance}),
    # ylim=[0.975, 1.0025],
    xlim=[1e3, 5e4],
    linewidth=2.5,
)
plot_1d_singleinstance_list(
    df=df_results_all,
    x_axis='reads',
    y_axis='inv_perf_ratio',
    dict_fixed={
        'instance': instance,
        'schedule': 'geometric'
    },
    ax=ax,
    list_dicts=[{'swe': i}
                for i in best_ensemble_sweeps],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    use_conf_interval=False,
    save_fig=False,
    default_dict=default_dict.update({'instance': instance}),
    use_colorbar=False,
    # ylim=[0.975, 1.0025],
    xlim=[1e3, 5e4],
)
plot_1d_singleinstance_list(
    df=df_progress_ternary_42,
    x_axis='cum_reads',
    y_axis='inv_perf_ratio',
    ax=ax,
    dict_fixed={'schedule': default_schedule},
    # label_plot='Ordered exploration',
    list_dicts=[{'run_per_solve': i}
                for i in rs],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    colors=['colormap'],
    colormap=plt.cm.get_cmap('tab10'),
    use_colorbar=False,
    use_conf_interval=False,
    save_fig=False,
    linewidth=1.5,
    markersize=10,
    style='.-',
    # ylim=[0.975, 1.0025],
    xlim=[5e3, 5e4],
)
# %%