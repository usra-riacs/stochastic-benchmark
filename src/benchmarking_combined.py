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

EPSILON = 1e-10

#%%
# %%
# Define default values

default_sweeps = 1000
total_reads = 1000
float_type = 'float32'
default_reads = 1000
default_boots = default_reads
total_reads = 1000
# TODO rename this total_reads parameter, remove redundancy with above
default_Tfactor = 1.0
default_schedule = 'geometric'
default_replicas = 1
default_p_hot = 50.0
default_p_cold = 1.0
parameters = ['schedule', 'sweeps', 'Tfactor']
suffix = 'C'
# %%
# Function to compute uniform statistics
def simpleCreateDnealResultsDataframes(
    df: pd.DataFrame = None,
    instance_list: List[int] = [0],
    parameters_dict: dict = None,
    parameters_list: list = None,
    boots_list: List[int] = [1000],
    data_path: str = None,
    results_path: str = None,
    pickle_path: str = None,
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
    prefix: str = '',
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
    # Create list of parameters
    # params = list(parameters_dict.keys())
    # Sort it alphabetically (ignoring uppercase)
    # params.sort(key=str.lower)
    # We will fix this for the moment
    params = ['schedule', 'sweeps', 'Tfactor']
    # Check that the parameters are columns in the dataframe
    if df is not None:
        assert all([i in df.columns for i in params])

    # Remove repeated elements in the parameters_dict values (sets)
    if parameters_dict is not None:
        for i, j in parameters_dict.items():
            parameters_dict[i] = set(j)
            # In case that the data is already in the dataframe, return it
            if all([k in df[i].values for (i, j) in parameters_dict.items() for k in j]):
                print('The dataframe has some data for the parameters')
                # The parameters dictionary has lists as values as the loop below makes the concatenation faster than running the loop for each parameter
                cond = [df[k].isin(v).astype(bool)
                        for k, v in parameters_dict.items()]
                cond_total = functools.reduce(lambda x, y: x & y, cond)
                if not all(boots in df[cond_total]['boots'].values for boots in boots_list):
                    print('The dataframe has missing bootstraps')
                else:
                    print('The dataframe already has all the data')
                    return df
            parameter_sets = itertools.product(
                *(parameters_dict[Name] for Name in parameters_dict))
            parameter_sets = list(parameter_sets)
    
    if parameters_list is not None:
        parameter_sets = parameters_list


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
            for parameter_set in parameter_sets:
                list_inputs = [instance] + [i for i in parameter_set]
                parameters = dict(zip(params, parameter_set))

                df_samples_name = 'df_' + prefix + str(instance) + "_"
                + '_'.join(str(vals) for vals in parameters.values()) + suffix + '.pkl'
                df_path = os.path.join(pickle_path, df_samples_name)
                if os.path.exists(df_path):
                    df_samples = pd.read_pickle(df_path)
                else:
                    # Main change is that if we do not find the file, we just don't load it
                    print('The pickle file does not exist')
                    df_samples = None

                for boots in boots_list:

                    # TODO Good place to replace with mask and isin1d()
                    # This generated the undersampling using bootstrapping, filtering by all the parameters values
                    if (df is not None) and \
                            (boots in df.loc[(df[list(parameters)] == pd.Series(parameters)).all(axis=1)]['boots'].values):
                        continue
                    else:
                        print("Generating results for instance:", instance,
                          ','.join(str(key) + ':' + str(val) for key,val in parameters.items()), boots)
                        print('Let us not do anything yet')
                        continue
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
# Function to generate stats aggregated dataframe
# TODO: this can be generalized by acknowledging that the boots are the resource R


def simpleGenerateStatsDataframe(
    df_all: List[dict] = None,
    stat_measures: List[str] = ['mean', 'median'],
    metrics_list: List[str] = None,
    instance_list: List[str] = None,
    parameters_dict: dict = None,
    parameters_list: list = None,
    lower_bounds: dict = None,
    upper_bounds: dict = None,
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
    df_all = simpleCreateDnealResultsDataframes(
        df=df_all,
        instance_list=instance_list,
        parameters_dict=parameters_dict,
        parameters_list=parameters_list,
        boots_list=resource_list,
        data_path=data_path,
        results_path=results_path,
        pickle_path=pickles_path,
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
    parameters = list(parameters_dict.keys())
    resources = ['boots']
    df_name = 'df_results_stats' + suffix
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

metrics_list = ['min_energy', 'tts',
                'perf_ratio', 'success_prob', 'mean_time', 'inv_perf_ratio']
bootstrap_iterations = 1000

# These lists of parameters are obsolte, they will be created from the index lists
sweeps_list = [i for i in range(1, 21, 1)] + [
    i for i in range(20, 201, 10)] + [
    i for i in range(200, 501, 20)] + [
    i for i in range(500, 1001, 100)]
schedules_list = ['geometric', 'linear']


parameters_dict = {
    'schedule': schedules_list,
    'sweeps': sweeps_list,
    'Tfactor': [default_Tfactor],
}

# This list is more complicated. In all honesty it should be uniform for all the parameter settings, but somethines we have more data for some
boots_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


instance_list = [i for i in range(20)] + [42]
training_instance_list = [i for i in range(20)]


# %%
# Parameters for the newly simulated instances
gap = 1
s = 0.99
conf_int = 68
fail_value = np.inf
ocean_df_flag = True

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
    'best_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
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
    'best_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    # 'tts': 'TTS to GS with 99% confidence \n [s * replica] ~ [MVM]',
}

# %%
# Join previous result dataframes
suffixes = ['','T','t']
current_path = os.getcwd()
data_path = os.path.join(current_path, '../data/sk/')
results_path = os.path.join(data_path, 'dneal/')
df_list = []
for suffix in suffixes:
    df_name = 'df_results' + suffix + '.pkl'
    df_path = os.path.join(results_path, df_name)
    df = pd.read_pickle(df_path)
    df_list.append(df)

df_results_all = pd.concat(df_list, axis=0)

# %%
parameters_list = df_results_all.sort_values('perf_ratio').drop_duplicates(subset=['schedule','sweeps','Tfactor','instance','boots']).set_index(['schedule','sweeps','Tfactor','instance','boots']).index.to_list()
parameters_dummy = {
    'schedule': [default_schedule],
    'sweeps': [default_sweeps],
    'Tfactor': [default_Tfactor],
}
# %%
# Generate stats results
use_raw_full_dataframe = False
use_raw_dataframes = False
use_raw_sample_pickles = False
overwrite_pickles = False

# TODO: this function assume we want all the combinatios of the parameters, while in reality, we might want to use those in a list
df_results_all_stats = simpleGenerateStatsDataframe(
    df_all=df_results_all,
    stat_measures=['mean', 'median'],
    instance_list=training_instance_list,
    parameters_dict=parameters_dummy,
    parameters_list=parameters_list,
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
)
# %%

# %%
