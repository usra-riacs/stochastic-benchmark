# %%
# Import the Dwave packages dimod and neal
import functools
import itertools
import os
import pickle
import pickle as pkl
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
from matplotlib import ticker
from pysa.sa import Solver
from scipy import sparse, stats
import seaborn as sns

from plotting import *
from retrieve_data import *
from do_dneal import *
from util_benchmark import *

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
results_path = os.path.join(current_path, '../data/sk/')
if not(os.path.exists(results_path)):
    print('Results directory ' + results_path +
          ' does not exist. We will create it.')
    os.makedirs(results_path)

dneal_results_path = os.path.join(results_path, 'dneal/')
if not(os.path.exists(dneal_results_path)):
    print('Dwave-neal results directory ' + dneal_results_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_results_path)

dneal_pickle_path = os.path.join(dneal_results_path, 'pickles/')
if not(os.path.exists(dneal_pickle_path)):
    print('Dwave-neal pickles directory' + dneal_pickle_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_pickle_path)

instance_path = os.path.join(results_path, 'instances/')
if not(os.path.exists(instance_path)):
    print('Instances directory ' + instance_path +
          ' does not exist. We will create it.')
    os.makedirs(instance_path)

plots_path = os.path.join(results_path, 'plots/')
if not(os.path.exists(plots_path)):
    print('Plots directory ' + plots_path +
          ' does not exist. We will create it.')
    os.makedirs(plots_path)
# %%
# Create instance 42
model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

prefix = "random_n_" + str(N) + "_inst_"
# TODO: this is prefix for file but we should get a nicer description for the plots
instance_name = prefix + str(instance)


# %%
# Compute random sample on problem and print average energy
random_energy, random_sample = randomEnergySampler(
    model_random, num_reads=1000, dwave_sampler=True)
df_random_sample = random_sample.to_pandas_dataframe(sample_column=True)
print('Average random energy = ' + str(random_energy))


# %%
# Run default Dwave-neal simulated annealing implementation
sim_ann_sampler = dimod.SimulatedAnnealingSampler()
default_sweeps = 1000
total_reads = 1000
default_reads = 1000
default_boots = default_reads
# TODO: check that 'geometric' is replaced accross the code with default_schedule
default_schedule = 'geometric'
default_name = prefix + str(instance) + '_' + default_schedule + '_' + \
    str(default_sweeps) + '.p'
df_default_name = 'df_' + default_name + 'kl'
rerun_default = False
if not os.path.exists(os.path.join(dneal_pickle_path, default_name)) or rerun_default:
    print('Running default D-Wave-neal simulated annealing implementation')
    start = time.time()
    default_samples = sim_ann_sampler.sample(
        model_random,
        num_reads=default_reads,
        num_sweeps=default_sweeps,)
    time_default = time.time() - start
    default_samples.info['timing'] = time_default
    with open(os.path.join(dneal_pickle_path, default_name), 'wb') as f:
        pickle.dump(default_samples, f)
else:
    with open(os.path.join(dneal_pickle_path, default_name), 'rb') as f:
        default_samples = pickle.load(f)
df_default_samples = default_samples.to_pandas_dataframe(sample_column=True)
df_default_samples['runtime (us)'] = int(
    1e6*default_samples.info['timing']/len(df_default_samples.index))
df_default_samples.to_pickle(os.path.join(dneal_pickle_path, df_default_name))
min_energy = df_default_samples['energy'].min()
print(min_energy)
# %%
# Load zipped results if using raw data
overwrite_pickles = False
use_raw_dataframes = True

# %%
# Function to generate samples dataframes or load them otherwise


def createDnealSamplesDataframeTemp(
    instance: int = 42,
    parameters: dict = None,
    total_reads: int = 1000,
    sim_ann_sampler=None,
    dneal_pickle_path: str = None,
    use_raw_dneal_pickles: bool = False,
    overwrite_pickles: bool = False,
) -> pd.DataFrame:
    '''
    Creates a dataframe with the samples for the dneal algorithm.

    Args:
        instance: The instance to load/create the samples for.
        parameters: The parameters to use for the dneal algorithm.
        schedule: The schedule to use for the dneal algorithm.
        sweep: The number of sweeps to use for the dneal algorithm.
        total_reads: The total number of reads to use for the dneal algorithm.
        sim_ann_sampler: The sampler to use for the simulated annealing algorithm.
        dneal_pickle_path: The path to the pickle files.
        use_raw_dneal_pickles: Whether to use the raw pickles or not.
        overwrite_pickles: Whether to overwrite the pickles or not.

    Returns:
        The dataframe with the samples for the dneal algorithm.
    '''
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
        '_'.join(str(vals) for vals in parameters.values()) + "T.p"
    df_samples_name = 'df_' + dict_pickle_name + 'kl'
    df_path = os.path.join(dneal_pickle_path, df_samples_name)
    if os.path.exists(df_path):
        try:
            df_samples = pd.read_pickle(df_path)
            return df_samples
        except (pkl.UnpicklingError, EOFError):
            os.replace(df_path, df_path + '.bak')
    if use_raw_dneal_pickles or not os.path.exists(df_path):
        # If you want to generate the data or load it here
        if sim_ann_sampler is None:
            sim_ann_sampler = neal.SimulatedAnnealingSampler()
        # Gather instance paths
        dict_pickle_name = os.path.join(dneal_pickle_path, dict_pickle_name)
        # If the instance data exists, load the data
        if os.path.exists(dict_pickle_name) and not overwrite_pickles:
            # print(pickle_name)
            samples = pickle.load(open(dict_pickle_name, "rb"))
            # Load dataframes
            # df_samples = pd.read_pickle(df_path)
        # If it does not exist, generate the data
        else:
            # TODO: We should be loading the data instead of regenerating it here
            # Fixing the random seed to get the same result
            np.random.seed(instance)
            J = np.random.rand(N, N)
            # We only consider upper triangular matrix ignoring the diagonal
            J = np.triu(J, 1)
            h = np.random.rand(N)
            model_random = dimod.BinaryQuadraticModel.from_ising(
                h, J, offset=0.0)
            start = time.time()
            samples = sim_ann_sampler.sample(
                model_random,
                num_reads=total_reads,
                num_sweeps=parameters['sweep'],
                beta_schedule_type=parameters['schedule'],
                beta_range=(default_samples.info['beta_range'][0]*parameters['Tfactor'],
                            default_samples.info['beta_range'][1]),
            )
            time_s = time.time() - start
            samples.info['timing'] = time_s
            pickle.dump(samples, open(dict_pickle_name, "wb"))
        # Generate Dataframes
        df_samples = samples.to_pandas_dataframe(sample_column=True)
        df_samples['runtime (us)'] = int(
            1e6*samples.info['timing']/len(df_samples.index))
        df_samples.to_pickle(df_path)

    return df_samples


# %%
# Function to update the dataframes
# TODO Remove all the list_* variables and name them as plurals instead
def createDnealResultsDataframes(
    df: pd.DataFrame = None,
    instance_list: List[int] = [0],
    parameters_dict: dict = None,
    boots_list: List[int] = [1000],
    dneal_results_path: str = None,
    dneal_pickle_path: str = None,
    use_raw_dataframes: bool = False,
    use_raw_dneal_pickles: bool = False,
    overwrite_pickles: bool = False,
    confidence_level: float = 68,
    gap: float = 1.0,
    bootstrap_iterations: int = 1000,
    s: float = 0.99,
    fail_value: float = np.inf,
    save_pickle: bool = True,
) -> pd.DataFrame:
    '''
    Function to create the dataframes for the experiments

    Args:
        df: The dataframe to be updated
        instance: The instance number
        boots: The number of bootstraps
        parameters_dict: The parameters dictionary with values as lists
        dneal_results_path: The path to the results
        dneal_pickle_path: The path to the pickle files
        use_raw_dataframes: If we want to use the raw data
        overwrite_pickles: If we want to overwrite the pickle files

    '''
    # Remove repeated elements in the parameters_dict values (sets)
    if parameters_dict is not None:
        for i, j in parameters_dict.items():
            parameters_dict[i] = set(j)
    # Create list of parameters sorted alphabetically (ingoring uppercase)
    params = list(parameters_dict.keys())
    params.sort(key=str.lower)
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
        df_name = "df_resultsT.pkl"
    else:
        df_name = "df_results_" + str(instance_list[0]) + "T.pkl"
    df_path = os.path.join(dneal_results_path, df_name)

    # If use_raw_dataframes compute the row
    if use_raw_dataframes or not os.path.exists(df_path):
        # TODO Remove all the list_* variables and name them as plurals instead
        list_results_dneal = []
        for instance in instance_list:
            random_energy = loadEnergyFromFile(os.path.join(
                results_path, 'random_energies.txt'), prefix + str(instance))
            min_energy = loadEnergyFromFile(os.path.join(
                results_path, 'gs_energies.txt'), prefix + str(instance))
            # We will assume that the insertion order in the keys is preserved (hence Python3.7+ only) and is sorted alphabetically
            combinations = itertools.product(
                *(parameters_dict[Name] for Name in parameters_dict))
            combinations = list(combinations)
            for combination in combinations:
                list_inputs = [instance] + [i for i in combination]
                # Question: Is there a way of extracting the parameters names as variables names from the dictionary keys?
                # For the moment, let's hard-code it
                schedule = combination[0]
                Tfactor = combination[1]
                # parameters = {'schedule': schedule,
                #                 # 'sweep': sweep,
                #               'Tfactor': Tfactor,
                #               }

                parameters = dict(zip(params, combination))
                df_samples = createDnealSamplesDataframeTemp(
                    instance=instance,
                    parameters=parameters,
                    total_reads=total_reads,
                    sim_ann_sampler=sim_ann_sampler,
                    dneal_pickle_path=dneal_pickle_path,
                    use_raw_dneal_pickles=use_raw_dneal_pickles,
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
                        )
                    list_results_dneal.append(
                        list_inputs + list_outputs)

        df_results_dneal = pd.DataFrame(list_results_dneal, columns=[
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
                [df, df_results_dneal], axis=0, ignore_index=True)
        else:
            df_new = df_results_dneal.copy()

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

# Define default behavior for the solver
total_reads = 1000
# TODO rename this total_reads parameter, remove redundancy with above
default_reads = 1000
default_sweeps = 1000
default_Tfactor = 1.0
default_schedule = 'geometric'
default_boots = total_reads
# %%
# Function to generate stats aggregated dataframe
# TODO: this can be generalized by acknowledging that the boots are the resource R


def generateStatsDataframe(
    df_all: List[dict] = None,
    stat_measures: List[str] = ['mean', 'median'],
    instance_list: List[str] = None,
    parameters_dict: dict = None,
    resource_list: List[int] = [default_boots],
    dneal_results_path: str = None,
    use_raw_full_dataframe: bool = False,
    use_raw_dataframes: bool = False,
    use_raw_dneal_pickles: bool = False,
    overwrite_pickles: bool = False,
    s: float = 0.99,
    confidence_level: float = 0.68,
    bootstrap_iterations: int = 1000,
    gap: float = 0.1,
    fail_value: float = None,
    save_pickle: bool = True,
) -> pd.DataFrame:
    '''
    Function to generate statistics from the aggregated dataframe

    Args:
        df_all: List of dictionaries containing the aggregated dataframe
        stat_measures: List of statistics to be calculated
        instance_list: List of instances to be considered
        parameters_dict: Dictionary of parameters to be considered, with list as values
        resource_list: List of resources to be considered
        dneal_results_path: Path to the directory containing the results
        use_raw_full_dataframe: If True, the full dataframe is used
        use_raw_dataframes: Boolean indicating whether to use the raw data for generating the aggregated dataframe
        use_raw_dneal_pickles: Boolean indicating whether to use the raw pickles for generating the aggregated pickles
        overwrite_pickles: Boolean indicating whether to overwrite the pickles
        s: The success factor (usually said as RTT within s% probability).
        confidence_level: Confidence level for the confidence intervals in bootstrapping
        bootstrap_iterations: Number of bootstrap iterations
        gap: The threshold for the considering a read successful w.r.t the performance ratio [%].
        fail_value: Value to be used for failed runs
        save_pickle: Boolean indicating whether to save the aggregated pickle
    '''
    df_all = createDnealResultsDataframes(
        df=df_all,
        instance_list=instance_list,
        parameters_dict=parameters_dict,
        boots_list=resource_list,
        dneal_results_path=dneal_results_path,
        dneal_pickle_path=dneal_pickle_path,
        use_raw_dataframes=use_raw_dataframes,
        use_raw_dneal_pickles=use_raw_dneal_pickles,
        overwrite_pickles=overwrite_pickles,
        s=s,
        confidence_level=confidence_level,
        bootstrap_iterations=bootstrap_iterations,
        gap=gap,
        fail_value=fail_value,
        save_pickle=True,
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
    df_name = 'df_results_statsT'
    df_path = os.path.join(dneal_results_path, df_name + '.pkl')
    df_all_stats = pd.read_pickle(df_path)
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
# Compute results for instance 42 using D-Wave Neal
use_raw_dataframes = True
use_raw_dneal_pickles = False
overwrite_pickles = False
instance = 42
metrics_list = ['min_energy', 'tts',
                'perf_ratio', 'success_prob', 'mean_time', 'inv_perf_ratio']
Tfactor_list = list(np.logspace(-1, 3, 35))
schedules_list = ['geometric', 'linear']
# schedules_list = ['geometric']
bootstrap_iterations = 1000
s = 0.99  # This is the success probability for the TTS calculation
gap = 1.0  # This is a percentual treshold of what the minimum energy should be
conf_int = 68  #
fail_value = np.inf
# Confidence interval for bootstrapping, value used to get standard deviation
confidence_level = 68
boots_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
# TODO there should be an attribute to the parameters and if they vary logarithmically, have a function that generates the list of values "equally" spaced in logarithmic space
sim_ann_sampler = neal.SimulatedAnnealingSampler()

df_name = "df_results_" + str(instance) + "T.pkl"
df_path = os.path.join(dneal_results_path, df_name)
if os.path.exists(df_path) and False:
    df_dneal_42 = pd.read_pickle(df_path)
else:
    df_dneal_42 = None

df_dneal_42 = createDnealResultsDataframes(
    df=df_dneal_42,
    instance_list=[instance],
    parameters_dict={'schedule': schedules_list, 'Tfactor': Tfactor_list},
    boots_list=boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_dneal_pickles=use_raw_dneal_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
)

# %%
# Define plot longer labels
labels = {
    'N': 'Number of variables',
    'instance': 'Random instance',
    'replicas': 'Number of replicas',
    'sweeps': 'Number of sweeps',
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
# Performance ratio vs sweeps for different bootstrap downsamples
default_dict = {'schedule': default_schedule,
                'Tfactor': default_Tfactor, 'boots': default_boots}
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='Tfactor',
    y_axis='perf_ratio',
    dict_fixed={'instance': 42, 'schedule': 'geometric'},
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict.update(
        {'instance': 42, 'reads': default_sweeps*default_boots}),
    use_colorbar=False,
    ylim=[0.9, 1.01],
)
# %%
# Inverse performance ratio vs sweeps for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='Tfactor',
    y_axis='inv_perf_ratio',
    dict_fixed={'instance': 42, 'schedule': 'geometric'},
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
)
# %%
# Performance ratio vs runs for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='reads',
    y_axis='perf_ratio',
    dict_fixed={'instance': 42, 'schedule': 'geometric'},
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    ylim=[0.95, 1.005],
)
# %%
# Performance ratio vs runs for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='reads',
    y_axis='inv_perf_ratio',
    dict_fixed={'instance': 42, 'schedule': 'geometric'},
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    # ylim=[0.95, 1.005]
)
# %%
# Mean time plot of some fixed parameter setting
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='Tfactor',
    y_axis='mean_time',
    dict_fixed={'instance': 42, 'boots': 1000},
    ax=ax,
    list_dicts=[{'schedule': i} for i in schedules_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
)
# %%
# Success probability of some fixed parameter setting
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='Tfactor',
    y_axis='success_prob',
    dict_fixed={'instance': 42, 'boots': 1000},
    ax=ax,
    list_dicts=[{'schedule': i} for i in schedules_list],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
)
# %%
# TTS Plot for all bootstrapping downsamples in both schedules
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='Tfactor',
    y_axis='tts',
    ax=ax,
    dict_fixed={'instance': 42},
    list_dicts=[{'schedule': i, 'boots': j}
                for i in schedules_list for j in [10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
)

# %%
# TTS Plot for all bootstrapping downsamples in both schedules
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='Tfactor',
    y_axis='tts',
    ax=ax,
    dict_fixed={'schedule': 'geometric', 'instance': 42},
    list_dicts=[{'boots': j}
                for j in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
)
# %%
# Loop over the dataframes with 4 values of TFactors and a sweep in boots then compute the results, and complete by creating main Dataframe
interesting_Tfactors = [
    df_dneal_42[df_dneal_42['boots'] == default_boots].nsmallest(1, 'tts')[
        'Tfactor'].values[0],
    0.1,
    10,
    100,
    1000,
    1,
]

# Iterating for all values of bootstrapping downsampling proves to be very expensive, rather use steps of 10
all_boots_list = list(range(1, 1001, 1))

df_dneal_42 = createDnealResultsDataframes(
    df=df_dneal_42,
    instance_list=[instance],
    parameters_dict={'schedule': schedules_list,
                     'Tfactor': interesting_Tfactors},
    boots_list=all_boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_dneal_pickles=use_raw_dneal_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
)

# %%
# Plot with performance ratio vs reads for interesting sweeps
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='reads',
    y_axis='perf_ratio',
    # instance=42,
    dict_fixed={
        'instance': 42,
        # 'schedule':'geometric'
    },
    ax=ax,
    list_dicts=[{'Tfactor': i, 'schedule': j}
                for j in schedules_list for i in interesting_Tfactors],
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
    df=df_dneal_42,
    x_axis='reads',
    y_axis='perf_ratio',
    # instance=42,
    dict_fixed={
        'instance': 42,
        'schedule': 'geometric'
    },
    ax=ax,
    list_dicts=[{'Tfactor': i} for i in interesting_Tfactors],
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
# Plot with inverse performance ratio vs reads for interesting sweeps
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='reads',
    y_axis='inv_perf_ratio',
    # instance=42,
    dict_fixed={
        'instance': 42,
        'schedule': 'geometric'
    },
    ax=ax,
    list_dicts=[{'Tfactor': i} for i in interesting_Tfactors],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    colors=['colormap'],
    # ylim=[0.95, 1.005],
    xlim=[1e3, default_sweeps*default_boots*1.1],
)
# %%
# Compute all instances with Dwave-neal
instance_list = [i for i in range(20)] + [42]
training_instance_list = [i for i in range(20)]
# %%
# Merge all results dataframes in a single one
schedules_list = ['geometric']
df_list = []
use_raw_dataframes = True
use_raw_dneal_pickles = False
all_boots_list = list(range(1, 1001, 1))
for instance in instance_list:
    df_name = "df_results_" + str(instance) + "T.pkl"
    df_path = os.path.join(dneal_results_path, df_name)
    if os.path.exists(df_path) and False:
        df_results_dneal = pd.read_pickle(df_path)
    else:
        df_results_dneal = None
    df_results_dneal = createDnealResultsDataframes(
        df=df_results_dneal,
        instance_list=[instance],
        parameters_dict={'schedule': schedules_list, 'Tfactor': Tfactor_list},
        boots_list=boots_list,
        dneal_results_path=dneal_results_path,
        dneal_pickle_path=dneal_pickle_path,
        use_raw_dataframes=use_raw_dataframes,
        use_raw_dneal_pickles=use_raw_dneal_pickles,
        overwrite_pickles=overwrite_pickles,
        s=s,
        confidence_level=conf_int,
        bootstrap_iterations=bootstrap_iterations,
        gap=gap,
        fail_value=fail_value,
        save_pickle=True,
    )

    # Loop over the dataframes with 4 values of sweeps and a sweep in boots then compute the results, and complete by creating main Dataframe
    interesting_Tfactors = [
        df_results_dneal[df_results_dneal['boots'] == default_boots].nsmallest(1, 'tts')[
            'Tfactor'].values[0],
        0.1,
        10,
        100,
        1000,
        1,
    ]

    df_results_dneal = createDnealResultsDataframes(
        df=df_results_dneal,
        instance_list=[instance],
        parameters_dict={'schedule': schedules_list,
                         'Tfactor': interesting_Tfactors},
        boots_list=all_boots_list,
        dneal_results_path=dneal_results_path,
        dneal_pickle_path=dneal_pickle_path,
        use_raw_dataframes=use_raw_dataframes,
        use_raw_dneal_pickles=use_raw_dneal_pickles,
        overwrite_pickles=overwrite_pickles,
        s=s,
        confidence_level=conf_int,
        bootstrap_iterations=bootstrap_iterations,
        gap=gap,
        fail_value=fail_value,
        save_pickle=True,
    )

    df_list.append(df_results_dneal)

df_results_all = pd.concat(df_list, ignore_index=True)
df_name = "df_resultsT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_results_all = cleanup_df(df_results_all)
df_results_all.to_pickle(df_path)

# %%
# Run all the instances with Dwave-neal
overwrite_pickles = False
use_raw_dataframes = True
use_raw_dneal_pickles = False
# schedules_list = ['geometric', 'linear']
schedules_list = ['geometric']

df_results_all = createDnealResultsDataframes(
    df=df_results_all,
    instance_list=instance_list,
    parameters_dict={'schedule': schedules_list, 'Tfactor': Tfactor_list},
    boots_list=boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_dneal_pickles=use_raw_dneal_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
)
# %%
# Generate stats results
use_raw_full_dataframe = True
use_raw_dataframes = False
use_raw_dneal_pickles = False
df_results_all_stats = generateStatsDataframe(
    df_all=df_results_all,
    stat_measures=['mean', 'median'],
    instance_list=training_instance_list,
    parameters_dict={'schedule': schedules_list, 'Tfactor': Tfactor_list},
    resource_list=boots_list,
    dneal_results_path=dneal_results_path,
    use_raw_full_dataframe=use_raw_full_dataframe,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_dneal_pickles=use_raw_dneal_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
)
# %%
# Generate plots for TTS of ensemble together with single instance (42)
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_dneal_42,
    x_axis='Tfactor',
    y_axis='tts',
    ax=ax,
    dict_fixed={
        'schedule': 'geometric',
        'instance': 42,
        },
    list_dicts=[{
        # 'instance': 42,
        'boots': j}
                for j in [1,5,10,100,1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    colormap=plt.cm.Dark2,
)
# plot_1d_singleinstance_list(
#     df=df_results_all_stats,
#     x_axis='Tfactor',
#     y_axis='median_tts',
#     ax=ax,
#     dict_fixed={'schedule': 'geometric'},
#     list_dicts=[{'boots': j}
#                 for j in [1,5,10,100,1000]],
#     labels=labels,
#     prefix=prefix,
#     log_x=True,
#     log_y=True,
#     colors=['colormap']
# )
# %%
# Generate plots for performance ratio of ensemble vs Tfactor
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_results_all_stats,
    x_axis='Tfactor',
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
    dict_fixed={'schedule': 'geometric', 'Tfactor': default_Tfactor},
    list_dicts=[{'boots': j}
                for j in all_boots_list[::10]],
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
    dict_fixed={'schedule': 'geometric'},
    list_dicts=[{'Tfactor': j}
                for j in interesting_Tfactors],
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
    y_axis='mean_success_prob',
    ax=ax,
    dict_fixed={'schedule': 'geometric'},
    list_dicts=[{'boots': j}
                for j in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    # ylim=[0.9, 1.005],
    # xlim=[1e2, 5e4],
)

# %%
# Gather all the data for the best tts of the ensemble training set for each instance
best_ensemble_Tfactor = []
df_list = []
stat_measures = ['mean', 'median']
use_raw_dataframes = True
for stat_measure in stat_measures:
    best_Tfactor = df_results_all_stats[df_results_all_stats['boots'] == default_boots].nsmallest(
        1, stat_measure + '_tts')['Tfactor'].values[0]
    best_ensemble_Tfactor.append(best_Tfactor)
for instance in instance_list:
    df_name = "df_results_" + str(instance) + "T.pkl"
    df_path = os.path.join(dneal_results_path, df_name)
    df_results_dneal = pd.read_pickle(df_path)
    df_results_dneal = createDnealResultsDataframes(
        df=df_results_dneal,
        instance_list=[instance],
        parameters_dict={'schedule': schedules_list,
                         'Tfactor': best_ensemble_Tfactor},
        boots_list=all_boots_list,
        dneal_results_path=dneal_results_path,
        dneal_pickle_path=dneal_pickle_path,
        use_raw_dataframes=use_raw_dataframes,
        use_raw_dneal_pickles=use_raw_dneal_pickles,
        overwrite_pickles=overwrite_pickles,
        s=s,
        confidence_level=conf_int,
        bootstrap_iterations=bootstrap_iterations,
        gap=gap,
        fail_value=fail_value,
        save_pickle=True,
    )
    df_list.append(df_results_dneal)

df_results_all = pd.concat(df_list, ignore_index=True)
df_results_all = cleanup_df(df_results_all)
df_name = "df_resultsT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_results_all.to_pickle(df_path)

# %%
# Reload all results with the best tts of the ensemble for each instance
df_results_all = createDnealResultsDataframes(
    df=df_results_all,
    instance_list=instance_list,
    parameters_dict={'schedule': schedules_list,
                     'Tfactor': best_ensemble_Tfactor},
    boots_list=all_boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_dneal_pickles=use_raw_dneal_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
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
df_name = "df_resultsT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_results_all.to_pickle(df_path)

if 'inv_perf_ratio' not in df_dneal_42.columns:
    df_dneal_42['inv_perf_ratio'] = 1 - df_dneal_42['perf_ratio'] + EPSILON
    df_dneal_42['inv_perf_ratio_conf_interval_lower'] = 1 - \
        df_dneal_42['perf_ratio_conf_interval_upper'] + EPSILON
    df_dneal_42['inv_perf_ratio_conf_interval_upper'] = 1 - \
        df_dneal_42['perf_ratio_conf_interval_lower'] + EPSILON
df_dneal_42 = cleanup_df(df_dneal_42)
df_name = "df_results_42T.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_dneal_42.to_pickle(df_path)
# %%
# Obtain the tts for each instance in the median and the mean of the ensemble accross the sweeps
# TODO generalize this code. In general, one parameter (or several) are fixed in certain interesting values and then for all instances with the all other values of remaining parameters we report the metric output, everything at 1000 bootstraps

for metric in ['perf_ratio', 'success_prob', 'tts', 'inv_perf_ratio']:

    df_results_all[metric + '_lower'] = df_results_all[metric] - \
        df_results_all[metric + '_conf_interval_lower']
    df_results_all[metric + '_upper'] = df_results_all[metric] - \
        df_results_all[metric + '_conf_interval_upper']

    # These following lines can be extracted from the loop
    df_default = df_results_all[(df_results_all['boots'] == default_boots) & (
        df_results_all['Tfactor'] == default_Tfactor)].set_index(['instance', 'schedule'])
    df_list = [df_default]
    keys_list = ['default']
    for i, Tfactor in enumerate(best_ensemble_Tfactor):
        df_list.append(df_results_all[(df_results_all['boots'] == default_boots) & (
            df_results_all['Tfactor'] == Tfactor)].set_index(['instance', 'schedule']))
        keys_list.append(stat_measures[i] + '_ensemble')
    # Until here can be done off-loop

    # Metrics that you want to minimize
    ascent = metric in ['tts', 'mean_time', 'inv_perf_ratio']
    df_best = df_results_all[
        (df_results_all['boots'] == default_boots)
    ].sort_values(metric, ascending=ascent).groupby(
        ['instance', 'schedule']).apply(
            pd.DataFrame.head, n=1
    ).droplevel(-1).drop(columns=['instance', 'schedule']
                         )

    df_list.append(df_best)
    keys_list.append('best')

    df_merged = pd.concat(df_list, axis=1, keys=keys_list,
                          names=['stat_metric', 'measure'])

    fig, ax = plt.subplots()
    df_merged.loc[
        (slice(None), slice(None)),  # (slice(None), 'geometric'),
        :
    ].plot.bar(
        y=[(stat_metric, metric) for stat_metric in keys_list],
        yerr=df_merged.loc[
            (slice(None), slice(None)),  # (slice(None), 'geometric'),
            (slice(None), [metric + '_lower', metric + '_upper'])
        ].to_numpy().T,
        ax=ax,
    )
    ax.set(title='Different performance of ' + metric +
           ' in instances ' + prefix + '\n' +
           'evaluated individually and with the ensemble')
    ax.set(ylabel=labels[metric])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if metric in ['tts', 'inv_perf_ratio']:
        ax.set(yscale='log')

df_results_all = cleanup_df(df_results_all)
df_name = "df_resultsT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_results_all.to_pickle(df_path)
# %%
# Plot with performance ratio vs reads for interesting sweeps
for instance in [3, 0, 7, 42]:

    interesting_Tfactors = [
        df_results_all[(df_results_all['boots'] == default_boots) & (df_results_all['instance'] == instance)].nsmallest(1, 'tts')[
            'Tfactor'].values[0],
        0.1,
        10,
        100,
        1000,
        1
    ] + list(set(best_ensemble_Tfactor))
    f, ax = plt.subplots()
    ax = plot_1d_singleinstance_list(
        df=df_results_all,
        x_axis='reads',
        y_axis='perf_ratio',
        dict_fixed={
            'instance': instance,
            'schedule': 'geometric'
        },
        ax=ax,
        list_dicts=[{'Tfactor': i}
                    for i in interesting_Tfactors],
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

    interesting_Tfactors = list(set([
        df_results_all[(df_results_all['boots'] == default_boots) & (df_results_all['instance'] == instance)].nsmallest(1, 'tts')[
            'Tfactor'].values[0],
        0.1,
        10,
        100,
        1000,
        1
    ] + best_ensemble_Tfactor))
    f, ax = plt.subplots()
    ax = plot_1d_singleinstance_list(
        df=df_results_all,
        x_axis='reads',
        y_axis='inv_perf_ratio',
        dict_fixed={
            'instance': instance,
            'schedule': 'geometric'
        },
        ax=ax,
        list_dicts=[{'Tfactor': i}
                    for i in interesting_Tfactors],
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
use_raw_dataframes = False
df_results_all_stats = generateStatsDataframe(
    df_all=df_results_all,
    stat_measures=['mean', 'median'],
    instance_list=training_instance_list,
    parameters_dict={'schedule': schedules_list,
                     'Tfactor': Tfactor_list},
    resource_list=boots_list,
    dneal_results_path=dneal_results_path,
    use_raw_full_dataframe=use_raw_full_dataframe,
    use_raw_dataframes=use_raw_dataframes,
    use_raw_dneal_pickles=use_raw_dneal_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
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
            dict_fixed={'schedule': 'geometric'},
            list_dicts=[{'Tfactor': i, 'instance': instance}
                        for i in [10, 100, default_Tfactor] + list(set(best_ensemble_Tfactor))],
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
        dict_fixed={'schedule': 'geometric'},
        list_dicts=[{'Tfactor': i}
                    for i in [10, 100, default_Tfactor] + list(set(best_ensemble_Tfactor))],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        ylim=[0.975, 1.0025],
        # xlim=[5e2, 5e4],
        xlim=[5e2, 5e4],
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
        dict_fixed={'schedule': 'geometric'},
        list_dicts=[{'Tfactor': i}
                    for i in [10, 100, default_Tfactor] + list(set(best_ensemble_Tfactor))],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        save_fig=False,
        xlim=[5e2, 5e4],
        use_colorbar=False,
        linewidth=2.5,
        colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
    )
# %%
# Create virtual best and virtual worst columns
# TODO This can be generalized as using as groups the parameters that are not dependent of the metric (e.g., schedule) or that signify different solvers
# TODO This needs to be functionalized

df_name = "df_results_virtT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
if use_raw_dataframes or os.path.exists(df_path) is False:
    df_virtual_all = df_results_all.groupby(
        ['schedule', 'reads']
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
        ['reads', 'schedule',
         'virt_best_perf_ratio',
         'virt_best_success_prob']
    ].sort_values('reads'
                  ).groupby(
        'schedule'
    ).expanding(min_periods=1).max().droplevel(-1).reset_index()

    # This is done as the virtual worst counts the worst case, computed as the minimum from last read to first
    df_virtual_worst_max = df_virtual_all[
        ['reads', 'schedule',
         'virt_worst_perf_ratio']
    ].sort_values('reads', ascending=False
                  ).groupby(
        'schedule'
    ).expanding(min_periods=1).min().droplevel(-1).reset_index()

    df_virtual_best_min = df_virtual_all[
        ['reads', 'schedule',
         'virt_best_tts',
         'virt_best_mean_time',
         'virt_best_inv_perf_ratio']
    ].sort_values('reads'
                  ).groupby(
        'schedule'
    ).expanding(min_periods=1).agg({
        'reads': np.max,
        'virt_best_tts': np.min,
        'virt_best_mean_time': np.min,
        'virt_best_inv_perf_ratio': np.min}
    ).droplevel(-1).reset_index()
    # df_virtual_best_min['reads'] = df_virtual_all[
    #     ['reads', 'schedule']
    # ].sort_values(['schedule','reads'])['reads']

    df_virtual_worst_min = df_virtual_all[
        ['reads', 'schedule',
         'virt_worst_inv_perf_ratio']
    ].sort_values('reads', ascending=False
                  ).groupby(
        'schedule'
    ).expanding(min_periods=1).agg({
        'reads': np.min,
        'virt_worst_inv_perf_ratio': np.max}
    ).droplevel(-1).reset_index()
    df_virtual_best_min['reads'] = df_virtual_all[
        ['reads', 'schedule']
    ].sort_values('schedule').sort_values('reads', ascending=False)['reads']

    df_virtual_best = df_virtual_best_max.merge(
        df_virtual_worst_max,
        on=['schedule', 'reads'],
        how='left')
    df_virtual_best = df_virtual_best.merge(
        df_virtual_best_min,
        on=['schedule', 'reads'],
        how='left')
    df_virtual_best = df_virtual_best.merge(
        df_virtual_worst_min,
        on=['schedule', 'reads'],
        how='left')
    df_virtual_best = cleanup_df(df_virtual_best)
    df_virtual_best.to_pickle(df_path)
else:
    df_virtual_best = pd.read_pickle(df_path)
    df_virtual_best = cleanup_df(df_virtual_best)

# %%
# Generate plots for performance ratio of ensemble vs reads with best and worst performance
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='virt_best_perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed={'schedule': 'geometric'},
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
        y_axis='virt_worst_perf_ratio',
        ax=ax,
        label_plot='Virtual worst',
        dict_fixed={'schedule': 'geometric'},
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
        dict_fixed={'schedule': 'geometric'},
        list_dicts=[{'Tfactor': i}
                    for i in [10, 100, 1000, default_Tfactor] + list(set(best_ensemble_Tfactor))],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        ylim=[0.975, 1.0025],
        # xlim=[5e2, 5e4],
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
        y_axis='virt_best_inv_perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed={'schedule': 'geometric'},
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
        dict_fixed={'schedule': 'geometric'},
        labels=labels,
        prefix=prefix,
        save_fig=False,
        log_x=True,
        log_y=False,
        linewidth=2.5,
        marker=None,
        color=['r'],
    )
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_inv_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        list_dicts=[{'Tfactor': i}
                    for i in [10, 100, 1000, default_Tfactor] + list(set(best_ensemble_Tfactor))],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        save_fig=False,
        use_colorbar=False,
        linewidth=1.5,
        markersize=1,
        colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
    )
# %%
# Random search for the ensemble
repetitions = 10  # Times to run the algorithm
rs = [1, 5, 10]  # resources per parameter setting (runs)
frac_r_exploration = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
R_budgets = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
parameters = ['schedule', 'Tfactor']
experiments = rs * repetitions
df_name = "df_progress_totalT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
use_raw_dataframes = True
df_search = df_results_all_stats[
    parameters + ['boots',
                            'median_perf_ratio', 'mean_perf_ratio', 'reads']
].set_index(
    parameters + ['boots']
)
if use_raw_dataframes or os.path.exists(df_path) is False:
    progress_list = []
    for R_budget in R_budgets:
        for frac_expl_total in frac_r_exploration:
            # budget for exploration (runs)
            R_exploration = int(R_budget*frac_expl_total)
            # budget for exploitation (runs)
            R_exploitation = R_budget - R_exploration
            for r in rs:
                for experiment in range(repetitions):
                    random_Tfactor = np.random.choice(
                        Tfactor_list, size=int(R_exploration / (r*default_sweeps)), replace=True)
                    # % Question: Should we replace these samples?
                    if r*default_sweeps > R_exploration:
                        print(
                            "R_exploration must be larger than single exploration step")
                        continue
                    series_list = []
                    total_reads = 0
                    for Tfactor in random_Tfactor:
                        series_list.append(
                            df_search.loc[
                                idx['geometric', Tfactor, r]]
                        )
                        total_reads += r
                        if total_reads > R_exploration:
                            converged = True
                            break
                    exploration_step = pd.concat(series_list, axis=1).T.rename_axis(
                        parameters + ['boots'])
                    exploration_step['median_perf_ratio'] = exploration_step['median_perf_ratio'].expanding(
                        min_periods=1).max()
                    exploration_step['mean_perf_ratio'] = exploration_step['mean_perf_ratio'].expanding(
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
                        parameters).loc[exploration_step.nlargest(1, 'median_perf_ratio').index]
                    exploitation_step['cum_reads'] = exploitation_step['reads'] + \
                        exploration_step['cum_reads'].max()
                    exploitation_step.sort_values(['cum_reads'], inplace=True)
                    exploitation_step = exploitation_step[exploitation_step['cum_reads'] <= R_budget]
                    exploitation_step['median_perf_ratio'].fillna(
                        0, inplace=True)
                    exploitation_step['median_perf_ratio'].clip(
                        lower=exploration_step['median_perf_ratio'].max(), inplace=True)
                    exploitation_step['median_perf_ratio'] = exploitation_step['median_perf_ratio'].expanding(
                        min_periods=1).max()
                    exploitation_step['mean_perf_ratio'].fillna(
                        0, inplace=True)
                    exploitation_step['mean_perf_ratio'].clip(
                        lower=exploration_step['mean_perf_ratio'].max(), inplace=True)
                    exploitation_step['mean_perf_ratio'] = exploitation_step['mean_perf_ratio'].expanding(
                        min_periods=1).max()
                    exploitation_step['experiment'] = experiment
                    exploitation_step['run_per_solve'] = r
                    exploitation_step['R_explor'] = R_exploration
                    exploitation_step['R_exploit'] = R_exploitation
                    progress_list.append(exploitation_step)
    df_progress_total = pd.concat(progress_list, axis=0)
    df_progress_total.reset_index(inplace=True)
    df_progress_total.to_pickle(df_path)
else:
    df_progress_total = pd.read_pickle(df_path)

if 'R_budget' not in df_progress_total.columns:
    df_progress_total['R_budget'] = df_progress_total['R_explor'] + \
        df_progress_total['R_exploit']


for stat_measure in stat_measures:
    if 'best_' + stat_measure + '_perf_ratio' not in df_progress_total.columns:
        df_progress_total[stat_measure + '_inv_perf_ratio'] = 1 - \
            df_progress_total[stat_measure + '_perf_ratio'] + EPSILON
        # df_progress_total['best_' + stat_measure + '_inv_perf_ratio'] = df_progress_total.sort_values(
        # ['cum_reads', 'R_budget']
        # ).groupby(['run_per_solve']
        # ).expanding(min_periods=1).min().droplevel(-1).reset_index()[stat_measure + '_inv_perf_ratio']
        df_progress_total['best_' + stat_measure + '_inv_perf_ratio'] = df_progress_total.sort_values(
            ['cum_reads', 'R_budget']
        ).expanding(min_periods=1).min()[stat_measure + '_inv_perf_ratio']
        df_progress_total['best_' + stat_measure + '_perf_ratio'] = 1 - \
            df_progress_total['best_' + stat_measure +
                              '_inv_perf_ratio'] + EPSILON
df_progress_total = cleanup_df(df_progress_total)
df_progress_total.to_pickle(df_path)

# %%
# Ternary search in the ensemble
# rs = [1, 5, 10]  # resources per parameter setting (runs)
rs = [5, 20, 50, 100]  # resources per parameter setting (runs)
# frac_r_exploration = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
frac_r_exploration = [0.05, 0.1, 0.2, 0.5, 0.75]
# R_budgets = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
R_budgets = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
parameters = ['schedule', 'Tfactor']
experiments = rs * repetitions
df_name = "df_progress_total_ternaryT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
use_raw_dataframes = True
df_search = df_results_all_stats[
    parameters + ['boots',
                            'median_perf_ratio', 'mean_perf_ratio', 'reads']
].set_index(
    parameters + ['boots']
)
if use_raw_dataframes or os.path.exists(df_path) is False:
    progress_list = []
    for R_budget in R_budgets:
        for frac_expl_total in frac_r_exploration:
            # budget for exploration (runs)
            R_exploration = int(R_budget*frac_expl_total)
            # budget for exploitation (runs)
            R_exploitation = R_budget - R_exploration
            for r in rs:
                if r*default_sweeps*4 > R_exploration:
                    print(
                        "R_exploration must be larger than four exploration steps")
                    continue
                
                
                series_list = []
                total_reads = 0
                
                lo = 0
                val_lo = Tfactor_list[lo]
                perf_lo = df_search.loc[
                    idx[default_schedule, val_lo, r]]['median_perf_ratio']
                series_list.append(df_search.loc[
                    idx[default_schedule, val_lo, r]])

                up = len(Tfactor_list) - 1
                val_up = Tfactor_list[up]
                perf_up = df_search.loc[
                    idx[default_schedule, val_up, r]]['median_perf_ratio']
                series_list.append(df_search.loc[
                    idx[default_schedule, val_up, r]])



                while lo <= up:
                    x1 = int(lo + (up - lo) / 3)
                    x2 = int(up - (up - lo) / 3)
                    val_x1 = Tfactor_list[x1]
                    perf_x1 = df_search.loc[
                        idx[default_schedule, val_x1, r]]['median_perf_ratio']
                    series_list.append(df_search.loc[
                        idx[default_schedule, val_x1, r]])
                    total_reads += r
                    if total_reads > R_exploration:
                        break
                    val_x2 = Tfactor_list[x2]
                    perf_x2 = df_search.loc[
                        idx[default_schedule, val_x2, r]]['median_perf_ratio']
                    series_list.append(df_search.loc[
                        idx[default_schedule, val_x2, r]])
                    total_reads += r
                    if total_reads > R_exploration:
                        break
                    if perf_x2 == perf_up:
                        up -= 1
                        val_up = Tfactor_list[up]
                        perf_up = df_search.loc[
                            idx[default_schedule, val_up, r]]['median_perf_ratio']
                        series_list.append(df_search.loc[
                            idx[default_schedule, val_up, r]])
                        total_reads += r
                    elif perf_x1 == perf_lo:
                        lo += 1
                        val_lo = Tfactor_list[lo]
                        perf_lo = df_search.loc[
                            idx[default_schedule, val_lo, r]]['median_perf_ratio']
                        series_list.append(df_search.loc[
                            idx[default_schedule, val_lo, r]])
                        total_reads += r
                    elif perf_x1 > perf_x2:
                        up = x2
                        val_up = Tfactor_list[up]
                        perf_up = df_search.loc[
                            idx[default_schedule, val_up, r]]['median_perf_ratio']
                    else:
                        lo = x1
                        val_lo = Tfactor_list[lo]
                        perf_lo = df_search.loc[
                            idx[default_schedule, val_lo, r]]['median_perf_ratio']

                    if total_reads > R_exploration:
                        break

                exploration_step = pd.concat(series_list, axis=1).T.rename_axis(
                    parameters + ['boots'])
                exploration_step['median_perf_ratio'] = exploration_step['median_perf_ratio'].expanding(
                    min_periods=1).max()
                exploration_step['mean_perf_ratio'] = exploration_step['mean_perf_ratio'].expanding(
                    min_periods=1).max()
                exploration_step.reset_index('boots', inplace=True)
                exploration_step['run_per_solve'] = r
                exploration_step['R_explor'] = R_exploration
                exploration_step['R_exploit'] = R_exploitation
                exploration_step['cum_reads'] = exploration_step.expanding(
                    min_periods=1)['reads'].sum().reset_index(drop=True).values
                progress_list.append(exploration_step)

                exploitation_step = df_search.reset_index().set_index(
                    parameters).loc[exploration_step.nlargest(1, 'median_perf_ratio').index]
                exploitation_step['cum_reads'] = exploitation_step['reads'] + \
                    exploration_step['cum_reads'].max()
                exploitation_step.sort_values(['cum_reads'], inplace=True)
                exploitation_step = exploitation_step[exploitation_step['cum_reads'] <= R_budget]
                exploitation_step['median_perf_ratio'].fillna(
                    0, inplace=True)
                exploitation_step['median_perf_ratio'].clip(
                    lower=exploration_step['median_perf_ratio'].max(), inplace=True)
                exploitation_step['median_perf_ratio'] = exploitation_step['median_perf_ratio'].expanding(
                    min_periods=1).max()
                exploitation_step['mean_perf_ratio'].fillna(
                    0, inplace=True)
                exploitation_step['mean_perf_ratio'].clip(
                    lower=exploration_step['mean_perf_ratio'].max(), inplace=True)
                exploitation_step['mean_perf_ratio'] = exploitation_step['mean_perf_ratio'].expanding(
                    min_periods=1).max()
                exploitation_step['run_per_solve'] = r
                exploitation_step['R_explor'] = R_exploration
                exploitation_step['R_exploit'] = R_exploitation
                progress_list.append(exploitation_step)
    df_progress_total_ternary = pd.concat(progress_list, axis=0)
    df_progress_total_ternary.reset_index(inplace=True)
    df_progress_total_ternary.to_pickle(df_path)
else:
    df_progress_total_ternary = pd.read_pickle(df_path)

if 'R_budget' not in df_progress_total_ternary.columns:
    df_progress_total_ternary['R_budget'] = df_progress_total_ternary['R_explor'] + \
        df_progress_total_ternary['R_exploit']


for stat_measure in stat_measures:
    if 'best_' + stat_measure + '_perf_ratio' not in df_progress_total_ternary.columns:
        df_progress_total_ternary[stat_measure + '_inv_perf_ratio'] = 1 - \
            df_progress_total_ternary[stat_measure + '_perf_ratio'] + EPSILON
        # df_progress_total['best_' + stat_measure + '_inv_perf_ratio'] = df_progress_total.sort_values(
        # ['cum_reads', 'R_budget']
        # ).groupby(['run_per_solve']
        # ).expanding(min_periods=1).min().droplevel(-1).reset_index()[stat_measure + '_inv_perf_ratio']
        df_progress_total_ternary['best_' + stat_measure + '_inv_perf_ratio'] = df_progress_total_ternary.sort_values(
            ['cum_reads', 'R_budget']
        ).expanding(min_periods=1).min()[stat_measure + '_inv_perf_ratio']
        df_progress_total_ternary['best_' + stat_measure + '_perf_ratio'] = 1 - \
            df_progress_total_ternary['best_' + stat_measure +
                              '_inv_perf_ratio'] + EPSILON
df_progress_total_ternary = cleanup_df(df_progress_total_ternary)
# df_progress_total_ternary.to_pickle(df_path)

# %%
# Plot for all the experiments trajectories
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_progress_total,
    x_axis='cum_reads',
    y_axis='best_median_perf_ratio',
    ax=ax,
    dict_fixed={
        'schedule': default_schedule,
        # 'R_budget': R_budgets[0],
        # 'R_explor': R_budgets[0]*frac_r_exploration[-1],
        # 'run_per_solve': rs[0],
    },
    # list_dicts=[{'experiment': i}
    #             for i in range(repetitions)],
    list_dicts=[{'run_per_solve': i}
                for i in rs],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    colors=['colormap'],
    colormap=plt.cm.get_cmap('rainbow'),
    use_colorbar=False,
    use_conf_interval=False,
    save_fig=False,
    # ylim=[0.98, 1.0025],
    # xlim=[rs[0]*default_sweeps, R_budgets[0]],
    linewidth=1.5,
    marker=None,
)
plot_1d_singleinstance_list(
    df=df_progress_total,
    x_axis='cum_reads',
    y_axis='median_perf_ratio',
    ax=ax,
    dict_fixed={
        'schedule': default_schedule,
        'R_budget': R_budgets[-1],
        'R_explor': R_budgets[-1]*frac_r_exploration[0],
        'run_per_solve': rs[-1],
    },
    list_dicts=[{'experiment': i}
                for i in range(repetitions)],
    # list_dicts=[{'run_per_solve': i}
    #             for i in rs],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    colors=['colormap'],
    colormap=plt.cm.get_cmap('rainbow'),
    use_colorbar=False,
    use_conf_interval=False,
    save_fig=False,
    ylim=[0.98, 1.0025],
    # xlim=[rs[0]*default_sweeps, R_budgets[0]],
    linewidth=1.5,
    marker=None,
)
plot_1d_singleinstance_list(
    df=df_progress_total_ternary,
    x_axis='cum_reads',
    y_axis='best_median_perf_ratio',
    ax=ax,
    dict_fixed={
        'schedule': default_schedule,
        'R_budget': R_budgets[-1],
        'R_explor': R_budgets[-1]*frac_r_exploration[0],
        # 'run_per_solve': rs[0],
    },
    # list_dicts=[{'experiment': i}
    #             for i in range(repetitions)],
    list_dicts=[{'run_per_solve': i}
                for i in rs],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    colors=['colormap'],
    colormap=plt.cm.get_cmap('viridis'),
    use_colorbar=False,
    use_conf_interval=False,
    save_fig=False,
    # ylim=[0.98, 1.0025],
    # xlim=[rs[0]*default_sweeps, R_budgets[0]],
    linewidth=1.5,
    marker=None,
)
# %%
# Average across the experiments with the same R_budget, R_explor, and run_per_solve for envelope
df_name = "df_progressT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
use_raw_dataframes = True
if use_raw_dataframes or os.path.exists(df_path) is False:
    df_progress = df_progress_total[
        ['schedule', 'cum_reads', 'R_budget', 'R_explor',
            'run_per_solve', 'experiment', 'median_perf_ratio', 'mean_perf_ratio'
        ]
    ].groupby(
        ['schedule', 'R_budget', 'R_explor', 'run_per_solve', 'cum_reads']
    ).apply(lambda s: pd.Series({
            # 'mean_cum_reads': np.min(s['cum_reads']),
            'median_median_perf_ratio': np.median(s['median_perf_ratio']),
            'mean_median_perf_ratio': np.mean(s['median_perf_ratio']),
            'median_mean_perf_ratio': np.median(s['mean_perf_ratio']),
            'mean_mean_perf_ratio': np.mean(s['mean_perf_ratio'])
            })
            ).reset_index()
    for stat_measure in stat_measures:
        for stat_measure_inv in stat_measures:
            df_progress[stat_measure + '_' + stat_measure_inv + '_inv_perf_ratio'] = 1 - \
                df_progress[stat_measure + '_' +
                            stat_measure_inv + '_perf_ratio'] + EPSILON

            df_progress['best_' + stat_measure + '_' + stat_measure_inv + '_inv_perf_ratio'] = df_progress.sort_values(
                ['cum_reads']
            ).expanding(min_periods=1).min()[stat_measure + '_' + stat_measure_inv + '_inv_perf_ratio']
            df_progress['best_' + stat_measure + '_' + stat_measure_inv + '_perf_ratio'] = 1 - \
                df_progress['best_' + stat_measure + '_' +
                            stat_measure_inv + '_inv_perf_ratio'] + EPSILON

    df_progress['frac_r_exploration'] = df_progress['R_explor'] / \
        df_progress['R_budget']

    df_progress = cleanup_df(df_progress)
    df_progress.to_pickle(df_path)
else:
    df_progress = pd.read_pickle(df_path)
# %%
# Average across the experiments with the same R_budget, R_explor, and run_per_solve for envelope
df_name = "df_progress_ternaryT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
use_raw_dataframes = True
if use_raw_dataframes or os.path.exists(df_path) is False:
    df_progress_ternary = df_progress_total_ternary[
        ['schedule', 'cum_reads', 'R_budget', 'R_explor',
            'run_per_solve', 'median_perf_ratio', 'mean_perf_ratio'
        ]
    ].groupby(
        ['schedule', 'R_budget', 'R_explor', 'run_per_solve', 'cum_reads']
    ).apply(lambda s: pd.Series({
            # 'mean_cum_reads': np.min(s['cum_reads']),
            'median_median_perf_ratio': np.median(s['median_perf_ratio']),
            'mean_median_perf_ratio': np.mean(s['median_perf_ratio']),
            'median_mean_perf_ratio': np.median(s['mean_perf_ratio']),
            'mean_mean_perf_ratio': np.mean(s['mean_perf_ratio'])
            })
            ).reset_index()
    for stat_measure in stat_measures:
        for stat_measure_inv in stat_measures:
            df_progress_ternary[stat_measure + '_' + stat_measure_inv + '_inv_perf_ratio'] = 1 - \
                df_progress_ternary[stat_measure + '_' +
                            stat_measure_inv + '_perf_ratio'] + EPSILON

            df_progress_ternary['best_' + stat_measure + '_' + stat_measure_inv + '_inv_perf_ratio'] = df_progress_ternary.sort_values(
                ['cum_reads']
            ).expanding(min_periods=1).min()[stat_measure + '_' + stat_measure_inv + '_inv_perf_ratio']
            df_progress_ternary['best_' + stat_measure + '_' + stat_measure_inv + '_perf_ratio'] = 1 - \
                df_progress_ternary['best_' + stat_measure + '_' +
                            stat_measure_inv + '_inv_perf_ratio'] + EPSILON

    df_progress_ternary['frac_r_exploration'] = df_progress_ternary['R_explor'] / \
        df_progress_ternary['R_budget']

    df_progress_ternary = cleanup_df(df_progress_ternary)
    df_progress_ternary.to_pickle(df_path)
else:
    df_progress_ternary = pd.read_pickle(df_path)
# %%
# Experimental plot not considering that perf_ratio at the end is optimal
# f, ax = plt.subplots()
# plot_1d_singleinstance_list(
#     df=df_progress,
#     x_axis='R_explor',
#     y_axis='mean_median_inv_perf_ratio',
#     ax=ax,
#     dict_fixed={
#         'R_budget': R_budgets[-1],
#         'cum_reads': R_budgets[-1],
#     },
#     # label_plot='Ordered exploration',
#     list_dicts=[{'run_per_solve': i}
#                 for i in rs],
#     labels=labels,
#     prefix=prefix,
#     log_x=True,
#     log_y=True,
#     colors=['colormap'],
#     colormap=plt.cm.get_cmap('viridis'),
#     use_colorbar=False,
#     use_conf_interval=False,
#     save_fig=False,
#     linewidth=1.5,
# )
# Equivalent seaborn plot
# palette = sns.color_palette("mako_r", 3)
# f,ax = plt.subplots()
# sns.lineplot(data=df_progress, x='R_explor', y='mean_median_inv_perf_ratio', style='R_budget', hue='run_per_solve', estimator=None, ci=None, ax=ax, palette= palette)
# ax.set(yscale='log')
# ax.set(xscale='log')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# TODO Translate all plots to seasborn, take special care to create fork to pass confidence intervals to plot
# Near future v 0.12.0 of seaborn will implement errorbar keyword which would allow us to reload confidence intervals instead of recomputing them every time. Check https://github.com/mwaskom/seaborn/pull/2407 for details
# %%
# Compute best random exploration-exploitation strategy for each R_budget
df_best_random_list = []
for R_budget in R_budgets:
    df_best_random_list.append(df_progress[
        (df_progress['schedule'] == default_schedule) &
        (df_progress['R_budget'] == R_budget)
    ].sort_values(['cum_reads']
                  ).nlargest(1, ['mean_median_perf_ratio', 'median_median_perf_ratio']
                             )[
        ['R_budget', 'frac_r_exploration', 'run_per_solve',
            'median_median_perf_ratio', 'mean_median_perf_ratio']
    ])

best_random_search_idx = pd.concat(df_best_random_list).set_index(
    ['R_budget', 'frac_r_exploration', 'run_per_solve']).index
df_best_random = df_progress.set_index(
    ['R_budget', 'frac_r_exploration', 'run_per_solve']
).loc[best_random_search_idx].reset_index()
df_best_random = cleanup_df(df_best_random)
# %%
# Compute best Ternary exploration-exploitation strategy for each R_budget
df_best_ternary_list = []
for R_budget in R_budgets:
    df_best_ternary_list.append(df_progress_ternary[
        (df_progress_ternary['schedule'] == default_schedule) &
        (df_progress_ternary['R_budget'] == R_budget)
    ].sort_values(['cum_reads']
                  ).nlargest(1, ['mean_median_perf_ratio', 'median_median_perf_ratio']
                             )[
        ['R_budget', 'frac_r_exploration', 'run_per_solve',
            'median_median_perf_ratio', 'mean_median_perf_ratio']
    ])

best_ternary_search_idx = pd.concat(df_best_ternary_list).set_index(
    ['R_budget', 'frac_r_exploration', 'run_per_solve']).index
df_best_ternary = df_progress_ternary.set_index(
    ['R_budget', 'frac_r_exploration', 'run_per_solve']
).loc[best_ternary_search_idx].reset_index()
df_best_ternary = cleanup_df(df_best_ternary)
# %%
# Generate plots for performance ratio of ensemble vs reads with best and worst performance
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='virt_best_perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed={'schedule': 'geometric'},
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
        y_axis='virt_worst_perf_ratio',
        ax=ax,
        label_plot='Virtual worst',
        dict_fixed={'schedule': 'geometric'},
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
        y_axis=stat_measure + '_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        list_dicts=[{'Tfactor': i}
                    for i in list(set(best_ensemble_Tfactor)) + [1000]],
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
    # plot_1d_singleinstance_list(
    #     df=df_progress,
    #     x_axis='cum_reads',
    #     # y_axis='mean_' + stat_measure + '_perf_ratio',
    #     y_axis=stat_measure + '_median_perf_ratio',
    #     ax=ax,
    #     dict_fixed={'schedule': 'geometric',
    #                 'R_budget': 5e4, 'run_per_solve': rs[0]},
    #     # label_plot='Ordered exploration',
    #     list_dicts=[{'frac_r_exploration': i}
    #                 for i in frac_r_exploration],
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
    #     xlim=[1e3, 1e6],
    #     linewidth=1.5,
    #     markersize=1,
    # )
    plot_1d_singleinstance(
        df=df_progress,
        x_axis='cum_reads',
        # y_axis='mean_' + stat_measure + '_perf_ratio',
        # y_axis=stat_measure + '_median_perf_ratio',
        y_axis='best_' + stat_measure + '_median_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': default_schedule},
        label_plot='Best random exploration exploitation',
        # list_dicts=[{'R_budget': i}
        #             for i in R_budgets],
        # list_dicts=[{'run_per_solve': i}
        #             for i in rs],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        use_conf_interval=False,
        save_fig=False,
        ylim=[0.975, 1.0025],
        xlim=[1e3, 1e6],
        linewidth=1.5,
        markersize=1,
        color=['g'],
    )
    plot_1d_singleinstance(
        df=df_progress_ternary,
        x_axis='cum_reads',
        # y_axis='mean_' + stat_measure + '_perf_ratio',
        # y_axis=stat_measure + '_median_perf_ratio',
        y_axis='best_' + stat_measure + '_median_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': default_schedule},
        label_plot='Best ternary exploration exploitation',
        # list_dicts=[{'R_budget': i}
        #             for i in R_budgets],
        # list_dicts=[{'run_per_solve': i}
        #             for i in rs],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        use_conf_interval=False,
        save_fig=False,
        ylim=[0.975, 1.0025],
        xlim=[1e3, 1e6],
        linewidth=1.5,
        markersize=1,
        color=['b'],
    )
    # plot_1d_singleinstance_list(
    #     df=df_best_random,
    #     x_axis='cum_reads',
    #     # y_axis='mean_' + stat_measure + '_perf_ratio',
    #     y_axis=stat_measure + '_median_perf_ratio',
    #     ax=ax,
    #     dict_fixed={'schedule': default_schedule},
    #     list_dicts=[{'R_budget': i}
    #                 for i in R_budgets],
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
    #     xlim=[1e3, 1e6],
    #     linewidth=1.5,
    #     markersize=1,
    # )
    # plot_1d_singleinstance_list(
    #     df=df_best_ternary,
    #     x_axis='cum_reads',
    #     # y_axis='mean_' + stat_measure + '_perf_ratio',
    #     y_axis=stat_measure + '_median_perf_ratio',
    #     ax=ax,
    #     dict_fixed={'schedule': default_schedule},
    #     list_dicts=[{'R_budget': i}
    #                 for i in R_budgets],
    #     labels=labels,
    #     prefix=prefix,
    #     log_x=True,
    #     log_y=False,
    #     colors=['colormap'],
    #     colormap=plt.cm.get_cmap('Dark2'),
    #     use_colorbar=False,
    #     use_conf_interval=False,
    #     save_fig=False,
    #     ylim=[0.975, 1.0025],
    #     xlim=[1e3, 1e6],
    #     linewidth=1.5,
    #     markersize=1,
    # )
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
        dict_fixed={'schedule': 'geometric'},
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
        dict_fixed={'schedule': 'geometric'},
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
    plot_1d_singleinstance(
        df=df_progress,
        x_axis='cum_reads',
        # y_axis='mean_' + stat_measure + '_perf_ratio',
        # y_axis=stat_measure + '_median_perf_ratio',
        y_axis='best_' + stat_measure + '_median_inv_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        label_plot='Best random exploration exploitation',
        # list_dicts=[{'R_budget': i}
        #             for i in R_budgets],
        # list_dicts=[{'run_per_solve': i}
        #             for i in rs],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        use_conf_interval=False,
        save_fig=False,
        ylim=[1e-10, 1e0],
        xlim=[1e3, R_budgets[-1]],
        linewidth=1.5,
        markersize=1,
        color=['g'],
    )
    plot_1d_singleinstance(
        df=df_progress_ternary,
        x_axis='cum_reads',
        # y_axis='mean_' + stat_measure + '_perf_ratio',
        # y_axis=stat_measure + '_median_perf_ratio',
        y_axis='best_' + stat_measure + '_median_inv_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        label_plot='Best ternary exploration exploitation',
        # list_dicts=[{'R_budget': i}
        #             for i in R_budgets],
        # list_dicts=[{'run_per_solve': i}
        #             for i in rs],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        use_conf_interval=False,
        save_fig=False,
        ylim=[1e-10, 1e0],
        xlim=[1e3, R_budgets[-1]],
        linewidth=1.5,
        markersize=1,
        color=['b'],
    )
    plot_1d_singleinstance_list(
        df=df_results_all_stats,
        x_axis='reads',
        y_axis=stat_measure + '_inv_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        list_dicts=[{'Tfactor': i}
                    for i in list(set(best_ensemble_Tfactor)) + [1000]],
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
        ylim=[1e-10, 1e0],
        xlim=[1e3, R_budgets[-1]],
    )
    # plot_1d_singleinstance_list(
    #     df=df_best_random,
    #     x_axis='cum_reads',
    #     # y_axis='mean_' + stat_measure + '_inv_perf_ratio',
    #     y_axis=stat_measure + '_median_inv_perf_ratio',
    #     ax=ax,
    #     dict_fixed={'schedule': 'geometric',
    #                 # 'frac_r_exploration': frac_r_exploration[-1],
    #                 # 'run_per_solve': rs[0]
    #                 },
    #     # label_plot='Ordered exploration',
    #     list_dicts=[{'R_budget': i}
    #                 for i in R_budgets],
    #     labels=labels,
    #     prefix=prefix,
    #     log_x=True,
    #     log_y=True,
    #     colors=['colormap'],
    #     colormap=plt.cm.get_cmap('tab10'),
    #     use_colorbar=False,
    #     use_conf_interval=False,
    #     save_fig=False,
    #     # ylim=[0.975, 1.0025],
    #     ylim=[1e-10, 1e0],
    #     xlim=[1e3, R_budgets[-1]],
    #     linewidth=1.5,
    #     markersize=1,
    # )

# %%
# Computing up ternary search across parameter
# We assume that the performance of the parameter is unimodal (in decreases and the increases)
# r = 1  # resource per parameter setting (runs)
rs = [1, 5, 10]
# R_budget = 550  # budget for exploitation (runs)
df_name = "df_progress_ternaryT.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_search = df_results_all_stats[
    parameters + ['boots',
     'median_perf_ratio', 'mean_perf_ratio', 'reads']
].set_index(
    parameters + ['boots']
)
use_raw_dataframes = True
if use_raw_dataframes or os.path.exists(df_path) is False:
    progress_list = []
    for r in rs:
        series_list = []
        lo = 0
        val_lo = Tfactor_list[lo]
        up = len(Tfactor_list) - 1
        val_up = Tfactor_list[up]
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
            val_x1 = Tfactor_list[x1]
            perf_x1 = df_search.loc[
                idx['geometric', val_x1, r]]['median_perf_ratio']
            series_list.append(df_search.loc[
                idx['geometric', val_x1, r]])
            val_x2 = Tfactor_list[x2]
            perf_x2 = df_search.loc[
                idx['geometric', val_x2, r]]['median_perf_ratio']
            series_list.append(df_search.loc[
                idx['geometric', val_x2, r]])
            if perf_x2 == perf_up:
                up -= 1
                val_up = Tfactor_list[up]
                perf_up = df_search.loc[
                    idx['geometric', val_up, r]]['median_perf_ratio']
                series_list.append(df_search.loc[
                    idx['geometric', val_up, r]])
            elif perf_x1 == perf_lo:
                lo += 1
                val_lo = Tfactor_list[lo]
                perf_lo = df_search.loc[
                    idx['geometric', val_lo, r]]['median_perf_ratio']
                series_list.append(df_search.loc[
                    idx['geometric', val_lo, r]])
            elif perf_x1 > perf_x2:
                up = x2
                val_up = Tfactor_list[up]
                perf_up = df_search.loc[
                    idx['geometric', val_up, r]]['median_perf_ratio']
            else:
                lo = x1
                val_lo = Tfactor_list[lo]
                perf_lo = df_search.loc[
                    idx['geometric', val_lo, r]]['median_perf_ratio']

        exploration_step = pd.concat(series_list, axis=1).T.rename_axis(
            parameters + ['boots'])
        exploration_step['median_perf_ratio'] = exploration_step['median_perf_ratio'].expanding(
            min_periods=1).max()
        exploration_step['mean_perf_ratio'] = exploration_step['mean_perf_ratio'].expanding(
            min_periods=1).max()
        exploration_step.reset_index('boots', inplace=True)
        exploration_step['run_per_solve'] = r
        exploration_step['cum_reads'] = exploration_step.expanding(
            min_periods=1)['reads'].sum().reset_index(drop=True).values
        progress_list.append(exploration_step)

        exploitation_step = df_results_all_stats[
            parameters + ['boots',
                'median_perf_ratio', 'mean_perf_ratio', 'reads']
        ].set_index(
            parameters).loc[exploration_step.nlargest(1, 'median_perf_ratio').index]
        exploitation_step['cum_reads'] = exploitation_step['reads'] + \
            exploration_step['cum_reads'].max()
        exploitation_step = exploitation_step[exploitation_step['cum_reads']
                                              <= default_reads*default_sweeps]
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
# Plots of ternary search together with the best performing schedule
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='virt_best_perf_ratio',
        ax=ax,
        label_plot='Virtual best',
        dict_fixed={'schedule': 'geometric'},
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
        y_axis='virt_worst_perf_ratio',
        ax=ax,
        label_plot='Virtual worst',
        dict_fixed={'schedule': 'geometric'},
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
        y_axis=stat_measure + '_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        list_dicts=[{'Tfactor': i}
                    for i in list(set(best_ensemble_Tfactor)) + [1000]],
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
        style='--',
        colormap=plt.cm.get_cmap('viridis'),
        colors=['colormap'],
    )
    plot_1d_singleinstance_list(
        df=df_best_random,
        x_axis='cum_reads',
        # y_axis='mean_' + stat_measure + '_perf_ratio',
        y_axis=stat_measure + '_median_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        # label_plot='Ordered exploration',
        list_dicts=[{'R_budget': i}
                    for i in R_budgets],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
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
        y_axis=stat_measure + '_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        # label_plot='Ordered exploration',
        list_dicts=[{'run_per_solve': i}
                    for i in rs],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        colors=['colormap'],
        colormap=plt.cm.get_cmap('tab10'),
        use_colorbar=False,
        use_conf_interval=False,
        save_fig=False,
        ylim=[0.975, 1.0025],
        xlim=[1e3, 1e6],
        linewidth=1.5,
        markersize=10,
        style='.-',
    )
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
        dict_fixed={'schedule': 'geometric'},
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
        dict_fixed={'schedule': 'geometric'},
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
        dict_fixed={'schedule': 'geometric'},
        list_dicts=[{'Tfactor': i}
                    for i in list(set(best_ensemble_Tfactor)) + [1000]],
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
    # plot_1d_singleinstance_list(
    #     df=df_best_random,
    #     x_axis='cum_reads',
    #     # y_axis='mean_' + stat_measure + '_inv_perf_ratio',
    #     y_axis=stat_measure + '_median_inv_perf_ratio',
    #     ax=ax,
    #     dict_fixed={'schedule': 'geometric'},
    #     # label_plot='Ordered exploration',
    #     list_dicts=[{'R_budget': i}
    #                 for i in R_budgets],
    #     labels=labels,
    #     prefix=prefix,
    #     log_x=True,
    #     log_y=True,
    #     colors=['colormap'],
    #     colormap=plt.cm.get_cmap('tab10'),
    #     use_colorbar=False,
    #     use_conf_interval=False,
    #     save_fig=False,
    #     linewidth=1.5,
    #     markersize=1,
    # )
    plot_1d_singleinstance(
        df=df_progress,
        x_axis='cum_reads',
        # y_axis='mean_' + stat_measure + '_perf_ratio',
        # y_axis=stat_measure + '_median_perf_ratio',
        y_axis='best_' + stat_measure + '_median_inv_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        label_plot='Best random exploration exploitation',
        # list_dicts=[{'R_budget': i}
        #             for i in R_budgets],
        # list_dicts=[{'run_per_solve': i}
        #             for i in rs],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=True,
        use_conf_interval=False,
        save_fig=False,
        ylim=[1e-10, 1e0],
        xlim=[1e3, R_budgets[-1]],
        linewidth=1.5,
        markersize=1,
    )
    plot_1d_singleinstance_list(
        df=df_progress_ternary,
        x_axis='cum_reads',
        y_axis=stat_measure + '_inv_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
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
        xlim=[1e3, 1e6],
    )
# %%
# Computing up ternary search across parameter for instance 42
# We assume that the performance of the parameter is unimodal (in decreases and the increases)
rs = [1, 5, 10]
df_name = "df_progress_ternary_42T.pkl"
df_path = os.path.join(dneal_results_path, df_name)
search_metric = 'perf_ratio'
compute_metric = 'perf_ratio'
if search_metric == 'tts':
    search_direction = -1  # -1 for decreasing, 1 for increasing
else:
    search_direction = 1
df_search = df_dneal_42[
    parameters + ['boots', 'reads'] +
    list(set([compute_metric, search_metric]))
].set_index(
    parameters + ['boots']
)
use_raw_dataframes = True
if use_raw_dataframes or os.path.exists(df_path) is False:
    progress_list = []
    for r in rs:
        series_list = []
        lo = 0
        val_lo = Tfactor_list[lo]
        up = len(Tfactor_list) - 1
        val_up = Tfactor_list[up]
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
            val_x1 = Tfactor_list[x1]
            perf_x1 = df_search.loc[
                idx[default_schedule, val_x1, r]][search_metric]
            series_list.append(df_search.loc[
                idx[default_schedule, val_x1, r]])
            val_x2 = Tfactor_list[x2]
            perf_x2 = df_search.loc[
                idx[default_schedule, val_x2, r]][search_metric]
            series_list.append(df_search.loc[
                idx[default_schedule, val_x2, r]])
            if perf_x2 == perf_up:
                up -= 1
                val_up = Tfactor_list[up]
                perf_up = df_search.loc[
                    idx[default_schedule, val_up, r]][search_metric]
                series_list.append(df_search.loc[
                    idx[default_schedule, val_up, r]])
            elif perf_x1 == perf_lo:
                lo += 1
                val_lo = Tfactor_list[lo]
                perf_lo = df_search.loc[
                    idx[default_schedule, val_lo, r]][search_metric]
                series_list.append(df_search.loc[
                    idx[default_schedule, val_lo, r]])
            elif search_direction*perf_x1 > search_direction*perf_x2:
                up = x2
                val_up = Tfactor_list[up]
                perf_up = df_search.loc[
                    idx[default_schedule, val_up, r]][search_metric]
            else:
                lo = x1
                val_lo = Tfactor_list[lo]
                perf_lo = df_search.loc[
                    idx[default_schedule, val_lo, r]][search_metric]

        exploration_step = pd.concat(series_list, axis=1).T.rename_axis(
            parameters + ['boots'])
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
                parameters).loc[exploration_step.nlargest(1, search_metric).index]
        else:
            exploitation_step = df_search.reset_index().set_index(
                parameters).loc[exploration_step.nsmallest(1, search_metric).index]
        exploitation_step['cum_reads'] = exploitation_step['reads'] + \
            exploration_step['cum_reads'].max()
        exploitation_step = exploitation_step[exploitation_step['cum_reads']
                                              <= default_reads*default_sweeps]
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
df_name = "df_progress_42T.pkl"
df_path = os.path.join(dneal_results_path, df_name)
compute_metric = 'perf_ratio'
df_search = df_dneal_42[
    parameters + ['boots', 'reads'] + [compute_metric]
].set_index(
    parameters + ['boots']
)
use_raw_dataframes = True
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
            random_Tfactor = np.random.choice(
                Tfactor_list, size=int(R_exploration / (r*default_sweeps)), replace=True)
            # % Question: Should we replace these samples?
            if r*default_sweeps > R_exploration:
                print(
                    "R_exploration must be larger than single exploration step")
                continue
            series_list = []
            total_reads = 0
            for Tfactor in random_Tfactor:
                series_list.append(df_search.loc[
                    idx[default_schedule, Tfactor, r]]
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
interesting_Tfactors = list(set([
    df_results_all[(df_results_all['boots'] == default_boots) & (df_results_all['instance'] == instance)].nsmallest(1, 'tts')[
        'Tfactor'].values[0],
    default_Tfactor,
    100,
    1000,

] + best_ensemble_Tfactor))
f, ax = plt.subplots()
# random_plot = sns.lineplot(
#     data=df_progress_total_42,
#     x='cum_reads',
#     y='perf_ratio',
#     hue='R_budget',
#     estimator='median',
#     ci=None,
#     ax=ax,
#     palette=sns.color_palette('rainbow', len(R_budgets)),
#     # legend=[str(i) for i in R_budgets],
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
    list_dicts=[{'Tfactor': i}
                for i in interesting_Tfactors],
    labels=labels,
    prefix=prefix,
    save_fig=False,
    log_x=True,
    log_y=False,
    use_conf_interval=False,
    default_dict=default_dict.update({'instance': instance}),
    use_colorbar=False,
    ylim=[0.975, 1.0025],
    xlim=[1e3, 1e6*1.1],
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
random_plot = sns.lineplot(
    data=df_progress_total_42,
    x='cum_reads',
    y='inv_perf_ratio',
    hue='R_budget',
    estimator='median',
    ci='sd',
    ax=ax,
    palette=sns.color_palette('rainbow', len(R_budgets)),
    legend=None,
    linewidth=2,
)
random_plot.legend(labels=['R_bu'+str(i) for i in R_budgets])
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
    xlim=[1e3, 1e6*1.1],
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
    list_dicts=[{'Tfactor': i}
                for i in [1, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict.update({'instance': instance}),
    use_colorbar=False,
    # ylim=[0.975, 1.0025],
    xlim=[1e3, 1e6*1.1],
)
# plot_1d_singleinstance_list(
#     df=df_progress_ternary_42,
#     x_axis='cum_reads',
#     y_axis='inv_perf_ratio',
#     ax=ax,
#     dict_fixed={'schedule': default_schedule},
#     # label_plot='Ordered exploration',
#     list_dicts=[{'run_per_solve': i}
#                 for i in rs],
#     labels=labels,
#     prefix=prefix,
#     log_x=True,
#     log_y=True,
#     colors=['colormap'],
#     colormap=plt.cm.get_cmap('tab10'),
#     use_colorbar=False,
#     use_conf_interval=False,
#     save_fig=False,
#     linewidth=1.5,
#     markersize=10,
#     style='.-',
#     # ylim=[0.975, 1.0025],
#     xlim=[1e3, 1e6*1.1],
# )
# %%
