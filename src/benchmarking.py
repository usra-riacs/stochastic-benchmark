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

from plotting import *
from retrieve_data import *

idx = pd.IndexSlice


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
# Create directories for results
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
# Create instance 42 and save it into disk
model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

prefix = "random_n_" + str(N) + "_inst_"
instance_name = prefix + str(instance)
instance_file_name = instance_name + ".txt"
instance_file = os.path.join(instance_path, instance_file_name)
if not os.path.exists(instance_file):
    text_file = open(instance_file, "w")
    text_file.write(model_random.to_coo())
    text_file.close()

# %%
# Plot Q graph
nx_graph = model_random.to_networkx_graph()
edges, bias = zip(*nx.get_edge_attributes(nx_graph, 'bias').items())
bias = np.array(bias)
nx.draw(nx_graph, node_size=15, pos=nx.spring_layout(nx_graph),
        alpha=0.25, edgelist=edges, edge_color=bias, edge_cmap=plt.cm.Blues)

# %%
# Define function to compute random sampled energy


def randomEnergySampler(
    model: dimod.BinaryQuadraticModel,
    num_reads: int = 1000,
    dwave_sampler: bool = False,
) -> float:
    '''
    Computes the energy of a random sampling.

    Args:
        num_reads: The number of samples to use.
        dwave_sampler: A boolean to use the D-Wave sampler or not.

    Returns:
        The energy of the random sampling.
    '''
    if dwave_sampler:
        randomSampler = dimod.RandomSampler()
        randomSample = randomSampler.sample(model, num_reads=num_reads)
        energies = [datum.energy for datum in randomSample.data(
            ['energy'], sorted_by='energy')]
    else:
        if model.vartype == Vartype.BINARY:
            state = np.random.randint(2, size=(model.num_variables, num_reads))
        else:
            randomSample = np.random.randint(
                2, size=(model.num_variables, num_reads)) * 2 - 1
            energies = [model.energy(randomSample[:, i])
                        for i in range(num_reads)]
    return np.mean(energies), randomSample


# %%
# Compute random sample on problem and print average energy
random_energy, random_sample = randomEnergySampler(
    model_random, num_reads=1000, dwave_sampler=True)
df_random_sample = random_sample.to_pandas_dataframe(sample_column=True)
print('Average random energy = ' + str(random_energy))


# %%
# Plot of obtained energies
# plotEnergyValuesDwaveSampleSet(random_sample,
#    title='Random sampling')
plotBarValues(df=df_random_sample, column_name='energy', sorted=True, skip=200,
              xlabel='Solution', ylabel='Energy', title='Random Sampling', save_fig=False, rot=0)


# %%
# Run default Dwave-neal simulated annealing implementation
sim_ann_sampler = dimod.SimulatedAnnealingSampler()
default_sweeps = 1000
total_reads = 1000
default_boots = total_reads
default_name = prefix + str(instance) + '_geometric_' + \
    str(default_sweeps) + '.p'
df_default_name = 'df_' + default_name + 'kl'
rerun_default = False
if not os.path.exists(os.path.join(dneal_pickle_path, default_name)) or rerun_default:
    print('Running default D-Wave-neal simulated annealing implementation')
    start = time.time()
    default_samples = sim_ann_sampler.sample(
        model_random,
        num_reads=1000,
        num_sweeps=1000,)
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
# Generate plots from the default simulated annealing run
# ax_enum = plotEnergyValuesDwaveSampleSet(sim_ann_sample_default,
#                              title='Simulated annealing with default parameters')
# ax_enum.set(ylim=[min_energy*(0.99)**np.sign(min_energy),
#             min_energy*(1.1)**np.sign(min_energy)])
plotBarValues(
    df=df_default_samples,
    column_name='energy',
    sorted=True,
    skip=200,
    xlabel='Solution',
    ylabel='Energy',
    title='Simulated Annealing with default parameters',
    save_fig=False,
    rot=0,
    ylim=[min_energy*(0.99)**np.sign(min_energy),
          min_energy*(1.1)**np.sign(min_energy)],
    legend=[],
)
# plot_energy_cfd(sim_ann_sample_default,
#                 title='Simulated annealing with default parameters', skip=10)
plotBarCounts(
    df=df_default_samples,
    column_name='energy',
    sorted=True,
    normalized=True,
    skip=10,
    xlabel='Energy',
    title='Simulated Annealing with default parameters',
    save_fig=False,
)

# %%
# Default Dwave-neal schedule plot
print(default_samples.info)
beta_schedule = np.geomspace(*default_samples.info['beta_range'], num=1000)
fig, ax = plt.subplots()
ax.plot(beta_schedule, '.')
ax.set_xlabel('Sweeps')
ax.set_ylabel('beta=Inverse temperature')
ax.set_title('Default Geometric temperature schedule')
# %%
# (Vectorized) Function to compute Resource to Target given a success_probability (array) float


def computeRTT(
    success_probability: float,
    s: float = 0.99,
    scale: float = 1.0,
    fail_value: float = None,
    size: int = 1000,
):
    '''
    Computes the resource to target metric given some success probabilty of getting to that target and a scale factor.

    Args:
        success_probability: The success probability of getting to the target.
        s: The success factor (usually said as RTT within s% probability).
        scale: The scale factor.
        fail_value: The value to return if the success probability is 0.

    Returns:
        The resource to target metric.
    '''
    if fail_value is None:
        fail_value = np.nan
    if success_probability == 0:
        return fail_value
    elif success_probability == 1:
        # Consider continuous TTS and TTS scaled by assuming success_probability=1 as success_probability=1/size*(1-1/10)
        return scale*np.log(1.0 - s) / np.log(1 - (1 - 1/10)/size)
    else:
        return scale*np.log(1.0 - s) / np.log(1 - success_probability)


computeRRT_vectorized = np.vectorize(computeRTT, excluded=(1, 2, 3, 4))
# %%
# Load zipped results if using raw data
overwrite_pickles = False
use_raw_data = False
# zip_name = os.path.join(dneal_results_path, 'results.zip')
# if os.path.exists(zip_name) and use_raw_data:
#     import zipfile
#     with zipfile.ZipFile(zip_name, 'r') as zip_ref:
#         zip_ref.extractall(dneal_pickle_path)
#     print('Results zip file has been extrated to ' + dneal_pickle_path)

# %%
# Function to load ground state solutions from solution file gs_energies.txt


def loadEnergyFromFile(data_file, instance_name):
    '''
    Loads the minimum energy of a given instance from file gs_energies.txt

    Args:
        data_file: The file to load the energies from.
        instance_name: The name of the instance to load the energy for.

    Returns:
        The minimum energy of the instance.

    '''
    energies = []
    with open(data_file, "r") as fin:
        for line in fin:
            if(line.split()[0] == instance_name):
                energies.append(float(line.split()[1]))

    return min(energies)

# %%
# Function to generate samples dataframes or load them otherwise


def createDnealSamplesDataframe(
    instance: int = 42,
    parameters: dict = None,
    total_reads: int = 1000,
    sim_ann_sampler=None,
    dneal_pickle_path: str = None,
    use_raw_pickles: bool = False,
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
        use_raw_pickles: Whether to use the raw pickles or not.
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
    dict_pickle_name = prefix + str(instance) + "_" + \
        '_'.join(str(vals) for vals in parameters.values()) + ".p"
    df_samples_name = 'df_' + dict_pickle_name + 'kl'
    df_path = os.path.join(dneal_pickle_path, df_samples_name)
    if os.path.exists(df_path):
        try:
            df_samples = pd.read_pickle(df_path)
            return df_samples
        except (pkl.UnpicklingError, EOFError):
            os.replace(df_path, df_path + '.bak')
    if use_raw_pickles or not os.path.exists(df_path):
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
# Function to retrieve results from Dataframes
def computeResultsList(
    df: pd.DataFrame,
    random_energy: float = 0.0,
    min_energy: float = None,
    downsample: int = 10,
    bootstrap_iterations: int = 1000,
    confidence_level: float = 68,
    gap: float = 1.0,
    s: float = 0.99,
    fail_value: float = np.inf,
    overwrite_pickles: bool = False,
) -> list:
    '''
    Compute a list of the results computed for analysis given a dataframe from a solver.

    Args:
        df: The dataframe from the solver.
        random_energy: The mean energy of the random sample.
        min_energy: The minimum energy of the samples.
        downsample: The downsampling sample for bootstrapping.
        bootstrap_iterations: The number of bootstrap samples.
        confidence_level: The confidence level for the bootstrap.
        gap: The threshold for the considering a read successful w.r.t the performance ratio [%].
        s: The success factor (usually said as RTT within s% probability).
        overwrite_pickles: If True, the pickles will be overwritten.

    Returns:
        A list of the results computed for analysis. Organized as follows
        [
            number of downsamples,
            bootstrapped mean minimum energy,
            boostrapped mean minimum energy confidence interval lower bound,
            boostrapped mean minimum energy confidence interval upper bound,
            bootstrapped performance ratio,
            bootstrapped performance ratio confidence interval lower bound,
            bootstrapped performance ratio confidence interval upper bound,
            bootstrapped success probability,
            boostrapped success probability confidence interval lower bound,
            boostrapped success probability confidence interval upper bound,
            boostrapped resource to target,
            boostrapped resource to target confidence interval lower bound,
            boostrapped resource to target confidence interval upper bound,
            boostrapped mean runtime,
            boostrapped mean runtime confidence interval lower bound,
            boostrapped mean runtime confidence interval upper bound,
        ]

    TODO: Here we assume the succes metric is the performance ratio, we can generalize that as any function of the parameters (use external function)
    TODO: Here we assume the energy is the response of the solver, we can generalize that as any column in the dataframe
    TODO: Here we only return a few parameters with confidence intervals w.r.t. the bootstrapping. We can generalize that as any possible outcome (use function)
    TODO: Since we are minimizing, computing the performance ratio gets the order of the minimum energy confidence interval inverted. Add parameter for maximization. Need to think what else should we change.
    '''

    aggregated_df_flag = False
    if min_energy is None:
        min_energy = df['energy'].min()

    success_val = random_energy - \
        (1.0 - gap/100.0)*(random_energy - min_energy)

    resamples = np.random.randint(0, len(df), size=(
        downsample, bootstrap_iterations)).astype(int)

    energies = df['energy'].values
    times = df['runtime (us)'].values
    # TODO Change this to be general for PySA dataframes
    if 'num_occurrences' in df.columns and not np.all(df['num_occurrences'].values == 1):
        print('The input dataframe is aggregated')
        occurrences = df['num_occurrences'].values
        aggregated_df_flag = True

    # Compute the minimum value of each bootstrap samples and its corresponding confidence interval based on the resamples
    min_boot_dist = np.apply_along_axis(
        func1d=np.min, axis=0, arr=energies[resamples])
    min_boot = np.mean(min_boot_dist)
    min_boot_conf_interval_lower = stats.scoreatpercentile(
        min_boot_dist, 50-confidence_level/2)
    min_boot_conf_interval_upper = stats.scoreatpercentile(
        min_boot_dist, 50+confidence_level/2)

    # Compute the mean time of each bootstrap samples and its corresponding confidence interval based on the resamples
    times_dist = np.apply_along_axis(
        func1d=np.mean, axis=0, arr=times[resamples])
    mean_time = np.mean(times_dist)
    mean_time_conf_interval_lower = stats.scoreatpercentile(
        times_dist, 50-confidence_level/2)
    mean_time_conf_interval_upper = stats.scoreatpercentile(
        times_dist, 50+confidence_level/2)

    # Compute the performance ratio of each bootstrap samples and its corresponding confidence interval based on the resamples
    perf_ratio = (random_energy - min_boot) / (random_energy - min_energy)
    perf_ratio_conf_interval_lower = (random_energy - min_boot_conf_interval_upper) / (
        random_energy - min_energy)
    perf_ratio_conf_interval_upper = (
        random_energy - min_boot_conf_interval_lower) / (random_energy - min_energy)

    # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
    if aggregated_df_flag:
        return []
        # TODO: One can think about deaggregating the dataframe here. Maybe check Dwave's code for this.
    else:
        success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
            x < success_val)/downsample, axis=0, arr=energies[resamples])
    success_prob = np.mean(success_prob_dist)
    success_prob_conf_interval_lower = stats.scoreatpercentile(
        success_prob_dist, 50-confidence_level/2)
    success_prob_conf_interval_upper = stats.scoreatpercentile(
        success_prob_dist, 50+confidence_level/2)

    # Compute the TTT within certain threshold of each bootstrap samples and its corresponding confidence interval based on the resamples
    tts_dist = computeRRT_vectorized(
        success_prob_dist, s=s, scale=1e-6*df['runtime (us)'].sum(), fail_value=fail_value)
    # Question: should we scale the TTS with the number of bootstrapping we do, intuition says we don't need to
    tts = np.mean(tts_dist)
    if np.isinf(tts) or np.isnan(tts) or tts == fail_value:
        tts_conf_interval_lower = fail_value
        tts_conf_interval_upper = fail_value
    else:
        # tts_conf_interval = computeRRT_vectorized(
        #     success_prob_conf_interval, s=0.99, scale=1e-6*df_default_samples['runtime (us)'].sum())
        tts_conf_interval_lower = stats.scoreatpercentile(
            tts_dist, 50-confidence_level/2)
        tts_conf_interval_upper = stats.scoreatpercentile(
            tts_dist, 50+confidence_level/2)
    # Question: How should we compute the confidence interval of the TTS? SHould we compute the function on the confidence interval of the probability or compute the confidence interval over the tts distribution?

    return [downsample, min_boot, min_boot_conf_interval_lower, min_boot_conf_interval_upper, perf_ratio, perf_ratio_conf_interval_lower, perf_ratio_conf_interval_upper, success_prob, success_prob_conf_interval_lower, success_prob_conf_interval_upper, tts, tts_conf_interval_lower, tts_conf_interval_upper, mean_time, mean_time_conf_interval_lower, mean_time_conf_interval_upper]


# %%
# Define clean up function


def cleanup_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Function to cleanup dataframes by:
    - From tuple-like confidence intervals by separating it into two columns.
    - Recomputing the reads column.
    - Defining the schedules columns as categoric.

    Args:
        df (pandas.DataFrame): Dataframe to be cleaned.

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    '''
    df_new = df.copy()
    for column in df_new.columns:
        if column.endswith('conf_interval'):
            df_new[column + '_lower'] = df_new[column].apply(lambda x: x[0])
            df_new[column + '_upper'] = df_new[column].apply(lambda x: x[1])
            df_new.drop(column, axis=1, inplace=True)
        elif column == 'schedule':
            df_new[column] = df_new[column].astype('category')
    if 'sweeps' in df_new.columns:
        df_new['reads'] = df_new['sweeps'] * df_new['boots']
        df_new['sweeps'] = df_new['sweeps'].astype('int', errors='ignore')
    else:
        df_new['reads'] = df_new['boots']
    df_new['reads'] = df_new['reads'].astype('int', errors='ignore')
    df_new['boots'] = df_new['boots'].astype('int', errors='ignore')
    return df_new


# %%
# TODO Remove all the list_* variables and name them as plurals instead
# Function to update the dataframes
def createDnealResultsDataframes(
    df: pd.DataFrame = None,
    instance_list: List[int] = [0],
    parameters_dict: dict = None,
    boots_list: List[int] = [1000],
    dneal_results_path: str = None,
    dneal_pickle_path: str = None,
    use_raw_data: bool = False,
    use_raw_pickles: bool = False,
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
        use_raw_data: If we want to use the raw data
        overwrite_pickles: If we want to overwrite the pickle files

    '''
    if parameters_dict is None:
        for i, j in parameters_dict.items():
            parameters_dict[i] = set(j)

    # Check that the parameters are columns in the dataframe
    if df is not None:
        assert all([i in df.columns for i in parameters_dict.keys()])
        # In case that the data is already in the dataframe, return it
        if all([k in df[i].values for (i, j) in parameters_dict.items() for k in j]):
            print('The dataframe has some data for the parameters')
            # The parameters dictionary has lists as values as the loop below makes the concatenation faster than running the loop for each parameter
            cond = [df[k].apply(lambda k: k == i).astype(bool)
                    for k, v in parameters_dict.items() for i in v]
            cond_total = functools.reduce(lambda x, y: x & y, cond)
            if all(boots in df[cond_total]['boots'].values for boots in boots_list):
                print('The dataframe already has all the data')
                return df

    # Create filename
    # TODO modify filenmaes inteligently to make it easier to work with
    if len(instance_list) > 1:
        df_name = "df_results.pkl"
    else:
        df_name = "df_results_" + str(instance_list[0]) + ".pkl"
    df_path = os.path.join(dneal_results_path, df_name)

    # If use_raw_data compute the row
    if use_raw_data or not os.path.exists(df_path):
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
                sweep = combination[1]

                parameters = {'schedule': schedule,
                              'sweep': sweep,
                              # 'Tfactor': 1.0,
                              }
                df_samples = createDnealSamplesDataframe(
                    instance=instance,
                    parameters=parameters,
                    total_reads=total_reads,
                    sim_ann_sampler=sim_ann_sampler,
                    dneal_pickle_path=dneal_pickle_path,
                    use_raw_pickles=use_raw_pickles,
                    overwrite_pickles=overwrite_pickles,
                )

                for boots in boots_list:

                    # TODO Good place to replace with mask and isin1d()
                    if (df is not None) and (boots in df[(df['schedule'] == schedule) & (df['sweeps'] == sweep)]['boots'].values):
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
            'instance', 'schedule', 'sweeps', 'boots',
            'min_energy', 'min_energy_conf_interval_lower', 'min_energy_conf_interval_upper',
            'perf_ratio', 'perf_ratio_conf_interval_lower', 'perf_ratio_conf_interval_upper',
            'success_prob', 'success_prob_conf_interval_lower', 'success_prob_conf_interval_upper',
            'tts', 'tts_conf_interval_lower', 'tts_conf_interval_upper',
            'mean_time', 'mean_time_conf_interval_lower', 'mean_time_conf_interval_upper'])
        if df is not None:
            df_new = pd.concat(
                [df, df_results_dneal], axis=0, ignore_index=True)
        else:
            df_new = df_results_dneal.copy()

        if save_pickle:
            df_new = cleanup_df(df_new)
            df_new.to_pickle(df_path)
    else:
        print("Loading the dataframe")
        df_new = pd.read_pickle(df_path)
    return df_new


# %%
# Create dictionaries for upper and lower bounds of confidence intervals
metrics_list = ['min_energy', 'tts',
                'perf_ratio', 'success_prob', 'mean_time']
lower_bounds = {key: None for key in metrics_list}
lower_bounds['success_prob'] = 0.0
lower_bounds['mean_time'] = 0.0
upper_bounds = {key: None for key in metrics_list}
upper_bounds['success_prob'] = 1.0
upper_bounds['perf_ratio'] = 1.0

# Define default behavior for the solver
total_reads = 1000
default_sweeps = 1000
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
    use_raw_data: bool = False,
    use_raw_pickles: bool = False,
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
        use_raw_data: Boolean indicating whether to use the raw data for generating the aggregated dataframe
        use_raw_pickles: Boolean indicating whether to use the raw pickles for generating the aggregated pickles
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
        use_raw_data=use_raw_data,
        use_raw_pickles=use_raw_pickles,
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
    df_name = 'df_results_stats'
    df_path = os.path.join(dneal_results_path, df_name + '.pkl')
    df_all_stats = pd.read_pickle(df_path)
    if all([stat_measure + '_' + metric + '_conf_interval_' + limit in df_all_stats.columns for stat_measure in stat_measures for metric in metrics_list for limit in ['lower', 'upper']]) and not use_raw_data:
        pass
    else:
        df_all_groups = df_all.set_index(
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
use_raw_data = False
use_raw_pickles = False
overwrite_pickles = False
instance = 42
metrics_list = ['min_energy', 'tts',
                'perf_ratio', 'success_prob', 'mean_time']
sweeps_list = [i for i in range(1, 21, 1)] + [
    i for i in range(20, 201, 10)] + [
    i for i in range(200, 501, 20)] + [
    i for i in range(500, 1001, 100)]
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

df_name = "df_results_" + str(instance) + ".pkl"
df_path = os.path.join(dneal_results_path, df_name)
if os.path.exists(df_path):
    df_results_dneal = pd.read_pickle(df_path)
else:
    df_results_dneal = None

df_results_dneal = createDnealResultsDataframes(
    df=df_results_dneal,
    instance_list=[instance],
    parameters_dict={'schedule': schedules_list, 'sweeps': sweeps_list},
    boots_list=boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_data=use_raw_data,
    use_raw_pickles=use_raw_pickles,
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
    'median_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'mean_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'tts': 'TTS ' + str(100*s) + '% confidence  \n (within ' + str(gap) + '% of best found) [s]',
    'median_tts': 'TTS ' + str(100*s) + '% confidence  \n (within ' + str(gap) + '% of best found) [s]',
    'mean_tts': 'TTS ' + str(100*s) + '% confidence  \n (within ' + str(gap) + '% of best found) [s]',
    'boots': 'Number of downsamples during bootrapping',
    'reads': 'Total number of reads (proportional to time)',
    'cum_reads': 'Total number of reads (proportional to time)',
    'min_energy': 'Minimum energy found',
    'mean_time': 'Mean time [us]',
    'Tfactor': 'Factor to multiply lower temperature by',
    'experiment': 'Experiment',
    # 'tts': 'TTS to GS with 99% confidence \n [s * replica] ~ [MVM]',
}

# %%
# Performance ratio vs sweeps for different bootstrap downsamples
default_dict = {'instance': 42, 'schedule': 'geometric',
                'sweeps': 1000, 'boots': 1000}
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_results_dneal,
    x_axis='sweeps',
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
    ylim=[0.95, 1.005]
)
# %%
# Performance ratio vs runs for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_results_dneal,
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
    ylim=[0.95, 1.005]
)
# %%
# Mean time plot of some fixed parameter setting
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_results_dneal,
    x_axis='sweeps',
    y_axis='mean_time',
    dict_fixed={'instance': 42, 'boots': 1000},
    ax=ax,
    list_dicts=[{'schedule': i} for i in schedules_list],
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
    df=df_results_dneal,
    x_axis='sweeps',
    y_axis='success_prob',
    dict_fixed={'instance': 42, 'boots': 1000},
    ax=ax,
    list_dicts=[{'schedule': i} for i in schedules_list],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
)
# %%
# TTS Plot for all bootstrapping downsamples in both schedules
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_results_dneal,
    x_axis='sweeps',
    y_axis='tts',
    ax=ax,
    dict_fixed={'instance': 42},
    list_dicts=[{'schedule': i, 'boots': j}
                for i in schedules_list for j in [10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
)
# %%
# Loop over the dataframes with 4 values of sweeps and a sweep in boots then compute the results, and complete by creating main Dataframe
interesting_sweeps = [
    df_results_dneal[df_results_dneal['boots'] == default_boots].nsmallest(1, 'tts')[
        'sweeps'].values[0],
    1,
    10,
    100,
    default_sweeps // 2,
    default_sweeps,
]

# Iterating for all values of bootstrapping downsampling proves to be very expensive, rather use steps of 10
all_boots_list = list(range(1, 1001, 1))

df_results_dneal = createDnealResultsDataframes(
    df=df_results_dneal,
    instance_list=[instance],
    parameters_dict={'schedule': schedules_list, 'sweeps': interesting_sweeps},
    boots_list=all_boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_data=use_raw_data,
    use_raw_pickles=use_raw_pickles,
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
    df=df_results_dneal,
    x_axis='reads',
    y_axis='perf_ratio',
    # instance=42,
    dict_fixed={
        'instance': 42,
        # 'schedule':'geometric'
    },
    ax=ax,
    list_dicts=[{'sweeps': i, 'schedule': j}
                for j in schedules_list for i in interesting_sweeps + [20]],
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
# Compute all instances with Dwave-neal
instance_list = [i for i in range(20)] + [42]
# %%
# Create all instances and save it into disk
for instance in instance_list:
    instance_file_name = prefix + str(instance) + ".txt"
    instance_file_name = os.path.join(instance_path, instance_file_name)

    if not os.path.exists(instance_file_name):
        # Fixing the random seed to get the same result
        np.random.seed(instance)
        J = np.random.rand(N, N)
        # We only consider upper triangular matrix ignoring the diagonal
        J = np.triu(J, 1)
        h = np.random.rand(N)
        model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

        text_file = open(instance_file_name, "w")
        text_file.write(model_random.to_coo())
        text_file.close()

# %%
# Compute random energy file
compute_random = False
if compute_random:
    for instance in instance_list:
        # Load problem instance
        np.random.seed(instance)
        J = np.random.rand(N, N)
        # We only consider upper triangular matrix ignoring the diagonal
        J = np.triu(J, 1)
        h = np.random.rand(N)
        ising_model = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)
        random_energy, _ = randomEnergySampler(
            ising_model, dwave_sampler=False)
        with open(os.path.join(results_path, "random_energies.txt"), "a") as gs_file:
            gs_file.write(prefix + str(instance) + " " +
                          str(random_energy) + " " + "best_found pysa\n")


# %%
# Merge all results dataframes in a single one
schedules_list = ['geometric']
df_list = []
use_raw_data = False
use_raw_pickles = False
all_boots_list = list(range(1, 1001, 1))
for instance in instance_list:
    df_name = "df_results_" + str(instance) + ".pkl"
    df_path = os.path.join(dneal_results_path, df_name)
    if os.path.exists(df_path):
        df_results_dneal = pd.read_pickle(df_path)
    else:
        df_results_dneal = None
    df_results_dneal = createDnealResultsDataframes(
        df=df_results_dneal,
        instance_list=[instance],
        parameters_dict={'schedule': schedules_list, 'sweeps': sweeps_list},
        boots_list=boots_list,
        dneal_results_path=dneal_results_path,
        dneal_pickle_path=dneal_pickle_path,
        use_raw_data=use_raw_data,
        use_raw_pickles=use_raw_pickles,
        overwrite_pickles=overwrite_pickles,
        s=s,
        confidence_level=conf_int,
        bootstrap_iterations=bootstrap_iterations,
        gap=gap,
        fail_value=fail_value,
        save_pickle=True,
    )

    # Loop over the dataframes with 4 values of sweeps and a sweep in boots then compute the results, and complete by creating main Dataframe
    interesting_sweeps = [
        df_results_dneal[df_results_dneal['boots'] == default_boots].nsmallest(1, 'tts')[
            'sweeps'].values[0],
        1,
        10,
        default_sweeps // 2,
        default_sweeps,
    ]

    df_results_dneal = createDnealResultsDataframes(
        df=df_results_dneal,
        instance_list=[instance],
        parameters_dict={'schedule': schedules_list,
                         'sweeps': interesting_sweeps},
        boots_list=all_boots_list,
        dneal_results_path=dneal_results_path,
        dneal_pickle_path=dneal_pickle_path,
        use_raw_data=use_raw_data,
        use_raw_pickles=use_raw_pickles,
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
df_name = "df_results.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_results_all = cleanup_df(df_results_all)
df_results_all.to_pickle(df_path)

# %%
# Run all the instances with Dwave-neal
overwrite_pickles = False
use_raw_data = False
use_raw_pickles = False
# schedules_list = ['geometric', 'linear']
schedules_list = ['geometric']

df_results_all = createDnealResultsDataframes(
    df=df_results_all,
    instance_list=instance_list,
    parameters_dict={'schedule': schedules_list, 'sweeps': sweeps_list},
    boots_list=boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_data=use_raw_data,
    use_raw_pickles=use_raw_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
)
# %%
# Compute preliminary ground state file with best found solution by Dwave-neal
compute_dneal_gs = False
if compute_dneal_gs:
    for instance in instance_list:
        # List all the pickled filed for an instance files
        pickle_list = createDnealExperimentFileList(
            directory=dneal_pickle_path,
            instance_list=[instance],
            prefix='df_' + prefix,
            suffix='.pkl'
        )
        min_energies = []
        min_energy = np.inf
        for file in pickle_list:
            df_samples = pd.read_pickle(file)
            if min_energy > df_samples['energy'].min():
                min_energy = df_samples['energy'].min()
                print(file)
                print(min_energy)
                min_energies.append(min_energy)
                min_df_samples = df_samples.copy()

        with open(os.path.join(results_path, "gs_energies.txt"), "a") as gs_file:
            gs_file.write(prefix + str(instance) + " " +
                          str(np.nanmin(min_energies)) + "  best_found dneal\n")
# %%
# Define function for ensemble averaging


def mean_conf_interval(
    x: pd.Series,
    key_string: str,
):
    '''
    Compute the mean and confidence interval of a series

    Args:
        x (pd.Series): Series to compute the mean and confidence interval
        key_string (str): String to use as key for the output dataframe

    Returns:
        pd.Series: Series with mean and confidence interval
    '''
    key_mean_string = 'mean_' + key_string
    result = {
        key_mean_string: x[key_string].mean(),
        key_mean_string + '_conf_interval_lower': x[key_string].mean() - np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string]))),
        key_mean_string + '_conf_interval_upper': x[key_string].mean() + np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string])))}
    return pd.Series(result)

# Define function for ensemble median


def median_conf_interval(
    x: pd.Series,
    key_string: str,
):
    '''
    Compute the median and confidence interval of a series (see http://mathworld.wolfram.com/StatisticalMedian.html for uncertainty propagation)

    Args:
        x (pd.Series): Series to compute the median and confidence interval
        key_string (str): String to use as key for the output dataframe

    Returns:
        pd.Series: Series with median and confidence interval
    '''
    key_median_string = 'median_' + key_string
    result = {
        key_median_string: x[key_string].median(),
        key_median_string + '_conf_interval_lower': x[key_string].median() - np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string]))) * np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1)),
        key_median_string + '_conf_interval_upper': x[key_string].median() + np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string]))) * np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1))}
    return pd.Series(result)

# Define function for ensemble metrics


def conf_interval(
    x: pd.Series,
    key_string: str,
    stat_measure: str = 'mean',
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
    key_median_string = stat_measure + '_' + key_string
    deviation = np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(
        x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string])))
    if stat_measure == 'mean':
        center = x[key_string].mean()
    else:
        center = x[key_string].median()
        deviation = deviation * \
            np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1))

    result = {
        key_median_string: center,
        key_median_string + '_conf_interval_lower': center - deviation,
        key_median_string + '_conf_interval_upper': center + deviation}
    return pd.Series(result)


# %%
# Generate stats results
use_raw_data = False
use_raw_pickles = False
df_results_all_stats = generateStatsDataframe(
    df_all=df_results_all,
    stat_measures=['mean', 'median'],
    instance_list=instance_list,
    parameters_dict={'schedule': schedules_list, 'sweeps': sweeps_list},
    resource_list=boots_list,
    dneal_results_path=dneal_results_path,
    use_raw_data=use_raw_data,
    use_raw_pickles=use_raw_pickles,
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
    df=df_results_all,
    x_axis='sweeps',
    y_axis='tts',
    ax=ax,
    dict_fixed={'schedule': 'geometric'},
    list_dicts=[{'instance': 42, 'boots': j}
                for j in [500, 1000]],
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
                for j in [500, 1000]],
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
    dict_fixed={'schedule': 'geometric', 'sweeps': 500},
    list_dicts=[{'boots': j}
                for j in all_boots_list[::10]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    ylim=[0.95, 1.005],
    # xlim=[1e2, 5e4],
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
    list_dicts=[{'sweeps': j}
                for j in interesting_sweeps],
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
# Gather all the data for the best tts of the ensemble for each instance
best_ensemble_sweeps = []
df_list = []
stat_measures = ['mean', 'median']
for stat_measure in stat_measures:
    best_sweep = df_results_all_stats[df_results_all_stats['boots'] == default_boots].nsmallest(
        1, stat_measure + '_tts')['sweeps'].values[0]
    best_ensemble_sweeps.append(best_sweep)
for instance in instance_list:
    df_name = "df_results_" + str(instance) + ".pkl"
    df_path = os.path.join(dneal_results_path, df_name)
    df_results_dneal = pd.read_pickle(df_path)
    df_results_dneal = createDnealResultsDataframes(
        df=df_results_dneal,
        instance_list=[instance],
        parameters_dict={'schedule': schedules_list,
                         'sweeps': best_ensemble_sweeps},
        boots_list=all_boots_list,
        dneal_results_path=dneal_results_path,
        dneal_pickle_path=dneal_pickle_path,
        use_raw_data=use_raw_data,
        use_raw_pickles=use_raw_pickles,
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
df_name = "df_results.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_results_all.to_pickle(df_path)

# %%
# Reload all results with the best tts of the ensemble for each instance
df_results_all = createDnealResultsDataframes(
    df=df_results_all,
    instance_list=instance_list,
    parameters_dict={'schedule': schedules_list,
                     'sweeps': best_ensemble_sweeps},
    boots_list=all_boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_data=use_raw_data,
    use_raw_pickles=use_raw_pickles,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_iterations=bootstrap_iterations,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
)

# %%
# Obtain the tts for each instance in the median and the mean of the ensemble accross the sweeps
# TODO generalize this code. In general, one parameter (or several) are fixed in certain interesting values and then for all instances with the all other values of remaining parameters we report the metric output, everything at 1000 bootstraps

for metric in ['perf_ratio', 'success_prob', 'tts']:

    df_results_all[metric + '_lower'] = df_results_all[metric] - \
        df_results_all[metric + '_conf_interval_lower']
    df_results_all[metric + '_upper'] = df_results_all[metric] - \
        df_results_all[metric + '_conf_interval_upper']

    # These following lineas can be extracted from the loop
    df_default = df_results_all[(df_results_all['boots'] == default_boots) & (
        df_results_all['sweeps'] == default_sweeps)].set_index(['instance', 'schedule'])
    df_list = [df_default]
    keys_list = ['default']
    for i, sweep in enumerate(best_ensemble_sweeps):
        df_list.append(df_results_all[(df_results_all['boots'] == default_boots) & (
            df_results_all['sweeps'] == sweep)].set_index(['instance', 'schedule']))
        keys_list.append(stat_measures[i])
    # Until here can be done off-loop

    # Metrics that you want to minimize
    ascent = metric in ['tts', 'mean_time']
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

df_results_all = cleanup_df(df_results_all)
df_name = "df_results.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_results_all.to_pickle(df_path)
# %%
# Plot with performance ratio vs reads for interesting sweeps
for instance in [42, 0, 19]:

    interesting_sweeps = [
        df_results_all[(df_results_all['boots'] == default_boots) & (df_results_all['instance'] == instance)].nsmallest(1, 'tts')[
            'sweeps'].values[0],
        10,
        default_sweeps//2,
        default_sweeps,
    ] + list(set(best_ensemble_sweeps))
    f, ax = plt.subplots()
    ax = plot_1d_singleinstance_list(
        df=df_results_all,
        x_axis='reads',
        y_axis='perf_ratio',
        # instance=42,
        dict_fixed={
            'instance': instance,
            'schedule': 'geometric'
        },
        ax=ax,
        list_dicts=[{'sweeps': i}
                    for i in interesting_sweeps],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        default_dict={'schedule': 'geometric',
                      'sweeps': default_sweeps, 'boots': default_boots},
        use_colorbar=False,
        ylim=[0.975, None],
        # xlim=[5e2, 5e4],
    )

# %%
# Regenerate the dataframe with the statistics to get the complete performance plot

# Regenerate the dataframe with the statistics to get the complete performance plot
df_results_all_stats = generateStatsDataframe(
    df_all=df_results_all,
    stat_measures=['mean', 'median'],
    instance_list=instance_list,
    parameters_dict={'schedule': schedules_list, 'sweeps': sweeps_list},
    resource_list=boots_list,
    dneal_results_path=dneal_results_path,
    use_raw_data=use_raw_data,
    use_raw_pickles=use_raw_pickles,
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
            list_dicts=[{'sweeps': i, 'instance': instance}
                        for i in [10, 500, default_sweeps] + list(set(best_ensemble_sweeps))],
            labels=labels,
            prefix=prefix,
            log_x=True,
            log_y=False,
            save_fig=False,
            ylim=[0.975, 1.0025],
            xlim=[5e2, 5e4],
            use_colorbar=False,
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
        list_dicts=[{'sweeps': i}
                    for i in [10, 500, default_sweeps] + list(set(best_ensemble_sweeps))],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        ylim=[0.975, 1.0025],
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
df_virtual_all = df_results_all.groupby(
    ['schedule', 'reads']
).apply(lambda s: pd.Series({
        'virt_best_tts': np.nanmin(s['tts']),
        'virt_best_perf_ratio': np.nanmax(s['perf_ratio']),
        'virt_best_success_prob': np.nanmax(s['success_prob']),
        'virt_best_mean_time': np.nanmin(s['mean_time']),
        'virt_worst_perf_ratio': np.nanmin(s['perf_ratio'])
        })
        ).reset_index()
df_virtual_max = df_virtual_all[
    ['reads', 'schedule',
     'virt_best_perf_ratio', 'virt_best_success_prob']].sort_values(
    'reads'
).groupby(
    'schedule'
).expanding(min_periods=1).max().droplevel(-1).reset_index()

df_virtual_max_w = df_virtual_all[
    ['reads', 'schedule',
     'virt_worst_perf_ratio']].sort_values(
    'reads', ascending=False
).groupby(
    'schedule'
).expanding(min_periods=1).min().droplevel(-1).reset_index()
if 'virt_worst_perf_ratio' in df_results_all.columns:
    df_results_all.drop('virt_worst_perf_ratio', axis=1, inplace=True)
if 'virt_best_perf_ratio' in df_results_all.columns:
    df_results_all.drop('virt_best_perf_ratio', axis=1, inplace=True)
if 'virt_best_success_prob' in df_results_all.columns:
    df_results_all.drop('virt_best_success_prob', axis=1, inplace=True)
if 'virt_best_mean_time' in df_results_all.columns:
    df_results_all.drop('virt_best_mean_time', axis=1, inplace=True)
if 'virt_best_tts' in df_results_all.columns:
    df_results_all.drop('virt_best_tts', axis=1, inplace=True)
df_results_all = df_results_all.merge(
    df_virtual_max,
    on=['schedule', 'reads'],
    how='outer')
df_results_all = df_results_all.merge(
    df_virtual_max_w,
    on=['schedule', 'reads'],
    how='outer')
df_virtual_min = df_virtual_all[
    ['reads', 'schedule',
     'virt_best_tts', 'virt_best_mean_time']].sort_values(
    'reads'
).groupby(
    'schedule'
).expanding(min_periods=1).max().droplevel(-1).reset_index()
df_results_all = df_results_all.merge(
    df_virtual_min,
    on=['schedule', 'reads'],
    how='outer')
df_results_all = cleanup_df(df_results_all)
df_name = "df_results.pkl"
df_path = os.path.join(dneal_results_path, df_name)
df_results_all.to_pickle(df_path)

# %%
# Generate plots for performance ratio of ensemble vs reads with best and worst performance
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    # for instance in instance_list:
    #     plot_1d_singleinstance_list(
    #         df=df_results_all,
    #         x_axis='reads',
    #         y_axis='perf_ratio',
    #         ax=ax,
    #         dict_fixed={'schedule': 'geometric'},
    #         list_dicts=[{'sweeps': i, 'instance': instance}
    #                     for i in [10, 500, default_sweeps] + list(set(best_ensemble_sweeps))],
    #         labels=labels,
    #         prefix=prefix,
    #         log_x=True,
    #         log_y=False,
    #         save_fig=False,
    #         ylim=[0.975, 1.0025],
    #         xlim=[5e2, 5e4],
    #         use_colorbar=False,
    #         alpha=0.25,
    #         colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
    #     )
    plot_1d_singleinstance(
        df=df_results_all,
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
        df=df_results_all,
        x_axis='reads',
        y_axis='virt_worst_perf_ratio',
        ax=ax,
        label_plot='Virtual worst',
        dict_fixed={'schedule': 'geometric'},
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
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
        list_dicts=[{'sweeps': i}
                    for i in [10, 500, default_sweeps] + list(set(best_ensemble_sweeps))],
        labels=labels,
        prefix=prefix,
        log_x=True,
        log_y=False,
        save_fig=False,
        ylim=[0.975, None],
        # xlim=[5e2, 5e4],
        use_colorbar=False,
        linewidth=1.5,
        markersize=1,
        colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
    )
# %%
# Defining which datapoints to take
repetitions = 10  # Times to run the algorithm
r = 1  # resource per parameter setting (runs)
rs = [1]  # Downsampling of each sweep result (runs/sweep)
experiments = rs * repetitions
R_exploration = 100  # budget for exploration (runs)
R_budget = 1e5  # budget for exploitation (runs)
R_exploitation = R_budget - R_exploration  # budget for exploitation (runs)
progress_list = []
for i, r in enumerate(experiments):
    random_sweeps = np.random.choice(
        sweeps_list, size=int(R_exploration / r), replace=True)
    # Conservative estimate of very unlikely scenario that we choose all sweeps=1
    # % Question: Should we replace these samples?
    series_list = []
    total_reads = 0
    for sweep in random_sweeps:
        series_list.append(
            df_results_all_stats.set_index(
                ['schedule', 'sweeps', 'boots']
            ).loc[
                idx['geometric', sweep, r]]
        )
        total_reads += sweep
        if total_reads > R_exploration:
            converged = True
            break
    df_progress = pd.concat(series_list, axis=1).T.rename_axis(
        ['schedule', 'sweeps', 'boots'])
    df_progress['median_perf_ratio'] = df_progress['median_perf_ratio'].expanding(
        min_periods=1).max()
    df_progress.reset_index('boots', inplace=True)
    df_progress['experiment'] = i
    df_progress['cum_reads'] = df_progress.groupby('experiment').expanding(
        min_periods=1)['reads'].sum().reset_index(drop=True).values
    progress_list.append(df_progress)

    exploitation_step = df_results_all_stats.set_index(
        ['schedule', 'sweeps']).loc[df_progress.nlargest(1, 'median_perf_ratio').index]
    exploitation_step['cum_reads'] = exploitation_step['reads'] + \
        df_progress['cum_reads'].max()
    exploitation_step['median_perf_ratio'].fillna(0, inplace=True)
    exploitation_step['median_perf_ratio'].clip(
        lower=df_progress['median_perf_ratio'].max(), inplace=True)
    exploitation_step['median_perf_ratio'] = exploitation_step['median_perf_ratio'].expanding(
        min_periods=1).max()
    exploitation_step['experiment'] = i
    exploitation_step = exploitation_step[exploitation_step['reads'] <= R_budget]
    progress_list.append(exploitation_step)
df_progress_total = pd.concat(progress_list, axis=0)
df_progress_total.reset_index(inplace=True)

# .T.reset_index().rename(columns={'level_0':'schedule', 'level_1':'Tfactor', 'level_2':'reads'})

# %%
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_progress_total,
    x_axis='cum_reads',
    y_axis='median_perf_ratio',
    ax=ax,
    dict_fixed={'schedule': 'geometric'},
    # label_plot='Ordered exploration',
    list_dicts=[{'experiment': i}
                for i in range(len(experiments))],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    colors=['colormap'],
    colormap=plt.cm.get_cmap('tab10'),
    use_colorbar=True,
    use_conf_interval=False,
    save_fig=False,
    ylim=[0.95, 1.0025],
    # xlim=[5e2, 5e4],
    linewidth=1.5,
    marker=None,
)
# %%
# Generate plots for performance ratio of ensemble vs reads with best and worst performance
for stat_measure in stat_measures:
    f, ax = plt.subplots()
    plot_1d_singleinstance(
        df=df_results_all,
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
        df=df_results_all,
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
        list_dicts=[{'sweeps': i}
                    for i in list(set(best_ensemble_sweeps))],
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
        colors=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
    )
    plot_1d_singleinstance_list(
        df=df_progress_total.reset_index(),
        x_axis='cum_reads',
        y_axis='median_perf_ratio',
        ax=ax,
        dict_fixed={'schedule': 'geometric'},
        # label_plot='Ordered exploration',
        list_dicts=[{'experiment': i}
                    for i in range(len(experiments))],
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
        xlim=[1e2, 5e4],
        linewidth=1.5,
        linestyle='--',
        markersize=1,
    )

# %%
