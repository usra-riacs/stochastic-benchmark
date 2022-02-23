# %%
# Import the Dwave packages dimod and neal
import functools
import itertools
import math
import os
import pickle
import pickle as pkl
import time
from collections import Counter
from ctypes.wintypes import DWORD
from functools import reduce
from gc import collect
from typing import List, Union

import dimod
import matplotlib.colors as mcolors
# Import Matplotlib to generate plots
import matplotlib.pyplot as plt
import neal
import networkx as nx
# Import numpy and scipy for certain numerical calculations below
import numpy as np
import pandas as pd
from dimod.vartypes import Vartype
from matplotlib import ticker
from pysa.sa import Solver
from scipy import sparse, stats

# %%
# Functions to retrieve instances files

# Define functions to extract data files


def getInstances(filename):  # instance number
    '''
    Extracts the instance from the filename assuming it is at the end before extension

    Args:
        filename: the name of the file

    Returns:
        instance: the instance number
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 2)[-1])


def createInstanceFileList(directory, instance_list):
    '''
    Creates a list of files in the directory for the isntances in the list

    Args:
        directory: the directory where the files are
        instance_list: the list of instances

    Returns:
        instance_file_list: the list of files    
    '''
    fileList = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and not f.endswith('.zip') and not f.endswith('.sh'))]
        # exclude gs_energies.txt files
        files = [f for f in files
                 if(not f.startswith('gs_energies'))]
        # Below, select only specifed n,s,alpha instances
        files = [f for f in files if(getInstances(f) in instance_list)]
        for f in files:
            fileList.append(root+"/"+f)
    return fileList


def getInstancePySAExperiment(filename):  # instance number
    '''
    Extracts the instance number from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        instance: the instance number    
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 9)[-9])


def getSweepsPySAExperiment(filename):
    '''
    Extracts the sweeps from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        sweeps: the number of sweeps
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 7)[-7])


def getPHot(filename):  # P hot
    '''
    Extracts the hot temperature transition probability from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        phot: the hot temperature transition probability
    '''
    return float(filename.rsplit(".", 1)[0].rsplit("_", 2)[-1])


def getPCold(filename):  # P cold
    '''
    Extracts the cold temperature transition probability from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        pcold: the cold temperature transition probability
    '''
    return float(filename.rsplit(".", 1)[0].rsplit("_", 3)[-3])


def getReplicas(filename):  # replicas
    '''
    Extracts the replicas from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        replicas: the number of replicas
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 5)[-5])


def createPySAExperimentFileList(
    directory: str,
    instance_list: Union[List[str], List[int]],
    rep_list: Union[List[str], List[int]] = None,
    sweep_list: Union[List[str], List[int]] = None,
    pcold_list: Union[List[str], List[float]] = None,
    phot_list: Union[List[str], List[float]] = None,
) -> list:
    '''
    Creates a list of experiment files in the directory for the instances in the instance_list, replicas in the rep_list, sweeps in the sweep_list, P cold in the pcold_list, and P hot in the phot_list

    Args:
        directory: the directory where the files are
        instance_list: the list of instances
        rep_list: the list of replicas
        sweep_list: the list of sweeps
        pcold_list: the list of P cold
        phot_list: the list of P hot

    Returns:
        experiment_file_list: the list of files

    '''
    fileList = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and
                    not f.endswith('.zip') and
                    not f.endswith('.sh') and
                    not f.endswith('.p') and
                    not f.startswith('results_'))]
        # exclude gs_energies.txt files
        files = [f for f in files
                 if(not f.startswith('gs_energies'))]
        # Below, select only specifed instances
        files = [f for f in files if(
            getInstancePySAExperiment(f) in instance_list)]
        # Consider replicas if provided list
        if rep_list is not None:
            files = [f for f in files if(
                getReplicas(f) in rep_list)]
        # Consider sweeps if provided list
        if sweep_list is not None:
            files = [f for f in files if(
                getSweepsPySAExperiment(f) in sweep_list)]
        # Consider pcold if provided list
        if pcold_list is not None:
            files = [f for f in files if(
                getPCold(f) in pcold_list)]
        # Consider phot if provided list
        if phot_list is not None:
            files = [f for f in files if(
                getPHot(f) in phot_list)]
        for f in files:
            fileList.append(root+"/"+f)

        # sort filelist by instance
        fileList = sorted(fileList, key=lambda x: getInstancePySAExperiment(x))
    return fileList


def getSchedule(filename):
    '''
    Extracts the schedule from the Dwave-neal experiment filename assuming the filename follows the naming convention prefix_instance_schedule_sweeps.extension

    Args:
        filename: the name of the file

    Returns:
        schedule: the schedule string
    '''
    return filename.rsplit(".", 1)[0].rsplit("_", 2)[-2]


def getSweepsDnealExperiment(filename):
    '''
    Extracts the sweeps from the Dwave-neal experiment filename assuming the filename follows the naming convention prefix_instance_schedule_sweeps.extension

    Args:
        filename: the name of the file

    Returns:
        sweep: the schedule string
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 1)[-1])


def getInstanceDnealExperiment(filename):
    '''
    Extracts the instance from the Dwave-neal experiment filename assuming the filename follows the naming convention prefix_instance_schedule_sweeps.extension

    Args:
        filename: the name of the file

    Returns:
        sweep: the sweep string
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 3)[-3])


def createDnealExperimentFileList(
    directory: str,
    instance_list: Union[List[str], List[int]],
    sweep_list: Union[List[str], List[int]] = None,
    schedule_list: List[str] = None,
) -> list:
    '''
    Creates a list of experiment files in the directory for the instances in the instance_list, sweeps in the sweep_list, and schedules in the schedule_list

    Args:
        directory: the directory where the files are
        instance_list: the list of instances
        sweep_list: the list of sweeps
        schedule_list: the list of schedules

    Returns:
        experiment_file_list: the list of files

    '''
    fileList = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and
                    not f.endswith('.zip') and
                    not f.endswith('.sh') and
                    not f.endswith('.p') and
                    not f.startswith('results_'))]
        # exclude gs_energies.txt files
        files = [f for f in files
                 if(not f.startswith('gs_energies'))]
        # Below, select only specifed instances
        files = [f for f in files if(
            getInstanceDnealExperiment(f) in instance_list)]
        # Consider sweeps if provided list
        if sweep_list is not None:
            files = [f for f in files if(
                getSweepsDnealExperiment(f) in sweep_list)]
        # Consider schedules if provided list
        if schedule_list is not None:
            files = [f for f in files if(
                getSchedule(f) in schedule_list)]
        for f in files:
            fileList.append(root+"/"+f)

        # sort filelist by instance
        fileList = sorted(
            fileList, key=lambda x: getInstanceDnealExperiment(x))
    return fileList


# %%
# Helper functions
# Some useful functions to get plots
def plotEnergyValuesDwaveSampleSet(
    results: dimod.SampleSet,
    title: str = None,
):
    '''
    Plots the energy values of the samples in a bar plot using as an impit a Dmid.sampleset.

    Args:
        results: A dimod.SampleSet object.
        title: A string to use as the plot title.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''

    _, ax = plt.subplots()

    energies = [datum.energy for datum in results.data(
        ['energy'], sorted_by='energy')]

    if results.vartype == Vartype.BINARY:
        samples = [''.join(c for c in str(datum.sample.values()).strip(
            ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        ax.set(xlabel='bitstring for solution')
    else:
        samples = np.arange(len(energies))
        ax.set(xlabel='solution')

    ax.bar(samples, energies)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel('Energy')
    if title:
        ax.set_title(str(title))
    print("minimum energy:", min(energies))
    return ax


def plotBarValues(
    df: pd.DataFrame,
    column_name: str,
    sorted: bool = True,
    skip: int = 1,
    xlabel: str = None,
    ylabel: str = None,
    title: str = None,
    save_fig: bool = False,
    **kwargs,
) -> plt.Figure:
    '''
    Plots the values of a column in a bar plot.

    Args:
        df: A pandas dataframe.
        column_name: A string of the column name.
        sorted: A boolean to sort the dataframe.
        skip: An integer of the number of rows to skip.
        xlabel: A string of the xlabel.
        ylabel: A string of the ylabel.
        title: A string of the title.
        save_fig: A boolean to save the figure.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    if sorted:
        df = df.sort_values(by=column_name)
    df.plot(y=column_name, kind='bar', ax=ax, **kwargs)
    ax.figure.tight_layout()
    new_ticks = np.arange(0, len(df.index) // skip)*skip
    # positions of each tick, relative to the indices of the x-values
    if column_name == 'sample':
        new_ticks
    ax.set_xticks(new_ticks)
    # labels
    ax.set_xticklabels(new_ticks)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if save_fig:
        plt.savefig(title+'.png')
    return ax


def plotBarCounts(
    df: pd.DataFrame,
    column_name: str,
    normalized: bool = False,
    sorted: bool = True,
    ascending: bool = False,
    skip: int = 1,
    xlabel: str = None,
    title: str = None,
    save_fig: bool = False,
    **kwargs,
) -> plt.Figure:
    '''
    Plots the counts of a column in a bar plot.

    Args:
        df: A pandas dataframe.
        column_name: A string of the column name.
        normalized: A boolean to normalize the counts.
        sorted: A boolean to sort the dataframe.
        ascending: A boolean to sort the plot in ascending order.
        skip: An integer of the number of rows to skip.
        xlabel: A string of the xlabel.
        title: A string of the title.
        save_fig: A boolean to save the figure.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    series = df[column_name].value_counts(normalize=normalized)
    if sorted:
        series = series.sort_values(ascending=ascending)
    series.plot(kind='bar', ax=ax, **kwargs)
    ax.figure.tight_layout()
    if column_name == 'sample':
        new_ticks = [str(list(state.values())).replace(', ', '')
                     for i, state in enumerate(series.keys()) if not i % skip]
    else:
        new_ticks = [t.get_text()[:7] for i, t in enumerate(ax.get_xticklabels()) if not i %
                     skip]
    ax.set_xticks(ax.get_xticks()[::skip])
    # labels
    ax.set_xticklabels(new_ticks)
    if xlabel:
        ax.set_xlabel(xlabel)
    if normalized:
        ax.set_ylabel('Probability')
    else:
        ax.set_ylabel('Count')
    if title:
        ax.set_title(title)
    if save_fig:
        plt.savefig(title+'.png')
    return ax


def plotSamplesDwaveSampleSet(
    results: dimod.SampleSet,
    title: str = None,
    skip: int = 1,
):
    '''
    Plots the samples of the samples in a histogram.

    Args:
        results: A dimod.SampleSet object.
        title: A string to use as the plot title.
        skip: An integer to skip every nth sample.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    energies = results.data_vectors['energy']
    if results.vartype == Vartype.BINARY:
        samples = [''.join(c for c in str(datum.sample.values()).strip(
            ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        ax.set_xlabel('bitstring for solution')
    else:
        samples = np.arange(len(results))
        ax.set_xlabel('solution')

    counts = Counter(samples)
    total = len(samples)
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None, ax=ax)

    ax.tick_params(axis='x', rotation=80)
    ax.set_xticklabels([t.get_text()[:7] if not i %
                       skip else "" for i, t in enumerate(ax.get_xticklabels())])
    ax.set_ylabel('Probabilities')
    if title:
        ax.set_title(str(title))
    print("minimum energy:", min(energies))
    return ax


def plotEnergyCFDDwaveSampleSet(
    results: dimod.SampleSet,
    title: str = None,
    skip: int = 1,
):
    '''
    Plots the energy values of the samples in a cumulative distribution function.

    Args:
        results: A dimod.SampleSet object.
        title: A string to use as the plot title.
        skip: An integer to skip every nth sample.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    # skip parameter given to avoid putting all xlabels
    energies = results.data_vectors['energy']
    occurrences = results.data_vectors['num_occurrences']
    counts = Counter(energies)
    total = sum(occurrences)
    counts = {}
    for index, energy in enumerate(energies):
        if energy in counts.keys():
            counts[energy] += occurrences[index]
        else:
            counts[energy] = occurrences[index]
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None, ax=ax)
    ax.set_xticklabels([t.get_text()[:7] if not i %
                       skip else "" for i, t in enumerate(ax.get_xticklabels())])

    ax.set_xlabel('Energy')
    ax.set_ylabel('Probabilities')
    if title:
        ax.set_title(str(title))
    print("minimum energy:", min(energies))
    return ax


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
default_name = prefix + str(instance) + '_geometric_1000.p'
df_default_name = 'df_' + default_name + 'kl'
rerun_default = False
if not os.path.exists(os.path.join(dneal_pickle_path, default_name)) or rerun_default:
    print('Running default D-Wave-neal simulated annealing implementation')
    start = time.time()
    default_samples = sim_ann_sampler.sample(model_random, num_reads=1000)
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
# Schedule plot
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
use_raw_data = True
zip_name = os.path.join(dneal_results_path, 'results.zip')
if os.path.exists(zip_name) and use_raw_data:
    import zipfile
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(dneal_pickle_path)
    print('Results zip file has been extrated to ' + dneal_pickle_path)

# %%
# Compute minimum value using MIP
# Ground state computation
compute_mip_gs = True
# Which type of MIP formulation to use ("qubo", "lcbo", "qcbo")
mip_formulation = "qubo"
if not compute_mip_gs:
    pass
else:
    import pyomo.environ as pyo
    from pyomo.opt import SolverStatus, TerminationCondition

    # Obtain ground states if possible using Mixed-Integer Formulation through Pyomo
    # Set up MIP optimization results directory
    mip_results_path = os.path.join(results_path, "mip_results/")
    # Create unexisting directory
    if not os.path.exists(mip_results_path):
        print('MIP results ' + mip_results_path +
              ' does not exist. We will create it.')
        os.makedirs(mip_results_path)
    # Compute optimal solution using MIP and save it into ground state file
    # Other solvers are available using GLPK, CBC (timeout), GAMS, Gurobi, or CPLEX
    solver_name = "gurobi"
    mip_solver = pyo.SolverFactory(solver_name)
    bqm_bin = model_random.change_vartype("BINARY", inplace=False)
    offset = bqm_bin.offset
    nx_graph_bin = bqm_bin.to_networkx_graph()

    # Create instance
    pyo_model = pyo.ConcreteModel(name="Random SK problem " + str(instance))
    # Define variables
    # Node variables
    pyo_model.x = pyo.Var(nx_graph_bin.nodes(), domain=pyo.Binary)
    obj_expr = offset
    for i, val in nx.get_node_attributes(nx_graph_bin, 'bias').items():
        obj_expr += val * pyo_model.x[i]

    if mip_formulation == "qubo":
        # Direct QUBO formulation
        obj_expr += pyo.quicksum(nx_graph_bin[i][j]['bias'] * pyo_model.x[i] * pyo_model.x[j]
                                 for (i, j) in nx_graph_bin.edges())
        # for (i, j) in nx_graph_bin.edges():
        #     # We want all edges to be sorted  with i-j and i<j
        #     assert(i < j)
        #     if i != j:
        #         obj_expr += nx_graph_bin[i][j]['bias'] * instance.x[i]*instance.x[j]
        #     else:
        #         print("Graph with self-edges" + str(i))

    elif mip_formulation == "lcbo":
        # Linear Constrained Binary Optimization

        # Edge variables
        pyo_model.y = pyo.Var(nx_graph_bin.edges(), domain=pyo.Binary)

        # add model constraints
        pyo_model.c1 = pyo.ConstraintList()
        pyo_model.c2 = pyo.ConstraintList()
        pyo_model.c3 = pyo.ConstraintList()

        for (i, j) in nx_graph_bin.edges():
            # We want all edges to be sorted  with i-j and i<j
            assert(i < j)
            if i != j:
                pyo_model.c1.add(pyo_model.y[i, j] <=
                                 pyo_model.x[i])
                pyo_model.c2.add(pyo_model.y[i, j] <=
                                 pyo_model.x[j])
                pyo_model.c3.add(pyo_model.y[i, j] >=
                                 pyo_model.x[i] + pyo_model.x[j] - 1)
                obj_expr += nx_graph_bin[i][j]['bias'] * pyo_model.y[i, j]
            else:
                print("Graph with self-edges" + str(i))
    else:
        print("Formulation not implemented yet")

    # Define the objective function
    pyo_model.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
    # Solve
    if solver_name == "gurobi":
        mip_solver.options['NonConvex'] = 2
        mip_solver.options['MIPGap'] = 1e-9
        mip_solver.options['TimeLimit'] = 30
    elif solver_name == "gams":
        mip_solver.options['solver'] = 'baron'
        mip_solver.options['solver'] = 'baron'
        mip_solver.options['add_options'] = 'option reslim=10;'

    results_dneal = mip_solver.solve(
        pyo_model,
        tee=True,
    )
    # result = opt_gams.solve(instance, tee=True)
    # Save solution
    obj_val = pyo_model.objective.expr()
    opt_sol = pd.DataFrame.from_dict(
        pyo_model.x.extract_values(), orient="index", columns=[str(pyo_model.x)])
    # Missing transformation back to spin variables here

    if (results_dneal.solver.status == SolverStatus.ok) and (results_dneal.solver.termination_condition == TerminationCondition.optimal):
        opt_sol.to_csv(mip_results_path + instance_name + "_" + str(obj_val) +
                       "_opt_sol.txt", header=None, index=True, sep=" ")
        with open(mip_results_path + "gs_energies.txt", "a") as gs_file:
            gs_file.write(instance_name + " " +
                          str(obj_val) + " " + str(results_dneal.solver.time) + " " + mip_formulation + " " + solver_name + "\n")

    else:
        opt_sol.to_csv(mip_results_path + instance_name + "_" + str(obj_val) +
                       "_sol.txt", header=None, index=True, sep=" ")
        with open(mip_results_path + "gs_energies.txt", "a") as gs_file:
            gs_file.write(instance_name + " " +
                          str(obj_val) + " " + str(results_dneal.solver.time) +
                          " " + str(results_dneal.solver.gap) + " " + mip_formulation + " " + solver_name + " suboptimal\n")

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
    schedule: str = 'geometric',
    sweep: int = 1000,
    total_reads: int = 1000,
    sim_ann_sampler=None,
    dneal_pickle_path: str = None,
    use_raw_data: bool = False,
    overwrite_pickles: bool = False,
) -> pd.DataFrame:
    '''
    Creates a dataframe with the samples for the dneal algorithm.
    '''
    # Gather instance names
    dict_pickle_name = prefix + str(instance) + "_" + schedule + \
        "_" + str(sweep) + ".p"
    df_samples_name = 'df_' + dict_pickle_name + 'kl'
    if use_raw_data or not os.path.exists(dneal_pickle_path + df_samples_name):
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
            # df_samples = pd.read_pickle(os.path.join(
            #     dneal_pickle_path, df_samples_name))
        # If it does not exist, generate the data
        else:
            start = time.time()
            samples = sim_ann_sampler.sample(
                model_random, num_reads=total_reads, num_sweeps=sweep, beta_schedule_type=schedule)
            time_s = time.time() - start
            samples.info['timing'] = time_s
            pickle.dump(samples, open(dict_pickle_name, "wb"))
        # Generate Dataframes
        df_samples = samples.to_pandas_dataframe(sample_column=True)
        df_samples['runtime (us)'] = int(
            1e6*samples.info['timing']/len(df_samples.index))
        df_samples.to_pickle(os.path.join(
            dneal_pickle_path, df_samples_name))
    else:
        df_samples = pd.read_pickle(os.path.join(
            dneal_pickle_path, df_samples_name))
    return df_samples


# %%
# Function to retrieve results from Dataframes
def computeResultsList(
    df: pd.DataFrame,
    random_energy: float = 0.0,
    min_energy: float = None,
    downsample: int = 10,
    bootstrap_samples: int = 1000,
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
        bootstrap_samples: The number of bootstrap samples.
        confidence_level: The confidence level for the bootstrap.
        gap: The threshold for the considering a read successful w.r.t the performance ratio [%].
        s: The success factor (usually said as RTT within s% probability).
        overwrite_pickles: If True, the pickles will be overwritten.

    Returns:
        A list of the results computed for analysis. Organized as follows
        [
            number of downsamples,
            bootstrapped mean minimum energy,
            boostrapped mean minimum energy confidence interval,
            bootstrapped performance ratio,
            bootstrapped performance ratio confidence interval,
            bootstrapped success probability,
            boostrapped success probability confidence interval,
            boostrapped resource to target,
            boostrapped resource to target confidence interval,
            boostrapped mean runtime,
            boostrapped mean runtime confidence interval,
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
        downsample, bootstrap_samples)).astype(int)

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
    min_boot_conf_interval = (stats.scoreatpercentile(
        min_boot_dist, 50-confidence_level/2), stats.scoreatpercentile(min_boot_dist, 50+confidence_level/2))

    # Compute the mean time of each bootstrap samples and its corresponding confidence interval based on the resamples
    times_dist = np.apply_along_axis(
        func1d=np.mean, axis=0, arr=times[resamples])
    mean_time = np.mean(times_dist)
    mean_time_conf_interval = (stats.scoreatpercentile(
        times_dist, 50-confidence_level/2), stats.scoreatpercentile(times_dist, 50+confidence_level/2))

    # Compute the performance ratio of each bootstrap samples and its corresponding confidence interval based on the resamples
    perf_ratio = (random_energy - min_boot) / (random_energy - min_energy)
    perf_ratio_conf_interval = ((random_energy - min_boot_conf_interval[1]) / (
        random_energy - min_energy), (random_energy - min_boot_conf_interval[0]) / (random_energy - min_energy))

    # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
    if aggregated_df_flag:
        return []
        # TODO: One can think about deaggregating the dataframe here. Maybe check Dwave's code for this.
    else:
        success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
            x < success_val)/downsample, axis=0, arr=energies[resamples])
    success_prob = np.mean(success_prob_dist)
    success_prob_conf_interval = (stats.scoreatpercentile(
        success_prob_dist, 50-confidence_level/2), stats.scoreatpercentile(success_prob_dist, 50+confidence_level/2))

    # Compute the TTT within certain threshold of each bootstrap samples and its corresponding confidence interval based on the resamples
    tts_dist = computeRRT_vectorized(
        success_prob_dist, s=s, scale=1e-6*df['runtime (us)'].sum(), fail_value=fail_value)
    # Question: should we scale the TTS with the number of bootstrapping we do, intuition says we don't need to
    tts = np.mean(tts_dist)
    if np.isinf(tts):
        tts_conf_interval = (np.inf, np.inf)
    else:
        # tts_conf_interval = computeRRT_vectorized(
        #     success_prob_conf_interval, s=0.99, scale=1e-6*df_default_samples['runtime (us)'].sum())
        tts_conf_interval = (stats.scoreatpercentile(
            tts_dist, 50-confidence_level/2), stats.scoreatpercentile(tts_dist, 50+confidence_level/2))
    # Question: How should we compute the confidence interval of the TTS? SHould we compute the function on the confidence interval of the probability or compute the confidence interval over the tts distribution?

    return [downsample, min_boot, min_boot_conf_interval, perf_ratio, perf_ratio_conf_interval, success_prob, success_prob_conf_interval, tts, tts_conf_interval, mean_time, mean_time_conf_interval]


# %%
# Function to update the dataframes
def createDnealResultsDataframes(
    df: pd.DataFrame = None,
    instance_list: List[int] = [0],
    parameters_dict: dict = None,
    boots_list: List[int] = [1000],
    dneal_results_path: str = None,
    dneal_pickle_path: str = None,
    use_raw_data: bool = False,
    overwrite_pickles: bool = False,
    confidence_level: float = 68,
    gap: float = 1.0,
    bootstrap_samples: int = 1000,
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
        parameters_dict: The parameters dictionary
        dneal_results_path: The path to the results
        dneal_pickle_path: The path to the pickle files
        use_raw_data: If we want to use the raw data
        overwrite_pickles: If we want to overwrite the pickle files

    '''

    # Check that the parameters are columns in the dataframe
    if df is not None:
        assert all([i in df.columns for i in parameters_dict.keys()])
        # In case that the data is already in the dataframe, return it
        if all([k in df[i].values for (i, j) in parameters_dict.items() for k in j]) and all([boots in df['boots'].values for boots in boots_list]):
            print('The dataframe already has all the data')
            return df

    # If use_raw_data compute the row
    if use_raw_data:
        list_results_dneal = []
        for instance in instance_list:
            if len(instance_list) > 1:
                df_name = "df_results.pkl"
            else:
                df_name = "df_results_" + str(instance) + ".pkl"
            df_path = os.path.join(dneal_results_path, df_name)
            random_energy = loadEnergyFromFile(os.path.join(
                results_path, 'random_energies.txt'), prefix + str(instance))
            min_energy = loadEnergyFromFile(os.path.join(
                results_path, 'gs_energies.txt'), prefix + str(instance))
            # We will assume that the insertion order in the keys is preserved (hence Python3.7+ only) and is sorted alphabetically
            combinations = itertools.product(
                *(parameters_dict[Name] for Name in sorted(parameters_dict)))
            combinations = list(combinations)
            for combination in combinations:
                list_inputs = [instance] + [i for i in combination]
                # Question: Is there a way of extracting the parameters names as variables names from the dictionary keys?
                # For the moment, let's hard-code it
                schedule = combination[0]
                sweep = combination[1]

                df_samples = createDnealSamplesDataframe(
                    instance=instance,
                    schedule=schedule,
                    sweep=sweep,
                    total_reads=total_reads,
                    sim_ann_sampler=sim_ann_sampler,
                    dneal_pickle_path=dneal_pickle_path,
                    use_raw_data=use_raw_data,
                    overwrite_pickles=overwrite_pickles,
                )

                for boots in boots_list:

                    if df is not None and schedule in df['schedule'].values and sweep in df['sweeps'].values and boots in df['boots'].values:
                        continue
                    else:
                        list_outputs = computeResultsList(
                            df=df_samples,
                            random_energy=random_energy,
                            min_energy=min_energy,
                            downsample=boots,
                            bootstrap_samples=bootstrap_samples,
                            confidence_level=confidence_level,
                            gap=gap,
                            s=s,
                            fail_value=fail_value,
                        )
                    list_results_dneal.append(
                        list_inputs + list_outputs)

        df_results_dneal = pd.DataFrame(list_results_dneal, columns=['instance', 'schedule', 'sweeps', 'boots', 'min_energy', 'min_energy_conf_interval',
                                        'perf_ratio', 'perf_ratio_conf_interval', 'success_prob', 'success_prob_conf_interval', 'tts', 'tts_conf_interval', 'mean_time', 'mean_time_conf_interval'])
        if df is not None:
            df = pd.concat(
                [df, df_results_dneal], axis=0, ignore_index=True)
        else:
            df = df_results_dneal

        if save_pickle:
            df.to_pickle(df_path)

    return df


# %%
# Compute results for instance 42 using D-Wave Neal
use_raw_data = True
overwrite_pickles = False
sweeps_list = [i for i in range(1, 21, 1)] + [
    i for i in range(20, 201, 10)] + [
    i for i in range(200, 501, 20)] + [
    i for i in range(500, 1001, 100)]
# sweeps_list = [1]
schedules_list = ['geometric', 'linear']
# schedules_list = ['geometric']
bootstrap_samples = 1000
total_reads = 1000
s = 0.99  # This is the success probability for the TTS calculation
gap = 1.0  # This is a percentual treshold of what the minimum energy should be
conf_int = 68  #
default_sweeps = 1000
default_boots = total_reads
fail_value = np.inf
# Confidence interval for bootstrapping, value used to get standard deviation
confidence_level = 68
boots_list = [1, 10, 100, default_boots]
sim_ann_sampler = neal.SimulatedAnnealingSampler()

df_name = "df_results_" + str(instance) + ".pkl"
df_path = os.path.join(dneal_results_path, df_name)
if os.path.exists(df_path) and not use_raw_data:
    df_results_dneal = pd.read_pickle(df_path)

df_results_dneal = createDnealResultsDataframes(
    df=df_results_dneal,
    instance_list=[42],
    parameters_dict={'schedule': schedules_list, 'sweeps': sweeps_list},
    boots_list=boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_data=use_raw_data,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_samples=bootstrap_samples,
    gap=gap,
    fail_value=fail_value,
)
# %%
# Plotting functions
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
    'min_energy': 'Minimum energy found',
    'mean_time': 'Mean time [us]',
    # 'tts': 'TTS to GS with 99% confidence \n [s * replica] ~ [MVM]',
}


def plot_1d_singleinstance(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    label_plot: str,
    dict_fixed: dict,
    ax: plt.Axes,
    log_x: bool = False,
    log_y: bool = False,
    save_fig: bool = False,
    default_dict: dict = None,
    **kwargs,
) -> plt.Axes:
    '''
    Function to plot 1D dependance figures

    Args:
        df: Dataframe with the data to plot
        x_axis: Name of the x axis
        y_axis: Name of the y axis
        label_plot: Name of the plot
        dict_fixed: Dictionary with the fixed values
        ax: Axes to plot on
        log_x: Logarithmic x axis
        log_y: Logarithmic y axis
        save_fig: Save the figure
        default_dict: Dictionary with the default values
        **kwargs: Additional arguments to pass to the plot function

    Returns:
        ax: Axes with the plot
    '''

    # Condition to enforce dict_fix conditions
    if dict_fixed is not None:
        cond = [df[k].apply(lambda k: k == v) for k, v in dict_fixed.items()]
        cond_total = functools.reduce(lambda x, y: x & y, cond)
        working_df = df[cond_total]
    else:
        working_df = df.copy()
    if y_axis == 'tts' or y_axis == 'tts_scaled':
        working_df.mask(
            working_df[y_axis] == fail_value).sort_values(
            x_axis
        ).plot(
            ax=ax,
            x=x_axis,
            y=y_axis,
            label=label_plot,
            style='-*',
            **kwargs,)
    else:
        working_df.sort_values(
            x_axis
        ).plot(
            ax=ax,
            x=x_axis,
            y=y_axis,
            label=label_plot,
            style='-*')

    if y_axis + '_conf_interval' in df.columns:
        conf_intervals = np.stack(working_df.sort_values(
            x_axis)[y_axis + '_conf_interval'].values).T
        ax.fill_between(
            working_df.sort_values(x_axis)[x_axis],
            conf_intervals[0],
            conf_intervals[1],
            alpha=0.2,
        )

    if default_dict is not None:
        cond_default = [df[k].apply(lambda k: k == v)
                        for k, v in default_dict.items()]
        cond_default_total = functools.reduce(lambda x, y: x & y, cond_default)
        default_val = df[cond_default_total][y_axis].values
        if default_val.size != 0:
            default_val = default_val[0]
            ax.axhline(default_val,
                       label='Default',
                       linestyle='--')

    ax.legend()
    if y_axis in labels.keys():
        ax.set(ylabel=labels[y_axis])
    else:
        ax.set(ylabel=y_axis)
    if x_axis in labels.keys():
        ax.set(xlabel=labels[x_axis])
    else:
        ax.set(xlabel=x_axis)
    if log_x:
        ax.set(xscale='log')
    if log_y:
        ax.set(yscale='log')
    ax.set(title='Plot N=' + str(N) +
           '\n' + y_axis + ' dependance with ' + x_axis + ', \n' +
                 ', '.join(str(key) + '=' + str(value) for key, value in dict_fixed.items()))

    if save_fig:
        plt.savefig(
            plots_path + y_axis + '_' + x_axis + '_fixed_' + '_'.join(str(key)
                                                                      for key in dict_fixed.keys())
            + '.png')

    return ax


def plot_1d_singleinstance_list(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    list_dicts: List[dict],
    ax: plt.Axes,
    dict_fixed: dict = None,
    log_x: bool = False,
    log_y: bool = False,
    use_colorbar: bool = False,
    save_fig: bool = False,
    default_dict: dict = None,
    colormap: plt.cm = None,
    **kwargs,
) -> plt.Axes:
    '''
    Function to plot 1D dependance figures

    Args:
        df: Dataframe with the data to plot
        x_axis: Name of the x axis
        y_axis: Name of the y axis
        list_dicts: List of dictionaries with the values to plot
        dict_fixed: Dictionary with the fixed values
        ax: Axes to plot on
        log_x: Logarithmic x axis
        log_y: Logarithmic y axis
        use_colorbar: Use colorbar
        save_fig: Save the figure
        **kwargs: Additional arguments to pass to the plot function

    Returns:
        ax: Axes with the plot
    '''
    if colormap is None:
        colormap = plt.cm.rainbow
    colors = colormap(np.linspace(0, 1, len(list_dicts)))
    title_str = 'Random N=' + \
        str(N) + '\n' + \
        y_axis + ' dependance with ' + x_axis

    # Condition to enforce dict_fix conditions
    if dict_fixed is not None:
        if 'instance' not in dict_fixed.keys():
            title_str += ', \n Ensemble'
        cond = [df[k].apply(lambda k: k == v) for k, v in dict_fixed.items()]
        cond_total = functools.reduce(lambda x, y: x & y, cond)
        working_df = df[cond_total]
        title_str += ', \n' + \
            ', '.join(str(key) + '=' + str(value)
                      for key, value in dict_fixed.items())
    else:
        working_df = df.copy()
    for i, line in enumerate(list_dicts):
        cond_part = [working_df[k].apply(lambda k: k == v)
                     for k, v in line.items()]
        cond_partial = functools.reduce(lambda x, y: x & y, cond_part)
        label_plot = ', '.join(str(key) + '=' + str(value)
                               for key, value in line.items())
        if 'instance' not in line.keys() and 'instance' not in dict_fixed.keys():
            label_plot = 'Ensemble, ' + label_plot
        if len(working_df[cond_partial]) == 0:
            continue
        else:
            if y_axis == 'tts' or y_axis == 'tts_scaled':
                working_df[cond_partial].mask(
                    working_df[cond_partial][y_axis] == fail_value).sort_values(
                    x_axis
                ).plot(
                    ax=ax,
                    x=x_axis,
                    y=y_axis,
                    label=label_plot,
                    color=colors[i],
                    style='-*',
                    **kwargs,)
            else:
                working_df[cond_partial].sort_values(
                    x_axis
                ).plot(
                    ax=ax,
                    x=x_axis,
                    y=y_axis,
                    label=label_plot,
                    color=colors[i],
                    style='-*',
                    **kwargs,)

        if y_axis + '_conf_interval' in df.columns and working_df[cond_partial][y_axis + '_conf_interval'].values.shape[0] > 1:
            conf_intervals = np.stack(working_df[cond_partial].sort_values(
                x_axis)[y_axis + '_conf_interval'].values).T
            ax.fill_between(
                working_df[cond_partial].sort_values(x_axis)[x_axis],
                conf_intervals[0],
                conf_intervals[1],
                alpha=0.2,
                color=colors[i],
            )

    if default_dict is not None:
        cond_default = [df[k].apply(lambda k: k == v)
                        for k, v in default_dict.items()]
        cond_default_total = functools.reduce(lambda x, y: x & y, cond_default)
        default_val = df[cond_default_total][y_axis].values
        if default_val.size != 0:
            default_val = default_val[0]
            ax.axhline(default_val,
                       label='Default',
                       linestyle='--')

    if use_colorbar:
        ax.get_legend().remove()
        # We assign the colormap according to the first key of the list_dicts, which we assume is sorted
        list_colorbar = [list(i.values())[0] for i in list_dicts]
        # Setup the colorbar
        normalize = mcolors.Normalize(
            vmin=min(list_colorbar), vmax=max(list_colorbar))
        scalarmappaple = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(list_colorbar)
        plt.colorbar(scalarmappaple, ax=ax, label=', '.join(
            labels[key] for key in line.keys()))
    else:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if y_axis in labels.keys():
        ax.set(ylabel=labels[y_axis])
    else:
        ax.set(ylabel=y_axis)
    if x_axis in labels.keys():
        ax.set(xlabel=labels[x_axis])
    else:
        ax.set(xlabel=x_axis)
    if log_x:
        ax.set(xscale='log')
    if log_y:
        ax.set(yscale='log')

    ax.set(title=title_str)

    if save_fig:
        plt.savefig(
            plots_path + y_axis + '_' + x_axis + '_fixed_' + '_'.join(str(key)
                                                                      for key in dict_fixed.keys())
            + '_list_' + '_'.join(str(key) for key in list_dicts[0].keys())
            + '.png')

    return ax


# %%
# Performance ratio vs sweeps for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_results_dneal,
    x_axis='sweeps',
    y_axis='perf_ratio',
    dict_fixed={'instance': 42, 'schedule': 'geometric'},
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict={'instance': 42, 'schedule': 'geometric',
                  'sweeps': 1000, 'boots': 1000},
    use_colorbar=False,
    ylim=[0.95, 1.005]
)
# %%
# Definition of reads metric as sweeps * total_reads
df_results_dneal['reads'] = df_results_dneal['boots'] * \
    df_results_dneal['sweeps']
df_results_dneal.to_pickle(df_path)
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
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict={'instance': 42, 'schedule': 'geometric',
                  'sweeps': 1000, 'boots': 1000},
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
    log_x=False,
    log_y=False,
    save_fig=False,
    default_dict={'instance': 42, 'schedule': 'geometric',
                  'sweeps': 1000, 'boots': 1000}
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
    log_x=False,
    log_y=False,
    save_fig=False,
    default_dict={'instance': 42, 'schedule': 'geometric',
                  'sweeps': 1000, 'boots': 1000}
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
    log_x=False,
    log_y=False,
    save_fig=False,
    default_dict={'instance': 42, 'schedule': 'geometric',
                  'sweeps': 1000, 'boots': 1000}
)
# %%
# Loop over the dataframes with 4 values of sweeps and a sweep in boots then compute the results, and complete by creating main Dataframe
interesting_sweeps = [
    df_results_dneal[df_results_dneal['boots'] == default_boots].nsmallest(1, 'tts')[
        'sweeps'].values[0],
    10,
    default_sweeps // 2,
    default_sweeps,
]

# Iterating for all values of bootstrapping downsampling proves to be very expensive, rather use steps of 10
all_boots_list = list(range(1, 1001, 10))

df_results_dneal = createDnealResultsDataframes(
    df=df_results_dneal,
    instance_list=[42],
    parameters_dict={'schedule': schedules_list, 'sweeps': interesting_sweeps},
    boots_list=all_boots_list,
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_data=use_raw_data,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_samples=bootstrap_samples,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
)
# %%
# Definition of reads metric as sweeps * total_reads
df_results_dneal['reads'] = df_results_dneal['boots'] * \
    df_results_dneal['sweeps']
df_results_dneal.to_pickle(df_path)

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
                for j in schedules_list for i in interesting_sweeps],
    # list_dicts=[{'sweeps': i for i in [190]}],
    # list_dicts=[{'boots': i for i in [1000]}],
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict={'schedule': 'geometric', 'sweeps': 1000, 'boots': 1000},
    use_colorbar=False,
    ylim=[0.95, 1.005],
    # xlim=[1e2, 5e4],
)
# %%
# Compute all instances with Dwave-neal
instance_list = [i for i in range(20)] + [42]


# %%
# Compute random energy file
compute_random = True

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
# Run all the instances with Dwave-neal
overwrite_pickles = False
use_raw_data = True
# schedules_list = ['geometric', 'linear']
schedules_list = ['geometric']

df_results_all = createDnealResultsDataframes(
    df=df_results_all,
    instance_list=instance_list,
    parameters_dict={'schedule': schedules_list, 'sweeps': sweeps_list},
    boots_list=boots_list + [500],
    dneal_results_path=dneal_results_path,
    dneal_pickle_path=dneal_pickle_path,
    use_raw_data=use_raw_data,
    overwrite_pickles=overwrite_pickles,
    s=s,
    confidence_level=conf_int,
    bootstrap_samples=bootstrap_samples,
    gap=gap,
    fail_value=fail_value,
    save_pickle=True,
)

# %%
# Definition of reads metric as sweeps * total_reads
all_results_name = "all_results.pkl"
all_results_name = os.path.join(dneal_pickle_path, all_results_name)
df_results_all['reads'] = df_results_all['boots'] * \
    df_results_all['sweeps']
df_results_all.to_pickle(all_results_name)
# #%%
# # If you wanto to use the raw data and process it here
# if use_raw_data:
#     for instance in instance_list:
#         # Fixing the random seed to get the same result
#         np.random.seed(instance)
#         J = np.random.rand(N, N)
#         # We only consider upper triangular matrix ignoring the diagonal
#         J = np.triu(J, 1)
#         h = np.random.rand(N)
#         model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)
#         for schedule in schedules_list:
#             for sweep in sweeps_list:
#                 # Gather instance names
#                 pickle_name = prefix + str(instance) + "_" + \
#                     schedule + "_" + str(sweep) + ".p"
#                 df_samples_name = 'df_' + pickle_name + 'kl'
#                 pickle_name = os.path.join(dneal_pickle_path, pickle_name)
#                 # If the instance data exists, load the data
#                 if os.path.exists(pickle_name) and not overwrite_pickles:
#                     samples = pickle.load(open(pickle_name, "rb"))
#                 # If it does not exist, generate the data
#                 else:
#                     start = time.time()
#                     samples = sim_ann_sampler.sample(
#                         model_random, num_reads=total_reads, num_sweeps=sweep, beta_schedule_type=schedule)
#                     time_s = time.time() - start
#                     samples.info['timing'] = time_s
#                     pickle.dump(samples, open(pickle_name, "wb"))

#                 # Generate the dataframe
#                 df_samples = samples.to_pandas_dataframe(
#                     sample_column=True)
#                 df_samples['runtime (us)'] = int(
#                     1e6*samples.info['timing']/len(df_samples.index))
#                 df_samples.to_pickle(os.path.join(
#                     dneal_pickle_path, df_samples_name))

# %%
# Compute preliminary ground state file with best found solution by Dwave-neal
compute_dneal_gs = True

if compute_dneal_gs:
    for instance in instance_list:
        # List all the pickled filed for an instance files
        pickle_list = createDnealExperimentFileList(directory=dneal_pickle_path,
                                                    instance_list=[instance])
        min_energies = []
        min_energy = 1000
        for file in pickle_list:
            # This is valid for old file format with dicts
            # samples = pickle.load(open(file, "rb"))
            # if min_energy > min(samples.data_vectors['energy']):
            #     min_energy = min(samples.data_vectors['energy'])
            #     print(file)
            #     print(min_energy)
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
# Get results pandas dataframe for each instance
# boots_list = [1, 10, 100, default_boots]
boots_list = [default_boots]
sweeps_list = [default_sweeps]
for sweep in sweeps_list:
    for boots in boots_list:
        df_results_dneal = createDnealResultsDataframes(
            df=df_results_dneal,
            instance=42,
            boots=boots,
            parameters_dict={'schedule': 'geometric', 'sweeps': sweep},
            dneal_results_path=dneal_results_path,
            dneal_pickle_path=dneal_pickle_path,
            use_raw_data=True,
            overwrite_pickles=False,
        )

# %%
# Join all the results in a single large dataframe
results_all_path = os.path.join(dneal_results_path, "df_results_all.pkl")
use_raw_data = True
if use_raw_data or not os.path.exists(results_all_path):
    df_results_all = pd.DataFrame()
    for instance in instance_list:
        df_path = os.path.join(
            dneal_results_path, "df_results_" + str(instance) + ".pkl")
        df_results_instance = pd.read_pickle(df_path)
        df_results_all = pd.concat(
            [df_results_all, df_results_instance], ignore_index=True)
    df_results_all.to_pickle(results_all_path)

# %%
# Split pandas dataframe confidence intervals into separate columns for easier processing
results_metrics_list = ['min_energy', 'tts',
                        'perf_ratio', 'success_prob', 'mean_time']
for metric in results_metrics_list:
    df_results_all[metric + '_conf_interval_lower'] = df_results_all[metric +
                                                                     '_conf_interval'].apply(lambda x: x[0])
    df_results_all[metric + '_conf_interval_upper'] = df_results_all[metric +
                                                                     '_conf_interval'].apply(lambda x: x[1])
# %%
# Define function for ensemble averaging


def mean_conf_interval(
    x: pd.Series,
    key_string: str,
):
    """
    Compute the mean and confidence interval of a series

    Args:
        x (pd.Series): Series to compute the mean and confidence interval
        key_string (str): String to use as key for the output dataframe

    Returns:
        pd.Series: Series with mean and confidence interval
    """
    key_conf_interval_string = 'mean_' + key_string + '_conf_interval'
    key_mean_string = 'mean_' + key_string
    result = {
        key_mean_string: x[key_string].mean(),
        key_conf_interval_string + '_lower': x[key_string].mean() - np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string]))),
        key_conf_interval_string + '_upper': x[key_string].mean() + np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string])))}
    return pd.Series(result)

# Define function for ensemble median


def median_conf_interval(
    x: pd.Series,
    key_string: str,
):
    """
    Compute the median and confidence interval of a series (see http://mathworld.wolfram.com/StatisticalMedian.html for uncertainty propagation)

    Args:
        x (pd.Series): Series to compute the median and confidence interval
        key_string (str): String to use as key for the output dataframe

    Returns:
        pd.Series: Series with median and confidence interval
    """
    key_conf_interval_string = 'median_' + key_string + '_conf_interval'
    key_mean_string = 'median_' + key_string
    result = {
        key_mean_string: x[key_string].median(),
        key_conf_interval_string + '_lower': x[key_string].median() - np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string]))) * np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1)),
        key_conf_interval_string + '_upper': x[key_string].median() + np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string]))) * np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1))}
    return pd.Series(result)


# %%
# Split large dataframe such that we can compute the statistics and confidence interval for each metric across the instances
parameters = ['schedule', 'sweeps']
df_results_all_groups = df_results_all.set_index(
    'instance').groupby(parameters + ['boots'])
df_list = []
for metric in results_metrics_list:
    df_results_all_mean = df_results_all_groups.apply(
        mean_conf_interval, key_string=metric)
    df_results_all_median = df_results_all_groups.apply(
        median_conf_interval, key_string=metric)
    df_list.append(df_results_all_mean)
    df_list.append(df_results_all_median)

df_results_all_stats = pd.concat(df_list, axis=1)
df_results_all_stats = df_results_all_stats.reset_index()

# %%
# Merge pandas dataframe confidence intervals into single column for easier processing
for metric in results_metrics_list:
    df_results_all_stats['mean_' + metric + '_conf_interval'] = df_results_all_stats[['mean_' +
                                                                                      metric + '_conf_interval_lower', 'mean_' + metric + '_conf_interval_upper']].apply(tuple, axis=1)
    df_results_all_stats['median_' + metric + '_conf_interval'] = df_results_all_stats[['median_' +
                                                                                        metric + '_conf_interval_lower', 'median_' + metric + '_conf_interval_upper']].apply(tuple, axis=1)


# %%
# Definition of reads metric as sweeps * total_reads
df_results_all_stats['reads'] = df_results_all_stats['boots'] * \
    df_results_all_stats['sweeps']

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
                for j in [100, 1000]],
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
    log_x=False,
    log_y=True,
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
                for j in [1, 10, 100, 1000]],
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
    dict_fixed={'schedule': 'geometric'},
    list_dicts=[{'boots': j}
                for j in [1, 10, 100, 1000]],
    log_x=True,
    log_y=False,
    save_fig=False,
    ylim=[0.95, 1.005],
    xlim=[1e2, 5e4],
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
    log_x=True,
    log_y=False,
    save_fig=False,
    ylim=[0.95, 1.005],
    xlim=[1e2, 5e4],
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
    log_x=True,
    log_y=False,
    save_fig=False,
    # ylim=[0.9, 1.005],
    # xlim=[1e2, 5e4],
)
# %%


def computeEnsembleResultsList(
    df: pd.DataFrame,
    random_energy: float = 0.0,
    min_energy: float = None,
    downsample: int = 10,
    bootstrap_samples: int = 500,
    confidence_level: float = 68,
    gap: float = 1.0,
    s: float = 0.99,
) -> list:
    '''
    Compute a list of the results computed for analysis given a dataframe from a solver.

    Args:
        df: The dataframe from the solver.
        random_energy: The mean energy of the random sample.
        min_energy: The minimum energy of the samples.
        downsample: The downsampling sample for bootstrapping.
        bootstrap_samples: The number of bootstrap samples.
        confidence_level: The confidence level for the bootstrap.
        gap: The threshold for the considering a read successful w.r.t the performance ratio [%].
        s: The success factor (usually said as RTT within s% probability).

    Returns:
        A list of the results computed for analysis. Organized as follows
        [
            number of downsamples,
            bootstrapped mean minimum energy,
            boostrapped mean minimum energy confidence interval,
            bootstrapped performance ratio,
            bootstrapped performance ratio confidence interval,
            bootstrapped success probability,
            boostrapped success probability confidence interval,
            boostrapped resource to target,
            boostrapped resource to target confidence interval,
            boostrapped mean runtime,
            boostrapped mean runtime confidence interval,
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
        len(df), bootstrap_samples)).astype(int)

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
    min_boot_conf_interval = (stats.scoreatpercentile(
        min_boot_dist, 50-confidence_level/2), stats.scoreatpercentile(min_boot_dist, 50+confidence_level/2))

    # Compute the mean time of each bootstrap samples and its corresponding confidence interval based on the resamples
    times_dist = np.apply_along_axis(
        func1d=np.mean, axis=0, arr=times[resamples])
    mean_time = np.mean(times_dist)
    mean_time_conf_interval = (stats.scoreatpercentile(
        times_dist, 50-confidence_level/2), stats.scoreatpercentile(times_dist, 50+confidence_level/2))

    # Compute the performance ratio of each bootstrap samples and its corresponding confidence interval based on the resamples
    perf_ratio = (random_energy - min_boot) / (random_energy - min_energy)
    perf_ratio_conf_interval = ((random_energy - min_boot_conf_interval[1]) / (
        random_energy - min_energy), (random_energy - min_boot_conf_interval[0]) / (random_energy - min_energy))

    # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
    if aggregated_df_flag:
        return []
        # TODO: One can think about deaggregating the dataframe here. Maybe check Dwave's code for this.
    else:
        success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
            x < success_val)/downsample, axis=0, arr=energies[resamples])
    success_prob = np.mean(success_prob_dist)
    success_prob_conf_interval = (stats.scoreatpercentile(
        success_prob_dist, 50-confidence_level/2), stats.scoreatpercentile(success_prob_dist, 50+confidence_level/2))

    # Compute the TTT within certain threshold of each bootstrap samples and its corresponding confidence interval based on the resamples
    tts_dist = computeRRT_vectorized(
        success_prob_dist, s=s, scale=1e-6*df['runtime (us)'].sum(), fail_value=np.inf)
    # Question: should we scale the TTS with the number of bootstrapping we do, intuition says we don't need to
    tts = np.mean(tts_dist)
    if np.isinf(tts):
        tts_conf_interval = (np.inf, np.inf)
    else:
        # tts_conf_interval = computeRRT_vectorized(
        #     success_prob_conf_interval, s=0.99, scale=1e-6*df_default_samples['runtime (us)'].sum())
        tts_conf_interval = (stats.scoreatpercentile(
            tts_dist, 50-confidence_level/2), stats.scoreatpercentile(tts_dist, 50+confidence_level/2))
    # Question: How should we compute the confidence interval of the TTS? Should we compute the function on the confidence interval of the probability or compute the confidence interval over the tts distribution?

    return [downsample, min_boot, min_boot_conf_interval, perf_ratio, perf_ratio_conf_interval, success_prob, success_prob_conf_interval, tts, tts_conf_interval, mean_time, mean_time_conf_interval]


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
# Run all instances with PySA
# Parameters: replicas = [1, 2, 4, 8], n_reads = 100, sweeps = 100, p_hot=50, p_cold = 1
# Input Parameters
total_reads = 1000
n_replicas_list = [1, 2, 4, 8]
# n_replicas_list = [4]
# sweeps = [i for i in range(
#     1, 21, 1)] + [i for i in range(
#         21, 101, 10)]
p_hot_list = [50.0]
p_cold_list = [1.0]
# instance_list = list(range(20)) + [42]
# instance_list = [1, 4, 11, 14, 15, 16] + [42]
# instance_list = [0,2,3,5,6,7,8,9,10,12,13,17,18,19]
# instance_list = [42]
use_raw_pickles = True
overwrite_pickle = False
float_type = 'float32'

# sweeps_list = [i for i in range(1, 21, 1)] + [
#     i for i in range(20, 501, 10)] + [
#     i for i in range(500, 1001, 20)]
# sweeps = [1000]

# Setup directory for PySA results
pysa_path = os.path.join(results_path, "pysa/")
# Create directory for PySA results
if not os.path.exists(pysa_path):
    os.makedirs(pysa_path)

if use_raw_pickles:

    # Setup directory for PySA pickles
    pysa_pickles_path = os.path.join(pysa_path, "pickles/")
    # Create directory for PySA pickles
    if not os.path.exists(pysa_pickles_path):
        os.makedirs(pysa_pickles_path)

    # List all the instances files
    file_list = createInstanceFileList(directory=instance_path,
                                       instance_list=instance_list)

    counter = 0
    for file in file_list:

        file_name = file.split(".txt")[0].rsplit("/", 1)[-1]
        print(file_name)

        data = np.loadtxt(file, dtype=float)
        M = sparse.coo_matrix(
            (data[:, 2], (data[:, 0], data[:, 1])), shape=(N, N))
        problem = M.A
        problem = problem+problem.T-np.diag(np.diag(problem))

        # Get solver
        solver = Solver(problem=problem, problem_type='ising',
                        float_type=float_type)

        for n_replicas in n_replicas_list:
            for n_sweeps in sweeps_list:
                for p_hot in p_hot_list:
                    for p_cold in p_cold_list:
                        counter += 1
                        print("file "+str(counter)+" of " + str(len(file_list)
                                                                * len(n_replicas_list) * len(sweeps_list) * len(p_hot_list) * len(p_cold_list)))

                        pickle_name = pysa_pickles_path + file_name + '_swe_' + str(n_sweeps) + '_rep_' + str(
                            n_replicas) + '_pcold_' + str(p_cold) + '_phot_' + str(p_hot) + '.pkl'

                        min_temp = 2 * \
                            np.min(np.abs(problem[np.nonzero(problem)])
                                   ) / np.log(100/p_cold)
                        min_temp_cal = 2*min(sum(abs(i)
                                                 for i in problem)) / np.log(100/p_cold)
                        max_temp = 2*max(sum(abs(i)
                                         for i in problem)) / np.log(100/p_hot)
                        if os.path.exists(pickle_name) and not overwrite_pickle:
                            print(pickle_name)
                            results_pysa = pd.read_pickle(pickle_name)
                            continue
                        print(pickle_name)
                        # Apply Metropolis
                        results_pysa = solver.metropolis_update(
                            num_sweeps=n_sweeps,
                            num_reads=total_reads,
                            num_replicas=n_replicas,
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
                        results_pysa.to_pickle(pickle_name)

# %%
# Compute preliminary ground state file with best found solution by PySA
compute_pysa_gs = True

if compute_pysa_gs:
    for instance in instance_list:
        # List all the pickled filed for an instance files
        pickle_list = createPySAExperimentFileList(directory=pysa_pickles_path,
                                                   instance_list=[instance])
        min_energies = []
        for file in pickle_list:
            df = pd.read_pickle(file)
            min_energies.append(df['best_energy'].min())

        with open(os.path.join(results_path, "gs_energies.txt"), "a") as gs_file:
            gs_file.write(prefix + str(instance) + " " +
                          str(np.nanmin(min_energies)) + " " + "best_found pysa\n")


# %%
# Load minimum found energy across each instance
def getMinPySAEnergy(
    directory: str,
    instance: Union[str, int],
) -> float:
    '''
    Load minimum found energy across each instance

    Args:
        directory: Directory where the PySA pickles are located
        instance: Instance number

    Returns:
        Minimum found energy
    '''
    # instance = int(instance_name.rsplit("_",1)[1])
    min_energies = [
        df_dneal[df_dneal['instance'] == instance]['best'].min()]
    file_list = createPySAExperimentFileList(
        directory=directory, instance_list=[instance])
    for file in file_list:
        df = pd.read_pickle(file)
        min_energies.append(df['best_energy'].min())
    return np.nanmin(min_energies)


# %%
# Create intermediate .data files with main information and unique_gs with unique groundstates information

# Set up directory for intermediate .data files
pysa_data_path = os.path.join(pysa_path, "data/")
# Create directory for intermediate .data files
if not os.path.exists(pysa_data_path):
    os.makedirs(pysa_data_path)


# Setup directory for unique ground states
pysa_gs_path = os.path.join(pysa_path, "unique_gs/")
# Create directory for unique ground states
if not os.path.exists(pysa_gs_path):
    os.makedirs(pysa_gs_path)

# Percentual tolerance to consider succesful runs
tol = 1

if use_raw_pickles:
    overwrite_files = True
    output_files_in_progress = []

    counter = 0
    for instance in instance_list:

        min_energy = loadEnergyFromFile(gs_file=results_path + "gs_energies.txt",
                                        instance_name=prefix + str(instance))

        # List all the instances files
        pickle_list = createPySAExperimentFileList(directory=pysa_pickles_path,
                                                   instance_list=[instance],
                                                   rep_list=n_replicas_list,
                                                   sweep_list=sweeps_list,
                                                   pcold_list=p_cold_list,
                                                   phot_list=p_hot_list)
        # print(pickle_list)
        for pickle_file in pickle_list:
            file_name = pickle_file.split(".pkl")[0].rsplit("/", 1)[-1]
            instance_name = file_name.split("_swe")[0]
            counter += 1
            output_file = pysa_data_path+file_name+".data"
            if overwrite_files or not os.path.exists(output_file):
                print(file_name + ": file "+str(counter) +
                      " of "+str(len(pickle_list)*len(instance_list)))
                if os.path.exists(pickle_file):
                    try:
                        results_pysa = pd.read_pickle(pickle_file)
                    except (pkl.UnpicklingError, EOFError):
                        os.replace(pickle_file, pickle_file + '.bak')
                        continue
                else:
                    print("Missing pickle file for " + file_name)
                    break

                sweeps_max = getSweepsPySAExperiment(pickle_file)

                num_sweeps = results_pysa["num_sweeps"][0]

                # Check that each file has as many reads as required
                n_reads_file = len(results_pysa["best_energy"])
                assert(total_reads == n_reads_file)

                if os.path.exists(output_file) and not(output_file in output_files_in_progress):
                    output_files_in_progress.append(output_file)
                    with open(output_file, "w") as fout:
                        fout.write("s read_num runtime(us) num_sweeps success_e" +
                                   str(-int(np.log10(tol)))+"\n")

                states_within_tolerance = []
                # Skip first read, as numba needs compile time
                runtimes = []
                successes = []
                for read_num in range(1, n_reads_file):
                    best_energy = results_pysa["best_energy"][read_num]
                    runtime = results_pysa["runtime (us)"][read_num]
                    runtimes.append(runtime)

                    best_sweeps = num_sweeps
                    success = 0
                    if(abs(float(best_energy)-float(min_energy))/float(min_energy) < tol/100):
                        best_sweeps = results_pysa["min_sweeps"][read_num]
                        states_within_tolerance.append(
                            results_pysa["best_state"][read_num])
                        success = 1
                    successes.append(success)

                    with open(output_file, "a") as fout:
                        fout.write("{s} {read} {runtime} {best_sweeps} {success}\n".
                                   format(s=num_sweeps,
                                          read=read_num,
                                          runtime=runtime,
                                          best_sweeps=best_sweeps,
                                          success=success))

                # Separate file with unique MAXCUT per instance
                unique_gs = np.unique(np.asarray(
                    states_within_tolerance), axis=0)
                with open(pysa_gs_path+output_file.split("/")[-1], "a") as fout:
                    fout.write("tol{tol} s{s} unqGS{unqGS} \n".format(
                        tol=-int(np.log10(tol)), s=num_sweeps, unqGS=len(unique_gs)))
            else:
                data = np.loadtxt(output_file, skiprows=1,
                                  usecols=(0, 1, 2, 3, 4))
                successes = data[:, 4]
                best_sweeps = data[:, 3]
                runtimes = data[:, 2]
# %%
# Create pickled Pandas framework with results for each instance

# Success rate for TTS computation
s = 0.99


for instance in instance_list:
    data_dict_name = "results_" + str(instance) + ".pickle"
    df_name = "df_results_" + str(instance) + ".pickle"

    file_list = createPySAExperimentFileList(directory=pysa_data_path,
                                             instance_list=[instance],
                                             rep_list=n_replicas_list,
                                             sweep_list=sweeps_list,
                                             pcold_list=p_cold_list,
                                             phot_list=p_hot_list)

    data_dict_path = os.path.join(pysa_path, data_dict_name)
    df_path = os.path.join(pysa_path, df_name)
    data_dict = {}
    counter = 0

    tts_list = []
    tts_scaled_list = []
    for file in file_list:
        counter += 1
        file_name = file.split(".data")[0].rsplit(
            "/", 1)[-1]
        print(file_name + ": file "+str(counter)+" of "+str(len(file_list)))

        # If you wanto to use the raw data and process it here
        if use_raw_data or not(os.path.exists(data_dict_path)) or not(os.path.exists(df_path)):
            instance = getInstancePySAExperiment(file)
            sweep = getSweepsPySAExperiment(file)
            replica = getReplicas(file)
            pcold = getPCold(file)
            phot = getPHot(file)
            #load in data, parameters
            # s,sweeps,runtime(us),best_sweeps,success
            data = np.loadtxt(file, skiprows=1, usecols=(0, 1, 2, 3, 4))

            # Computation of TTS across mean value of all reads in each PySA run
            successes = data[:, 4]
            best_sweeps = data[:, 3]
            runtimes = data[:, 2]
            success_rate = np.mean(successes)
            mean_time = np.mean(runtimes)  # us
            if success_rate == 0:
                tts_scaled = 1e15
                tts = 1e15
            # Consider continuous TTS and TTS scaled by assuming s=1 as s=1-1/1000*(1-1/10)
            elif success_rate == 1:
                tts_scaled = 1e-6*mean_time * \
                    np.log(1-s) / np.log(1-0.999+0.0001)  # s
                tts = replica*1e-6*mean_time * \
                    np.log(1-s) / np.log(1-0.999+0.00001)  # s * replica
            else:
                tts_scaled = 1e-6*mean_time * \
                    np.log(1-s) / np.log(1-success_rate)  # s
                tts = replica*1e-6*mean_time * \
                    np.log(1-s) / np.log(1-success_rate)  # s * replica
            tts_scaled_list.append(tts_scaled)
            tts_list.append(tts)
            if instance not in data_dict.keys():
                data_dict[instance] = {}
            if sweep not in data_dict[instance].keys():
                data_dict[instance][sweep] = {}
            if replica not in data_dict[instance][sweep].keys():
                data_dict[instance][sweep][replica] = {}
            if pcold not in data_dict[instance][sweep][replica].keys():
                data_dict[instance][sweep][replica][pcold] = {}
            if phot not in data_dict[instance][sweep][replica][pcold].keys():
                data_dict[instance][sweep][replica][pcold][phot] = {}

            # data_dict[instance][sweep][replica][pcold][phot]['success'] = data[:,4]
            data_dict[instance][sweep][replica][pcold][phot]['best_sweep'] = best_sweeps
            data_dict[instance][sweep][replica][pcold][phot]['success_rate'] = success_rate
            data_dict[instance][sweep][replica][pcold][phot]['mean_time'] = mean_time
            data_dict[instance][sweep][replica][pcold][phot]['tts'] = tts
            data_dict[instance][sweep][replica][pcold][phot]['tts_scaled'] = tts_scaled
            if(len(data[:, 1]) < replica):
                print('Missing replicas for instance' + str(file))
                print(len(data[:, 1]))
                pass

            # Save results dictionary in case that we are interested in reusing them
            pkl.dump(data_dict, open(data_dict_name, "wb"))

            # Create Pandas framework for the results
            # Restructure dictionary to dictionary of tuple keys -> values
            data_dict_2 = {(instance, sweep, replica, pcold, phot):
                           (data_dict[instance][sweep][replica][pcold][phot]['best_sweep'],
                            data_dict[instance][sweep][replica][pcold][phot]['success_rate'],
                            data_dict[instance][sweep][replica][pcold][phot]['mean_time'],
                            data_dict[instance][sweep][replica][pcold][phot]['tts'],
                            data_dict[instance][sweep][replica][pcold][phot]['tts_scaled'])
                           for instance in data_dict.keys()
                           for sweep in data_dict[instance].keys()
                           for replica in data_dict[instance][sweep].keys()
                           for pcold in data_dict[instance][sweep][replica].keys()
                           for phot in data_dict[instance][sweep][replica][pcold].keys()}

            # Construct dataframe from dictionary
            df_dneal = pd.DataFrame.from_dict(
                data_dict_2, orient='index').reset_index()

            # Split column of tuples to multiple columns
            df_dneal[['instance', 'sweeps', 'replicas', 'pcold',
                      'phot']] = df_dneal['index'].apply(pd.Series)

            # Clean up: remove unwanted columns, rename and sort
            df_dneal = df_dneal.drop('index', 1).\
                rename(columns={0: 'best_sweep', 1: 'success_rate', 2: 'mean_time', 3: 'tts', 4: 'tts_scaled'}).\
                sort_index(axis=1)

            df_dneal.to_pickle(df_path)
        else:  # Just reload processed datafile
            # data_dict = pkl.load(open(results_name, "rb"))
            pass
# %%
# %%
