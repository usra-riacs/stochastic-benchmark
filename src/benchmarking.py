# %%
# Import the Dwave packages dimod and neal
from ctypes.wintypes import DWORD
from gc import collect
import itertools
import math
import os
import pickle
import pickle as pkl
import time
from collections import Counter
from functools import reduce
from itertools import chain
from typing import List, Union

import dimod
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
random_energy, random_sample = randomEnergySampler(model_random, num_reads=1000,dwave_sampler=True)
df_random_sample = random_sample.to_pandas_dataframe(sample_column=True)
print('Average random energy = ' + str(random_energy))


# %%
# Plot of obtained energies
# plotEnergyValuesDwaveSampleSet(random_sample,
                #    title='Random sampling')
plotBarValues(df=df_random_sample,column_name='energy',sorted=True,skip=200,xlabel='Solution',ylabel='Energy',title='Random Sampling',save_fig=False,rot=0)


# %%
# Run default Dwave-neal simulated annealing implementation
sim_ann_sampler = dimod.SimulatedAnnealingSampler()
start = time.time()
sim_ann_sample_default = sim_ann_sampler.sample(model_random, num_reads=1000)
time_default = time.time() - start
df_sim_ann_sample_default = sim_ann_sample_default.to_pandas_dataframe(sample_column=True)
df_sim_ann_sample_default['runtime (um)'] = int(1e6*time_default/1000)
min_energy = df_sim_ann_sample_default['energy'].min()
print(min_energy)


# %%
# Generate plots from the default simulated annealing run
# ax_enum = plotEnergyValuesDwaveSampleSet(sim_ann_sample_default,
#                              title='Simulated annealing with default parameters')
# ax_enum.set(ylim=[min_energy*(0.99)**np.sign(min_energy),
#             min_energy*(1.1)**np.sign(min_energy)])
plotBarValues(
    df=df_sim_ann_sample_default,
    column_name='energy',
    sorted=True,
    skip=200,
    xlabel='Solution',
    ylabel='Energy',
    title='Simulated Annealing with default parameters',
    save_fig=False,
    rot=0,
    ylim=[min_energy*(0.99)**np.sign(min_energy),min_energy*(1.1)**np.sign(min_energy)],
    )
# plot_energy_cfd(sim_ann_sample_default,
#                 title='Simulated annealing with default parameters', skip=10)
plotBarCounts(
    df=df_sim_ann_sample_default,
    column_name='energy',
    sorted=True,
    normalized=True,
    skip=10,
    xlabel='Energy',
    title='Simulated Annealing with default parameters',
    save_fig=False,
    rot=0,
)




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

def loadMinEnergy(gs_file, instance_name):
    '''
    Loads the minimum energy of a given instance from file gs_energies.txt

    Args:
        gs_file: The file to load the energies from.
        instance_name: The name of the instance to load the energy for.

    Returns:
        The minimum energy of the instance.    

    '''
    energies = []
    with open(gs_file, "r") as fin:
        for line in fin:
            if(line.split()[0] == instance_name):
                energies.append(float(line.split()[1]))

    return min(energies)


# %%
# Compute results for instance 42 using D-Wave Neal
s = 0.99  # This is the success probability for the TTS calculation
treshold = 5.0  # This is a percentual treshold of what the minimum energy should be
# sweeps_list = [i for i in range(1, 250, 1)] + [
#     i for i in range(250, 1001, 10)]
# sweeps_list = [i for i in range(1, 21, 1)] + [
#     i for i in range(20, 501, 10)] + [
#     i for i in range(500, 1001, 20)]
sweeps_list = [i for i in range(1, 21, 5)] + [
    i for i in range(20, 501, 20)] + [
    i for i in range(500, 1001, 50)]
schedules_list = ['geometric', 'linear']
# schedules_list = ['geometric']
total_reads = 1000
default_sweeps = 1000
n_boot = 500
conf_int = 68  # Confidence interval for bootstrapping
default_boots = default_sweeps
boots_list = [1, 10, default_boots]
results_name = "results_" + str(instance) + ".pkl"
results_file = os.path.join(dneal_results_path, results_name)
results_dneal = {}
results_dneal['p'] = {}
results_dneal['min_energy'] = {}
results_dneal['random_energy'] = {}
results_dneal['tts'] = {}
results_dneal['ttsci'] = {}
results_dneal['t'] = {}
results_dneal['best'] = {}
results_dneal['bestci'] = {}
# If you wanto to use the raw data and process it here
if use_raw_data or not(os.path.exists(results_file)):
    # If you want to generate the data or load it here
    overwrite_pickles = False
    sim_ann_sampler = neal.SimulatedAnnealingSampler()
    random_energy, random_sample = randomEnergySampler(
        model_random, num_reads=total_reads, dwave_sampler=True)

    for boot in boots_list:
        results_dneal['p'][boot] = {}
        results_dneal['tts'][boot] = {}
        results_dneal['ttsci'][boot] = {}
        results_dneal['best'][boot] = {}
        results_dneal['bestci'][boot] = {}

    for schedule in schedules_list:
        probs = {k: [] for k in boots_list}
        time_to_sol = {k: [] for k in boots_list}
        prob_np = {k: [] for k in boots_list}
        ttscs = {k: [] for k in boots_list}
        times = []
        b = {k: [] for k in boots_list}
        bnp = {k: [] for k in boots_list}
        bcs = {k: [] for k in boots_list}
        for sweep in sweeps_list:
            # Gather instance names
            pickle_name = prefix + str(instance) + "_" + schedule + \
                "_" + str(sweep) + ".p"
            pickle_name = os.path.join(dneal_pickle_path, pickle_name)
            # If the instance data exists, load the data
            if os.path.exists(pickle_name) and not overwrite_pickles:
                # print(pickle_name)
                samples = pickle.load(open(pickle_name, "rb"))
                time_s = samples.info['timing']
            # If it does not exist, generate the data
            else:
                start = time.time()
                samples = sim_ann_sampler.sample(
                    model_random, num_reads=total_reads, num_sweeps=sweep, beta_schedule_type=schedule)
                time_s = time.time() - start
                samples.info['timing'] = time_s
                pickle.dump(samples, open(pickle_name, "wb"))
            # Compute statistics
            energies = samples.data_vectors['energy']
            occurrences = samples.data_vectors['num_occurrences']
            total_counts = sum(occurrences)
            times.append(time_s)
            min_energy = loadMinEnergy(
                results_path + "gs_energies.txt", prefix + str(instance))
            # success = min_energy*(1.0 + treshold/100.0)**np.sign(min_energy)
            success = random_energy - \
                (random_energy - min_energy)*(1.0 - treshold/100.0)

            # Best of boot samples es computed via n_boot bootstrappings
            boot_dist = {}
            pr_dist = {}
            cilo = {}
            ciup = {}
            pr = {}
            pr_cilo = {}
            pr_ciup = {}
            for boot in boots_list:
                boot_dist[boot] = []
                pr_dist[boot] = []
                for i in range(int(n_boot)):
                    resampler = np.random.randint(0, total_reads, boot)
                    sample_boot = energies.take(resampler, axis=0)
                    # Compute the best along that axis
                    boot_dist[boot].append(min(sample_boot))

                    occurences = occurrences.take(resampler, axis=0)
                    counts = {}
                    for index, energy in enumerate(sample_boot):
                        if energy in counts.keys():
                            counts[energy] += occurences[index]
                        else:
                            counts[energy] = occurences[index]
                    pr_dist[boot].append(
                        sum(counts[key] for key in counts.keys() if key < success)/boot)

                b[boot].append(np.mean(boot_dist[boot]))
                # Confidence intervals from bootstrapping the best out of boot
                bnp[boot] = np.array(boot_dist[boot])
                cilo[boot] = np.apply_along_axis(
                    stats.scoreatpercentile, 0, bnp[boot], 50.-conf_int/2.)
                ciup[boot] = np.apply_along_axis(
                    stats.scoreatpercentile, 0, bnp[boot], 50.+conf_int/2.)
                bcs[boot].append((cilo[boot], ciup[boot]))
                # Confidence intervals from bootstrapping the TTS of boot
                prob_np[boot] = np.array(pr_dist[boot])
                pr[boot] = np.mean(prob_np[boot])
                probs[boot].append(pr[boot])
                if prob_np[boot].all() == 0:
                    time_to_sol[boot].append(np.inf)
                    ttscs[boot].append((np.inf, np.inf))
                else:
                    pr_cilo[boot] = np.apply_along_axis(
                        stats.scoreatpercentile, 0, prob_np[boot], 50.-conf_int/2.)
                    pr_ciup[boot] = np.apply_along_axis(
                        stats.scoreatpercentile, 0, prob_np[boot], 50.+conf_int/2.)
                    time_to_sol[boot].append(
                        time_s*math.log10(1-s)/math.log10(1-pr[boot]+1e-9))
                    ttscs[boot].append((time_s*math.log10(1-s)/math.log10(
                        1-pr_cilo[boot]), time_s*math.log10(1-s)/math.log10(1-pr_ciup[boot]+1e-9)))

        results_dneal['t'][schedule] = times
        results_dneal['min_energy'][schedule] = min_energy
        results_dneal['random_energy'][schedule] = random_energy
        for boot in boots_list:
            results_dneal['p'][boot][schedule] = probs[boot]
            results_dneal['tts'][boot][schedule] = time_to_sol[boot]
            results_dneal['ttsci'][boot][schedule] = ttscs[boot]
            results_dneal['best'][boot][schedule] = [
                (random_energy - energy) / (random_energy - min_energy) for energy in b[boot]]
            results_dneal['bestci'][boot][schedule] = [tuple((random_energy - element) / (
                random_energy - min_energy) for element in energy) for energy in bcs[boot]]

    # Save results file in case that we are interested in reusing them
    pickle.dump(results_dneal, open(results_file, "wb"))
else:  # Just reload processed datafile
    results_dneal = pickle.load(open(results_file, "rb"))

# %%
# Transform current Dwave-neal nested dictionary into Pandas DataFrame

# Restructure dictionary to dictionary of tuple keys -> values
data_dict_2 = {(instance, schedule, sweep, boot):
               (results_dneal['t'][schedule][i],
                results_dneal['min_energy'][schedule],
                results_dneal['random_energy'][schedule],
                results_dneal['p'][boot][schedule][i],
                results_dneal['tts'][boot][schedule][i],
                results_dneal['ttsci'][boot][schedule][i],
                results_dneal['best'][boot][schedule][i],
                results_dneal['bestci'][boot][schedule][i])
               for i, sweep in enumerate(sweeps_list)
               for schedule in results_dneal['t'].keys()
               for boot in results_dneal['p'].keys()}

# Construct dataframe from dictionary
df_dneal = pd.DataFrame.from_dict(
    data_dict_2, orient='index').reset_index()

# Split column of tuples to multiple columns
df_dneal[['instance', 'schedule', 'sweeps', 'boots']
         ] = df_dneal['index'].apply(pd.Series)

# Clean up: remove unwanted columns, rename and sort
df_dneal = df_dneal.drop('index', 1).\
    rename(columns={0: 'mean_times', 1: 'min_energy', 2: 'random_energy', 3: 'success_rate', 4: 'tts', 5: 'tts_ci', 6: 'best', 7: 'best_ci'}).\
    sort_index(axis=1)


df_name = "df_results_" + str(instance) + ".pkl"
df_path = os.path.join(dneal_results_path, df_name)


df_dneal.to_pickle(df_path)
# df_dneal[df_dneal['schedule']=='geometric'].plot.scatter(x='sweeps',y='success_rate')

# %%
# Compute all instances with Dwave-neal

# instance_list = [0, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19]
# instance_list = [10,11,14,15,16,19,1,2,3,42,4,5,6,17]
instance_list = [i for i in range(20)] + [42]

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
            samples = pickle.load(open(file, "rb"))
            if min_energy > min(samples.data_vectors['energy']):
                min_energy = min(samples.data_vectors['energy'])
                print(file)
                print(min_energy)
                min_energies.append(min_energy)

        with open(os.path.join(results_path, "gs_energies.txt"), "a") as gs_file:
            gs_file.write(prefix + str(instance) + " " +
                          str(np.nanmin(min_energies)) + "  best_found dneal\n")

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

        min_energy = loadMinEnergy(gs_file=results_path + "gs_energies.txt",
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
# instance_list = [10,11,14,15,16,19,1,2,3,42,4,5,6]
# instance_list = list(range(20)) + [42]
overwrite_pickles = False
s = 0.99  # This is the success probability for the TTS calculation
treshold = 5.0  # This is a percentual treshold of what the minimum energy should be
# schedules = ['geometric', 'linear']
schedules = ['geometric']
total_reads = 1000
default_sweeps = 1000
# boots = [1, 10, 100, default_sweeps]
boots = [1, 10, default_boots]
all_results = {}

all_results_name = "all_results.pkl"
all_results_name = os.path.join(dneal_pickle_path, all_results_name)
# If you wanto to use the raw data and process it here
if use_raw_data or not(os.path.exists(all_results_name)):

    for instance in instance_list:
        all_results[instance] = {}
        all_results[instance]['p'] = {}
        all_results[instance]['min_energy'] = {}
        all_results[instance]['random_energy'] = {}
        all_results[instance]['tts'] = {}
        all_results[instance]['ttsci'] = {}
        all_results[instance]['t'] = {}
        all_results[instance]['best'] = {}
        all_results[instance]['bestci'] = {}

        # Fixing the random seed to get the same result
        np.random.seed(instance)
        J = np.random.rand(N, N)
        # We only consider upper triangular matrix ignoring the diagonal
        J = np.triu(J, 1)
        h = np.random.rand(N)
        model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

        # randomSample = randomSampler.sample(
        #     model_random, num_reads=total_reads)
        # random_energies = [datum.energy for datum in randomSample.data(
        #     ['energy'])]
        # random_energy = np.mean(random_energies)

        default_pickle_name = prefix + str(instance) + "_geometric_1000.p"
        default_pickle_name = os.path.join(
            dneal_pickle_path, default_pickle_name)
        if os.path.exists(default_pickle_name) and not overwrite_pickles:
            sim_ann_sample_default = pickle.load(open(default_pickle_name, "rb"))
            time_default = sim_ann_sample_default.info['timing']
        else:
            start = time.time()
            sim_ann_sample_default = sim_ann_sampler.sample(
                model_random, num_reads=1000)
            time_default = time.time() - start
            sim_ann_sample_default.info['timing'] = time_default
            pickle.dump(sim_ann_sample_default, open(default_pickle_name, "wb"))
        energies = [datum.energy for datum in sim_ann_sample_default.data(
            ['energy'], sorted_by='energy')]
        min_energy = energies[0]
        for schedule in schedules:

            # all_results[instance]['t'][schedule] = {}
            # all_results[instance]['min_energy'][schedule] = {}
            # all_results[instance]['random_energy'][schedule] = {}
            # all_results[instance]['p'][schedule] = {}
            # all_results[instance]['tts'][schedule] = {}
            # all_results[instance]['ttsci'][schedule] = {}
            # all_results[instance]['best'][schedule] = {}
            # all_results[instance]['bestci'][schedule] = {}

            # # probs = []
            # probs = {k: [] for k in boots}
            # time_to_sol = {k: [] for k in boots}
            # prob_np = {k: [] for k in boots}
            # ttscs = {k: [] for k in boots}
            # times = []
            # b = {k: [] for k in boots}
            # bnp = {k: [] for k in boots}
            # bcs = {k: [] for k in boots}
            for sweep in sweeps_list:
                # Gather instance names
                pickle_name = prefix + str(instance) + "_" + \
                    schedule + "_" + str(sweep) + ".p"
                pickle_name = os.path.join(dneal_pickle_path, pickle_name)
                # If the instance data exists, load the data
                if os.path.exists(pickle_name) and not overwrite_pickles:
                    samples = pickle.load(open(pickle_name, "rb"))
                    time_s = samples.info['timing']
                # If it does not exist, generate the data
                else:
                    start = time.time()
                    samples = sim_ann_sampler.sample(
                        model_random, num_reads=1000, num_sweeps=sweep, beta_schedule_type=schedule)
                    time_s = time.time() - start
                    samples.info['timing'] = time_s
                    pickle.dump(samples, open(pickle_name, "wb"))
                # Compute statistics
            #     energies = samples.data_vectors['energy']
            #     occurrences = samples.data_vectors['num_occurrences']
            #     total_counts = sum(occurrences)
            #     times.append(time_s)
            #     if min(energies) < min_energy:
            #         min_energy = min(energies)
            #         # print("A better solution of " + str(min_energy) + " was found for sweep " + str(sweep))
            #     # success = min_energy*(1.0 + treshold/100.0)**np.sign(min_energy)
            #     success = random_energy - \
            #         (random_energy - min_energy)*(1.0 - treshold/100.0)

            #     # Best of boot samples es computed via n_boot bootstrapping
            #     ci = 68
            #     boot_dist = {}
            #     pr_dist = {}
            #     cilo = {}
            #     ciup = {}
            #     pr = {}
            #     pr_cilo = {}
            #     pr_ciup = {}
            #     for boot in boots:
            #         boot_dist[boot] = []
            #         pr_dist[boot] = []
            #         for i in range(int(n_boot)):
            #             resampler = np.random.randint(0, total_reads, boot)
            #             sample_boot = energies.take(resampler, axis=0)
            #             # Compute the best along that axis
            #             boot_dist[boot].append(min(sample_boot))

            #             occurences = occurrences.take(resampler, axis=0)
            #             counts = {}
            #             for index, energy in enumerate(sample_boot):
            #                 if energy in counts.keys():
            #                     counts[energy] += occurences[index]
            #                 else:
            #                     counts[energy] = occurences[index]
            #             pr_dist[boot].append(
            #                 sum(counts[key] for key in counts.keys() if key < success)/boot)
            #         prob_np[boot] = np.array(pr_dist[boot])
            #         pr[boot] = np.mean(prob_np[boot])
            #         probs[boot].append(pr[boot])

            #         b[boot].append(np.mean(boot_dist[boot]))
            #         # Confidence intervals from bootstrapping the best out of boot
            #         bnp[boot] = np.array(boot_dist[boot])
            #         cilo[boot] = np.apply_along_axis(
            #             stats.scoreatpercentile, 0, bnp[boot], 50.-ci/2.)
            #         ciup[boot] = np.apply_along_axis(
            #             stats.scoreatpercentile, 0, bnp[boot], 50.+ci/2.)
            #         bcs[boot].append((cilo[boot], ciup[boot]))
            #         # Confidence intervals from bootstrapping the TTS of boot
            #         if prob_np[boot].all() == 0:
            #             time_to_sol[boot].append(np.inf)
            #             ttscs[boot].append((np.inf, np.inf))
            #         else:
            #             pr_cilo[boot] = np.apply_along_axis(
            #                 stats.scoreatpercentile, 0, prob_np[boot], 50.-ci/2.)
            #             pr_ciup[boot] = np.apply_along_axis(
            #                 stats.scoreatpercentile, 0, prob_np[boot], 50.+ci/2.)
            #             time_to_sol[boot].append(
            #                 time_s*math.log10(1-s)/math.log10(1-pr[boot]+1e-9))
            #             ttscs[boot].append((time_s*math.log10(1-s)/math.log10(
            #                 1-pr_cilo[boot]+1e-9), time_s*math.log10(1-s)/math.log10(1-pr_ciup[boot]+1e-9)))

            # all_results[instance]['t'][schedule][default_boots] = times
            # all_results[instance]['min_energy'][schedule][default_boots] = min_energy
            # all_results[instance]['random_energy'][schedule][default_boots] = random_energy
            # for boot in boots:
            #     all_results[instance]['p'][schedule][boot] = probs[boot]
            #     all_results[instance]['tts'][schedule][boot] = time_to_sol[boot]
            #     all_results[instance]['ttsci'][schedule][boot] = ttscs[boot]
            #     all_results[instance]['best'][schedule][boot] = [
            #         (random_energy - energy) / (random_energy - min_energy) for energy in b[boot]]
            #     all_results[instance]['bestci'][schedule][boot] = [tuple(
            #         (random_energy - element) / (random_energy - min_energy) for element in energy) for energy in bcs[boot]]

    # Save results file in case that we are interested in reusing them
    # pickle.dump(all_results, open(all_results_name, "wb"))
# else:  # Just reload processed datafile
    # all_results = pickle.load(open(all_results_name, "rb"))


# %%
