# %%
from tqdm.auto import tqdm
from scipy import sparse
from pysa.sa import Solver
import pandas as pd
import numpy as np
import os
import datetime
import sys
import itertools
import pickle

# jobid = int(os.getenv('PBS_ARRAY_INDEX'))
jobid = 42


# Input Parameters
total_reads = 1000
float_type = 'float32'
# Using 'float64' is about ~20% slower, can be neccessary
overwrite_pickles = False
instances_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/instances"
pickles_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/pysa/pickles"
# instances_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/instances"
# pickles_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/pysa/pickles_try"

sweeps = []
replicas = []
sizes = []
instances = []
Tcfactors = []
Thfactors = []

# Jobid instances 
# Fixed sweep, replicas, Tcfactor, Thfactors
# Input parameter size
sweeps = [1000]
replicas = [8]
Tcfactors = [0]
Thfactors = [-1.5]
# sizes.append(int(str(sys.argv[1])))
sizes = [100]
instances.append(int(jobid))

# Single instance sweep study series of replicas, parameters size, alpha, and instance
# Jobid instance 
# Fixed replicas, Tcfactor, Thfactors
# Input parameter size
# all_sweeps = [1] + [i for i in range(2, 21, 2)] + [
#                 i for i in range(20, 51, 5)] + [
#                 i for i in range(50, 101, 10)] + [
#                 i for i in range(100, 201, 20)] + [
#                 i for i in range(200, 501, 50)] + [
#                 i for i in range(500, 1001, 100)]# + [
#                 i for i in range(1000, 2001, 200)] + [
#                 i for i in range(2000, 5001, 500)] + [
#                 i for i in range(5000, 10001, 1000)]
# # sweep_idx = jobid % len(sweeps)
# # sweeps.append(all_sweeps[sweep_idx])
# sweeps = all_sweeps
# sizes.append(int(str(sys.argv[1])))
# # replicas = [2**i for i in range(0,8)]
# replicas.append(int(str(sys.argv[2])))
# instances.append(int(jobid))
# Tcfactors = [1]
# Thfactors = list(np.linspace(-3, 1, num=33, endpoint=True))


# %%
# Function to generate samples dataframes or load them otherwise


def createPySASamplesDataframe(
    size: int = 100,
    instance: int = 42,
    parameters: dict = None,
    total_reads: int = 1000,
    float_type: str = "float32",
    prefix: str = "",
    instances_path: str = "",
    pickles_path: str = None,
    save_pickles: bool = True,
    overwrite_pickles: bool = False,
) -> pd.DataFrame:
    '''
    Creates a dataframe with the samples for the pysa algorithm.
    Here we assume that the parameters are:
    - sweep: int Sweeps to be performed
    - replicas: int Number of replicas
    - Tcfactor: float Cold temperature power of 10 factor from Dwave deviation, e.g., 10**Tcfactor*min_delta_energy/log(100/1)
    - Thfactor: float Hot temperature power of 10 factor from Dwave deviation, e.g., 10**Thfactor*max_delta_energy/log(100/50)

    Args:
        instance: The instance to load/create the samples for.
        parameters: The parameters to use for PySA.
        total_reads: The total number of reads to use in PySA.
        float_type: The float type to use in PySA.
        prefix: The prefix to use for the samples.
        instance_path: The path to the instance files.
        pickles_path: The path to the pickle files.
        save_pickles: Whether to save the pickles or not.
        overwrite_pickles: Whether to overwrite the pickles or not.

    Returns:
        The dataframe with the samples for the pysa algorithm.
    '''
    # TODO This can be further generalized to use arbitrary parameter dictionaries

    if parameters is None:
        parameters = {
            'sweep': 100,
            'replica': 1,
            'Tcfactor': 0.0,
            'Thfactor': 0.0,
        }

    # Gather instance names
    # TODO: We need to adress renaming problems, one proposal is to be very judicious about the keys order in parameters and be consistent with naming, another idea is sorting them alphabetically before joining them
    instance_name = prefix + str(instance)
    df_samples_name = instance_name + "_" + \
        '_'.join(str(keys) + '_' + str(vals)
                 for keys, vals in parameters.items()) + ".pkl"
    df_path = os.path.join(pickles_path, df_samples_name)
    if os.path.exists(df_path) and not overwrite_pickles:
        try:
            df_samples = pd.read_pickle(df_path)
        except (pickle.UnpicklingError, EOFError):
            print('Pickle file ' + df_path +
                  ' is corrupted. We will create a new one.')
            os.replace(df_path, df_path + '.bak')
            # TODO: How to jump to other branch of conditional?
    else:
        file_path = os.path.join(instances_path, instance_name + ".txt")

        data = np.loadtxt(file_path, dtype=float)
        M = sparse.coo_matrix(
            (data[:, 2], (data[:, 0], data[:, 1])), shape=(size, size))
        problem = M.A
        problem = problem+problem.T-np.diag(np.diag(problem))

        # Get solver
        solver = Solver(
            problem=problem,
            problem_type='ising',
            float_type=float_type,
        )

        min_temp = np.power(10,parameters['Tcfactor']) * np.min(np.abs(problem[np.nonzero(problem)])
                   ) / np.log(100/1)
        min_temp_cal = np.power(10,parameters['Tcfactor']) * min(sum(abs(i)
                                 for i in problem)) / np.log(100/1)
        max_temp = np.power(10,parameters['Thfactor']) * max(sum(abs(i)
                             for i in problem)) / np.log(100/50)

        df_samples = solver.metropolis_update(
            num_sweeps=parameters['sweep'],
            num_reads=total_reads,
            num_replicas=parameters['replica'],
            update_strategy='random',
            min_temp=min_temp,
            max_temp=max_temp,
            initialize_strategy='random',
            recompute_energy=True,
            sort_output_temps=True,
            parallel=True,  # True by default
            use_pt=True,
            verbose=True,
        )
        if save_pickles:
            df_samples.drop(columns=['states','energies','num_sweeps','problem_type','float_type','temps','best_state','init_time (us)'], inplace=True)
            df_samples.to_pickle(df_path)

    return df_samples



# %%
# Main execution

# Define parameters dict
default_parameters = {
    'sweep': [1000],
    'replica': [1],
    'Tcfactor': [0.0],
    'Thfactor': [0.0],
}

parameters_dict = {
    'sweep': sweeps,
    'replica': replicas,
    'Tcfactor': Tcfactors,
    'Thfactor': Thfactors,
}
# Add default parametes in case they are missing and remove repeated elements in the parameters_dict values (sets)
if parameters_dict is not None:
    for i, j in parameters_dict.items():
        if len(j) == 0:
            parameters_dict[i] = default_parameters[i]
        else:
            parameters_dict[i] = set(j)
# Create list of parameters
parameters = list(parameters_dict.keys())

parameter_sets = itertools.product(
                *(parameters_dict[Name] for Name in parameters_dict))
parameter_sets = list(parameter_sets)
total_runs = np.product([len(i) for i in parameters_dict.values()])
counter = 0
for size in sizes:
    prefix = "random_n_"+str(size)+"_inst_"
    for instance in instances:
        for parameter_set in parameter_sets:
            counter += 1
            print("run "+str(counter)+" of "+str(total_runs))
            parameters = dict(zip(parameters, parameter_set))
            df_samples = createPySASamplesDataframe(
                size=size,
                instance=instance,
                parameters=parameters,
                total_reads=total_reads,
                float_type=float_type,
                prefix = prefix,
                pickles_path=pickles_path,
                instances_path=instances_path,
                save_pickles=True,
                overwrite_pickles=overwrite_pickles,
            )

# %%
print(datetime.datetime.now())