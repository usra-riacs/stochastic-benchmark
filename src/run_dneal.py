# %%
from tqdm.auto import tqdm
from scipy import sparse
import time
import pandas as pd
import numpy as np
import os
import datetime
import sys
import itertools
import pickle
import dimod
import neal

# jobid = int(os.getenv('PBS_ARRAY_INDEX'))
jobid = 42


# Input Parameters
total_reads = 1000
# Using 'float64' is about ~20% slower, can be neccessary
overwrite_pickles = False
# instances_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/instances"
# pickles_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/pysa/pickles"
instances_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/instances"
pickles_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/dneal/pickles_try"

schedules = []
sweeps = []
replicas = []
sizes = []
instances = []
Tcfactors = []
Thfactors = []

# Jobid instances 
# Fixed sweep, schedule, Tcfactor, Thfactors
# Input parameter size
schedules = ['geometric']
sweeps = [1000]
Tcfactors = [0]
Thfactors = [-1.5]
# sizes.append(int(str(sys.argv[1])))
sizes = [100]
instances.append(int(jobid))

# Single instance sweep study series of replicas, parameters size, alpha, and instance
# Jobid instance 
# Fixed schedule, Tcfactor, Thfactors
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
# schedules = ['geometric']
# instances.append(int(jobid))
# Tcfactors = [1]
# Thfactors = list(np.linspace(-3, 1, num=33, endpoint=True))


# %%
# Function to generate samples dataframes or load them otherwise



def createDnealSamplesDataframe(
    size: int = 100,
    instance: int = 42,
    parameters: dict = None,
    total_reads: int = 1000,
    sampler = None,
    prefix: str = "",
    instances_path: str = "",
    pickles_path: str = None,
    save_pickles: bool = True,
    use_raw_sample_pickles: bool = False,
    overwrite_pickles: bool = False,
) -> pd.DataFrame:
    '''
    Creates a dataframe with the samples for the dneal algorithm.

    Args:
        size: Size of the problem instance.
        instance: The instance to load/create the samples for.
        parameters: The parameters to use for the dneal algorithm.
        schedule: The schedule to use for the dneal algorithm.
        sweep: The number of sweeps to use for the dneal algorithm.
        total_reads: The total number of reads to use for the dneal algorithm.
        sampler: The sampler to use for the simulated annealing algorithm.
        prefix: The prefix to use for the pickle files.
        instances_path: The path to the instances.
        pickle_path: The path to the pickle files.
        save_pickles: Whether to save the pickles or not.
        use_raw_sample_pickles: Whether to use the raw sample pickles or not.
        overwrite_pickles: Whether to overwrite the pickles or not.
        

    Returns:
        The dataframe with the samples for the dneal algorithm.
    '''


    if parameters is None:
        parameters = {
            'schedule': 'geometric',
            'sweep': 100,
            'Tcfactor': 0.0,
            'Thfactor': 0.0,
        }

    # Gather instance names
    # TODO: We need to adress renaming problems, one proposal is to be very judicious about the keys order in parameters and be consistent with naming, another idea is sorting them alphabetically before joining them
    instance_name = prefix + str(instance)
    dict_samples_name = instance_name + "_" + \
        '_'.join(str(keys) + '_' + str(vals)
                 for keys, vals in parameters.items()) + ".p"
    df_samples_name = instance_name + "_" + \
        '_'.join(str(keys) + '_' + str(vals)
                 for keys, vals in parameters.items()) + ".pkl"
    dict_path = os.path.join(pickles_path, dict_samples_name)
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
        if os.path.exists(dict_path) and use_raw_sample_pickles:
            with open(dict_path, 'rb') as f:
                samples = pickle.load(f)
        else:
            file_path = os.path.join(instances_path, instance_name + ".txt")

            data = np.loadtxt(file_path, dtype=float)
            M = sparse.coo_matrix(
                (data[:, 2], (data[:, 0], data[:, 1])), shape=(size, size))
            J = M.A
            problem = M.A
            problem = problem+problem.T-np.diag(np.diag(problem))
            h = np.diagonal(M.A)
            np.fill_diagonal(J,0)

            model_random = dimod.BinaryQuadraticModel.from_ising(
                h, J, offset=0.0)
            min_temp = np.power(10,parameters['Tcfactor']) * np.min(np.abs(problem[np.nonzero(problem)])
                    ) / np.log(100/1)
            min_temp_cal = np.power(10,parameters['Tcfactor']) * min(sum(abs(i)
                                    for i in problem)) / np.log(100/1)
            max_temp = np.power(10,parameters['Thfactor']) * max(sum(abs(i)
                                for i in problem)) / np.log(100/50)
            if sampler is None:
                sampler = neal.SimulatedAnnealingSampler()
            start = time.time()
            samples = sampler.sample(
                model_random,
                num_reads=total_reads,
                num_sweeps=parameters['sweep'],
                beta_schedule_type=parameters['schedule'],
                beta_range=(1/max_temp, 1/min_temp),
                initial_states_generator="random"
            )
            time_s = time.time() - start
            samples.info['timing'] = time_s
            if save_pickles:
                with open(dict_path, 'wb') as f:
                    pickle.dump(samples, f)
        # Generate Dataframes
        df_samples = samples.to_pandas_dataframe(sample_column=True)
        df_samples.drop(columns=['sample'], inplace=True)
        df_samples['runtime (us)'] = int(
            1e6*samples.info['timing']/len(df_samples.index))
        if save_pickles:
            df_samples.to_pickle(df_path)

    return df_samples

# %%
# Main execution

# Define parameters dict
default_parameters = {
    'schedule': 'geometric',
    'sweep': [1000],
    'replica': [1],
    'Tcfactor': [0.0],
    'Thfactor': [0.0],
}

parameters_dict = {
    'schedule': schedules,
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
sampler = neal.SimulatedAnnealingSampler()
counter = 0
for size in sizes:
    prefix = "random_n_"+str(size)+"_inst_"
    for instance in instances:
        for parameter_set in parameter_sets:
            counter += 1
            print("run "+str(counter)+" of "+str(total_runs))
            parameters = dict(zip(parameters, parameter_set))
            df_samples = createDnealSamplesDataframe(
                size=size,
                instance=instance,
                parameters=parameters,
                total_reads=total_reads,
                prefix = prefix,
                pickles_path=pickles_path,
                instances_path=instances_path,
                sampler=sampler,
                save_pickles=True,
                use_raw_sample_pickles=False,
                overwrite_pickles=overwrite_pickles,
            )

# %%
print(datetime.datetime.now())