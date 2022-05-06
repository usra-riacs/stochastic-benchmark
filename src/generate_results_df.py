# %%
import pandas as pd
import numpy as np
import os
import datetime
import sys
import itertools
import pickle
import os
import functools
from typing import List, Tuple, Union


# %%
# jobid = int(os.getenv('PBS_ARRAY_INDEX'))
# jobid = 42


# Input Parameters
total_reads = 1000
overwrite_pickles = False
if int(str(sys.argv[2])) == 1:
    ocean_df_flag = True
else:
    ocean_df_flag = False
compute_best_found_flag = False
use_raw_dataframes = True
# instances_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/instances"
# data_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/"
# if ocean_df_flag:
#     pickles_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/dneal/pickles"
#     results_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/dneal"
# else:
#     pickles_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/pysa/pickles"
#     results_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/pysa"


instances_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/instances"
data_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/"
if ocean_df_flag:
    pickles_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/dneal/pickles"
    results_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/dneal/"
else:
    pickles_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/pysa/pickles"
    results_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/pysa/"

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
replicas = [8]
Tcfactors = [0]
Thfactors = [-1.5]
# sizes.append(int(str(sys.argv[1])))
# sizes = [100]
# instances.append(int(jobid))


all_sweeps = [1] + [i for i in range(2, 21, 2)] + [
    i for i in range(20, 51, 5)] + [
    i for i in range(50, 101, 10)] + [
    i for i in range(100, 201, 20)] + [
    i for i in range(200, 501, 50)] + [
    i for i in range(500, 1001, 100)]  # + [
# i for i in range(1000, 2001, 200)] + [
# i for i in range(2000, 5001, 500)] + [
# i for i in range(5000, 10001, 1000)]
# sweep_idx = jobid % len(sweeps)
# sweeps.append(all_sweeps[sweep_idx])
sweeps = all_sweeps
sizes.append(int(str(sys.argv[1])))
# sizes = [100,200]
replicas = [2**i for i in range(0, 4)]
# replicas.append(int(str(sys.argv[2])))
# instances.append(int(jobid))
instances = [i for i in range(0,20)] + [42]
Tcfactors = [0.0]
Thfactors = list(np.linspace(-3, 1, num=33, endpoint=True))
# Thfactors = [0.0]


EPSILON = 1e-10
confidence_level = 68
s = 0.99
gap = 1.0

bootstraps = [int(i*10**j) for j in [0, 1, 2]
              for i in [1, 1.5, 2, 3, 5, 7]] + [int(1e3)]

bootstrap_iterations = 1000
fail_value = np.inf

# %%

# Functions to retrieve instances files
# Define functions to extract data files

# TODO there is a way to generalize this using a single function and the parameters dict, searching for the dict key in the name and parsing the value after it, generalizing it to any parameter.


def getInstance(filename, prefix):
    '''
    Extracts the instance from the Dwave-neal experiment filename assuming the filename follows the naming convention prefix_instance_....extension

    Args:
        filename: the name of the file
        prefix: the prefix of the files

    Returns:
        sweep: the sweep string
    '''
    return int(filename.rsplit(".", 1)[0].split(prefix, 1)[1].split("_")[0])


def createInstanceFileList(directory, instances):
    '''
    Creates a list of files in the directory for the instances in the list

    Args:
        directory: the directory where the files are
        instances: the list of instances

    Returns:
        instance_files: the list of files
    '''
    files = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and not f.endswith('.zip') and not f.endswith('.sh'))]
        # exclude best_found.txt files
        files = [f for f in files
                 if(not f.startswith('best_found'))]
        # Below, select only specifed n,s,alpha instances
        files = [f for f in files if(getInstance(f) in instances)]
        for f in files:
            files.append(root+"/"+f)
    return files


def getSweepPySAExperiment(filename):
    '''
    Extracts the sweeps from the PySA experiment filename assuming the filename follows the naming convention 
    prefix_instance_sweep_sweep_replica_replica_Tcfactor_Tcfactor_Thfactor_Thfactor.extension

    Args:
        filename: the name of the file

    Returns:
        sweeps: the number of sweeps
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 7)[-7])


def getThfactorExperiment(filename):  # Thfactor
    '''
    Extracts the hot temperature transition probability from the PySA experiment filename assuming the filename follows the naming convention 
    prefix_instance_sweep_sweep_replica_replica_Tcfactor_Tcfactor_Thfactor_Thfactor.extension

    Args:
        filename: the name of the file

    Returns:
        Thfactor: the hot temperature transition probability
    '''
    return float(filename.rsplit(".", 1)[0].rsplit("_", 2)[-1])


def getTcfactorExperiment(filename):  # Tcfactor
    '''
    Extracts the cold temperature transition probability from the PySA experiment filename assuming the filename follows the naming convention 
    prefix_instance_sweep_sweep_replica_replica_Tcfactor_Tcfactor_Thfactor_Thfactor.extension

    Args:
        filename: the name of the file

    Returns:
        Tcfactor: the cold temperature transition probability
    '''
    return float(filename.rsplit(".", 1)[0].rsplit("_", 3)[-3])


def getReplicaPySAExperiment(filename):  # replicas
    '''
    Extracts the replicas from the PySA experiment filename assuming the filename follows the naming convention 
    prefix_instance_sweep_sweep_replica_replica_Tcfactor_Tcfactor_Thfactor_Thfactor.extension

    Args:
        filename: the name of the file

    Returns:
        replicas: the number of replicas
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 5)[-5])


def createPySAExperimentFileList(
    directory: str,
    instances: Union[List[str], List[int]],
    replicas: Union[List[str], List[int]] = None,
    sweeps: Union[List[str], List[int]] = None,
    Tcfactors: Union[List[str], List[float]] = None,
    Thfactors: Union[List[str], List[float]] = None,
    prefix: str = "",
) -> list:
    '''
    Creates a list of experiment files in the directory for the instance in  instances, replica in replicas, sweep in sweeps, Tcfactor in Tcfactors, and Thfactor in Thfactors

    Args:
        directory: the directory where the files are
        instances: the list of instances
        replicas: the list of replicas
        sweeps: the list of sweeps
        Tcfactors: the list of Tcfactor
        Thfactors: the list of Thfactor
        prefix: the prefix of the files

    Returns:
        experiment_files: the list of files

    '''
    files_list = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and
                    not f.endswith('.zip') and
                    not f.endswith('.sh') and
                    not f.endswith('.p') and
                    f.startswith(prefix))]
        # exclude best_found.txt files
        files = [f for f in files
                 if(not f.startswith('best_found'))]
        # Below, select only specifed instances
        files = [f for f in files if(
            getInstance(f, prefix) in instances)]
        # Consider replicas if provided list
        if replicas is not None:
            files = [f for f in files if(
                getReplicaPySAExperiment(f) in replicas)]
        # Consider sweeps if provided list
        if sweeps is not None:
            files = [f for f in files if(
                getSweepPySAExperiment(f) in sweeps)]
        # Consider Tcfactor if provided list
        if Tcfactors is not None:
            files = [f for f in files if(
                getTcfactorExperiment(f) in Tcfactors)]
        # Consider Thfactor if provided list
        if Thfactors is not None:
            files = [f for f in files if(
                getThfactorExperiment(f) in Thfactors)]
        for f in files:
            files_list.append(root+"/"+f)

        # sort files_list by instance
        files_list = sorted(files_list, key=lambda x: getInstance(x, prefix))
    return files_list


def getScheduleDnealExperiment(filename):
    '''
    Extracts the schedule from the Dwave-neal experiment filename assuming the filename follows the naming convention 
    prefix_instance_schedule_schedule_sweep_sweep_Tcfactor_Tcfactor_Thfactor_Thfactor.extension

    Args:
        filename: the name of the file
        prefix: the prefix of the files

    Returns:
        schedule: the schedule string
    '''
    return filename.rsplit(".", 1)[0].rsplit("_", 8)[-7]


def getSweepDnealExperiment(filename):
    '''
    Extracts the sweeps from the Dwave-neal experiment filename assuming the filename follows the naming convention prefix_instance_schedule_sweeps.extension

    Args:
        filename: the name of the file
        prefix: the prefix of the file

    Returns:
        sweep: the schedule string
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 5)[-5])


def createDnealExperimentFileList(
    directory: str,
    instances: Union[List[str], List[int]],
    sweeps: Union[List[str], List[int]] = None,
    schedules: List[str] = None,
    Tcfactors: Union[List[str], List[float]] = None,
    Thfactors: Union[List[str], List[float]] = None,
    prefix: str = "",
    suffix: str = "",
) -> list:
    '''
    Creates a list of experiment files in the directory for the instances in the instances, sweeps in the sweeps, and schedules in the schedules

    Args:
        directory: the directory where the files are
        instances: the list of instances
        sweeps: the list of sweeps
        schedules: the list of schedules
        prefix: the prefix of the experiment files

    Returns:
        experiment_files: the list of files

    '''
    files_list = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and
                    not f.endswith('.zip') and
                    not f.endswith('.sh') and
                    f.endswith(suffix) and
                    f.startswith(prefix))]
        # exclude best_found.txt files
        files = [f for f in files
                 if(not f.startswith('best_found'))]
        # Below, select only specifed instances
        files = [f for f in files if(
            getInstance(f, prefix) in instances)]
        # Consider sweeps if provided list
        if sweeps is not None:
            files = [f for f in files if(
                getSweepDnealExperiment(f) in sweeps)]
        # Consider schedules if provided list
        if schedules is not None:
            files = [f for f in files if(
                getScheduleDnealExperiment(f) in schedules)]
        # Consider Tcfactor if provided list
        if Tcfactors is not None:
            files = [f for f in files if(
                getTcfactorExperiment(f) in Tcfactors)]
        # Consider Thfactor if provided list
        if Thfactors is not None:
            files = [f for f in files if(
                getThfactorExperiment(f) in Thfactors)]
        for f in files:
            files_list.append(root+"/"+f)

        # sort files_list by instance
        files_list = sorted(
            files_list, key=lambda x: getInstance(x, prefix))
    return files_list

# %%
# Function to create ground-state file


def createBestFound(
    instances: Union[List[str], List[int]],
    pickles_path: str,
    results_path: str,
    response_column: str = None,
    response_direction: int = -1,
    prefix: str = "",
    ocean_df_flag: bool = True,
):
    if ocean_df_flag:
        response_column = 'energy'
    else:  # Assuming PySA output dataframe
        response_column = 'best_energy'
    for instance in instances:
        # List all the pickled filed for an instance files
        if ocean_df_flag:
            files = createDnealExperimentFileList(
                directory=pickles_path,
                instances=[instance],
                prefix=prefix,
                suffix='.pkl',
            )

        else:
            files = createPySAExperimentFileList(
                directory=pickles_path,
                instances=[instance],
                prefix=prefix,
            )
        best_responses = []
        for file in files:
            df = pd.read_pickle(file)
            if response_direction == -1:  # Minimization
                best_responses.append(df[response_column].min())
            else:  # Maximization
                best_responses.append(df[response_column].max())

        if len(best_responses) == 0:
            print('No files found for instance {}'.format(instance))
        else:
            with open(os.path.join(results_path, "best_found.txt"), "a") as gs_file:
                if response_direction == -1:
                    best_response = np.nanmin(best_responses)
                else:
                    best_response = np.nanmax(best_responses)
                if ocean_df_flag:
                    gs_file.write(prefix + str(instance) + " " +
                                str(best_response) + " " + "best_found dneal\n")
                else:
                    gs_file.write(prefix + str(instance) + " " +
                                str(best_response) + " " + "best_found pysa\n")


# %%
# Function to load ground state solutions from solution file best_found.txt


def loadResponseFromFile(data_file, instance_name):
    '''
    Loads the reponse (energy) of a given instance 
    from file with rows having the instances

    Args:
        data_file: The file to load the responses (energies) from.
        instance_name: The name of the instance to load the response (energy) for.

    Returns:
        The response (energy) of the instance.

    '''
    energies = []
    with open(data_file, "r") as fin:
        for line in fin:
            if(line.split()[0] == instance_name):
                energies.append(float(line.split()[1]))

    if len(energies) == 0:
        print("No response found for instance: " + instance_name)
        return None
    return min(energies)


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
        # Consider continuous RTT and RTT scaled by assuming success_probability=1 as success_probability=1/size*(1-1/10)
        return scale*np.log(1.0 - s) / np.log(1 - (1 - 1/10)/size)
    else:
        return scale*np.log(1.0 - s) / np.log(1 - success_probability)


computeRTT_vectorized = np.vectorize(computeRTT, excluded=(1, 2, 3, 4))

# %%
# Function to retrieve results from Dataframes


def computeResultsList(
    df: pd.DataFrame,
    random_value: float = 0.0,
    response_column: str = None,
    response_direction: int = -1,
    best_value: float = None,
    success_metric: str = 'perf_ratio',
    resource_column: str = None,
    downsample: int = 10,
    bootstrap_iterations: int = 1000,
    confidence_level: float = 68,
    gap: float = 1.0,
    s: float = 0.99,
    fail_value: float = np.inf,
    overwrite_pickles: bool = False,
    ocean_df_flag: bool = True,
) -> list:
    '''
    Compute a list of the results computed for analysis given a dataframe from a solver.

    Args:
        df: The dataframe contaning results.
        random_value: The mean response (energy) of the random sample.
        response_column: The column name of the response (energy) of the sample.
        response_direction: The direction of the best response (minimum energy) of the sample.
        best_value: The best value of the response (energy) of the sample.
        downsample: The downsampling sample for bootstrapping.
        bootstrap_iterations: The number of bootstrap samples.
        confidence_level: The confidence level for the bootstrap.
        gap: The threshold for the considering a read successful w.r.t the performance ratio [%].
        s: The success factor (usually said as RTT within s% probability).
        ocean_df_flag: If True, the dataframe is from ocean sdk.

    Returns:
        A list of the results computed for analysis. Organized as follows
        [
            number of downsamples,
            bootstrapped mean best response (minimum energy),
            bootstrapped mean best response (minimum energy) confidence interval lower bound,
            bootstrapped mean best response (minimum energy) confidence interval upper bound,
            bootstrapped performance ratio,
            bootstrapped performance ratio confidence interval lower bound,
            bootstrapped performance ratio confidence interval upper bound,
            bootstrapped success probability,
            bootstrapped success probability confidence interval lower bound,
            bootstrapped success probability confidence interval upper bound,
            bootstrapped resource to target,
            bootstrapped resource to target confidence interval lower bound,
            bootstrapped resource to target confidence interval upper bound,
            bootstrapped mean runtime,
            bootstrapped mean runtime confidence interval lower bound,
            bootstrapped mean runtime confidence interval upper bound,
            bootstrapped inverse performance ratio,
            bootstrapped inverse performance ratio confidence interval lower bound,
            bootstrapped inverse performance ratio confidence interval upper bound,
        ]

    TODO: Here we assume the succes metric is the performance ratio, we can generalize that as any function of the parameters (use external function)
    TODO: Here we only return a few parameters with confidence intervals w.r.t. the bootstrapping. We can generalize that as any possible outcome (use function)
    '''

    aggregated_df_flag = False
    if response_column is None:
        if ocean_df_flag:
            response_column = 'energy'
        else:  # Assume it is a PySA dataframe
            response_column = 'best_energy'
    if resource_column is None:
        resource_column = 'runtime (us)'

    if best_value is None:
        if response_direction == - 1:  # Minimization
            best_value = df[response_column].min()
        else:  # Maximization
            best_value = df[response_column].max()

    if response_direction == - 1:  # Minimization
        success_val = random_value - \
            (1.0 - gap/100.0)*(random_value - best_value)
    else:  # Maximization
        success_val = (1.0 - gap/100.0) * \
            (best_value - random_value) - random_value
        # TODO Here we only include relative performance ratio. Consider other objectives as in benchopt

    resamples = np.random.randint(0, len(df), size=(
        downsample, bootstrap_iterations), dtype=np.intp)

    responses = df[response_column].values
    times = df[resource_column].values
    rtt_factor = 1e-6*df[resource_column].sum()
    # TODO Change this to be general for PySA dataframes
    if 'num_occurrences' in df.columns and not np.all(df['num_occurrences'].values == 1):
        print('The input dataframe is aggregated')
        # occurrences = df['num_occurrences'].values
        aggregated_df_flag = True

    # Compute the minimum value of each bootstrap samples and its corresponding confidence interval based on the resamples
    if response_direction == - 1:  # Minimization
        response_dist = np.apply_along_axis(
            func1d=np.min, axis=0, arr=responses[resamples])
    else:  # Maximization
        response_dist = np.apply_along_axis(
            func1d=np.max, axis=0, arr=responses[resamples])
        # TODO This could be generalized as the X best samples
    response = np.mean(response_dist)
    response_conf_interval_lower = np.nanpercentile(
        response_dist, 50-confidence_level/2)
    response_conf_interval_upper = np.nanpercentile(
        response_dist, 50+confidence_level/2)

    # Compute the resource (time) of each bootstrap samples and its corresponding confidence interval based on the resamples
    resource_dist = np.apply_along_axis(
        func1d=np.mean, axis=0, arr=times[resamples])
    mean_time = np.mean(resource_dist)
    mean_time_conf_interval_lower = np.nanpercentile(
        resource_dist, 50-confidence_level/2)
    mean_time_conf_interval_upper = np.nanpercentile(
        resource_dist, 50+confidence_level/2)

    # Compute the success metric (performance ratio) of each bootstrap samples and its corresponding confidence interval based on the resamples
    if success_metric == 'perf_ratio':
        perf_ratio = (random_value - response) / (random_value - best_value)
        perf_ratio_conf_interval_lower = (random_value - response_conf_interval_upper) / (
            random_value - best_value)
        perf_ratio_conf_interval_upper = (
            random_value - response_conf_interval_lower) / (random_value - best_value)
    else:
        print("Success metric not implemented yet")
        # TODO here the input could be a function

    # Compute the inverse success metric (performance ratio) of each bootstrap samples and its corresponding confidence interval based on the resamples
    if success_metric == 'perf_ratio':
        inv_perf_ratio = 1 - (random_value - response) / \
            (random_value - best_value) + EPSILON
        inv_perf_ratio_conf_interval_lower = 1 - (random_value - response_conf_interval_lower) / (
            random_value - best_value) + EPSILON
        inv_perf_ratio_conf_interval_upper = 1 - (
            random_value - response_conf_interval_upper) / (random_value - best_value) + EPSILON
    else:
        print("Success metric not implemented yet")
        # TODO here the input could be a function

    # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
    if aggregated_df_flag:
        print('Aggregated dataframe')
        return []
        # TODO: One can think about desaggregating the dataframe here. Maybe check Dwave's code for this.
    else:
        if response_direction == -1:  # Minimization

            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x < success_val)/downsample, axis=0, arr=responses[resamples])
        else:  # Maximization
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x > success_val)/downsample, axis=0, arr=responses[resamples])
    # Consider np.percentile instead to reduce package dependency. We need to benchmark and test alternative
    success_prob = np.mean(success_prob_dist)
    success_prob_conf_interval_lower = np.nanpercentile(
        success_prob_dist, 50-confidence_level/2)
    success_prob_conf_interval_upper = np.nanpercentile(
        success_prob_dist, 50+confidence_level/2)

    # Compute the resource to target (RTT) within certain threshold of each bootstrap samples and its corresponding confidence interval based on the resamples
    rtt_dist = computeRTT_vectorized(
        success_prob_dist, s=s, scale=rtt_factor, fail_value=fail_value)
    # Question: should we scale the RTT with the number of bootstrapping we do, intuition says we don't need to
    rtt = np.mean(rtt_dist)
    if np.isinf(rtt) or np.isnan(rtt) or rtt == fail_value:
        rtt_conf_interval_lower = fail_value
        rtt_conf_interval_upper = fail_value
    else:
        # rtt_conf_interval = computeRTT_vectorized(
        #     success_prob_conf_interval, s=0.99, scale=1e-6*df_default_samples['runtime (us)'].sum())
        rtt_conf_interval_lower = np.nanpercentile(
            rtt_dist, 50-confidence_level/2)
        rtt_conf_interval_upper = np.nanpercentile(
            rtt_dist, 50+confidence_level/2)
    # Question: How should we compute the confidence interval of the RTT? Should we compute the function on the confidence interval of the probability or compute the confidence interval over the RTT distribution?

    return [downsample,
            response, response_conf_interval_lower, response_conf_interval_upper,
            perf_ratio, perf_ratio_conf_interval_lower, perf_ratio_conf_interval_upper,
            success_prob, success_prob_conf_interval_lower, success_prob_conf_interval_upper,
            rtt, rtt_conf_interval_lower, rtt_conf_interval_upper,
            mean_time, mean_time_conf_interval_lower, mean_time_conf_interval_upper,
            inv_perf_ratio, inv_perf_ratio_conf_interval_lower, inv_perf_ratio_conf_interval_upper]


# %%
# Function to update the dataframes
# TODO Remove all the list_* variables and name them as plurals instead
# TODO: Prefix is assumed given directly to the file

def createResultsDataframes(
    df: pd.DataFrame = None,
    instances: List[int] = [0],
    parameters_dict: dict = None,
    bootstraps: List[int] = [1000],
    data_path: str = None,
    results_path: str = None,
    pickles_path: str = None,
    use_raw_dataframes: bool = False,
    confidence_level: float = 68,
    gap: float = 1.0,
    bootstrap_iterations: int = 1000,
    s: float = 0.99,
    fail_value: float = np.inf,
    save_pickle: bool = True,
    ocean_df_flag: bool = True,
) -> pd.DataFrame:
    '''
    Function to create the dataframes for the experiments

    Args:
        df: The dataframe to be updated
        instances: The instance number
        bootstraps: The number of bootstraps
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

    Returns:
        The results dataframe

    '''
    # Remove repeated elements in the parameters_dict values (sets)
    if parameters_dict is not None:
        for i, j in parameters_dict.items():
            parameters_dict[i] = set(j)
    # Create list of parameters
    parameter_names = list(parameters_dict.keys())
    # Sort it alphabetically (ignoring uppercase)
    # parameter_names.sort(key=str.lower)
    # Check that the parameters are columns in the dataframe
    if df is not None:
        assert all([i in df.columns for i in parameter_names])
        # In case that the data is already in the dataframe, return it
        if all([k in df[i].values for (i, j) in parameters_dict.items() for k in j]):
            print('The dataframe has some data for the parameters')
            # The parameters dictionary has lists as values as the loop below makes the concatenation faster than running the loop for each parameter
            cond = [df[k].isin(v).astype(bool)
                    for k, v in parameters_dict.items()]
            cond_total = functools.reduce(lambda x, y: x & y, cond)
            if all(bootstrap in df[cond_total]['boots'].values for bootstrap in bootstraps):
                print('The dataframe already has all the data')
                return df

    # Create filename
    if len(instances) > 1:
        df_name = prefix + 'df_results.pkl'
    else:
        df_name = prefix + str(instances[0]) + '_df_results.pkl'
    df_path = os.path.join(results_path, df_name)

    # If use_raw_dataframes compute the row
    if use_raw_dataframes or not os.path.exists(df_path):
        results = []
        for instance in instances:
            random_energy = loadResponseFromFile(os.path.join(
                data_path, 'random_energies.txt'), prefix + str(instance))
            best_value = loadResponseFromFile(os.path.join(
                data_path, 'best_found.txt'), prefix + str(instance))
            # We will assume that the insertion order in the keys is preserved (hence Python3.7+ only) and is sorted alphabetically
            parameter_sets = itertools.product(
                *(parameters_dict[Name] for Name in parameters_dict))
            parameter_sets = list(parameter_sets)
            total_runs = np.product([len(i) for i in parameters_dict.values()])
            counter = 0
            for parameter_set in parameter_sets:
                counter += 1
                instance_name = prefix + str(instance)
                parameters = dict(zip(parameter_names, parameter_set))
                df_samples_name = instance_name + "_" + \
                    '_'.join(str(keys) + '_' + str(vals)
                             for keys, vals in parameters.items()) + ".pkl"
                df_samples_path = os.path.join(pickles_path, df_samples_name)
                if os.path.exists(df_samples_path):
                    print(prefix,instance,": analyzing ",counter," of ",total_runs)
                    df_samples = pd.read_pickle(df_samples_path)
                else:
                    print('Missing file ' + df_samples_name)
                    continue
                inputs = [instance] + [i for i in parameter_set]

                for boots in bootstraps:

                    # TODO Good place to replace with mask and isin1d()
                    # This generated the undersampling using bootstrapping, filtering by all the parameters values
                    if (df is not None) and \
                            (boots in df.loc[(df[list(parameters)] == pd.Series(parameters)).all(axis=1)]['boots'].values):
                        continue
                    else:
                        # print("Generating results for instance:", instance,
                        #   "schedule:", schedule, "sweep:", sweep, "boots:", boots)
                        outputs = computeResultsList(
                            df=df_samples,
                            random_value=random_energy,
                            best_value=best_value,
                            downsample=boots,
                            bootstrap_iterations=bootstrap_iterations,
                            confidence_level=confidence_level,
                            gap=gap,
                            s=s,
                            fail_value=fail_value,
                            ocean_df_flag=ocean_df_flag,
                        )
                    results.append(
                        inputs + outputs)
                if counter % 100 == 0:
                    with open(os.path.join(results_path, df_name + '.pickle'), 'wb') as f:
                        pickle.dump(results, f)
        # TODO: Organize these column names to be created automatically from metric list
        df_results = pd.DataFrame(results,
                                  columns=['instance'] + parameter_names + 
                                  ['boots',
                                    'min_energy', 'min_energy_conf_interval_lower', 'min_energy_conf_interval_upper',
                                    'perf_ratio', 'perf_ratio_conf_interval_lower', 'perf_ratio_conf_interval_upper',
                                    'success_prob', 'success_prob_conf_interval_lower', 'success_prob_conf_interval_upper',
                                    'rtt', 'rtt_conf_interval_lower', 'rtt_conf_interval_upper',
                                    'mean_time', 'mean_time_conf_interval_lower', 'mean_time_conf_interval_upper',
                                    'inv_perf_ratio', 'inv_perf_ratio_conf_interval_lower', 'inv_perf_ratio_conf_interval_upper',
                                    ])
        if df is not None:
            df_new = pd.concat(
                [df, df_results], axis=0, ignore_index=True)
        else:
            df_new = df_results.copy()
        if save_pickle:
            # df_new = cleanup_df(df_new)
            df_new.to_pickle(df_path)

    else:
        print("Loading the dataframe")
        df_new = pd.read_pickle(df_path)
    return df_new

# %%
# List experiment files and determine if there are any missing from experiment


# Define parameters dict
default_parameters = {
    'sweep': [1000],
    'schedule': ['geometric'],
    'replica': [1],
    'Tcfactor': [0.0],
    'Thfactor': [0.0],
}

if ocean_df_flag:
    parameters_dict = {
        'schedule': schedules,
        'sweep': sweeps,
        'Tcfactor': Tcfactors,
        'Thfactor': Thfactors,
    }
else:
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
parameter_names = list(parameters_dict.keys())

parameter_sets = itertools.product(
    *(parameters_dict[Name] for Name in parameters_dict))
parameter_sets = list(parameter_sets)
complete_files = []

for size in sizes:
    prefix = "random_n_"+str(size)+"_inst_"
    parameters_in_files = []
    for instance in instances:
        if ocean_df_flag:
            files = createDnealExperimentFileList(
                directory=pickles_path,
                instances=[instance],
                sweeps=None,
                schedules=None,
                Tcfactors=None,
                Thfactors=None,
                prefix=prefix,
                suffix='.pkl',)
            for file in files:
                parameters_in_files.append((
                    getScheduleDnealExperiment(file),
                    getSweepDnealExperiment(file),
                    getTcfactorExperiment(file),
                    getThfactorExperiment(file),
                ))
        else:
            files = createPySAExperimentFileList(
                directory=pickles_path,
                instances=[instance],
                replicas=None,
                sweeps=None,
                Tcfactors=None,
                Thfactors=None,
                prefix=prefix,)
            for file in files:
                parameters_in_files.append((
                    getSweepPySAExperiment(file),
                    getReplicaPySAExperiment(file),
                    getTcfactorExperiment(file),
                    getThfactorExperiment(file),
                ))
        for parameter_set in parameter_sets:
            if parameter_set not in parameters_in_files:
                print("Missing parameter set:", parameter_set,
                      "for instance:", instance, "of size", size)
        complete_files.extend(files)

# %%
# Compute best found file
if compute_best_found_flag:
    for size in sizes:
        prefix = "random_n_"+str(size)+"_inst_"
        createBestFound(
            instances=instances,
            pickles_path=pickles_path,
            results_path=data_path,
            prefix=prefix,
            ocean_df_flag=ocean_df_flag,
        )

# %%
# Main execution
for size in sizes:
    prefix = "random_n_"+str(size)+"_inst_"
    for instance in instances:
        df_name = prefix + str(instance) + '_df_results.pkl'
        df_path = os.path.join(results_path, df_name)
        if os.path.exists(df_path) and not use_raw_dataframes:
            df_samples = pd.read_pickle(df_path)
        else:
            df_samples = None
        df_samples = createResultsDataframes(
            df=df_samples,
            instances=[instance],
            parameters_dict=parameters_dict,
            bootstraps=bootstraps,
            data_path=data_path,
            results_path=results_path,
            pickles_path=pickles_path,
            confidence_level=confidence_level,
            gap=gap,
            bootstrap_iterations=bootstrap_iterations,
            use_raw_dataframes=use_raw_dataframes,
            s=s,
            fail_value=fail_value,
            save_pickle=True,
            ocean_df_flag=ocean_df_flag,
        )

# %%
print(datetime.datetime.now())
