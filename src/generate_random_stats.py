# %%
import datetime
import functools
import itertools
import os
import pickle
import sys
from typing import List, Tuple, Union
import time
import numpy as np
import pandas as pd
from pysa.sa import Solver
from scipy import sparse
from tqdm.auto import tqdm

#%%
# Define instance parameters
instances = [i for i in range(1,51)]
# instances = [i for i in range(1,5)]
evaluate_instances = [i for i in range(41,51)]
# instances = [i for i in range(41,51)]
size = int(str(sys.argv[1]))
alpha = float(str(sys.argv[2]))
sizes = []
alphas = []

# sizes.append(int(str(sys.argv[1])))
# alphas.append(float(str(sys.argv[2])))
sizes = [size]
alphas = [alpha]

wishart_instances = True
# Input Parameters
total_reads = 200
overwrite_pickles = False
if 0 == 1:
    ocean_df_flag = True
else:
    ocean_df_flag = False
compute_best_found_flag = False
use_raw_dataframes = False

bootstrap_iterations = 1000


# jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
jobid = 0

schedules = []
sweeps = []
replicas = []
Tcfactors = []
Thfactors = []

sweeps = [0]
replicas = [1]

all_sweeps = [0]
sweeps = all_sweeps
# replicas = [2**i for i in range(0, 4)]
replicas = [1]
# instances = [i for i in range(0,20)] + [42]
pcolds = [1.00]
phots = [50.0]


EPSILON = 1e-10
confidence_level = 68
s = 0.99
gap = 1.0

# bootstraps = list(set([int(i*10**j) for j in [0, 1]
#               for i in [1, 1.5, 2, 3, 5, 7]] + [int(1e2),int(2e2)]
# bootstraps = list(set([int(7e5)]))
bootstraps = list(set([int(i*10**j) for j in [0, 1, 2, 3, 4, 5]
                for i in [1, 1.5, 2, 3, 5, 7]]))+ [int(1e6)]

# bootstrap_iterations = 1000
max_bootstraps = 1e6
fail_value = np.inf

# All instances with fixed sweeps and replicas, parameter size and alpha
#sweeps = [1000]
#replicas = [100]
#sizes.append(int(str(sys.argv[1])))
#alphas.append(float(str(sys.argv[2])))
#instances.append(int(jobid))

# Single instance sweep study series of replicas, parameters size, alpha, and instance
# %%
# alphas.append(float(str(sys.argv[2])))
# instances.append(int(str(sys.argv[3])))



def getAlpha(filename):
    return float(filename.split(".txt")[0].rsplit("_", 3)[-3])


def getN(filename):
    return int(filename.split(".txt")[0].rsplit("_", 5)[-5])


def getS(filename):  # instance number
    return int(filename.split(".txt")[0].rsplit("_", 2)[-1])


def createFileList(directory, sizes, instances, alphas):
    fileList = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and not f.endswith('.zip') and not f.endswith('.sh'))]
        # exclude gs_energies.txt files
        files = [f for f in files
                 if(not f.startswith('gs_energies'))]
        # Below, select only specifed n,s,alpha instances
        files = [f for f in files if(getS(f) in instances and getAlpha(
            f) in alphas and getN(f) in sizes)]
        for f in files:
            fileList.append(root+"/"+f)
    return fileList
# %%
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
        print("No response found for instance: " + instance_name + "in datafile: " + data_file)
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
    parameter_sets = None,
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
        df_name = prefix + 'df_results_random.pkl'
    else:
        df_name = prefix + str(instances[0]) + '_df_results_random.pkl'
    df_path = os.path.join(results_path, df_name)
    print(df_path)

    if parameter_sets is None:
        parameter_sets = itertools.product(
            *(parameters_dict[Name] for Name in parameters_dict))
        parameter_sets = list(parameter_sets)
    parameter_sets = set(parameter_sets)
    total_runs = len(parameter_sets)

    # If use_raw_dataframes compute the row
    if use_raw_dataframes or not os.path.exists(df_path):
        results = []
        for instance in instances:
            # TODO To be completed before by computing this random energy, currently we assume it is zero
            # random_energy = loadResponseFromFile(os.path.join(
            #     data_path, 'random_energies.txt'), prefix + str(instance))
            random_energy = 0
            best_value = loadResponseFromFile(os.path.join(
                data_path, 'gs_energies.txt'), prefix + str(instance) + '.txt')
            # We will assume that the insertion order in the keys is preserved (hence Python3.7+ only) and is sorted alphabetically
            counter = 0
            for parameter_set in parameter_sets:
                counter += 1
                instance_name = prefix + str(instance)
                parameters = dict(zip(parameter_names, parameter_set))
                # df_samples_name = instance_name + "_" + \
                #     '_'.join(str(keys) + '_' + str(vals)
                #              for keys, vals in parameters.items()) + ".pkl"
                # TODO This can be generalized as above but it fails as pcold has 3 digits and phot only 2
                df_samples_name = instance_name + "_swe_{:.0f}_rep_{:.0f}_pcold_{:.2f}_phot_{:.1f}.pkl".format(parameter_set[0], parameter_set[1], parameter_set[2], parameter_set[3])
                df_samples_path = os.path.join(pickles_path, df_samples_name)
                if os.path.exists(df_samples_path):
                    print(prefix,instance,": analyzing ",counter," of ",total_runs)
                    try:
                        df_samples = pd.read_pickle(df_samples_path)
                    except (pickle.UnpicklingError, EOFError):
                        print('Pickle file ' + df_path + ' is corrupted. We will create a new one.')
                        os.replace(df_path, df_path + '.bak')
                        continue
                    # df_samples = pd.read_pickle(df_samples_path)
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
                        bootstrap_iterations = int(min(10**(np.floor(np.log10(max_bootstraps // boots)+0.9)),1000))
                        outputs = computeResultsList(
                            df=df_samples,
                            random_value=random_energy,
                            best_value=best_value/2.0,
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

# %%

# Define function for ensemble metrics


def conf_interval(
    x: pd.Series,
    key_string: str,
    stat_measure = 'median',
    confidence_level: float = 68,
    bootstrap_iterations: int = 1000,
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
    key_estimator_string = str(stat_measure) + '_' + key_string
    mean_deviation = np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(
        x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string])))

    if isinstance(stat_measure, str):

        # Allow named numpy functions
        f = getattr(np, stat_measure)

        # Try to use nan-aware version of function if necessary
        missing_data = np.isnan(np.sum(np.column_stack(x[key_string])))

        if missing_data and not stat_measure.startswith("nan"):
            nanf = getattr(np, f"nan{stat_measure}", None)
            if nanf is None:
                print("Data contain nans but no nan-aware version of `{func}` found")
            else:
                f = nanf
    elif isinstance(stat_measure, int) or isinstance(stat_measure, float):
        f = getattr(np, 'nanpercentile')
        # TODO I need to see how to instantiate this
    else:
        f = stat_measure

    if isinstance(stat_measure, int) or isinstance(stat_measure, float):
        center = np.nanpercentile(x[key_string], stat_measure)
        # TODO Fix this
        lower_interval = center
        upper_interval = center
        bootstrap_confidence_interval = False
        if bootstrap_confidence_interval:
                boot_dist = []
                # Rationale here is that we perform bootstrapping over the entire data set but considering original confidence intervals, which we assume resemble standard deviation from a normally distributed error population. This is in line with the data generation but we might want to fix it
                for i in range(int(bootstrap_iterations)):
                    resampler = np.random.randint(0, len(x[key_string]), len(x[key_string]), dtype=np.intp)  # intp is indexing dtype
                    sample = x[key_string].values.take(resampler, axis=0)
                    sample_ci_upper = x[key_string + '_conf_interval_upper'].values.take(resampler, axis=0)
                    sample_ci_lower = x[key_string + '_conf_interval_upper'].values.take(resampler, axis=0)
                    sample_std = (sample_ci_upper-sample_ci_lower)/2
                    sample_error = np.random.normal(0, sample_std, len(sample))
                    boot_dist.append(np.percentile(sample + sample_error, q=stat_measure/100))
                np.array(boot_dist)
                p = 50 - confidence_level / 2, 50 + confidence_level / 2
                (lower_interval, upper_interval) = np.nanpercentile(boot_dist, p, axis=0)
    else:
        center = f(x[key_string])
    if stat_measure == 'mean':
        # center = np.mean(x[key_string])
        lower_interval = center - mean_deviation
        upper_interval = center + mean_deviation
    elif stat_measure == 'median':
        # center = np.median(x[key_string])
        median_deviation = mean_deviation * \
            np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1))
        lower_interval = center - median_deviation
        upper_interval = center + median_deviation

    result = {
        key_estimator_string: center,
        key_estimator_string + '_conf_interval_lower': lower_interval,
        key_estimator_string + '_conf_interval_upper': upper_interval}
    results = pd.Series(result)
    return results

# %%

def generateStatsDataframe(
    df_all: List[dict] = None,
    stat_measures: List[str] = ['mean', 'median'],
    instance_list: List[str] = None,
    parameters_names: List[str] = None,
    metrics: List[str] = ['min_energy', 'perf_ratio', 'success_prob', 'rtt', 'mean_time', 'inv_perf_ratio'],
    results_path: str = None,
    use_raw_dataframes: bool = False,
    confidence_level: float = 68,
    bootstrap_iterations: int = 1000,
    save_pickle: bool = True,
    lower_bounds: List[float] = None,
    upper_bounds: List[float] = None,
    prefix: str = '',
) -> pd.DataFrame:
    '''
    Function to generate statistics from the aggregated dataframe

    Args:
    TODO review parameters list
        df_all: List of dictionaries containing the aggregated dataframe
        stat_measures: List of statistics to be calculated
        instance_list: List of instances to be considered
        parameters_names: List of parameters' names
        resource_list: List of resources to be considered
        results_path: Path to the directory containing the results
        use_raw_dataframes: Boolean indicating whether to use the raw data for generating the aggregated dataframe
        confidence_level: Confidence level for the confidence intervals in bootstrapping
        bootstrap_iterations: Number of bootstrap iterations
        save_pickle: Boolean indicating whether to save the aggregated pickle
        ocean_df_flag: Boolean indicating whether to use the ocean dataframe
        suffix: Suffix to be added to the pickle name
    '''
    if df_all is None:
        print("Please provide the aggregated dataframe")
        return None
    # Split large dataframe such that we can compute the statistics and confidence interval for each metric across the instances
    # TODO This can be improved by the lambda function version of the approach defining an input parameter for the function as a dictionary. Currently this is too slow
    # Not succesful reimplementation of the operation above
    # df_stats_all = df_results_all.set_index(
    #     'instance').groupby(parameters + ['boots']
    #     ).apply(lambda s: pd.Series({
    #         stat_measure + '_' + metric : conf_interval(s,metric, stat_measure) for metric in metrics_list for stat_measure in stat_measures})
    #     ).reset_index()

    # Create filename
    df_name = prefix + 'df_stats_random.pkl'
    df_path = os.path.join(results_path, df_name)
    if os.path.exists(df_path):
        df_all_stats = pd.read_pickle(df_path)
    else:
        df_all_stats = pd.DataFrame()

    resources = ['boots']

    if all([str(stat_measure) + '_' + metric + '_conf_interval_' + limit in df_all_stats.columns for stat_measure in stat_measures for metric in metrics for limit in ['lower', 'upper']]) and not use_raw_dataframes:
        pass
    else:
        df_all_groups = df_all[
            df_all['instance'].isin(instance_list)
            ].set_index(
                'instance'
            ).groupby(
                parameters_names + resources
                )
        # Remove all groups with fewer than a single instance
        df_filtered = df_all_groups.filter(lambda x: len(x) > 1)
        df_groups = df_filtered.groupby(
                parameters_names + resources
                )
        dataframes = []
        # This function could resemble what is done inside of seaborn to bootstrap everything https://github.com/mwaskom/seaborn/blob/77e3b6b03763d24cc99a8134ee9a6f43b32b8e7b/seaborn/regression.py#L159
        counter = 0
        for metric in metrics:
            for stat_measure in stat_measures:

                df_all_estimator = df_groups.apply(
                    conf_interval,
                    key_string = metric,
                    stat_measure = stat_measure,
                    confidence_level = confidence_level,
                    bootstrap_iterations = bootstrap_iterations,
                    )
                dataframes.append(df_all_estimator)

                # Save intermediate file after 10 metric x stat_measures computations
                if counter % 10 == 0:
                    with open(df_path + '.pickle', 'wb') as f:
                        pickle.dump(dataframes, f)
                counter += 1
        if all([len(i) == 0 for i in dataframes]):
            print('No dataframes to merge')
            return None

        df_all_stats = pd.concat(dataframes, axis=1).reset_index()

    df_stats = df_all_stats.copy()

    for stat_measure in stat_measures:
        for key, value in lower_bounds.items():
            if str(stat_measure) + '_' + key + '_conf_interval_lower' in df_stats.columns:
                df_stats[str(stat_measure) + '_' + key + '_conf_interval_lower'].clip(
                    lower=value, inplace=True)
        for key, value in upper_bounds.items():
            if str(stat_measure) + '_' + key + '_conf_interval_upper' in df_stats.columns:
                df_stats[str(stat_measure) + '_' + key + '_conf_interval_upper'].clip(
                    upper=value, inplace=True)

    # TODO Implement resource_factor as in generate_ws_dataframe.py
    if 'replica' in parameters_names:
        df_stats['reads'] = df_stats['sweep'] * df_stats['replica'] * df_stats['boots']
    elif 'rep' in parameters_names:
        df_stats['reads'] = df_stats['boots']
    else:
        df_stats['reads'] = df_stats['sweep'] * df_stats['boots']

    if save_pickle:
        # df_stats = cleanup_df(df_stats)
        print(df_path)
        df_stats.to_pickle(df_path)

    return df_stats


# %%
# Main

# Input Parameters
n_reads = 100000
float_type = 'float32'

# All instances with 1000 sweeps
# Load files

# instances_path = "/home/x-bernalde/repos/wishart/instance_generation/wishart_planting_N_{:.0f}_alpha_{:.2f}/instances".format(size, alpha)
instances_path = "/home/bernalde/repos/wishart/instance_generation/wishart_planting_N_{:.0f}_alpha_{:.2f}/instances".format(size, alpha)
# data_path = "/home/x-bernalde/repos/wishart/instance_generation/wishart_planting_N_{:.0f}_alpha_{:.2f}/".format(size, alpha)
data_path = "/home/bernalde/repos/wishart/instance_generation/wishart_planting_N_{:.0f}_alpha_{:.2f}/".format(size, alpha)
# wishart_path = "/home/x-bernalde/repos/wishart/instance_generation"
wishart_path = "/home/bernalde/repos/wishart/instance_generation"
wishart_raw_path = "/anvil/scratch/x-bernalde/wishart/results/"
# results_path = "/anvil/scratch/x-bernalde/wishart/results/"
results_path = "/home/bernalde/repos/stochastic-benchmark/data/wishart/results_df/"
pickles_path = results_path

file_list = createFileList(wishart_path, sizes, instances, alphas)

print(file_list)

# %%
# List experiments


# Define parameters dict
default_parameters = {
    'swe': [1000],
    'rep': [1],
    'pcold': [1.00],
    'phot': [50.0],
}

parameters_dict = {
    'swe': sweeps,
    'rep': replicas,
    'pcold': pcolds,
    'phot': phots,
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

parameter_sets_dict = dict.fromkeys(sizes)
for size in sizes:
    parameter_sets_dict[size] = dict.fromkeys(instances)
    # TODO Should this also vary with alpha?
    prefix = 'wishart_planting_N_{:.0f}_alpha_{:.2f}_inst_'.format(size, alpha)


# %%

counter = 0
for file in file_list:

    file_name = file.split(".txt")[0].rsplit("/", 1)[-1]
    print(file_name)

    data = np.loadtxt(file, dtype=int)
    N = int(file.split(".txt")[0].rsplit("_", 5)[-5])
    M = sparse.coo_matrix(
        (data[:, 2], (data[:, 0], data[:, 1])), shape=(N, N))
    problem = M.A
    problem = (problem + problem.T) / 2

    # Get solver
    # solver = Solver(problem=problem, problem_type='ising',
                    # float_type=float_type)

    for n_replicas in replicas:
        for n_sweeps in sweeps:
            for p_hot in phots:
                for p_cold in pcolds:
                    counter += 1
                    print("file "+str(counter)+" of "+str(len(file_list) *
                                                        len(replicas) *
                                                        len(sweeps) *
                                                        len(phots) *
                                                        len(pcolds)))

                    pickle_name = results_path + file_name + \
                        '_swe_' + str(n_sweeps) + \
                        '_rep_' + str(n_replicas) + \
                        '_pcold_' + "%.2f" % round(p_cold,2) + \
                        '_phot_' + "%.1f" % round(p_hot,2) + '.pkl'

                    pickle_scratch_name = wishart_raw_path + file_name + \
                        '_swe_' + str(n_sweeps) + \
                        '_rep_' + str(n_replicas) + \
                        '_pcold_' + "%.2f" % round(p_cold,2) + \
                        '_phot_' + "%.1f" % round(p_hot,2) + '.pkl'

                    print(pickle_name)
                    if os.path.exists(pickle_name) or os.path.exists(pickle_scratch_name):
                        print(pickle_name)
                        #res_1 = pd.read_pickle(pickle_name)
                        pass
                    else:
                        # Generate random sample of largest size of sample to perform measurement.
                        max_reads = int(1e6)

                        
                        local_fields = np.copy(np.diag(problem))
                        couplings = np.copy(problem)
                        np.fill_diagonal(couplings, 0)


                        # Initialize to random state
                        start_time = time.time()
                        states = 2 * np.random.randint(
                            2, size=(max_reads, N)) - 1
                        # Get initial energies
                        energies = np.array([
                            np.dot(state, np.dot(couplings, state)/2 + local_fields)
                            for state in states
                        ])
                        final_time = time.time() - start_time
                        random_df = pd.DataFrame(energies, columns=['best_energy'])
                        random_df['runtime (us)'] = final_time / max_reads * 1e6

                        random_df.to_pickle(pickle_name)

# %%
# Main execution
counter = 0
for size in sizes:
    prefix = 'wishart_planting_N_{:.0f}_alpha_{:.2f}_inst_'.format(size, alpha)
    for instance in instances:
        counter += 1

        df_name = prefix + str(instance) + '_df_results_random.pkl'
        df_path = os.path.join(results_path, df_name)
        if os.path.exists(df_path) and not use_raw_dataframes:
            df_samples = pd.read_pickle(df_path)
        else:
            df_samples = None
            df_samples = createResultsDataframes(
                df=df_samples,
                instances=[instance],
                parameters_dict=parameters_dict,
                parameter_sets=parameter_sets_dict[size][instance],
                bootstraps=bootstraps,
                data_path=data_path,
                results_path=results_path,
                pickles_path=pickles_path,
                confidence_level=confidence_level,
                gap=gap,
                bootstrap_iterations=None,
                use_raw_dataframes=use_raw_dataframes,
                s=s,
                fail_value=fail_value,
                save_pickle=True,
                ocean_df_flag=ocean_df_flag,
            )

# %%
# Create dictionaries for upper and lower bounds of confidence intervals
metrics = ['min_energy', 'rtt',
                'perf_ratio', 'success_prob', 'mean_time', 'inv_perf_ratio']
lower_bounds = {key: None for key in metrics}
lower_bounds['success_prob'] = 0.0
lower_bounds['mean_time'] = 0.0
lower_bounds['inv_perf_ratio'] = EPSILON
upper_bounds = {key: None for key in metrics}
upper_bounds['success_prob'] = 1.0
upper_bounds['perf_ratio'] = 1.0


# %%
# Join all the results in a single dataframe

default_boots = total_reads

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
elif wishart_instances:
    parameters_dict = {
        'swe': sweeps,
        'rep': replicas,
        'pcold': pcolds,
        'phot': phots,
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
parameters_names = list(parameters_dict.keys())

parameter_sets = itertools.product(
    *(parameters_dict[Name] for Name in parameters_dict))
parameter_sets = list(parameter_sets)
complete_files = []

for size in sizes:
    for alpha in alphas:
        # prefix = "random_n_"+str(size)+"_inst_"
        prefix = 'wishart_planting_N_{:.0f}_alpha_{:.2f}_inst_'.format(size, alpha)
        df_name_all = prefix + 'df_results_random.pkl'
        df_path_all = os.path.join(results_path, df_name_all)
        if os.path.exists(df_path_all) and not use_raw_dataframes:
            df_results_all = pd.read_pickle(df_path_all)
        else:
            df_results_list = []
            for instance in instances:
                df_name = prefix + str(instance) + '_df_results_random.pkl'
                df_path = os.path.join(results_path, df_name)
                df_results_list.append(pd.read_pickle(df_path))
            df_results_all = pd.concat(df_results_list, ignore_index=True)
            df_results_all.to_pickle(df_path_all)


        # prefix = 'wishart_planting_N_{:.0f}_alpha_{:.2f}_inst_'.format(size, alpha)
        # df_stats_name = prefix + 'eval_df_stats_random.pkl'
        # df_stats_path = os.path.join(results_path, df_name)
        # df_stats = df_results_all.copy()
        # df_stats['reads'] = df_stats['boots']
        # # df_stats = cleanup_df(df_stats)
        # print(df_stats_path)
        # df_stats.to_pickle(df_stats_path)



# %%
# Main execution

for size in sizes:
    for alpha in alphas:
        # prefix = "random_n_"+str(size)+"_inst_"
        prefix = 'wishart_planting_N_{:.0f}_alpha_{:.2f}_inst_'.format(size, alpha)
        df_name_all = prefix + 'df_results_random.pkl'
        df_path_all = os.path.join(results_path, df_name_all)
        if os.path.exists(df_path_all):
            df_results_all = pd.read_pickle(df_path_all)
        else:
            df_results_all = None
        df_results_all_stats = generateStatsDataframe(
            df_all=df_results_all,
            stat_measures=['median', 10, 50, 90],
            instance_list=instances,
            parameters_names=parameters_names,
            metrics = metrics,
            results_path=results_path,
            use_raw_dataframes=use_raw_dataframes,
            confidence_level=confidence_level,
            bootstrap_iterations=bootstrap_iterations,
            save_pickle=True,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            prefix = prefix + 'all_',
        )
        df_results_all_stats = generateStatsDataframe(
            df_all=df_results_all,
            stat_measures=['median', 10, 50, 90],
            instance_list=evaluate_instances,
            parameters_names=parameters_names,
            metrics = metrics,
            results_path=results_path,
            use_raw_dataframes=use_raw_dataframes,
            confidence_level=confidence_level,
            bootstrap_iterations=bootstrap_iterations,
            save_pickle=True,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            prefix = prefix + 'eval_',
        )

# %%
print(datetime.datetime.now())

# %%