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
jobid = 42


# Input Parameters
total_reads = 1000
overwrite_pickles = False
# if 0 == 1:
if int(str(sys.argv[2])) == 1:
    ocean_df_flag = True
else:
    ocean_df_flag = False
compute_best_found_flag = False
use_raw_dataframes = True
instances_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/instances"
data_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/"
if ocean_df_flag:
    pickles_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/dneal/pickles"
    results_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/dneal"
else:
    pickles_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/pysa/pickles"
    results_path = "/nobackup/dbernaln/repos/stochastic-benchmark/data/sk/pysa"


# instances_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/instances"
# data_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/"
# if ocean_df_flag:
#     pickles_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/dneal/pickles"
#     results_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/dneal/"
# else:
#     pickles_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/pysa/pickles"
#     results_path = "/home/bernalde/repos/stochastic-benchmark/data/sk/pysa/"

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
sizes.append(int(str(sys.argv[1])))
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
# sizes.append(int(str(sys.argv[1])))
# sizes = [100]
replicas = [2**i for i in range(0, 4)]
# replicas.append(int(str(sys.argv[2])))
# instances.append(int(jobid))
instances = [i for i in range(0,20)] + [42]
training_instances = [i for i in range(0,20)]
Tcfactors = [0.0]
Thfactors = list(np.linspace(-3, 1, num=33, endpoint=True))
# Thfactors = [0.0]


EPSILON = 1e-10
confidence_level = 68

bootstrap_iterations = 1000

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

# Define function for ensemble metrics


def conf_interval(
    x: pd.Series,
    key_string: str,
    stat_measure: str = 'median',
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
    key_estimator_string = stat_measure + '_' + key_string
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
    else:
        f = stat_measure
    
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
    else:
        # TODO I need to debug this part
        boot_dist = []
        # Rationale here is that we perform bootstrapping over the entire data set but considering original confidence intervals, which we assume resemble standard deviation from a normally distributed error population. THis is in line with the data generation but we might want to fix it
        for i in range(int(bootstrap_iterations)):
            resampler = np.random.randint(0, len(x[key_string], len(x[key_string]), dtype=np.intp))  # intp is indexing dtype
            sample = x[key_string].values.take(resampler, axis=0)
            sample_ci_upper = x[key_string + '_conf_interval_upper'].values.take(resampler, axis=0)
            sample_ci_lower = x[key_string + '_conf_interval_upper'].values.take(resampler, axis=0)
            sample_std = (sample_ci_upper-sample_ci_lower)/2
            sample_error = np.random.normal(0, sample_std, len(sample))
            boot_dist.append(f(sample + sample_error))
        np.array(boot_dist)
        p = 50 - confidence_level / 2, 50 + confidence_level / 2
        (lower_interval, upper_interval) = np.nanpercentile(boot_dist, p, axis=0)

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
    parameters_dict: dict = None,
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
        parameters_dict: Dictionary of parameters to be considered, with list as values
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

    parameters_names = list(parameters_dict.keys())
    resources = ['boots']
    # Create filename
    df_name = prefix + 'df_stats.pkl'
    df_path = os.path.join(results_path, df_name)
    if os.path.exists(df_path):
        df_all_stats = pd.read_pickle(df_path)
    else:
        df_all_stats = pd.DataFrame()
    if all([stat_measure + '_' + metric + '_conf_interval_' + limit in df_all_stats.columns for stat_measure in stat_measures for metric in metrics for limit in ['lower', 'upper']]) and not use_raw_dataframes:
        pass
    else:
        df_all_groups = df_all[
            df_all['instance'].isin(instance_list)
            ].set_index(
                'instance'
            ).groupby(
                parameters_names + resources
                )
        dataframes = []
        # This function could resemble what is done inside of seaborn to bootstrap everything https://github.com/mwaskom/seaborn/blob/77e3b6b03763d24cc99a8134ee9a6f43b32b8e7b/seaborn/regression.py#L159
        for metric in metrics:
            for stat_measure in stat_measures:
                
                df_all_estimator = df_all_groups.apply(
                    conf_interval,
                    key_string = metric,
                    stat_measure = stat_measure,
                    confidence_level = confidence_level,
                    bootstrap_iterations = bootstrap_iterations,
                    )
                dataframes.append(df_all_estimator)
        if all([len(i) == 0 for i in dataframes]):
            print('No dataframes to merge')
            return None

        df_all_stats = pd.concat(dataframes, axis=1).reset_index()

    df_stats = df_all_stats.copy()

    for stat_measure in stat_measures:
        for key, value in lower_bounds.items():
            df_stats[stat_measure + '_' + key + '_conf_interval_lower'].clip(
                lower=value, inplace=True)
        for key, value in upper_bounds.items():
            df_stats[stat_measure + '_' + key + '_conf_interval_upper'].clip(
                upper=value, inplace=True)
    # TODO Implement resource_factor as in generate_ws_dataframe.py
    if 'replica' in parameters_names:
        df_stats['reads'] = df_stats['sweep'] * df_stats['replica'] * df_stats['boots']
    else:
        df_stats['reads'] = df_stats['sweep'] * df_stats['boots']

    if save_pickle:
        # df_stats = cleanup_df(df_stats)
        df_stats.to_pickle(df_path)

    return df_stats

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
    df_name_all = prefix + 'df_results.pkl'
    df_path_all = os.path.join(results_path, df_name_all)
    if os.path.exists(df_path_all) and not use_raw_dataframes:
        df_results_all = pd.read_pickle(df_path_all)
    else:
        df_results_list = []
        for instance in instances:
            df_name = prefix + str(instance) + '_df_results.pkl'
            df_path = os.path.join(results_path, df_name)
            df_results_list.append(pd.read_pickle(df_path))
        df_results_all = pd.concat(df_results_list, ignore_index=True)
        df_results_all.to_pickle(df_path_all)
        
            

# %%
# Main execution

for size in sizes:
    prefix = "random_n_"+str(size)+"_inst_"
    df_name_all = prefix + 'df_results.pkl'
    df_path_all = os.path.join(results_path, df_name_all)
    if os.path.exists(df_path_all):
        df_results_all = pd.read_pickle(df_path_all)
    else:
        df_results_all = None
    df_results_all_stats = generateStatsDataframe(
        df_all=df_results_all,
        stat_measures=['median'],
        instance_list=training_instances,
        parameters_dict=parameters_dict,
        metrics = metrics,
        results_path=results_path,
        use_raw_dataframes=use_raw_dataframes,
        confidence_level=confidence_level,
        bootstrap_iterations=bootstrap_iterations,
        save_pickle=True,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        prefix = prefix,
    )

# %%
print(datetime.datetime.now())

# %%
