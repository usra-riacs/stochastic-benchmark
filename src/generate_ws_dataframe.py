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
from bisect import bisect_left
from math import hypot
import random

idx = pd.IndexSlice

# %%
# jobid = int(os.getenv('PBS_ARRAY_INDEX'))
jobid = 42


# Input Parameters
total_reads = 1000
overwrite_pickles = False
# if int(str(sys.argv[2])) == 1:
if 1 == 1:
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
# sizes.append(int(str(sys.argv[1])))
sizes = [200]
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
# %%


# %%
# Join all the results in a single dataframe

default_boots = total_reads
minimum_boots = 1

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
# Function to interpolate dataframes across a resource column
def interpolate_df(
        dataframe: pd.DataFrame = None,
        resource_column: str = 'reads',
        prefix: str = '',
        parameters_dict: dict = None,
        default_boots: int = default_boots,
        minimum_boots: int = minimum_boots,
        resource_proportional_parameters: list = ['sweep', 'replica'],
        idx = pd.IndexSlice,
        save_pickle: bool = True,
    ):
    if dataframe is None:
        print('Error: Dataframe is None')
        return None
    if len(dataframe) == 0:
        print('Error: Dataframe is empty')
        return None
    df = dataframe.copy()
    dataframes = []
    parameter_names = list(parameters_dict.keys())
    parameter_sets = itertools.product(
        *(parameters_dict[Name] for Name in parameters_dict))
    parameter_sets = list(parameter_sets)
    r_indices = []
    if resource_column not in df.columns:
        df[resource_column] = df['boots']
        for r_parameters in resource_proportional_parameters:
            if r_parameters in parameter_names:
                df[resource_column] *= df_results[r_parameters]
    resouce_values = df[resource_column].values
    resouce_values = np.sort(np.unique(resouce_values))
    instances = [0]
    if 'instance' in df.columns:
        instances = df['instance'].unique().tolist()
    df_index = df.set_index(parameter_names).copy()
    for r_parameters in resource_proportional_parameters:
        if r_parameters in parameter_names:
            r_indices.append(parameter_names.index(r_parameters))
    
    for instance in instances:
        for parameter_set in parameter_sets:
            if parameter_set not in df_index.index.to_list():
                print('Parameter set',parameter_set,'not found')
                continue# For each parameter setting remove repeated reads
            df_values = df_index.loc[idx[parameter_set]].copy()
            if 'instance' in df.columns:
                df_values = df_values.loc[df_values['instance'] == instance].copy()
            df_original = df_values.drop_duplicates(
                subset=resource_column, keep='first').copy()
            resource_factor = 1
            for r_index in r_indices:
                resource_factor *= parameter_set[r_index]
            # Set interpolation points for the responses at all the relevant reads values
            interpolate_resource = resouce_values[
                np.where(
                    (resouce_values <= default_boots*resource_factor) & 
                    (resouce_values >= minimum_boots*resource_factor)
                    )
                ]
            # Create a dataframe with the interesting reads as index and all the columns
            dummy_df = pd.DataFrame(
                np.NaN,
                index = interpolate_resource,
                columns = df_original.columns
                )
            dummy_df.drop(columns=resource_column, inplace=True)
            # Fill out the values that we have certain
            df_interpolate = dummy_df.combine_first(
                df_original.set_index(resource_column)).copy()
            # Interpolate for all the other values (not extrapolated)
            df_interpolate = df_interpolate.interpolate(
                method='linear', limit_area='inside'
                ).dropna(how='all').reset_index().rename(
                    columns={'index':resource_column}).copy()
            # Computing the boots column
            df_interpolate['boots'] = df_interpolate[resource_column]/resource_factor
            # Reading the parameter columns
            for key, value in zip(parameter_names,parameter_set):
                df_interpolate[key] = value
            if 'instance' in df.columns:
                df_interpolate['instance'] = instance
            dataframes.append(df_interpolate)

    if all([len(i) == 0 for i in dataframes]):
        print('No dataframes to merge')
        return None
    df_interpolated = pd.concat(dataframes).reset_index(drop=True)
    if save_pickle:
        df_name_interpolated = prefix.rsplit('.')[0] + '_interp.pkl'
        df_path_interpolated = os.path.join(results_path, df_name_interpolated)
        df_interpolated.to_pickle(df_path_interpolated)
    return df_interpolated
# %%
def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

# %%
# Main execution

# Interpolate results accross resources
for size in sizes:
    prefix = "random_n_"+str(size)+"_inst_"
    df_name_stats = prefix + 'df_stats.pkl'
    df_path_stats = os.path.join(results_path, df_name_stats)
    if not os.path.exists(df_path_stats):
        df_stats = None
    else:
        df_stats = pd.read_pickle(df_path_stats)
        if overwrite_pickles:
            df_stats_interpolated = interpolate_df(
                dataframe=df_stats,
                resource_column='reads',
                prefix=df_name_stats,
                parameters_dict=parameters_dict,
                default_boots=default_boots,
                minimum_boots=minimum_boots,
                resource_proportional_parameters=['sweep', 'replica'],
                idx = idx,
                save_pickle=True,
            )
        else:
            df_name_interpolated = df_name_stats.rsplit('.')[0] + '_interp.pkl'
            df_path_interpolated = os.path.join(results_path, df_name_interpolated)
            df_stats_interpolated = pd.read_pickle(df_path_interpolated)
    df_name_results = prefix + 'df_results.pkl'
    df_path_results = os.path.join(results_path, df_name_results)
    if not os.path.exists(df_path_results):
        df_results = None
    else:
        df_results = pd.read_pickle(df_path_results)
        if overwrite_pickles:
            df_results_interpolated = interpolate_df(
                dataframe=df_results,
                resource_column='reads',
                prefix=df_name_results,
                parameters_dict=parameters_dict,
                default_boots=default_boots,
                minimum_boots=minimum_boots,
                resource_proportional_parameters=['sweep', 'replica'],
                idx = idx,
                save_pickle=True,
            )
        else:
            df_name_interpolated = df_name_results.rsplit('.')[0] + '_interp.pkl'
            df_path_interpolated = os.path.join(results_path, df_name_interpolated)
            df_results_interpolated = pd.read_pickle(df_path_interpolated)

# %%

# Create virtual best and virtual worst columns
# TODO This can be generalized as using as groups the parameters that are not dependent of the metric (e.g., schedule) or that signify different solvers
# TODO This needs to be functionalized
stale_parameters = []
varying_parameters = []
numeric_parameters = []
for parameter_name in parameter_names:
    if len(locals()[parameter_name+'s']) == 1:
        stale_parameters.append(parameter_name)
    else:
        varying_parameters.append(parameter_name)
        if isinstance(locals()[parameter_name+'s'][0],int) or isinstance(locals()[parameter_name+'s'][0],float):
            numeric_parameters.append(parameter_name)

df_name_virtual = prefix + 'df_virtual.pkl'
df_path_virtual = os.path.join(results_path, df_name_virtual)

# Computation of virtual best perf_ratio as commended by Davide, several points: 1) the perf_ratio is computed as the maximum (we are assuming we care about the max) of for each instance for each read, 2) the median of this idealized solver (that has the best parameters for each case) across the instances is computed

response_column='perf_ratio'
response_direction=1

# df_virtual = df_virtual.merge(
if response_direction == -1: # Minimization
    df_virtual = df_results_interpolated[[response_column] + parameter_names + ['instance','reads']].set_index(
            parameter_names
            ).groupby(['instance','reads']
            )[response_column].min().reset_index().set_index(
                ['instance']
                ).groupby(['reads']
                ).median().reset_index().sort_values(
                ['reads']).rename(columns={response_column: 'virt_best_'+response_column})
else: # Maximization
    df_virtual = df_results_interpolated[[response_column] + parameter_names + ['instance','reads']].set_index(
            parameter_names
            ).groupby(['instance','reads']
            )[response_column].max().reset_index().set_index(
                ['instance']
                ).groupby(['reads']
                ).median().reset_index().sort_values(
                ['reads']).rename(columns={response_column: 'virt_best_'+response_column})
        #     ).expanding(min_periods=1).max(),
        # on=['reads'],
        # how='left')
# TODO Generalize direction of search

df_virtual = df_virtual.merge(df_results_interpolated[
    [response_column] + parameter_names + ['instance','reads']
    ].set_index(
            parameter_names
            ).groupby(['instance','reads']
            )[response_column].max().reset_index().set_index(
                ['instance']
                ).groupby(['reads']
                ).max().reset_index().sort_values(
                ['reads']).rename(columns={response_column: 'envelope_'+response_column}),
    on=['reads'],
    how='left')

# Recipe to obtain parameter setting that optimizes the median of the response (across instances) in the statistics dataframe for each read
recipe_lazy = df_stats_interpolated[
        ['median_'+response_column] + parameter_names + ['reads']
    ].set_index(
        parameter_names
    ).groupby(['reads']
        ).idxmax().rename(
            columns={'median_'+response_column:'recipe'})

df_virtual = df_virtual.merge(
    df_stats_interpolated[
        ['median_'+response_column] + parameter_names + ['reads']
    ].set_index(
            parameter_names
            ).groupby(['reads']
            ).max().reset_index().rename(columns={'median_perf_ratio':'lazy_perf_ratio'}))

recipe_lazy = recipe_lazy.merge(
    recipe_lazy['recipe'].apply(pd.Series).rename(
        columns={i: parameter_names[i] for i in range(len(parameter_names))}
        ).reset_index(),
        on=['reads'],
        how='left')

recipe_lazy=recipe_lazy.merge(
    df_stats_interpolated[
        ['median_'+response_column] + parameter_names + ['reads']
    ].set_index(
            parameter_names
            ).groupby(['reads']
            ).max().reset_index(),
        on=['reads'],
        how='left')

# Recipe to obtain parameter setting that optimizes the response  in the results dataframe for each instance and read, then takes the mean of the parameters across instances, for each read.
# There is an extra projection step into the parameter values
recipe_mean_best_params = df_results_interpolated[
        [response_column] + parameter_names + ['instance','reads']
    ].set_index(
    numeric_parameters
    ).groupby(['instance','reads']
        )[response_column].idxmax().apply(pd.Series).reset_index().set_index(
            ['instance']
            ).groupby(['reads']
            ).mean().rename(columns={i: numeric_parameters[i] for i in range(len(numeric_parameters))}).reset_index()

for stale_parameter in stale_parameters:
    recipe_mean_best_params[stale_parameter] = locals()[stale_parameter+'s'][0]

# Project the numeric parameters in the recipe to the evaluated parameters
for numeric_parameter in numeric_parameters:
    # parameter_list = locals()[numeric_parameter+'s']
    parameter_list = np.sort(df_results_interpolated[numeric_parameter].unique())
    recipe_mean_best_params[numeric_parameter]=recipe_mean_best_params[numeric_parameter].apply(lambda x: take_closest(parameter_list,x))

# # Project the reads to the closest value in boots_list*sweeps
# recipe_mean_best_params['boots'] = recipe_mean_best_params.index/recipe_mean_best_params['sweeps']
# recipe_mean_best_params['boots']=recipe_mean_best_params['boots'].apply(lambda x: take_closest(boots_list,x))
# recipe_mean_best_params.index = recipe_mean_best_params['boots']*recipe_mean_best_params['sweeps']
# recipe_mean_best_params.index.rename('reads',inplace=True)

# Join parameters in recipe to single column
recipe_mean_best_params['params'] = recipe_mean_best_params[parameter_names].apply(tuple, axis = 1)
df_results_interpolated['params'] = df_results_interpolated[parameter_names].apply(tuple, axis = 1)

# Projecting parameter setting absent in the original dataframe to one that is available using the Euclidean norm.
# TODO The euclidean norm does not take into account the fact that the scale of the parameters is different. Moreover, it does not work with non-numerical data
# TODO performance improvement here. This is by no means efficient.
included_parameters = df_results_interpolated[numeric_parameters].values
dist_eucl = lambda a, b: hypot(b[0]-a[0], b[0]-a[0])
if not all([i in df_results_interpolated['params'].values for i in recipe_mean_best_params['params'].values]):
    for index, row in recipe_mean_best_params.iterrows():
        if row['params'] not in set(df_results_interpolated['params'].tolist()):
            non_included_parameters = row[numeric_parameters].values
            print('These parameters are not included in the original database', non_included_parameters)
            new_parameters = min(included_parameters, key=lambda co: dist_eucl(co, non_included_parameters))
            print('Projected parameters as', new_parameters)
            for i, numeric_parameter in enumerate(numeric_parameters):
                recipe_mean_best_params[numeric_parameter][index] = new_parameters[i]

# Join parameters in recipe to single column
recipe_mean_best_params['params'] = recipe_mean_best_params[parameter_names].apply(tuple, axis = 1)
recipe_mean_best_params['recipe'] = recipe_mean_best_params[parameter_names + ['reads']].apply(tuple, axis = 1)

dummy_df = df_stats_interpolated[
        ['median_'+response_column] + parameter_names + ['reads']    
    ].set_index(
    parameter_names + ['reads']
    ).loc[pd.MultiIndex.from_tuples(recipe_mean_best_params['recipe']
    )].copy()

dummy_df.index.rename(parameter_names + ['reads'], inplace=True)

df_virtual = df_virtual.merge(
    dummy_df.reset_index()[
        ['median_'+response_column] + ['reads']
    ],
    on=['reads'],
    how='left')

recipe_mean_best_params = recipe_mean_best_params.merge(
    dummy_df.reset_index()[
        ['median_'+response_column] + ['reads']
    ],
    on=['reads'],
    how='left')

df_name_virtual = df_name_results.rsplit('.')[0] + '_virtual.pkl'
df_path_virtual = os.path.join(results_path, df_name_virtual)
df_virtual.to_pickle(df_path_virtual)

df_name_recipe_mean_best = df_name_results.rsplit('.')[0] + '_recipe_mean_best.pkl'
df_path_recipe_mean_best = os.path.join(results_path, df_name_recipe_mean_best)
recipe_mean_best_params.to_pickle(df_path_recipe_mean_best)

df_name_recipe_mean_best = df_name_results.rsplit('.')[0] + '_recipe_mean_best.pkl'
df_path_recipe_mean_best = os.path.join(results_path, df_name_recipe_mean_best)
recipe_mean_best_params.to_pickle(df_path_recipe_mean_best)

# %%
# Function to perform alternative processing of progress dataframes


def process_df_progress(
    df_progress: pd.DataFrame = None,
    compute_metrics: list = ['perf_ratio'],
    stat_measures: list = ['median'],
    maximizing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Function to process progress dataframes, computing statistics across experiments and reporting best behavior for each budget.

    Args:
        df_progress: Dataframe containing progress data
        compute_metrics: List of metrics to compute
        stat_measures: List of statistics to compute
        maximizing: Boolean indicating whether to maximize or minimize the metric

    Returns:
        Tuple of dictionaries of which there are
            df_progress_processed: Processed dataframe
            df_progress_end: Processed dataframe


    '''
    if df_progress is None:
        return None

    experiment_setting = ['R_budget', 'R_explor', 'run_per_solve']
    individual_run = experiment_setting.copy()
    if 'experiment' in df_progress.columns:
        individual_run += ['experiment']

    if maximizing:
        opt_stats = ['max', 'idxmax']
        del_str = 'max_'
    else:
        opt_stats = ['min', 'idxmin']
        del_str = 'min_'

    df_progress_end = df_progress[
        individual_run + compute_metrics + ['cum_reads']
    ].loc[
        df_progress.sort_values(
            ['cum_reads'],
            ascending=False,
        ).groupby(
            individual_run
        )['cum_reads'].idxmax()
    ].copy()

    for compute_metric in compute_metrics:
        if 'inv_' in compute_metric:
            df_progress[compute_metric.replace(
                'inv_', '')] = 1 - df_progress[compute_metric] + EPSILON
            expanding_metric = compute_metric
            df_progress_end[compute_metric.replace(
                'inv_', '')] = 1 - df_progress_end[compute_metric] + EPSILON
        else:
            position = compute_metric.index('perf_ratio')
            df_progress[compute_metric[:position] + 'inv_' +
                        compute_metric[position:]] = df_progress[compute_metric] + EPSILON
            df_progress_end[compute_metric[:position] + 'inv_' +
                            compute_metric[position:]] = df_progress_end[compute_metric] + EPSILON
            expanding_metric = compute_metric[:position] + \
                'inv_' + compute_metric[position:]
    df_progress['best_inv_perf_ratio'] = df_progress.sort_values(
        ['cum_reads', 'R_budget']
    ).expanding(min_periods=1).min()[expanding_metric]
    df_progress['best_perf_ratio'] = 1 - \
        df_progress['best_inv_perf_ratio'] + EPSILON
    
    if 'f_explor' not in df_progress:
        df_progress['f_explor'] = df_progress['R_explor'] / \
            df_progress['R_budget']

    # df_progress = cleanup_df(df_progress)

    if 'f_explor' not in df_progress_end:
        df_progress_end['f_explor'] = df_progress_end['R_explor'] / \
            df_progress_end['R_budget']
    # df_progress_end = cleanup_df(df_progress_end)

    df_progress_best = df_progress_end.groupby(
        experiment_setting
    )[
        compute_metrics
    ].agg(stat_measures).groupby(
        ['R_budget']
    ).agg(
        opt_stats
    ).copy()

    df_progress_best.columns = ["_".join(reversed(pair)).replace(
        del_str, '') for pair in df_progress_best.columns]
    df_progress_best.reset_index(inplace=True)

    for stat_measure in stat_measures:
        for compute_metric in compute_metrics:
            if 'inv_' in compute_metric:
                df_progress_best[stat_measure + '_' + compute_metric.replace('inv_', '')] = 1 - \
                    df_progress_best[stat_measure +
                                     '_' + compute_metric] + EPSILON
            else:
                position = compute_metric.index('perf_ratio')
                df_progress_best[stat_measure + '_' + compute_metric[:position] + 'inv_' +
                                 compute_metric[position:]] = 1 - df_progress_best[stat_measure + '_' + compute_metric] + EPSILON

    # df_progress_best = cleanup_df(df_progress_best)

    for column in df_progress_best.columns:
        if 'idx' in column:
            df_progress_best=df_progress_best.merge(df_progress_best[
                'idxmean_median_perf_ratio'
                ].apply(pd.Series).rename(
                columns={i: experiment_setting[i] for i in range(len(experiment_setting))}
                ), on='R_budget', how='left')

    if 'f_explor' not in df_progress_best.columns:
        df_progress_best['f_explor'] = df_progress_best['R_explor'] / \
            df_progress_best['R_budget']
    return df_progress_best, df_progress_end

# %%
# Random search for the ensemble
repetitions = 10  # Times to run the algorithm
rs = [10, 20, 50, 100, 200, 500, 1000]  # resources per parameter setting (reads/resource_factor = boots)
frac_r_exploration = [0.05, 0.1, 0.2, 0.5, 0.75]
R_budgets = [i*10**j for i in [1, 1.5, 2, 3, 5, 7] for j in [3, 4, 5]] + [1e6] # Total search budget (reads)
experiments = rs * repetitions
df_name_progress = df_name_results.rsplit('.')[0] + '_progress_binary.pkl'
df_path_progress = os.path.join(results_path, df_name_progress)
search_metric = 'median_'+response_column
compute_metric = 'median_'+response_column
df_search = df_stats_interpolated[
    parameter_names + list(set([compute_metric] + [search_metric])) + ['boots','reads']
].set_index(
    parameter_names + ['boots']
).sort_index()
df_reads = df_stats_interpolated[
    parameter_names + ['reads']
].set_index(
    parameter_names
).sort_index()

r_indices = []
resource_proportional_parameters = ['sweep', 'replica']
for r_parameters in resource_proportional_parameters:
    if r_parameters in parameter_names:
        r_indices.append(parameter_names.index(r_parameters))
# Set the parameter sets as those in the results dataframe
parameter_sets = list(set(df_results_interpolated['params'].values))
use_raw_dataframes = False
if os.path.exists(df_path_progress) and not use_raw_dataframes:
    df_progress_total = pd.read_pickle(df_path_progress)
else:
    progress_list = []
    for R_budget in R_budgets:
        for frac_expl_total in frac_r_exploration:
            # budget for exploration (runs)
            R_exploration = int(R_budget*frac_expl_total)
            # budget for exploitation (runs)
            R_exploitation = R_budget - R_exploration
            for r in rs:
                if r > R_exploration:
                    print(
                        "R_exploration must be larger than single exploration step")
                    continue
                for experiment in range(repetitions):
                    random_parameter_sets = random.choices(
                        parameter_sets, k=int(R_exploration / r))
                    # Conservative estimate of very unlikely scenario that we choose all sweeps=1
                    # % Question: Should we replace these samples? We are replacing now

                    
                    all_reads = df_reads.loc[
                        idx[random_parameter_sets[0]]
                    ]
                    resource_factor = 1
                    for r_index in r_indices:
                        resource_factor *= random_parameter_sets[0][r_index]
                    actual_reads = take_closest(all_reads.values,r*resource_factor)[0]
                    # Start random exploration with best found point
                    series_list = [df_search.loc[idx[random_parameter_sets[0]+((actual_reads / resource_factor),)]]]
                    total_reads = actual_reads

                    if total_reads > R_exploration:
                        # TODO: There should be a better way of having parameters that affect runtime making an appear. An idea, having a function f(params) = runs that we can call
                        print(
                            "R_exploration must be larger than single exploration step")
                        continue
                        # We allow it to run at least once assuming that
                    for random_parameter_set in random_parameter_sets:
                        resource_factor = 1
                        for r_index in r_indices:
                            resource_factor *= random_parameter_set[r_index]
                        total_reads += r*resource_factor
                        if total_reads > R_exploration:
                            converged = True
                            break
                        series_list.append(
                            df_search.loc[
                                idx[random_parameter_set + (r,)]
                            ]
                        )
                    exploration_step = pd.concat(series_list, axis=1).T.rename_axis(
                        parameter_names + ['boots'])
                    exploration_step[compute_metric] = exploration_step[compute_metric].expanding(
                        min_periods=1).max()
                    exploration_step.reset_index('boots', inplace=True)
                    exploration_step['experiment'] = experiment
                    exploration_step['run_per_solve'] = r
                    exploration_step['R_explor'] = R_exploration
                    exploration_step['R_exploit'] = R_exploitation
                    exploration_step['R_budget'] = R_budget
                    exploration_step['cum_reads'] = exploration_step.groupby('experiment').expanding(
                        min_periods=1)['reads'].sum().reset_index(drop=True).values
                    progress_list.append(exploration_step)
                    # Exploitation step
                    exploitation_step = df_search.reset_index().set_index(
                        parameter_names).loc[exploration_step.nlargest(1, compute_metric).index]
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
                    exploitation_step['R_budget'] = R_budget
                    progress_list.append(exploitation_step)
    df_progress_total = pd.concat(progress_list, axis=0)
    df_progress_total.reset_index(inplace=True)
    df_progress_total.to_pickle(df_path_progress)

df_progress_total['f_explor'] = df_progress_total['R_explor'] / df_progress_total['R_budget']
df_progress_total.to_pickle(df_path_progress)
# for stat_measure in ['median']:
#     if 'best_' + stat_measure + '_perf_ratio' not in df_progress_total.columns:
#         df_progress_total[stat_measure + '_inv_perf_ratio'] = 1 - \
#             df_progress_total[stat_measure + '_perf_ratio'] + EPSILON
#         df_progress_total['best_' + stat_measure + '_inv_perf_ratio'] = df_progress_total.sort_values(
#             ['cum_reads', 'R_budget']
#         ).expanding(min_periods=1).min()[stat_measure + '_inv_perf_ratio']
#         df_progress_total['best_' + stat_measure + '_perf_ratio'] = 1 - \
#             df_progress_total['best_' + stat_measure +
#                               '_inv_perf_ratio'] + EPSILON
# df_progress_total = cleanup_df(df_progress_total)
# df_progress_total.to_pickle(df_path)
# %%
df_progress_best, df_progress_end = process_df_progress(
    df_progress=df_progress_total,
    compute_metrics=['median_perf_ratio'],
    stat_measures=['mean'],
    maximizing=True,
)
# %%
print(datetime.datetime.now())

# %%