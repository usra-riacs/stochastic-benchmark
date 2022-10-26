from dataclasses import dataclass, field
from itertools import product

from tqdm import tqdm

import pandas as pd
from utils_ws import *

import stats
import names

@dataclass
class RandomSearchParameters:
    """
    Defines parameters for Random search experiments
    
    Parameters
    ----------
    budgets : list[int]
        Defines total resource budget
    exploration_fracs : list[float]
        Fractions of the budgets that should be used for explorations
    Nexperiments : int
        Number of experiments to run
    taus : list[int]
        List of how many time steps to explore each parameter setting for
    optimization_dir : int
        Whether we want to maximize (1) or minimize(-1) the response
    parameter_names : list[str]
        The parameter names - should be columns in the dataframe
    key : str
        Indicates the response column
    stat_measure : stats.StatsMeasure
        How the experimental results should be aggregated over experiments
        
    """
    budgets: list = field(default_factory=lambda: [i*10**j for i in [1, 1.5, 2, 3, 5, 7]
             for j in [3, 4, 5]] + [1e6] )
    exploration_fracs: list = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.5, 0.75])
    Nexperiments: int = 10
    taus: list = field(default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000, 10000])
    optimization_dir: int = 1
    parameter_names: list = field(default_factory=lambda: ['sweep', 'replica'])
    key: str = 'PerfRatio'
    restrict: str = str()
    stat_measure: stats.StatsMeasure = stats.Mean()

def prepare_search(stats_df: pd.DataFrame, rsParams: RandomSearchParameters):
    """
    Prepares search on stats_df by aligning taus with resource values seen in stats_df
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Dataframe that random search will be conducted on
    rsParams : RandomSearchParameters
        Parameters to align with stats_df resources
        
    Returns
    -------
        None : modifies rsParams.taus
    """
    resource_values = list(stats_df['resource'])
    rsParams.taus = np.unique([take_closest(resource_values, r) for r in rsParams.taus])
    
def summarize_experiments(df: pd.DataFrame, rsParams: RandomSearchParameters):
    """
    Aggregates experiments using rsParams.stat_measure over TotalBudget, ExplorationBudget and taus.
    For each
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the results from all experiments
    rsParams : RandomSearchParameters
    
    Returns
    -------
    best_agg_alloc : pd.DataFrame
        Best allocation of taus and exploration fraction for each value of the TotalBudget
    exp_at_best : pd.DataFrame
        Experiments that correspond to the best_agg_alloc values. Used for determining best found parameters
        and response for each experiment.
    """
    group_on = ['TotalBudget', 'ExplorationBudget', 'tau']
    summary = df.groupby(group_on).agg(rsParams.stat_measure)
    summary.reset_index(inplace=True)
    
    if rsParams.optimization_dir == 1:
        best_agg_alloc = summary.sort_values(
            rsParams.key, ascending=False).drop_duplicates('TotalBudget')
        
    elif rsParams.optimization_dir == -1:
        best_agg_alloc = summary.sort_values(
            rsParams.key, ascending=True).drop_duplicates('TotalBudget')
    
    df_list = []
    for t in best_agg_alloc['TotalBudget']:
        row = best_agg_alloc[best_agg_alloc['TotalBudget'] == t]
        df_list.append(df[(df['tau'] == row['tau'].iloc[0]) &\
                              (df['ExplorationBudget'] == row['ExplorationBudget'].iloc[0]) &\
                              (df['TotalBudget'] == t)])
    exp_at_best = pd.concat(df_list, ignore_index=True)
    return best_agg_alloc, exp_at_best

def single_experiment(df_stats: pd.DataFrame, rsParams: RandomSearchParameters, budget: float, explore_frac:float, tau:int):
    """
    Runs experiment on df_stats for a single setting of the exploration parameters (budget, explore_frac, and tau)
    
    Parameters
    ----------
    df_stats : pd.DataFrame
        Dataframe of stats that are used in experiments
    rsParams : RandomSearchParameters
    budget : int 
        Total budget
    explore_frac : float 
        Fraction of budget to use exploring
    tau : int
        Timesteps to use at each exploration step
    
    Returns
    -------
    Dataframe representing entire run of the experiment
    """
    explore_budget = budget * explore_frac
    tau = take_closest(list(df_stats['resource']), tau)
    if explore_budget < tau:
        return
    if np.isclose(tau, 0.):
        return
    if len(rsParams.restrict) == 0:
        df_tau = df_stats[df_stats['resource'] == tau].copy()
    else:
        df_tau = df_stats[(df_stats['resource'] == tau)
                        & (df_stats[rsParams.restric] == True)].copy()
    df_tau = df_tau.sample(n = int(explore_budget / tau), replace=True)
    
    
    if rsParams.optimization_dir == 1:
        best_pars = df_tau.loc[[df_tau[rsParams.key].idxmax()]]
        best_val = df_tau[rsParams.key].max()
    elif rsParams.optimization_dir == -1:
        best_pars = df_tau.loc[[df_tau[rsParams.key].idxmin()]]
        best_val = df_tau[rsParams.key].min()
    
    df_tau['exploit'] = 0
    df_tau['resource_step'] = df_tau['resource'].copy()
    best_pars = best_pars[rsParams.parameter_names]

    exploit_df = df_stats.merge(best_pars, on=rsParams.parameter_names)
    exploit_df['exploit'] = 1
    exploit_df.loc[:, rsParams.key].fillna(0., inplace=True)
    names_dict = names.filename2param(rsParams.key)
    names_dict.update({'ConfInt': 'lower'})
    CIlower = names.param2filename(names_dict, '')
    names_dict.update({'ConfInt': 'upper'})
    CIupper = names.param2filename(names_dict, '')
    if rsParams.optimization_dir == 1:
        exploit_df.loc[:, [rsParams.key, CIlower, CIupper]].clip(lower=best_val, inplace=True)
    elif rsParams.optimization_dir == -1:
        exploit_df.loc[:, [rsParams.key, CIlower, CIupper]].clip(upper=best_val, inplace=True)
        
    exploit_df['resource_step'] = exploit_df['resource'].diff().fillna(exploit_df['resource'])

    df_experiment = pd.concat([df_tau, exploit_df], ignore_index=True)
    df_experiment['tau'] = tau
    df_experiment['TotalBudget'] = budget
    df_experiment['ExplorationBudget'] = explore_budget
    df_experiment['CummResource'] = df_experiment['resource_step'].expanding(min_periods=1).sum()
    df_experiment = df_experiment[df_experiment['CummResource'] <= budget]
    return df_experiment
        
def run_experiments(df_stats: pd.DataFrame, rsParams: RandomSearchParameters):
    """
    Runs all experiments and returns concatenated results
    """
    final_values = []
    total = len(rsParams.budgets) * len(rsParams.exploration_fracs) * len(rsParams.taus) *  rsParams.Nexperiments
    pbar = tqdm(product(rsParams.budgets, rsParams.exploration_fracs, rsParams.taus, range(rsParams.Nexperiments)), total=total)
    for budget, explore_frac, tau, experiment in pbar:
        df_experiment = single_experiment(df_stats, rsParams, budget, explore_frac, tau)
        if df_experiment is None:
            continue
        df_experiment['Experiment'] = experiment
        final_values.append(df_experiment.iloc[[-1]])
    if len(final_values) == 0:
        return pd.DataFrame(columns=(list(df_stats.columns) + ['exploit', 'tau', 'TotalBudget', 'ExplorationBudget', 'CummResource']))
    
    else:
        return pd.concat(final_values, ignore_index=True)

def apply_allocations(df_stats: pd.DataFrame, rsParams: RandomSearchParameters, best_agg_alloc: pd.DataFrame):
    """
    Applies best allocations to a new dataframe and returns results (useful for evaluating allocations on the 
    testing dataset)
    """
    final_values = []
    for _, row in best_agg_alloc.iterrows():
        budget = row['TotalBudget']
        explore_budget = row['ExplorationBudget']
        tau = row['tau']
        explore_frac = float(explore_budget) / float(budget)
        for experiment in range(rsParams.Nexperiments):
            df_experiment = single_experiment(df_stats, rsParams, budget, explore_frac, tau)
            if df_experiment is None:
                continue
            df_experiment['Experiment'] = experiment
            final_values.append(df_experiment.iloc[[-1]])
    return pd.concat(final_values, ignore_index=True)
                                
def RandomExploration(df_stats: pd.DataFrame, rsParams: RandomSearchParameters):
    """
    Runs random exploration experiments
    
    Returns
    -------
    best_agg_alloc : Best allocation of exploration meta-parameters (explore frac and tau)
    exp_at_best : Corresponding experiments at the best allocations
    final_values : The final values of all experiments
    """
    prepare_search(df_stats, rsParams)
    final_values = run_experiments(df_stats, rsParams) #The final values (i.e., last response for each experiment)
    best_agg_alloc, exp_at_best = summarize_experiments(final_values, rsParams)
    return best_agg_alloc, exp_at_best, final_values
    