from dataclasses import dataclass, field
from itertools import product

from tqdm import tqdm

import pandas as pd
from utils_ws import *

import stats

@dataclass
class RandomSearchParameters:
    budgets: list = field(default_factory=lambda: [i*10**j for i in [1, 1.5, 2, 3, 5, 7]
             for j in [3, 4, 5]] + [1e6] )
    exploration_fracs: list = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.5, 0.75])
    Nexperiments: int = 10
    taus: list = field(default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000, 10000])
    optimization_dir: int = 1
    parameter_names: list = field(default_factory=lambda: ['sweep', 'replica'])
    key: str = 'PerfRatio'
    stat_measure: stats.StatsMeasure = stats.Mean()

def prepare_search(stats_df: pd.DataFrame, rsParams: RandomSearchParameters):
    resource_values = list(stats_df['resource'])
#     print(resource_values)
#     print(rsParams.taus)
    rsParams.taus = np.unique([take_closest(resource_values, r) for r in rsParams.taus])
    
def summarize_experiments(df: pd.DataFrame, rsParams: RandomSearchParameters):
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
    explore_budget = budget * explore_frac
    if explore_budget < tau:
        return
    df_tau = df_stats[df_stats['resource'] == tau].copy()
    df_tau = df_tau.sample(n = int(explore_budget / tau), replace=True)
    
    if rsParams.optimization_dir == 1:
        best_pars = df_tau.loc[[df_tau[rsParams.key].idxmax()]]
        best_val = df_tau[rsParams.key].max()
    elif rsParams.optimization_dir == -1:
        best_pars = df_tau.loc[[df_tau[rsParams.key].idxmin()]]
        best_val = df_tau[rsParams.key].min()
    
    df_tau['exploit'] = 0
    best_pars = best_pars[rsParams.parameter_names]

    exploit_df = df_stats.merge(best_pars, on=rsParams.parameter_names)
    exploit_df['exploit'] = 1
    exploit_df.loc[:, rsParams.key].fillna(0., inplace=True)

    if rsParams.optimization_dir == 1:
        exploit_df.loc[:, rsParams.key].clip(lower=best_val, inplace=True)
    elif rsParams.optimization_dir == -1:
        exploit_df.loc[:, rsParams.key].clip(upper=best_val, inplace=True)

    df_experiment = pd.concat([df_tau, exploit_df], ignore_index=True)
    df_experiment['tau'] = tau
    df_experiment['TotalBudget'] = budget
    df_experiment['ExplorationBudget'] = explore_budget
    df_experiment['CummResource'] = df_experiment['resource'].expanding(min_periods=1).sum()
    df_experiment = df_experiment[df_experiment['CummResource'] <= budget]
    
    return df_experiment
        
def run_experiments(df_stats: pd.DataFrame, rsParams: RandomSearchParameters):
    final_values = []
    total = len(rsParams.budgets) * len(rsParams.exploration_fracs) * len(rsParams.taus) *  rsParams.Nexperiments
    pbar = tqdm(product(rsParams.budgets, rsParams.exploration_fracs, rsParams.taus, range(rsParams.Nexperiments)), total=total)
    for budget, explore_frac, tau, experiment in pbar:
        df_experiment = single_experiment(df_stats, rsParams, budget, explore_frac, tau)
        if df_experiment is None:
            continue
        df_experiment['Experiment'] = experiment
        final_values.append(df_experiment.iloc[[-1]])
    return pd.concat(final_values, ignore_index=True)

def apply_allocations(df_stats: pd.DataFrame, rsParams: RandomSearchParameters, best_agg_alloc: pd.DataFrame):
    final_values = []
    for _, row in best_agg_alloc.iterrows():
        budget = row['TotalBudget']
        explore_budget = row['ExplorationBudget']
        tau = row['tau']
        explore_frac = float(explore_budget) / float(budget)
        for experiment in range(rsParams.Nexperiments):
            df_experiment = single_experiment(df_stats, rsParams, budget, explore_frac, tau)
            df_experiment['Experiment'] = experiment
            final_values.append(df_experiment.iloc[[-1]])
    return pd.concat(final_values, ignore_index=True)
                                
def RandomExploration(df_stats: pd.DataFrame, rsParams: RandomSearchParameters):
    prepare_search(df_stats, rsParams)
    final_values = run_experiments(df_stats, rsParams)
    best_agg_alloc, exp_at_best = summarize_experiments(final_values, rsParams)
    return best_agg_alloc, exp_at_best, final_values
    