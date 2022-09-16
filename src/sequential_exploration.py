from dataclasses import dataclass, field
from itertools import product

from tqdm import tqdm

import pandas as pd
import df_utils
import names
from utils_ws import *

import stats
tqdm.pandas()

@dataclass
class SequentialSearchParameters:
    budgets: list = field(default_factory=lambda: [i*10**j for i in [1, 1.5, 2, 3, 5, 7]
             for j in [3, 4, 5]] + [1e6] )
    exploration_fracs: list = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.5, 0.75])
    taus: list = field(default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000, 10000])
    order_col: str = 'order'
    optimization_dir: int = 1
    parameter_names: list = field(default_factory=lambda: ['sweep', 'replica'])
    key: str = 'PerfRatio'
    stat_measure: stats.StatsMeasure = stats.Mean()

def prepare_search(stats_df: pd.DataFrame, ssParams: SequentialSearchParameters):
    resource_values = list(stats_df['resource'])
    ssParams.taus = np.unique([take_closest(resource_values, r) for r in ssParams.taus])
    
def summarize_experiments(df: pd.DataFrame, ssParams: SequentialSearchParameters):
    group_on = ['TotalBudget', 'ExplorationBudget', 'tau']
    summary = df.groupby(group_on).agg(ssParams.stat_measure)
    summary.reset_index(inplace=True)
    
    if ssParams.optimization_dir == 1:
        best_agg_alloc = summary.sort_values(
            ssParams.key, ascending=False).drop_duplicates('TotalBudget')
        
    elif ssParams.optimization_dir == -1:
        best_agg_alloc = summary.sort_values(
            ssParams.key, ascending=True).drop_duplicates('TotalBudget')
    
    df_list = []
    for t in best_agg_alloc['TotalBudget']:
        row = best_agg_alloc[best_agg_alloc['TotalBudget'] == t]
        df_list.append(df[(df['tau'] == row['tau'].iloc[0]) &\
                              (df['ExplorationBudget'] == row['ExplorationBudget'].iloc[0]) &\
                              (df['TotalBudget'] == t)])
    exp_at_best = pd.concat(df_list, ignore_index=True)
    return best_agg_alloc, exp_at_best

def SequentialExplorationSingle(df_stats: pd.DataFrame, ssParams: SequentialSearchParameters, budget: float, explore_frac:float, tau:int):
    explore_budget = budget * explore_frac
    if explore_budget < tau:
#         print('Sequential search experiment terminated due to budget')
        return
    
    df_tau = df_stats[df_stats['resource'] == tau].copy()
    df_tau.sort_values(by=ssParams.order_col, ascending=True, inplace=True, ignore_index=True)
    df_tau.dropna(axis=0, how='any', subset=[ssParams.order_col, ssParams.key], inplace=True)
    
    if len(df_tau) == 0:
#         print('Sequential search experiment terminated due to not enough data')
        return
    
    n = int(explore_budget / tau)
    df_tau = df_tau.iloc[0:min(n, len(df_tau))]
    
    if ssParams.optimization_dir == 1:
        best_pars = df_tau.loc[[df_tau[ssParams.key].idxmax()]]
        best_val = df_tau[ssParams.key].max()
    elif ssParams.optimization_dir == -1:
        best_pars = df_tau.loc[[df_tau[ssParams.key].idxmin()]]
        best_val = df_tau[ssParams.key].min()
    
    df_tau['exploit'] = 0
    best_pars = best_pars[ssParams.parameter_names]

    exploit_df = df_stats.merge(best_pars, on=ssParams.parameter_names)
    exploit_df['exploit'] = 1
    exploit_df.loc[:, ssParams.key].fillna(0., inplace=True)
    
    names_dict = names.filename2param(ssParams.key)
    names_dict.update({'ConfInt': 'lower'})
    CIlower = names.param2filename(names_dict, '')
    names_dict.update({'ConfInt': 'upper'})
    CIupper = names.param2filename(names_dict, '')

    if ssParams.optimization_dir == 1:
        exploit_df.loc[:, [ssParams.key, CIlower, CIupper]].clip(lower=best_val, inplace=True)
    elif ssParams.optimization_dir == -1:
        exploit_df.loc[:, [ssParams.key, CIlower, CIupper]].clip(upper=best_val, inplace=True)

    df_experiment = pd.concat([df_tau, exploit_df], ignore_index=True)
    df_experiment['tau'] = tau
    df_experiment['TotalBudget'] = budget
    df_experiment['ExplorationBudget'] = explore_budget
    df_experiment['CummResource'] = df_experiment['resource'].expanding(min_periods=1).sum()
    df_experiment = df_experiment[df_experiment['CummResource'] <= budget]
    
    return df_experiment
        
def run_experiments(df_stats: pd.DataFrame, ssParams: SequentialSearchParameters):
    final_values = []
    total = len(ssParams.budgets) * len(ssParams.exploration_fracs) * len(ssParams.taus)
    pbar = tqdm(product(ssParams.budgets, ssParams.exploration_fracs, ssParams.taus), total=total)
    for budget, explore_frac, tau in pbar:
        df_experiment = SequentialExplorationSingle(df_stats, ssParams, budget, explore_frac, tau)
        if df_experiment is None:
            continue
        final_values.append(df_experiment.iloc[[-1]])
    if len(final_values) == 0:
        return pd.DataFrame(columns=(list(df_stats.columns)\
                                     + ['exploit', 'tau', 'TotalBudget', 'ExplorationBudget', 'CummResource']))
    
    else:
        return pd.concat(final_values, ignore_index=True)

def apply_allocations(df_stats: pd.DataFrame, ssParams: SequentialSearchParameters, best_agg_alloc: pd.DataFrame, group_on: list):
    final_values = []
    for _, row in best_agg_alloc.iterrows():
        budget = row['TotalBudget']
        explore_budget = row['ExplorationBudget']
        tau = row['tau']
        explore_frac = float(explore_budget) / float(budget)
        df_experiment = df_utils.applyParallel(df_stats.groupby(group_on),
                                              lambda df: SequentialExplorationSingle(df, ssParams, budget, explore_frac, tau))
#         df_experiment = SequentialExplorationSingle(df_stats, ssParams, budget, explore_frac, tau)
        final_values.append(df_experiment.iloc[[-1]])
    return pd.concat(final_values, ignore_index=True)
                                
def SequentialExploration(df_stats: pd.DataFrame, ssParams: SequentialSearchParameters, group_on: list):
    prepare_search(df_stats, ssParams)
#     final_values = df_utils.applyParallel(df_stats.groupby(group_on), lambda df: run_experiments(df, ssParams))
    final_values = df_stats.groupby(group_on).progress_apply(lambda df: run_experiments(df, ssParams))
    best_agg_alloc, exp_at_best = summarize_experiments(final_values, ssParams)
    return best_agg_alloc, exp_at_best, final_values