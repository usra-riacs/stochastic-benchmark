import dill
import math
import numpy as np
import os

import bootstrap
import df_utils
import interpolate
import random_exploration
import sequential_exploration
import stochastic_benchmark
import names
import stats
import success_metrics
import training
from utils_ws import *

def max_clique_setup(p):
    here = '../notebooks/example_data/max_clique_p={}'.format(p)
    parameter_names = ['sweeps']
    instance_cols = ['N', 'n', 'idx']

    def resource_fcn(df):
        return df['sweeps'] * df['boots']

    shared_args = {'response_col':'Energy',\
                  'resource_col':'MeanTime',\
                  'response_dir':-1,\
                  'confidence_level':68,\
                  'random_value':0.}

    metric_args = {}
    metric_args['Response'] = {'opt_sense':-1}
    metric_args['SuccessProb'] = {'gap':1.0, 'response_dir':-1}
    metric_args['RTT'] = {'fail_value': np.nan, 'RTT_factor':1.,\
                          'gap':1.0, 's':0.99}

    sms = [success_metrics.Response,
          success_metrics.PerfRatio,
          success_metrics.InvPerfRatio,
          success_metrics.SuccessProb,
          success_metrics.Resource,
          success_metrics.RTT]

    def update_rules(self, df):
        GTMinEnergy = df['GTMinEnergy'].iloc[0]
        MinEnergy = df['Energy'].min()
        self.shared_args['best_value'] = min(GTMinEnergy, MinEnergy)
        self.metric_args['RTT']['RTT_factor'] = df['MeanTime'].iloc[0]

    agg = 'count'
    nboots = 500
    bsParams = bootstrap.BootstrapParameters(shared_args,update_rules,agg,metric_args,sms)
    bs_iter_class = bootstrap.BSParams_iter()
    bsparams_iter = bs_iter_class(bsParams, nboots)

    metrics = ['Response', 'RTT', 'PerfRatio', 'SuccProb', 'MeanTime', 'InvPerfRatio']
    sp = stats.StatsParameters(metrics=metrics, stats_measures=[stats.Median()])

    sb = stochastic_benchmark.stochastic_benchmark(parameter_names, here,
                                                   instance_cols=instance_cols,
                                                   stat_params = sp,
                                                  bsParams_iter=bsparams_iter,
                                                  resource_fcn=resource_fcn)

    budgets = [i*10**j for i in [1, 1.5, 2, 3, 5, 7]
                 for j in [1, 2, 3, 4, 5]] + [10**6]
    
    exploration_fracs = [0.05, 0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8]
    taus = [10, 20, 50, 100, 200, 500, 750, 1000, 2500, 5000, 10000]
    parameter_names = ['sweeps']
    key = names.param2filename({'Key': 'PerfRatio', 'Metric':'median'}, '')

    rsParams = random_exploration.RandomSearchParameters(budgets=budgets,\
                                                        exploration_fracs=exploration_fracs,\
                                                        taus=taus,\
                                                        parameter_names=parameter_names,\
                                                        key=key)

    ssParams = sequential_exploration.SequentialSearchParameters(budgets=budgets,\
                                                                exploration_fracs=exploration_fracs,\
                                                                taus=taus,\
                                                                parameter_names=parameter_names,\
                                                                key='Key=PerfRatio')
 
    sb.run_baseline()
    
    
    sb.run_ProjectionExperiment('TrainingStats')
    sb.run_ProjectionExperiment('TrainingResults')
    sb.run_RandomSearchExperiment(rsParams)
    sb.run_SequentialSearchExperiment(ssParams)
    return sb

def wishart_setup(N, alpha, suffix=None):
    if suffix is None:
        here = '../notebooks/example_data/wishart_N={}_alpha={}'.format(N, alpha)
        parameter_names = ['swe', 'rep', 'pcold', 'phot']
    else:
        here = '../notebooks/example_data/wishart_N={}_alpha={}_{}'.format(N, alpha, suffix)
        parameter_names = ['swe', 'rep', 'phot']
    def resource_fcn(df):
        return df['swe'] * df['rep'] * df['boots']
    sb = stochastic_benchmark.stochastic_benchmark(parameter_names, here=here,\
                                                   train_test_split=0.8, resource_fcn=resource_fcn)
    key = names.param2filename({'Key': 'PerfRatio', 'Metric':'median'}, '')
    
    
    add_hyperopt_trial(sb, offset=0)
    sb.run_baseline()
    recipes,_ = sb.baseline.evaluate()
    recipes.reset_index(inplace=True)
    
    resource_values = list(recipes['resource'])
    budgets = [i*10**j for i in [1, 1.5, 2, 3, 5, 7]
             for j in [3, 4, 5]] + [1e6]
    budgets = np.unique([take_closest(resource_values, b) for b in budgets])
    
    rsParams = random_exploration.RandomSearchParameters(
        budgets=budgets,
        parameter_names=parameter_names,
        key=key)
    
    ssParams = sequential_exploration.SequentialSearchParameters(
        budgets=budgets,
        parameter_names=parameter_names,
        key='Key=PerfRatio')
    
    sb.run_ProjectionExperiment('TrainingStats')
    sb.run_ProjectionExperiment('TrainingResults')
    sb.run_RandomSearchExperiment(rsParams)
    sb.run_SequentialSearchExperiment(ssParams)

    return sb

def mod_wishart_setup(N, alpha, mod, suffix=None):
    if suffix is None:
        here = '../notebooks/example_data/wishart_N={}_alpha={}_mod{}'.format(N, alpha, mod)
        parameter_names = ['swe', 'rep', 'pcold', 'phot']
    else:
        here = '../notebooks/example_data/wishart_N={}_alpha={}_{}'.format(N, alpha, suffix)
        parameter_names = ['swe', 'rep', 'phot']
    
    def resource_fcn(df):
        df['mod_rep'] = (df['rep'] / mod).apply(np.ceil)
        return df['swe'] * df['mod_rep'] * df['boots']
    sb = stochastic_benchmark.stochastic_benchmark(parameter_names, here=here,\
                                                   train_test_split=0.8, resource_fcn=resource_fcn)
    key = names.param2filename({'Key': 'PerfRatio', 'Metric':'median'}, '')
    
    add_hyperopt_trial(sb, offset=0)
#     if not (('order' in sb.interp_results.columns) and ('order' in sb.bs_results.columns)):
#         print('calling hpo')
#         add_hyperopt_trial(sb, offset=0)
    sb.run_baseline()
    recipes,_ = sb.baseline.evaluate()
    recipes.reset_index(inplace=True)
    
    resource_values = list(recipes['resource'])
    budgets = [i*10**j for i in [1, 1.5, 2, 3, 5, 7]
             for j in [3, 4, 5]] + [1e6]
    budgets = np.unique([take_closest(resource_values, b) for b in budgets])
    
    rsParams = random_exploration.RandomSearchParameters(
        budgets=budgets,
        parameter_names=parameter_names,
        key=key)
    
    ssParams = sequential_exploration.SequentialSearchParameters(
        budgets=budgets,
        parameter_names=parameter_names,
        key='Key=PerfRatio')
    
    sb.run_ProjectionExperiment('TrainingStats')
    sb.run_ProjectionExperiment('TrainingResults')
    sb.run_RandomSearchExperiment(rsParams)
    sb.run_SequentialSearchExperiment(ssParams)

    return sb

def add_hyperopt_trial(sb, offset=100):
    ### Bandaid for adding hyperopt ordering data. Add after sb is initialized but before trials are run
    
    base_dir = sb.here.cwd
    hpo_pickle_path = os.path.join(base_dir, 'processing_data/hyperopt_trial.pickle')
    
    with open(hpo_pickle_path, 'rb') as in_strm:
        trials = dill.load(in_strm)

    vals = trials.vals
    for df in [sb.interp_results, sb.bs_results]:
        df['order'] = np.nan
        count = 0
        for idx in range(offset, len(vals['sweeps'])):
            sweeps = vals['sweeps'][idx]
            replicas = vals['replicas'][idx] 
            pcold = vals['pcold'][idx]
            phot = vals['phot'][idx]

            df_idx = df.index[(df['swe'] == sweeps) 
                              & (df['rep'] == replicas)
                              & (df['pcold'] == pcold)
                              & (df['phot'] == phot)]

            if len(df_idx) == 0:
#                 print('No rows found for idx {}'.format(idx))
                continue
            else:
                df.loc[df_idx, 'order'] = count
                count += 1
    sb.bs_results.to_pickle(sb.here.bootstrap)
    sb.interp_results.to_pickle(sb.here.interpolate)
    
def skpleiades_setup(n):
    here = 'example_data/sk_pleiades_n={}'.format(n)
    parameter_names = ['sweep', 'Tcfactor', 'Thfactor']
    key = names.param2filename({'Key': 'PerfRatio', 'Metric':'median'}, '')
    rsParams = random_exploration.RandomSearchParameters(parameter_names=parameter_names,
        key=key)
    sb = stochastic_benchmark.stochastic_benchmark(parameter_names, here=here)
    sb.run_baseline()
    sb.run_ProjectionExperiment('TrainingStats')
    sb.run_ProjectionExperiment('TrainingResults')
    sb.run_RandomSearchExperiment(rsParams)
    return sb