from collections import defaultdict
import dill
import glob
import math
from multiprocess import Pool
import numpy as np
import os
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import sys
from tqdm import tqdm

sys.path.append('../../src') #TODO set path to point to src of stochastic-benchmark
import bootstrap
import interpolate
import names
import random_exploration
import sequential_exploration
import stats
import stochastic_benchmark
import success_metrics
from utils_ws import *

import wishart_runs #TODO I need to find the file I sent previous -> add to path if necessary

alpha = '0.50'
# datadir = '/home/bernalde/repos/stochastic-benchmark/examples/wishart_n_50_alpha_{}/data'.format(alpha)  #TODO set this to where the pickled datafiles are stored
datadir = '/home/bernalde/repos/stochastic-benchmark/examples/wishart_n_50_alpha_{}/rerun_data'.format(alpha) #TODO set this to where the pickled datafiles are stored


def compress_order(df_single):
    df_order = df_single[['warmstart={}_hpo_order={}'.format(h, hpo_trial) for h in [0, 1] for hpo_trial in range(10)]]
    count_df = df_order.sum(axis = 0, min_count=1).to_frame().T
    copy_cols = ['grid_search', 'GTMinEnergy', 'Energy', 'count', 'MeanTime']
    copy_row = df_single[copy_cols].iloc[0].values
    count_df[copy_cols] = copy_row
    return count_df

def instance_df(instance_num):
    df_list = []
    for h in [0, 1]:
        for hpo_trial in range(10):
            hpo_filename = 'hpoTrials_warmstart={}_trial={}_inst={}.pkl'.format(h, hpo_trial, instance_num)
            hpoTrial_file = os.path.join(datadir, hpo_filename)
            with open(hpoTrial_file, 'rb') as f:
                hpoTrial = dill.load(f)
            
            if h == 1:
                hpo_offset = 675 #TODO replace with the number of trials in the grid search
            else:
                hpo_offset = 0

            record = []
            if h == 1 and hpo_trial == 0:
                start_idx = 0
            elif h == 0:
                start_idx = 0
            else:
                start_idx = 675
            for idx in range(start_idx, start_idx + 100):
                sweeps = int(hpoTrial.vals['sweeps'][idx])
                replicas = int(hpoTrial.vals['replicas'][idx])
                pcold = np.round(hpoTrial.vals['pcold'][idx], decimals=2)
                phot = np.maximum(0.1, np.round(hpoTrial.vals['phot'][idx], decimals=1))
                
                order = idx if idx >= hpo_offset else np.nan
                df_filename, obj_filename = wishart_runs.logname(instance_num, sweeps, replicas, pcold, phot)
                temp_df = pd.read_pickle(df_filename)
                with open(obj_filename, 'rb') as f:
                    obj = dill.load(f)
                    mean_time = dill.load(f)

                temp_df['MeanTime'] = mean_time
                temp_df['warmstart={}_hpo_order={}'.format(h, hpo_trial)] = hpo_trial
                temp_df['grid_search'] = int(np.isnan(order))
                temp_df['instance'] = instance_num
                df_list.append(temp_df)
      
    df = pd.concat(df_list, ignore_index=True)
    # df = df.groupby(['sweeps', 'replicas', 'pcold', 'phot', 'instance']).apply(compress_order)
    # df.reset_index(inplace=True)
    return df

def prepare_raw_data():
    exp_raw_dir = os.path.join(os.getcwd(), 'wishart_n_50_alpha={}/exp_raw'.format(alpha))
    if not os.path.exists(exp_raw_dir):
        os.makedirs(exp_raw_dir)
    def f(instance_num):
        df_filename = os.path.join(exp_raw_dir, 'raw_results_inst={}.pkl'.format(instance_num))
        df = instance_df(instance_num)
        df.to_pickle(df_filename)

    with Pool() as p:
        p.map(f, range(1, 51))
        
def random_values(instance_num):
    #TODO if you want to update the random values for each instance write this here.
    # calling a lookup table is probably the fastest
    rv = 0.
    return rv

def postprocess_linear(recipe):
    x_range = (10**3, 10**6)
    post_recipe_dict = {}
    
    parameter_names = ['sweeps', 'replicas', 'pcold', 'phot']
    x = np.array(recipe['resource'])
    range_idx = np.where((x >= x_range[0]) & (x <= x_range[1]))
    x = x[range_idx]
    post_recipe_dict['resource'] = x
    x = x.reshape((-1, 1))
    for param in parameter_names:
        param_vals = np.array(recipe[param])
        param_vals = param_vals[range_idx]
        model = LinearRegression().fit(x, param_vals)
        param_pred = model.predict(x)
        post_recipe_dict[param] = param_pred
    pred_recipe = pd.DataFrame.from_dict(post_recipe_dict)
    return pred_recipe

def postprocess_polynomial(recipe, deg):
    post_recipe_dict = {}
    post_recipe_dict['resource'] = np.array(recipe['resource'].copy())
    parameter_names = ['sweeps', 'replicas', 'pcold', 'phot']
    x = np.array(recipe['resource']).reshape((-1, 1))
    for param in parameter_names:
        param_vals = np.array(recipe[param])
        model=make_pipeline(PolynomialFeatures(deg),LinearRegression()).fit(x,param_vals)
        param_pred = model.predict(x)
        post_recipe_dict[param] = param_pred
    pred_recipe = pd.DataFrame.from_dict(post_recipe_dict)
    return pred_recipe
        
def postprocess_custom(recipe):
    post_recipe_dict = {}
    post_recipe_dict['resource'] = np.array(recipe['resource'].copy())
    parameter_names = ['sweeps', 'replicas', 'pcold', 'phot']
    x = np.array(recipe['resource']).reshape((-1, 1))
    for param in parameter_names:
        param_vals = np.array(recipe[param])
        if param == 'sweeps':
            model = TransformedTargetRegressor(regressor=LinearRegression(),
                func=np.log, inverse_func = np.exp).fit(x, param_vals)
        else:
            model = LinearRegression().fit(x, param_vals)
        param_pred = model.predict(x)
        post_recipe_dict[param] = param_pred
    pred_recipe = pd.DataFrame.from_dict(post_recipe_dict)
    return pred_recipe

def postprocess_random(meta_params):
    # print(meta_params)
    x_range = (10**3, 10**6)
    post_recipe_dict = {}
    
    parameter_names = ['ExploreFrac', 'tau']
    x = np.array(meta_params['TotalBudget'])
    range_idx = np.where((x >= x_range[0]) & (x <= x_range[1]))
    x = x[range_idx]
    post_recipe_dict['TotalBudget'] = x
    x = x.reshape((-1, 1))
    for param in parameter_names:
        param_vals = np.array(meta_params[param])
        param_vals = param_vals[range_idx]
        if param == 'ExploreFrac':
            post_recipe_dict[param] = np.mean(param_vals) * np.ones_like(param_vals)
        elif param == 'tau':
            # model = LinearRegression().fit(x, param_vals)
            # param_pred = model.predict(x)
            # param_pred = np.clip(param_pred, None, 10**4)
            # TODO doing a step fit to the values
            param_pred = param_vals.copy()
            range_idx_1 = np.where(x <= 5*10**4)[0]
            param_pred[range_idx_1] = 10**2
            range_idx_2 = np.where(x > 5*10**4)[0]
            param_pred[range_idx_2] = 10**4
            post_recipe_dict[param] = param_pred
    pred_recipe = pd.DataFrame.from_dict(post_recipe_dict)
    pred_recipe['ExplorationBudget'] = pred_recipe['TotalBudget'] * pred_recipe['ExploreFrac']
    return pred_recipe

def stoch_bench_setup():
    # Set up basic information 
    alpha = '0.5'
    # path to working directory
    here = os.path.join('/home/robin/stochastic-benchmark/examples', 'wishart_n_50_alpha_{}/'.format(alpha))
    parameter_names = ['sweeps', 'replicas', 'pcold', 'phot']
    instance_cols = ['instance'] #indicates how instances should be grouped, default is ['instance']

    ## Response information 
    response_key = 'PerfRatio' # Column with the response, default is 'PerfRatio'
    response_dir = 1 # whether we want to maximize (1) or minimize (-1), default is 1

    ## Optimizations informations
    recover = True #Whether we want to read dataframes when available, default is True
    reduce_mem = True #Whether we want to segment bootstrapping and interpolation to reduce memory usage, default is True
    smooth = True #Whether virtual best should be monontonized, default is True

    sb = stochastic_benchmark.stochastic_benchmark(parameter_names, here, instance_cols, response_key, response_dir, recover, reduce_mem, smooth)

    # Set up bootstrapped parameters
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

    def update_rules(self, df):  #These update the bootstrap parameters for each group 
        GTMinEnergy = df['GTMinEnergy'].iloc[0] 
        self.shared_args['best_value'] = GTMinEnergy #update best value for each instance
        self.metric_args['RTT']['RTT_factor'] = df['MeanTime'].iloc[0]

    agg = 'count' #aggregated column
    #success metric we want to calculate
    sms = [success_metrics.Response,
            success_metrics.PerfRatio,
            success_metrics.InvPerfRatio,
            success_metrics.SuccessProb,
            success_metrics.Resource,
            success_metrics.RTT]
    boots_range = range(50, 1001, 50) 
    ssOrderCols = ['warmstart={}_hpo_order={}'.format(h, hpo_trial) for h in [0, 1] for hpo_trial in range(10)]
    bsParams = bootstrap.BootstrapParameters(shared_args=shared_args,
                                                update_rule=update_rules,
                                                agg=agg,
                                                metric_args=metric_args,
                                                success_metrics=sms,
                                                keep_cols=ssOrderCols)
    bs_iter_class = bootstrap.BSParams_range_iter()
    bsparams_iter = bs_iter_class(bsParams, boots_range)

    #How names should be parsed from raw filesnames
    def group_name_fcn(raw_filename):
        raw_filename = os.path.basename(raw_filename)
        start_idx = raw_filename.index('inst')
        end_idx = raw_filename.index('.')
        return raw_filename[start_idx: end_idx]

    # Run bootstrap
    sb.run_Bootstrap(bsparams_iter, group_name_fcn)

    # Interpolate 
    def resource_fcn(df):
        return df['sweeps'] * df['replicas'] * df['boots']
    iParams = interpolate.InterpolationParameters(resource_fcn,
                                                        parameters=parameter_names,
                                                        ignore_cols = ssOrderCols)

    sb.run_Interpolate(iParams)

    # Set up Stats computations
    train_test_split = 0.8
    metrics = ['Response', 'RTT', 'PerfRatio', 'SuccProb', 'MeanTime', 'InvPerfRatio']
    stParams = stats.StatsParameters(metrics=metrics, stats_measures=[stats.Median()])

    sb.run_Stats(stParams, train_test_split)
    # Run virtual best baseline
    sb.run_baseline()
    sb.run_ProjectionExperiment('TrainingStats', lambda x : postprocess_linear(x), 'linear')
    sb.run_ProjectionExperiment('TrainingResults', lambda x : postprocess_linear(x), 'linear')
    #Set up Random search parameters and sequential search paramters

    # Make sure search budgets align with the baselines - needed for the distance
    recipes,_ = sb.baseline.evaluate()
    recipes.reset_index(inplace=True)
    resource_values = list(recipes['resource'])
    budgets = [i*10**j for i in [1, 1.5, 2, 3, 5, 7]
                for j in [3, 4, 5]] + [1e6]
    budgets = np.unique([take_closest(resource_values, b) for b in budgets])

    # which columns determin the order in sequential search experiments
    ssOrderCols0 = ['warmstart=0_hpo_order={}'.format(hpo_trial) for hpo_trial in range(10)] 
    ssOrderCols1 = ['warmstart=1_hpo_order={}'.format(hpo_trial) for hpo_trial in range(10)] 


    # Which column you are optimizing. Different from initialization b/c aggregated metrics include the name
    key = names.param2filename({'Key': 'PerfRatio', 'Metric':'median'}, '')

    rsParams = random_exploration.RandomSearchParameters(
        budgets=budgets,
        parameter_names=parameter_names,
        key=key)
        
    ssParams0 = sequential_exploration.SequentialSearchParameters(
        budgets=budgets,
        order_cols=ssOrderCols0,
        parameter_names=parameter_names,
        key='Key=PerfRatio')

    ssParams1 = sequential_exploration.SequentialSearchParameters(
        budgets=budgets,
        order_cols=ssOrderCols1,
        parameter_names=parameter_names,
        key='Key=PerfRatio')

    sb.run_RandomSearchExperiment(rsParams, postprocess=postprocess_random, postprocess_name='custom')
    sb.run_SequentialSearchExperiment(ssParams0, id_name='cold', postprocess=postprocess_random, postprocess_name='custom')
    sb.run_SequentialSearchExperiment(ssParams1, id_name='warm', postprocess=postprocess_random, postprocess_name='custom')

    sb.run_StaticRecommendationExperiment(sb.experiments[0])
    sb.run_StaticRecommendationExperiment(sb.experiments[1])

    testing_results = sb.interp_results[sb.interp_results['train'] == 0].copy()
    testing_instances = list(np.unique(testing_results['instance']))
    # for idx in [5, 6]:
    #     parameters_list = sb.experiments[idx].list_runs()
    #     if idx == 5:
    #         filename = os.path.join(sb.here.checkpoints, 'FixedRecommendation_id=cold.pkl')
    #     else:
    #         filename = os.path.join(sb.here.checkpoints, 'FixedRecommendation_id=warm.pkl')
    #     if os.path.exists(filename):
    #         rerun_df = pd.read_pickle(filename)
    #     else:
    #         results_dict = run_experiments(parameters_list, testing_instances)
    #         rerun_df = process_rerun(sb, results_dict)
    #         rerun_df.to_pickle(filename)
        
    #     sb.experiments[idx].attach_runs(rerun_df, process=False)

    return sb

class experiment_results():
    def __init__(self, resource_list):
        self.resource_list = resource_list
        self.experiments = {}
    
def compress_experiments(parameters_list):
    parameters_dict = defaultdict(list)
    for params in parameters_list:
        resource = params.resource
        sweeps = int(params.sweeps)
        replicas = int(params.replicas)
        pcold = np.round(params.pcold, decimals = 2)
        phot = np.maximum(0.1, np.round(params.phot, decimals = 1))
        parameters_dict[(sweeps, replicas, pcold, phot)].append(resource)
    return parameters_dict

def run_experiments(parameters_list, testing_instances):
    parameters_dict = compress_experiments(parameters_list)
    results_dict = {k:experiment_results(v) for k, v, in parameters_dict.items()}

    for instance_num in testing_instances:
        for params, results in results_dict.items():
            exp_params = (0, *params)
            result = wishart_runs.rerun_pysa(exp_params, instance_num)
            result['instance'] = instance_num
            results.experiments[instance_num] = result

    return results_dict

def process_rerun(sb, results_dict):
    bs_params = next(sb.bsParams_iter)
    resource_col = bs_params.shared_args['resource_col']
    response_col = bs_params.shared_args['response_col'] 
    agg = bs_params.agg

    def evaluate_single(df_single, n_reads):
        bs_params.update_rule(bs_params, df_single)
        resources = df_single[resource_col].values
        responses = df_single[response_col].values
        resources = np.repeat(resources, df_single[agg])
        responses = np.repeat(responses, df_single[agg])

        if agg is not None:
            p =  list(df_single[bs_params.agg] / df_single[bs_params.agg].sum())
            resamples = np.random.choice(len(df_single), (n_reads, bs_params.bootstrap_iterations), p=p)
        else:
            resamples = np.random.randint(0, len(df_single), size=(n_reads, bs_params.bootstrap_iterations), dtype=np.intp)
        responses = df_single[bs_params.shared_args['response_col']].values[resamples]
        resources = df_single[bs_params.shared_args['resource_col']].values[resamples]

        bs_df = pd.DataFrame()
        for metric_ref in bs_params.success_metrics:
            metric = metric_ref(bs_params.shared_args,bs_params.metric_args[metric_ref.__name__])
            metric.evaluate(bs_df, responses, resources)
        for col in bs_params.keep_cols:
            if col in df_single.columns:
                val = df_single[col].iloc[0]
                bs_df[col] = val
        
        return bs_df
    
    def prepare_param(k):
        exp_res = results_dict[k]
        sweeps = k[0]
        replicas = k[1]
        pcold = k[2]
        phot = k[3]
        res_list = []
        for resource in exp_res.resource_list:
            n_reads = math.floor(resource / (sweeps * replicas))
            for instance_num, df_single in exp_res.experiments.items():
                res = evaluate_single(df_single, n_reads)
                res['instance'] = instance_num
                res['resource'] = resource
                res_list.append(res)
        if len(res_list) == 0:
            return 
        res = pd.concat(res_list, ignore_index=True)
        # except:
        #     print('failing interal for parameters {}'.format(k))
        res['sweeps'] = sweeps
        res['replicas'] = replicas
        res['pcold'] = pcold
        res['phot'] = phot
        return res
    # with Pool() as p:
    #     ret_list = p.map(prepare_param, results_dict.keys())
    ret_list = []
    for k in tqdm(results_dict.keys()):
        ret = prepare_param(k)
        ret_list.append(ret)
    try:
        ret = pd.concat(ret_list, ignore_index=True)
    except:
        print('failing external loop')
    return ret
        
def main():
    # prepare_raw_data()
    sb = stoch_bench_setup()
    
    return 

if __name__ == '__main__':
    main()

