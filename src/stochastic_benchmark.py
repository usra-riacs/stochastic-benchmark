import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import seaborn.objects as so
import seaborn as sns


import bootstrap
import df_utils
import interpolate
import random_exploration
import sequential_exploration
import stats
import success_metrics
import training
import names
import utils_ws



def prepare_bootstrap(nboots = 1000, 
                      response_col = names.param2filename({'Key': 'MinEnergy'}, ''),
                      resource_col = names.param2filename({'Key': 'MeanTime'}, '')):    
    shared_args = {'response_col':response_col,\
              'resource_col':resource_col,\
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
    bsParams = bootstrap.BootstrapParameters(shared_args, metric_args, sms)
    
    bs_iter_class = bootstrap.BSParams_iter()
    bsparams_iter = bs_iter_class(bsParams, nboots)
    return bsparams_iter

def sweep_boots_resource(df):
    return df['sweep'] * df['boots']

class ProjectionExperiment:
    def __init__(self, parent, project_from):
        self.parent = parent
        self.name = 'Projection from {}'.format(project_from)
        self.project_from = project_from
        self.populate()
        
    def populate(self):
        rec_path = os.path.join(self.parent.here.checkpoints, 'Projection_from={}.pkl'.format(self.project_from))
        
        # Prepare the recipes
        if self.project_from == 'TrainingStats':
            br_train_path = os.path.join(self.parent.here.checkpoints, 'BestRecommended_train.pkl')
            if os.path.exists(br_train_path):
                self.recipe = pd.read_pickle(br_train_path)
            else:
                response_col = names.param2filename(
            {'Key': self.parent.response_key, 'Metric': self.parent.stat_params.stats_measures[0].name}, '')
                self.recipe = training.best_parameters(self.parent.training_stats.copy(),
                                     parameter_names=self.parent.parameter_names,
                                     response_col=response_col,
                                     response_dir=1,
                                     resource_col='resource',
                                     additional_cols=['boots'],
                                    smooth=self.parent.smooth)
                self.recipe.to_pickle(br_train_path)

        elif self.project_from == 'TrainingResults':
            vb_train_path = os.path.join(self.parent.here.checkpoints, 'VirtualBest_train.pkl')
            if os.path.exists(vb_train_path):
                self.vb_train = pd.read_pickle(vb_train_path)
            else:
                response_col = names.param2filename({'Key': self.parent.response_key}, '')
                training_results = self.parent.interp_results[self.parent.interp_results['train'] == 1].copy()
                self.vb_train = training.virtual_best(training_results,\
                                   parameter_names=self.parent.parameter_names,\
                                   response_col=response_col,\
                                   response_dir=1,\
                                   groupby = self.parent.instance_cols,\
                                   resource_col='resource',\
                                    smooth=self.parent.smooth)
                self.vb_train.to_pickle(vb_train_path)

            self.recipe = training.best_recommended(self.vb_train.copy(),
                              parameter_names=self.parent.parameter_names,
                              resource_col='resource',
                              additional_cols=['boots']).reset_index()
        else:
            raise NotImplementedError('Projection from {} has not been implemented'.format(project_from))
        
        # Run the projections
        if os.path.exists(rec_path):
            self.rec_params = pd.read_pickle(rec_path)
        else:
            testing_results = self.parent.interp_results[self.parent.interp_results['train'] == 0].copy()
            self.rec_params = training.evaluate(testing_results,\
                                                self.recipe,\
                                                training.scaled_distance,\
                                                parameter_names=self.parent.parameter_names)
            self.rec_params.to_pickle(rec_path)
    
    def evaluate(self):
        params_df = self.rec_params.loc[:, ['resource'] + self.parent.parameter_names].copy()
        params_df = params_df.groupby('resource').mean()
        params_df.reset_index(inplace=True)
        
        base = names.param2filename({'Key': self.parent.response_key}, '')
        CIlower = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'lower'}, '')
        CIupper = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'upper'}, '')
        eval_df = self.rec_params.copy()
        eval_df.rename(columns = {
            base :'response',
            CIlower :'response_lower',
            CIupper :'response_upper',
        }, inplace=True
        )
        eval_df = eval_df.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        eval_df = eval_df.groupby('resource').median()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df
    
    def evaluate_monotone(self):
        params_df, eval_df = self.evaluate()
        joint = params_df.merge(eval_df, on='resource')
        joint = df_utils.monotone_df(joint, 'resource', 'response', 1)
        params_df = joint.loc[:, ['resource'] + self.parent.parameter_names]
        eval_df = joint.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        return params_df, eval_df
    
class RandomSearchExperiment:
    def __init__(self, parent, rsParams):
        self.parent = parent
        self.name = 'RandomSearch'
        self.rsParams = rsParams
        self.populate()
        
    def populate(self):
        strategy_path = os.path.join(self.parent.here.checkpoints, 'RandomSearch_strategy.pkl')
        eval_train_path = os.path.join(self.parent.here.checkpoints, 'RandomSearch_evalTrain.pkl')
        eval_test_path = os.path.join(self.parent.here.checkpoints, 'RandomSearch_evalTest.pkl')
        if os.path.exists(strategy_path):
            self.strategy = pd.read_pickle(strategy_path)
        else:
            self.strategy, self.eval_train, _ = random_exploration.RandomExploration(self.parent.training_stats, self.rsParams)
            self.strategy.to_pickle(strategy_path)
            self.eval_train.to_pickle(eval_train_path)
        
        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
            self.eval_test = random_exploration.apply_allocations(self.parent.testing_stats.copy(), self.rsParams, self.strategy)
            self.eval_test.to_pickle(eval_test_path)
            
    def evaluate(self):
        params_df = self.eval_test.loc[:, ['TotalBudget'] + self.parent.parameter_names]
        params_df = params_df.groupby('TotalBudget').mean()
        params_df.reset_index(inplace=True)
        params_df.rename(columns = {
            'TotalBudget' : 'resource'}, inplace=True)
        
        base = names.param2filename({'Key': self.parent.response_key,
                                    'Metric': self.parent.stat_params.stats_measures[0].name}, '')
        CIlower = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'lower',
                                        'Metric': self.parent.stat_params.stats_measures[0].name}, '')
        CIupper = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'upper',
                                        'Metric': self.parent.stat_params.stats_measures[0].name}, '')
        eval_df = self.eval_test.copy()
        eval_df.drop('resource', axis=1, inplace=True)
        eval_df.rename(columns = {
            'TotalBudget' : 'resource',
            base :'response',
            CIlower :'response_lower',
            CIupper :'response_upper',
        },inplace=True
        )
        
        eval_df = eval_df.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        eval_df = eval_df.groupby('resource').median()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df
    
    def evaluate_monotone(self):
        params_df, eval_df = self.evaluate()
        joint = params_df.merge(eval_df, on='resource')
        joint = df_utils.monotone_df(joint, 'resource', 'response', 1)
        params_df = joint.loc[:, ['resource'] + self.parent.parameter_names]
        eval_df = joint.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        return params_df, eval_df
    
class SequentialSearchExperiment:
    def __init__(self, parent, ssParams):
        self.parent = parent
        self.name = 'SequentialSearch'
        self.ssParams = ssParams
        self.populate()
        
    def populate(self):
        strategy_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_strategy.pkl')
        eval_train_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_evalTrain.pkl')
        eval_test_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_evalTest.pkl')
        if os.path.exists(strategy_path):
            self.strategy = pd.read_pickle(strategy_path)
            self.eval_train = pd.read_pickle(eval_train_path)
        else:
            training_results = self.parent.interp_results[self.parent.interp_results['train'] == 1].copy()
            self.strategy, self.eval_train, _ = sequential_exploration.SequentialExploration(training_results, self.ssParams, group_on=self.parent.instance_cols)
            self.strategy.to_pickle(strategy_path)
            self.eval_train.to_pickle(eval_train_path)
        
        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
            try:
                testing_results = self.parent.interp_results[self.parent.interp_results['train'] == 0].copy()
                self.eval_test = sequential_exploration.apply_allocations(testing_results,
                                                                          self.ssParams,
                                                                          self.strategy,
                                                                          self.parent.instance_cols)
                self.eval_test.to_pickle(eval_test_path)
            except:
                print('Not enough test data for sequential search. Evaluating on train.')
    
    def evaluate(self):
        if hasattr(self, 'eval_test'):
            params_df = self.eval_test.loc[:, ['TotalBudget'] + self.parent.parameter_names]
            eval_df = self.eval_test.copy()
        else:
            params_df = self.eval_train.loc[:, ['TotalBudget'] + self.parent.parameter_names]
            eval_df = self.eval_train.copy()
        
        
        for col in params_df.columns:
            if params_df[col].dtype == 'object':
                params_df.loc[:, col] = params_df.loc[:, col].astype(float)

        temp = params_df.groupby('TotalBudget').mean()
        params_df.reset_index(inplace=True)
        params_df.rename(columns = {
            'TotalBudget' : 'resource'}, inplace=True)
        base = names.param2filename({'Key': self.parent.response_key}, '')
        CIlower = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'lower'}, '')
        CIupper = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'upper'}, '')
        
        eval_df.drop('resource', axis=1, inplace=True)
        eval_df.rename(columns = {
            'TotalBudget' : 'resource',
            base :'response',
            CIlower :'response_lower',
            CIupper :'response_upper',
        },inplace=True
        )
        
        eval_df = eval_df.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        eval_df = eval_df.groupby('resource').median()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df
    
    def evaluate_monotone(self):
        params_df, eval_df = self.evaluate()
        joint = params_df.merge(eval_df, on='resource')
        joint = df_utils.monotone_df(joint, 'resource', 'response', 1)
        params_df = joint.loc[:, ['resource'] + self.parent.parameter_names]
        eval_df = joint.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        return params_df, eval_df
        
class VirtualBestBaseline:
    def __init__(self, parent):
        self.parent = parent
        self.name = 'VirtualBest'
        self.populate()
        
    def savename(self):
        return os.path.join(self.parent.here.checkpoints, 'VirtualBest_test.pkl')
    
    def populate(self):
        if os.path.exists(self.savename()):
            self.rec_params = pd.read_pickle(self.savename())
        else:
            response_col = names.param2filename({'Key': self.parent.response_key}, '')
            testing_results = self.parent.interp_results[self.parent.interp_results['train'] == 0].copy()
            self.rec_params = training.virtual_best(testing_results,\
                               parameter_names=self.parent.parameter_names,\
                               response_col=response_col,\
                               response_dir=1,\
                               groupby = self.parent.instance_cols,\
                               resource_col='resource',\
                                smooth=self.parent.smooth)
            self.rec_params.to_pickle(self.savename())
                
    def evaluate(self):
        params_df = self.rec_params.loc[:, ['resource'] + self.parent.parameter_names]
        params_df = params_df.groupby('resource').mean()
        
        base = names.param2filename({'Key': self.parent.response_key}, '')
        CIlower = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'lower'}, '')
        CIupper = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'upper'}, '')
        eval_df = self.rec_params.copy()
        eval_df.rename(columns = {
            base :'response',
            CIlower :'response_lower',
            CIupper :'response_upper',
        },inplace=True
        )
        
        eval_df = eval_df.loc[:, ['resource', 'response']]
        eval_df = eval_df.groupby('resource').median()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df

class Plotting:
    def __init__(self, parent):
        self.parent = parent
        self.colors = ['blue', 'green', 'red', 'purple']
        self.assign_colors()
        self.xscale='log'
    
    def set_colors(self, cp):
        self.colors = cp
        self.assign_colors()
        
    def set_xlims(self, xlims):
        self.xlims = xlims
    
    def make_legend(self, ax):
        color_patches = [mpatches.Patch(color=self.parent.baseline.color, label=self.parent.baseline.name)]
        color_patches = color_patches + [mpatches.Patch(color=experiment.color, label=experiment.name)
                        for experiment in self.parent.experiments]
        ax.legend(handles=[cpatch for cpatch in color_patches])
    
    def apply_shared(self, p):
        if type(p) is dict:
            for k, v in p.items():
                p[k] = self.apply_shared(v)
            return p
            
        p = p.scale(x=self.xscale)
        if hasattr(self, 'xlims'):
            p = p.limit(x=self.xlims)
        
        fig = plt.figure()
        p = p.on(fig).plot()
        ax = fig.axes[0]
        self.make_legend(ax)
            
        return fig
        
    def assign_colors(self):
        self.parent.baseline.color = 'black'
        for idx, experiment in enumerate(self.parent.experiments):
            experiment.color = self.colors[idx]
    
    def plot_parameters(self):
        params_df,_ = self.parent.baseline.evaluate()
        p = {}
        for param in self.parent.parameter_names:
            p[param] = (so.Plot(data=params_df, x='resource', y=param)
                        .add(so.Line(color = self.parent.baseline.color, marker='x'))
                       )
        for experiment in self.parent.experiments:
            params_df, _ = experiment.evaluate_monotone()
            for param in self.parent.parameter_names:
                p[param] = (p[param].add(so.Line(color=experiment.color, marker='x'),
                                         data=params_df, x='resource', y=param)
                            .scale(x='log'))
        p = self.apply_shared(p)
            
        return p
    
    def plot_parameters_distance(self):
        recipes,_ = self.parent.baseline.evaluate()

        all_params_list = []
        count = 0
        for experiment in self.parent.experiments:
            params_df, _ = experiment.evaluate_monotone()
            params_df['exp_idx'] = count
            all_params_list.append(params_df)
            count += 1
        
        all_params = pd.concat(all_params_list, ignore_index=True)
        dist_params_list = []

        for _, recipe in recipes.reset_index().iterrows():
            res_df = all_params[all_params['resource'] == recipe['resource']]
            temp_df_eval = training.scaled_distance(res_df, recipe, self.parent.parameter_names)
            temp_df_eval.loc[:,'resource'] = recipe['resource']
            dist_params_list.append(temp_df_eval)
        all_params = pd.concat(dist_params_list, ignore_index=True)
        
        p = so.Plot(data=all_params, x='resource', y='distance_scaled')
        for idx, experiment in enumerate(self.parent.experiments):
            params_df = all_params[all_params['exp_idx'] == idx]
            p = (p.add(so.Line(color=experiment.color, marker='x'),
                      data=params_df, x='resource', y='distance_scaled'))

        p = self.apply_shared(p)
        
        return p
    
    def plot_performance(self):
        _, eval_df = self.parent.baseline.evaluate()
        eval_df = df_utils.monotone_df(eval_df, 'resource', 'response', 1)
        p = (so.Plot(data=eval_df, x='resource', y='response')
             .add(so.Line(color = self.parent.baseline.color, marker="x"))
            )
        
        for experiment in self.parent.experiments:
            _, eval_df = experiment.evaluate_monotone()
            p = (p.add(so.Line(color=experiment.color, marker="x"), data=eval_df, x='resource', y='response')
                 .add(so.Band(alpha=0.2, color=experiment.color), data=eval_df, x='resource',
                      ymin='response_lower', ymax='response_upper')
                )

#         _, eval_df = self.parent.baseline.evaluate()
#         eval_df = df_utils.monotone_df(eval_df, 'resource', 'response', 1)
#         eval_df['Experiment'] = self.parent.baseline.name
        
#         p = (so.Plot(data=eval_df, x='resource', y='response')
#              .add(so.Line(color = self.parent.baseline.color, marker="x"))
#             )
        
#         joint_list = []
#         for experiment in self.parent.experiments:
#             _, eval_df = experiment.evaluate_monotone()
#             eval_df['Experiment'] = experiment.name
#             joint_list.append(eval_df)
            
#         joint = pd.concat(joint_list, ignore_index=True)
#         p = sns.lineplot(data=joint, x='resource', y='response', hue='Experiment')
#             p = (p.add(so.Line(color=experiment.color, marker="x"), data=eval_df, x='resource', y='response')
#                  .add(so.Band(alpha=0.2, color=experiment.color), data=eval_df, x='resource',
#                       ymin='response_lower', ymax='response_upper')
#                 )
            
        
        
        p = self.apply_shared(p)
        return p

    def plot_random_metaparams(self, experiment):
        metaparams = ['ExploreFrac','tau']
        df = experiment.train_exp_at_best.copy()
        df['ExploreFrac'] = df['ExplorationBudget'] / df['TotalBudget']
        
        fig = plt.figure()
        sfigs = fig.subfigures(1, len(metaparams),facecolor='White')
        p = {}
        for idx, param in enumerate(metaparams):
            p[idx] = (
                so.Plot(data=df, x='TotalBudget', y=param)
                .scale(x="log")
                .add(so.Line(color=experiment.color, marker='x'), so.Agg())
            )
            figname = os.path.join(self.here.plots, 'metaparam_{}.pdf'.format(param))
            
            p[idx].show()
            p[idx].save(figname)
        return p
        
class stochastic_benchmark:
    def __init__(self, 
                 parameter_names,
                 here=os.getcwd(),
                 instance_cols=['instance'],
                 bsParams_iter = prepare_bootstrap(),
                 stat_params = stats.StatsParameters(stats_measures=[stats.Median()]),
                 resource_fcn = sweep_boots_resource,
                 response_key = 'PerfRatio',
                 train_test_split = 0.5,
                 recover=True,
                smooth=True,
                baseline = 'VirtualBest',
                experiments = ['Projection', 'RandomSearch', 'SequentialSearch']):
        
        self.here = names.paths(here)
        self.parameter_names = parameter_names
        self.instance_cols = instance_cols
        self.bsParams_iter = bsParams_iter
        self.stat_params = stat_params
        self.resource_fcn = resource_fcn
        self.response_key = response_key
        self.train_test_split = train_test_split
        self.recover=recover
        self.smooth = smooth
        
        ## Dataframes needed for experiments and baselines
        self.bs_results = None
        self.interp_results = None
        self.training_stats = None
        self.testing_stats = None
        
        self.experiments = []
        
        #Recursive file recovery 
        while any([v is None for v in [self.interp_results, self.training_stats, self.testing_stats]]):
            self.populate_training_stats()
            self.populate_testing_stats()
            self.populate_interp_results()
            self.populate_bs_results()
        
    def populate_training_stats(self):
        if self.training_stats is None:
            if os.path.exists(self.here.training_stats) and self.recover:
                self.training_stats = pd.read_pickle(self.here.training_stats)
            elif self.interp_results is not None:
                training_results = self.interp_results[self.interp_results['train'] == 1]
                print('Computing training stats')
                self.training_stats = stats.Stats(training_results, self.stat_params,
                                             self.parameter_names + ['boots', 'resource'])
                self.training_stats.to_pickle(self.here.training_stats)
                
    def populate_testing_stats(self):
        if self.testing_stats is None:
            if os.path.exists(self.here.testing_stats) and self.recover:
                self.testing_stats = pd.read_pickle(self.here.testing_stats)
            elif self.interp_results is not None:
                testing_results = self.interp_results[self.interp_results['train'] == 0]
                print('Computing testing stats')
                self.testing_stats = stats.Stats(testing_results, self.stat_params,
                                             self.parameter_names + ['boots', 'resource'])
                self.testing_stats.to_pickle(self.here.testing_stats)
            
    def populate_interp_results(self):
        if self.interp_results is None:
            if os.path.exists(self.here.interpolate) and self.recover:
                self.interp_results = pd.read_pickle(self.here.interpolate)
                if 'train' not in self.interp_results.columns:
                    self.interp_results = training.split_train_test(self.interp_results, self.instance_cols, train_test_split)
                    self.interp_results.to_pickle(self.here.interpolate)

            elif self.bs_results is not None:
                iParams = interpolate.InterpolationParameters(self.resource_fcn,
                                                              parameters=self.parameter_names)
                print('Interpolating results with parameters: ', iParams)
                self.interp_results = interpolate.Interpolate(self.bs_results,
                                                              iParams, self.parameter_names+self.instance_cols)
                self.interp_results.to_pickle(self.here.interpolate)
                self.interp_results = training.split_train_test(self.interp_results, self.instance_cols, self.train_test_split)
                self.interp_results.to_pickle(self.here.interpolate)
    
    def populate_bs_results(self):
        if self.bs_results is None:
            if os.path.exists(self.here.bootstrap) and self.recover:
                self.bs_results = pd.read_pickle(self.here.bootstrap)
            else:
                group_on = self.parameter_names + self.instance_cols
                if not hasattr(self, 'raw_data'):
                    self.raw_data = df_utils.read_exp_raw(self.here.raw_data)
                self.bs_results = bootstrap.Bootstrap(self.raw_data, group_on, self.bsParams_iter)
                self.bs_results.to_pickle(self.here.bootstrap)
            
    def run_baseline(self):
        self.baseline = VirtualBestBaseline(self)
    def run_ProjectionExperiment(self, project_from):
        self.experiments.append(ProjectionExperiment(self, project_from))
    def run_RandomSearchExperiment(self, rsParams):
        self.experiments.append(RandomSearchExperiment(self, rsParams))
    def run_SequentialSearchExperiment(self, ssParams):
        self.experiments.append(SequentialSearchExperiment(self, ssParams))
    def initPlotting(self):
        self.plots = Plotting(self)
    