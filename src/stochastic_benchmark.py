from collections import namedtuple
import copy
import glob
from math import floor
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
from random import choice
import seaborn.objects as so
import seaborn as sns
import warnings


import bootstrap
import df_utils
import interpolate
from plotting import *
import random_exploration
import sequential_exploration
import stats
import success_metrics
import training
import names
import utils_ws

median = True



def default_bootstrap(nboots = 1000, 
                      response_col = names.param2filename({'Key': 'MinEnergy'}, ''),
                      resource_col = names.param2filename({'Key': 'MeanTime'}, '')):
    """
    Default bootstrapping parameters
    """
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
    """
    Default resource computation - resource = sweeps * boots
    """
    return df['sweep'] * df['boots']

class Experiment:
    def __init__(self):
        return 
    def evaluate(self):
       raise NotImplementedError(
            "Evaluate should be overriden by a subclass of Experiment") 
    def evaluate_monotone(self):
        """
        Monotonizes the response and parameters from evaluate
        
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        res = self.evaluate()
        if len(res) == 2:
            params_df, eval_df = res
        elif len(res) == 3:
            params_df, eval_df, preproc_params = res
        joint = params_df.merge(eval_df, on='resource')
        joint = df_utils.monotone_df(joint, 'resource', 'response', 1)
        params_df = joint.loc[:, ['resource'] + self.parent.parameter_names]
        eval_df = joint.loc[:, ['resource','response', 'response_lower', 'response_upper']]

        if len(res) == 2:
            return params_df, eval_df
        elif len(res) == 3:
            return params_df, eval_df, preproc_params

class ProjectionExperiment(Experiment):
    """
    Holds information needed for projection experiments. 
    Used for evaluating performance of a recipe on the test set if the user cannot re-run experiments.
    Recipes can be post-processed by a user-defined function (e.g., smoothed fit) and queried for running
    evaluatation experiments.
    
    Attributes
    ----------
    parent : stochatic_benchmark
    name : str
        name for pretty printing
    project_from : str 
        'TrainingStats' or 'TrainingResults'
    recipe : pd.DataFrame
        Recommended parameters for each resource (can be postprocessed). This is not projected
    rec_params : pd.DataFrame
        Projected recommended parameters
    """
    def __init__(self, parent, project_from, postprocess=None, postprocess_name=None):
        self.parent = parent
        self.name = 'Projection from {}'.format(project_from)
        self.project_from = project_from
        self.postprocess = postprocess
        self.postprocess_name = postprocess_name
        self.populate()

        
    def populate(self):
        """
        Adds recipe depending on source. Currently only projection from the best recommended from the training stats or results are available.
        Any addition recipe specifications should be implemented here
        """
        if self.postprocess is not None:
            rec_path = os.path.join(self.parent.here.checkpoints, 'Projection_from={}_postprocess={}.pkl'.format(self.project_from, self.postprocess_name))
        else:
            rec_path = os.path.join(self.parent.here.checkpoints, 'Projection_from={}.pkl'.format(self.project_from))
        
        # Prepare the recipes
        if self.project_from == 'TrainingStats':
            if self.postprocess is not None:
                br_train_path_post = os.path.join(self.parent.here.checkpoints, 'BestRecommended_train_postprocess={}.pkl'.format(self.postprocess_name))
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

            if self.postprocess is not None:
                self.preproc_recipe = self.recipe.copy()
                self.recipe = self.postprocess(self.recipe)
                self.recipe.to_pickle(br_train_path_post)
                    
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
            
            if self.postprocess is not None:
                self.preproc_recipe = self.recipe.copy()
                self.recipe = self.postprocess(self.recipe)
                    
        else:
            raise NotImplementedError('Projection from {} has not been implemented'.format(self.project_from))
        
        # Run the projections
        if os.path.exists(rec_path):
            self.rec_params = pd.read_pickle(rec_path)
        else:
            print('Evaluating recommended parameters on testing results')
            testing_results = self.parent.interp_results[self.parent.interp_results['train'] == 0].copy()
            self.rec_params = training.evaluate(testing_results,\
                                                self.recipe,\
                                                training.scaled_distance,\
                                                parameter_names=self.parent.parameter_names,\
                                                group_on = self.parent.instance_cols)
            self.rec_params.to_pickle(rec_path)
        
    def evaluate(self, monotone=False):
        """
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        # params_df = self.rec_params.loc[:, ['resource'] + self.parent.parameter_names].copy()
        
        # base = names.param2filename({'Key': self.parent.response_key}, '')
        # CIlower = names.param2filename({'Key': self.parent.response_key,
        #                                 'ConfInt':'lower'}, '')
        # CIupper = names.param2filename({'Key': self.parent.response_key,
        #                                 'ConfInt':'upper'}, '')
        # eval_df = self.rec_params.copy()
        # eval_df.rename(columns = {
        #     base :'response',
        #     CIlower :'response_lower',
        #     CIupper :'response_upper',
        # }, inplace=True
        # )

        # joint = self.rec_params.copy()
        # base = names.param2filename({'Key': self.parent.response_key}, '')
        # CIlower = names.param2filename({'Key': self.parent.response_key,
        #                                 'ConfInt':'lower'}, '')
        # CIupper = names.param2filename({'Key': self.parent.response_key,
        #                                 'ConfInt':'upper'}, '')
        # joint.rename(columns = {
        #     base :'response',
        #     CIlower :'response_lower',
        #     CIupper :'response_upper',
        # }, inplace=True
        # )
        # joint = joint.loc[:, ['resource'] + self.parent.parameter_names + 
        # ['response', 'response_lower', 'response_upper'] + self.parent.instance_cols] 

        # extrapolate_from = self.parent.interp_results.loc[self.parent.interp_results['train'] == 0].copy()
        # extrapolate_from.rename(columns = {
        #     base :'response',
        #     CIlower :'response_lower',
        #     CIupper :'response_upper',
        # }, inplace=True
        # )

        # def mono(df):
        #     # res = df_utils.monotone_df(joint, 'resource', 'response', 1,
        #     # extrapolate_from=extrapolate_from, match_on = self.parent.parameter_names + self.parent.instance_cols)
        #     res = df_utils.monotone_df(joint, 'resource', 'response', 1)
        #     return res

        # joint = joint.groupby(self.parent.instance_cols).apply(mono)
        

        # params_df = joint.loc[:, ['resource'] + self.parent.parameter_names]
        # eval_df = joint.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        # params_df = params_df.groupby('resource').mean()
        # params_df.reset_index(inplace=True)

        # eval_df = eval_df.groupby('resource').median()
        # eval_df.reset_index(inplace=True)




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
        if median:
            eval_df = eval_df.groupby('resource').median()
        else:
            eval_df = eval_df.groupby('resource').mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df
    
    def evaluate_monotone(self):
        """
        Monotonizes the response and parameters from evaluate
        
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df, eval_df = self.evaluate()
    
        joint = params_df.merge(eval_df, on='resource')
        extrapolate_from=self.parent.testing_stats.copy()
        base = names.param2filename({'Key': self.parent.response_key,
                                    'Metric': self.parent.stat_params.stats_measures[0].name}, '')
        CIlower = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'lower',
                                        'Metric': self.parent.stat_params.stats_measures[0].name}, '')
        CIupper = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'upper',
                                        'Metric': self.parent.stat_params.stats_measures[0].name}, '')
        extrapolate_from.rename(columns = {
            base :'response',
            CIlower :'response_lower',
            CIupper :'response_upper',
        },inplace=True
        )

        # joint = df_utils.monotone_df(joint, 'resource', 'response', 1,
        #     extrapolate_from=extrapolate_from, match_on = self.parent.parameter_names)
        joint = df_utils.monotone_df(joint, 'resource', 'response', 1)
        params_df = joint.loc[:, ['resource'] + self.parent.parameter_names]
        eval_df = joint.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        return params_df, eval_df
    

class StaticRecommendationExperiment(Experiment):
    """
    Holds parameters for fixed recommendation experiments
    
    Attributes
    ----------
    parent : stochastic_benchmark
    name : str 
        name for pretty printing
    rec_params : pd.DataFrame
        Recommended parameters for evaluation
    """
    
    def __init__(self, parent, init_from):
        self.parent = parent
        self.name = 'FixedRecommendation'
        
        if type(init_from) == ProjectionExperiment:
            self.rec_params = init_from.recipe
            if init_from.postprocess is not None:
                self.preproc_rec_params = init_from.preproc_recipe.copy()

        elif type(init_from) == pd.DataFrame:
            self.rec_params = init_from
        else:
            warn_str = 'init_from type is not supported. No recommended parameters are set.'
            warnings.warn(warn_str)
        
    def list_runs(self):
        """
        Returns a list of experiments evaluate.
        """
        parameter_names = "resource " + ' '.join(self.parent.parameter_names)
        Parameter = namedtuple("Parameter", parameter_names)
        runs = []
        for _, row in self.rec_params.iterrows():
            runs.append(Parameter(row['resource'], *[row[k] for k in self.parent.parameter_names]))
        return runs
    
    def attach_runs(self, df, process=True):
        """
        Attaches reruns of experiment to 
        """
        if type(df) == str:
            df = pd.read_pickle(df)
        if process:
            group_on = self.parent.instance_cols + ['resource']
            self.eval_df = self.parent.evaluate_without_bootstrap(df, group_on)
        else:
            self.eval_df = df
        self.parent.baseline.recalibrate(self.eval_df)
    
    def evaluate(self):
        """
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df = self.rec_params.loc[:, ['resource'] + self.parent.parameter_names].copy()
        preproc_params = self.preproc_rec_params.loc[:, ['resource'] + self.parent.parameter_names].copy()
        # params_df = params_df.groupby('resource').mean()
        # params_df.reset_index(inplace=True)
        
        base = names.param2filename({'Key': self.parent.response_key}, '')
        CIlower = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'lower'}, '')
        CIupper = names.param2filename({'Key': self.parent.response_key,
                                        'ConfInt':'upper'}, '')
        eval_df = self.eval_df.copy()
        eval_df.rename(columns = {
            base :'response',
            CIlower :'response_lower',
            CIupper :'response_upper',
        }, inplace=True
        )
        eval_df = eval_df.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        eval_df = eval_df.groupby('resource').mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df, preproc_params
    
    
class RandomSearchExperiment(Experiment):
    """
    Holds parameters needed for random search experiment
    
    Attributes
    ----------
    parent : stochatic_benchmark
    name : str
        name for pretty printing
    meta_params : pd.DataFrame
        Best metaparameters (Exploration budget and Tau) 
    eval_train : pd.DataFrame
        Resulting parameters of meta_params on training set
    eval_test : pd.DataFrame
        Resulting parameters of meta_params on testing set    
    """
    def __init__(self, parent, rsParams, postprocess=None, postprocess_name=None):
        self.parent = parent
        self.name = 'RandomSearch'
        self.rsParams = rsParams
        self.meta_parameter_names = ['ExploreFrac', 'tau']
        self.resource = 'TotalBudget'
        self.postprocess = postprocess
        self.postprocess_name = postprocess_name
        self.populate()
        
    def populate(self):
        """
        Populates meta_params, eval_train, eval_test 
        """
        meta_params_path = os.path.join(self.parent.here.checkpoints, 'RandomSearch_meta_params.pkl')
        eval_train_path = os.path.join(self.parent.here.checkpoints, 'RandomSearch_evalTrain.pkl')
        
        if self.postprocess is None:
            eval_test_path = os.path.join(self.parent.here.checkpoints, 'RandomSearch_evalTest.pkl')
        else:
            eval_test_path = os.path.join(self.parent.here.checkpoints, 
                'RandomSearch_evalTest_postprocess={}.pkl'.format(self.postprocess_name)) 

        if os.path.exists(meta_params_path):
            self.meta_params = pd.read_pickle(meta_params_path)
        else:
            self.meta_params, self.eval_train, _ = random_exploration.RandomExploration(self.parent.training_stats, self.rsParams)
            self.meta_params.to_pickle(meta_params_path)
            self.eval_train.to_pickle(eval_train_path)
        self.meta_params['ExploreFrac'] = self.meta_params['ExplorationBudget'] / self.meta_params['TotalBudget']

        if self.postprocess is not None:
            self.preproc_meta_params = self.meta_params.copy()
            self.meta_params = self.postprocess(self.meta_params)
        
        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
            print('\t Evaluating random search on test')
            self.eval_test = random_exploration.apply_allocations(self.parent.testing_stats.copy(), self.rsParams, self.meta_params)
            self.eval_test.to_pickle(eval_test_path)
            
    def evaluate(self):
        """
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
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
        if median:
            eval_df = eval_df.groupby('resource').median()
        else:
            eval_df = eval_df.groupby('resource').mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df
    
    
class SequentialSearchExperiment(Experiment):
    """
    Holds parameters needed for sequential search experiment
    
    Attributes
    ----------
    parent : stochatic_benchmark
    name : str
        name for pretty printing
    meta_params : pd.DataFrame
        Best metaparameters (Exploration budget and Tau) 
    eval_train : pd.DataFrame4
        Resulting parameters of meta_params on training set
    eval_test : pd.DataFrame
        Resulting parameters of meta_params on testing set    
    """
    def __init__(self, parent, ssParams, id_name=None, postprocess=None, postprocess_name=None):
        self.parent = parent
        if id_name is None:
            self.name = 'SequentialSearch'
        else:
            self.name = 'SequentialSearch_{}'.format(id_name)
        self.ssParams = ssParams
        self.id_name = id_name
        self.meta_parameter_names = ['ExploreFrac', 'tau']
        self.resource = 'TotalBudget'
        self.postprocess = postprocess
        self.postprocess_name = postprocess_name
        self.populate()
        
    def populate(self):
        if self.id_name is None:
            meta_params_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_meta_params.pkl')
            eval_train_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_evalTrain.pkl')
            if self.postprocess is None:
                eval_test_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_evalTest.pkl')
            else:
                eval_test_path = os.path.join(self.parent.here.checkpoints, 
                    'SequentialSearch_evalTest_postprocess={}.pkl'.format(self.postprocess_name)) 
        else:
            meta_params_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_meta_params_id={}.pkl'.format(self.id_name))
            eval_train_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_evalTrain_id={}.pkl'.format(self.id_name))
            if self.postprocess is None:
                eval_test_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_evalTest_id={}.pkl'.format(self.id_name))    
            else:
                eval_test_path = os.path.join(self.parent.here.checkpoints, 
                    'SequentialSearch_evalTest_id={}_postprocess={}.pkl'.format(self.id_name, self.postprocess_name))

        if os.path.exists(meta_params_path):
            self.meta_params = pd.read_pickle(meta_params_path)
            self.eval_train = pd.read_pickle(eval_train_path)
        else:
            training_results = self.parent.interp_results[self.parent.interp_results['train'] == 1].copy()
            self.meta_params, self.eval_train, _ = sequential_exploration.SequentialExploration(training_results, self.ssParams, group_on=self.parent.instance_cols)
            self.meta_params.to_pickle(meta_params_path)
            self.eval_train.to_pickle(eval_train_path)
        self.meta_params['ExploreFrac'] = self.meta_params['ExplorationBudget'] / self.meta_params['TotalBudget']
        if self.postprocess is not None:
            self.preproc_meta_params = self.meta_params.copy()
            self.meta_params = self.postprocess(self.meta_params)
        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
            # try:
            print('\t Evaluating sequential search on test')
            testing_results = self.parent.interp_results[self.parent.interp_results['train'] == 0].copy()
            self.eval_test = sequential_exploration.apply_allocations(testing_results,
                                                                        self.ssParams,
                                                                        self.meta_params,
                                                                        self.parent.instance_cols)
            self.eval_test.to_pickle(eval_test_path)
            # except:
            #     print('Not enough test data for sequential search. Evaluating on train.')
    
    def evaluate(self):
        """
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
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
        if median:
            eval_df = eval_df.groupby('resource').median()
        else:
            eval_df = eval_df.groupby('resource').mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df
    
        
class VirtualBestBaseline:
    """
    Calculates virtual best on an instance by instance basis
    
    Attributes
    ----------
    parent : stochatic_benchmark
    name : str
        name for pretty printing
    rec_params : pd.DataFrame
        Dataframe of best paremeters per instance and resource level
    
    """
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
                               response_dir=self.parent.response_dir,\
                               groupby = self.parent.instance_cols,\
                               resource_col='resource',\
                                smooth=self.parent.smooth)
            self.rec_params.to_pickle(self.savename())
                
    def recalibrate(self, new_df):
        """
        Parameters
        ----------
        new_df : pd.DataFrame
            pandas dataframe with the new data. Should only have columns 
            ['resource'. *(parameters_names), response, response_lower, response_upper]
            response cols should match name of results columns
        Updates params and evaluation to take in new data
        """
        base = names.param2filename({'Key': self.parent.response_key}, '')
        joint_cols = ['resource', base] + self.parent.parameter_names + self.parent.instance_cols
        new_df = new_df.loc[:, joint_cols]
        joint = pd.concat([self.rec_params.loc[:, joint_cols], new_df], ignore_index=True)
        
        self.rec_params = training.virtual_best(joint,\
                               parameter_names=self.parent.parameter_names,\
                               response_col=base,\
                               response_dir=self.parent.response_dir,\
                               groupby = self.parent.instance_cols,\
                               resource_col='resource',\
                                additional_cols=[],\
                                smooth=self.parent.smooth)

    def evaluate(self):
        """
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
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
        if median:
            eval_df = eval_df.groupby('resource').median()
        else:
            eval_df = eval_df.groupby('resource').mean()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df

        
class stochastic_benchmark:
    """
    Attributes
    ----------
    parameter_names : list[str]
        list of parameter names
    here : str
        path to parent directory of data
    instance_cols : list[str]
        Columns that define an instance. i.e., datapoints that match on all of these cols are the same instance
    bsParams_iter : 
        Iterator that yields bootstrap parameters
    iParams : 
        Interpolation parameters
    stat_params : stats.StatsParameters
        Parameters for computing stats dataframes
    resource_fcn : callable(pd.DataFrame)
        Function that writes a 'resource' function depending on dataframe parameters
    response_key : str
        Column that we want to optimize
    train_test_split : float
        Fraction of instances that should be training data
    recover : bool
        Whether dataframes should be recovered where available or generated from scratch
    """
    def __init__(self, 
                 parameter_names,
                 here=os.getcwd(),
                 instance_cols=['instance'],
                 bsParams_iter = default_bootstrap(),
                 iParams = None,
                 stat_params = stats.StatsParameters(stats_measures=[stats.Median()]),
                 resource_fcn = sweep_boots_resource,
                 response_key = 'PerfRatio',
                 response_dir = 1,
                 train_test_split = 0.5,
                 recover=True,
                 reduce_mem=True,
                 group_name_fcn=None,
                smooth=True):
        
        self.here = names.paths(here)
        self.parameter_names = parameter_names
        self.instance_cols = instance_cols
        self.bsParams_iter = bsParams_iter
        self.stat_params = stat_params
        self.resource_fcn = resource_fcn
        self.response_key = response_key
        self.response_dir = response_dir
        self.train_test_split = train_test_split
        self.recover = recover
        self.reduce_mem = reduce_mem
        self.group_name_fcn = group_name_fcn
        self.smooth = smooth
        
        if iParams is None:
            self.iParams = interpolate.InterpolationParameters(self.resource_fcn,
                                                  parameters=self.parameter_names)
        else:
            self.iParams = iParams
        
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
            # self.populate_bs_results()
        
        
    def populate_training_stats(self):
        """
        Tries to recover or computes training stats
        """
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
        """
        Tries to recover or computes testing stats
        """
        if self.testing_stats is None:
            if os.path.exists(self.here.testing_stats) and self.recover:
                self.testing_stats = pd.read_pickle(self.here.testing_stats)
                
            elif self.interp_results is not None:
                testing_results = self.interp_results[self.interp_results['train'] == 0]
                print('Computing testing stats')
                if len(testing_results) ==0:
                    self.testing_stats = pd.DataFrame()
                    
                else:
                    self.testing_stats = stats.Stats(testing_results, self.stat_params,
                                                 self.parameter_names + ['boots', 'resource'])
                    self.testing_stats.to_pickle(self.here.testing_stats)
            
    def populate_interp_results(self):
        """
        Tries to recover or computes interpolated results
        """
        if self.interp_results is None:
            if os.path.exists(self.here.interpolate) and self.recover:
                self.interp_results = pd.read_pickle(self.here.interpolate)
                if 'train' not in self.interp_results.columns:
                    self.interp_results = training.split_train_test(self.interp_results, self.instance_cols, self.train_test_split)
                    self.interp_results.to_pickle(self.here.interpolate)

            elif self.bs_results is not None:
                print(self.bs_results)
                if self.reduce_mem:
                    print('Interpolating results with parameters: ', self.iParams)
                    self.interp_results = interpolate.Interpolate_reduce_mem(self.bs_results,
                                                                self.iParams, self.parameter_names+self.instance_cols)
                else:
                    print('Interpolating results with parameters: ', self.iParams)
                    self.interp_results = interpolate.Interpolate(self.bs_results,
                                                                self.iParams, self.parameter_names+self.instance_cols)
                    
                base = names.param2filename({'Key': self.response_key}, '')
                CIlower = names.param2filename({'Key': self.response_key,
                                                'ConfInt':'lower'}, '')
                CIupper = names.param2filename({'Key': self.response_key,
                                                'ConfInt':'upper'}, '')
                self.interp_results.dropna(subset=[base, CIlower, CIupper], inplace=True)

                self.interp_results = training.split_train_test(self.interp_results, self.instance_cols, self.train_test_split)
                self.interp_results.to_pickle(self.here.interpolate)
                self.bs_results = None
            else:
                self.populate_bs_results(self.group_name_fcn)
    
    def populate_bs_results(self, group_name_fcn=None):
        """
        Tries to recover or computes bootstrapped results
        """

        if self.bs_results is None:
            if self.reduce_mem:
                def raw2bs_names(raw_filename):
                    group_name = group_name_fcn(raw_filename)
                    bs_filename = os.path.join(self.here.checkpoints, 'bootstrapped_results_{}.pkl'.format(group_name))
                    return bs_filename
                    
                self.raw_data = glob.glob(os.path.join(self.here.raw_data, '*.pkl'))
                bs_names = [raw2bs_names(raw_file) for raw_file in self.raw_data]

                if all([os.path.exists(bs_name) for bs_name in bs_names]) and len(bs_names) > 1 and self.recover:
                    print('Reading bootstrapped results')
                    self.bs_results = bs_names
                else:
                    group_on = self.parameter_names + self.instance_cols
                    if not hasattr(self, 'raw_data'):
                        print('Running bootstrapped results')
                        self.raw_data = glob.glob(os.path.join(self.here.raw_data, '*.pkl'))
                    self.bs_results = bootstrap.Bootstrap_reduce_mem(self.raw_data, group_on, self.bsParams_iter, self.here.checkpoints, group_name_fcn)
            
            else:
                if os.path.exists(self.here.bootstrap) and self.recover:
                    print('Reading bootstrapped results')
                    self.bs_results = pd.read_pickle(self.here.bootstrap)
                else:
                    print('Running bootstrapped results')
                    group_on = self.parameter_names + self.instance_cols
                    if not hasattr(self, 'raw_data'): 
                        self.raw_data = df_utils.read_exp_raw(self.here.raw_data)
                    
                    progress_dir = os.path.join(self.here.progress, 'bootstrap/')
                    if not os.path.exists(progress_dir):
                        os.makedirs(progress_dir)
                    
                    self.bs_results = bootstrap.Bootstrap(self.raw_data, group_on, self.bsParams_iter, progress_dir)
                    self.bs_results.to_pickle(self.here.bootstrap)
    
 
    def evaluate_without_bootstrap(self, df, group_on):
        """"
        Runs same computations evaluations as bootstrap without bootstrapping
        """
        bs_params = next(self.bsParams_iter)
        resource_col = bs_params.shared_args['resource_col']
        response_col = bs_params.shared_args['response_col'] 
        agg = bs_params.agg
        
        def evaluate_single(df_single):
            bs_params.update_rule(bs_params, df_single)
            resources = df_single[resource_col].values
            responses = df_single[response_col].values
            resources = np.repeat(resources, df_single[agg])
            responses = np.repeat(responses, df_single[agg])

            bs_df = pd.DataFrame()
            for metric_ref in bs_params.success_metrics:
                metric = metric_ref(bs_params.shared_args,bs_params.metric_args[metric_ref.__name__])
                metric.evaluate(bs_df, responses, resources)
            for col in bs_params.keep_cols:
                if col in df_single.columns:
                    val = df_single[col].iloc[0]
                    bs_df[col] = val
            
            return bs_df
        full_eval = df.groupby(group_on).apply(lambda df : evaluate_single(df)).reset_index()
        full_eval.drop(columns = ['level_{}'.format(len(group_on))], inplace=True)
        return full_eval
            
    def run_baseline(self):
        """
        Adds virtual best baseline
        """
        print('Runnng baseline')
        self.baseline = VirtualBestBaseline(self)
    def run_ProjectionExperiment(self, project_from, postprocess=None, postprocess_name=None):
        """
        Runs projections experiments
        """
        print("Running projection experiment")
        self.experiments.append(ProjectionExperiment(self, project_from, postprocess, postprocess_name))
    def run_RandomSearchExperiment(self, rsParams, postprocess=None, postprocess_name=None):
        """
        Runs random search experiments
        """
        print('Running random search experiment')
        self.experiments.append(RandomSearchExperiment(self, rsParams, 
            postprocess=postprocess, postprocess_name=postprocess_name))
    def run_SequentialSearchExperiment(self, ssParams, id_name=None, postprocess=None, postprocess_name=None):
        """
        Runs sequential search experiments
        """
        print('Running sequential search experiment')
        self.experiments.append(SequentialSearchExperiment(self, ssParams, 
            id_name, postprocess=postprocess, postprocess_name=postprocess_name))
    def run_StaticRecommendationExperiment(self, init_from):
        """
        Runs static recommendation experiments
        """
        print('Running static recommendation experiment')
        self.experiments.append(StaticRecommendationExperiment(self, init_from))
    def initPlotting(self):
        """
        Sets up plotting - this should be run after all experiments are run
        """
        self.plots = Plotting(self)
    
