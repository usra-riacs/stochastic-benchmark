from collections import namedtuple
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import seaborn.objects as so
import seaborn as sns
import warnings


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
            "Evaluate should be overriden by a subclass of SuccessMetrics") 
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
        joint = df_utils.monotone_df(joint, 'resource', 'response', 1)
        params_df = joint.loc[:, ['resource'] + self.parent.parameter_names]
        eval_df = joint.loc[:, ['resource','response', 'response_lower', 'response_upper']]
        return params_df, eval_df

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
        self.attached = None
        
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
                br_train_path = os.path.join(self.parent.here.checkpoints, 'BestRecommended_train_postprocess={}.pkl'.format(self.postprocess_name))
            else:
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
                if self.postprocess is not None:
                    self.recipe = self.postprocess(self.recipe)
                    
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
            
            if self.postprocess is not None:
                    self.recipe = self.postprocess(self.recipe)
                    
        else:
            raise NotImplementedError('Projection from {} has not been implemented'.format(self.project_from))
        
        # Run the projections
        if os.path.exists(rec_path):
            self.rec_params = pd.read_pickle(rec_path)
        else:
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
        eval_df = eval_df.groupby('resource').median()
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
    
    def attach_runs(self, df):
        """
        Attaches reruns of experiment to 
        """
        if type(df) == str:
            df = pd.read_pickle(df)
        
        group_on = self.parent.instance_cols + ['resource']
        self.eval_df = self.parent.evaluate_without_bootstrap(self, df, group_on)
    
    def evaluate(self):
        """
        Returns
        -------
        params_df : pd.DataFrame
            Dataframe of recommended parameters
        eval_df : pd.DataFrame
            Dataframe of responses, renamed to generic columns for compatibility
        """
        params_df = self.eval_df.loc[:, ['resource'] + self.parent.parameter_names].copy()
        params_df = params_df.groupby('resource').mean()
        params_df.reset_index(inplace=True)
        
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
        eval_df = eval_df.groupby('resource').median()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df
    
    
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
    def __init__(self, parent, rsParams):
        self.parent = parent
        self.name = 'RandomSearch'
        self.rsParams = rsParams
        self.meta_parameter_names = ['ExploreFrac', 'tau']
        self.resource = 'TotalBudget'
        self.populate()
        
    def populate(self):
        """
        Populates meta_params, eval_train, eval_test 
        """
        meta_params_path = os.path.join(self.parent.here.checkpoints, 'RandomSearch_meta_params.pkl')
        eval_train_path = os.path.join(self.parent.here.checkpoints, 'RandomSearch_evalTrain.pkl')
        eval_test_path = os.path.join(self.parent.here.checkpoints, 'RandomSearch_evalTest.pkl')
        
        if os.path.exists(meta_params_path):
            self.meta_params = pd.read_pickle(meta_params_path)
        else:
            self.meta_params, self.eval_train, _ = random_exploration.RandomExploration(self.parent.training_stats, self.rsParams)
            self.meta_params.to_pickle(meta_params_path)
            self.eval_train.to_pickle(eval_train_path)
        self.meta_params['ExploreFrac'] = self.meta_params['ExplorationBudget'] / self.meta_params['TotalBudget']
        
        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
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
        eval_df = eval_df.groupby('resource').median()
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
    eval_train : pd.DataFrame
        Resulting parameters of meta_params on training set
    eval_test : pd.DataFrame
        Resulting parameters of meta_params on testing set    
    """
    def __init__(self, parent, ssParams):
        self.parent = parent
        self.name = 'SequentialSearch'
        self.ssParams = ssParams
        self.meta_parameter_names = ['ExploreFrac', 'tau']
        self.resource = 'TotalBudget'
        self.populate()
        
    def populate(self):
        meta_params_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_meta_params.pkl')
        eval_train_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_evalTrain.pkl')
        eval_test_path = os.path.join(self.parent.here.checkpoints, 'SequentialSearch_evalTest.pkl')
        if os.path.exists(meta_params_path):
            self.meta_params = pd.read_pickle(meta_params_path)
            self.eval_train = pd.read_pickle(eval_train_path)
        else:
            training_results = self.parent.interp_results[self.parent.interp_results['train'] == 1].copy()
            self.meta_params, self.eval_train, _ = sequential_exploration.SequentialExploration(training_results, self.ssParams, group_on=self.parent.instance_cols)
            self.meta_params.to_pickle(meta_params_path)
            self.eval_train.to_pickle(eval_train_path)
        self.meta_params['ExploreFrac'] = self.meta_params['ExplorationBudget'] / self.meta_params['TotalBudget']
        
        if os.path.exists(eval_test_path):
            self.eval_test = pd.read_pickle(eval_test_path)
        else:
            try:
                testing_results = self.parent.interp_results[self.parent.interp_results['train'] == 0].copy()
                self.eval_test = sequential_exploration.apply_allocations(testing_results,
                                                                          self.ssParams,
                                                                          self.meta_params,
                                                                          self.parent.instance_cols)
                self.eval_test.to_pickle(eval_test_path)
            except:
                print('Not enough test data for sequential search. Evaluating on train.')
    
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
        eval_df = eval_df.groupby('resource').median()
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
                               response_dir=1,\
                               groupby = self.parent.instance_cols,\
                               resource_col='resource',\
                                smooth=self.parent.smooth)
            self.rec_params.to_pickle(self.savename())
                
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
        eval_df = eval_df.groupby('resource').median()
        eval_df.reset_index(inplace=True)
        return params_df, eval_df

        

class Plotting:
    """
    Plotting helpers for coordinating plots
    
    Attributes
    ----------
    parent : stochatic_benchmark
    colors : list[str]
        Color palette for experiments. Baseline will always be black
    xcale : str
        scale for shared x axis
    xlims : tuple
        limits for shared x axis
    """
    
    def __init__(self, parent):
        self.parent = parent
        self.colors = ['blue', 'green', 'red', 'purple']
        self.assign_colors()
        self.xscale='log'

    
    def set_colors(self, cp):
        """
        Sets color palette and reassigns colors to experiments
        """
        self.colors = cp
        self.assign_colors()
        
    def set_xlims(self, xlims):
        """
        Sets limits for shared x 
        """
        self.xlims = xlims
    
    def make_legend(self, ax, baseline_bool, experiment_bools):
        """
        Makes legend for each experiment
        """
        if baseline_bool:
            color_patches = [mpatches.Patch(color=self.parent.baseline.color, label=self.parent.baseline.name)]
        else:
            color_patches = []
            
        color_patches = color_patches + [mpatches.Patch(color=experiment.color, label=experiment.name)
                        for idx, experiment in enumerate(self.parent.experiments)
                                        if experiment_bools[idx]]
        ax.legend(handles=[cpatch for cpatch in color_patches])
    
    def apply_shared(self, p, baseline_bool=True, experiment_bools=None):
        """
        Apply shared plot components (xscale, xlim, legends)
        """
        if experiment_bools is None:
            experiment_bools = [True] * len(self.parent.experiments)
        
        if type(p) is dict:
            for k, v in p.items():
                p[k] = self.apply_shared(v, baseline_bool, experiment_bools)
            return p
            
        p = p.scale(x=self.xscale)
        if hasattr(self, 'xlims'):
            p = p.limit(x=self.xlims)
        
        fig = plt.figure()
        p = p.on(fig).plot()
        ax = fig.axes[0]
        self.make_legend(ax, baseline_bool, experiment_bools)
            
        return fig
        
    def assign_colors(self):
        """
        Assigns colors to experiments
        """
        self.parent.baseline.color = 'black'
        for idx, experiment in enumerate(self.parent.experiments):
            experiment.color = self.colors[idx]
    
    def plot_parameters(self):
        """
        Plots the recommnded parameters for each experiment
        """
        params_df,_ = self.parent.baseline.evaluate()
        p = {}
        for param in self.parent.parameter_names:
            p[param] = (so.Plot(data=params_df, x='resource', y=param)
                        .add(so.Line(color = self.parent.baseline.color, linestyle='--'))
                       )
        for experiment in self.parent.experiments:
            metaflag = hasattr(experiment, 'meta_params')
            params_df, _ = experiment.evaluate_monotone()
            for param in self.parent.parameter_names:
                if metaflag:
                    p[param] = (p[param].add(so.Line(color=experiment.color, linestyle=':'),
                                         data=params_df, x='resource', y=param)
                            .scale(x='log'))
                else:
                    p[param] = (p[param].add(so.Line(color=experiment.color, marker='x'),
                                         data=params_df, x='resource', y=param)
                            .scale(x='log'))                     
                            
        p = self.apply_shared(p)
            
        return p
    
    def plot_parameters_distance(self):
        """
        Plots the scaled distance between the recommended parameters and virtual best
        """
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
            metaflag = hasattr(experiment, 'meta_params')
            params_df = all_params[all_params['exp_idx'] == idx]
            if metaflag:
                p = (p.add(so.Line(color=experiment.color, linestyle=':'),
                      data=params_df, x='resource', y='distance_scaled'))
            else:
                p = (p.add(so.Line(color=experiment.color, marker='x'),
                      data=params_df, x='resource', y='distance_scaled'))

        p = self.apply_shared(p, baseline_bool=False)
        
        return p
    
    def plot_performance(self):
        """
        Plots the monotonized performance for each experiment
        """
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
        
        p = self.apply_shared(p)
        return p

    def plot_meta_parameters(self):
        """
        Plots meta parameters for experiments that have them (random search and sequential search)
        """
        plots_dict = {}
        for idx, experiment in enumerate(self.parent.experiments):
            exp_plot_dict ={}
            if hasattr(experiment, 'meta_params'):
                for param in experiment.meta_parameter_names:
                    exp_plot_dict[param] = (so.Plot(data = experiment.meta_params, x=experiment.resource, y=param)
                         .add(so.Line(color=experiment.color, marker ='x'))
                        )
                baseline_bool = False
                experiment_bools = [False] * len(self.parent.experiments)
                experiment_bools[idx] = True
                expl_plot_dict = self.apply_shared(exp_plot_dict,
                                                   baseline_bool=baseline_bool,
                                                   experiment_bools=experiment_bools)
                plots_dict[experiment.name] = exp_plot_dict
                    
        return plots_dict
        
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
                 train_test_split = 0.5,
                 recover=True,
                smooth=True):
        
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
            self.populate_bs_results()
        
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
                print('Interpolating results with parameters: ', self.iParams)
                self.interp_results = interpolate.Interpolate(self.bs_results,
                                                              self.iParams, self.parameter_names+self.instance_cols)
                self.interp_results.to_pickle(self.here.interpolate)
                self.interp_results = training.split_train_test(self.interp_results, self.instance_cols, self.train_test_split)
                self.interp_results.to_pickle(self.here.interpolate)
    
    def populate_bs_results(self):
        """
        Tries to recover or computes bootstrapped results
        """
        if self.bs_results is None:
            if os.path.exists(self.here.bootstrap) and self.recover:
                print('Reading bootstrapped results')
                self.bs_results = pd.read_pickle(self.here.bootstrap)
            else:
                group_on = self.parameter_names + self.instance_cols
                if not hasattr(self, 'raw_data'):
                    self.raw_data = df_utils.read_exp_raw(self.here.raw_data)
                print('Running bootstrapped results')
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
        def evaluate_single(df_single):
            bs_params.update_rule(bs_params, df_single)
            resources = df_single[resource_col].values
            responses = df_single[response_col].values

            bs_df = pd.DataFrame()
            for metric_ref in bs_params.success_metrics:
                metric = metric_ref(bs_params.shared_args,bs_params.metric_args[metric_ref.__name__])
                metric.evaluate(bs_df, responses, resources)
            for col in bs_params.keep_cols:
                if col in df_single.columns:
                    val = df_single[col].iloc[0]
                    bs_df[col] = val
            
            return bs_df
        full_eval = df.groupby(group_on).apply(lambda df : evaluate_single(df))
        return full_eval
            
    def run_baseline(self):
        """
        Adds virtual best baseline
        """
        self.baseline = VirtualBestBaseline(self)
    def run_ProjectionExperiment(self, project_from, postprocess=None, postprocess_name=None):
        """
        Runs projections experiments
        """
        self.experiments.append(ProjectionExperiment(self, project_from, postprocess, postprocess_name))
    def run_RandomSearchExperiment(self, rsParams):
        """
        Runs random search experiments
        """
        self.experiments.append(RandomSearchExperiment(self, rsParams))
    def run_SequentialSearchExperiment(self, ssParams):
        """
        Runs sequential search experiments
        """
        self.experiments.append(SequentialSearchExperiment(self, ssParams))
    def initPlotting(self):
        """
        Sets up plotting - this should be run after all experiments are run
        """
        self.plots = Plotting(self)
    
