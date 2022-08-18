import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


import bootstrap
import df_utils
import interpolate
import stats
import training
import names

def prepare_bootstrap(nboots = 1000, 
                      response_col = names.param2filename({'Key': 'MinEnergy'}, ''),
                      resource_col = names.param2filename({'Key': 'MeanTime'}, '')):
    bs_iter_class = bootstrap.BSParams_iter()
    bsparams_iter = bs_iter_class(response_col, resource_col, nboots)
    return bsparams_iter

def sweep_boots_resource(df):
    return df['sweep'] * df['boots']

class stochastic_benchmark:
    def __init__(self, 
                 parameter_names,
                 here=os.getcwd(),
                 instance_cols=['instance'],
                 bs_params_iter = prepare_bootstrap(),
                 stat_params = stats.StatsParameters(stats_measures=[stats.Median()]),
                 resource_fcn = sweep_boots_resource,
                 train_test_split = 0.5,
                 recover=True):
        
        self.here = names.paths(here)
        self.parameter_names = parameter_names
        self.instance_cols = instance_cols
        self.bs_params_iter = bs_params_iter
        self.stat_params = stat_params
        self.resource_fcn = resource_fcn
        self.train_test_split = train_test_split
        self.recover=recover
        
        self.bs_results = None
        self.interp_results = None
        self.training_stats = None
        self.testing_stats = None
        
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
                self.training_stats = stats.Stats(training_results, self.stat_params,
                                             self.parameter_names + ['boots', 'resource'])
                self.training_stats.to_pickle(self.here.training_stats)
                
    def populate_testing_stats(self):
        if self.testing_stats is None:
            if os.path.exists(self.here.testing_stats) and self.recover:
                self.testing_stats = pd.read_pickle(self.here.testing_stats)
            elif self.interp_results is not None:
                testing_results = self.interp_results[self.interp_results['train'] == 0]
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
                group_on = self.parameters_names + self.instance_cols
                self.bs_results = bootstrap.Bootstrap(self.raw_data, group_on, self.bsparams_iter)
                self.bs_results.to_pickle(self.here.bootstrap)
            
            
    def set_virtual_best(self):
        training_results = self.interp_results[self.interp_results['train'] == 1]
        self.virtual_best = training.virtual_best(training_results,\
                           parameter_names=self.parameter_names,\
                           response_col='Key=PerfRatio',\
                           response_dir=1,\
                           resource_col='resource')
    
        return 
    
    def set_recommended_parameters(self):
        self.best_recommended = {}
        response_col = names.param2filename(
            {'Key': 'PerfRatio', 'Metric': self.stat_params.stats_measures[0].name}, '')
        self.best_recommended['stats'] =\
        training.best_parameters(self.training_stats,
                                 parameter_names=self.parameter_names,
                                 response_col=response_col,
                                 response_dir=1,
                                 resource_col='resource',
                                 additional_cols=['boots'])
        
        if not hasattr(self, 'virtual_best'):
            self.set_virtual_best()
        self.best_recommended['results'] =\
        training.best_recommended(self.virtual_best,
                                  parameter_names=self.parameter_names,
                                  resource_col='resource',
                                  additional_cols=['boots']).reset_index()
         
    def evaluate_recommendations(self):
        self.projections = {}
        if not hasattr(self, 'best_recommended'):
            self.set_recommended_parameters()
            
        testing_results = self.interp_results[self.interp_results['train'] == 0]
        for k, v in self.best_recommended.items():
            self.projections[k] = training.evaluate(testing_results,
                                                    v,
                                                    training.scaled_distance,
                                                    parameter_names=self.parameter_names)
    def plot_parameters(self):
        if not hasattr(self, 'best_recommended'):
            self.set_recommended_parameters()
        for k, v in self.best_recommended.items():
            fig, axs = plt.subplots(1, len(self.parameter_names), figsize=(len(self.parameter_names)*5 + 1, 5))
            fig.suptitle('Best recommended parameters generated from {}'.format(k))

            for idx, param in enumerate(self.parameter_names):
                sns.lineplot(x='resource', y=param, data=v, ax=axs[idx])
                axs[idx].set_xscale('log')
            figname = os.path.join(self.here.plots, 'RecParams_Gen={}.pdf'.format(k))
            plt.savefig(figname)
            # plt.close(fig)
            
    def plot_performance(self):
        if not hasattr(self, 'projections'):
            self.evaluate_recommendations()
        for k, v in self.projections.items():
            fig = plt.figure()
            ax = plt.gca()
            
            key = 'PerfRatio'
            fig.suptitle('{} from ({})'.format(key, k))
            key = names.param2filename({'Key': key}, '')
            sns.lineplot(x='resource', y=key, data=v, ax=ax)
            
            ax.set_xscale('log')
            figname = os.path.join(self.here.plots, '{}_Gen={}.pdf'.format(key, k))
            plt.savefig(figname)


