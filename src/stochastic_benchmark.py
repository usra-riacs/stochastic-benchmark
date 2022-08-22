import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


import bootstrap
import df_utils
import interpolate
import random_exploration
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
        self.virtual_best = {}
        
        if os.path.exists(self.here.virtual_best['train']):
            self.virtual_best['train'] = pd.read_pickle(self.here.virtual_best['train'])
        else:
            training_results = self.interp_results[self.interp_results['train'] == 1]
            self.virtual_best['train'] = training.virtual_best(training_results,\
                               parameter_names=self.parameter_names,\
                               response_col='Key=PerfRatio',\
                               response_dir=1,\
                               resource_col='resource')
            self.virtual_best['train'].to_pickle(self.here.virtual_best['train'])
        
        if os.path.exists(self.here.virtual_best['test']):
            self.virtual_best['test'] = pd.read_pickle(self.here.virtual_best['test'])
        else:
            testing_results = self.interp_results[self.interp_results['train'] == 0]
            self.virtual_best['test'] = training.virtual_best(testing_results,\
                               parameter_names=self.parameter_names,\
                               response_col='Key=PerfRatio',\
                               response_dir=1,\
                               resource_col='resource')
            self.virtual_best['test'].to_pickle(self.here.virtual_best['test'])

        return 
    
    def set_recommended_parameters(self):
        self.best_recommended = {}
        response_col = names.param2filename(
            {'Key': 'PerfRatio', 'Metric': self.stat_params.stats_measures[0].name}, '')
        
        if os.path.exists(self.here.best_rec['stats']):
            self.best_recommended['stats'] = pd.read_pickle(self.here.best_rec['stats'])
        else:
            self.best_recommended['stats'] =\
            training.best_parameters(self.training_stats,
                                     parameter_names=self.parameter_names,
                                     response_col=response_col,
                                     response_dir=1,
                                     resource_col='resource',
                                     additional_cols=['boots'])
            self.best_recommended['stats'].to_pickle(self.here.best_rec['stats'])
        
        if not hasattr(self, 'virtual_best'):
            self.set_virtual_best()
        
        if os.path.exists(self.here.best_rec['results']):
            self.best_recommended['results'] = pd.read_pickle(self.here.best_rec['results'])
        else:
            self.best_recommended['results'] =\
            training.best_recommended(self.virtual_best['train'],
                                      parameter_names=self.parameter_names,
                                      resource_col='resource',
                                      additional_cols=['boots']).reset_index()
            self.best_recommended['results'].to_pickle(self.here.best_rec['results'])
         
    def project_recs(self):
        self.projections = {}
        if not hasattr(self, 'best_recommended'):
            self.set_recommended_parameters()
            
        testing_results = self.interp_results[self.interp_results['train'] == 0]
        for k, v in self.best_recommended.items():
            if os.path.exists(self.here.projections[k]):
                self.projections[k] = pd.read_pickle(self.here.projections[k])
            else:
                self.projections[k] = training.evaluate(testing_results,
                                                        v,
                                                        training.scaled_distance,
                                                        parameter_names=self.parameter_names)
                self.projections[k].to_pickle(self.here.projections[k])
    
    def run_random_exploration(self):
        key = names.param2filename({'Key': 'PerfRatio', 'Metric':'median'}, '')
        rsParams = random_exploration.RandomSearchParameters(parameter_names=self.parameter_names, key=key)
        if os.path.exists(self.here.best_agg_alloc):
            self.best_agg_alloc = pd.read_pickle(self.here.best_agg_alloc)
            self.train_exp_at_best = pd.read_pickle(self.here.train_exp_at_best)
            self.final_values = pd.read_pickle(self.here.final_values)
            
        else:
            self.best_agg_alloc, self.train_exp_at_best, self.final_values = random_exploration.RandomExploration(self.training_stats, rsParams)
            self.best_agg_alloc.to_pickle(self.here.best_agg_alloc)
            self.train_exp_at_best.to_pickle(self.here.train_exp_at_best)
            self.final_values.to_pickle(self.here.final_values)
            
            
        if os.path.exists(self.here.test_exp_at_best):
            self.test_exp_at_best = pd.read_pickle(self.here.test_exp_at_best)
        else:
            self.test_exp_at_best = random_exploration.apply_allocations(self.testing_stats, rsParams, self.best_agg_alloc)
            self.test_exp_at_best.to_pickle(self.here.test_exp_at_best)
        
    def plot_parameters(self):
        if not hasattr(self, 'best_recommended'):
            self.set_recommended_parameters()
        for k, v in self.best_recommended.items():
            fig, axs = plt.subplots(1, len(self.parameter_names), figsize=(len(self.parameter_names)*5 + 1, 5))
            fig.suptitle('Best recommended parameters generated from {}'.format(k))

            for idx, param in enumerate(self.parameter_names):
                ax = axs[idx]
                sns.lineplot(x='TotalBudget', y=param, data=self.train_exp_at_best, ax=ax, label='Random on train')
                sns.lineplot(x='TotalBudget', y=param, data=self.test_exp_at_best, ax=ax, label='Random on test')
                sns.lineplot(x='resource', y=param, data=v, ax=ax, label='Best rec')
                sns.lineplot(x='resource', y=param, data=self.virtual_best['test'], ci='sd',\
                         estimator='median', ax=ax, label = 'Virtual best')
                axs[idx].set_xscale('log')
                
#             figname = os.path.join(self.here.plots, 'RecParams_Gen={}.pdf'.format(k))
#             plt.savefig(figname)
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
            
            sns.lineplot(x='resource', y=key, data=self.virtual_best['test'], ci='sd',\
                         estimator='median', label = 'Virtual best')
            
            sns.lineplot(x='resource', y=key, data=v, ci='sd', estimator='median',\
                         label = 'Projected best rec.')
            
            sns.lineplot(x='TotalBudget', y='Key=PerfRatio_Metric=median',\
             data = self.test_exp_at_best, ci='sd', estimator='median', label = 'Random exploration')
            
            ax.set_xscale('log')
#             figname = os.path.join(self.here.plots, '{}_Gen={}.pdf'.format(key, k))
#             plt.savefig(figname)


