import df_utils
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import training

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
        self.colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        self.colors = sns.color_palette("tab10", len(self.parent.experiments))
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
            try:
                _, eval_df = experiment.evaluate_monotone()
                p = (p.add(so.Line(color=experiment.color, marker="x"), data=eval_df, x='resource', y='response')
                    .add(so.Band(alpha=0.2, color=experiment.color), data=eval_df, x='resource',
                        ymin='response_lower', ymax='response_upper')
                    )
            except:
                continue
        
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
