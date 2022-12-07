import df_utils
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import training
import os
import numpy as np
from matplotlib.collections import LineCollection

monotone = False
dir_path = os.path.dirname(os.path.realpath(__file__))
ws_style = os.path.join(dir_path,'ws.mplstyle')

plt.style.use(ws_style)
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
            
    def plot_parameters_separate(self):
        """Plot the parameters (Virtual Best and projection experiments)
        Create a separate figure for each parameter

        Returns:
            figs: Dictionary of figure handles. The keys are the parameter names
            axes: Dictionary of axis handles. The keys are the parameter names
        """
        # For each resource value, obtain the best parameter value from VirtualBestBaseline
                       
        figs = dict()
        axes = dict()
        
        for param in self.parent.parameter_names:
            # Create one figure for each parameter 
            figs[param], axes[param] = plt.subplots(1, 1)
            
        # Get the best parameters from the Virtual Baseline
        params_df, eval_df = self.parent.baseline.evaluate()
        eval_df = df_utils.monotone_df(eval_df, 'resource', 'response', 1)
            
        # plot the virtual baseline paramters
        for param in self.parent.parameter_names:
            points = np.array([params_df.index.values, params_df[param].values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(eval_df['response'].min(), eval_df['response'].max())
            lc = LineCollection(segments, cmap='Spectral', norm=norm)
            # Set the values used for colormapping
            lc.set_array(eval_df['response'])
            lc.set_label(self.parent.baseline.name)
            lc.set_linewidth(8)
            lc.set_alpha(0.75)
            line = axes[param].add_collection(lc)
            cbar = figs[param].colorbar(line, ax=axes[param])
            cbar.ax.tick_params()
            cbar.set_label(self.parent.response_key)  
        
            _ = axes[param].plot(params_df.index, params_df[param], 'o', ms=2, mec='k', alpha=0.25)
            
        # Plot parameters from experiments
        for experiment in self.parent.experiments:
            # Choose whether to monotomize experiment parameters
            if monotone:
                res = experiment.evaluate_monotone()
            else:
                res = experiment.evaluate()
            params_df = res[0]
            eval_df = res[1]
            
            for param in self.parent.parameter_names:
                if not hasattr(experiment, 'meta_params'):
                    # Plot only if experiment does not have meta_parameters
                    _ = axes[param].plot(params_df['resource'], params_df[param], 'o-', ms=2, lw=1.5, color=experiment.color, label=experiment.name)
                if len(res) == 3:
                    # Len=3 only if postprocessing was used. In that case also plot the recipe before the postprocessing was done
                    preproc_params = res[2]
                    axes[param].plot(preproc_params['resource'], preproc_params[param], color=experiment.color, marker='x', linestyle=':', ms=2, lw=1.5)
        
        # Finally, add more properties such as labels, legend, etc.
        for param in self.parent.parameter_names:
            axes[param].grid(axis="y")
            axes[param].set_ylabel(param)
            axes[param].set_xscale(self.xscale)
            axes[param].set_xlabel("Resource")
            axes[param].legend()
            figs[param].tight_layout()
        
        return figs, axes
        
        
            
    
    def plot_parameters_prior(self):
        """
        This one is obsolete, and will be deleted soon.
        Plots the recommnded parameters for each experiment
        """
        # For each resource value, obtain the best parameter value from VirtualBestBaseline
        params_df,_ = self.parent.baseline.evaluate()
        
        # Store params_df to a csv file
        save_loc = os.path.join(self.parent.here.checkpoints, 'params_plotting')
        if not os.path.exists(save_loc) : os.makedirs(save_loc)
        save_file = os.path.join(save_loc, 'baseline.csv')
        params_df.to_csv(save_file)
        
        p = {}
        for param in self.parent.parameter_names:
            p[param] = (so.Plot(data=params_df, x='resource', y=param)
                        .add(so.Line(color = self.parent.baseline.color, linestyle='--'))
                       )
        for experiment in self.parent.experiments:
            metaflag = hasattr(experiment, 'meta_params')
            if monotone:
                res = experiment.evaluate_monotone()
            else:
                res = experiment.evaluate()
            params_df = res[0]
            
            save_file = os.path.join(save_loc, experiment.name+'.csv')
            params_df.to_csv(save_file)
            
            if len(res) == 3:
                preproc_params = res[2]

            for param in self.parent.parameter_names:
                if metaflag:
                    pass
                    # Commented out because meta_params might be too much information for these plots
                    # p[param] = (p[param].add(so.Line(color=experiment.color, linestyle=':'),
                    #                      data=params_df, x='resource', y=param)
                    #         .scale(x='log'))
                else:
                    p[param] = (p[param].add(so.Line(color=experiment.color, marker='o'),
                                         data=params_df, x='resource', y=param)
                            .scale(x='log'))
                if len(res) == 3:
                   p[param] = (p[param].add(so.Line(color=experiment.color, marker='x', linestyle=':'),
                                         data=preproc_params, x='resource', y=param)
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
            if monotone:
                params_df = experiment.evaluate_monotone()[0]
            else:
                params_df = experiment.evaluate()[0]
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
                p = (p.add(so.Line(color=experiment.color, marker='x', linestyle=':'),
                      data=params_df, x='resource', y='distance_scaled'))
            else:
                p = (p.add(so.Line(color=experiment.color, marker='o'),
                      data=params_df, x='resource', y='distance_scaled'))

        p = self.apply_shared(p, baseline_bool=False)
        
        return p
    
    def plot_performance(self):
        """
        Plots the monotonized performance for each experiment
        """
        _, eval_df = self.parent.baseline.evaluate()
        eval_df = df_utils.monotone_df(eval_df, 'resource', 'response', 1)
        # Store eval_df to a csv file
        save_loc = os.path.join(self.parent.here.checkpoints, 'performance_plotting')
        if not os.path.exists(save_loc) : os.makedirs(save_loc)
        save_file = os.path.join(save_loc, 'baseline.csv')
        eval_df.to_csv(save_file)
        
        
        p = (so.Plot(data=eval_df, x='resource', y='response')
             .add(so.Line(color = self.parent.baseline.color, marker='o'))
            )
        
        
        for experiment in self.parent.experiments:
            try:
                if monotone and not experiment.name == 'SequentialSearch_cold':
                    res = experiment.evaluate_monotone()
                else:
                    res = experiment.evaluate()
                eval_df = res[1]
                # Store eval_df to a csv file
                save_file = os.path.join(save_loc, experiment.name+'.csv')
                eval_df.to_csv(save_file)
                
                
                p = (p.add(so.Line(color=experiment.color, marker='o'), data=eval_df, x='resource', y='response', lw=7)
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
                         .add(so.Line(color=experiment.color, marker ='o'))
                        )
                    if hasattr(experiment, 'preproc_meta_params'):
                       exp_plot_dict[param] = exp_plot_dict[param].add(
                            so.Line(color=experiment.color, marker='x', linestyle ='--'),
                            data=experiment.preproc_meta_params, x=experiment.resource, y=param)
                baseline_bool = False
                experiment_bools = [False] * len(self.parent.experiments)
                experiment_bools[idx] = True
                exp_plot_dict = self.apply_shared(exp_plot_dict,
                                                   baseline_bool=baseline_bool,
                                                   experiment_bools=experiment_bools)
                plots_dict[experiment.name] = exp_plot_dict
                    
        return plots_dict
