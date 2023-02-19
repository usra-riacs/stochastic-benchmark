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
plot_vb_CI = True
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
        
    def plot_parameters_together(self):
        """Plot the parameters (Virtual Best and projection experiments)
        Create a single figure with a subfigure corresponding to each parameter

        Returns:
            fig: Figure handle
            axes: Dictionary of axis handles. The keys are the parameter names
        """
        
        fig, axes_list = plt.subplots(len(self.parent.parameter_names), 1)
        
        # Convert axes_list to a dictionary
        axes = dict()
        for ind, param in enumerate(self.parent.parameter_names):
            axes[param] = axes_list[ind]
            
        # Get the best parameters from the Virtual Baseline
        params_df, eval_df = self.parent.baseline.evaluate()
        eval_df = df_utils.monotone_df(eval_df, 'resource', 'response', 1)
        
        # Before plotting, store params_df to a csv file
        self.store_baseline_params(params_df)
        
            
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
            _ = axes[param].plot(params_df.index, params_df[param], 'o', ms=2, mec='k', alpha=0.25)
        
        cbar = fig.colorbar(line, ax=axes_list.ravel().tolist())
        cbar.ax.tick_params()
        cbar.set_label(self.parent.response_key) 
        
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
                    # Store experiment parameters to csv before plotting
                    self.store_expt_params(experiment.name, res)
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
            if hasattr(self, "xlims"):
                axes[param].set_xlim(self.xlims)
            # axes[param].legend()
        handles, labels = axes_list[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=[0.5,0],loc='upper center')
        # plt.legend()
        # fig.tight_layout()
        
        return fig, axes
        
        
    def store_baseline_params(self, params_df):
        """
        Store dataframe which has the data that is plotted for baseline parameters, to a csv file
        Parameters:

        """
        save_loc = os.path.join(self.parent.here.checkpoints, 'params_plotting')
        if not os.path.exists(save_loc) : os.makedirs(save_loc)
        save_file = os.path.join(save_loc, 'baseline.csv')
        params_df.to_csv(save_file)
        
    def store_expt_params(self, experiment_name, res):
        """
        Store data that will be used for plotting parameters from experiment
        Parameters:
            experiment_name (str): Name of the experiment (i.e. experiment.name)
            res (list): list with 2 or 3 items. res[0] is params_df (i.e. final params from expt), while res[2] (if it exists) contains parameters before post-processing.
        """
        save_loc = os.path.join(self.parent.here.checkpoints, 'params_plotting')
        save_file = os.path.join(save_loc, experiment_name+'.csv')
        params_df = res[0]
        params_df.to_csv(save_file)
        if len(res) == 3:
            # Len=3 only if postprocessing was used. 
            preproc_params = res[2]
            save_file = os.path.join(save_loc, experiment_name+'params.csv')
            preproc_params.to_csv(save_file)
    
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
        
        # Before plotting, store params_df to a csv file
        self.store_baseline_params(params_df)
        
            
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
                    # Store experiment parameters to csv file before plotting
                    self.store_expt_params(experiment.name, res)
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
            if hasattr(self, "xlims"):
                axes[param].set_xlim(self.xlims)
            axes[param].legend()
            figs[param].tight_layout()
        
        return figs, axes
    
    
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
        
        fig, axs = plt.subplots(1, 1)
        axs.plot(all_params['resource'], all_params['distance_scaled'])
        
        for idx, experiment in enumerate(self.parent.experiments):
            metaflag = hasattr(experiment, 'meta_params')
            params_df = all_params[all_params['exp_idx'] == idx]
            if metaflag:
                axs.plot(params_df['resource'], params_df['distance_scaled'], marker="x", linestyle=":", color=experiment.color, label=experiment.name)
            else:
                axs.plot(params_df['resource'], params_df['distance_scaled'], marker="o", color=experiment.color, label=experiment.name)
        
        axs.grid(axis="y")
        axs.set_ylabel("distance_scaled")
        axs.set_xscale(self.xscale)
        axs.set_xlabel("Resource")
        axs.legend(loc="best")
        fig.tight_layout()
        
        return fig, axs
    
    def plot_performance(self):
        """
        Plots the monotonized performance for each experiment (with the baseline)
        """
        # If saved data for virtual best exists, simply load it. Otherwise, compute the curve from baseline.
        save_loc = os.path.join(self.parent.here.checkpoints, 'performance_plotting')
        if not os.path.exists(save_loc) : os.makedirs(save_loc)
        save_file = os.path.join(save_loc, 'baseline.csv')
        if os.path.exists(save_file):
            eval_df = pd.read_csv(save_file)
        else:            
            _, eval_df = self.parent.baseline.evaluate()
            eval_df = df_utils.monotone_df(eval_df, 'resource', 'response', 1)
            # Store eval_df to a csv file
            eval_df.to_csv(save_file)
        
        
        fig, axs = plt.subplots(1, 1)
        if plot_vb_CI:
            axs.fill_between(eval_df['resource'], eval_df["response_lower"], \
                eval_df["response_upper"],alpha=0.25, color='k', lw=0)
        _ = axs.plot(eval_df['resource'], eval_df['response'], 'o-', ms=5, lw=1, color=self.parent.baseline.color, label=self.parent.baseline.name)
        
        
        for experiment in self.parent.experiments:
            try:
                save_file = os.path.join(save_loc, experiment.name+'.csv')
                if os.path.exists(save_file):
                    eval_df = pd.read_csv(save_file)
                else:
                    if monotone and not experiment.name == 'SequentialSearch_cold':
                        res = experiment.evaluate_monotone()
                    else:
                        res = experiment.evaluate()
                    eval_df = res[1]
                    # Store eval_df to a csv file
                    eval_df.to_csv(save_file)
                
                # Add confidence intervals
                axs.fill_between(eval_df['resource'], eval_df["response_lower"], eval_df["response_upper"],alpha=0.25, color=experiment.color, lw=0)#, label="CI Mean"+legend_str) # color='b',
                # Add mean/median line
                _ = axs.plot(eval_df['resource'], eval_df['response'], 'o-', ms=5, lw=1, color=experiment.color, label=experiment.name)
            except:
                continue
            
        axs.grid(axis="y")
        axs.set_ylabel(self.parent.response_key)
        axs.set_xscale(self.xscale)
        axs.set_xlabel("Resource")
        if hasattr(self, "xlims"):
            axs.set_xlim(self.xlims)
        axs.legend(loc="lower right")
        fig.tight_layout()
        return fig, axs
    
    def plot_meta_parameters(self):
        """
        Plots meta parameters for experiments that have them (random search and sequential search)
        """
        figs = dict()
        axes = dict()
        for idx, experiment in enumerate(self.parent.experiments):
            exp_figs = dict()
            exp_axes = dict()
            if hasattr(experiment, 'meta_params'):
                for param in experiment.meta_parameter_names:
                    # Create a figure for each parameter and each experiment
                    fig, axs = plt.subplots(1, 1)
                    experiment.meta_params.sort_values(by=experiment.resource, inplace=True)
                    axs.plot(experiment.meta_params[experiment.resource], experiment.meta_params[param], color=experiment.color, marker ='o',
                                         label = experiment.name)
                    if hasattr(experiment, 'preproc_meta_params'):
                        experiment.preproc_meta_params.sort_values(by=experiment.resource, inplace=True)
                        axs.plot(experiment.preproc_meta_params[experiment.resource], experiment.preproc_meta_params[param],
                                             color=experiment.color, marker ='x', linestyle = '--')
                    axs.grid(axis="y")
                    axs.set_ylabel(param)
                    axs.set_xscale(self.xscale)
                    axs.set_xlabel(experiment.resource)
                    axs.legend(loc="best")
                    fig.tight_layout()
                    exp_figs[param] = fig
                    exp_axes[param] = axs
                baseline_bool = False
                experiment_bools = [False] * len(self.parent.experiments)
                experiment_bools[idx] = True
                # exp_plot_dict = self.apply_shared(exp_plot_dict,
                #                                    baseline_bool=baseline_bool,
                #                                    experiment_bools=experiment_bools)
                # plots_dict[experiment.name] = exp_plot_dict
            figs[experiment.name] = exp_figs
            axes[experiment.name] = exp_axes
        return figs, axes