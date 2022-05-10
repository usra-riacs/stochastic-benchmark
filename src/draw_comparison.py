import pandas as pd
import numpy as np
import os
import sys

statistic = '50'
size = 200
ocean_df_flag = True
data_path = "/home/bernalde/repos/stochastic-benchmark/data/sk_pleiades/"
plots_path = os.path.join(data_path, 'plots/')
if ocean_df_flag:
    results_path = os.path.join(data_path, 'dneal/')
else:
    results_path = os.path.join(data_path, 'pysa/')

draw_plots = True
prefix = "random_n_"+str(size)+"_inst_"
df_name_stats = prefix + 'df_stats.pkl'
    # df_name_stats = prefix + 'df_stats.pkl'
df_path_stats = os.path.join(results_path, df_name_stats)
if not os.path.exists(df_path_stats):
    print('No file found: ' + df_path_stats)
else:
    df_stats = pd.read_pickle(df_path_stats)
    df_name_stats_interpolated = df_name_stats.rsplit('.')[
        0] + '_interp.pkl'
    df_path_stats_interpolated = os.path.join(
        results_path, df_name_stats_interpolated)
    df_stats_interpolated = pd.read_pickle(df_path_stats_interpolated)


default_parameters = {
    'sweep': [1000],
    'schedule': ['geometric'],
    'replica': [1],
    'Tcfactor': [0.0],
    'Thfactor': [0.0],
}


if ocean_df_flag:
    parameters_dict = {
        'schedule': [],
        'sweep': [],
        'Tcfactor': [],
        'Thfactor': [],
    }
else:
    parameters_dict = {
        'sweep': [],
        'replica': [],
        'Tcfactor': [],
        'Thfactor': [],
    }
# Add default parametes in case they are missing and remove repeated elements in the parameters_dict values (sets)
if parameters_dict is not None:
    for i, j in parameters_dict.items():
        if len(j) == 0:
            parameters_dict[i] = default_parameters[i]
        else:
            parameters_dict[i] = set(j)

parameter_names = list(parameters_dict.keys())


# %%
# Import plotting libraries
if draw_plots:
    import matplotlib.pyplot as plt
    import seaborn as sns
# %%
# Windows Stickers Plot
if draw_plots:
    # Windows sticker plot with virtual best, suggested parameters, best from TTS, default parameters, random search, and ternary search
    fig, ax = plt.subplots()
    sns.lineplot(x='reads', y='virt_best_'+statistic+'_perf_ratio', data=df_virtual,
                 ax=ax, estimator=None, label='Quantile '+statistic+' virtual best')
    sns.lineplot(x='reads', y=statistic+'_lazy_perf_ratio', data=df_virtual,
                 ax=ax, estimator=None, label='Quantile '+statistic+' suggested mean parameter')
    best_tts_param = df_stats_interpolated.nsmallest(
        1, str(100-int(statistic))+'_rtt'
    ).set_index(parameter_names).index.values
    best_tts = df_stats_interpolated[parameter_names + ['reads', statistic+'_perf_ratio']].set_index(
        parameter_names
    ).loc[
        best_tts_param].reset_index()
    default_params = []
    if parameters_dict is not None:
        for i, j in parameters_dict.items():
            default_params.append(default_parameters[i][0])
    default_performance = df_stats_interpolated[parameter_names + ['reads', statistic+'_perf_ratio']].set_index(
        parameter_names
    ).loc[
        tuple(default_params)].reset_index()
    random_params = df_stats_interpolated.sample(
        n=1).set_index(parameter_names).index.values
    random_performance = df_stats_interpolated[parameter_names + ['reads', statistic+'_perf_ratio']].set_index(
        parameter_names
    ).loc[
        random_params].reset_index()
    sns.lineplot(x='reads', y=statistic+'_perf_ratio', data=best_tts, ax=ax, estimator=None, label='Quantile '+statistic+' best TTS parameter \n' +
                 " ".join([str(parameter_names[i] + ':' + str(best_tts_param[0][i])) for i in range(len(parameter_names))]))
    sns.lineplot(x='reads', y=statistic+'_perf_ratio', data=default_performance, ax=ax, estimator=None, label='Quantile '+statistic+' default parameter \n' +
                 " ".join([str(parameter_names[i] + ':' + str(default_params[i])) for i in range(len(parameter_names))]))
    sns.lineplot(x='reads', y=statistic+'_perf_ratio', data=random_performance, ax=ax, estimator=None, label='Quantile '+statistic+' random parameter \n' +
                 " ".join([str(parameter_names[i] + ':' + str(random_params[0][i])) for i in range(len(parameter_names))]))
    sns.lineplot(x='R_budget', y='mean_'+statistic+'_perf_ratio', data=df_progress_best,
                 ax=ax, estimator=None, label='Quantile '+statistic+'  best random search expl-expl')
    sns.lineplot(x='R_budget', y='mean_'+statistic+'_perf_ratio', data=df_progress_best_ternary,
                 ax=ax, estimator=None, label='Quantile '+statistic+'  best ternary search expl-expl')
    ax.set(xscale='log')
    ax.set(ylim=[0.98, 1.001])
    ax.set(xlim=[5e2, 1e6])
    ax.set(ylabel='Performance ratio = \n (random - best found) / (random - min)')
    ax.set(xlabel='Total number of spin variable reads (proportional to time)')
    if ocean_df_flag:
        solver = 'dneal'
    else:
        solver = 'pysa'
    ax.set(title='Windows sticker plot \n instances SK ' +
           prefix.rsplit('_inst')[0] + '\n solver ' + solver)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fancybox=True)
    if ocean_df_flag:
        solver = 'dneal'
    else:
        solver = 'pysa'
    plot_name = 'windows_sticker_comparison_'+str(size)+'_'+statistic+'.png'
    plt.savefig(os.path.join(plots_path,plot_name), bbox_extra_artists=(lgd,), bbox_inches='tight')