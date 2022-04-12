# %%
import os
import pandas as pd
import numpy as np
from plotting import *
import matplotlib.pyplot as plt
#%%
s = 0.99
EPSILON = 1e-10
gap = 1

current_path = os.getcwd()
data_path = os.path.join(current_path, '../data/sk/')

results_path = os.path.join(data_path, 'dneal/')

labels = {
    'N': 'Number of variables',
    'instance': 'Random instance',
    'replicas': 'Number of replicas',
    'sweeps': 'Number of sweeps',
    'rep': 'Number of replicas',
    'swe': 'Number of sweeps',
    'swe': 'Number of sweeps',
    'pcold': 'Probability of dEmin flip at cold temperature',
    'phot': 'Probability of dEmax flip at hot temperature',
    'mean_time': 'Mean time [us]',
    'success_prob': 'Success probability \n (within ' + str(gap) + '% of best found)',
    'median_success_prob': 'Success probability \n (within ' + str(gap) + '% of best found)',
    'mean_success_prob': 'Success probability \n (within ' + str(gap) + '% of best found)',
    'perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'median_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'mean_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'median_mean_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'mean_mean_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'median_median_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'mean_median_perf_ratio': 'Performance ratio \n (random - best found) / (random - min)',
    'tts': 'TTS ' + str(100*s) + '% confidence  \n (within ' + str(gap) + '% of best found) [s]',
    'median_tts': 'TTS ' + str(100*s) + '% confidence  \n (within ' + str(gap) + '% of best found) [s]',
    'mean_tts': 'TTS ' + str(100*s) + '% confidence  \n (within ' + str(gap) + '% of best found) [s]',
    'boots': 'Number of downsamples during bootrapping',
    'reads': 'Total number of reads (proportional to time)',
    'cum_reads': 'Total number of reads (proportional to time)',
    'mean_cum_reads': 'Total number of reads (proportional to time)',
    'min_energy': 'Minimum energy found',
    'mean_time': 'Mean time [us]',
    'Tfactor': 'Factor to multiply lower temperature by',
    'experiment': 'Experiment',
    'inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'median_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'mean_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'median_mean_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'median_median_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'mean_mean_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    'mean_median_inv_perf_ratio': 'Inverse performance ratio \n (best found  - min) / (random - min) + ' + str(EPSILON),
    # 'tts': 'TTS to GS with 99% confidence \n [s * replica] ~ [MVM]',
}
suffixes = ['','T','t','C']
dict_names = {'sweeps':'','Tfactor with sweeps=1000':'T','Tfactor with sweeps=100':'t','both sweeps and Tfactor':'C'}
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
                u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
prefix = "random_n_100_inst_"
f, ax = plt.subplots()

pysa_results_path = os.path.join(data_path, 'pysa/')
counter = 0
df_name = "df_results_virt" + 'P' + ".pkl"
df_path = os.path.join(pysa_results_path, df_name)
df_virtual_best_pysa = pd.read_pickle(df_path)
plot_1d_singleinstance(
        df=df_virtual_best_pysa,
        x_axis='reads',
        y_axis='perf_ratio',
        ax=ax,
        label_plot='Virtual best PySA',
        dict_fixed=None,
        prefix=prefix,
        save_fig=False,
        labels=labels,
        log_x=True,
        log_y=False,
        ylim=[0.975, 1.0025],
        xlim=[5e2, 1e6],
        linewidth=1.5,
        markersize=1,
        color = ['k']
    )
for key, val in dict_names.items():
    df_name = "df_results_virt" + val + ".pkl"
    df_path = os.path.join(results_path, df_name)
    df_virtual_best = pd.read_pickle(df_path)
    
    plot_1d_singleinstance(
        df=df_virtual_best,
        x_axis='reads',
        y_axis='perf_ratio',
        ax=ax,
        label_plot='Virtual best ' + key,
        dict_fixed=None,
        prefix=prefix,
        save_fig=False,
        labels=labels,
        log_x=True,
        log_y=False,
        ylim=[0.975, 1.0025],
        xlim=[5e2, 1e6],
        linewidth=1.5,
        markersize=1,
        color = colors[counter]
    )
    counter += 1
# %%
