# %%
# Import the Dwave packages dimod and neal
import functools
import itertools
import os
import pickle
import time
import random
from typing import List, Union

import dimod
# Import Matplotlib to edit plots
import matplotlib.pyplot as plt
import neal
import networkx as nx
# Import numpy and scipy for certain numerical calculations below
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from pysa.sa import Solver
from scipy import sparse, stats

from plotting import *
from retrieve_data import *
from do_dneal import *
from util_benchmark import *

idx = pd.IndexSlice

# %% Specify problem to be solved

instance_class = 'sk'
N = 100  # Number of variables
prefix = "random_n_" + str(N) + "_inst_"
# suffix = '_' + str(N)
suffix = ''

# Specify single instance
instance = 42

# Specify all instances
instance_list = [i for i in range(20)] + [42]
training_instance_list = [i for i in range(20)]


# %%
# Define default values

default_sweeps = 1000
total_reads = 1000
float_type = 'float32'
default_reads = 1000
default_boots = default_reads
total_reads = 1000
# TODO rename this total_reads parameter, remove redundancy with above
default_Tfactor = 1.0
default_schedule = 'geometric'
default_replicas = 1
default_p_hot = 50.0
default_p_cold = 1.0
parameters_list = ['schedule', 'sweeps', 'Tfactor']
ocean_df_flag = True
default_dict = {
    'schedule': default_schedule,
    'sweeps': default_sweeps,
    'Tfactor': default_Tfactor,
    'boots': default_boots,
}

# %%
# Define experiment setting
sweeps_list = [i for i in range(1, 21, 1)] + [
    i for i in range(20, 201, 10)] + [
    i for i in range(200, 501, 20)] + [
    i for i in range(500, 1001, 25)]
Tfactor_list = [default_Tfactor]
# schedules_list = ['geometric', 'linear']
schedules_list = ['geometric']

parameters_dict = {
    'schedule': schedules_list,
    'sweeps': sweeps_list,
    'Tfactor': Tfactor_list,
}


# %%
# Specify and if non-existing, create directories for results

current_path = os.getcwd()
data_path = os.path.join(current_path, '../data/' + instance_class + '/')
if not(os.path.exists(data_path)):
    print('Data directory ' + data_path +
          ' does not exist. We will create it.')
    os.makedirs(data_path)

dneal_results_path = os.path.join(data_path, 'dneal/')
if not(os.path.exists(dneal_results_path)):
    print('Dwave-neal results directory ' + dneal_results_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_results_path)

dneal_pickle_path = os.path.join(dneal_results_path, 'pickles/')
if not(os.path.exists(dneal_pickle_path)):
    print('Dwave-neal pickles directory' + dneal_pickle_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_pickle_path)

pysa_results_path = os.path.join(data_path, 'pysa/')
if not(os.path.exists(pysa_results_path)):
    print('PySA results directory ' + pysa_results_path +
          ' does not exist. We will create it.')
    os.makedirs(pysa_results_path)

pysa_pickle_path = os.path.join(pysa_results_path, 'pickles/')
if not(os.path.exists(pysa_pickle_path)):
    print('PySA pickles directory' + pysa_pickle_path +
          ' does not exist. We will create it.')
    os.makedirs(pysa_pickle_path)

instance_path = os.path.join(data_path, 'instances/')
if not(os.path.exists(instance_path)):
    print('Instances directory ' + instance_path +
          ' does not exist. We will create it.')
    os.makedirs(instance_path)

plots_path = os.path.join(data_path, 'plots/')
if not(os.path.exists(plots_path)):
    print('Plots directory ' + plots_path +
          ' does not exist. We will create it.')
    os.makedirs(plots_path)

if ocean_df_flag:
    results_path = dneal_results_path
else:
    results_path = pysa_results_path

# %%
# Import single instance datafile

df_name_single_instance = "df_results_" + str(instance) + suffix + ".pkl"
df_path_single_instance = os.path.join(results_path, df_name_single_instance)
if os.path.exists(df_path_single_instance):
    df_single_instance = pd.read_pickle(df_path_single_instance)
else:
    df_single_instance = None


# %%
# Performance ratio vs sweeps for different bootstrap downsamples
dict_fixed = {'instance': instance, 'schedule': default_schedule}
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_single_instance,
    x_axis='sweeps',
    y_axis='perf_ratio',
    dict_fixed=dict_fixed,
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict.update(
        {'instance': 42, 'reads': default_sweeps*default_boots}),
    use_colorbar=False,
    ylim=[0.975, 1.0025],
    colors=['colormap'],
)
# %%
# Inverse performance ratio vs sweeps for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_single_instance,
    x_axis='sweeps',
    y_axis='inv_perf_ratio',
    dict_fixed=dict_fixed,
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    colors=['colormap'],
)
# %%
# Performance ratio vs runs for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_single_instance,
    x_axis='reads',
    y_axis='perf_ratio',
    dict_fixed=dict_fixed,
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    ylim=[0.975, 1.0025],
    colors=['colormap'],
)
# %%
# Performance ratio vs runs for different bootstrap downsamples
f, ax = plt.subplots()
ax = plot_1d_singleinstance_list(
    df=df_single_instance,
    x_axis='reads',
    y_axis='inv_perf_ratio',
    dict_fixed=dict_fixed,
    ax=ax,
    list_dicts=[{'boots': i} for i in [1, 10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    use_colorbar=False,
    colors=['colormap'],
    # ylim=[0.95, 1.005]
)
# %%
# TTS Plot for all bootstrapping downsamples in both schedules
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_single_instance,
    x_axis='reads',
    y_axis='tts',
    ax=ax,
    dict_fixed={'schedule': 'geometric', 'instance': 42},
    list_dicts=[{'boots': j}
                for j in [10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=True,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    colors=['colormap'],
)
# %%
# Mean time plot of some fixed parameter setting
dict_fixed = {'instance': instance, 'boots': default_boots}
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_single_instance,
    x_axis='sweeps',
    y_axis='mean_time',
    dict_fixed=dict_fixed,
    ax=ax,
    list_dicts=[{'schedule': i} for i in schedules_list],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
)
# %%
# Success probability of some fixed parameter setting
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_single_instance,
    x_axis='sweeps',
    y_axis='success_prob',
    dict_fixed=dict_fixed,
    ax=ax,
    list_dicts=[{'schedule': i} for i in schedules_list],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=False,
    save_fig=False,
    default_dict=default_dict,
)
# %%
# TTS Plot for all bootstrapping downsamples in both schedules
dict_fixed = {'instance': instance}
f, ax = plt.subplots()
plot_1d_singleinstance_list(
    df=df_single_instance,
    x_axis='sweeps',
    y_axis='tts',
    ax=ax,
    dict_fixed=dict_fixed,
    list_dicts=[{'schedule': i, 'boots': j}
                for i in schedules_list for j in [10, 100, 1000]],
    labels=labels,
    prefix=prefix,
    log_x=False,
    log_y=True,
    save_fig=False,
    default_dict=default_dict,
    colors=['colormap'],
)

# %%
