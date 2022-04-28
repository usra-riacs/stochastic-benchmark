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

EPSILON = 1e-10

# %%
# Specify instance 0 of the lattice problem
N = 4  # Length of the lattice
instance = 0
# %%
# Specify and if non-existing, create directories for results
current_path = os.getcwd()
data_path = os.path.join(current_path, '../data/toroidal2d_rand/')
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
suffix = '_200'
ocean_df_flag = True
results_path = dneal_results_path

# %%
# Load the given instance and its corresponding ground truth
np.random.seed(instance)  # Fixing the random seed to get the same result
J = np.random.rand(N, N)
# We only consider upper triangular matrix ignoring the diagonal
J = np.triu(J, 1)
h = np.random.rand(N)