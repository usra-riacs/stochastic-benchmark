# Import the Dwave packages dimod and neal
import dimod
import neal
# Import Matplotlib to generate plots
import matplotlib.pyplot as plt
# Import numpy and scipy for certain numerical calculations below
import numpy as np
import math
from collections import Counter
import pandas as pd
from itertools import chain
import time
import networkx as nx
import os
import pickle
from scipy import stats
from matplotlib import ticker
import pyomo.environ as pyo


# Create own random instances
N = 50  # Number of variables
np.random.seed(42)  # Fixing the random seed to get the same result
J = np.random.choice([-1, 1], size=(N, N))
# J = np.random.rand(N, N)
# We only consider upper triangular matrix ignoring the diagonal
J = np.triu(J, 1)
h = np.zeros(N)

model_sk = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

# We do not need to worry about the transformation to QUBO since dimod takes care of it
Q, c = model_sk.to_qubo()

# Define the model
model_pyo = pyo.ConcreteModel(name='Random SK model N=50 modeled as linaerized MIP')

I = range(len(model_sk))
J = range(len(model_sk))
#Define the original variables
model_pyo.x = pyo.Var(I, domain=pyo.Binary)
# Define the edges variables
model_pyo.y = pyo.Var(I, J, domain=pyo.Binary)

obj_expr = c

# add model constraints
model_pyo.c1 = pyo.ConstraintList()
model_pyo.c2 = pyo.ConstraintList()
model_pyo.c3 = pyo.ConstraintList()
for (i,j) in Q.keys():
    if i != j:
        model_pyo.c1.add(model_pyo.y[i,j] >= model_pyo.x[i] + model_pyo.x[j] - 1)
        model_pyo.c2.add(model_pyo.y[i,j] <= model_pyo.x[i])
        model_pyo.c3.add(model_pyo.y[i,j] <= model_pyo.x[j])
        obj_expr += Q[i,j]*model_pyo.y[i,j]
    else:
        obj_expr += Q[i,j]*model_pyo.x[i]

# Define the objective function
model_pyo.objective = pyo.Objective(expr = obj_expr, sense=pyo.minimize)
# Print the model
model_pyo.display()

opt_gurobi = pyo.SolverFactory('gurobi')

result_obj = opt_gurobi.solve(model_pyo, tee=True, timelimit=10)