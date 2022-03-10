import numpy as np
import dimod
import os
import networkx as nx
import matplotlib.pyplot as plt

from plotting import *
from do_dneal import *
# %%
# Specify instance 42
N = 100  # Number of variables
instance = 42
np.random.seed(instance)  # Fixing the random seed to get the same result
J = np.random.rand(N, N)
# We only consider upper triangular matrix ignoring the diagonal
J = np.triu(J, 1)
h = np.random.rand(N)

# %%
# Create directories for results
current_path = os.getcwd()
results_path = os.path.join(current_path, '../data/sk/')
if not(os.path.exists(results_path)):
    print('Results directory ' + results_path +
          ' does not exist. We will create it.')
    os.makedirs(results_path)

dneal_results_path = os.path.join(results_path, 'dneal/')
if not(os.path.exists(dneal_results_path)):
    print('Dwave-neal results directory ' + dneal_results_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_results_path)

dneal_pickle_path = os.path.join(dneal_results_path, 'pickles/')
if not(os.path.exists(dneal_pickle_path)):
    print('Dwave-neal pickles directory' + dneal_pickle_path +
          ' does not exist. We will create it.')
    os.makedirs(dneal_pickle_path)

instance_path = os.path.join(results_path, 'instances/')
if not(os.path.exists(instance_path)):
    print('Instances directory ' + instance_path +
          ' does not exist. We will create it.')
    os.makedirs(instance_path)

# %%
# Create instance 42 and save it into disk
model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

prefix = "random_n_" + str(N) + "_inst_"
instance_name = prefix + str(instance)
instance_file_name = instance_name + ".txt"
instance_file = os.path.join(instance_path, instance_file_name)
if not os.path.exists(instance_file):
    text_file = open(instance_file, "w")
    text_file.write(model_random.to_coo())
    text_file.close()

# %%
# Plot Q graph
nx_graph = model_random.to_networkx_graph()
edges, bias = zip(*nx.get_edge_attributes(nx_graph, 'bias').items())
bias = np.array(bias)
nx.draw(nx_graph, node_size=15, pos=nx.spring_layout(nx_graph),
        alpha=0.25, edgelist=edges, edge_color=bias, edge_cmap=plt.cm.Blues)


# %%
# Compute random sample on problem and print average energy
random_energy, random_sample = randomEnergySampler(
    model_random, num_reads=1000, dwave_sampler=True)
df_random_sample = random_sample.to_pandas_dataframe(sample_column=True)
print('Average random energy = ' + str(random_energy))

# %%
# Plot of obtained energies
# plotEnergyValuesDwaveSampleSet(random_sample,
#    title='Random sampling')
plotBarValues(df=df_random_sample, column_name='energy', sorted=True, skip=200,
              xlabel='Solution', ylabel='Energy', title='Random Sampling', save_fig=False, rot=0)


# %%
# Generate plots from the default simulated annealing run
# ax_enum = plotEnergyValuesDwaveSampleSet(sim_ann_sample_default,
#                              title='Simulated annealing with default parameters')
# ax_enum.set(ylim=[min_energy*(0.99)**np.sign(min_energy),
#             min_energy*(1.1)**np.sign(min_energy)])
plotBarValues(
    df=df_default_samples,
    column_name='energy',
    sorted=True,
    skip=200,
    xlabel='Solution',
    ylabel='Energy',
    title='Simulated Annealing with default parameters',
    save_fig=False,
    rot=0,
    ylim=[min_energy*(0.99)**np.sign(min_energy),
          min_energy*(1.1)**np.sign(min_energy)],
    legend=[],
)
# plot_energy_cfd(sim_ann_sample_default,
#                 title='Simulated annealing with default parameters', skip=10)
plotBarCounts(
    df=df_default_samples,
    column_name='energy',
    sorted=True,
    normalized=True,
    skip=10,
    xlabel='Energy',
    title='Simulated Annealing with default parameters',
    save_fig=False,
)

# %%
# Default Dwave-neal schedule plot
print(default_samples.info)
beta_schedule = np.geomspace(*default_samples.info['beta_range'], num=1000)
fig, ax = plt.subplots()
ax.plot(beta_schedule, '.')
ax.set_xlabel('Sweeps')
ax.set_ylabel('beta=Inverse temperature')
ax.set_title('Default Geometric temperature schedule')