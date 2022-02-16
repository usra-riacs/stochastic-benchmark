# %%
# Import packages
import math
import os
import pickle
import time
import zipfile
from collections import Counter
from itertools import chain

import dimod
import matplotlib.pyplot as plt
import neal
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy import stats


# %%
# Some useful functions to get plots
def plot_energy_values(
    results: dimod.SampleSet,
    title: str = None,
):
    '''
    Plots the energy values of the samples in a histogram.

    Args:
        results: A dimod.SampleSet object.
        title: A string to use as the plot title.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''

    _, ax = plt.subplots()

    energies = [datum.energy for datum in results.data(
        ['energy'], sorted_by='energy')]

    if results.vartype == 'Vartype.BINARY':
        samples = [''.join(c for c in str(datum.sample.values()).strip(
            ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        ax.set(xlabel='bitstring for solution')
    else:
        samples = np.arange(len(energies))
        ax.set(xlabel='solution')

    ax.bar(samples, energies)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel('Energy')
    if title:
        ax.set_title(str(title))
    print("minimum energy:", min(energies))
    return ax


def plot_samples(
    results: dimod.SampleSet,
    title: str = None,
    skip: int = 1,
):
    '''
    Plots the samples of the samples in a histogram.

    Args:
        results: A dimod.SampleSet object.
        title: A string to use as the plot title.
        skip: An integer to skip every nth sample.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    if results.vartype == 'Vartype.BINARY':
        samples = [''.join(c for c in str(datum.sample.values()).strip(
            ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        ax.set_xlabel('bitstring for solution')
    else:
        samples = np.arange(len(energies))
        ax.set_xlabel('solution')

    counts = Counter(samples)
    total = len(samples)
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None, ax=ax)

    ax.tick_params(axis='x', rotation=80)
    ax.set_xticklabels([t.get_text()[:7] if not i %
                       skip else "" for i, t in enumerate(ax.get_xticklabels())])
    ax.set_ylabel('Probabilities')
    if title:
        ax.set_title(str(title))
    print("minimum energy:", min(energies))
    return ax


def plot_energy_cfd(
    results: dimod.SampleSet,
    title: str = None,
    skip: int = 1,
):
    '''
    Plots the energy values of the samples in a cumulative distribution function.

    Args:
        results: A dimod.SampleSet object.
        title: A string to use as the plot title.
        skip: An integer to skip every nth sample.

    Returns:
        A matplotlib.pyplot.Figure object.
    '''
    _, ax = plt.subplots()
    # skip parameter given to avoid putting all xlabels
    energies = results.data_vectors['energy']
    occurrences = results.data_vectors['num_occurrences']
    counts = Counter(energies)
    total = sum(occurrences)
    counts = {}
    for index, energy in enumerate(energies):
        if energy in counts.keys():
            counts[energy] += occurrences[index]
        else:
            counts[energy] = occurrences[index]
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None, ax=ax)
    ax.set_xticklabels([t.get_text()[:7] if not i %
                       skip else "" for i, t in enumerate(ax.get_xticklabels())])

    ax.set_xlabel('Energy')
    ax.set_ylabel('Probabilities')
    if title:
        ax.set_title(str(title))
    print("minimum energy:", min(energies))
    return ax

# %%
# Let's define a model, with 100 variables and random weights, to see how this performance changes.
# Assume that we are interested at the instance created with random weights coming from a seed of 42.


# %%
N = 100  # Number of variables
np.random.seed(42)  # Fixing the random seed to get the same result
J = np.random.rand(N, N)
# We only consider upper triangular matrix ignoring the diagonal
J = np.triu(J, 1)
h = np.random.rand(N)


# %%
model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)


# %%
nx_graph = model_random.to_networkx_graph()
edges, bias = zip(*nx.get_edge_attributes(nx_graph, 'bias').items())
bias = np.array(bias)
nx.draw(nx_graph, node_size=15, pos=nx.spring_layout(nx_graph),
        alpha=0.25, edgelist=edges, edge_color=bias, edge_cmap=plt.cm.Blues)


# %% [markdown]
# For a problem of this size we cannot do a complete enumeration ($2^{100} \approx 1.2e30$) but we can randomly sample the distribution of energies to have a baseline for our later comparisons.

# %%
randomSampler = dimod.RandomSampler()
randomSample = randomSampler.sample(model_random, num_reads=1000)
energies = [datum.energy for datum in randomSample.data(
    ['energy'], sorted_by='energy')]
random_energy = np.mean(energies)
print('Average random energy = ' + str(random_energy))


# %%
plot_energy_values(randomSample,
                   title='Random sampling')


# %%
simAnnSampler = dimod.SimulatedAnnealingSampler()
start = time.time()
simAnnSamplesDefault = simAnnSampler.sample(model_random, num_reads=1000)
timeDefault = time.time() - start
energies = [datum.energy for datum in simAnnSamplesDefault.data(
    ['energy'], sorted_by='energy')]
min_energy = energies[0]
print(min_energy)


# %%
ax_enum = plot_energy_values(simAnnSamplesDefault,
                             title='Simulated annealing with default parameters')
ax_enum.set(ylim=[min_energy*(0.99)**np.sign(min_energy),
            min_energy*(1.1)**np.sign(min_energy)])
plot_energy_cfd(simAnnSamplesDefault,
                title='Simulated annealing with default parameters', skip=10)


# %% [markdown]
# Notice that the minimum energy coming from the random sampling and the one from the simulated annealing are very different.
# Moreover, the distributions that both lead to are extremely different too.

# %%
print(simAnnSamplesDefault.info)
beta_schedule = np.geomspace(
    *simAnnSamplesDefault.info['beta_range'], num=1000)
fig, ax = plt.subplots()
ax.plot(beta_schedule, '.')
ax.set_xlabel('Sweeps')
ax.set_ylabel('beta=Inverse temperature')
ax.set_title('Default Geometric temperature schedule')

# %%
# Define function to compute random sampled energy


def random_energy_sampler(
    model: dimod.BinaryQuadraticModel,
    num_reads: int = 1000,
    dwave_sampler: bool = False,
) -> float:
    '''
    Computes the energy of a random sampling.

    Args:
        num_reads: The number of samples to use.
        dwave_sampler: A boolean to use the D-Wave sampler or not.

    Returns:
        The energy of the random sampling.
    '''
    if dwave_sampler:
        randomSampler = dimod.RandomSampler()
        randomSample = randomSampler.sample(model, num_reads=num_reads)
        energies = [datum.energy for datum in randomSample.data(
            ['energy'], sorted_by='energy')]
    else:
        state = np.random.randint(2, size=(model.num_variables, num_reads))
        energies = [model.energy(state[:, i]) for i in range(num_reads)]
    return np.mean(energies)

# %%
# To get a scaled version of this success equivalent for all instances, we will define this success with respect to the metric:
# $$
# \frac{found - random}{minimum - random}
# $$
# Where $found$ corresponds to the best found solution within our sampling, $random$ is the mean of the random sampling shown above, and $minimum$ corresponds to the best found solution to our problem during the exploration. Consider that this minimum might not be the global minimum.
# This metric is very informative given that the best performance you can have is 1, being at the minimum, and negative values would correspond to a method that at best behaves worse that the random sampling.
# Success now is counted as being within certain treshold of this value of 1.
# This new way of measuring each method is very similar to the approximation ratio of approximation algorithms, therefore we will use that terminology from now on.


# %%
# Define directory for results and create it if it does not exist
current_path = os.getcwd()
pickle_path = os.path.join(current_path, '../data/sk/dneal/pickles')
if not(os.path.exists(pickle_path)):
    print('Dwave-neal pickle directory ' + pickle_path +
          ' does not exist. We will create it.')
    os.makedirs(pickle_path)

# %%
# Extract results if there is a zip file containing the pickles
zip_name = os.path.join(pickle_path, 'results.zip')
overwrite_pickles = False
use_raw_data = True
if os.path.exists(zip_name) and use_raw_data:
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(pickle_path)
    print('Results zip file has been extrated to ' + pickle_path)

# %%
s = 0.99  # This is the success probability for the TTS calculation
treshold = 5.0  # This is a percentual treshold of what the minimum energy should be
sweeps = list(chain(np.arange(1, 250, 1), np.arange(250, 1001, 10)))
schedules = ['geometric', 'linear']
# schedules = ['geometric']
total_reads = 1000
default_sweeps = 1000
n_boot = 500
ci = 68  # Confidence interval for bootstrapping
default_boots = default_sweeps
boots = [1, 10, default_boots]
min_energy = -239.7094652034834
# Add function to compute actual minimum
instance = 42
results_name = "results_" + str(instance) + ".pkl"
results_name = os.path.join(pickle_path, results_name)
results = {}
results['p'] = {}
results['min_energy'] = {}
results['random_energy'] = {}
results['tts'] = {}
results['ttsci'] = {}
results['t'] = {}
results['best'] = {}
results['bestci'] = {}
# If you wanto to use the raw data and process it here
if use_raw_data or not(os.path.exists(results_name)):
    # If you want to generate the data or load it here
    overwrite_pickles = False

    for boot in boots:
        results['p'][boot] = {}
        results['tts'][boot] = {}
        results['ttsci'][boot] = {}
        results['best'][boot] = {}
        results['bestci'][boot] = {}

    for schedule in schedules:
        probs = {k: [] for k in boots}
        time_to_sol = {k: [] for k in boots}
        prob_np = {k: [] for k in boots}
        ttscs = {k: [] for k in boots}
        times = []
        b = {k: [] for k in boots}
        bnp = {k: [] for k in boots}
        bcs = {k: [] for k in boots}
        for sweep in sweeps:
            # Gather instance names
            pickle_name = str(instance) + "_" + schedule + \
                "_" + str(sweep) + ".p"
            pickle_name = os.path.join(pickle_path, pickle_name)
            # If the instance data exists, load the data
            if os.path.exists(pickle_name) and not overwrite_pickles:
                # print(pickle_name)
                samples = pickle.load(open(pickle_name, "rb"))
                time_s = samples.info['timing']
            # If it does not exist, generate the data
            else:
                start = time.time()
                samples = simAnnSampler.sample(
                    model_random, num_reads=total_reads, num_sweeps=sweep, beta_schedule_type=schedule)
                time_s = time.time() - start
                samples.info['timing'] = time_s
                pickle.dump(samples, open(pickle_name, "wb"))
            # Compute statistics
            energies = samples.data_vectors['energy']
            occurrences = samples.data_vectors['num_occurrences']
            total_counts = sum(occurrences)
            times.append(time_s)
            if min(energies) < min_energy:
                min_energy = min(energies)
                print("A better solution of " + str(min_energy) +
                      " was found for sweep " + str(sweep))
            # success = min_energy*(1.0 + treshold/100.0)**np.sign(min_energy)
            success = random_energy - \
                (random_energy - min_energy)*(1.0 - treshold/100.0)

            # Best of boot samples es computed via n_boot bootstrappings
            boot_dist = {}
            pr_dist = {}
            cilo = {}
            ciup = {}
            pr = {}
            pr_cilo = {}
            pr_ciup = {}
            for boot in boots:
                boot_dist[boot] = []
                pr_dist[boot] = []
                for i in range(int(n_boot)):
                    resampler = np.random.randint(0, total_reads, boot)
                    sample_boot = energies.take(resampler, axis=0)
                    # Compute the best along that axis
                    boot_dist[boot].append(min(sample_boot))

                    occurences = occurrences.take(resampler, axis=0)
                    counts = {}
                    for index, energy in enumerate(sample_boot):
                        if energy in counts.keys():
                            counts[energy] += occurences[index]
                        else:
                            counts[energy] = occurences[index]
                    pr_dist[boot].append(
                        sum(counts[key] for key in counts.keys() if key < success)/boot)

                b[boot].append(np.mean(boot_dist[boot]))
                # Confidence intervals from bootstrapping the best out of boot
                bnp[boot] = np.array(boot_dist[boot])
                cilo[boot] = np.apply_along_axis(
                    stats.scoreatpercentile, 0, bnp[boot], 50.-ci/2.)
                ciup[boot] = np.apply_along_axis(
                    stats.scoreatpercentile, 0, bnp[boot], 50.+ci/2.)
                bcs[boot].append((cilo[boot], ciup[boot]))
                # Confidence intervals from bootstrapping the TTS of boot
                prob_np[boot] = np.array(pr_dist[boot])
                pr[boot] = np.mean(prob_np[boot])
                probs[boot].append(pr[boot])
                if prob_np[boot].all() == 0:
                    time_to_sol[boot].append(np.inf)
                    ttscs[boot].append((np.inf, np.inf))
                else:
                    pr_cilo[boot] = np.apply_along_axis(
                        stats.scoreatpercentile, 0, prob_np[boot], 50.-ci/2.)
                    pr_ciup[boot] = np.apply_along_axis(
                        stats.scoreatpercentile, 0, prob_np[boot], 50.+ci/2.)
                    time_to_sol[boot].append(
                        time_s*math.log10(1-s)/math.log10(1-pr[boot]+1e-9))
                    ttscs[boot].append((time_s*math.log10(1-s)/math.log10(
                        1-pr_cilo[boot]), time_s*math.log10(1-s)/math.log10(1-pr_ciup[boot]+1e-9)))

        results['t'][schedule] = times
        results['min_energy'][schedule] = min_energy
        results['random_energy'][schedule] = random_energy
        for boot in boots:
            results['p'][boot][schedule] = probs[boot]
            results['tts'][boot][schedule] = time_to_sol[boot]
            results['ttsci'][boot][schedule] = ttscs[boot]
            results['best'][boot][schedule] = [
                (random_energy - energy) / (random_energy - min_energy) for energy in b[boot]]
            results['bestci'][boot][schedule] = [tuple((random_energy - element) / (
                random_energy - min_energy) for element in energy) for energy in bcs[boot]]

    # Save results file in case that we are interested in reusing them
    pickle.dump(results, open(results_name, "wb"))
else:  # Just reload processed datafile
    results = pickle.load(open(results_name, "rb"))


# %% [markdown]
# After gathering all the results, we would like to see the progress of the approximation ration with respect to the increasing number of sweeps.
# To account for the stochasticity of this method, we are bootstrapping all of our results with different values of the bootstrapping sample, and each confidence interval corresponds to a standard deviation away from the mean.

# %%
fig, ax = plt.subplots()
for boot in boots:
    for schedule in schedules:
        ax.plot(sweeps, results['best'][boot][schedule], label=str(
            schedule) + ', ' + str(boot) + ' reads')
        bestnp = np.stack(results['bestci'][boot][schedule], axis=0).T
        ax.fill_between(sweeps, bestnp[0], bestnp[1], alpha=0.25)
ax.set(xlabel='Sweeps')
ax.set(ylabel='Approximation ratio = \n ' +
       '(best found - random sample) / (min energy - random sample)')
ax.set_title('Simulated annealing approximation ratio of Ising 42 N=100\n' +
             ' with varying schedule, ' + str(n_boot) + ' bootstrap re-samples, and sweeps')
plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))
ax.set(xscale='log')
ax.set(ylim=[0.8, 1.01])
# ax.set(xlim=[1,200])


# %% [markdown]
# Now, besides looking at the sweeps, which are our parameter, we want to see how the performance changes with respect to the number of shots, which in this case are proportional to the computational time/effort that it takes to solve the problem.

# %%
fig, ax = plt.subplots()
for boot in boots:
    reads = [s * boot for s in sweeps]
    for schedule in schedules:
        ax.plot(reads, results['best'][boot][schedule], label=str(
            schedule) + ' with ' + str(boot) + ' reads')
        bestnp = np.stack(results['bestci'][boot][schedule], axis=0).T
        ax.fill_between(reads, bestnp[0], bestnp[1], alpha=0.25)
ax.set(xlabel='Total number of reads')
ax.set(ylabel='Approximation ratio = \n ' +
       '(best found - random sample) / (min energy - random sample)')
ax.set_title('Simulated annealing approximation ratio of Ising 42 N=100\n' +
             ' with varying schedule, ' + str(n_boot) + ' bootstrap re-samples, and sweeps')
plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))
ax.set(xscale='log')
ax.set(ylim=[0.8, 1.01])
# ax.set(xlim=[1,200])


# %%
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Simulated annealing expected runtime of \n' +
             ' instance ' + str(instance) + ' Ising N=100 with varying schedule and sweeps')

for schedule in schedules:
    ax1.plot(sweeps, results['t'][schedule], '-', label=schedule)
ax1.hlines(results['t']['geometric'][-1], sweeps[0], sweeps[-1],
           linestyle='--', label='default', colors='b')

ax1.set(ylabel='Time [s]')
# ax1.set(xlim=[1,200])


for schedule in schedules:
    ax2.semilogy(sweeps, results['p'][default_sweeps]
                 [schedule], '-', label=schedule)
ax2.hlines(results['p'][default_sweeps]['geometric'][-1], sweeps[0], sweeps[-1],
           linestyle='--', label='default', colors='b')
# ax2.set(xlim=[1,200])

ax2.set(ylabel='Success Probability \n (within ' +
        str(treshold) + ' % of best found)')
ax2.set(xlabel='Sweeps')
plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.3))

# Add plot going all the way to 1000 sweeps


# %%
fig1, ax1 = plt.subplots()

for boot in reversed(boots):
    for schedule in schedules:
        ax1.plot(sweeps, results['tts'][boot][schedule],
                 label=schedule + "_boot" + str(boot))
        ttsnp = np.stack(results['ttsci'][boot][schedule], axis=0).T
        ax1.fill_between(sweeps, ttsnp[0], ttsnp[1], alpha=0.25)


ax1.hlines(results['tts'][total_reads]['geometric'][-1], sweeps[0], sweeps[-1],
           linestyle='--', label='default', colors='b')

ax1.set(yscale='log')
ax1.set(ylim=[3, 1e3])
# ax1.set(xlim=[1,200])

ax1.set(ylabel='Time To Solution within ' +
        str(treshold) + ' % of best found [s]')
ax1.set(xlabel='Sweeps')
ax.set_title('Simulated annealing expected runtime of random Ising N=100\n' +
             ' with varying schedule, ' + str(n_boot) + ' bootstrap re-samples, and sweeps')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
           ncol=2, fancybox=False, shadow=False)

ax2 = plt.axes([.45, .55, .4, .3])
for schedule in schedules:
    min_tts = min(results['tts'][default_sweeps][schedule])
    min_index = results['tts'][default_sweeps][schedule].index(min_tts)
    min_sweep = sweeps[results['tts'][default_sweeps][schedule].index(min_tts)]
    print("minimum TTS for " + schedule + " schedule = " +
          str(min_tts) + "s at sweep = " + str(min_sweep))
    for boot in reversed(boots):
        ax2.semilogy(sweeps[min_index-10:min_index+10], results['tts']
                     [boot][schedule][min_index-10:min_index+10], '-s')
ax2.hlines(results['tts'][default_sweeps]['geometric'][-1], sweeps[min_index-10], sweeps[min_index+10],
           linestyle='--', label='default', colors='b')

ax2.set(ylabel='TTS [s]')
ax2.set(xlabel='Sweeps')


# %%
min_beta_schedule = np.geomspace(
    *simAnnSamplesDefault.info['beta_range'], num=min_sweep)
fig, ax = plt.subplots()
ax.plot(beta_schedule, '.')
ax.plot(min_beta_schedule, '.')
ax.set_xlabel('Sweeps')
ax.set_ylabel('beta=Inverse temperature')
ax.set_title('Geometric temperature schedule')
plt.legend(['Default', 'Best'])


# %%
fig, ax = plt.subplots()
# Best of boot samples es computed via n_boot bootstrapping
n_boot = 500
boots = range(1, 1000, 1)
interest_sweeps = [min_sweep, default_sweeps, 10, 500]
approx_ratio = {}
approx_ratioci = {}

for schedul in schedules:
    approx_ratio[schedule] = {}
    approx_ratioci[schedule] = {}

# Gather instance names
instance = 42
for sweep in interest_sweeps:
    for schedule in schedules:
        if sweep in approx_ratio[schedule] and sweep in approx_ratioci[schedule]:
            pass
        else:
            min_energy = results['min_energy'][schedule]
            random_energy = results['random_energy'][schedule]

            pickle_name = str(instance) + "_" + schedule + \
                "_" + str(sweep) + ".p"
            pickle_name = os.path.join(pickle_path, pickle_name)
            # If the instance data exists, load the data
            if os.path.exists(pickle_name) and not overwrite_pickles:
                # print(pickle_name)
                samples = pickle.load(open(pickle_name, "rb"))
                time_s = samples.info['timing']
            # If it does not exist, generate the data
            else:
                start = time.time()
                samples = simAnnSampler.sample(
                    model_random, num_reads=total_reads, num_sweeps=sweep, beta_schedule_type=schedule)
                time_s = time.time() - start
                samples.info['timing'] = time_s
                pickle.dump(samples, open(pickle_name, "wb"))
            # Compute statistics
            energies = samples.data_vectors['energy']
            if min(energies) < min_energy:
                min_energy = min(energies)
                print("A better solution of " + str(min_energy) +
                      " was found for sweep " + str(sweep))

            b = []
            bcs = []
            probs = []
            time_to_sol = []
            for boot in boots:
                boot_dist = []
                pr_dist = []
                for i in range(int(n_boot - boot + 1)):
                    resampler = np.random.randint(0, total_reads, boot)
                    sample_boot = energies.take(resampler, axis=0)
                    # Compute the best along that axis
                    boot_dist.append(min(sample_boot))

                b.append(np.mean(boot_dist))
                # Confidence intervals from bootstrapping the best out of boot
                bnp = np.array(boot_dist)
                cilo = np.apply_along_axis(
                    stats.scoreatpercentile, 0, bnp, 50.-ci/2.)
                ciup = np.apply_along_axis(
                    stats.scoreatpercentile, 0, bnp, 50.+ci/2.)
                bcs.append((cilo, ciup))

            approx_ratio[schedule][sweep] = [
                (random_energy - energy) / (random_energy - min_energy) for energy in b]
            approx_ratioci[schedule][sweep] = [tuple((random_energy - element) / (
                random_energy - min_energy) for element in energy) for energy in bcs]

        ax.plot([shot*sweep for shot in boots], approx_ratio[schedule]
                [sweep], label=str(sweep) + ' sweeps')
        approx_ratio_bestci_np = np.stack(
            approx_ratioci[schedule][sweep], axis=0).T
        ax.fill_between([shot*sweep for shot in boots],
                        approx_ratio_bestci_np[0], approx_ratio_bestci_np[1], alpha=0.25)
ax.set(xscale='log')
ax.set(ylim=[0.9, 1.01])
ax.set(xlim=[1e2, 1e4])
ax.set(xlabel='Total number of reads (equivalent to time)')
ax.set(ylabel='Approximation ratio = \n ' +
       '(best found - random sample) / (min energy - random sample)')
ax.set_title('Simulated annealing approximation ratio of Ising ' + str(instance) + ' N=100\n' +
             ' with varying schedule, ' + str(n_boot) + ' bootstrap re-samples, and sweeps')
plt.legend()


# %% [markdown]
# Here see how using the optimal number of sweeps is better than using other values (including the default recommended by the solver) in terms of solving this problem.
# Obviously, we only know this after running the experiments and verifying it ourselves. This is not the usual case, so we want to see how well can we do if we solve similar (but no the same instances).
# Here we will generate 20 random instances from the same distribution and size but different random seed.

# %%
overwrite_pickles = False
s = 0.99  # This is the success probability for the TTS calculation
treshold = 5.0  # This is a percentual treshold of what the minimum energy should be
sweeps = list(chain(np.arange(1, 250, 1), np.arange(250, 1001, 10)))
# schedules = ['geometric', 'linear']
schedules = ['geometric']
total_reads = 1000
default_sweeps = 1000
# boots = [1, 10, 100, default_sweeps]
boots = [1, 10, default_boots]
all_results = {}
instances = range(20)

all_results_name = "all_results.pkl"
all_results_name = os.path.join(pickle_path, all_results_name)
# If you wanto to use the raw data and process it here
if use_raw_data or not(os.path.exists(all_results_name)):

    for instance in instances:
        all_results[instance] = {}
        all_results[instance]['p'] = {}
        all_results[instance]['min_energy'] = {}
        all_results[instance]['random_energy'] = {}
        all_results[instance]['tts'] = {}
        all_results[instance]['ttsci'] = {}
        all_results[instance]['t'] = {}
        all_results[instance]['best'] = {}
        all_results[instance]['bestci'] = {}

        # Fixing the random seed to get the same result
        np.random.seed(instance)
        J = np.random.rand(N, N)
        # We only consider upper triangular matrix ignoring the diagonal
        J = np.triu(J, 1)
        h = np.random.rand(N)
        model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

        randomSample = randomSampler.sample(
            model_random, num_reads=total_reads)
        random_energies = [datum.energy for datum in randomSample.data(
            ['energy'])]
        random_energy = np.mean(random_energies)

        default_pickle_name = str(instance) + "_geometric_1000.p"
        default_pickle_name = os.path.join(pickle_path, default_pickle_name)
        if os.path.exists(default_pickle_name) and not overwrite_pickles:
            simAnnSamplesDefault = pickle.load(open(default_pickle_name, "rb"))
            timeDefault = simAnnSamplesDefault.info['timing']
        else:
            start = time.time()
            simAnnSamplesDefault = simAnnSampler.sample(
                model_random, num_reads=1000)
            timeDefault = time.time() - start
            simAnnSamplesDefault.info['timing'] = timeDefault
            pickle.dump(simAnnSamplesDefault, open(default_pickle_name, "wb"))
        energies = [datum.energy for datum in simAnnSamplesDefault.data(
            ['energy'], sorted_by='energy')]
        min_energy = energies[0]
        for schedule in schedules:

            all_results[instance]['t'][schedule] = {}
            all_results[instance]['min_energy'][schedule] = {}
            all_results[instance]['random_energy'][schedule] = {}
            all_results[instance]['p'][schedule] = {}
            all_results[instance]['tts'][schedule] = {}
            all_results[instance]['ttsci'][schedule] = {}
            all_results[instance]['best'][schedule] = {}
            all_results[instance]['bestci'][schedule] = {}

            # probs = []
            probs = {k: [] for k in boots}
            time_to_sol = {k: [] for k in boots}
            prob_np = {k: [] for k in boots}
            ttscs = {k: [] for k in boots}
            times = []
            b = {k: [] for k in boots}
            bnp = {k: [] for k in boots}
            bcs = {k: [] for k in boots}
            for sweep in sweeps:
                # Gather instance names
                pickle_name = str(instance) + "_" + \
                    schedule + "_" + str(sweep) + ".p"
                pickle_name = os.path.join(pickle_path, pickle_name)
                # If the instance data exists, load the data
                if os.path.exists(pickle_name) and not overwrite_pickles:
                    samples = pickle.load(open(pickle_name, "rb"))
                    time_s = samples.info['timing']
                # If it does not exist, generate the data
                else:
                    start = time.time()
                    samples = simAnnSampler.sample(
                        model_random, num_reads=1000, num_sweeps=sweep, beta_schedule_type=schedule)
                    time_s = time.time() - start
                    samples.info['timing'] = time_s
                    pickle.dump(samples, open(pickle_name, "wb"))
                # Compute statistics
                energies = samples.data_vectors['energy']
                occurrences = samples.data_vectors['num_occurrences']
                total_counts = sum(occurrences)
                times.append(time_s)
                if min(energies) < min_energy:
                    min_energy = min(energies)
                    # print("A better solution of " + str(min_energy) + " was found for sweep " + str(sweep))
                # success = min_energy*(1.0 + treshold/100.0)**np.sign(min_energy)
                success = random_energy - \
                    (random_energy - min_energy)*(1.0 - treshold/100.0)

                # Best of boot samples es computed via n_boot bootstrapping
                ci = 68
                boot_dist = {}
                pr_dist = {}
                cilo = {}
                ciup = {}
                pr = {}
                pr_cilo = {}
                pr_ciup = {}
                for boot in boots:
                    boot_dist[boot] = []
                    pr_dist[boot] = []
                    for i in range(int(n_boot)):
                        resampler = np.random.randint(0, total_reads, boot)
                        sample_boot = energies.take(resampler, axis=0)
                        # Compute the best along that axis
                        boot_dist[boot].append(min(sample_boot))

                        occurences = occurrences.take(resampler, axis=0)
                        counts = {}
                        for index, energy in enumerate(sample_boot):
                            if energy in counts.keys():
                                counts[energy] += occurences[index]
                            else:
                                counts[energy] = occurences[index]
                        pr_dist[boot].append(
                            sum(counts[key] for key in counts.keys() if key < success)/boot)
                    prob_np[boot] = np.array(pr_dist[boot])
                    pr[boot] = np.mean(prob_np[boot])
                    probs[boot].append(pr[boot])

                    b[boot].append(np.mean(boot_dist[boot]))
                    # Confidence intervals from bootstrapping the best out of boot
                    bnp[boot] = np.array(boot_dist[boot])
                    cilo[boot] = np.apply_along_axis(
                        stats.scoreatpercentile, 0, bnp[boot], 50.-ci/2.)
                    ciup[boot] = np.apply_along_axis(
                        stats.scoreatpercentile, 0, bnp[boot], 50.+ci/2.)
                    bcs[boot].append((cilo[boot], ciup[boot]))
                    # Confidence intervals from bootstrapping the TTS of boot
                    if prob_np[boot].all() == 0:
                        time_to_sol[boot].append(np.inf)
                        ttscs[boot].append((np.inf, np.inf))
                    else:
                        pr_cilo[boot] = np.apply_along_axis(
                            stats.scoreatpercentile, 0, prob_np[boot], 50.-ci/2.)
                        pr_ciup[boot] = np.apply_along_axis(
                            stats.scoreatpercentile, 0, prob_np[boot], 50.+ci/2.)
                        time_to_sol[boot].append(
                            time_s*math.log10(1-s)/math.log10(1-pr[boot]+1e-9))
                        ttscs[boot].append((time_s*math.log10(1-s)/math.log10(
                            1-pr_cilo[boot]+1e-9), time_s*math.log10(1-s)/math.log10(1-pr_ciup[boot]+1e-9)))

            all_results[instance]['t'][schedule][default_boots] = times
            all_results[instance]['min_energy'][schedule][default_boots] = min_energy
            all_results[instance]['random_energy'][schedule][default_boots] = random_energy
            for boot in boots:
                all_results[instance]['p'][schedule][boot] = probs[boot]
                all_results[instance]['tts'][schedule][boot] = time_to_sol[boot]
                all_results[instance]['ttsci'][schedule][boot] = ttscs[boot]
                all_results[instance]['best'][schedule][boot] = [
                    (random_energy - energy) / (random_energy - min_energy) for energy in b[boot]]
                all_results[instance]['bestci'][schedule][boot] = [tuple(
                    (random_energy - element) / (random_energy - min_energy) for element in energy) for energy in bcs[boot]]

    # Save results file in case that we are interested in reusing them
    pickle.dump(all_results, open(all_results_name, "wb"))
else:  # Just reload processed datafile
    all_results = pickle.load(open(all_results_name, "rb"))


# %%
def bootstrap(data, n_boot=1000, ci=68):
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        # Median ignoring nans instead of mean
        boot_dist.append(np.nanmedian(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return (s1, s2)


def tsplotboot(ax, x, data, error_est, **kw):
    if x is None:
        x = np.arange(data.shape[1])
    # Median ignoring nans instead of mean
    est = np.nanmedian(data, axis=0)
    mask = ~np.isnan(est)
    if error_est == 'bootstrap':
        cis = bootstrap(data)
    elif error_est == 'std':
        sd = np.nanstd(data, axis=0)
        cis = (est - sd, est + sd)
    ax.fill_between(x[mask], cis[0][mask], cis[1][mask], alpha=0.35, **kw)
    ax.plot(x[mask], est[mask], **kw)
    ax.margins(x=0)


# %% [markdown]
# Now we bootstrap our solutions with respect to the whole set of isntances, or ensemble, and we use the median which respresents the solution better than the mean.

# %%
fig, ax1 = plt.subplots()
for boot in reversed(boots):
    for schedule in schedules:
        results_array = np.array(
            [np.array(all_results[i]['tts'][schedule][boot]) for i in range(20)])
        tsplotboot(ax1, x=np.asarray(sweeps), data=results_array, error_est='bootstrap',
                   label="Ensemble " + schedule + ' with ' + str(boot) + ' reads')

        ax1.plot(sweeps, results['tts'][boot][schedule],
                 label="Instance " + schedule + "_boot" + str(boot))


ax1.set(yscale='log')

ax1.set(ylabel='Time To Solution within ' +
        str(treshold) + ' % of best found [s]')
ax1.set(xlabel='Sweeps')
plt.title('Simulated annealing expected runtime of \n' +
          ' Ising 42 N=100 with varying schedule and sweeps')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
           ncol=3, fancybox=False, shadow=False)

ax2 = plt.axes([.45, .55, .4, .3])
for schedule in schedules:
    results_array = np.array(
        [np.array(all_results[i]['tts'][schedule][default_boots]) for i in range(20)])
    min_median_tts = min(np.nanmedian(results_array, axis=0))
    min_median_index = np.argmin(np.nanmedian(results_array, axis=0))
    min_median_sweep = sweeps[min_median_index]
    min_tts = min(results['tts'][default_boots][schedule])
    min_index = results['tts'][default_boots][schedule].index(min_tts)
    min_sweep = sweeps[results['tts'][default_boots][schedule].index(min_tts)]
    if min_sweep < min_median_sweep:
        index_lo = min_index
        index_hi = min_median_index
    else:
        index_lo = min_median_index
        index_hi = min_index
    print("minimum median TTS for " + schedule + " schedule = " +
          str(min_median_tts) + "s at sweep = " + str(min_median_sweep))
    ax2.semilogy(sweeps[index_lo-5:index_hi+5],
                 np.median([all_results[i]['tts'][schedule][default_boots]
                           for i in range(20)], axis=0)
                 [index_lo-5:index_hi+5], '-s')
    print("minimum TTS for instance 42 with " + schedule +
          " schedule = " + str(min_tts) + "s at sweep = " + str(min_sweep))
    ax2.semilogy(sweeps[index_lo-5:index_hi+5],
                 results['tts'][default_sweeps][schedule][index_lo-5:index_hi+5], '-s')


ax2.hlines(np.median([all_results[i]['tts'][schedule][default_sweeps][-1] for i in range(20)]),
           sweeps[index_lo-5], sweeps[index_hi+5],
           linestyle='--', label='default', colors='b')

ax2.set(ylabel='TTS [s]')
ax2.set(xlabel='Sweeps')


# %%
fig, ax = plt.subplots()
for boot in boots:
    for schedule in schedules:
        best_array = np.array(
            [np.array(all_results[i]['best'][schedule][boot]) for i in range(20)])
        # min_median_best = min(np.nanmedian(results_array, axis=0))
        tsplotboot(ax, x=np.asarray(sweeps), data=best_array, error_est='bootstrap',
                   label="Ensemble " + schedule + ' with ' + str(boot) + ' reads')
ax.set(xlabel='Sweeps')
ax.set(ylabel='Approximation ratio \n = best found / min energy')
ax.set_title('Simulated annealing approximation ratio of \n' +
             ' Ensemble of Ising N=100 with varying schedule, sweeps, and number of reads')
plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))
ax.set(xscale='log')
ax.set(ylim=[0.8, 1.01])
# ax.set(xlim=[1,200])


# %%
fig, ax = plt.subplots()
for boot in boots:
    reads = [s * boot for s in sweeps]
    for schedule in schedules:
        best_array = np.array(
            [np.array(all_results[i]['best'][schedule][boot]) for i in range(20)])
        # min_median_best = min(np.nanmedian(results_array, axis=0))
        tsplotboot(ax, x=np.asarray(reads), data=best_array, error_est='bootstrap',
                   label="Ensemble " + schedule + ' with ' + str(boot) + ' reads')
ax.set(xlabel='Total number of reads')
ax.set(ylabel='Approximation ratio \n = best found / min energy')
ax.set_title('Simulated annealing approximation ratio of \n' +
             ' Ensemble of Ising N=100 with varying schedule, sweeps, and number of reads')
plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))
ax.set(xscale='log')
ax.set(ylim=[0.8, 1.01])
# ax.set(xlim=[1,200])


# %%
fig, ax = plt.subplots()
instances = np.arange(20)
default_boots = 1000
indices = [np.argmin(all_results[i]['tts']['geometric']
                     [default_boots]) for i in range(20)]
min_sweeps = [sweeps[i] for i in indices]
print('Optimal sweeps for each instance ' + str(min_sweeps))
minima = [np.min(all_results[i]['tts']['geometric'][default_boots])
          for i in range(20)]
default = [all_results[i]['tts']['geometric'][default_sweeps][-1]
           for i in range(20)]
median_all = [all_results[i]['tts']['geometric']
              [default_boots][min_median_index] for i in range(20)]


ax.bar(instances-0.2, minima, width=0.2, color='b',
       align='center', label='virtual best')
ax.bar(instances, median_all, width=0.2,
       color='g', align='center', label='median')
ax.bar(instances+0.2, default, width=0.2,
       color='r', align='center', label='default')
ax.xaxis.get_major_locator().set_params(integer=True)
plt.xlabel('Instance')
ax.set(ylabel='Time To Solution within ' +
       str(treshold) + ' % of best found [s]')
plt.legend()


# %% [markdown]
# Notice how much performance would we be losing if we had used the default value for all these instances, and how much we could eventually win if we knew the best for each.

# %%
fig, ax = plt.subplots()
interest_sweeps = [min_sweep, total_reads, 10, 500]
interest_sweeps.append(min_median_sweep)
interest_sweeps.append(120)
shots = range(1, 1000, 1)
n_boot = 100

# Gather instance names
instance = 42
for sweep in interest_sweeps:
    for schedule in schedules:
        if sweep in approx_ratio[schedule] and sweep in approx_ratioci[schedule]:
            pass
        else:
            min_energy = results['min_energy'][schedule]
            random_energy = results['random_energy'][schedule]

            pickle_name = str(instance) + "_" + schedule + \
                "_" + str(sweep) + ".p"
            pickle_name = os.path.join(pickle_path, pickle_name)
            # If the instance data exists, load the data
            if os.path.exists(pickle_name) and not overwrite_pickles:
                # print(pickle_name)
                samples = pickle.load(open(pickle_name, "rb"))
                time_s = samples.info['timing']
            # If it does not exist, generate the data
            else:
                start = time.time()
                samples = simAnnSampler.sample(
                    model_random, num_reads=total_reads, num_sweeps=sweep, beta_schedule_type=schedule)
                time_s = time.time() - start
                samples.info['timing'] = time_s
                pickle.dump(samples, open(pickle_name, "wb"))
            # Compute statistics
            energies = samples.data_vectors['energy']
            if min(energies) < min_energy:
                min_energy = min(energies)
                print("A better solution of " + str(min_energy) +
                      " was found for sweep " + str(sweep))
                results['min_energy'][schedule] = min_energy

            b = []
            bcs = []
            probs = []
            time_to_sol = []
            for shot in shots:
                shot_dist = []
                pr_dist = []
                for i in range(int(n_boot - shot + 1)):
                    resampler = np.random.randint(0, total_reads, shot)
                    sample_shot = energies.take(resampler, axis=0)
                    # Compute the best along that axis
                    shot_dist.append(min(sample_shot))

                b.append(np.mean(shot_dist))
                # Confidence intervals from bootstrapping the best out of shot
                bnp = np.array(shot_dist)
                cilo = np.apply_along_axis(
                    stats.scoreatpercentile, 0, bnp, 50.-ci/2.)
                ciup = np.apply_along_axis(
                    stats.scoreatpercentile, 0, bnp, 50.+ci/2.)
                bcs.append((cilo, ciup))

            approx_ratio[schedule][sweep] = [
                (random_energy - energy) / (random_energy - min_energy) for energy in b]
            approx_ratioci[schedule][sweep] = [tuple((random_energy - element) / (
                random_energy - min_energy) for element in energy) for energy in bcs]

        ax.plot([shot*sweep for shot in shots], approx_ratio[schedule]
                [sweep], label=str(sweep) + ' sweeps')
        approx_ratio_bestci_np = np.stack(
            approx_ratioci[schedule][sweep], axis=0).T
        ax.fill_between([shot*sweep for shot in shots],
                        approx_ratio_bestci_np[0], approx_ratio_bestci_np[1], alpha=0.25)
ax.set(xscale='log')
ax.set(ylim=[0.95, 1.001])
ax.set(xlim=[1e2, 1e4])
ax.set(xlabel='Total number of reads (equivalent to time)')
ax.set(ylabel='Approximation ratio = \n ' +
       '(best found - random sample) / (min energy - random sample)')
ax.set_title('Simulated annealing approximation ratio of Ising ' + str(instance) + ' N=100\n' +
             ' with varying schedule, ' + str(n_boot) + ' bootstrap re-samples, and sweeps')
plt.legend()


# %% [markdown]
# This example shows that for our beloved instance 42, using the mean of the best accross the ensemble is better than using the default, but not as good as if we knew from scratch what would have made the best case.

# %% [markdown]
# After figuring out what would be the best parameter for our instance of interest, it would be nice to see what the ensemble performance is. We have several choices, either going with the (arbitrary) default values, or using the mean of the best performance we have found up to that point. There is an unachievable goal, which would be the case where we knew the best solution of each case, which we call the virtual best.
# This helps us understand how much is at stake with the choice of parameters we make.

# %%
fig, ax = plt.subplots()
default_sweeps = 1000
interest_sweeps = [min_median_sweep, default_sweeps]
interest_sweeps.append('best')
# n_boot = 100
# Careful, this cell is particularly expensive, so try it with n_boot = 100


all_approx_ratio = {}

# Gather instance names
for instance in instances:
    all_approx_ratio[instance] = {}
    for schedul in schedules:
        all_approx_ratio[instance][schedule] = {}

    for sweep in interest_sweeps:
        flag_best = False
        if sweep == 'best':
            sweep = sweeps[indices[instance]]
            flag_best = True
        for schedule in schedules:
            if sweep in all_approx_ratio[instance][schedule]:
                pass
            else:
                min_energy = all_results[instance]['min_energy'][schedule][default_sweeps]
                random_energy = all_results[instance]['random_energy'][schedule][default_sweeps]

                pickle_name = str(instance) + "_" + \
                    schedule + "_" + str(sweep) + ".p"
                pickle_name = os.path.join(pickle_path, pickle_name)
                # If the instance data exists, load the data
                if os.path.exists(pickle_name) and not overwrite_pickles:
                    # print(pickle_name)
                    samples = pickle.load(open(pickle_name, "rb"))
                    time_s = samples.info['timing']
                # If it does not exist, generate the data
                else:
                    start = time.time()
                    samples = simAnnSampler.sample(
                        model_random, num_reads=total_reads, num_sweeps=sweep, beta_schedule_type=schedule)
                    time_s = time.time() - start
                    samples.info['timing'] = time_s
                    pickle.dump(samples, open(pickle_name, "wb"))
                # Compute statistics
                energies = samples.data_vectors['energy']
                if min(energies) < min_energy:
                    min_energy = min(energies)
                    print("A better solution of " + str(min_energy) +
                          " was found for sweep " + str(sweep))

                b = []
                for shot in shots:
                    shot_dist = []
                    for i in range(int(n_boot - shot + 1)):
                        resampler = np.random.randint(0, total_reads, shot)
                        sample_shot = energies.take(resampler, axis=0)
                        # Compute the best along that axis
                        shot_dist.append(min(sample_shot))

                    b.append(np.mean(shot_dist))
            if flag_best:
                all_approx_ratio[instance][schedule]['best'] = [
                    (random_energy - energy) / (random_energy - min_energy) for energy in b]
                ax.plot([shot*sweep for shot in shots], all_approx_ratio[instance]
                        [schedule]['best'], color='lightgray', label=None, alpha=0.5)
            else:
                all_approx_ratio[instance][schedule][sweep] = [
                    (random_energy - energy) / (random_energy - min_energy) for energy in b]
                ax.plot([shot*sweep for shot in shots], all_approx_ratio[instance]
                        [schedule][sweep], color='lightgray', label=None, alpha=0.25)

for sweep in interest_sweeps:
    approx_ratio_array = np.array(
        [np.array(all_approx_ratio[i][schedule][sweep]) for i in instances])
    label_plot = "Ensemble " + schedule + ' with ' + str(sweep) + ' sweeps'
    if sweep == 'best':
        sweep = sweeps[indices[instance]]
        label_plot = "Ensemble " + schedule + ' with virtual best sweeps'
    reads = [shot*sweep for shot in shots]
    tsplotboot(ax, x=np.asarray(reads), data=approx_ratio_array,
               error_est='bootstrap', label=label_plot)

ax.set(xscale='log')
ax.set(ylim=[0.95, 1.001])
ax.set(xlim=[1e2, 1e4])
ax.set(xlabel='Total number of reads (equivalent to time)')
ax.set(ylabel='Approximation ratio = \n ' +
       '(best found - random sample) / (min energy - random sample)')
ax.set_title('Simulated annealing approximation ratio of Ising Ensemble N=100\n' +
             ' with varying schedule, ' + str(n_boot) + ' bootstrap re-samples, and sweeps')
plt.legend()


# %% [markdown]
# As you can see, there is a gap between the case with the best mean performance and the virtual best. This difference is arguably small, but you can imagine that with a larger number of parameters, this difference can become larger and larger, making the search of good parameters more worthy and complicated. We are actively working on such parameter setting strategies, and expect to make progress in this area (keep tuned!).

# %%
results_array
min_median_tts = min(np.nanmedian(results_array, axis=0))
print(min_median_tts)


# %%
