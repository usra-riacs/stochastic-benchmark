# %%
import dimod
import numpy as np
import os
from typing import List, Union
import pandas as pd
from retrieve_data import *
# %%
# Define function to compute random sampled energy


def randomEnergySampler(
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
        if model.vartype == dimod.Vartype.BINARY:
            state = np.random.randint(2, size=(model.num_variables, num_reads))
        else:
            randomSample = np.random.randint(
                2, size=(model.num_variables, num_reads)) * 2 - 1
            energies = [model.energy(randomSample[:, i])
                        for i in range(num_reads)]
    return np.mean(energies), randomSample

# %%
# zip_name = os.path.join(dneal_results_path, 'results.zip')
# if os.path.exists(zip_name) and use_raw_dataframes:
#     import zipfile
#     with zipfile.ZipFile(zip_name, 'r') as zip_ref:
#         zip_ref.extractall(dneal_pickle_path)
#     print('Results zip file has been extrated to ' + dneal_pickle_path)

# %%

# %%
# Functionalize the file creation
# Create all instances and save it into disk
def createRandomSKModel(
    N: int,
    instance_list: List[int] = None,
    prefix: str = '',
    instance_path: str = '',
):
    '''
    Creates a random model and saves it into disk.

    Args:
        N: The number of variables in the model.
        instance_list: A list of integers to define the instances by setting the sandom seed in numpy.
        prefix: A string to prefix the file name.
        instance_path: A string to define the path to save the instances.
    '''
    for instance in instance_list:
        instance_file_name = prefix + str(instance) + ".txt"
        instance_file_name = os.path.join(instance_path, instance_file_name)

        if not os.path.exists(instance_file_name):
            # Fixing the random seed to get the same result
            np.random.seed(instance)
            J = np.random.rand(N, N)
            # We only consider upper triangular matrix ignoring the diagonal
            J = np.triu(J, 1)
            h = np.random.rand(N)
            model_random = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)

            text_file = open(instance_file_name, "w")
            text_file.write(model_random.to_coo())
            text_file.close()

# %%
# Compute random energy file
def computeRandomSK(
    N,
    prefix,
    instance_list,
    data_path,
    ):
    '''
    Compute the random energy file for Sherry-Kirkpatrick model.

    Args:
        N: The number of variables in the model.
        prefix: A string to prefix the file name.
        instance_list: A list of integers to define the instances by setting the sandom seed in numpy.
        data_path: A string to define the path to save the instances.
        
    '''
    for instance in instance_list:
        # Load problem instance
        np.random.seed(instance)
        J = np.random.rand(N, N)
        # We only consider upper triangular matrix ignoring the diagonal
        J = np.triu(J, 1)
        h = np.random.rand(N)
        ising_model = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.0)
        random_energy, _ = randomEnergySampler(
            ising_model, dwave_sampler=False)
        with open(os.path.join(data_path, "random_energies.txt"), "a") as gs_file:
            gs_file.write(prefix + str(instance) + " " +
                          str(random_energy) + " " + "random\n")



# %%
# Compute preliminary ground state file with best found solution by Dwave-neal
def computeDnealGS(
    prefix,
    instance_list,
    dneal_pickle_path,
    data_path,
    ):
    '''    
    Compute the preliminary ground state file with best found solution by Dwave-neal.

    Args:
        prefix: A string to prefix the file name.
        instance_list: A list of integers to define the instances by setting the sandom seed in numpy.
        dneal_pickle_path: A string to define the path where the Dneal pickle paths are.
        data_path: A string to define the path to save the instances.
    '''
    for instance in instance_list:
        # List all the pickled filed for an instance files
        pickle_list = createDnealExperimentFileList(
            directory=dneal_pickle_path,
            instance_list=[instance],
            prefix='df_' + prefix,
            suffix='.pkl'
        )
        min_energies = []
        min_energy = np.inf
        for file in pickle_list:
            df_samples = pd.read_pickle(file)
            if min_energy > df_samples['energy'].min():
                min_energy = df_samples['energy'].min()
                print(file)
                print(min_energy)
                min_energies.append(min_energy)
                min_df_samples = df_samples.copy()
        
        if len(min_energies) > 0:
            with open(os.path.join(data_path, "gs_energies.txt"), "a") as gs_file:
                gs_file.write(prefix + str(instance) + " " +
                            str(np.nanmin(min_energies)) + "  best_found dneal\n")
# %%
if __name__ == "__main__":
    
    # Create directories and specify instances details

    N = 200  # Number of variables
    prefix = "random_n_" + str(N) + "_inst_"
    instance_list = list(range(20)) + [42]
    # Specify and if non-existing, create directories for results
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../data/sk/')
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
        
    results_path = dneal_results_path

    create_instace_files = True
    if create_instace_files:
        instance_path = os.path.join(data_path, 'instances')
        createRandomSKModel(
            N=N,
            instance_list=instance_list,
            prefix=prefix,
            instance_path=data_path
        )

        
    compute_random = False
    if compute_random:
        computeRandomSK(N = N, prefix = prefix, instance_list = instance_list, data_path = data_path)
    compute_dneal_gs = False
    if compute_dneal_gs:
        computeDnealGS(prefix=prefix, instance_list=instance_list, dneal_pickle_path=dneal_pickle_path, data_path=data_path)
