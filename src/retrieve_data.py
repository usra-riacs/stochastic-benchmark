
import os
from typing import List, Union

# Functions to retrieve instances files
# Define functions to extract data files

def getInstances(filename):  # instance number
    '''
    Extracts the instance from the filename assuming it is at the end before extension

    Args:
        filename: the name of the file

    Returns:
        instance: the instance number
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 2)[-1])

# TODO rename all variables *_list as plurals


def createInstanceFileList(directory, instance_list):
    '''
    Creates a list of files in the directory for the instances in the list

    Args:
        directory: the directory where the files are
        instance_list: the list of instances

    Returns:
        instance_file_list: the list of files
    '''
    fileList = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and not f.endswith('.zip') and not f.endswith('.sh'))]
        # exclude gs_energies.txt files
        files = [f for f in files
                 if(not f.startswith('gs_energies'))]
        # Below, select only specifed n,s,alpha instances
        files = [f for f in files if(getInstances(f) in instance_list)]
        for f in files:
            fileList.append(root+"/"+f)
    return fileList


def getInstancePySAExperiment(filename):  # instance number
    '''
    Extracts the instance number from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        instance: the instance number
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 9)[-9])


def getSweepsPySAExperiment(filename):
    '''
    Extracts the sweeps from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        sweeps: the number of sweeps
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 7)[-7])


def getPHot(filename):  # P hot
    '''
    Extracts the hot temperature transition probability from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        phot: the hot temperature transition probability
    '''
    return float(filename.rsplit(".", 1)[0].rsplit("_", 2)[-1])


def getPCold(filename):  # P cold
    '''
    Extracts the cold temperature transition probability from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        pcold: the cold temperature transition probability
    '''
    return float(filename.rsplit(".", 1)[0].rsplit("_", 3)[-3])


def getReplicas(filename):  # replicas
    '''
    Extracts the replicas from the PySA experiment filename assuming the filename follows the naming convention prefix_instance_sweeps_replicas_pcold_phot.extension

    Args:
        filename: the name of the file

    Returns:
        replicas: the number of replicas
    '''
    return int(filename.rsplit(".", 1)[0].rsplit("_", 5)[-5])


def createPySAExperimentFileList(
    directory: str,
    instance_list: Union[List[str], List[int]],
    rep_list: Union[List[str], List[int]] = None,
    sweep_list: Union[List[str], List[int]] = None,
    pcold_list: Union[List[str], List[float]] = None,
    phot_list: Union[List[str], List[float]] = None,
    prefix: str = "",
) -> list:
    '''
    Creates a list of experiment files in the directory for the instances in the instance_list, replicas in the rep_list, sweeps in the sweep_list, P cold in the pcold_list, and P hot in the phot_list

    Args:
        directory: the directory where the files are
        instance_list: the list of instances
        rep_list: the list of replicas
        sweep_list: the list of sweeps
        pcold_list: the list of P cold
        phot_list: the list of P hot
        prefix: the prefix of the files

    Returns:
        experiment_file_list: the list of files

    '''
    fileList = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and
                    not f.endswith('.zip') and
                    not f.endswith('.sh') and
                    not f.endswith('.p') and
                    f.startswith(prefix))]
        # exclude gs_energies.txt files
        files = [f for f in files
                 if(not f.startswith('gs_energies'))]
        # Below, select only specifed instances
        files = [f for f in files if(
            getInstancePySAExperiment(f) in instance_list)]
        # Consider replicas if provided list
        if rep_list is not None:
            files = [f for f in files if(
                getReplicas(f) in rep_list)]
        # Consider sweeps if provided list
        if sweep_list is not None:
            files = [f for f in files if(
                getSweepsPySAExperiment(f) in sweep_list)]
        # Consider pcold if provided list
        if pcold_list is not None:
            files = [f for f in files if(
                getPCold(f) in pcold_list)]
        # Consider phot if provided list
        if phot_list is not None:
            files = [f for f in files if(
                getPHot(f) in phot_list)]
        for f in files:
            fileList.append(root+"/"+f)

        # sort filelist by instance
        fileList = sorted(fileList, key=lambda x: getInstancePySAExperiment(x))
    return fileList


def getSchedule(filename,prefix):
    '''
    Extracts the schedule from the Dwave-neal experiment filename assuming the filename follows the naming convention prefix_instance_schedule_sweeps.extension

    Args:
        filename: the name of the file
        prefix: the prefix of the files

    Returns:
        schedule: the schedule string
    '''
    return filename.rsplit(".", 1)[0].split(prefix, 1)[1].split("_")[1]


def getSweepsDnealExperiment(filename,prefix):
    '''
    Extracts the sweeps from the Dwave-neal experiment filename assuming the filename follows the naming convention prefix_instance_schedule_sweeps.extension

    Args:
        filename: the name of the file
        prefix: the prefix of the file

    Returns:
        sweep: the schedule string
    '''
    return int(filename.rsplit(".", 1)[0].split(prefix)[1].split("_")[2])


def getInstanceDnealExperiment(filename,prefix):
    '''
    Extracts the instance from the Dwave-neal experiment filename assuming the filename follows the naming convention prefix_instance_schedule_sweeps.extension

    Args:
        filename: the name of the file
        prefix: the prefix of the files

    Returns:
        sweep: the sweep string
    '''
    return int(filename.rsplit(".", 1)[0].split(prefix, 1)[1].split("_")[0])


def createDnealExperimentFileList(
    directory: str,
    instance_list: Union[List[str], List[int]],
    sweep_list: Union[List[str], List[int]] = None,
    schedule_list: List[str] = None,
    prefix: str = "",
    suffix: str = "",
) -> list:
    '''
    Creates a list of experiment files in the directory for the instances in the instance_list, sweeps in the sweep_list, and schedules in the schedule_list

    Args:
        directory: the directory where the files are
        instance_list: the list of instances
        sweep_list: the list of sweeps
        schedule_list: the list of schedules
        prefix: the prefix of the experiment files

    Returns:
        experiment_file_list: the list of files

    '''
    fileList = []
    for root, dirs, files in os.walk(directory):
        # exclude hidden, compressed, or bash files
        files = [f for f in files
                 if(not f.startswith('.') and
                    not f.endswith('.zip') and
                    not f.endswith('.sh') and
                    f.endswith(suffix) and
                    f.startswith(prefix))]
        # exclude gs_energies.txt files
        files = [f for f in files
                 if(not f.startswith('gs_energies'))]
        # Below, select only specifed instances
        files = [f for f in files if(
            getInstanceDnealExperiment(f,prefix) in instance_list)]
        # Consider sweeps if provided list
        if sweep_list is not None:
            files = [f for f in files if(
                getSweepsDnealExperiment(f,prefix) in sweep_list)]
        # Consider schedules if provided list
        if schedule_list is not None:
            files = [f for f in files if(
                getSchedule(f,prefix) in schedule_list)]
        for f in files:
            fileList.append(root+"/"+f)

        # sort filelist by instance
        fileList = sorted(
            fileList, key=lambda x: getInstanceDnealExperiment(x,prefix))
    return fileList



# %%
# Function to load ground state solutions from solution file gs_energies.txt


def loadEnergyFromFile(data_file, instance_name):
    '''
    Loads the minimum energy of a given instance from file gs_energies.txt

    Args:
        data_file: The file to load the energies from.
        instance_name: The name of the instance to load the energy for.

    Returns:
        The minimum energy of the instance.

    '''
    energies = []
    with open(data_file, "r") as fin:
        for line in fin:
            if(line.split()[0] == instance_name):
                energies.append(float(line.split()[1]))

    if len(energies) == 0:
        print("No energy found for instance: " + instance_name)
        return None
    return min(energies)
