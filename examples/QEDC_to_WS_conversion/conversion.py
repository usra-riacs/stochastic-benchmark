# Functions for storing and plotting windows stickers plots
# new functions

import pickle
import pandas as pd
import numpy as np
import os
import glob
import maxcut_benchmark

times_list = ["elapsed_time", "exec_time", "opt_exec_time", "create_time"]
cumul_time_list = ["cumul_"+ttype for ttype in times_list]
metrics_list = ["approx_ratio", 'bestcut_ratio', 'cvar_ratio', 'gibbs_ratio']
other_quantities_to_store_load = ["thetas_array", "unique_sizes", "unique_counts", "cumul_counts", "fidelity", "hf_fidelity", "quantile_optgaps"]
col_tuples = [("times", j) for j in times_list] + [("cumul times", j) for j in cumul_time_list] + [("metrics", j) for j in metrics_list] + [("other", j) for j in other_quantities_to_store_load]
col_multi_index = pd.MultiIndex.from_tuples(col_tuples)
all_quantities_list = times_list + cumul_time_list + metrics_list + other_quantities_to_store_load

def flatten(xss):
    "Flatten a list of lists"
    return [x for xs in xss for x in xs]


def json_to_pkl(folder, load_from_json = False, target_folder = None, target_file_name = None, monotonized = True):
    """
    If a pickle file exists, load and return gen_props and dataframe with only one metric (the one that was used as the approximation ratio).
    The returned and stored dataframe will have columns corresponding to each restart, and the row indices will be the cobyla iteration number
    Args:
        folder (str): folder location where the json files are stored
        target_folder (str): location where the pkl file is to be stored (or from which, the dataframe is to be loaded from). Default value is None, in which case the target_folder will be set to be the same as 'folder'.
        target_file_name (str): name of pkl file in which the dataframe will be stored. If None, it will be called all_metrics.pkl
        monotonized (bool): If True, the metric values are made non-decreasing as a function of the iteration number, for each restart
        

    Returns:
        gen_prop (dict) : of inputs to the run method
        df (dataframe) : Indices are minimizer iteration number, and columns are restart indices
    """
    
    def transform_col(col, monotone):
        """
        for every column, fill nan values with the value preceeding the first nan value
        """
        # Check if there is any nan value. If not, then return the column as is
        
                
        if not np.isnan(col).any():
            # do nothing
            col = col
        else:
            # Otherwise, find the last non-nan value
            pos = np.isnan(col).argmax() - 1
            fillval = col[pos]
            col[pos:] = fillval
        if monotone:
            col = np.maximum.accumulate(col)
        
        return col
    
    # Check if pkl file already exists
    if target_folder is None:
        target_folder = folder
    if target_file_name is None:
        target_file_name = 'all_metrics.pkl'
    pkl_file_name = os.path.join(target_folder, target_file_name)
    
    if os.path.exists(pkl_file_name) and not load_from_json:
        # load into memory and return values
        with open(pkl_file_name, "rb") as f:
            # df = pd.read_pickle(f)
            # gen_prop = pd.read_pickle(f)
            df = pickle.load(f)
            gen_prop = pickle.load(f)
    else:
        # First, load the data from the folder
        print(f"Loading from json files...")
        if not os.path.exists(target_folder) : os.makedirs(target_folder)
        
        gen_prop = maxcut_benchmark.load_all_metrics(folder)
        mini_iters = gen_prop['max_iter']
        restarts = gen_prop['max_circuits']
       
        # create a multiindex dataframe, indexed by 1) minimizer iteration number and 2)
        num_qubits = gen_prop['min_qubits']
        detail_2_width = metrics.circuit_metrics_detail_2[str(num_qubits)]
        # for each restart, get the time series of the metric
        # all_qtys_dict = {qty : [] for qty in all_quantities_list}
        metric = gen_prop['objective_func_type']
        metrics_array = np.empty((mini_iters, restarts))
        metrics_array[:] = np.nan
        # Now load quantities from metrics to the all_qtys_dict
        for restart_ind in range(restarts):
            # For each restart, get data
            restart_dict = detail_2_width[restart_ind + 1]
            for mini_iter_ind in restart_dict.keys():
                metrics_array[mini_iter_ind, restart_ind] = restart_dict[mini_iter_ind][metric]
        metrics_array = np.apply_along_axis(lambda col: transform_col(col, monotonized), 0, metrics_array)
        
        # Create the dataframe
        df = pd.DataFrame(metrics_array, columns=[i for i in range(1, restarts + 1)])
        df['iterations'] = list(range(1, mini_iters + 1)) # iterations will start with 1, since they are multiplied to other quantities in order to compute the resource value.
        df.set_index('iterations', inplace=True)
                        
        with open(pkl_file_name, "wb") as f:
            # df = pd.read_pickle(f)
            # gen_prop = pd.read_pickle(f)
            pickle.dump(df, f)
            pickle.dump(gen_prop, f)
        
    return df, gen_prop

def row_func(row, downsample, bootstrap_iterations, metric, confidence_level = 64):
    """Takes in a series (a row from a dataframe), and does bootstrapping, to obtain mean and confidence intervals. 
    Args:
        row (pd.Series): one row of dataframe (containing values for each instance)
        downsample (int): number of observations in each bootstrap sample
        bootstrap_iterations (int): number of bootstrap iters
    Returns:
        pd.Series: with indices Mean, CI_l, CI_u
    """
    fact = erfinv(confidence_level / 100.) * np.sqrt(2.)


    resamples = np.random.choice(len(row), size=(downsample, bootstrap_iterations))
    resamples = row.values[resamples]
    resamples = np.max(resamples, axis=0)
    std = np.std(resamples)
    mean = np.mean(resamples)
    CI_l = mean - fact * std
    CI_u = mean + fact * std
    key = "Key=" + metric
    return pd.Series([mean, CI_l, CI_u], index=[key, "ConfInt=lower_" + key, "ConfInt=upper_"+key])


def do_bootstrap(df, downsample, bootstrap_iterations, metric, confidence_level = 64):
    """To simulate smaller number of restarts, do bootstrapping.

    Args:
        df (dataframe): indices are cobyla iterations and columns are restarts
        downsample (int): number of elements in each sample
        confidence_interval (int/float): default is 64
    """
    new_df = df.apply(lambda row: row_func(row, downsample,bootstrap_iterations, metric, confidence_level = confidence_level), axis=1, result_type="expand")
    return new_df

def pkl_to_sb_pkl_bootstrapped(pkl_folder, bs_restarts_list, bootstrap_iterations, instance_number, confidence_level, sb_target_folder = "checkpoints", sb_target_file = ""):
    """Convert raw data from pkl format to bootstrapped data (also to be stored in pkl format)

    Args:
        pkl_folder (_type_): _description_
        bs_restarts_list (_type_): _description_
        bootstrap_iterations (_type_): _description_
        instance_number (_type_): _description_
        confidence_level (_type_): _description_
        sb_target_folder (str, optional): _description_. Defaults to "".
        sb_target_file (str, optional): _description_. Defaults to "".

    Returns:
        pandas.dataframe: 
    """
    # Read all files from the pkl_folder
    raw_data = glob.glob(os.path.join(pkl_folder, '*.pkl'))
    
    # Check if pkl file and folder already exists
    if not os.path.exists(sb_target_folder): os.makedirs(sb_target_folder)
    pkl_file_name = os.path.join(sb_target_folder, sb_target_file)
    if os.path.exists(pkl_file_name):
        print("Bootstrap file for instance already found. Loading file")
        #load the dataframe, and return
        df = pd.read_pickle(pkl_file_name)
        print("Bootstrap dataframe loaded into memory")
        return df

    # If not, then load from the pkl_folder, do bootstrapping and then combine into dataframe
    print("Bootstrap file for instance not found. Loading raw file and implementing bootstrapping.")
    df_list = []
    for pkl_file in raw_data: 
        # load data from each file in the folder.
        # Each file corresponds to one choice of num_shots and rounds.
        with open(pkl_file, 'rb') as f:
            # Each raw file contains the metric values as a function of cobyla iteration number, for all the restarts.
            df = pickle.load(f)
            gen_prop = pickle.load(f)
        num_shots = gen_prop.get('num_shots')
        rounds = gen_prop.get('rounds')
        metric = gen_prop.get('objective_func_type')
        # Do bootstrapping now
        # df has indices = cobyla iterations and columns = restarts
        # For each minimizer iteration index, do bootstrapping to simulate many runs. For each simulated "run", keep only the max obtained value. Thus obtain many max values. Keep the mean and CI values for these quantities.
        
        
        for bs_restarts in bs_restarts_list:
            new_df = do_bootstrap(df, downsample = bs_restarts, bootstrap_iterations = bootstrap_iterations, metric = metric, confidence_level = 64)
            new_df.reset_index(inplace=True)
            new_df['boots'] = bs_restarts
            new_df['shots'] = num_shots
            new_df['rounds'] = rounds
            new_df['instance'] = instance_number
            
            df_list.append(new_df)
        
    df = pd.concat(df_list, ignore_index=True, axis=0)
    print("Bootstrap dataframe loaded onto memory")
    # Store df to pkl file
    df.to_pickle(pkl_file_name)
    
    return df
