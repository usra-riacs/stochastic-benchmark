import glob
from multiprocess import Pool
import os
import pandas as pd
import names
import numpy as np

# import parameters
# TODO PyLance is crying here because parameters is not defined

EPSILON = 1e-10
confidence_level = 68
s = 0.99
gap = 1.0

#Taken from https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby/29281494#29281494
def applyParallel(dfGrouped, func):
    with Pool() as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def read_exp_raw(exp_raw_dir, name_params=[]):
    """
    Generates a combined dataframe of all experiments in exp_raw_dir

    Parameters
    ----------
    exp_raw_dir : str
        directory of raw data files, parameter names that should be extracted from filename
    name_params : list
        list of parameter names


    Returns
    -------
    df : pd.dataframe
        dataframe of all experiments in exp_raw_dir
        Directory containing raw experiment data
    """
    filelist = glob.glob(os.path.join(exp_raw_dir, '*.pkl'))
    df_list = []
    for f in filelist:
        temp_df = pd.read_pickle(f)
        # expand parameters in filename
        if len(name_params) >= 1:
            params_dict = names.filename2param(os.path.basename(f))
            for p in name_params:
                temp_df[p] = params_dict[p]
        df_list.append(temp_df)
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all


def parameter_set(df, param_names):
    """
    Obtain the list of parameter settings in columns param_names from a dataframe and create 'params' column in it

    Parameters
    ----------
    df : pd.dataframe
        dataframe containing parameter settings
    param_names : list
        list of parameter names

    Returns
    -------
    param_set : list
        list of unique parameter settings
    """
    df['params'] = df[param_names].apply(tuple, axis=1)
    param_set = df['params'].unique()
    return param_set


def get_best(df, response_col, response_dir, group_on):
    """ 
    Sorts according to the best response for each parameter set in dataframe fd (specificied by response_col and response_dir) grouped by group on

    Parameters
    ----------
    df : pd.dataframe
        dataframe containing parameter settings
    response_col : str
        column name of response
    response_dir : float
        direction of response (either 'min'=-1 or 'max'=1)
        TODO There has to be a better way of doing this
    group_on : list
        list of column names to group on

    Returns
    -------
    df_best : pd.dataframe
    """
    if response_dir == -1:  # Minimization
        best_df = df.sort_values(
            response_col, ascending=True).drop_duplicates(group_on)
    else:
        best_df = df.sort_values(
            response_col, ascending=False).drop_duplicates(group_on)
    return best_df


def rename_df(df):
    """
    Rename columns in dataframe df from old format to newer one
    
    Parameters
    ----------
    df : pd.dataframe
        dataframe

    Returns
    -------
    df : pd.dataframe
        dataframe with renamed columns
    """
    rename_dict = {'min_energy': names.param2filename({'Key': 'MinEnergy'}, ''),
                   'min_energy_conf_interval_lower': names.param2filename({'Key': 'MinEnergy', 'ConfInt': 'lower'}, ''),
                   'min_energy_conf_interval_upper': names.param2filename({'Key': 'MinEnergy', 'ConfInt': 'upper'}, ''),
                   'perf_ratio': names.param2filename({'Key': 'PerfRatio'}, ''),
                   'perf_ratio_conf_interval_lower': names.param2filename({'Key': 'PerfRatio', 'ConfInt': 'lower'}, ''),
                   'perf_ratio_conf_interval_upper': names.param2filename({'Key': 'PerfRatio', 'ConfInt': 'upper'}, ''),
                   'success_prob': names.param2filename({'Key': 'SuccProb'}, ''),
                   'success_prob_conf_interval_lower': names.param2filename({'Key': 'SuccProb', 'ConfInt': 'lower'}, ''),
                   'success_prob_conf_interval_upper': names.param2filename({'Key': 'SuccProb', 'ConfInt': 'upper'}, ''),
                   'rtt': names.param2filename({'Key': 'RTT'}, ''),
                   'rtt_conf_interval_lower': names.param2filename({'Key': 'RTT', 'ConfInt': 'lower'}, ''),
                   'rtt_conf_interval_upper': names.param2filename({'Key': 'RTT', 'ConfInt': 'upper'}, ''),
                   'mean_time': names.param2filename({'Key': 'MeanTime'}, ''),
                   'mean_time_conf_interval_lower': names.param2filename({'Key': 'MeanTime', 'ConfInt': 'lower'}, ''),
                   'mean_time_conf_interval_upper': names.param2filename({'Key': 'MeanTime', 'ConfInt': 'upper'}, ''),
                   'inv_perf_ratio': names.param2filename({'Key': 'InvPerfRatio'}, ''),
                   'inv_perf_ratio_conf_interval_lower': names.param2filename({'Key': 'InvPerfRatio', 'ConfInt': 'lower'}, ''),
                   'inv_perf_ratio_conf_interval_upper': names.param2filename({'Key': 'InvPerfRatio', 'ConfInt': 'upper'}, '')}

    df.rename(columns=rename_dict, inplace=True)
    return df
