import glob
import os
import pandas as pd
import names
import numpy as np

import parameters
# TODO PyLance is creying here because parameters is not defined

EPSILON = 1e-10
confidence_level = 68
s = 0.99
gap = 1.0


def read_exp_raw(exp_raw_dir, name_params = []):
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
    if response_dir == -1: # Minimization
        best_df = df.sort_values(response_col, ascending=True).drop_duplicates(group_on)
    else:
        best_df = df.sort_values(response_col, ascending=False).drop_duplicates(group_on)
    return best_df

