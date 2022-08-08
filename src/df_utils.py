import glob
import os
import pandas as pd
import names
import numpy as np

import parameters

EPSILON = 1e-10
confidence_level = 68
s = 0.99
gap = 1.0


def read_exp_raw(exp_raw_dir, name_params = []):
    """ Input: directory of raw data files, parameter names that should be extracted from filename
        Outputs: combined dataframe of files in raw data directory
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
    """ Input: dataframe and parameters of interest
        Outputs: list of numerical values for parameters encountered in df
    """
    df['params'] = df[param_names].apply(tuple, axis=1)
    param_set = df['params'].unique()
    return param_set

def get_best(df, response_col, response_dir, group_on):
    """ 
    Outputs: dataframe rows with best response (specificied by response_col and response_dir)
    Grouped by group on
    """
    if response_dir == -1: # Minimization
        best_df = df.sort_values(response_col, ascending=True).drop_duplicates(group_on)
    else:
        best_df = df.sort_values(response_col, ascending=False).drop_duplicates(group_on)
    return best_df

