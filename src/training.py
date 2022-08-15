# from collections import defaultdict
# from dataclasses import dataclass, field
import numpy as np

import pandas as pd
from typing import List, Tuple, Union
import warnings


def best_recommended(df: pd.DataFrame,
                     parameter_names: List[str],
                     response_col: str,
                     response_dir: int,
                     resource_col: str = 'resource'):
    """
    Returns the best recommended parameter set for each instance in df

    Parameters
    ----------
    df : pd.dataframe
        dataframe containing parameter settings
    parameter_names : list
        list of parameter names
    response_col : str
        column name of response
    response_dir : float
        direction of response (either 'min'=-1 or 'max'=1)
        TODO There has to be a better way of doing this
    resource_col : str
        column name of resource

    Returns
    -------
    df_best : pd.dataframe
    """
    if response_dir == 1:
        best = df.sort_values(
            response_col, ascending=False).drop_duplicates(resource_col)
    elif response_dir == -1:
        best = df.sort_values(
            response_col, ascending=True).drop_duplicates(resource_col)
    else:
        warnings.warn('Unsupported response_dir, maximizing performance')
        best = df.sort_values(
            response_col, ascending=False).drop_duplicates(resource_col)

    best = best[np.unique([resource_col, response_col] + parameter_names)
                ].sort_values(resource_col, ascending=True)
    return best


def virtual_best(df: pd.DataFrame,
                 parameter_names: List[str],
                 response_col: str,
                 response_dir: int,
                 resource_col: str = 'resource'):
    """
    Returns the virtual best parameter set for each instance in df

    Parameters
    ----------
    df : pd.dataframe
        dataframe containing parameter settings
    parameter_names : list
        list of parameter names
    response_col : str
        column name of response
    response_dir : float
        direction of response (either 'min'=-1 or 'max'=1)
        TODO There has to be a better way of doing this
    resource_col : str
        column name of resource

    Returns
    -------
    df_best : pd.dataframe
    """
    def br(df): return best_recommended(df, parameter_names,
                                        response_col, response_dir, resource_col)
    vb = df.groupby('instance').apply(br).reset_index()
    vb.drop('level_1', axis=1, inplace=True)
    return vb


def split_train_test(df: pd.DataFrame, split_on: List[str], ptrain: float):
    """
    Splits dataframe df into train and test dataframes according to split_on and ptrain

    Parameters
    ----------
    df : pd.dataframe
        dataframe containing parameter settings
    split_on : list
        list of column names to split on
    ptrain : float
        proportion of data to use for training

    Returns
    -------
    df_train : pd.dataframe
    """
    df = df.groupby(split_on).apply(lambda df: pd.DataFrame.from_dict(
        {'train': [np.random.binomial(1, ptrain)]})).merge(df, on=split_on)
    return df
