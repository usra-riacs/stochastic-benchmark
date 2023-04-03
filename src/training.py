# from collections import defaultdict
# from dataclasses import dataclass, field
import numpy as np

import pandas as pd
from typing import List, Tuple, Union
import warnings

import df_utils

check_split_validity = True # While splitting instances into test and train sets, ensure that each set is non-empty


def best_parameters(df: pd.DataFrame,
                     parameter_names: List[str],
                     response_col: str,
                     response_dir: int,
                     resource_col: str = 'resource',
                   additional_cols: List[str] = ['boots'],
                   smooth=False):
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
    if smooth:
        best = df_utils.monotone_df(best, resource_col, response_col, response_dir)
            
    best = best[np.unique([resource_col, response_col] + parameter_names + additional_cols)
                ].sort_values(resource_col, ascending=True)
    return best


def virtual_best(df: pd.DataFrame,
                 parameter_names: List[str],
                 response_col: str,
                 response_dir: int,
                 groupby: List[str]=['instance'],
                 resource_col: str = 'resource',
                additional_cols: List[str] = ['boots'],
                smooth=False):
    """
    Returns best parameters for each instance and resource level
    
    Parameters
    ----------
    parameter_names : list[str]
    response_col : str
    response_dir : int
        Maximizing (1) or minimizing(-1)
    groupby : list[str]
        columns that define and instance
    resouce_col : str
    additional_cols : list[str]
        Additional columns that should be kept in the dataframe
    smooth: bool
        Whether to monotonize the dataframe
    """
    
    def br(df) : return best_parameters(df, parameter_names, response_col, response_dir, resource_col, additional_cols, smooth)
    vb = df.groupby(groupby).apply(br).reset_index()
    vb.drop('level_{}'.format(len(groupby)), axis=1, inplace=True)
    return vb


def split_train_test(df: pd.DataFrame, split_on: List[str], ptrain: float):
    """
    Create column, 'train' that splits training and testing instances
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to write the train column to
    split_on : list[str]
        List of columns that define an instance (i.e., if they match on all columns in split_on, they will have the same label)
    ptrain : float
        Fraction of instances should be a training instance
    """
    if not check_split_validity:
        df = df.groupby(split_on).apply(lambda df : pd.DataFrame.from_dict({'train': [np.random.binomial(1, ptrain)]})).merge(df, on=split_on)
    else:
        valid_split = False
        while not valid_split:
            df = df.groupby(split_on).apply(lambda df : pd.DataFrame.from_dict({'train': [np.random.binomial(1, ptrain)]})).merge(df, on=split_on)
            train_col = df['train'].unique()
            if 1 not in train_col or 0 not in train_col:
                # delete train column, and redo
                del df['train']
                raise Warning("Testing and training sets are not both non-empty. Redoing split. To remove this warning, set training.check_split_validity=False")
            else:
                # The split is valid, so exit while, and return df
                valid_split = True
    return df

def best_recommended(vb: pd.DataFrame,
                     parameter_names: List[str],
                     resource_col: str = 'resource',
                    additional_cols: List[str]=[]):
    """
    Returns aggregated recommneded parameters
    """
    
    br = vb.groupby(resource_col).mean()
    return br[parameter_names + additional_cols]

def evaluate_single(df_eval: pd.DataFrame,
                    recipes: pd.DataFrame,
                    distance_fcn,
                    parameter_names: List[str],
                    resource_col: str = 'resource'):
    """
    Parameters
    ----------
    df_eval : pd.DataFrame
        Dataframe with points that should be used for projection
    recipes : pd.DataFrame
        recipes that should be tried out on df_eval
    distance_fcn : callable
        Function that defines distance from df_eval parameters to recipe parameters
    parameter_names : list[str]
    resource_col : str
    """
    df_list = []
    def argmin(df) : return df[df['distance_scaled'] == df['distance_scaled'].min()]
    for _,recipe in recipes.iterrows():
        temp_df_eval = argmin(distance_fcn(df_eval[df_eval[resource_col]==recipe[resource_col]], recipe, parameter_names))
        temp_df_eval.loc[:,resource_col] = recipe[resource_col]
        for p in parameter_names:
            temp_df_eval['{}_rec'.format(p)] = recipe[p]
        df_list.append(temp_df_eval)
    res = pd.concat(df_list, ignore_index=True)
    return res

def scaled_distance(df_eval: pd.DataFrame,
                    recipe: pd.DataFrame,
                    parameter_names: List[str]):
    """
    Scaled distance function that defines distance from every point in df_eval to one recipe
    
    """
    local_df_eval = df_eval.copy()
    local_df_eval['distance_scaled'] = 0.
    for colname in parameter_names:
        maxval = df_eval[colname].max()
        minval = df_eval[colname].min()

        if maxval == minval:
            local_df_eval.loc[:, [colname + '_scaled']] = (df_eval[colname] != recipe[colname]).astype(float)
            local_df_eval.loc[:, 'distance_scaled'] += (local_df_eval[colname + '_scaled'])**2
            
        else:
            local_df_eval.loc[:, [colname + '_scaled']] = (df_eval[colname] - minval) / (maxval - minval)
            recipe.loc[colname + '_scaled'] = (recipe[colname] - minval) / (maxval - minval)
            local_df_eval.loc[:, 'distance_scaled'] += (local_df_eval[colname + '_scaled'] - recipe[colname + '_scaled'].copy())**2
    return local_df_eval

def evaluate(df: pd.DataFrame,
             recipes: pd.DataFrame,
             distance_fcn,
             parameter_names: List[str],
             resource_col: str = 'resource',
             group_on=[]):
    """
    Divides df by instance and applies projection from recipes onto df
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to do the projection from
    recipes : pd.DataFrame
        Recipes to try out
    distance_fcn : Callable
        Computes distance between parameters for projection
    parameters_names : list[str]
    resource_col : str
    group_on : list[str]
        list of columns that define an instance
    """
    if len(group_on) == 0:
        return evaluate_single(df, recipes, distance_fcn, parameter_names, resource_col)
    else:
        def eval_fcn(df) : return evaluate_single(df, recipes, distance_fcn, parameter_names, resource_col)
        df_eval = df.groupby(group_on).apply(eval_fcn).reset_index(drop=True)
        return df_eval
    
