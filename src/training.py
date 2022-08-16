# from collections import defaultdict
# from dataclasses import dataclass, field
import numpy as np

import pandas as pd
from typing import List, Tuple, Union
import warnings



def best_parameters(df: pd.DataFrame, 
                 parameter_names: List[str],
                 response_col: str,
                 response_dir: int,
                 resource_col: str = 'resource'):
    if response_dir == 1:
        best = df.sort_values(response_col, ascending=False).drop_duplicates(resource_col)
    elif response_dir == -1:
        best = df.sort_values(response_col, ascending=True).drop_duplicates(resource_col)
    else:
        warnings.warn('Unsupported response_dir, maximizing performance')
        best = df.sort_values(response_col, ascending=False).drop_duplicates(resource_col)
        
    best = best[np.unique([resource_col, response_col] + parameter_names)].sort_values(resource_col, ascending=True)
    return best

def virtual_best(df: pd.DataFrame,
                 parameter_names: List[str],
                 response_col: str,
                 response_dir: int,
                 resource_col: str = 'resource'):
    def br(df) : return best_parameters(df, parameter_names, response_col, response_dir, resource_col)
    vb = df.groupby('instance').apply(br).reset_index()
    vb.drop('level_1', axis=1, inplace=True)
    return vb

def split_train_test(df: pd.DataFrame, split_on: List[str], ptrain: float):
    df = df.groupby(split_on).apply(lambda df : pd.DataFrame.from_dict({'train': [np.random.binomial(1, ptrain)]})).merge(df, on=split_on)
    return df

def best_recommended(vb: pd.DataFrame,
                     parameter_names: List[str],
                     resource_col: str = 'resource'):
    
    br = vb.groupby(resource_col).mean()
    return br[parameter_names]

def evaluate_single(df_eval: pd.DataFrame,
                    recipes: pd.DataFrame,
                    distance_fcn,
                    parameter_names: List[str],
                    resource_col: str = 'resource'):
    df_list = []
    def argmin(df): return df[df['distance_scaled'] == df['distance_scaled'].min()]
    for _,recipe in recipes.iterrows():
        temp_df_eval = argmin(distance_fcn(df_eval, recipe, parameter_names))
        temp_df_eval.loc[:,resource_col] = recipe[resource_col]
        for p in parameter_names:
            temp_df_eval['{}_rec'.format(p)] = recipe[p]
        df_list.append(temp_df_eval)

    return pd.concat(df_list, ignore_index=True)

def scaled_distance(df_eval: pd.DataFrame,
                    recipe: pd.DataFrame,
                    parameter_names: List[str]):
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
    if len(group_on) == 0:
        return evaluate_single(df, recipes, distance_fcn, parameter_names, resource_col)
    else:
        def eval_fcn(df) : return evaluate_single(df, recipes, distance_fcn, parameter_names, resource_col)
        df_eval = df.groupby(group_on).apply(eval_fcn).reset_index(drop=True)
        return df_eval
    