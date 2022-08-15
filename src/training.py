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
    def br(df) : return best_recommended(df, parameter_names, response_col, response_dir, resource_col)
    vb = df.groupby('instance').apply(br).reset_index()
    vb.drop('level_1', axis=1, inplace=True)
    return vb

def split_train_test(df: pd.DataFrame, split_on: List[str], ptrain: float):
    df = df.groupby(split_on).apply(lambda df : pd.DataFrame.from_dict({'train': [np.random.binomial(1, ptrain)]})).merge(df, on=split_on)
    return df