from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from typing import Callable
import itertools
from utils_ws import *

tqdm.pandas()

default_ninterp = 100


@dataclass
class InterpolationParameters:
    """
    Parameters for dataframe interpolation
    """
    resource_fcn: Callable[[pd.DataFrame], pd.Series]
    parameters: list = field(default_factory=lambda: ['sweep', 'replica'])
    resource_value_type: str = 'log' # 'log', 'data', or 'manual' indicates how interpolation points should be generated
    resource_values: list = field(default_factory=list)
    group_on: str = 'instance'
    min_boots: int = 1
    ignore_cols: list = field(default_factory=lambda: [])

    def __post_init__(self):
        if self.resource_value_type not in ['manual', 'data', 'log']:
            warn_str = 'Unsupported resource value type: {}. Setting value type to log.'.format(
                self.resource_value_type)
            warnings.warn(warn_str)
            self.resource_value_type = 'log'

        if self.resource_value_type == 'manual' and (len(self.resource_values) == 0):
            warn_str = 'Manual resource value type requires resource values. Setting value type to log.'
            warnings.warn(warn_str)
            self.resource_value_type = 'log'

        elif self.resource_value_type in ['data', 'log'] and (len(self.resource_values) >= 0):
            warn_str = 'Resource value type {} does not support passing in values. Removing.'.format(
                self.resource_value_type)
            warnings.warn(warn_str)
            self.resource_value = []

            
def generateResourceColumn(df: pd.DataFrame, interp_params: InterpolationParameters):
    """
    Generates a resource column for the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to generate resource column for.
    interp_params : InterpolationParameters
        Parameters for interpolation.

    Returns
    -------
    pd.DataFrame
        Dataframe with resource column generated.
    """
    df['resource'] = interp_params.resource_fcn(df)
    
    if interp_params.resource_value_type == 'data':
        interp_params.resource_values = df['resource'].values
    elif interp_params.resource_value_type == 'log':
        interp_params.resource_values = gen_log_space(
            min(df['resource'].values), max(df['resource'].values), default_ninterp)
    
    interp_params.resource_values = np.sort(
        np.unique(interp_params.resource_values))

def InterpolateSingle(df_single: pd.DataFrame, interp_params: InterpolationParameters, group_on):
    """
    Interpolates a dataframe based on a single column.

    Parameters
    ----------
    df_single : pd.DataFrame
        Dataframe to interpolate.
    interp_params : InterpolationParameters
        Parameters for interpolation.
    group_on : str
        Grouping parameter for the dataframe.

    Returns
    -------
    pd.DataFrame
        Interpolated dataframe.    
    """
    # max boots may vary across df_singles (is this desirable?) Leaving in to make sure no extrapolation occurs
    interpolate_resource = interp_params.resource_values[
        np.where(
            (interp_params.resource_values <= take_closest(interp_params.resource_values, df_single['resource'].max())) &
            (interp_params.resource_values >= take_closest(interp_params.resource_values, df_single['resource'].min()))
        )
    ]
    if not df_single['resource'].is_unique:
        df_single.drop_duplicates('resource', inplace=True)
        warn_str = 'Dataframe has duplicate resources. Dropping duplicates, but consider re-running bootstrap'
        warnings.warn(warn_str)
    
    df_single.set_index('resource', inplace=True)
    df_single.sort_index(inplace=True)
    df_out = pd.DataFrame(index=interpolate_resource)
    df_out.index.name = 'resource'

    for colname, col in df_single.iteritems():
        col = pd.to_numeric(col, errors='ignore')
        if colname in group_on:
            continue
        elif colname in interp_params.ignore_cols:
            df_out[colname] = df_single[colname].iloc[0]
        elif np.issubdtype(col, int) or np.issubdtype(col, float):
            df_out[colname] = np.interp(
                interpolate_resource, df_single.index, col, left=np.nan)
        else:
            df_out[colname] = col.copy()
    
    return df_out


def Interpolate(df: pd.DataFrame, interp_params: InterpolationParameters, group_on):
    """
    Complete interpolation function to include preparation, resource columns and actual interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to interpolate.
    interp_params : InterpolationParameters
        Parameters for interpolation.
    group_on : str
        Grouping parameter for the dataframe.

    Returns
    -------
    pd.DataFrame
        Interpolated dataframe.
    """
    generateResourceColumn(df, interp_params)
    def dfInterp(df): return InterpolateSingle(df, interp_params, group_on)
    df_interp = df.groupby(group_on).progress_apply(dfInterp)
    df_interp.reset_index(inplace=True)
    return df_interp

def Interpolate_reduce_mem(df_list:list, interp_params: InterpolationParameters, group_on):
    """
    Complete interpolation function to include preparation, resource columns and actual interpolation.

    Parameters
    ----------
    df_list : list[str]
        list of bootstrapped results to interpolate on
    interp_params : InterpolationParameters
        Parameters for interpolation.
    group_on : str
        Grouping parameter for the dataframe.

    Returns
    -------
    pd.DataFrame
        Interpolated dataframe.
    """
    df_interp_list = []
    for df_name in df_list:
        df = pd.read_pickle(df_name)
        generateResourceColumn(df, interp_params)
        temp_df_interp = df.groupby(group_on).progress_apply(lambda df: InterpolateSingle(df, interp_params, group_on))
        temp_df_interp.reset_index(inplace=True)
        df_interp_list.append(temp_df_interp)
    df_interp = pd.concat(df_interp_list, ignore_index=True)
    return df_interp