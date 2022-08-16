from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import warnings

import itertools
from utils_ws import *


default_boots = 1000


@dataclass
class InterpolationParameters:
    """
    Parameters for dataframe interpolation
    """
    resource_col: str = 'reads'
    resource_proportional_parameters: list = field(
        default_factory=lambda: ['sweep', 'replica'])
    parameters: list = field(default_factory=lambda: ['sweep', 'replica'])
    # 'log', 'data', or 'manual' indicates how interpolation points should be generated
    resource_value_type: str = 'log'
    resource_values: list = field(default_factory=list)
    group_on: str = 'instance'
    min_boots: int = 1

    def __post_init__(self):
        for r_param in self.resource_proportional_parameters:
            if r_param not in self.parameters:
                self.resource_proportional_parameters.remove(r_param)
                warn_str = 'Resource proportional parameter, {}, is not in the parameters. Removing.'.format(
                    r_param)
                warnings.warn(warn_str)

        if self.resource_value_type not in ['manual', 'data', 'log']:
            warn_str = 'Unsupported resource value type: {}. Setting value type to log.'.format(
                self.resource_value_type)
            warnings.warn(warn_str)
            self.resource_value_type = 'log'

        if self.resource_value_type == 'manual' and self.resource_values is None:
            warn_str = 'Manual resource value type requires resource values. Setting value type to log.'
            warnings.warn(warn_str)
            self.resource_value_type = 'log'

        elif self.resource_value_type in ['data', 'log'] and self.resource_values is not None:
            warn_str = 'Resource value type {} does not support passing in values. Removing.'.format(
                self.resource_value_type)
            warnings.warn(warn_str)
            self.resource_value = 'None'


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
    df['resource'] = df[interp_params.resource_col].copy()
    if interp_params.resource_value_type == 'data':
        interp_params.resource_values = df['resource'].values
    elif interp_params.resource_value_type == 'log':
        interp_params.resource_values = gen_log_space(
            min(df['resource'].values), max(df['resource'].values), default_boots // 10)
    interp_params.resource_values = np.sort(
        np.unique(interp_params.resource_values))


def prepareInterpolation(df: pd.DataFrame, interp_params: InterpolationParameters):
    """
    Prepares dataframe for interpolation by defining resource column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to prepare for interpolation.
    interp_params : InterpolationParameters
        Parameters for interpolation.

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolation parameters applied.
    """
    if interp_params.resource_col not in df.columns:
        warnings.warn(
            "resource_col is not a column of the dataframe, setting resource_col to boots")
        interp_params.resource_col = 'boots'
        for r_param in interp_params.resource_proportional_parameters:
            df['resource'] *= df[r_param]


def getResourceFactor(df_single: pd.DataFrame, interp_params: InterpolationParameters):
    """
    Gets the resource factor for a single dataframe.

    Parameters
    ----------
    df_single : pd.DataFrame
        Dataframe to get resource factor for.
    interp_params : InterpolationParameters
        Parameters for interpolation.

    Returns
    -------
    r_factor : float
        Resource factor for the dataframe.
    """
    r_param_vals = [df_single[r_param].iloc[0]
                    for r_param in interp_params.resource_proportional_parameters]
    r_factor = np.prod(r_param_vals)
    return r_factor


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
    max_boots = df_single['boots'].max()
    r_factor = getResourceFactor(df_single, interp_params)
    interpolate_resource = interp_params.resource_values[
        np.where(
            (interp_params.resource_values <= take_closest(interp_params.resource_values, max_boots*r_factor)) &
            (interp_params.resource_values >= take_closest(
                interp_params.resource_values, interp_params.min_boots*r_factor))
        )
    ]
    df_single.set_index('resource', inplace=True)
    df_single.sort_index(inplace=True)

    df_out = pd.DataFrame(index=interpolate_resource)
    df_out.index.name = 'resource'

    for colname, col in df_single.iteritems():
        col = pd.to_numeric(col, errors='ignore')
        if colname in group_on:
            continue
        elif np.issubdtype(col, int) or np.issubdtype(col, float):
            # print(col)
            df_out[colname] = np.interp(
                interpolate_resource, df_single.index, col, left=np.nan)
        else:
            warn_str = '{} is not a numeric type. Column was left unchanged'.format(
                colname)
            warning.warn(warn_str)
            # PyLance warning that "warning" is not defined
            df_out[colname] = col
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
    prepareInterpolation(df, interp_params)
    generateResourceColumn(df, interp_params)
    def dfInterp(df): return InterpolateSingle(df, interp_params, group_on)
    df_interp = df.groupby(group_on).apply(dfInterp).reset_index()
    return df_interp


# Kept for testing, can delete later
# %%
# Function to interpolate dataframes across a resource column
def interpolate_dfOld(
    dataframe: pd.DataFrame = None,
    resource_column: str = 'reads',
    prefix: str = '',
    parameters_dict: dict = None,  # dictionary of lists of parameter values
    default_boots: int = 1000,
    minimum_boots: int = 1,
    resource_proportional_parameters: list = ['sweep', 'replica'],
    idx=pd.IndexSlice,
    results_path: str = None,
    save_pickle: bool = True,
    overwrite_pickles: bool = True,
    all_datapoints: bool = False,
    resource_values: list = None,
):
    # Get rid of this check, the user should always be passing in a DF
    if dataframe is None:
        print('Error: Dataframe is None')
        return None

    if len(dataframe) == 0:
        print('Error: Dataframe is empty')
        return None

    df = dataframe.copy()
    # Generates parameter names from dictionary
    parameter_names = list(parameters_dict.keys())
    parameter_sets = itertools.product(
        *(parameters_dict[Name] for Name in parameters_dict))
    # Lists out every combination of parameters from parameters dict
    parameter_sets = list(parameter_sets)
    r_indices = []

    # check this in a separate function and set resource to boots instead of modifying df
    if resource_column not in df.columns:
        df[resource_column] = df['boots']
        for r_parameters in resource_proportional_parameters:
            if r_parameters in parameter_names:
                df[resource_column] *= df[r_parameters]

    if resource_values is None:  # If resource values are not passed in then generate from datapoints or log space
        if all_datapoints:
            resource_values = df[resource_column].values
        else:
            resource_values = gen_log_space(min(df[resource_column].values), max(
                df[resource_column].values), default_boots // 10)
    resource_values = np.sort(np.unique(resource_values))

    instances = [0]
    if 'instance' in df.columns:
        instances = df['instance'].unique().tolist()
    df_index = df.set_index(parameter_names).sort_index().copy()
    for r_parameters in resource_proportional_parameters:
        if r_parameters in parameter_names:
            r_indices.append(parameter_names.index(r_parameters))

    dataframes = []
    boots_rec_dict = {}
    for instance in instances:
        # df_name_partial = prefix.rsplit(
        #     '.',1)[0] + str(instance) + '_partial.pkl'
        # df_path_partial = os.path.join(results_path, df_name_partial)
        # if os.path.exists(df_path_partial) and not overwrite_pickles:
        #     print('Loaded partial dataframe from file')
        #     df_interpolate = pd.read_pickle(df_path_partial)
        #     dataframes_instance = [df_interpolate]
        # else:
        dataframes_instance = []
        # for parameter_set in parameter_sets:
        #     if parameter_set not in df_index.index.to_list():
        #         print('Parameter set', parameter_set, 'not found')
        #         continue  # For each parameter setting remove repeated reads
        # df_values = df_index.loc[idx[parameter_set]].copy()
        if 'instance' in df.columns:
            df_values = df_index.loc[df_index['instance'] == instance].copy(
            )
        else:
            df_values = df_index.copy()
        for parameter_set in set(df_values.index.to_list()):
            df_original = df_values.loc[idx[parameter_set]].copy()
            # Reading the parameter columns
            for key, value in zip(parameter_names, parameter_set):
                df_original[key] = value
            if 'params' in df_original.columns:
                df_original.drop(columns=['params'], inplace=True)
            if len(df_original) == 0:
                print('No data for parameter set',
                      parameter_set, 'with instance', instance)
                continue
            resource_factor = 1
            for r_index in r_indices:
                resource_factor *= parameter_set[r_index]
                # resource_factor *= index[r_index]
            # Set interpolation points for the responses at all the relevant reads values
            maximum_boots = df_original['boots'].max()
            boots_rec_dict[tuple(parameter_set)] = (
                maximum_boots, resource_factor)
            interpolate_resource = resource_values[
                np.where(
                    (resource_values <= take_closest(resource_values, maximum_boots*resource_factor)) &
                    (resource_values >= take_closest(
                        resource_values, minimum_boots*resource_factor))
                )
            ]
            if all_datapoints:
                # Create a dataframe with the interesting reads as index and all the columns
                dummy_df = pd.DataFrame(
                    np.NaN,
                    index=interpolate_resource,
                    columns=df_index.columns
                )
                dummy_df.drop(columns=resource_column, inplace=True)
                # Fill out the values that we have certain
                dummy_df.update(df_original.set_index(resource_column))
                df_interpolate = dummy_df.copy()
                # Interpolate for all the other values (without extrapolating)
                df_interpolate = df_interpolate.interpolate(
                    method='linear', limit_area='inside'
                ).dropna(how='all').reset_index().rename(
                    columns={'index': resource_column})
            else:
                df_interpolate = interp(df_original.set_index(
                    resource_column).sort_index(), interpolate_resource)
                df_interpolate = df_interpolate.reset_index().rename(
                    columns={'index': resource_column})
            # Computing the boots column
            # display(df_interpolate)
            # print(len(df_interpolate['boots']))
            # print(len(df_interpolate[resource_column]))
            # print(resource_factor)
            # print(df_interpolate[resource_column])
            # print(df_interpolate[resource_column] / \
            #     resource_factor)
            df_interpolate['boots'] = df_interpolate[resource_column] / \
                resource_factor
            if 'instance' in df.columns:
                df_interpolate['instance'] = instance
            dataframes_instance.append(df_interpolate)
        df_interpolate = pd.concat(
            dataframes_instance).reset_index(drop=True)
        # df_interpolate.to_pickle(df_path_partial)

    if len(instances) == 1:
        dataframes = dataframes_instance
    else:
        for instance in instances:
            df_name_partial = prefix.rsplit(
                '.', 1)[0] + str(instance) + '_partial.pkl'
            df_path_partial = os.path.join(results_path, df_name_partial)
            # df_interpolate = pd.read_pickle(df_path_partial)
            # dataframes.append(df_interpolate)

#     if all([len(i) == 0 for i in dataframes]):
#         print('No dataframes to merge')
#         return None
    df_interpolated = pd.concat(dataframes).reset_index()
#     if save_pickle:
#         df_name_interpolated = prefix.rsplit('.',1)[0] + '_interp.pkl'
#         df_path_interpolated = os.path.join(results_path, df_name_interpolated)
#         df_interpolated.to_pickle(df_path_interpolated)
    return df_interpolated
    # return boots_rec_dict
# %%
