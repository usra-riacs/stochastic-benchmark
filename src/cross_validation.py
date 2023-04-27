import pandas as pd
import numpy as np
from scipy.special import erfinv
import os
import stats
import interpolate
import utils_ws

parameters_dict = dict()
performance_dict = dict()

parameters_summarized_dict = dict()  # Dict of dicts. dict[expt][param] --> dataframe with mean and CI for expt and param
performance_summarized_dict = dict()

def baseline_evaluate(rec_params, parameter_names, response_col):
    params_df = rec_params.loc[:, ['resource'] + parameter_names]
    params_df = params_df.groupby('resource').mean()
    params_df.reset_index(inplace=True)
    eval_df = rec_params.copy()
    eval_df = eval_df.loc[:, ['resource', response_col]]
    eval_df = eval_df.groupby('resource').mean()
    eval_df.reset_index(inplace=True)
    return params_df, eval_df

def all_ci(df_single, colname, confidence_level=68):
    """
    Averaging of parameter values
    Args:
        df_single (pd.dataframe): with two columns: resource, and parameter name
        colname (str): parameter name
        confidence_level (int, optional): Defaults to 68.

    Returns:
        pd.Dataframe: _description_
    """
    fact = erfinv(confidence_level / 100.) * np.sqrt(2.)
    data = df_single[colname]
    mean = np.nanmean(data)
    std = np.nanstd(data)
    ddict = dict()
    ddict['mean'] = [mean]
    ddict['CI_l'] = [mean - fact * std]
    ddict['CI_u'] = [mean + fact * std]
    # ddict['resource'] = [mean - fact * std]
    return pd.DataFrame.from_dict(ddict)

def propagate_ci(df, stats_measure):
    """
    Propagate CIs for resource vs performance
    Input is a dataframe with columns 'response', 'response_lower' and 'response_upper'
    Each row corresponds to data from a different split.
    These are assumed to the median and +- 68% CI.
    It is assumed that the distribution is Gaussian, so that 'response' is the mean
    
    Args:
        df (dataframe): Gotten from groupby.apply
        propagate_option (int): 1 or 2 
            If 1, do inverse variance weighing
            If 2, mean=average of medians
    """
    if stats_measure == "mean":
        sm = stats.Mean()
    elif stats_measure == "median":
        sm = stats.Median()
    else: 
        raise ValueError("The value of stats_measure can only be mean or median")
    
    cent, CIlower, CIupper = sm.ConfInts(base=df["response"], lower=df["response_lower"], upper=df["response_upper"])
    
    ddict = dict()
    ddict['mean'] = [cent]
    ddict['CI_l'] = [CIlower]
    ddict['CI_u'] = [CIupper]
    return pd.DataFrame.from_dict(ddict)

def load_parameters(folders, list_of_expts):
    """
    For each split folder (in folders), parameters data is stored in a csv file corresponding to each experiment (in list_of_expts)
    For each experiment, load data from all splits, and concatenate into a dataframe. Store it in parameters_dict.

    Args:
        folders (list of strings): List of checkpoints folder location for each split.
        list_of_expts (list of strings) 
    """
    list_of_expts = ['baseline'] + list_of_expts
    # Create a dictionary for baseline, and each experiment
    for expt in list_of_expts:
        # concatenate all folders 
        df_list = []
        for split_ind, folder in enumerate(folders):
            # Load dataframe
            file_name = os.path.join(folder, 'params_plotting', expt + '.csv')
            df = pd.read_csv(file_name)
            df['split_ind'] = split_ind
            df_list.append(df)
        df_concat_this_expt = pd.concat(df_list, axis=0, ignore_index=True)
        if 'Unnamed: 0' in df_concat_this_expt.columns:
            df_concat_this_expt.drop(columns='Unnamed: 0', inplace=True)
        parameters_dict[expt] = df_concat_this_expt
        
def process_params_across_splits(parameter_names, confidence_level=68):
    """
    Obtain the mean and CI's for each parameter, extracted from each experiment
    To do this, for each experiment, load the data stored in parameters_dict[experiment]
    Next, for each value of resource, obtain the mean and CI's for each parameter (across various splits)
    Store processed data in parameters_summarized_dict

    Args:
        parameter_names (list of strings): List of parameters
        confidence_level (float)
    """
    for expt, curr_param_df in parameters_dict.items():
        expt_dict = dict()
        for param in parameter_names:
            expt_param_df = curr_param_df[["resource", param]].groupby('resource').apply(lambda col : all_ci(col, param, confidence_level)).reset_index()
            expt_param_df.drop('level_1', axis=1, inplace=True)
            expt_param_df.sort_values(by='resource', inplace=True)
            expt_dict[param] = expt_param_df
        parameters_summarized_dict[expt] = expt_dict

def process_performance_across_splits(stats_measure = "mean", mean_median_method = "simple_average"):
    """
    For any given value of resource and experiment type, each split contains a center and a confidence interval value for the performance
    Propagate these CIs to obtain a single center value and CI for each resource value
    Store processed data in performance_summarized_dict

    Args:
        stats_measure (str): can be "mean" or "median"
        mean_median_method (str): can be "simple_average" or "inverse_variance_weighing"
    """
    stats.mean_median_method = mean_median_method
    
    performance_summarized_dict.clear()
    for expt, curr_perf_df in performance_dict.items():
        if expt == 'baseline':
            # For the baseline, each split only has response vs resource.
            # There are no confidence intervals
            # Hence, there is no need to 'propagate the CIs'
            perf_df = curr_perf_df[["resource", "response"]].groupby('resource').apply(lambda col : all_ci(col, "response")).reset_index()
            perf_df.drop('level_1', axis=1, inplace=True)
            perf_df.sort_values(by='resource', inplace=True)
        else:
            # Need to propagate the CI's
            
            perf_df = curr_perf_df.groupby('resource').apply(lambda df: propagate_ci(df, stats_measure)).reset_index()
            perf_df.drop('level_1', axis=1, inplace=True)
            perf_df.sort_values(by='resource', inplace=True)
            
        performance_summarized_dict[expt] = perf_df


def interpolate_raw_performance(df_many_splits_performance_raw, group_on = ['split_ind']):
    """Given performance data for various test-train splits, choose a grid of values of resource, and obtain interpolated values of response and response confidence intervals at those values for each grid value

    Args:
        df_many_splits_performance_raw (dataframes): Combined raw data loaded from csv files corresponding to all test-train splits. Contains columns for resource, response, response CIs and split index (split_ind). 
        
    Returns:
        pd.dataframe: dataframe with 
    """
    # The resource grid for different instances can be different. 
    # First, choose a grid
    min_res, max_res = df_many_splits_performance_raw['resource'].min(), df_many_splits_performance_raw['resource'].max()
    interp_grid_for_resources = utils_ws.gen_log_space(
            min_res, max_res, interpolate.default_ninterp)
    
    iParams = interpolate.InterpolationParameters(resource_fcn=lambda x:None, 
                                                  parameters = [],
                                                  ignore_cols = ['count', 'index'],
                                                  resource_values= interp_grid_for_resources)
    
    # For each split_ind, do interpolation separately over the resource grid
    temp_df_interp = df_many_splits_performance_raw.groupby(group_on).\
        progress_apply(lambda df: interpolate.InterpolateSingle(df, iParams, group_on))
    temp_df_interp.reset_index(inplace=True)
    return temp_df_interp
        
def load_performance(folders, list_of_expts, split_ind_cols = ['split_ind'], interpolate_flag = True):
    """Load performance data for baseline and experiments for all test-train splits into memory.

    Args:
        folders (list of str): Directories of checkpoints folders corresponding to the different splits
        list_of_expts (list of str)
        split_ind_cols (list of str, optional): Columns that distinguish different test-train splits. Defaults to ['split_ind'].
        interpolate_flag (bool, optional): Whether to interpolate the response to a fixed grid for all splits. Defaults to True.
    """
    list_of_expts = ['baseline'] + list_of_expts
    # Create a dictionary for baseline, and each experiment
    for expt in list_of_expts:
        # concatenate all folders 
        df_list = []
        for split_ind, folder in enumerate(folders):
            # Load data from csv files
            file_name = os.path.join(folder, 'performance_plotting', expt + '.csv')
            df = pd.read_csv(file_name)
            df['split_ind'] = split_ind
            df_list.append(df)
        
        df_concat_this_expt = pd.concat(df_list, axis=0, ignore_index=True)
        if 'Unnamed: 0' in df_concat_this_expt.columns:
            df_concat_this_expt.drop(columns='Unnamed: 0', inplace=True)
            
        if interpolate_flag:
            df_concat_this_expt = interpolate_raw_performance(df_concat_this_expt, group_on = split_ind_cols)
        performance_dict[expt] = df_concat_this_expt

    
    
def proj_expt_evaluate(rec_params, parameter_names, response_col):
    params_df = rec_params.loc[:, ['resource'] + parameter_names].copy()
    params_df = params_df.groupby('resource').mean()
    params_df.reset_index(inplace=True)
    
    eval_df = rec_params.copy()
    eval_df.rename(columns = {
                response_col :'response',
                "ConfInt=lower_"+response_col :'response_lower',
                "ConfInt=upper_"+response_col :'response_upper',
            }, inplace=True
            )
    eval_df = eval_df.loc[:, ['resource','response', 'response_lower', 'response_upper']]
    eval_df = eval_df.groupby('resource').mean()
    eval_df.reset_index(inplace=True)
    
    return params_df, eval_df


def random_exp_evaluate(eval_test, parameter_names, response_col):
    """
    Returns
    -------
    params_df : pd.DataFrame
        Dataframe of recommended parameters
    eval_df : pd.DataFrame
        Dataframe of responses, renamed to generic columns for compatibility
    """
    params_df = eval_test.loc[:, ['TotalBudget'] + parameter_names]
    params_df = params_df.groupby('TotalBudget').mean()
    params_df.reset_index(inplace=True)
    params_df.rename(columns = {
        'TotalBudget' : 'resource'}, inplace=True)
    
    eval_df = eval_test.copy()
    eval_df.drop('resource', axis=1, inplace=True)
    eval_df.rename(columns = {
        'TotalBudget' : 'resource',
        response_col : 'response',
        # base :'response',
        'ConfInt=lower_'+response_col :'response_lower',
        'ConfInt=upper_'+response_col :'response_upper',
    },inplace=True
    )
    
    eval_df = eval_df.loc[:, ['resource','response', 'response_lower', 'response_upper']]
    eval_df = eval_df.groupby('resource').mean()
    eval_df.reset_index(inplace=True)
    return params_df, eval_df

def seq_search_evaluate(eval_test, parameter_names, response_col):
    """
    Returns
    -------
    params_df : pd.DataFrame
        Dataframe of recommended parameters
    eval_df : pd.DataFrame
        Dataframe of responses, renamed to generic columns for compatibility
    """
    params_df = eval_test.loc[:, ['TotalBudget'] + parameter_names]
    eval_df = eval_test.copy()
    
    
    for col in params_df.columns:
        if params_df[col].dtype == 'object':
            params_df.loc[:, col] = params_df.loc[:, col].astype(float)

    params_df.reset_index(inplace=True)
    params_df.rename(columns = {
        'TotalBudget' : 'resource'}, inplace=True)
    # base = names.param2filename({'Key': self.parent.response_key}, '')
    # CIlower = names.param2filename({'Key': self.parent.response_key,
    #                                 'ConfInt':'lower'}, '')
    # CIupper = names.param2filename({'Key': self.parent.response_key,
    #                                 'ConfInt':'upper'}, '')
    
    eval_df.drop('resource', axis=1, inplace=True)
    eval_df.rename(columns = {
        'TotalBudget' : 'resource',
        response_col : 'response',
        # base :'response',
        'ConfInt=upper_'+response_col :'response_lower',
        'ConfInt=lower_'+response_col :'response_upper',
    },inplace=True
    )
    
    eval_df = eval_df.loc[:, ['resource','response', 'response_lower', 'response_upper']]
    eval_df = eval_df.groupby('resource').mean()
    eval_df.reset_index(inplace=True)
    return params_df, eval_df


def create_eval_params_dfs(filename, folders, parameter_names, type, response_col):
    df_list_perf = []
    df_list_params = []
    df_list_raw = []
    for splitind, folder in enumerate(folders):
        # Load rec_params for this split
        curfile = os.path.join(folder, filename)
        rec_params = pd.read_pickle(curfile) # for random exploration, this is called evaltest
        # get params df and eval_df
        # these are the dataframes created while plotting the performance and strategy plots for each test-train split
        if type == 'baseline_evaluate':
            params_df, eval_df = baseline_evaluate(rec_params, parameter_names, response_col=response_col)
        elif type == "proj_expt_evaluate":
            params_df, eval_df = proj_expt_evaluate(rec_params, parameter_names, response_col=response_col)
        elif type == "random_exp_evaluate":
            eval_test = rec_params
            params_df, eval_df = random_exp_evaluate(eval_test, parameter_names, response_col = response_col)
        elif type == 'seq_search_evaluate':
            eval_test = rec_params
            params_df, eval_df = seq_search_evaluate(eval_test, parameter_names, response_col = response_col)
        
        rec_params['split'] = splitind + 1
        params_df['split'] = splitind + 1
        eval_df['split'] = splitind + 1
        
        df_list_raw.append(rec_params)
        df_list_params.append(params_df)
        df_list_perf.append(eval_df)

    df_raw = pd.concat(df_list_raw, axis=0, ignore_index=True)
    df_params = pd.concat(df_list_params, axis=0, ignore_index=True)
    df_perf = pd.concat(df_list_perf, axis=0, ignore_index=True)
    return df_raw, df_params, df_perf
