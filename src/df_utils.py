import glob
from multiprocess import Pool
import os
import pandas as pd
import names
import numpy as np

EPSILON = 1e-10
confidence_level = 68
s = 0.99
gap = 1.0


def applyParallel(dfGrouped, func):
    """
    Apply pandas groupby operation using multiprocessing.
    Taken from https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby/29281494#29281494

    Parameters
    ----------
    dfGrouped : pandas.GroupBy
        groupby object to perform function on
    func : Callable
        Function to apply to each group

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the results from applying func to each group of dfGrouped.
    """
    with Pool() as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)


def monotone_df(
    df, resource_col, response_col, opt_sense, extrapolate_from=None, match_on=None
):
    """
    Makes dataframe monotone with the response column as a function of the response column.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to monotonize
    resource_col : str
        column considered the resource
    response_col : str
        column with the response metric
    opt_sense : int
        Indicates whether response should be increasing (1) or decreasing (-1) as a function of resou
    extrapolate_from : pandas.DataFrame
        DataFrame to extrapolate from if the response is not monotonic against the resource.
    match_on : list[str]
        Columns to match on when extrapolating
        
    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the parameters and response rolled-over if the response is not monotonic against the resource.
    """

    if opt_sense == 1:
        df.sort_values(response_col, ascending=False, inplace=True)
        df.drop_duplicates(resource_col, inplace=True)
    elif opt_sense == -1:
        df = df.sort_values(response_col, ascending=True).drop_duplicates(resource_col)

    df.sort_values(resource_col, ascending=True, inplace=True)
    # df.reset_index(inplace=True)

    if opt_sense == 1:
        running_val = df[response_col].cummax()
    elif opt_sense == -1:
        running_val = df[response_col].cummin()
    dont_extrap = extrapolate_from is None
    count = 0
    rolling_cols = [cols for cols in df.columns if cols != resource_col]
    for idx, row in df.iterrows():
        if count == 0:
            count += 1
            prev_row = row.copy()
        elif running_val[idx] != row[response_col]:
            if dont_extrap:
                df.loc[idx, rolling_cols] = prev_row[rolling_cols].copy()
            else:
                matched_row = extrapolate_from[
                    (extrapolate_from[resource_col] == row[resource_col])
                    & np.all(
                        [extrapolate_from[col] == row[col] for col in match_on], axis=0
                    )
                ]
                if len(matched_row) == 0:
                    print("No matched row found for")
                    print(row)
                    continue
                else:
                    matched_row = matched_row.iloc[0]
                # print(matched_row[rolling_cols])
                # print(df.loc[idx, rolling_cols])
                df.loc[idx, rolling_cols] = matched_row[rolling_cols].copy()
        else:
            prev_row = row.copy()
    df.drop(columns=[c for c in df.columns if "level" in c])
    return df


def eval_cumm(df, group_on, resource_col, response_col, opt_sense):
    """
    Cumulatively evaluates all resources below a set resource value and selects best parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to perform operation on
    group_on : list[str]
        Columns to group on
    resource_col : str
        Column considered the resource
    response_col : str
        Column with the response metric

    Returns
    -------
    df : pd.DataFrame
        Datafram with the cummulative evaluation
    """

    def cummSingle(single_df):
        single_df = monotone_df(single_df.copy(), resource_col, response_col, 1)
        single_df.loc[:, "cummulative" + resource_col] = (
            single_df[resource_col].expanding(min_periods=1).sum()
        )
        return single_df

    cumm_df = df.groupby(group_on).apply(cummSingle)
    return cumm_df


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
    filelist = glob.glob(os.path.join(exp_raw_dir, "*.pkl"))
    df_list = []
    for f in filelist:
        temp_df = pd.read_pickle(f)
        # expand parameters in filename
        if len(name_params) >= 1:
            params_dict = names.filename2param(os.path.basename(f))
            for p in name_params:
                temp_df[p] = params_dict[p]
        df_list.append(temp_df)
    if len(df_list) == 0:
        raise Exception("No raw data found at: {}".format(exp_raw_dir))
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
    df["params"] = df[param_names].apply(tuple, axis=1)
    param_set = df["params"].unique()
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
    response_dir : int
        direction of response (either 'min'=-1 or 'max'=1)
        TODO There has to be a better way of doing this
    group_on : list
        list of column names to group on

    Returns
    -------
    df_best : pd.dataframe
    """
    if response_dir == -1:  # Minimization
        best_df = df.sort_values(response_col, ascending=True).drop_duplicates(group_on)
    else:
        best_df = df.sort_values(response_col, ascending=False).drop_duplicates(
            group_on
        )
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
    rename_dict = {
        "min_energy": names.param2filename({"Key": "MinEnergy"}, ""),
        "min_energy_conf_interval_lower": names.param2filename(
            {"Key": "MinEnergy", "ConfInt": "lower"}, ""
        ),
        "min_energy_conf_interval_upper": names.param2filename(
            {"Key": "MinEnergy", "ConfInt": "upper"}, ""
        ),
        "perf_ratio": names.param2filename({"Key": "PerfRatio"}, ""),
        "perf_ratio_conf_interval_lower": names.param2filename(
            {"Key": "PerfRatio", "ConfInt": "lower"}, ""
        ),
        "perf_ratio_conf_interval_upper": names.param2filename(
            {"Key": "PerfRatio", "ConfInt": "upper"}, ""
        ),
        "success_prob": names.param2filename({"Key": "SuccProb"}, ""),
        "success_prob_conf_interval_lower": names.param2filename(
            {"Key": "SuccProb", "ConfInt": "lower"}, ""
        ),
        "success_prob_conf_interval_upper": names.param2filename(
            {"Key": "SuccProb", "ConfInt": "upper"}, ""
        ),
        "rtt": names.param2filename({"Key": "RTT"}, ""),
        "rtt_conf_interval_lower": names.param2filename(
            {"Key": "RTT", "ConfInt": "lower"}, ""
        ),
        "rtt_conf_interval_upper": names.param2filename(
            {"Key": "RTT", "ConfInt": "upper"}, ""
        ),
        "mean_time": names.param2filename({"Key": "MeanTime"}, ""),
        "mean_time_conf_interval_lower": names.param2filename(
            {"Key": "MeanTime", "ConfInt": "lower"}, ""
        ),
        "mean_time_conf_interval_upper": names.param2filename(
            {"Key": "MeanTime", "ConfInt": "upper"}, ""
        ),
        "inv_perf_ratio": names.param2filename({"Key": "InvPerfRatio"}, ""),
        "inv_perf_ratio_conf_interval_lower": names.param2filename(
            {"Key": "InvPerfRatio", "ConfInt": "lower"}, ""
        ),
        "inv_perf_ratio_conf_interval_upper": names.param2filename(
            {"Key": "InvPerfRatio", "ConfInt": "upper"}, ""
        ),
    }

    df.rename(columns=rename_dict, inplace=True)
    return df
