from dataclasses import dataclass
import numpy as np


@dataclass
class BootstrapParameters:
    """
    Parameters for the bootstrap method.
    """
    random_value: float = 0.0
    response_col: str = None
    response_dir: int = -1
    best_value: float = None
    success_metric: str = 'perf_ratio'
    resource_col: str = None
    downsample: int = 10
    bootstrap_iterations: int = 1000
    confidence_level: float = 68
    gap: float = 1.0
    s: float = 0.99
    fail_value: float = np.inf


# def initBootstrap(df, bs_params, resamples):
def initBootstrap(df, bs_params):
    """
    Initialize the bootstrap method.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.

    Returns
    -------
    resamples : list
        List of resamples.
    responses : numpy.ndarray
        Array of responses.
    times : numpy.ndarray
        Array of times.
    """
    resamples = np.random.randint(0, len(df), size=(
        bs_params.downsample, bs_params.bootstrap_iterations), dtype=np.intp)
    responses = df[bs_params.response_col].values
    times = df[bs_params.resource_col].values

    if bs_params.best_value is None:
        if bs_params.response_dir == - 1:  # Minimization
            bs_params.best_value = df[bs_params.response_col].min()
        else:  # Maximization
            bs_params.best_value = df[bs_params.response_col].max()
    return resamples, responses, times
    # return responses, times

# def BootstrapSingle(df, bs_params, resamples):


def BootstrapSingle(df, bs_params):
    """
    Bootstrap single function.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.

    Returns
    -------
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    """
    resamples, responses, times = initBootstrap(df, bs_params)
    # responses, times = initBootstrap(df, bs_params, resamples)
    bs_df = pd.DataFrame()
    # TODO Pylance is crying here because pd is not defined
    computeResponse(df, bs_df, bs_params, resamples, responses)
    computePerfRatio(df, bs_df, bs_params)
    computeSuccessProb(df, bs_df, bs_params, resamples, responses)
    computeRTT(df, bs_df, bs_params, resamples, responses)
    computeResource(df, bs_df, bs_params, resamples, times)
    computeInvPerfRatio(df, bs_df, bs_params)
    return bs_df


def Bootstrap(df, group_on, bootstraps):
    """
    Bootstrap function.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    group_on : str
        Column name to group on.
    bootstraps : int
        Number of bootstraps.

    Returns
    -------
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    """
    df_list = []
    for boots in bootstraps:
        bs_params = BootstrapParameters()
        bs_params.downsample = boots
        bs_params.response_col = 'min_energy'
        # THESE NEED TO BE PASSED IN!!!!@!, maybe pass in list of bs_params
        bs_params.resource_col = 'sweep'
        def dfBS(df): return BootstrapSingle(df, bs_params)
        # this creates an additional 'level_x' col that should be droped
        temp_df = df.groupby(group_on).apply(dfBS).reset_index()
        temp_df['boots'] = boots
        df_list.append(temp_df)
    grouped_df = pd.concat(df_list, ignore_index=True)
    return grouped_df


def computeResponse(df, bs_df, bs_params, resamples, responses):
    """
    Compute the response of each bootstrap samples and its corresponding confidence interval based on the resamples.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.
    resamples : numpy.ndarray
        Array of resamples.
    responses : numpy.ndarray
        Array of responses.

    Returns
    -------
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    """
    # Compute the minimum value of each bootstrap samples and its corresponding confidence interval based on the resamples
    if bs_params.response_dir == - 1:  # Minimization
        response_dist = np.apply_along_axis(
            func1d=np.min, axis=0, arr=responses[resamples])
    else:  # Maximization
        response_dist = np.apply_along_axis(
            func1d=np.max, axis=0, arr=responses[resamples])
        # TODO This could be generalized as the X best samples
    bs_df['response'] = [np.mean(response_dist)]
    bs_df['response_conf_interval_lower'] = np.nanpercentile(
        response_dist, 50-confidence_level/2)
    bs_df['response_conf_interval_upper'] = np.nanpercentile(
        response_dist, 50+confidence_level/2)
    # TODO PyLance is crying here because confidence_level is not defined


def computePerfRatio(df, bs_df, bs_params):
    """
    Compute the performance ratio of each bootstrap samples and its corresponding confidence interval based on the resamples.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.
    """
    # Compute the success metric (performance ratio) of each bootstrap samples and its corresponding confidence interval based on the resamples
    if bs_params.success_metric == 'perf_ratio':
        bs_df['perf_ratio'] = (bs_params.random_value - bs_df['response'])\
            / (bs_params.random_value - bs_params.best_value)
        bs_df['perf_ratio_conf_interval_lower'] = (bs_params.random_value - bs_df['response_conf_interval_upper']) \
            / (bs_params.random_value - bs_params.best_value)
        bs_df['perf_ratio_conf_interval_upper'] = (bs_params.random_value - bs_df['response_conf_interval_lower'])\
            / (bs_params.random_value - bs_params.best_value)
    else:
        print("Success metric not implemented yet")
        # TODO here the input could be a function


def computeInvPerfRatio(df, bs_df, bs_params):
    """
    Compute the inverse performance ratio of each bootstrap samples and its corresponding confidence interval based on the resamples.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.
    """
    # Compute the inverse success metric (performance ratio) of each bootstrap samples and its corresponding confidence interval based on the resamples
    if bs_params.success_metric == 'perf_ratio':
        bs_df['inv_perf_ratio'] = 1 - (bs_params.random_value - bs_df['response']) / \
            (bs_params.random_value - bs_params.best_value) + EPSILON
        bs_df['inv_perf_ratio_conf_interval_lower'] = 1 - (bs_params.random_value - bs_df['response_conf_interval_lower'])\
            / (bs_params.random_value - bs_params.best_value) + EPSILON
        bs_df['inv_perf_ratio_conf_interval_upper'] = 1 - (bs_params.random_value - bs_df['response_conf_interval_upper'])\
            / (bs_params.random_value - bs_params.best_value) + EPSILON
        # TODO PyLance is crying here because EPSILON is not defined
    else:
        print("Success metric not implemented yet")
        # TODO here the input could be a function


def computeSuccessProb(df, bs_df, bs_params, resamples, responses):
    """
    Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.
    resamples : numpy.ndarray
        Array of resamples.
    responses : numpy.ndarray
        Array of responses.
    """
    aggregated_df_flag = False
    if bs_params.response_dir == - 1:  # Minimization
        success_val = bs_params.random_value - \
            (1.0 - gap/100.0)*(bs_params.random_value - bs_params.best_value)
    else:  # Maximization
        success_val = (1.0 - gap/100.0) * \
            (bs_params.best_value - bs_params.random_value) - bs_params.random_value
    # TODO Here we only include relative performance ratio. Consider other objectives as in benchopt
    # TODO PyLance is crying here because gap is not defined

    # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
    if aggregated_df_flag:
        print('Aggregated dataframe')
        return []
        # TODO: One can think about desaggregating the dataframe here. Maybe check Dwave's code for this.
    else:
        if bs_params.response_dir == -1:  # Minimization
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x < success_val)/bs_params.downsample, axis=0, arr=responses[resamples])
        else:  # Maximization
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x > success_val)/bs_params.downsample, axis=0, arr=responses[resamples])
    # Consider np.percentile instead to reduce package dependency. We need to benchmark and test alternative
    bs_df['success_prob'] = np.mean(success_prob_dist)
    bs_df['success_prob_conf_interval_lower'] = np.nanpercentile(
        success_prob_dist, 50 - bs_params.confidence_level/2)
    bs_df['success_prob_conf_interval_upper'] = np.nanpercentile(
        success_prob_dist, 50 + bs_params.confidence_level/2)


def computeResource(df, bs_df, bs_params, resamples, times):
    """
    Compute the resource of each bootstrap samples and its corresponding confidence interval based on the resamples.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.
    resamples : numpy.ndarray
        Array of resamples.
    times : numpy.ndarray
        Array of times.
    """
    # Compute the resource (time) of each bootstrap samples and its corresponding confidence interval based on the resamples
    resource_dist = np.apply_along_axis(
        func1d=np.mean, axis=0, arr=times[resamples])
    bs_df['mean_time'] = np.mean(resource_dist)
    bs_df['mean_time_conf_interval_lower'] = np.nanpercentile(
        resource_dist, 50 - bs_params.confidence_level/2)
    bs_df['mean_time_conf_interval_upper'] = np.nanpercentile(
        resource_dist, 50 + bs_params.confidence_level/2)


def computeRTT(df, bs_df, bs_params, resamples, responses):
    """
    Compute the RTT of each bootstrap samples and its corresponding confidence interval based on the resamples.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    bs_df : pandas.DataFrame
        DataFrame containing the bootstrap results.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.
    resamples : numpy.ndarray
        Array of resamples.
    responses : numpy.ndarray
        Array of responses.
    """
    if bs_params.response_dir == - 1:  # Minimization
        success_val = bs_params.random_value - \
            (1.0 - gap/100.0)*(bs_params.random_value - bs_params.best_value)
    else:  # Maximization
        success_val = (1.0 - gap/100.0) * \
            (bs_params.best_value - bs_params.random_value) - bs_params.random_value
    aggregated_df_flag = False
    # Compute the resource to target (RTT) within certain threshold of each bootstrap
    # samples and its corresponding confidence interval based on the resamples
    rtt_factor = 1e-6*df[bs_params.resource_col].sum()

    # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
    if aggregated_df_flag:
        print('Aggregated dataframe')
        return []
        # TODO: One can think about desaggregating the dataframe here. Maybe check Dwave's code for this.
    else:
        if bs_params.response_dir == -1:  # Minimization
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x < success_val)/bs_params.downsample, axis=0, arr=responses[resamples])
        else:  # Maximization
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x > success_val)/bs_params.downsample, axis=0, arr=responses[resamples])

    rtt_dist = computeRTT_vectorized(
        success_prob_dist, bs_params, scale=rtt_factor)
    # Question: should we scale the RTT with the number of bootstrapping we do, intuition says we don't need to
    rtt = np.mean(rtt_dist)
    bs_df['rtt'] = rtt
    if np.isinf(rtt) or np.isnan(rtt) or rtt == bs_params.fail_value:
        bs_df['rtt_conf_interval_lower'] = bs_params.fail_value
        bs_df['rtt_conf_interval_upper'] = bs_params.fail_value
    else:
        # rtt_conf_interval = computeRTT_vectorized(
        #     success_prob_conf_interval, s=0.99, scale=1e-6*df_default_samples['runtime (us)'].sum())
        bs_df['rtt_conf_interval_lower'] = np.nanpercentile(
            rtt_dist, 50-confidence_level/2)
        bs_df['rtt_conf_interval_upper'] = np.nanpercentile(
            rtt_dist, 50+confidence_level/2)
        # TODO: Pylance is crying here because confidence interval is not defined.
    # Question: How should we compute the confidence interval of the RTT? Should we compute the function on the confidence interval of the probability or compute the confidence interval over the RTT distribution?


def computeRTTSingle(success_probability: float, bs_params, scale: float = 1.0, size: int = 1000):
    '''
    Computes the resource to target metric given some success probabilty of getting to that target and a scale factor.

    Parameters
    ----------
    success_probability : float
        The success probability of getting to the target.
    bs_params : BootstrapParameters
        Parameters for the bootstrap method.
    scale : float
        The scale factor.
    size : int
        The number of bootstrap samples to use.

    Returns
    -------
    rtt : float
        The resource to target metric.
    '''
    # Defaults to np.inf but then overwrites (if None to nan)
    if bs_params.fail_value is None:
        bs_params.fail_value = np.nan
    if success_probability == 0:
        return bs_params.fail_value
    elif success_probability == 1:
        # Consider continuous RTT and RTT scaled by assuming success_probability=1 as success_probability=1/size*(1-1/10)
        return scale*np.log(1.0 - bs_params.s) / np.log(1 - (1 - 1/10)/size)
    else:
        return scale*np.log(1.0 - bs_params.s) / np.log(1 - success_probability)


computeRTT_vectorized = np.vectorize(computeRTTSingle, excluded=(1, 2, 3))


# Everything below is for testing purposes, and can be deleted once tested
# Bootstrapping done over all reads for particular param setting and instance
def computeResultsList(
    df: pd.DataFrame,
    resamples,
    random_value: float = 0.0,
    response_column: str = None,
    response_direction: int = -1,
    best_value: float = None,
    success_metric: str = 'perf_ratio',
    resource_column: str = None,
    downsample: int = 10,
    bootstrap_iterations: int = 1000,
    confidence_level: float = 68,
    gap: float = 1.0,
    s: float = 0.99,
    fail_value: float = np.inf,
    overwrite_pickles: bool = False,
    ocean_df_flag: bool = True
) -> list:
    '''
    Compute a list of the results computed for analysis given a dataframe from a solver.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    resamples : numpy.ndarray
        Array of resamples.
    random_value : float
        The random value to use for the bootstrap.
    response_column : str
        The column containing the response.
    response_direction : int
        The direction of the response.
    best_value : float
        The best value to use for the bootstrap.
    success_metric : str
        The success metric to use for the bootstrap.
    resource_column : str
        The column containing the resource.
    downsample : int
        The downsample to use for the bootstrap.
    bootstrap_iterations : int
        The number of bootstrap iterations to use.
    confidence_level : float
        The confidence level to use for the bootstrap.
    gap : float
        The gap to use for the bootstrap.
    s : float
        The s to use for the bootstrap.
    fail_value : float
        The fail value to use for the bootstrap.
    overwrite_pickles : bool

    Returns
    -------
    list
        A list of the results computed for analysis.

    TODO: Here we assume the succes metric is the performance ratio, we can generalize that as any function of the parameters (use external function)
    TODO: Here we only return a few parameters with confidence intervals w.r.t. the bootstrapping. We can generalize that as any possible outcome (use function)
    '''

    aggregated_df_flag = False
    if response_column is None:
        if ocean_df_flag:
            response_column = 'energy'
        else:  # Assume it is a PySA dataframe
            response_column = 'best_energy'
    if resource_column is None:
        resource_column = 'runtime (us)'

    if best_value is None:
        if response_direction == - 1:  # Minimization
            best_value = df[response_column].min()
        else:  # Maximization
            best_value = df[response_column].max()

    if response_direction == - 1:  # Minimization
        success_val = random_value - \
            (1.0 - gap/100.0)*(random_value - best_value)
    else:  # Maximization
        success_val = (1.0 - gap/100.0) * \
            (best_value - random_value) - random_value
        # TODO Here we only include relative performance ratio. Consider other objectives as in benchopt

    # resamples = np.random.randint(0, len(df), size=(
    #     downsample, bootstrap_iterations), dtype=np.intp)

    responses = df[response_column].values
    times = df[resource_column].values
    rtt_factor = 1e-6*df[resource_column].sum()
    # TODO Change this to be general for PySA dataframes
    if 'num_occurrences' in df.columns and not np.all(df['num_occurrences'].values == 1):
        print('The input dataframe is aggregated')
        # occurrences = df['num_occurrences'].values
        aggregated_df_flag = True

    # Compute the minimum value of each bootstrap samples and its corresponding confidence interval based on the resamples
    if response_direction == - 1:  # Minimization
        response_dist = np.apply_along_axis(
            func1d=np.min, axis=0, arr=responses[resamples])
    else:  # Maximization
        response_dist = np.apply_along_axis(
            func1d=np.max, axis=0, arr=responses[resamples])
        # TODO This could be generalized as the X best samples
    response = np.mean(response_dist)
    response_conf_interval_lower = np.nanpercentile(
        response_dist, 50-confidence_level/2)
    response_conf_interval_upper = np.nanpercentile(
        response_dist, 50+confidence_level/2)

    # Compute the resource (time) of each bootstrap samples and its corresponding confidence interval based on the resamples
    resource_dist = np.apply_along_axis(
        func1d=np.mean, axis=0, arr=times[resamples])
    mean_time = np.mean(resource_dist)
    mean_time_conf_interval_lower = np.nanpercentile(
        resource_dist, 50-confidence_level/2)
    mean_time_conf_interval_upper = np.nanpercentile(
        resource_dist, 50+confidence_level/2)

    # Compute the success metric (performance ratio) of each bootstrap samples and its corresponding confidence interval based on the resamples
    if success_metric == 'perf_ratio':
        perf_ratio = (random_value - response) / (random_value - best_value)
        perf_ratio_conf_interval_lower = (random_value - response_conf_interval_upper) / (
            random_value - best_value)
        perf_ratio_conf_interval_upper = (
            random_value - response_conf_interval_lower) / (random_value - best_value)
    else:
        print("Success metric not implemented yet")
        # TODO here the input could be a function

    # Compute the inverse success metric (performance ratio) of each bootstrap samples and its corresponding confidence interval based on the resamples
    if success_metric == 'perf_ratio':
        inv_perf_ratio = 1 - (random_value - response) / \
            (random_value - best_value) + EPSILON
        inv_perf_ratio_conf_interval_lower = 1 - (random_value - response_conf_interval_lower) / (
            random_value - best_value) + EPSILON
        inv_perf_ratio_conf_interval_upper = 1 - (
            random_value - response_conf_interval_upper) / (random_value - best_value) + EPSILON
        # TODO PyLance is crying here because EPSILON is not defined
    else:
        print("Success metric not implemented yet")
        # TODO here the input could be a function

    # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
    if aggregated_df_flag:
        print('Aggregated dataframe')
        return []
        # TODO: One can think about desaggregating the dataframe here. Maybe check Dwave's code for this.
    else:
        if response_direction == -1:  # Minimization

            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x < success_val)/downsample, axis=0, arr=responses[resamples])
        else:  # Maximization
            success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
                x > success_val)/downsample, axis=0, arr=responses[resamples])
    # Consider np.percentile instead to reduce package dependency. We need to benchmark and test alternative
    success_prob = np.mean(success_prob_dist)
    success_prob_conf_interval_lower = np.nanpercentile(
        success_prob_dist, 50-confidence_level/2)
    success_prob_conf_interval_upper = np.nanpercentile(
        success_prob_dist, 50+confidence_level/2)

    # Compute the resource to target (RTT) within certain threshold of each bootstrap samples and its corresponding confidence interval based on the resamples
    rtt_dist = computeRTT_vectorizedOld(
        success_prob_dist, s=s, scale=rtt_factor, fail_value=fail_value)
    # Question: should we scale the RTT with the number of bootstrapping we do, intuition says we don't need to
    rtt = np.mean(rtt_dist)
    if np.isinf(rtt) or np.isnan(rtt) or rtt == fail_value:
        rtt_conf_interval_lower = fail_value
        rtt_conf_interval_upper = fail_value
    else:
        # rtt_conf_interval = computeRTT_vectorized(
        #     success_prob_conf_interval, s=0.99, scale=1e-6*df_default_samples['runtime (us)'].sum())
        rtt_conf_interval_lower = np.nanpercentile(
            rtt_dist, 50-confidence_level/2)
        rtt_conf_interval_upper = np.nanpercentile(
            rtt_dist, 50+confidence_level/2)
    # Question: How should we compute the confidence interval of the RTT? Should we compute the function on the confidence interval of the probability or compute the confidence interval over the RTT distribution?

    return [downsample,
            response, response_conf_interval_lower, response_conf_interval_upper,
            perf_ratio, perf_ratio_conf_interval_lower, perf_ratio_conf_interval_upper,
            success_prob, success_prob_conf_interval_lower, success_prob_conf_interval_upper,
            rtt, rtt_conf_interval_lower, rtt_conf_interval_upper,
            mean_time, mean_time_conf_interval_lower, mean_time_conf_interval_upper,
            inv_perf_ratio, inv_perf_ratio_conf_interval_lower, inv_perf_ratio_conf_interval_upper]


def computeRTTOld(
    success_probability: float,
    s: float = 0.99,
    scale: float = 1.0,
    fail_value: float = None,
    size: int = 1000,
):
    '''
    Computes the resource to target metric given some success probabilty of getting to that target and a scale factor.

    Parameters
    ----------
    success_probability : float
        The success probability of getting to the target.
    s : float
        The confidence level of the success probability.
    scale : float
        The scale factor of the resource to target metric.
    fail_value : float
        The value to return if the resource to target metric is infinite or nan.
    size : int
        The size of the random sample to use.

    Returns
    -------
    float
        The resource to target metric.
    '''
    if fail_value is None:
        fail_value = np.nan
    if success_probability == 0:
        return fail_value
    elif success_probability == 1:
        # Consider continuous RTT and RTT scaled by assuming success_probability=1 as success_probability=1/size*(1-1/10)
        return scale*np.log(1.0 - s) / np.log(1 - (1 - 1/10)/size)
    else:
        return scale*np.log(1.0 - s) / np.log(1 - success_probability)


computeRTT_vectorizedOld = np.vectorize(computeRTTOld, excluded=(1, 2, 3, 4))
