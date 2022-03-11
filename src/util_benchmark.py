import numpy as np
import pandas as pd
from scipy import sparse, stats

EPSILON = 1e-10

# %%
# Define the default solver parameters
default_sweeps = 1000
total_reads = 1000
default_reads = 1000
default_boots = default_reads


# %%
# (Vectorized) Function to compute Resource to Target given a success_probability (array) float


def computeRTT(
    success_probability: float,
    s: float = 0.99,
    scale: float = 1.0,
    fail_value: float = None,
    size: int = 1000,
):
    '''
    Computes the resource to target metric given some success probabilty of getting to that target and a scale factor.

    Args:
        success_probability: The success probability of getting to the target.
        s: The success factor (usually said as RTT within s% probability).
        scale: The scale factor.
        fail_value: The value to return if the success probability is 0.

    Returns:
        The resource to target metric.
    '''
    if fail_value is None:
        fail_value = np.nan
    if success_probability == 0:
        return fail_value
    elif success_probability == 1:
        # Consider continuous TTS and TTS scaled by assuming success_probability=1 as success_probability=1/size*(1-1/10)
        return scale*np.log(1.0 - s) / np.log(1 - (1 - 1/10)/size)
    else:
        return scale*np.log(1.0 - s) / np.log(1 - success_probability)


computeRRT_vectorized = np.vectorize(computeRTT, excluded=(1, 2, 3, 4))


# %%
# Function to retrieve results from Dataframes
def computeResultsList(
    df: pd.DataFrame,
    random_energy: float = 0.0,
    min_energy: float = None,
    downsample: int = 10,
    bootstrap_iterations: int = 1000,
    confidence_level: float = 68,
    gap: float = 1.0,
    s: float = 0.99,
    fail_value: float = np.inf,
    overwrite_pickles: bool = False,
) -> list:
    '''
    Compute a list of the results computed for analysis given a dataframe from a solver.

    Args:
        df: The dataframe from the solver.
        random_energy: The mean energy of the random sample.
        min_energy: The minimum energy of the samples.
        downsample: The downsampling sample for bootstrapping.
        bootstrap_iterations: The number of bootstrap samples.
        confidence_level: The confidence level for the bootstrap.
        gap: The threshold for the considering a read successful w.r.t the performance ratio [%].
        s: The success factor (usually said as RTT within s% probability).
        overwrite_pickles: If True, the pickles will be overwritten.

    Returns:
        A list of the results computed for analysis. Organized as follows
        [
            number of downsamples,
            bootstrapped mean minimum energy,
            boostrapped mean minimum energy confidence interval lower bound,
            boostrapped mean minimum energy confidence interval upper bound,
            bootstrapped performance ratio,
            bootstrapped performance ratio confidence interval lower bound,
            bootstrapped performance ratio confidence interval upper bound,
            bootstrapped success probability,
            boostrapped success probability confidence interval lower bound,
            boostrapped success probability confidence interval upper bound,
            boostrapped resource to target,
            boostrapped resource to target confidence interval lower bound,
            boostrapped resource to target confidence interval upper bound,
            boostrapped mean runtime,
            boostrapped mean runtime confidence interval lower bound,
            boostrapped mean runtime confidence interval upper bound,
            bootstrapped inverse performance ratio,
            bootstrapped inverse performance ratio confidence interval lower bound,
            bootstrapped inverse performance ratio confidence interval upper bound,
        ]

    TODO: Here we assume the succes metric is the performance ratio, we can generalize that as any function of the parameters (use external function)
    TODO: Here we assume the energy is the response of the solver, we can generalize that as any column in the dataframe
    TODO: Here we only return a few parameters with confidence intervals w.r.t. the bootstrapping. We can generalize that as any possible outcome (use function)
    TODO: Since we are minimizing, computing the performance ratio gets the order of the minimum energy confidence interval inverted. Add parameter for maximization. Need to think what else should we change.
    '''

    aggregated_df_flag = False
    if min_energy is None:
        min_energy = df['energy'].min()

    success_val = random_energy - \
        (1.0 - gap/100.0)*(random_energy - min_energy)

    resamples = np.random.randint(0, len(df), size=(
        downsample, bootstrap_iterations)).astype(int)

    energies = df['energy'].values
    times = df['runtime (us)'].values
    # TODO Change this to be general for PySA dataframes
    if 'num_occurrences' in df.columns and not np.all(df['num_occurrences'].values == 1):
        print('The input dataframe is aggregated')
        occurrences = df['num_occurrences'].values
        aggregated_df_flag = True

    # Compute the minimum value of each bootstrap samples and its corresponding confidence interval based on the resamples
    min_boot_dist = np.apply_along_axis(
        func1d=np.min, axis=0, arr=energies[resamples])
    min_boot = np.mean(min_boot_dist)
    min_boot_conf_interval_lower = stats.scoreatpercentile(
        min_boot_dist, 50-confidence_level/2)
    min_boot_conf_interval_upper = stats.scoreatpercentile(
        min_boot_dist, 50+confidence_level/2)

    # Compute the mean time of each bootstrap samples and its corresponding confidence interval based on the resamples
    times_dist = np.apply_along_axis(
        func1d=np.mean, axis=0, arr=times[resamples])
    mean_time = np.mean(times_dist)
    mean_time_conf_interval_lower = stats.scoreatpercentile(
        times_dist, 50-confidence_level/2)
    mean_time_conf_interval_upper = stats.scoreatpercentile(
        times_dist, 50+confidence_level/2)

    # Compute the performance ratio of each bootstrap samples and its corresponding confidence interval based on the resamples
    perf_ratio = (random_energy - min_boot) / (random_energy - min_energy)
    perf_ratio_conf_interval_lower = (random_energy - min_boot_conf_interval_upper) / (
        random_energy - min_energy)
    perf_ratio_conf_interval_upper = (
        random_energy - min_boot_conf_interval_lower) / (random_energy - min_energy)

    # Compute the inverse performance ratio of each bootstrap samples and its corresponding confidence interval based on the resamples
    inv_perf_ratio = 1 - (random_energy - min_boot) / \
        (random_energy - min_energy) + EPSILON
    inv_perf_ratio_conf_interval_lower = 1 - (random_energy - min_boot_conf_interval_lower) / (
        random_energy - min_energy) + EPSILON
    inv_perf_ratio_conf_interval_upper = 1 - (
        random_energy - min_boot_conf_interval_upper) / (random_energy - min_energy) + EPSILON

    # Compute the success probability of each bootstrap samples and its corresponding confidence interval based on the resamples
    if aggregated_df_flag:
        return []
        # TODO: One can think about deaggregating the dataframe here. Maybe check Dwave's code for this.
    else:
        success_prob_dist = np.apply_along_axis(func1d=lambda x: np.sum(
            x < success_val)/downsample, axis=0, arr=energies[resamples])
    success_prob = np.mean(success_prob_dist)
    success_prob_conf_interval_lower = stats.scoreatpercentile(
        success_prob_dist, 50-confidence_level/2)
    success_prob_conf_interval_upper = stats.scoreatpercentile(
        success_prob_dist, 50+confidence_level/2)

    # Compute the TTT within certain threshold of each bootstrap samples and its corresponding confidence interval based on the resamples
    tts_dist = computeRRT_vectorized(
        success_prob_dist, s=s, scale=1e-6*df['runtime (us)'].sum(), fail_value=fail_value)
    # Question: should we scale the TTS with the number of bootstrapping we do, intuition says we don't need to
    tts = np.mean(tts_dist)
    if np.isinf(tts) or np.isnan(tts) or tts == fail_value:
        tts_conf_interval_lower = fail_value
        tts_conf_interval_upper = fail_value
    else:
        # tts_conf_interval = computeRRT_vectorized(
        #     success_prob_conf_interval, s=0.99, scale=1e-6*df_default_samples['runtime (us)'].sum())
        tts_conf_interval_lower = stats.scoreatpercentile(
            tts_dist, 50-confidence_level/2)
        tts_conf_interval_upper = stats.scoreatpercentile(
            tts_dist, 50+confidence_level/2)
    # Question: How should we compute the confidence interval of the TTS? Should we compute the function on the confidence interval of the probability or compute the confidence interval over the tts distribution?

    return [downsample, min_boot, min_boot_conf_interval_lower, min_boot_conf_interval_upper, perf_ratio, perf_ratio_conf_interval_lower, perf_ratio_conf_interval_upper, success_prob, success_prob_conf_interval_lower, success_prob_conf_interval_upper, tts, tts_conf_interval_lower, tts_conf_interval_upper, mean_time, mean_time_conf_interval_lower, mean_time_conf_interval_upper, inv_perf_ratio, inv_perf_ratio_conf_interval_lower, inv_perf_ratio_conf_interval_upper]


# %%
# Define clean up function


def cleanup_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Function to cleanup dataframes by:
    - From tuple-like confidence intervals by separating it into two columns.
    - Recomputing the reads column.
    - Defining the schedules columns as categoric.

    Args:
        df (pandas.DataFrame): Dataframe to be cleaned.

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    '''
    df_new = df.copy()
    int_columns = ['reads', 'boots', 'sweeps',
                   'R_budget', 'R_exploit', 'R_explore', 'cum_reads', 'swe', 'rep']
    cat_columns = ['schedule', 'instance']
    for column in df_new.columns:
        if column.endswith('conf_interval'):
            df_new[column + '_lower'] = df_new[column].apply(lambda x: x[0])
            df_new[column + '_upper'] = df_new[column].apply(lambda x: x[1])
            df_new.drop(column, axis=1, inplace=True)
        elif column == 'schedule':
            df_new[column] = df_new[column].astype('category')
        elif column.endswith('_conf_interval_lower'):
            # df_new[column] = np.nanmin(df_new[[column, column.removesuffix('_conf_interval_lower')]], axis=1)
            # Only valid for Python 3.9+ PEP-616
            df_new[column] = np.nanmin(df_new[[column, column[:-20]]], axis=1)
        elif column.endswith('_conf_interval_upper'):
            # df_new[column] = np.nanmax(df_new[[column, column.removesuffix('_conf_interval_upper')]], axis=1)
            # Only valid for Python 3.9+ PEP-616
            df_new[column] = np.nanmax(df_new[[column, column[:-20]]], axis=1)
    if 'boots' in df_new.columns:
        if 'sweeps' in df_new.columns:
            df_new['reads'] = df_new['sweeps'] * df_new['boots']
        elif 'swe' in df_new.columns and 'rep' in df_new.columns:
            df_new['reads'] = df_new['swe'] * df_new['rep'] * df_new['boots']
        else:
            df_new['reads'] = default_sweeps * df_new['boots']

    for column in df_new.columns:
        if column in int_columns:
            df_new[column] = df_new[column].astype('int', errors='ignore')
        elif column in cat_columns:
            df_new[column] = df_new[column].astype('category')

    return df_new


# %%
# Define function for ensemble averaging


def mean_conf_interval(
    x: pd.Series,
    key_string: str,
):
    '''
    Compute the mean and confidence interval of a series

    Args:
        x (pd.Series): Series to compute the mean and confidence interval
        key_string (str): String to use as key for the output dataframe

    Returns:
        pd.Series: Series with mean and confidence interval
    '''
    key_mean_string = 'mean_' + key_string
    result = {
        key_mean_string: x[key_string].mean(),
        key_mean_string + '_conf_interval_lower': x[key_string].mean() - np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string]))),
        key_mean_string + '_conf_interval_upper': x[key_string].mean() + np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string])))}
    return pd.Series(result)

# Define function for ensemble median


def median_conf_interval(
    x: pd.Series,
    key_string: str,
):
    '''
    Compute the median and confidence interval of a series (see http://mathworld.wolfram.com/StatisticalMedian.html for uncertainty propagation)

    Args:
        x (pd.Series): Series to compute the median and confidence interval
        key_string (str): String to use as key for the output dataframe

    Returns:
        pd.Series: Series with median and confidence interval
    '''
    key_median_string = 'median_' + key_string
    result = {
        key_median_string: x[key_string].median(),
        key_median_string + '_conf_interval_lower': x[key_string].median() - np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string]))) * np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1)),
        key_median_string + '_conf_interval_upper': x[key_string].median() + np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string]))) * np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1))}
    return pd.Series(result)

# Define function for ensemble metrics


def conf_interval(
    x: pd.Series,
    key_string: str,
    stat_measure: str = 'mean',
):
    '''
    Compute the mean or median and confidence interval of a series (see http://mathworld.wolfram.com/StatisticalMedian.html for uncertainty propagation)

    Args:
        x (pd.Series): Series to compute the median and confidence interval
        key_string (str): String to use as key for the output dataframe
        stat_measure (str): String to use as key for the output dataframe

    Returns:
        pd.Series: Series with median and confidence interval
    '''
    key_median_string = stat_measure + '_' + key_string
    deviation = np.sqrt(sum((x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower'])*(
        x[key_string + '_conf_interval_upper']-x[key_string + '_conf_interval_lower']))/(4*len(x[key_string])))
    if stat_measure == 'mean':
        center = x[key_string].mean()
    else:
        center = x[key_string].median()
        deviation = deviation * \
            np.sqrt(np.pi*len(x[key_string])/(2*len(x[key_string])-1))

    result = {
        key_median_string: center,
        key_median_string + '_conf_interval_lower': center - deviation,
        key_median_string + '_conf_interval_upper': center + deviation}
    return pd.Series(result)

