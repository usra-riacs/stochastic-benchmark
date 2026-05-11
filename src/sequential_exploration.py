from dataclasses import dataclass, field
from itertools import product

from tqdm import tqdm
import logging

import pandas as pd
import df_utils
import names
from utils_ws import *

import stats

logger = logging.getLogger(__name__)

tqdm.pandas()


@dataclass
class SequentialSearchParameters:
    """
    Defines parameters for sequential search experiments

    Parameters
    ----------
    budgets : list[int]
        Defines total resource budget
    exploration_fracs : list[float]
        Fractions of the budgets that should be used for explorations
    taus : list[int]
        List of how many time steps to explore each parameter setting for
    order_cols : list[str]
        Columns that contain the ordering for sequential search. Each item corresponds to a new experiment
    optimization_dir : int
        Whether we want to maximize (1) or minimize(-1) the response
    parameter_names : list[str]
        The parameter names - should be columns in the dataframe
    key : str
        Indicates the response column
    stat_measure : stats.StatsMeasure
        How the experimental results should be aggregated over experiments

    """

    budgets: list = field(
        default_factory=lambda: [
            i * 10**j for i in [1, 1.5, 2, 3, 5, 7] for j in [3, 4, 5]
        ]
        + [1e6]
    )
    exploration_fracs: list = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.75]
    )
    taus: list = field(
        default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    )
    order_cols: list = field(default_factory=lambda: ["order"])
    optimization_dir: int = 1
    parameter_names: list = field(default_factory=lambda: ["sweep", "replica"])
    key: str = "PerfRatio"
    stat_measure: stats.StatsMeasure = stats.Mean()


def prepare_search(stats_df: pd.DataFrame, ssParams: SequentialSearchParameters):
    """
    Prepares search on stats_df by aligning taus with resource values seen in stats_df

    Parameters
    ----------
    stats_df : pd.DataFrame
        Dataframe that random search will be conducted on
    rsParams : RandomSearchParameters
        Parameters to align with stats_df resources

    Returns
    -------
        None : modifies rsParams.taus
    """
    resource_values = list(stats_df["resource"])
    ssParams.taus = np.unique([take_closest(resource_values, r) for r in ssParams.taus])


def summarize_experiments(df: pd.DataFrame, ssParams: SequentialSearchParameters):
    """
    Aggregates experiments using rsParams.stat_measure over TotalBudget, ExplorationBudget and taus.
    For each

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the results from all experiments
    ssParams : SequentialSearchParameters

    Returns
    -------
    best_agg_alloc : pd.DataFrame
        Best allocation of taus and exploration fraction for each value of the TotalBudget
    exp_at_best : pd.DataFrame
        Experiments that correspond to the best_agg_alloc values. Used for determining best found parameters
        and response for each experiment.
    """
    group_on = ["TotalBudget", "ExplorationBudget", "tau"]
    summary = df.groupby(group_on).agg(ssParams.stat_measure)
    summary.reset_index(inplace=True)

    if ssParams.optimization_dir == 1:
        best_agg_alloc = summary.sort_values(
            ssParams.key, ascending=False
        ).drop_duplicates("TotalBudget")

    elif ssParams.optimization_dir == -1:
        best_agg_alloc = summary.sort_values(
            ssParams.key, ascending=True
        ).drop_duplicates("TotalBudget")

    df_list = []
    for t in best_agg_alloc["TotalBudget"]:
        row = best_agg_alloc[best_agg_alloc["TotalBudget"] == t]
        df_list.append(
            df[
                (df["tau"] == row["tau"].iloc[0])
                & (df["ExplorationBudget"] == row["ExplorationBudget"].iloc[0])
                & (df["TotalBudget"] == t)
            ]
        )
    exp_at_best = pd.concat(df_list, ignore_index=True)
    return best_agg_alloc, exp_at_best


def SequentialExplorationSingle(
    df_stats: pd.DataFrame,
    ssParams: SequentialSearchParameters,
    experiment: int,
    budget: float,
    explore_frac: float,
    tau: int,
):
    """
    Runs experiment on df_stats for a single setting of the exploration parameters (budget, explore_frac, and tau)

    Parameters
    ----------
    df_stats : pd.DataFrame
        Dataframe of stats that are used in experiments
    ssParams : SequentialSearchParameters
    budget : int
        Total budget
    explore_frac : float
        Fraction of budget to use exploring
    tau : int
        Timesteps to use at each exploration step

    Returns
    -------
    Dataframe representing entire run of the experiment
    """
    explore_budget = budget * explore_frac
    if explore_budget < tau:
        # print('Sequential search experiment terminated due to budget')
        return None
    tau = take_closest(list(df_stats["resource"]), tau)
    df_tau = df_stats[df_stats["resource"] == tau].copy()
    # print(ssParams.order_cols[experiment])
    df_tau.sort_values(
        by=ssParams.order_cols[experiment],
        ascending=True,
        inplace=True,
        ignore_index=True,
    )
    df_tau.dropna(
        axis=0,
        how="any",
        subset=[ssParams.order_cols[experiment], ssParams.key],
        inplace=True,
    )

    if len(df_tau) == 0:
        logger.warning(
            "Sequential search experiment terminated due to insufficient data"
        )
        return None

    n = int(explore_budget / tau)
    df_tau = df_tau.iloc[0 : min(n, len(df_tau))]

    if df_tau.size:
        if ssParams.optimization_dir == 1:
            best_pars = df_tau.loc[[df_tau[ssParams.key].idxmax()]]
            best_val = df_tau[ssParams.key].max()
        elif ssParams.optimization_dir == -1:
            best_pars = df_tau.loc[[df_tau[ssParams.key].idxmin()]]
            best_val = df_tau[ssParams.key].min()
    else:
        logger.warning(
            "Sequential search experiment data is empty. Budget %s expl_frac %s tau %s",
            budget,
            explore_frac,
            tau,
        )
        return None

    df_tau["exploit"] = 0
    df_tau["resource_step"] = df_tau["resource"].copy()
    best_pars = best_pars[ssParams.parameter_names]

    exploit_df = df_stats.merge(best_pars, on=ssParams.parameter_names)
    exploit_df["exploit"] = 1
    exploit_df[ssParams.key] = exploit_df[ssParams.key].fillna(0.0)

    names_dict = names.filename2param(ssParams.key)
    names_dict.update({"ConfInt": "lower"})
    CIlower = names.param2filename(names_dict, "")
    names_dict.update({"ConfInt": "upper"})
    CIupper = names.param2filename(names_dict, "")

    if ssParams.optimization_dir == 1:
        exploit_df.loc[:, [ssParams.key, CIlower, CIupper]].clip(
            lower=best_val, inplace=True
        )
    elif ssParams.optimization_dir == -1:
        exploit_df.loc[:, [ssParams.key, CIlower, CIupper]].clip(
            upper=best_val, inplace=True
        )

    exploit_df["resource_step"] = (
        exploit_df["resource"].diff().fillna(exploit_df["resource"])
    )

    df_experiment = pd.concat([df_tau, exploit_df], ignore_index=True)
    df_experiment["tau"] = tau
    df_experiment["TotalBudget"] = budget
    df_experiment["ExplorationBudget"] = explore_budget
    df_experiment["CummResource"] = (
        df_experiment["resource_step"].expanding(min_periods=1).sum()
    )
    df_experiment = df_experiment[df_experiment["CummResource"] <= budget]

    return df_experiment


def run_experiments(df_stats: pd.DataFrame, ssParams: SequentialSearchParameters):
    """
    Runs all experiments and returns concatenated results

    Parameters
    ----------
    df_stats : pd.DataFrame
        Dataframe of stats that are used in experiments
    ssParams : SequentialSearchParameters

    Returns
    -------
    If single experiment, returns dataframe representing entire run of the experiment
    If multiple experiments, returns dataframe representing entire run of all experiments
    """
    final_values = []
    total = (
        len(ssParams.budgets)
        * len(ssParams.exploration_fracs)
        * len(ssParams.taus)
        * len(ssParams.order_cols)
    )
    pbar = tqdm(
        product(
            ssParams.budgets,
            ssParams.exploration_fracs,
            ssParams.taus,
            range(len(ssParams.order_cols)),
        ),
        total=total,
    )
    for budget, explore_frac, tau, experiment in pbar:
        df_experiment = SequentialExplorationSingle(
            df_stats, ssParams, experiment, budget, explore_frac, tau
        )
        if df_experiment is None:
            continue
        df_experiment["Experiment"] = experiment
        final_values.append(df_experiment.iloc[[-1]])
    if len(final_values) == 0:
        return pd.DataFrame(
            columns=(
                list(df_stats.columns)
                + [
                    "exploit",
                    "tau",
                    "TotalBudget",
                    "ExplorationBudget",
                    "CummResource",
                    "Experiment",
                ]
            )
        )

    else:
        return pd.concat(final_values, ignore_index=True)


def apply_allocations(
    df_stats: pd.DataFrame,
    ssParams: SequentialSearchParameters,
    best_agg_alloc: pd.DataFrame,
    group_on: list,
):
    """
    Applies best allocations to a new dataframe and returns results (useful for evaluating allocations on the
    testing dataset)

    Parameters
    ----------
    df_stats : pd.DataFrame
        Dataframe of stats that are used in experiments
    ssParams : SequentialSearchParameters
    best_agg_alloc : pd.DataFrame
        Dataframe of best allocations
    group_on : list
        List of columns to group on

    Returns
    -------
    pd.DataFrame of results
    """
    final_values = []
    logger.debug("%s", len(best_agg_alloc))
    for _, row in tqdm(best_agg_alloc.iterrows()):
        # print(row)
        budget = row["TotalBudget"]
        explore_budget = row["ExplorationBudget"]
        tau = row["tau"]
        # print(row['tau'], tau)
        # print('------')
        explore_frac = float(explore_budget) / float(budget)
        for experiment in range(len(ssParams.order_cols)):

            def fcn(df):
                res = SequentialExplorationSingle(
                    df, ssParams, experiment, budget, explore_frac, tau
                )
                if res is not None:
                    return res.iloc[[-1]]
                else:
                    return pd.DataFrame()

            df_experiment = df_utils.applyParallel(df_stats.groupby(group_on), fcn)
            if len(df_experiment) >= 1:
                final_values.append(df_experiment)

            # Note: Previous sequential implementation (manual iteration with res_list)
            # was replaced with applyParallel for better performance and cleaner error handling.
            # See git history for the sequential loop pattern if needed for debugging.

    return pd.concat(final_values, ignore_index=True)


def SequentialExploration(
    df_stats: pd.DataFrame, ssParams: SequentialSearchParameters, group_on: list
):
    """
    Runs sequential exploration experiments

    Parameters
    ----------
    df_stats : pd.DataFrame
        Dataframe of stats that are used in experiments
    ssParams : SequentialSearchParameters
    group_on : list
        List of columns to group on

    Returns
    -------
    best_agg_alloc : Best allocation of exploration meta-parameters (explore frac and tau)
    exp_at_best : Corresponding experiments at the best allocations
    final_values : The final values of all experiments
    """
    prepare_search(df_stats, ssParams)
    final_values = df_utils.applyParallel(
        df_stats.groupby(group_on), lambda df: run_experiments(df, ssParams)
    )
    #     final_values = df_stats.groupby(group_on).progress_apply(lambda df: run_experiments(df, ssParams))
    best_agg_alloc, exp_at_best = summarize_experiments(final_values, ssParams)
    return best_agg_alloc, exp_at_best, final_values
