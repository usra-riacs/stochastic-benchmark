import pytest
import pandas as pd
import numpy as np
from stats import StatsParameters, Mean, Median, StatsSingle, applyBounds, Stats


def test_StatsSingle():
    df_single = pd.DataFrame(
        {
            "Key=A": [1, 2, 3, 4, 5],
            "ConfInt=lower_Key=A": [0.8, 1.8, 2.8, 3.8, 4.8],
            "ConfInt=upper_Key=A": [1.2, 2.2, 3.2, 4.2, 5.2],
        }
    )
    stats_params = StatsParameters(
        metrics=["A"],
        stats_measures=[Mean(), Median()],
        lower_bounds={"A": 1},
        upper_bounds={"A": 5},
    )
    result = StatsSingle(df_single, stats_params)
    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame"
    assert len(result) == 1, "The DataFrame should have 1 row"
    assert set(result.columns) == {
        "Key=A_Metric=mean",
        "ConfInt=lower_Key=A_Metric=mean",
        "ConfInt=upper_Key=A_Metric=mean",
        "Key=A_Metric=median",
        "ConfInt=lower_Key=A_Metric=median",
        "ConfInt=upper_Key=A_Metric=median",
        "count",
    }, "The DataFrame should have the correct columns"


def test_applyBounds():
    df = pd.DataFrame(
        {
            "Key=A_Metric=mean": [0.5, 2, 3, 4, 5.5],
            "ConfInt=lower_Key=A_Metric=mean": [0.4, 1.8, 2.8, 3.8, 5.4],
            "ConfInt=upper_Key=A_Metric=mean": [0.6, 2.2, 3.2, 4.2, 5.6],
            "Key=A_Metric=median": [0.5, 2, 3, 4, 5.5],
            "ConfInt=lower_Key=A_Metric=median": [0.4, 1.8, 2.8, 3.8, 5.4],
            "ConfInt=upper_Key=A_Metric=median": [0.6, 2.2, 3.2, 4.2, 5.6],
        }
    )
    stats_params = StatsParameters(
        metrics=["A"],
        stats_measures=[Mean(), Median()],
        lower_bounds={"A": 1},
        upper_bounds={"A": 5},
    )
    applyBounds(df, stats_params)
    assert isinstance(df, pd.DataFrame), "The result should be a DataFrame"
    assert len(df) == 5, "The DataFrame should have 5 rows"
    assert set(df.columns) == {
        "Key=A_Metric=mean",
        "ConfInt=lower_Key=A_Metric=mean",
        "ConfInt=upper_Key=A_Metric=mean",
        "Key=A_Metric=median",
        "ConfInt=lower_Key=A_Metric=median",
        "ConfInt=upper_Key=A_Metric=median",
    }, "The DataFrame should have the correct columns"
    assert (
        df["ConfInt=lower_Key=A_Metric=mean"].min() >= 1
    ), "Lower bound should be applied to the mean"
    assert (
        df["ConfInt=upper_Key=A_Metric=mean"].max() <= 5
    ), "Upper bound should be applied to the mean"
    assert (
        df["ConfInt=lower_Key=A_Metric=median"].min() >= 1
    ), "Lower bound should be applied to the median"
    assert (
        df["ConfInt=upper_Key=A_Metric=mean"].max() <= 5
    ), "Upper bound should be applied to the median"
    assert (
        df["ConfInt=lower_Key=A_Metric=mean"].min() >= 1
    ), "Lower bound should be applied to the mean"
    assert (
        df["ConfInt=upper_Key=A_Metric=mean"].max() <= 5
    ), "Upper bound should be applied to the mean"
    assert (
        df["ConfInt=lower_Key=A_Metric=median"].min() >= 1
    ), "Lower bound should be applied to the median"
    assert (
        df["ConfInt=upper_Key=A_Metric=mean"].max() <= 5
    ), "Upper bound should be applied to the median"


def test_Stats():
    df = pd.DataFrame(
        {
            "Key=A": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "ConfInt=lower_Key=A": [0.8, 1.8, 2.8, 3.8, 4.8, 0.8, 1.8, 2.8, 3.8, 4.8],
            "ConfInt=upper_Key=A": [1.2, 2.2, 3.2, 4.2, 5.2, 1.2, 2.2, 3.2, 4.2, 5.2],
            "Key=B": [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
            "ConfInt=lower_Key=B": [4.8, 3.8, 2.8, 1.8, 0.8, 4.8, 3.8, 2.8, 1.8, 0.8],
            "ConfInt=upper_Key=B": [5.2, 4.2, 3.2, 2.2, 1.2, 5.2, 4.2, 3.2, 2.2, 1.2],
            "C": ["x", "x", "x", "x", "x", "y", "y", "y", "y", "y"],
        }
    )

    stats_params = StatsParameters(
        metrics=["A", "B"],
        stats_measures=[Mean(), Median()],
        lower_bounds={"A": 1, "B": 2},
        upper_bounds={"A": 5, "B": 6},
    )

    center = np.array([1, 2, 3, 4, 5])
    confint_lower = np.array([0.8, 1.8, 2.8, 3.8, 4.8])
    confint_upper = np.array([1.2, 2.2, 3.2, 4.2, 5.2])
    weights = np.array([1, 1, 1, 1, 1])

    deviation = (confint_upper - confint_lower) / 2
    mean_center = np.mean(center)
    mean_deviation = np.sqrt(
        np.sum(weights**2 * deviation**2) / np.sum(weights) ** 2
    )
    mean_confint_lower = mean_center - mean_deviation
    mean_confint_upper = mean_center + mean_deviation
    median_center = np.median(center)
    median_deviation = mean_deviation * np.sqrt(np.pi / 2)
    median_confint_lower = median_center - median_deviation
    median_confint_upper = median_center + median_deviation

    result = Stats(df, stats_params, ["C"])

    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame"
    assert len(result) == 2, "The DataFrame should have 2 rows"
    assert set(result.columns) == {
        "C",
        "Key=A_Metric=mean",
        "ConfInt=lower_Key=A_Metric=mean",
        "ConfInt=upper_Key=A_Metric=mean",
        "Key=A_Metric=median",
        "ConfInt=lower_Key=A_Metric=median",
        "ConfInt=upper_Key=A_Metric=median",
        "Key=B_Metric=mean",
        "ConfInt=lower_Key=B_Metric=mean",
        "ConfInt=upper_Key=B_Metric=mean",
        "Key=B_Metric=median",
        "ConfInt=lower_Key=B_Metric=median",
        "ConfInt=upper_Key=B_Metric=median",
        "count",
    }, "The DataFrame should have the correct columns"
    assert (
        result["Key=A_Metric=mean"] == [mean_center, mean_center]
    ).all(), "The mean should be calculated correctly"
    assert (
        result["Key=A_Metric=median"] == [median_center, median_center]
    ).all(), "The median should be calculated correctly"
    assert (
        result["Key=B_Metric=mean"] == [mean_center, mean_center]
    ).all(), "The mean should be calculated correctly"
    assert (
        result["Key=B_Metric=median"] == [median_center, median_center]
    ).all(), "The median should be calculated correctly"
    assert (result["count"] == [5, 5]).all(), "The count should be calculated correctly"
    assert (result["C"] == ["x", "y"]).all(), "The groupby column should be preserved"
    assert (
        result["ConfInt=lower_Key=A_Metric=mean"]
        == [mean_confint_lower, mean_confint_lower]
    ).all(), "The lower bound should be applied"
    assert (
        result["ConfInt=upper_Key=A_Metric=mean"]
        == [mean_confint_upper, mean_confint_upper]
    ).all(), "The upper bound should be applied"
    assert (
        result["ConfInt=lower_Key=A_Metric=median"]
        == [median_confint_lower, median_confint_lower]
    ).all(), "The lower bound should be applied"
    assert (
        result["ConfInt=upper_Key=A_Metric=median"]
        == [median_confint_upper, median_confint_upper]
    ).all(), "The upper bound should be applied"
    assert (
        result["ConfInt=lower_Key=B_Metric=mean"]
        == [mean_confint_lower, mean_confint_lower]
    ).all(), "The lower bound should be applied"
    assert (
        result["ConfInt=upper_Key=B_Metric=mean"]
        == [mean_confint_upper, mean_confint_upper]
    ).all(), "The upper bound should be applied"
    assert (
        result["ConfInt=lower_Key=B_Metric=median"]
        == [median_confint_lower, median_confint_lower]
    ).all(), "The lower bound should be applied"
    assert (
        result["ConfInt=upper_Key=B_Metric=median"]
        == [median_confint_upper, median_confint_upper]
    ).all(), "The upper bound should be applied"


def test_Stats_2():
    mean_A = 2.85
    mean_B = 3.0
    dev_A = 0.2
    dev_B = 0.34
    center_A = [mean_A + diff for diff in [-2, -1, 0, 1, 2]]
    center_B = [mean_B + diff for diff in [-2, -1, 0, 1, 2]]
    A = center_A + center_A[::-1]
    B = center_B[::-1] + center_B
    confint_lower_A = [i - dev_A for i in A]
    confint_upper_A = [i + dev_A for i in A]
    confint_lower_B = [i - dev_B for i in B]
    confint_upper_B = [i + dev_B for i in B]

    df = pd.DataFrame(
        {
            "Key=A": A,
            "ConfInt=lower_Key=A": confint_lower_A,
            "ConfInt=upper_Key=A": confint_upper_A,
            "Key=B": B,
            "ConfInt=lower_Key=B": confint_lower_B,
            "ConfInt=upper_Key=B": confint_upper_B,
            "C": ["x", "x", "x", "x", "x", "y", "y", "y", "y", "y"],
        }
    )

    stats_params = StatsParameters(
        metrics=["A", "B"],
        stats_measures=[Mean(), Median()],
        lower_bounds={"A": 1, "B": 2},
        upper_bounds={"A": 5, "B": 6},
    )

    center_A = np.array(center_A)
    ci_lower_A = np.array(center_A - dev_A)
    ci_upper_A = np.array(center_A + dev_A)
    weights_A = np.array([1, 1, 1, 1, 1])

    deviation_A = (ci_upper_A - ci_lower_A) / 2
    mean_center_A = np.mean(center_A)
    mean_deviation_A = np.sqrt(
        np.sum(weights_A**2 * deviation_A**2) / np.sum(weights_A) ** 2
    )
    mean_ci_lower_A = mean_center_A - mean_deviation_A
    mean_ci_upper_A = mean_center_A + mean_deviation_A
    median_center_A = np.median(center_A)
    median_deviation_A = mean_deviation_A * np.sqrt(np.pi / 2)
    median_ci_lower_A = median_center_A - median_deviation_A
    median_ci_upper_A = median_center_A + median_deviation_A

    center_B = np.array(center_B)
    ci_lower_B = np.array(center_B - dev_B)
    ci_upper_B = np.array(center_B + dev_B)
    weights_B = np.array([1, 1, 1, 1, 1])

    deviation_B = (ci_upper_B - ci_lower_B) / 2
    mean_center_B = np.mean(center_B)
    mean_deviation_B = np.sqrt(
        np.sum(weights_B**2 * deviation_B**2) / np.sum(weights_B) ** 2
    )
    mean_ci_lower_B = mean_center_B - mean_deviation_B
    mean_ci_upper_B = mean_center_B + mean_deviation_B
    median_center_B = np.median(center_B)
    median_deviation_B = mean_deviation_B * np.sqrt(np.pi / 2)
    median_ci_lower_B = median_center_B - median_deviation_B
    median_ci_upper_B = median_center_B + median_deviation_B

    result = Stats(df, stats_params, ["C"])

    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame"
    assert len(result) == 2, "The DataFrame should have 2 rows"
    assert set(result.columns) == {
        "C",
        "Key=A_Metric=mean",
        "ConfInt=lower_Key=A_Metric=mean",
        "ConfInt=upper_Key=A_Metric=mean",
        "Key=A_Metric=median",
        "ConfInt=lower_Key=A_Metric=median",
        "ConfInt=upper_Key=A_Metric=median",
        "Key=B_Metric=mean",
        "ConfInt=lower_Key=B_Metric=mean",
        "ConfInt=upper_Key=B_Metric=mean",
        "Key=B_Metric=median",
        "ConfInt=lower_Key=B_Metric=median",
        "ConfInt=upper_Key=B_Metric=median",
        "count",
    }, "The DataFrame should have the correct columns"
    assert np.array(result["Key=A_Metric=mean"]) == pytest.approx(
        mean_center_A
    ), "The mean of metric A should be calculated correctly"
    assert np.array(result["Key=A_Metric=median"]) == pytest.approx(
        median_center_A
    ), "The median of metric A should be calculated correctly"
    assert np.array(result["Key=B_Metric=mean"]) == pytest.approx(
        mean_center_B
    ), "The mean of metric B should be calculated correctly"
    assert np.array(result["Key=B_Metric=median"]) == pytest.approx(
        median_center_B
    ), "The median of metric B should be calculated correctly"
    assert (result["count"] == [5, 5]).all(), "The count should be calculated correctly"
    assert (result["C"] == ["x", "y"]).all(), "The groupby column should be preserved"
    assert np.array(result["ConfInt=lower_Key=A_Metric=mean"]) == pytest.approx(
        mean_ci_lower_A
    ), "The lower bound to the mean of metric A should be applied"
    assert np.array(result["ConfInt=upper_Key=A_Metric=mean"]) == pytest.approx(
        mean_ci_upper_A
    ), "The upper bound to the mean of metric A should be applied"
    assert np.array(result["ConfInt=lower_Key=A_Metric=median"]) == pytest.approx(
        median_ci_lower_A
    ), "The lower bound to the median of metric A should be applied"
    assert np.array(result["ConfInt=upper_Key=A_Metric=median"]) == pytest.approx(
        median_ci_upper_A
    ), "The upper bound to the median of metric A should be applied"
    assert np.array(result["ConfInt=lower_Key=B_Metric=mean"]) == pytest.approx(
        mean_ci_lower_B
    ), "The lower bound to the mean of metric B should be applied"
    assert np.array(result["ConfInt=upper_Key=B_Metric=mean"]) == pytest.approx(
        mean_ci_upper_B
    ), "The upper bound to the mean of metric B should be applied"
    assert np.array(result["ConfInt=lower_Key=B_Metric=median"]) == pytest.approx(
        median_ci_lower_B
    ), "The lower bound to the median of metric B should be applied"
    assert np.array(result["ConfInt=upper_Key=B_Metric=median"]) == pytest.approx(
        median_ci_upper_B
    ), "The upper bound to the median of metric B should be applied"
