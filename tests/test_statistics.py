import pandas as pd

from src.reddit_pbl.statistics import compare_groups, point_biserial


def test_compare_groups_returns_group_means_and_difference():
    df = pd.DataFrame(
        {
            "LINK_SENTIMENT": [-1, -1, 1, 1],
            "Average word length": [4.0, 5.0, 7.0, 9.0],
        }
    )

    result = compare_groups(
        df,
        group_col="LINK_SENTIMENT",
        value_col="Average word length",
        negative_value=-1,
        positive_value=1,
    )

    assert result["negative_mean"] == 4.5
    assert result["positive_mean"] == 8.0
    assert result["difference_negative_minus_positive"] == -3.5


def test_point_biserial_is_positive_when_feature_is_larger_for_negative_class():
    y_negative = pd.Series([1, 1, 0, 0])
    feature = pd.Series([10.0, 8.0, 2.0, 1.0])

    result = point_biserial(y_negative, feature)

    assert result > 0.9
