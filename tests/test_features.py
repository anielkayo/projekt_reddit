import pandas as pd

from src.reddit_pbl.features import (
    add_pair_history_features,
    add_structural_features,
    sentiment_to_binary_negative,
)


def test_sentiment_to_binary_negative_maps_minus_one_to_one():
    values = pd.Series([-1, 1, -1, 1])

    result = sentiment_to_binary_negative(values)

    assert result.tolist() == [1, 0, 1, 0]


def test_add_pair_history_features_uses_only_previous_24_hours_for_same_pair():
    df = pd.DataFrame(
        {
            "TIMESTAMP": pd.to_datetime(
                [
                    "2017-01-01 00:00:00",
                    "2017-01-01 12:00:00",
                    "2017-01-02 01:00:00",
                    "2017-01-02 02:00:00",
                    "2017-01-02 03:00:00",
                ]
            ),
            "SOURCE_SUBREDDIT": ["a", "a", "a", "b", "a"],
            "TARGET_SUBREDDIT": ["b", "b", "b", "a", "c"],
            "LIWC_Anger": [0.10, 0.30, 0.50, 0.90, 0.70],
        }
    )

    result = add_pair_history_features(df)

    assert result["prev_pair_interactions_24h"].tolist() == [0, 1, 1, 0, 0]
    assert result["prev_pair_mean_anger_24h"].round(3).tolist() == [0.0, 0.1, 0.3, 0.0, 0.0]
    assert result["prev_pair_max_anger_24h"].round(3).tolist() == [0.0, 0.1, 0.3, 0.0, 0.0]


def test_add_structural_features_counts_previous_source_target_and_pair_activity():
    df = pd.DataFrame(
        {
            "TIMESTAMP": pd.to_datetime(
                [
                    "2017-01-01 00:00:00",
                    "2017-01-01 01:00:00",
                    "2017-01-01 02:00:00",
                    "2017-01-01 03:00:00",
                ]
            ),
            "SOURCE_SUBREDDIT": ["a", "a", "b", "a"],
            "TARGET_SUBREDDIT": ["b", "c", "a", "b"],
        }
    )

    result = add_structural_features(df)

    assert result["source_previous_links"].tolist() == [0, 1, 0, 2]
    assert result["target_previous_links"].tolist() == [0, 0, 0, 1]
    assert result["pair_previous_links"].tolist() == [0, 0, 0, 1]
