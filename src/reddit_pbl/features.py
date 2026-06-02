from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd


def sentiment_to_binary_negative(values: pd.Series) -> pd.Series:
    """Map LINK_SENTIMENT values to 1 for negative links and 0 otherwise."""
    return (values == -1).astype(int)


def add_pair_history_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "TIMESTAMP",
    source_col: str = "SOURCE_SUBREDDIT",
    target_col: str = "TARGET_SUBREDDIT",
    anger_col: str = "LIWC_Anger",
    window_hours: int = 24,
) -> pd.DataFrame:
    """Add prior 24h interaction and anger features for each directed subreddit pair."""
    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    result = result.sort_values(timestamp_col).reset_index(drop=True)

    histories: dict[tuple[str, str], deque[tuple[pd.Timestamp, float]]] = defaultdict(deque)
    counts: list[int] = []
    mean_anger: list[float] = []
    max_anger: list[float] = []
    window = pd.Timedelta(hours=window_hours)

    for row in result.itertuples(index=False):
        timestamp = getattr(row, timestamp_col)
        pair = (getattr(row, source_col), getattr(row, target_col))
        anger = float(getattr(row, anger_col))
        history = histories[pair]

        while history and timestamp - history[0][0] > window:
            history.popleft()

        anger_values = [item[1] for item in history]
        counts.append(len(history))
        mean_anger.append(float(sum(anger_values) / len(anger_values)) if anger_values else 0.0)
        max_anger.append(float(max(anger_values)) if anger_values else 0.0)

        history.append((timestamp, anger))

    result["prev_pair_interactions_24h"] = counts
    result["prev_pair_mean_anger_24h"] = mean_anger
    result["prev_pair_max_anger_24h"] = max_anger
    return result


def add_structural_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "TIMESTAMP",
    source_col: str = "SOURCE_SUBREDDIT",
    target_col: str = "TARGET_SUBREDDIT",
) -> pd.DataFrame:
    """Add simple network-history features computed only from previous rows."""
    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    result = result.sort_values(timestamp_col).reset_index(drop=True)

    source_counts: defaultdict[str, int] = defaultdict(int)
    target_counts: defaultdict[str, int] = defaultdict(int)
    pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    neighbors: defaultdict[str, set[str]] = defaultdict(set)

    previous_source_links: list[int] = []
    previous_target_links: list[int] = []
    previous_pair_links: list[int] = []
    common_neighbors: list[int] = []

    for row in result.itertuples(index=False):
        source = getattr(row, source_col)
        target = getattr(row, target_col)
        pair = (source, target)

        previous_source_links.append(source_counts[source])
        previous_target_links.append(target_counts[target])
        previous_pair_links.append(pair_counts[pair])
        common_neighbors.append(len(neighbors[source].intersection(neighbors[target])))

        source_counts[source] += 1
        target_counts[target] += 1
        pair_counts[pair] += 1
        neighbors[source].add(target)
        neighbors[target].add(source)

    result["source_previous_links"] = previous_source_links
    result["target_previous_links"] = previous_target_links
    result["pair_previous_links"] = previous_pair_links
    result["common_neighbors_so_far"] = common_neighbors
    return result
