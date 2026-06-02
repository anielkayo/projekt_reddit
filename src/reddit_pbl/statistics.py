from __future__ import annotations

import math

import pandas as pd


def compare_groups(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    negative_value: int = -1,
    positive_value: int = 1,
) -> dict[str, float]:
    """Return simple descriptive statistics for negative and positive links."""
    negative = pd.to_numeric(df.loc[df[group_col] == negative_value, value_col], errors="coerce").dropna()
    positive = pd.to_numeric(df.loc[df[group_col] == positive_value, value_col], errors="coerce").dropna()

    negative_mean = float(negative.mean())
    positive_mean = float(positive.mean())
    return {
        "negative_n": int(negative.shape[0]),
        "positive_n": int(positive.shape[0]),
        "negative_mean": negative_mean,
        "positive_mean": positive_mean,
        "difference_negative_minus_positive": negative_mean - positive_mean,
        "negative_median": float(negative.median()),
        "positive_median": float(positive.median()),
    }


def point_biserial(y_binary: pd.Series, feature: pd.Series) -> float:
    """Compute a point-biserial correlation for a binary label and numeric feature."""
    data = pd.DataFrame(
        {
            "y": pd.to_numeric(y_binary, errors="coerce"),
            "x": pd.to_numeric(feature, errors="coerce"),
        }
    ).dropna()
    if data["y"].nunique() != 2 or data["x"].nunique() < 2:
        return math.nan
    return float(data["y"].corr(data["x"]))
