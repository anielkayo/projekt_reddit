from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.reddit_pbl.columns import COGNITIVE_COMPLEXITY_FEATURES, LINGUISTIC_FEATURES, STRUCTURAL_FEATURES
from src.reddit_pbl.features import add_pair_history_features, add_structural_features, sentiment_to_binary_negative


REQUIRED_COLUMNS = [
    "POST_ID",
    "Content_Sentiment",
    "TIMESTAMP",
    "Content_Score",
    "Raw_Content",
    "Raw_Title",
    "SOURCE_SUBREDDIT",
    "TARGET_SUBREDDIT",
    "LINK_SENTIMENT",
    "LIWC_Anger",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the project CSV with the default pandas parser."""
    return pd.read_csv(path)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error when a required project column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_dataset(df: pd.DataFrame, *, train_fraction: float = 0.8) -> pd.DataFrame:
    """Return a deterministic, analysis-ready dataset without modifying the input."""
    validate_required_columns(df)
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    result = df.copy()
    result["TIMESTAMP"] = pd.to_datetime(result["TIMESTAMP"], errors="raise")
    result = result.sort_values(["TIMESTAMP", "POST_ID"]).reset_index(drop=True)

    result["Raw_Title"] = result["Raw_Title"].fillna("").astype(str)
    result["Raw_Content"] = result["Raw_Content"].fillna("").astype(str)
    result["combined_text"] = result["Raw_Title"].str.strip() + "\n" + result["Raw_Content"].str.strip()
    result["is_negative_link"] = sentiment_to_binary_negative(result["LINK_SENTIMENT"])

    result = add_pair_history_features(result)
    result = add_structural_features(result)
    result["high_previous_anger_24h"] = _high_previous_anger_flag(result)
    result["split_chronological"] = _chronological_split(len(result), train_fraction=train_fraction)
    return result


def build_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize data quality and model-readiness checks for prepared data."""
    validate_required_columns(df)
    timestamps = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    target = sentiment_to_binary_negative(df["LINK_SENTIMENT"])

    report: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_post_ids": int(df["POST_ID"].duplicated().sum()),
        "timestamp_missing_or_bad": int(timestamps.isna().sum()),
        "timestamp_min": str(timestamps.min()),
        "timestamp_max": str(timestamps.max()),
        "negative_links": int(target.sum()),
        "positive_links": int((target == 0).sum()),
        "negative_rate": float(target.mean()),
        "unique_source_subreddits": int(df["SOURCE_SUBREDDIT"].nunique()),
        "unique_target_subreddits": int(df["TARGET_SUBREDDIT"].nunique()),
    }

    if "split_chronological" in df.columns:
        split_counts = df["split_chronological"].value_counts()
        report["train_rows"] = int(split_counts.get("train", 0))
        report["test_rows"] = int(split_counts.get("test", 0))

    return report


def build_feature_metadata(df: pd.DataFrame) -> dict[str, list[str] | str]:
    """List columns by role so notebooks and classifiers can reuse the same schema."""
    numeric_features = [
        column
        for column in LINGUISTIC_FEATURES + STRUCTURAL_FEATURES + ["Content_Score"]
        if column in df.columns
    ]
    categorical_features = [
        column
        for column in ["Content_Sentiment", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]
        if column in df.columns
    ]
    cognitive_features = [column for column in COGNITIVE_COMPLEXITY_FEATURES if column in df.columns]
    indicator_features = [column for column in ["high_previous_anger_24h"] if column in df.columns]

    return {
        "target": "is_negative_link",
        "timestamp": "TIMESTAMP",
        "text_features": ["combined_text", "Raw_Title", "Raw_Content"],
        "numeric_features": numeric_features,
        "indicator_features": indicator_features,
        "categorical_features": categorical_features,
        "cognitive_complexity_features": cognitive_features,
        "structural_features": [column for column in STRUCTURAL_FEATURES if column in df.columns],
        "split_column": "split_chronological",
    }


def write_prepared_artifacts(
    prepared_df: pd.DataFrame,
    *,
    prepared_dataset_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the prepared dataset plus compact reports used by notebooks."""
    prepared_dataset_path = Path(prepared_dataset_path)
    output_dir = Path(output_dir)
    prepared_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_report = build_quality_report(prepared_df)
    metadata = build_feature_metadata(prepared_df)

    quality_report_csv = output_dir / "data_quality_report.csv"
    metadata_json = output_dir / "prepared_dataset_metadata.json"
    markdown_report = output_dir / "data_preparation_report.md"

    prepared_df.to_csv(prepared_dataset_path, index=False)
    pd.DataFrame(
        [{"metric": metric, "value": value} for metric, value in quality_report.items()]
    ).to_csv(quality_report_csv, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_report.write_text(_markdown_report(quality_report, metadata), encoding="utf-8")

    return {
        "prepared_dataset": prepared_dataset_path,
        "quality_report_csv": quality_report_csv,
        "metadata_json": metadata_json,
        "markdown_report": markdown_report,
    }


def _high_previous_anger_flag(df: pd.DataFrame) -> pd.Series:
    history = df.loc[df["prev_pair_interactions_24h"] > 0, "prev_pair_mean_anger_24h"]
    if history.empty:
        return pd.Series(False, index=df.index)

    threshold = history.quantile(0.75)
    return (df["prev_pair_interactions_24h"] > 0) & (df["prev_pair_mean_anger_24h"] >= threshold)


def _chronological_split(rows: int, *, train_fraction: float) -> list[str]:
    cutoff = int(rows * train_fraction)
    return ["train" if index < cutoff else "test" for index in range(rows)]


def _markdown_report(quality_report: dict[str, Any], metadata: dict[str, list[str] | str]) -> str:
    return "\n".join(
        [
            "# Raport przygotowania danych",
            "",
            "## Jakosc i gotowosc",
            "",
            f"- Liczba obserwacji: {quality_report['rows']}",
            f"- Liczba kolumn po przygotowaniu: {quality_report['columns']}",
            f"- Braki danych lacznie: {quality_report['missing_values_total']}",
            f"- Zduplikowane POST_ID: {quality_report['duplicate_post_ids']}",
            f"- Zakres czasu: {quality_report['timestamp_min']} - {quality_report['timestamp_max']}",
            f"- Linki negatywne: {quality_report['negative_links']}",
            f"- Odsetek linkow negatywnych: {quality_report['negative_rate']:.4f}",
            f"- Wiersze train: {quality_report.get('train_rows', 0)}",
            f"- Wiersze test: {quality_report.get('test_rows', 0)}",
            "",
            "## Kolumny modelowe",
            "",
            f"- Target: `{metadata['target']}`",
            f"- Split: `{metadata['split_column']}`",
            f"- Tekst: {', '.join(f'`{column}`' for column in metadata['text_features'])}",
            f"- Liczba cech numerycznych: {len(metadata['numeric_features'])}",
            f"- Liczba cech wskaznikowych: {len(metadata['indicator_features'])}",
            f"- Liczba cech strukturalnych: {len(metadata['structural_features'])}",
            "",
            "## Dopasowanie do hipotez",
            "",
            "- H1: uzyj `high_previous_anger_24h`, `prev_pair_mean_anger_24h` i targetu `is_negative_link`.",
            "- H2: uzyj listy `cognitive_complexity_features` oraz porownania wzgledem `is_negative_link`.",
            "- H3: uzyj `combined_text`, cech numerycznych, cech strukturalnych i `split_chronological`.",
            "",
        ]
    )
