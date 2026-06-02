import pandas as pd

from src.reddit_pbl.preparation import (
    build_feature_metadata,
    build_quality_report,
    prepare_dataset,
    write_prepared_artifacts,
)


def _sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "POST_ID": ["p3", "p1", "p4", "p2"],
            "Content_Sentiment": [0, -1, 1, 0],
            "TIMESTAMP": [
                "2017-01-02 01:00:00",
                "2017-01-01 00:00:00",
                "2017-01-02 02:00:00",
                "2017-01-01 12:00:00",
            ],
            "Content_Score": [0.70, 0.95, 0.80, 0.65],
            "Raw_Content": ["third body", "first body", "fourth body", "second body"],
            "Raw_Title": ["third", "first", "fourth", "second"],
            "SOURCE_SUBREDDIT": ["a", "a", "b", "a"],
            "TARGET_SUBREDDIT": ["b", "b", "a", "b"],
            "LINK_SENTIMENT": [-1, 1, 1, -1],
            "LIWC_Anger": [0.50, 0.10, 0.90, 0.30],
            "Average word length": [5.0, 4.0, 6.0, 4.5],
            "Number of words": [20.0, 10.0, 30.0, 15.0],
        }
    )


def test_prepare_dataset_adds_targets_text_history_and_chronological_split():
    result = prepare_dataset(_sample_dataset(), train_fraction=0.5)

    assert result["POST_ID"].tolist() == ["p1", "p2", "p3", "p4"]
    assert result["is_negative_link"].tolist() == [0, 1, 1, 0]
    assert result["combined_text"].tolist()[0] == "first\nfirst body"
    assert result["prev_pair_interactions_24h"].tolist() == [0, 1, 1, 0]
    assert result["prev_pair_mean_anger_24h"].round(3).tolist() == [0.0, 0.1, 0.3, 0.0]
    assert result["source_previous_links"].tolist() == [0, 1, 2, 0]
    assert result["target_previous_links"].tolist() == [0, 1, 2, 0]
    assert result["pair_previous_links"].tolist() == [0, 1, 2, 0]
    assert result["high_previous_anger_24h"].tolist() == [False, False, True, False]
    assert result["split_chronological"].tolist() == ["train", "train", "test", "test"]


def test_build_quality_report_summarizes_model_readiness():
    prepared = prepare_dataset(_sample_dataset(), train_fraction=0.5)

    report = build_quality_report(prepared)

    assert report["rows"] == 4
    assert report["columns"] == len(prepared.columns)
    assert report["missing_values_total"] == 0
    assert report["duplicate_post_ids"] == 0
    assert report["negative_links"] == 2
    assert report["negative_rate"] == 0.5
    assert report["train_rows"] == 2
    assert report["test_rows"] == 2


def test_build_feature_metadata_lists_target_text_and_indicator_features():
    prepared = prepare_dataset(_sample_dataset(), train_fraction=0.5)

    metadata = build_feature_metadata(prepared)

    assert metadata["target"] == "is_negative_link"
    assert metadata["split_column"] == "split_chronological"
    assert "combined_text" in metadata["text_features"]
    assert "high_previous_anger_24h" in metadata["indicator_features"]


def test_write_prepared_artifacts_creates_dataset_report_and_metadata(tmp_path):
    prepared = prepare_dataset(_sample_dataset(), train_fraction=0.5)

    paths = write_prepared_artifacts(
        prepared,
        prepared_dataset_path=tmp_path / "prepared.csv",
        output_dir=tmp_path / "outputs",
    )

    assert paths["prepared_dataset"].exists()
    assert paths["quality_report_csv"].exists()
    assert paths["metadata_json"].exists()
    assert paths["markdown_report"].exists()
    assert pd.read_csv(paths["prepared_dataset"]).shape[0] == 4
    assert '"target": "is_negative_link"' in paths["metadata_json"].read_text(encoding="utf-8")
    assert "negative_rate" in paths["quality_report_csv"].read_text(encoding="utf-8")
