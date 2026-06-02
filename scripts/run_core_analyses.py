from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reddit_pbl.columns import COGNITIVE_COMPLEXITY_FEATURES, LINGUISTIC_FEATURES, STRUCTURAL_FEATURES
from src.reddit_pbl.features import add_pair_history_features, add_structural_features, sentiment_to_binary_negative
from src.reddit_pbl.statistics import compare_groups, point_biserial


DATA_PATH = Path("database/NajnowszaWersjaBazy1205.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42


def _combined_text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["Raw_Title"].fillna("").astype(str)
        + "\n"
        + frame["Raw_Content"].fillna("").astype(str)
    )


def _metrics(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "negative_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "negative_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "negative_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "tn_fp_fn_tp": confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist(),
    }


def run_hypothesis_1(df: pd.DataFrame) -> pd.DataFrame:
    features = add_pair_history_features(df)
    features["is_negative_link"] = sentiment_to_binary_negative(features["LINK_SENTIMENT"])

    history_rows = features[features["prev_pair_interactions_24h"] > 0].copy()
    threshold = history_rows["prev_pair_mean_anger_24h"].quantile(0.75)
    features["high_previous_anger_24h"] = (
        (features["prev_pair_interactions_24h"] > 0)
        & (features["prev_pair_mean_anger_24h"] >= threshold)
    )

    table = pd.crosstab(features["high_previous_anger_24h"], features["is_negative_link"])
    table = table.reindex(index=[False, True], columns=[0, 1], fill_value=0)
    odds_ratio, fisher_p = stats.fisher_exact(table.to_numpy())
    chi2, chi2_p, _, _ = stats.chi2_contingency(table.to_numpy())

    summary = pd.DataFrame(
        {
            "group": ["no_high_previous_anger", "high_previous_anger"],
            "rows": [int(table.loc[False].sum()), int(table.loc[True].sum())],
            "negative_links": [int(table.loc[False, 1]), int(table.loc[True, 1])],
            "negative_rate": [
                table.loc[False, 1] / table.loc[False].sum(),
                table.loc[True, 1] / table.loc[True].sum(),
            ],
            "anger_threshold_q75": [threshold, threshold],
            "odds_ratio_high_vs_other": [odds_ratio, odds_ratio],
            "fisher_p_value": [fisher_p, fisher_p],
            "chi2": [chi2, chi2],
            "chi2_p_value": [chi2_p, chi2_p],
        }
    )
    summary.to_csv(OUTPUT_DIR / "hypothesis_1_emotional_escalation.csv", index=False)
    return summary


def run_hypothesis_2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in COGNITIVE_COMPLEXITY_FEATURES:
        comparison = compare_groups(
            df,
            group_col="LINK_SENTIMENT",
            value_col=feature,
            negative_value=-1,
            positive_value=1,
        )
        negative = pd.to_numeric(df.loc[df["LINK_SENTIMENT"] == -1, feature], errors="coerce").dropna()
        positive = pd.to_numeric(df.loc[df["LINK_SENTIMENT"] == 1, feature], errors="coerce").dropna()
        pooled_std = np.sqrt(((negative.var(ddof=1) + positive.var(ddof=1)) / 2))
        mann = stats.mannwhitneyu(negative, positive, alternative="two-sided")
        rows.append(
            {
                "feature": feature,
                **comparison,
                "cohens_d_negative_minus_positive": (
                    comparison["difference_negative_minus_positive"] / pooled_std if pooled_std else np.nan
                ),
                "point_biserial_negative": point_biserial(sentiment_to_binary_negative(df["LINK_SENTIMENT"]), df[feature]),
                "mannwhitney_u": mann.statistic,
                "mannwhitney_p_value": mann.pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "hypothesis_2_cognitive_complexity.csv", index=False)
    return result


def run_classic_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    df = add_pair_history_features(df)
    df = add_structural_features(df)
    df["combined_text"] = _combined_text(df)
    y = sentiment_to_binary_negative(df["LINK_SENTIMENT"])

    numeric_features = [col for col in LINGUISTIC_FEATURES + STRUCTURAL_FEATURES if col in df.columns]
    hf_features = ["Content_Sentiment", "Content_Score"]
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    text_selector = FunctionTransformer(lambda x: x["combined_text"], validate=False)

    models = {
        "Dummy most frequent": DummyClassifier(strategy="most_frequent"),
        "TF-IDF word 1-2 + Logistic Regression balanced": Pipeline(
            [
                ("text", text_selector),
                ("tfidf", TfidfVectorizer(max_features=50000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "TF-IDF word 1-2 + Linear SVC balanced": Pipeline(
            [
                ("text", text_selector),
                ("tfidf", TfidfVectorizer(max_features=50000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
                ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Properties + Logistic Regression balanced": Pipeline(
            [
                ("prep", ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric_features)])),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Properties + Random Forest balanced": Pipeline(
            [
                ("prep", ColumnTransformer([("num", SimpleImputer(), numeric_features)])),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "HF Content sentiment + Logistic Regression balanced": Pipeline(
            [
                ("prep", ColumnTransformer([("hf", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), hf_features)])),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "HF Content sentiment + Properties + Logistic Regression balanced": Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("hf", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), hf_features),
                            ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric_features),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "TF-IDF + Properties + Logistic Regression balanced": Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            (
                                "text",
                                TfidfVectorizer(max_features=30000, min_df=2, ngram_range=(1, 2), sublinear_tf=True),
                                "combined_text",
                            ),
                            ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric_features),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rows.append(_metrics(name, y_test, predictions))

    hf_direct = (X_test["Content_Sentiment"] == -1).astype(int).to_numpy()
    rows.append(_metrics("HF Content_Sentiment direct baseline", y_test, hf_direct))

    result = pd.DataFrame(rows).sort_values(["negative_f1", "macro_f1"], ascending=False)
    result.to_csv(OUTPUT_DIR / "classic_model_benchmark.csv", index=False)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    print("Dataset:", df.shape)
    print("Running hypothesis 1...")
    print(run_hypothesis_1(df).to_string(index=False))
    print("Running hypothesis 2...")
    print(run_hypothesis_2(df).to_string(index=False))
    print("Running classic benchmark...")
    print(run_classic_benchmark(df).to_string(index=False))


if __name__ == "__main__":
    main()
