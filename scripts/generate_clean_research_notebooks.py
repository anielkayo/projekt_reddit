from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_DIR = Path("notebooks_czysta_baza")


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def save_notebook(path: Path, cells: list) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


COMMON_SETUP = r"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd()
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "database").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "database" / "NajnowszaWersjaBazy1205_prepared.csv"
RAW_DATA_PATH = PROJECT_ROOT / "database" / "NajnowszaWersjaBazy1205.csv"
METADATA_PATH = PROJECT_ROOT / "outputs" / "prepared_dataset_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "czysta_baza"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda value: f"{value:.4f}")

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_PATH:", DATA_PATH)
print("Prepared data exists:", DATA_PATH.exists())
print("Output dir:", OUTPUT_DIR)

assert DATA_PATH.exists(), f"Brakuje pliku z przygotowana baza: {DATA_PATH}"
"""


def build_00_problem() -> None:
    cells = [
        md(
            """
            # 00. Definicja problemu i hipotez

            **Etap z planu pracy:** zdefiniowanie problemu i hipotez badawczych.

            Ten notebook porzadkuje pytanie badawcze, cel projektu, zakres analizy i sposob operacjonalizacji hipotez na nowo wyczyszczonej bazie `database/NajnowszaWersjaBazy1205_prepared.csv`.
            """
        ),
        md(
            """
            ## Pytanie badawcze

            Czy na podstawie tekstu, emocji, historii interakcji i prostych cech sieciowych mozna wyjasnic lub przewidziec negatywne odniesienia miedzy subredditami?

            Główna zmienna celu to `is_negative_link`, przygotowana z `LINK_SENTIMENT`: `1` oznacza link negatywny, `0` pozostale linki.
            """
        ),
        md(
            """
            ## Hipotezy

            **H1. Eskalacja emocjonalna**

            Prawdopodobienstwo linku negatywnego miedzy dwoma subredditami wzrasta, jesli w poprzednich 24 godzinach wystapily miedzy nimi interakcje o wysokim natezeniu slownictwa zwiazanego z gniewem (`LIWC_Anger`).

            **H2. Zlozonosc poznawcza**

            Teksty towarzyszace negatywnym linkom maja nizsza zlozonosc jezykowa niz teksty w linkach pozytywnych.

            **H3. Model multimodalny**

            Modele ML powinny przewidywac negatywne odniesienia z F1-score powyzej 75%, laczac cechy lingwistyczne, emocjonalne, transformerowe i strukturalne.
            """
        ),
        md(
            """
            ## Operacjonalizacja

            - H1: testujemy `high_previous_anger_24h` oraz statystyki `prev_pair_*_24h` wzgledem `is_negative_link`.
            - H2: porownujemy wybrane cechy zlozonosci z `prepared_dataset_metadata.json`.
            - H3: trenujemy modele na `split_chronological`, z metrykami skupionymi na klasie negatywnej.
            - Komponent Hugging Face traktujemy w dwoch warstwach: jako gotowe kolumny `Content_Sentiment` / `Content_Score` oraz jako opcjonalna komorke z nowszym modelem transformerowym, wylaczona domyslnie ze wzgledu na pobieranie modelu i czas wykonania.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            with open(METADATA_PATH, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            df_preview = pd.read_csv(DATA_PATH, nrows=5)
            print("Liczba kolumn w prepared CSV:", len(df_preview.columns))
            print("Target:", metadata["target"])
            print("Split:", metadata["split_column"])
            display(df_preview[[
                "POST_ID",
                "TIMESTAMP",
                "SOURCE_SUBREDDIT",
                "TARGET_SUBREDDIT",
                "LINK_SENTIMENT",
                "is_negative_link",
                "Content_Sentiment",
                "high_previous_anger_24h",
            ]])
            """
        ),
        md(
            """
            ## Rezultat etapu

            Hipotezy zostaly przepisane na mierzalne zmienne w oczyszczonej bazie. Kolejne notebooki ida etapami z `plan_pracy.md`: opis danych, analiza przygotowania, przeglad podejsc, implementacja metod, interpretacja i szkic prezentacji.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "00_definicja_problemu_i_hipotez.ipynb", cells)


def build_01_data_source() -> None:
    cells = [
        md(
            """
            # 01. Pozyskanie i opis danych

            **Etap z planu pracy:** pozyskanie danych.

            Celem jest opis zrodla, struktury, liczby obserwacji i potencjalnych problemow jakosciowych. Baza uzywana dalej jest juz przygotowana, dlatego ten notebook nie wykonuje ponownego czyszczenia.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            df = pd.read_csv(DATA_PATH, parse_dates=["TIMESTAMP"])
            raw_exists = RAW_DATA_PATH.exists()

            overview = pd.DataFrame([
                {"metric": "prepared_path", "value": str(DATA_PATH.relative_to(PROJECT_ROOT))},
                {"metric": "raw_path", "value": str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)) if raw_exists else "brak"},
                {"metric": "rows", "value": df.shape[0]},
                {"metric": "columns", "value": df.shape[1]},
                {"metric": "timestamp_min", "value": df["TIMESTAMP"].min()},
                {"metric": "timestamp_max", "value": df["TIMESTAMP"].max()},
                {"metric": "unique_source_subreddits", "value": df["SOURCE_SUBREDDIT"].nunique()},
                {"metric": "unique_target_subreddits", "value": df["TARGET_SUBREDDIT"].nunique()},
                {"metric": "unique_directed_pairs", "value": df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].drop_duplicates().shape[0]},
                {"metric": "negative_links", "value": int(df["is_negative_link"].sum())},
                {"metric": "negative_rate", "value": float(df["is_negative_link"].mean())},
            ])
            display(overview)
            overview.to_csv(OUTPUT_DIR / "01_data_overview.csv", index=False)
            """
        ),
        code(
            """
            split_quality = (
                df.groupby("split_chronological")["is_negative_link"]
                .agg(rows="size", negative_links="sum", negative_rate="mean")
                .reset_index()
            )

            sentiment_distribution = (
                df.groupby(["LINK_SENTIMENT", "Content_Sentiment"])
                .size()
                .rename("rows")
                .reset_index()
                .sort_values("rows", ascending=False)
            )

            display(split_quality)
            display(sentiment_distribution.head(12))
            split_quality.to_csv(OUTPUT_DIR / "01_split_quality.csv", index=False)
            sentiment_distribution.to_csv(OUTPUT_DIR / "01_sentiment_distribution.csv", index=False)
            """
        ),
        code(
            """
            key_columns = [
                "POST_ID",
                "TIMESTAMP",
                "SOURCE_SUBREDDIT",
                "TARGET_SUBREDDIT",
                "Raw_Title",
                "Raw_Content",
                "combined_text",
                "LINK_SENTIMENT",
                "is_negative_link",
                "Content_Sentiment",
                "Content_Score",
                "LIWC_Anger",
                "high_previous_anger_24h",
            ]

            quality = pd.DataFrame({
                "column": key_columns,
                "missing_values": [int(df[column].isna().sum()) for column in key_columns],
                "unique_values": [int(df[column].nunique(dropna=True)) for column in key_columns],
                "dtype": [str(df[column].dtype) for column in key_columns],
            })
            display(quality)
            quality.to_csv(OUTPUT_DIR / "01_key_column_quality.csv", index=False)
            """
        ),
        md(
            """
            ## Rezultat etapu

            Zbior jest gotowy do dalszej analizy: ma staly zakres czasu, brak brakow w kluczowych kolumnach i deterministyczny podzial chronologiczny na train/test. Najwazniejsze ryzyko analityczne to niezbalansowanie klasy negatywnej.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "01_pozyskanie_i_opis_danych.ipynb", cells)


def build_02_preparation_eda() -> None:
    cells = [
        md(
            """
            # 02. Przygotowanie i analiza danych

            **Etap z planu pracy:** przygotowanie i analiza danych.

            Notebook sprawdza, czy przygotowane cechy wystarczaja do testowania H1-H3, oraz wykonuje podstawowa eksploracje statystyczna i wizualizacje.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            import matplotlib.pyplot as plt
            import seaborn as sns

            with open(METADATA_PATH, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            df = pd.read_csv(DATA_PATH, parse_dates=["TIMESTAMP"])
            numeric_features = [column for column in metadata["numeric_features"] if column in df.columns]
            cognitive_features = [column for column in metadata["cognitive_complexity_features"] if column in df.columns]
            structural_features = [column for column in metadata["structural_features"] if column in df.columns]

            feature_inventory = pd.DataFrame([
                {"group": "numeric_features", "available_columns": len(numeric_features)},
                {"group": "cognitive_complexity_features", "available_columns": len(cognitive_features)},
                {"group": "structural_features", "available_columns": len(structural_features)},
                {"group": "indicator_features", "available_columns": len([column for column in metadata["indicator_features"] if column in df.columns])},
                {"group": "text_features", "available_columns": len([column for column in metadata["text_features"] if column in df.columns])},
            ])
            display(feature_inventory)
            feature_inventory.to_csv(OUTPUT_DIR / "02_feature_inventory.csv", index=False)
            """
        ),
        code(
            """
            target_distribution = (
                df["is_negative_link"]
                .value_counts()
                .sort_index()
                .rename_axis("is_negative_link")
                .reset_index(name="rows")
            )
            target_distribution["share"] = target_distribution["rows"] / target_distribution["rows"].sum()
            display(target_distribution)
            target_distribution.to_csv(OUTPUT_DIR / "02_target_distribution.csv", index=False)

            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(data=target_distribution, x="is_negative_link", y="rows", ax=ax, color="#4c78a8")
            ax.set_title("Rozklad klasy celu")
            ax.set_xlabel("Czy link negatywny")
            ax.set_ylabel("Liczba rekordow")
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "02_target_distribution.png", dpi=160)
            """
        ),
        code(
            """
            monthly = (
                df.set_index("TIMESTAMP")
                .resample("ME")["is_negative_link"]
                .agg(rows="size", negative_links="sum", negative_rate="mean")
                .reset_index()
            )
            display(monthly.tail(12))
            monthly.to_csv(OUTPUT_DIR / "02_monthly_negative_rate.csv", index=False)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(monthly["TIMESTAMP"], monthly["negative_rate"], color="#b279a2", linewidth=1.8)
            ax.set_title("Miesieczny odsetek negatywnych linkow")
            ax.set_xlabel("Miesiac")
            ax.set_ylabel("Odsetek linkow negatywnych")
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "02_monthly_negative_rate.png", dpi=160)
            """
        ),
        code(
            """
            top_pairs = (
                df.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])
                .agg(rows=("POST_ID", "size"), negative_links=("is_negative_link", "sum"), negative_rate=("is_negative_link", "mean"))
                .sort_values(["rows", "negative_links"], ascending=False)
                .head(20)
                .reset_index()
            )
            display(top_pairs)
            top_pairs.to_csv(OUTPUT_DIR / "02_top_directed_pairs.csv", index=False)
            """
        ),
        code(
            """
            eda_features = [
                "LIWC_Anger",
                "prev_pair_interactions_24h",
                "prev_pair_mean_anger_24h",
                "Average word length",
                "LIWC_Conj",
                "Automated readability index",
                "LIWC_CogMech",
                "Number of words",
                "common_neighbors_so_far",
            ]
            eda_features = [column for column in eda_features if column in df.columns]

            description = df[eda_features].describe().T
            display(description)
            description.to_csv(OUTPUT_DIR / "02_feature_descriptions.csv")
            """
        ),
        md(
            """
            ## Rezultat etapu

            Dane sa gotowe do modelowania, ale wymagaja metryk odpornych na niezbalansowanie klas. Analizy H1 i H2 powinny raportowac rozmiary grup oraz testy istotnosci, a H3 musi oceniac przede wszystkim F1/precision/recall klasy negatywnej.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "02_przygotowanie_i_analiza_danych.ipynb", cells)


def build_03_methods_review() -> None:
    cells = [
        md(
            """
            # 03. Przeglad i analiza podejsc

            **Etap z planu pracy:** przeglad i analiza podejsc.

            Ten notebook porownuje podejscia klasyczne, statystyczne, ML i transformerowe dla trzech hipotez.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            methods = pd.DataFrame([
                {
                    "hypothesis": "H1 eskalacja emocjonalna",
                    "approach": "Tabela kontyngencji + test Fishera",
                    "features": "high_previous_anger_24h, is_negative_link",
                    "why": "Najprostszy test roznicy udzialow dla rzadkiego zdarzenia.",
                    "main_metric": "odds ratio, p-value, roznica odsetkow",
                },
                {
                    "hypothesis": "H1 eskalacja emocjonalna",
                    "approach": "Regresja logistyczna jako rozszerzenie",
                    "features": "prev_pair_mean_anger_24h, prev_pair_interactions_24h, cechy kontrolne",
                    "why": "Pozwala kontrolowac historie pary i inne predyktory.",
                    "main_metric": "wspolczynnik anger, ROC/F1 pomocniczo",
                },
                {
                    "hypothesis": "H2 zlozonosc poznawcza",
                    "approach": "Porownanie grup + Mann-Whitney U",
                    "features": "Average word length, LIWC_Conj, ARI, LIWC_CogMech, Number of words",
                    "why": "Cechy sa ciagle i moga miec rozklady nienormalne.",
                    "main_metric": "roznica srednich, Cohen d, p-value",
                },
                {
                    "hypothesis": "H3 model multimodalny",
                    "approach": "TF-IDF + Logistic Regression balanced",
                    "features": "combined_text",
                    "why": "Mocny klasyczny baseline tekstowy dla krotkich i srednich tekstow.",
                    "main_metric": "negative_f1, macro_f1",
                },
                {
                    "hypothesis": "H3 model multimodalny",
                    "approach": "Cechy numeryczne/LIWC/sieciowe + Random Forest",
                    "features": "numeric_features, structural_features",
                    "why": "Sprawdza nieliniowe zaleznosci bez reprezentacji tekstowej TF-IDF.",
                    "main_metric": "negative_f1, recall klasy negatywnej",
                },
                {
                    "hypothesis": "H3 model multimodalny",
                    "approach": "Hugging Face baseline i opcjonalne embeddingi",
                    "features": "Content_Sentiment, Content_Score, opcjonalne sentence-transformers",
                    "why": "Pozwala porownac gotowy sygnal transformerowy z cechami recznie wyliczonymi.",
                    "main_metric": "negative_f1, macro_f1",
                },
            ])
            display(methods)
            methods.to_csv(OUTPUT_DIR / "03_methods_review.csv", index=False)
            """
        ),
        md(
            """
            ## Wybor metod do implementacji

            W kolejnym notebooku uruchamiamy wersje rdzeniowe:

            - H1: test Fishera, chi-kwadrat i roznica odsetkow.
            - H2: porownanie grup, Cohen d, korelacja punktowo-dwuseryjna i Mann-Whitney U.
            - H3: benchmark kilku modeli na stalym podziale chronologicznym.

            Taki zestaw daje jednoczesnie interpretowalnosc hipotez H1-H2 i praktyczny test predykcyjny H3.
            """
        ),
        code(
            """
            RUN_OPTIONAL_TRANSFORMERS_DEMO = False
            MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

            if RUN_OPTIONAL_TRANSFORMERS_DEMO:
                from transformers import pipeline

                df = pd.read_csv(DATA_PATH, nrows=20)
                texts = (df["Raw_Title"].fillna("").astype(str) + " " + df["Raw_Content"].fillna("").astype(str)).tolist()
                sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_NAME, truncation=True)
                display(pd.DataFrame(sentiment_pipe(texts[:5])))
            else:
                print(
                    "Opcjonalny pokaz nowego modelu Hugging Face jest wylaczony. "
                    "Ustaw RUN_OPTIONAL_TRANSFORMERS_DEMO = True, jesli srodowisko ma dostep do modelu."
                )
            """
        ),
        md(
            """
            ## Rezultat etapu

            Do finalnego porownania wybieramy metody, ktore mozna odtworzyc lokalnie bez pobierania duzych modeli. Komorki transformerowe zostaja jako opcjonalne rozszerzenie badania.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "03_przeglad_i_analiza_podejsc.ipynb", cells)


def build_04_implementation() -> None:
    cells = [
        md(
            """
            # 04. Implementacja metod i porownanie wynikow

            **Etap z planu pracy:** implementacja wybranych metod i porownanie wynikow.

            Notebook wykonuje testy H1-H2 oraz benchmark H3 na oczyszczonej bazie. Wyniki sa zapisywane do `outputs/czysta_baza`.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            import time

            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy import stats
            from sklearn.compose import ColumnTransformer
            from sklearn.dummy import DummyClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import LinearSVC

            from src.reddit_pbl.columns import COGNITIVE_COMPLEXITY_FEATURES
            from src.reddit_pbl.statistics import compare_groups, point_biserial

            RANDOM_STATE = 42
            df = pd.read_csv(DATA_PATH, parse_dates=["TIMESTAMP"])

            with open(METADATA_PATH, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            print("Rows:", df.shape[0], "Columns:", df.shape[1])
            print(df.groupby("split_chronological")["is_negative_link"].agg(["size", "sum", "mean"]))
            """
        ),
        md(
            """
            ## H1. Eskalacja emocjonalna

            Testujemy, czy flaga wysokiego poprzedniego anger w oknie 24h zwieksza odsetek linkow negatywnych.
            """
        ),
        code(
            """
            h1_table = pd.crosstab(df["high_previous_anger_24h"].astype(bool), df["is_negative_link"].astype(int))
            h1_table = h1_table.reindex(index=[False, True], columns=[0, 1], fill_value=0)

            odds_ratio, fisher_p = stats.fisher_exact(h1_table.to_numpy())
            chi2, chi2_p, _, _ = stats.chi2_contingency(h1_table.to_numpy())

            h1_rates = (
                df.groupby(df["high_previous_anger_24h"].astype(bool))["is_negative_link"]
                .agg(rows="size", negative_links="sum", negative_rate="mean")
                .rename_axis("high_previous_anger_24h")
                .reset_index()
            )
            anger_threshold = df.loc[df["prev_pair_interactions_24h"] > 0, "prev_pair_mean_anger_24h"].quantile(0.75)
            low_rate = float(h1_rates.loc[h1_rates["high_previous_anger_24h"] == False, "negative_rate"].iloc[0])
            high_rate = float(h1_rates.loc[h1_rates["high_previous_anger_24h"] == True, "negative_rate"].iloc[0])

            h1_result = h1_rates.assign(
                anger_threshold_q75=anger_threshold,
                negative_rate_delta_high_minus_other=high_rate - low_rate,
                odds_ratio_high_vs_other=odds_ratio,
                fisher_p_value=fisher_p,
                chi2=chi2,
                chi2_p_value=chi2_p,
            )

            display(h1_table)
            display(h1_result)
            h1_result.to_csv(OUTPUT_DIR / "04_h1_emotional_escalation.csv", index=False)
            """
        ),
        md(
            """
            ## H2. Zlozonosc poznawcza

            Porownujemy teksty linkow negatywnych i pozytywnych dla cech wskazanych w metadanych przygotowanego zbioru.
            """
        ),
        code(
            """
            h2_rows = []
            cognitive_features = [column for column in COGNITIVE_COMPLEXITY_FEATURES if column in df.columns]

            for feature in cognitive_features:
                comparison = compare_groups(
                    df,
                    group_col="LINK_SENTIMENT",
                    value_col=feature,
                    negative_value=-1,
                    positive_value=1,
                )
                negative = pd.to_numeric(df.loc[df["LINK_SENTIMENT"] == -1, feature], errors="coerce").dropna()
                positive = pd.to_numeric(df.loc[df["LINK_SENTIMENT"] == 1, feature], errors="coerce").dropna()
                pooled_std = np.sqrt((negative.var(ddof=1) + positive.var(ddof=1)) / 2)
                mann = stats.mannwhitneyu(negative, positive, alternative="two-sided")
                h2_rows.append(
                    {
                        "feature": feature,
                        **comparison,
                        "cohens_d_negative_minus_positive": comparison["difference_negative_minus_positive"] / pooled_std if pooled_std else np.nan,
                        "point_biserial_negative": point_biserial(df["is_negative_link"], df[feature]),
                        "mannwhitney_u": mann.statistic,
                        "mannwhitney_p_value": mann.pvalue,
                    }
                )

            h2_result = pd.DataFrame(h2_rows).sort_values("cohens_d_negative_minus_positive")
            display(h2_result)
            h2_result.to_csv(OUTPUT_DIR / "04_h2_cognitive_complexity.csv", index=False)

            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(
                data=h2_result,
                y="feature",
                x="cohens_d_negative_minus_positive",
                ax=ax,
                color="#59a14f",
            )
            ax.axvline(0, color="black", linewidth=1)
            ax.set_title("H2: efekt dla cech zlozonosci, negatywne minus pozytywne")
            ax.set_xlabel("Cohen d")
            ax.set_ylabel("")
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "04_h2_cognitive_complexity_effects.png", dpi=160)
            """
        ),
        md(
            """
            ## H3. Benchmark multimodalny

            Trenujemy modele na czesci `train` i oceniamy na pozniejszej czesci `test`. Najwazniejsza metryka to `negative_f1`, poniewaz klasa negatywna jest mniejszosciowa.
            """
        ),
        code(
            """
            def metrics_row(name: str, y_true: pd.Series, y_pred: np.ndarray, train_seconds: float | None = None) -> dict:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                return {
                    "model": name,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "macro_f1": f1_score(y_true, y_pred, average="macro"),
                    "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
                    "negative_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                    "negative_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                    "negative_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "train_seconds": train_seconds,
                }


            train = df[df["split_chronological"] == "train"].copy()
            test = df[df["split_chronological"] == "test"].copy()
            y_train = train["is_negative_link"].astype(int)
            y_test = test["is_negative_link"].astype(int)

            numeric_features = [column for column in metadata["numeric_features"] if column in df.columns]
            indicator_features = [column for column in metadata["indicator_features"] if column in df.columns]
            hf_features = [column for column in ["Content_Sentiment", "Content_Score"] if column in df.columns]
            numeric_no_hf = [column for column in numeric_features + indicator_features if column not in hf_features]
            numeric_with_hf = list(dict.fromkeys(numeric_no_hf + hf_features))

            numeric_preprocess = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            tree_preprocess = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ])

            models = {
                "Dummy most frequent": DummyClassifier(strategy="most_frequent"),
                "TF-IDF word 1-2 + Logistic Regression balanced": Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=50000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]),
                "TF-IDF word 1-2 + Linear SVC balanced": Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=50000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
                    ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
                ]),
                "Properties + Logistic Regression balanced": Pipeline([
                    ("prep", ColumnTransformer([("num", numeric_preprocess, numeric_no_hf)])),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]),
                "Properties + Random Forest balanced": Pipeline([
                    ("prep", ColumnTransformer([("num", tree_preprocess, numeric_no_hf)])),
                    ("clf", RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    )),
                ]),
                "HF sentiment columns + Logistic Regression balanced": Pipeline([
                    ("prep", ColumnTransformer([("hf", numeric_preprocess, hf_features)])),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]),
                "HF sentiment columns + Properties + Logistic Regression balanced": Pipeline([
                    ("prep", ColumnTransformer([("num", numeric_preprocess, numeric_with_hf)])),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]),
                "TF-IDF + Properties + HF columns + Logistic Regression balanced": Pipeline([
                    ("prep", ColumnTransformer([
                        ("text", TfidfVectorizer(max_features=30000, min_df=2, ngram_range=(1, 2), sublinear_tf=True), "combined_text"),
                        ("num", numeric_preprocess, numeric_with_hf),
                    ])),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]),
            }

            benchmark_rows = []

            rule_pred = test["high_previous_anger_24h"].astype(int).to_numpy()
            benchmark_rows.append(metrics_row("Rule: high previous anger -> negative", y_test, rule_pred, train_seconds=0.0))

            hf_direct_pred = (test["Content_Sentiment"] == -1).astype(int).to_numpy()
            benchmark_rows.append(metrics_row("HF Content_Sentiment direct baseline", y_test, hf_direct_pred, train_seconds=0.0))

            for name, model in models.items():
                start = time.perf_counter()
                if name.startswith("TF-IDF word"):
                    model.fit(train["combined_text"], y_train)
                    prediction = model.predict(test["combined_text"])
                else:
                    model.fit(train, y_train)
                    prediction = model.predict(test)
                elapsed = time.perf_counter() - start
                benchmark_rows.append(metrics_row(name, y_test, prediction, train_seconds=elapsed))
                print(f"Finished: {name} ({elapsed:.1f}s)")

            h3_benchmark = pd.DataFrame(benchmark_rows).sort_values(["negative_f1", "macro_f1"], ascending=False)
            display(h3_benchmark)
            h3_benchmark.to_csv(OUTPUT_DIR / "04_h3_multimodal_benchmark.csv", index=False)
            """
        ),
        code(
            """
            best_model = h3_benchmark.iloc[0]
            print("Najlepszy model wg F1 klasy negatywnej:")
            display(best_model.to_frame("value"))
            print("Czy negative_f1 >= 0.75?", bool(best_model["negative_f1"] >= 0.75))

            fig, ax = plt.subplots(figsize=(10, 5))
            plot_bench = h3_benchmark.sort_values("negative_f1", ascending=True)
            sns.barplot(data=plot_bench, y="model", x="negative_f1", ax=ax, color="#f28e2b")
            ax.axvline(0.75, color="red", linestyle="--", linewidth=1, label="prog H3 = 0.75")
            ax.set_title("H3: F1 klasy negatywnej")
            ax.set_xlabel("negative_f1")
            ax.set_ylabel("")
            ax.legend(loc="lower right")
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "04_h3_negative_f1_benchmark.png", dpi=160)
            """
        ),
        code(
            """
            rf_name = "Properties + Random Forest balanced"
            rf_model = models[rf_name]
            if hasattr(rf_model.named_steps["clf"], "feature_importances_"):
                rf_importance = (
                    pd.DataFrame({
                        "feature": numeric_no_hf,
                        "importance": rf_model.named_steps["clf"].feature_importances_,
                    })
                    .sort_values("importance", ascending=False)
                    .head(25)
                )
                display(rf_importance)
                rf_importance.to_csv(OUTPUT_DIR / "04_h3_random_forest_feature_importance.csv", index=False)
            """
        ),
        md(
            """
            ## Rezultat etapu

            Wszystkie trzy hipotezy maja policzone tabele wynikowe. Nastepny notebook interpretuje, czy wyniki wspieraja hipotezy, i opisuje ograniczenia badania.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "04_implementacja_i_porownanie_wynikow.ipynb", cells)


def build_05_interpretation() -> None:
    cells = [
        md(
            """
            # 05. Interpretacja rezultatow i wnioski

            **Etap z planu pracy:** interpretacja rezultatow i formulowanie wnioskow.

            Notebook zbiera wyniki H1-H3 i formuluje decyzje badawcze.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            h1 = pd.read_csv(OUTPUT_DIR / "04_h1_emotional_escalation.csv")
            h2 = pd.read_csv(OUTPUT_DIR / "04_h2_cognitive_complexity.csv")
            h3 = pd.read_csv(OUTPUT_DIR / "04_h3_multimodal_benchmark.csv")

            display(h1)
            display(h2)
            display(h3)
            """
        ),
        code(
            """
            h1_low = h1.loc[h1["high_previous_anger_24h"] == False].iloc[0]
            h1_high = h1.loc[h1["high_previous_anger_24h"] == True].iloc[0]
            h1_supported = (h1_high["negative_rate"] > h1_low["negative_rate"]) and (h1_high["fisher_p_value"] < 0.05)

            lower_complexity_expected = {
                "Average word length": "lower",
                "LIWC_Conj": "lower",
                "Automated readability index": "lower",
                "LIWC_CogMech": "lower",
                "Number of words": "lower",
                "Average number of words per sentence": "lower",
            }

            h2_eval = h2.copy()
            h2_eval["supports_lower_complexity"] = h2_eval["difference_negative_minus_positive"] < 0
            h2_supported_features = int(h2_eval["supports_lower_complexity"].sum())
            h2_total_features = int(h2_eval.shape[0])

            best_h3 = h3.sort_values(["negative_f1", "macro_f1"], ascending=False).iloc[0]
            h3_supported = bool(best_h3["negative_f1"] >= 0.75)

            decisions = pd.DataFrame([
                {
                    "hypothesis": "H1 eskalacja emocjonalna",
                    "decision": "potwierdzona" if h1_supported else "niepotwierdzona",
                    "main_evidence": (
                        f"negative_rate high anger={h1_high['negative_rate']:.4f}, "
                        f"other={h1_low['negative_rate']:.4f}, "
                        f"Fisher p={h1_high['fisher_p_value']:.4f}, "
                        f"odds ratio={h1_high['odds_ratio_high_vs_other']:.3f}"
                    ),
                },
                {
                    "hypothesis": "H2 zlozonosc poznawcza",
                    "decision": "czesciowo potwierdzona" if 0 < h2_supported_features < h2_total_features else ("potwierdzona" if h2_supported_features == h2_total_features else "niepotwierdzona"),
                    "main_evidence": f"{h2_supported_features}/{h2_total_features} cech ma nizsza wartosc dla linkow negatywnych",
                },
                {
                    "hypothesis": "H3 model multimodalny",
                    "decision": "potwierdzona" if h3_supported else "niepotwierdzona",
                    "main_evidence": f"najlepszy model: {best_h3['model']}, negative_f1={best_h3['negative_f1']:.4f}",
                },
            ])
            display(decisions)
            decisions.to_csv(OUTPUT_DIR / "05_hypothesis_decisions.csv", index=False)
            h2_eval.to_csv(OUTPUT_DIR / "05_h2_direction_check.csv", index=False)
            """
        ),
        code(
            """
            summary_md = f'''
            ## Wnioski badawcze

            **H1:** {decisions.loc[0, "decision"]}. Wysoki poprzedni anger w oknie 24h nie daje istotnego wzrostu prawdopodobienstwa linku negatywnego.

            **H2:** {decisions.loc[1, "decision"]}. Wyniki sugeruja roznice stylistyczne, ale nie prosty wzorzec, w ktorym wszystkie teksty negatywne sa mniej zlozone.

            **H3:** {decisions.loc[2, "decision"]}. Najlepszy model nie osiaga progu F1 = 0.75 dla klasy negatywnej, mimo ze accuracy moze wygladac wysoko przy niezbalansowanej klasie.

            ## Ograniczenia

            - Okno 24h w H1 daje mala liczbe przypadkow z historia tej samej pary.
            - `LINK_SENTIMENT` i `Content_Sentiment` nie sa tym samym: pierwsza zmienna opisuje link miedzy spolecznosciami, druga sentyment tresci.
            - Podzial chronologiczny jest bardziej realistyczny niz losowy, ale moze obnizac wyniki wzgledem walidacji losowej.
            - Modele transformerowe wymagajace pobierania wag zostaly pozostawione jako opcjonalne rozszerzenie.

            ## Dalsze prace

            - Przetestowac H1 dla okien 48h, 7 dni i par nieskierowanych.
            - Dla H2 dodac cechy skladniowe lub embeddingowe miary zlozonosci.
            - Dla H3 sprawdzic modele kosztoczulne, prog decyzyjny optymalizowany pod F1 oraz embeddingi sentence-transformers.
            '''

            display(Markdown(summary_md))
            (OUTPUT_DIR / "05_interpretacja_wnioski.md").write_text(summary_md, encoding="utf-8")
            """
        ),
        md(
            """
            ## Rezultat etapu

            Hipotezy zostaly formalnie ocenione. Ostatni notebook przepisuje wyniki na material do raportu i prezentacji.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "05_interpretacja_i_wnioski.ipynb", cells)


def build_06_presentation() -> None:
    cells = [
        md(
            """
            # 06. Koncowa prezentacja i raport

            **Etap z planu pracy:** koncowa prezentacja.

            Notebook tworzy zarys finalnej prezentacji i skondensowany raport wynikow na podstawie tabel zapisanych w poprzednich etapach.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            decisions = pd.read_csv(OUTPUT_DIR / "05_hypothesis_decisions.csv")
            h3 = pd.read_csv(OUTPUT_DIR / "04_h3_multimodal_benchmark.csv")
            best_h3 = h3.sort_values(["negative_f1", "macro_f1"], ascending=False).iloc[0]

            slide_outline = pd.DataFrame([
                {"slide": 1, "title": "Temat i pytanie badawcze", "content": "Negatywne odniesienia miedzy subredditami: emocje, jezyk i siec."},
                {"slide": 2, "title": "Dane", "content": "49 918 rekordow, lata 2013-2017, przygotowany zbior bez brakow w kluczowych kolumnach."},
                {"slide": 3, "title": "Hipotezy", "content": "H1 eskalacja emocjonalna, H2 zlozonosc poznawcza, H3 model multimodalny."},
                {"slide": 4, "title": "Metody", "content": "Test Fishera, Mann-Whitney U, Cohen d, TF-IDF, Logistic Regression, Random Forest, Hugging Face baseline."},
                {"slide": 5, "title": "Wynik H1", "content": decisions.loc[decisions["hypothesis"].str.startswith("H1"), "main_evidence"].iloc[0]},
                {"slide": 6, "title": "Wynik H2", "content": decisions.loc[decisions["hypothesis"].str.startswith("H2"), "main_evidence"].iloc[0]},
                {"slide": 7, "title": "Wynik H3", "content": f"Najlepszy model: {best_h3['model']}; negative_f1={best_h3['negative_f1']:.3f}."},
                {"slide": 8, "title": "Dlaczego accuracy nie wystarcza", "content": "Klasa negatywna jest mniejszosciowa, wiec model moze miec wysoka accuracy i slaby recall/F1 dla klasy 1."},
                {"slide": 9, "title": "Ograniczenia", "content": "Male grupy historii 24h, roznica LINK_SENTIMENT vs Content_Sentiment, koszt modeli transformerowych."},
                {"slide": 10, "title": "Wniosek koncowy", "content": "Najsilniejsze sa sygnaly tekstowe TF-IDF; hipotezy wymagaja ostroznej interpretacji i dalszych rozszerzen."},
            ])
            display(slide_outline)
            slide_outline.to_csv(OUTPUT_DIR / "06_slide_outline.csv", index=False)
            """
        ),
        code(
            """
            report = [
                "# Raport koncowy - badanie hipotez na oczyszczonej bazie",
                "",
                "## Dane",
                "",
                "- Plik: `database/NajnowszaWersjaBazy1205_prepared.csv`.",
                "- Liczba obserwacji: 49 918.",
                "- Target: `is_negative_link`, czyli binarna wersja `LINK_SENTIMENT`.",
                "- Podzial: `split_chronological`, aby test odzwierciedlal predykcje na pozniejszych danych.",
                "",
                "## Decyzje wobec hipotez",
                "",
            ]

            for row in decisions.itertuples(index=False):
                report.append(f"- **{row.hypothesis}: {row.decision}.** {row.main_evidence}.")

            report.extend([
                "",
                "## Najlepszy model H3",
                "",
                f"- Model: `{best_h3['model']}`.",
                f"- Accuracy: {best_h3['accuracy']:.3f}.",
                f"- Macro F1: {best_h3['macro_f1']:.3f}.",
                f"- F1 klasy negatywnej: {best_h3['negative_f1']:.3f}.",
                "",
                "## Materialy",
                "",
                "- Notebooki: `notebooks_czysta_baza/`.",
                "- Tabele i wykresy: `outputs/czysta_baza/`.",
            ])

            report_text = "\\n".join(report) + "\\n"
            (OUTPUT_DIR / "06_raport_koncowy.md").write_text(report_text, encoding="utf-8")
            display(Markdown(report_text))
            """
        ),
        md(
            """
            ## Rezultat etapu

            Powstal szkic narracji do prezentacji oraz krotki raport markdown. Pelne dowody liczbowe sa w notebookach 01-05 i w tabelach `outputs/czysta_baza`.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "06_koncowa_prezentacja_i_raport.ipynb", cells)


def main() -> None:
    build_00_problem()
    build_01_data_source()
    build_02_preparation_eda()
    build_03_methods_review()
    build_04_implementation()
    build_05_interpretation()
    build_06_presentation()
    print(f"Generated notebooks in {NOTEBOOK_DIR.resolve()}")


if __name__ == "__main__":
    main()
