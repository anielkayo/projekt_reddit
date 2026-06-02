from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_DIR = Path("notebooks")


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
import sys

import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "database" / "NajnowszaWersjaBazy1205.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_PATH exists:", DATA_PATH.exists())
"""


def build_00_problem() -> None:
    cells = [
        md(
            """
            # 00. Zdefiniowanie problemu i hipotez

            Ten notebook odpowiada etapowi: zdefiniowanie problemu, pytania badawczego i hipotez.
            Nie zmienia bazy danych. Plik `database/NajnowszaWersjaBazy1205.csv` jest traktowany jako gotowy i wyczyszczony zbiór wejściowy.
            """
        ),
        md(
            """
            ## Problem badawczy

            Analizujemy, kiedy link lub odniesienie między dwoma subredditami ma negatywny sentyment.
            Najważniejsza etykieta modelowana w projekcie to `LINK_SENTIMENT`, gdzie `-1` oznacza link negatywny, a `1` link pozytywny.

            Dodatkowo baza zawiera `Content_Sentiment`, czyli sentyment treści policzony modelem transformerowym w poprzednim notebooku projektu.
            W benchmarku traktujemy go jako Hugging Face baseline, ale głównym celem predykcji pozostaje `LINK_SENTIMENT`.
            """
        ),
        md(
            """
            ## Hipotezy

            **H1. Eskalacja emocjonalna**

            Prawdopodobieństwo wystąpienia linku negatywnego między dwoma subredditami wzrasta, jeśli w poprzednich 24 godzinach wystąpiły między nimi interakcje o wysokim natężeniu słownictwa związanego z gniewem (`LIWC_Anger`).

            **H2. Złożoność poznawcza**

            Teksty towarzyszące negatywnym linkom mają niższą złożoność językową, na przykład krótsze słowa, mniej spójników logicznych i niższe wskaźniki czytelności, niż teksty w linkach pozytywnych.

            **H3. Model multimodalny**

            Modele uczenia maszynowego potrafią przewidzieć wystąpienie negatywnego odniesienia między dwiema społecznościami ze skutecznością F1 powyżej 75%, łącząc cechy lingwistyczne, emocjonalne i strukturalne sieci.
            """
        ),
        md(
            """
            ## Operacjonalizacja

            - H1: tworzymy cechy historii pary subredditów z poprzednich 24 godzin: liczba wcześniejszych interakcji, średni `LIWC_Anger`, maksymalny `LIWC_Anger`.
            - H2: porównujemy grupy `LINK_SENTIMENT = -1` i `LINK_SENTIMENT = 1` dla cech złożoności: średnia długość słowa, `LIWC_Conj`, `Automated readability index`, `LIWC_CogMech`, liczba słów.
            - H3: porównujemy klasyczne modele ML, cechy tekstowe TF-IDF, cechy lingwistyczne, cechy sieciowe i warianty Hugging Face.

            Przy ocenie modeli najważniejszy jest F1 dla klasy negatywnej oraz macro F1, ponieważ zbiór jest silnie niezbalansowany.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            preview = pd.read_csv(DATA_PATH, nrows=5)
            print("Kolumny:", len(preview.columns))
            display(preview[["POST_ID", "TIMESTAMP", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "LINK_SENTIMENT", "Content_Sentiment"]])
            """
        ),
        md(
            """
            ## Wniosek po etapie

            Zakres badania jest możliwy do wykonania na gotowej bazie. Nie wykonujemy ponownego czyszczenia danych, tylko wykorzystujemy istniejące kolumny tekstowe, LIWC, sentymentowe, czasowe i sieciowe.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "00_definicja_problemu_i_hipotez.ipynb", cells)


def build_01_data() -> None:
    cells = [
        md(
            """
            # 01. Pozyskanie i opis danych

            Ten notebook odpowiada etapowi: opis źródła danych, liczby obserwacji, struktury zbioru i potencjalnych problemów jakościowych.
            Nie zapisuje zmian do bazy wejściowej.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            df = pd.read_csv(DATA_PATH)
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

            print("Wymiary bazy:", df.shape)
            print("Zakres czasu:", df["TIMESTAMP"].min(), "->", df["TIMESTAMP"].max())
            print("Liczba unikalnych par SOURCE -> TARGET:", df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].drop_duplicates().shape[0])
            """
        ),
        code(
            """
            sentiment_counts = pd.DataFrame({
                "LINK_SENTIMENT": df["LINK_SENTIMENT"].value_counts().sort_index(),
                "Content_Sentiment": df["Content_Sentiment"].value_counts().sort_index(),
            })
            display(sentiment_counts)
            """
        ),
        code(
            """
            key_columns = [
                "POST_ID", "TIMESTAMP", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT",
                "Raw_Title", "Raw_Content", "LINK_SENTIMENT", "Content_Sentiment",
                "LIWC_Anger", "Average word length", "LIWC_Conj", "Automated readability index",
            ]
            missing = df[key_columns].isna().sum().to_frame("missing_values")
            display(missing)
            """
        ),
        code(
            """
            numeric_cols = [
                "LIWC_Anger", "Average word length", "LIWC_Conj",
                "Automated readability index", "LIWC_CogMech", "Number of words"
            ]
            display(df[numeric_cols].describe().T)
            """
        ),
        md(
            """
            ## Wniosek po etapie

            Baza ma gotową strukturę do modelowania: identyfikatory postów, czas, pary subredditów, tekst, sentyment linku oraz cechy LIWC i właściwości tekstu.
            Zgodnie z założeniem projektu nie wykonujemy czyszczenia danych, tylko sprawdzamy ich strukturę i kompletność w kolumnach używanych dalej.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "01_pozyskanie_i_opis_danych.ipynb", cells)


def build_02_h1() -> None:
    cells = [
        md(
            """
            # 02. Hipoteza 1: Eskalacja emocjonalna

            Sprawdzamy, czy link negatywny jest częstszy, gdy w poprzednich 24 godzinach ta sama para `SOURCE_SUBREDDIT -> TARGET_SUBREDDIT` miała wysokie natężenie `LIWC_Anger`.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            from scipy import stats
            from src.reddit_pbl.features import add_pair_history_features, sentiment_to_binary_negative
            """
        ),
        md(
            """
            ## Metoda

            1. Sortujemy rekordy po czasie.
            2. Dla każdej skierowanej pary subredditów liczymy wcześniejsze interakcje z ostatnich 24 godzin.
            3. Wysoki anger definiujemy jako średni wcześniejszy `LIWC_Anger` co najmniej na poziomie 75 percentyla wśród rekordów z jakąkolwiek wcześniejszą interakcją.
            4. Porównujemy odsetek negatywnych linków i liczymy test Fishera oraz iloraz szans.
            """
        ),
        code(
            """
            df = pd.read_csv(DATA_PATH)
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

            features = add_pair_history_features(df)
            features["is_negative_link"] = sentiment_to_binary_negative(features["LINK_SENTIMENT"])

            history_rows = features[features["prev_pair_interactions_24h"] > 0].copy()
            anger_threshold = history_rows["prev_pair_mean_anger_24h"].quantile(0.75)

            features["high_previous_anger_24h"] = (
                (features["prev_pair_interactions_24h"] > 0)
                & (features["prev_pair_mean_anger_24h"] >= anger_threshold)
            )

            table = pd.crosstab(features["high_previous_anger_24h"], features["is_negative_link"])
            table = table.reindex(index=[False, True], columns=[0, 1], fill_value=0)
            display(table)

            odds_ratio, fisher_p = stats.fisher_exact(table.to_numpy())
            chi2, chi2_p, _, _ = stats.chi2_contingency(table.to_numpy())
            print("Próg wysokiego anger:", anger_threshold)
            print("Odds ratio:", odds_ratio)
            print("Fisher p-value:", fisher_p)
            print("Chi2 p-value:", chi2_p)
            """
        ),
        code(
            """
            rates = (
                features.groupby("high_previous_anger_24h")["is_negative_link"]
                .agg(rows="size", negative_links="sum", negative_rate="mean")
                .reset_index()
            )
            display(rates)

            rates.to_csv(OUTPUT_DIR / "notebook_h1_rates.csv", index=False)
            """
        ),
        md(
            """
            ## Wniosek

            W uruchomionej analizie rdzeniowej wynik nie wspiera H1: grupa z wysokim wcześniejszym `LIWC_Anger` ma bardzo podobny odsetek linków negatywnych do pozostałych rekordów, a test Fishera nie wskazuje istotnej różnicy.

            Interpretacyjnie ważne jest też to, że tylko niewielka liczba rekordów ma historię tej samej pary w poprzednich 24 godzinach. Hipotezę warto powtórzyć także dla okien 48h i 7 dni albo dla par nieskierowanych.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "02_hipoteza_1_eskalacja_emocjonalna.ipynb", cells)


def build_03_h2() -> None:
    cells = [
        md(
            """
            # 03. Hipoteza 2: Złożoność poznawcza

            Sprawdzamy, czy teksty w linkach negatywnych mają niższą złożoność językową niż teksty w linkach pozytywnych.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            from scipy import stats
            from src.reddit_pbl.columns import COGNITIVE_COMPLEXITY_FEATURES
            from src.reddit_pbl.features import sentiment_to_binary_negative
            from src.reddit_pbl.statistics import compare_groups, point_biserial
            """
        ),
        code(
            """
            df = pd.read_csv(DATA_PATH)

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
                pooled_std = np.sqrt((negative.var(ddof=1) + positive.var(ddof=1)) / 2)
                mann = stats.mannwhitneyu(negative, positive, alternative="two-sided")
                rows.append({
                    "feature": feature,
                    **comparison,
                    "cohens_d_negative_minus_positive": comparison["difference_negative_minus_positive"] / pooled_std,
                    "point_biserial_negative": point_biserial(sentiment_to_binary_negative(df["LINK_SENTIMENT"]), df[feature]),
                    "mannwhitney_p_value": mann.pvalue,
                })

            results = pd.DataFrame(rows)
            display(results.sort_values("cohens_d_negative_minus_positive"))
            results.to_csv(OUTPUT_DIR / "notebook_h2_complexity.csv", index=False)
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt

            plot_data = results.sort_values("cohens_d_negative_minus_positive")
            ax = plot_data.plot.barh(
                x="feature",
                y="cohens_d_negative_minus_positive",
                figsize=(9, 4),
                legend=False,
                title="Efekt: negatywne minus pozytywne linki",
            )
            ax.axvline(0, color="black", linewidth=1)
            ax.set_xlabel("Cohen d")
            plt.tight_layout()
            """
        ),
        md(
            """
            ## Wniosek

            Wyniki są mieszane. Negatywne linki mają niższy `Automated readability index` i niższą średnią liczbę słów na zdanie, co częściowo pasuje do hipotezy niższej złożoności.
            Jednocześnie mają nieco więcej spójników (`LIWC_Conj`), więcej słów i wyższy `LIWC_CogMech`, więc hipotezy nie można przyjąć w całości.

            Najrozsądniejszy wniosek: negatywne linki różnią się stylem i długością tekstu, ale nie jest to prosta zależność "negatywny = zawsze prostszy poznawczo".
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "03_hipoteza_2_zlozonosc_poznawcza.ipynb", cells)


def build_04_h3() -> None:
    cells = [
        md(
            """
            # 04. Hipoteza 3: Benchmark modeli sentymentu

            Ten notebook porównuje klasyczne modele ML oraz podejścia Hugging Face w zadaniu przewidywania `LINK_SENTIMENT`.
            Najważniejsza metryka to F1 dla klasy negatywnej, ponieważ klasa `-1` jest mniejszościowa.
            """
        ),
        code(COMMON_SETUP),
        md(
            """
            ## Modele porównywane w analizie rdzeniowej

            - Dummy most frequent jako baseline nierównowagi klas.
            - TF-IDF word 1-2 + Logistic Regression z `class_weight="balanced"`.
            - TF-IDF word 1-2 + Linear SVC z wagami klas.
            - Właściwości tekstu, LIWC i cechy sieciowe + Logistic Regression.
            - Właściwości tekstu, LIWC i cechy sieciowe + Random Forest.
            - TF-IDF + właściwości + Logistic Regression.
            - Hugging Face direct baseline: istniejące `Content_Sentiment` jako predyktor negatywnego linku.
            - Hugging Face derived features: `Content_Sentiment` i `Content_Score` w Logistic Regression.

            Pełny kod benchmarku jest w `scripts/run_core_analyses.py`.
            """
        ),
        code(
            """
            benchmark_path = OUTPUT_DIR / "classic_model_benchmark.csv"
            if not benchmark_path.exists():
                import subprocess
                subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "run_core_analyses.py")], check=True)

            benchmark = pd.read_csv(benchmark_path)
            display(benchmark.sort_values(["negative_f1", "macro_f1"], ascending=False))
            """
        ),
        code(
            """
            best = benchmark.sort_values(["negative_f1", "macro_f1"], ascending=False).iloc[0]
            print("Najlepszy model według F1 klasy negatywnej:")
            print(best[["model", "accuracy", "macro_f1", "negative_precision", "negative_recall", "negative_f1"]])
            print()
            print("Czy F1 klasy negatywnej przekracza 75%?", bool(best["negative_f1"] >= 0.75))
            """
        ),
        md(
            """
            ## Opcjonalny benchmark embeddingów Hugging Face

            Poniższa komórka uruchamia model `sentence-transformers/all-MiniLM-L6-v2` albo inny model embeddingowy z Hugging Face.
            W lokalnym środowisku Windows może wymagać poprawnego PyTorch. W Colabie najprościej uruchomić ją z GPU.

            Domyślnie `RUN_HF_EMBEDDINGS = False`, żeby przypadkowo nie pobierać dużych modeli i nie generować długo embeddingów.
            Po ustawieniu na `True` wyniki zostaną dopisane do tabeli benchmarkowej.
            """
        ),
        code(
            """
            RUN_HF_EMBEDDINGS = False
            HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
            HF_SAMPLE_SIZE = 12000

            if RUN_HF_EMBEDDINGS:
                from sentence_transformers import SentenceTransformer
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
                from sklearn.model_selection import train_test_split
                from src.reddit_pbl.features import sentiment_to_binary_negative

                df = pd.read_csv(DATA_PATH)
                df["combined_text"] = df["Raw_Title"].fillna("").astype(str) + "\\n" + df["Raw_Content"].fillna("").astype(str)
                y = sentiment_to_binary_negative(df["LINK_SENTIMENT"])

                if HF_SAMPLE_SIZE is not None and HF_SAMPLE_SIZE < len(df):
                    sample = df.assign(y=y).sample(HF_SAMPLE_SIZE, random_state=42, stratify=y)
                    y = sample["y"]
                    texts = sample["combined_text"].tolist()
                else:
                    texts = df["combined_text"].tolist()

                X_train_text, X_test_text, y_train, y_test = train_test_split(
                    texts, y, test_size=0.2, random_state=42, stratify=y
                )

                model = SentenceTransformer(HF_MODEL_NAME)
                X_train = model.encode(X_train_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
                X_test = model.encode(X_test_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

                clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)

                hf_result = pd.DataFrame([{
                    "model": f"HF embeddings {HF_MODEL_NAME} + Logistic Regression balanced",
                    "accuracy": accuracy_score(y_test, pred),
                    "macro_f1": f1_score(y_test, pred, average="macro"),
                    "weighted_f1": f1_score(y_test, pred, average="weighted"),
                    "negative_precision": precision_score(y_test, pred, pos_label=1, zero_division=0),
                    "negative_recall": recall_score(y_test, pred, pos_label=1, zero_division=0),
                    "negative_f1": f1_score(y_test, pred, pos_label=1, zero_division=0),
                    "tn_fp_fn_tp": confusion_matrix(y_test, pred, labels=[0, 1]).ravel().tolist(),
                }])
                display(pd.concat([benchmark, hf_result], ignore_index=True).sort_values(["negative_f1", "macro_f1"], ascending=False))
            else:
                print("Pominięto embeddingowy benchmark HF. Ustaw RUN_HF_EMBEDDINGS = True, aby go uruchomić.")
            """
        ),
        md(
            """
            ## Wniosek

            W lokalnie wykonanym benchmarku hipoteza H3 w wersji F1 > 75% nie została potwierdzona.
            Najlepszy wynik dla klasy negatywnej osiągnął model TF-IDF + Logistic Regression balanced, z F1 klasy negatywnej około 0.38.

            Random Forest ma wysoką accuracy, ale słabo rozpoznaje klasę negatywną. To pokazuje, że przy niezbalansowanych danych nie wolno wybierać modelu tylko po accuracy.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "04_hipoteza_3_benchmark_modeli.ipynb", cells)


def build_05_conclusions() -> None:
    cells = [
        md(
            """
            # 05. Interpretacja wyników i końcowe wnioski

            Ten notebook zbiera wyniki z poprzednich etapów, formułuje decyzje wobec hipotez i daje krótką strukturę końcowej prezentacji.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            h1 = pd.read_csv(OUTPUT_DIR / "hypothesis_1_emotional_escalation.csv")
            h2 = pd.read_csv(OUTPUT_DIR / "hypothesis_2_cognitive_complexity.csv")
            bench = pd.read_csv(OUTPUT_DIR / "classic_model_benchmark.csv")

            display(h1)
            display(h2.sort_values("cohens_d_negative_minus_positive"))
            display(bench.sort_values(["negative_f1", "macro_f1"], ascending=False))
            """
        ),
        md(
            """
            ## Decyzje wobec hipotez

            **H1: niepotwierdzona w podstawowej operacjonalizacji.**
            Wysoki wcześniejszy `LIWC_Anger` w oknie 24h nie zwiększa wyraźnie prawdopodobieństwa negatywnego linku.

            **H2: częściowo potwierdzona, ale z zastrzeżeniami.**
            Część cech wskazuje na niższą czytelność lub krótsze zdania dla negatywnych linków, ale inne cechy pokazują większą długość tekstu i więcej markerów poznawczych.

            **H3: niepotwierdzona dla progu F1 > 75%.**
            Modele przewidują klasę większościową dobrze, ale negatywne linki są trudne do uchwycenia. Najlepszy lokalny wynik F1 dla klasy negatywnej jest daleko poniżej 0.75.
            """
        ),
        md(
            """
            ## Ograniczenia

            - `LINK_SENTIMENT` i `Content_Sentiment` mogą mierzyć różne zjawiska: sentyment linku społecznościowego kontra sentyment treści posta.
            - Okno 24h w H1 jest restrykcyjne i daje mało przypadków z historią tej samej pary.
            - Dane są niezbalansowane, więc macro F1 i F1 klasy negatywnej są ważniejsze niż accuracy.
            - Benchmark HF embeddingów może wymagać środowiska Colab/GPU i poprawnego PyTorch.
            """
        ),
        md(
            """
            ## Propozycja slajdów końcowych

            1. Temat i pytanie badawcze.
            2. Opis danych: 49 918 rekordów, lata 2013-2017, pary subredditów.
            3. Hipotezy H1, H2, H3.
            4. Metodologia: cechy LIWC, cechy historii par, TF-IDF, klasyczne ML, Hugging Face.
            5. Wyniki H1: brak wsparcia dla eskalacji 24h.
            6. Wyniki H2: efekt mieszany, negatywność nie oznacza prostej niższej złożoności.
            7. Wyniki H3: najlepszy model i porównanie metryk.
            8. Dlaczego accuracy myli przy niezbalansowanych danych.
            9. Ograniczenia i dalsze prace.
            10. Końcowy wniosek badawczy.
            """
        ),
    ]
    save_notebook(NOTEBOOK_DIR / "05_interpretacja_i_wnioski_koncowe.ipynb", cells)


def main() -> None:
    build_00_problem()
    build_01_data()
    build_02_h1()
    build_03_h2()
    build_04_h3()
    build_05_conclusions()
    print(f"Generated notebooks in {NOTEBOOK_DIR.resolve()}")


if __name__ == "__main__":
    main()
