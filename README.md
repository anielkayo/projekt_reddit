# Analiza sentymentu linków między subredditami

Projekt bada hipotezy dotyczące negatywnych odniesień między społecznościami Reddita.
Główna baza wejściowa to `database/NajnowszaWersjaBazy1205.csv`. Plik jest traktowany
jako gotowy i wyczyszczony, więc notebooki nie wykonują ponownego czyszczenia ani nie
modyfikują bazy.

## Struktura repozytorium

- `PBL_dataset_analysis.ipynb` - oryginalny, roboczy notebook projektu.
- `database/NajnowszaWersjaBazy1205.csv` - gotowa baza wejściowa.
- `notebooks/00_definicja_problemu_i_hipotez.ipynb` - problem badawczy, pytania i hipotezy.
- `notebooks/01_pozyskanie_i_opis_danych.ipynb` - opis źródła, struktury i jakości danych.
- `notebooks/02_hipoteza_1_eskalacja_emocjonalna.ipynb` - test hipotezy eskalacji emocjonalnej.
- `notebooks/03_hipoteza_2_zlozonosc_poznawcza.ipynb` - test hipotezy złożoności poznawczej.
- `notebooks/04_hipoteza_3_benchmark_modeli.ipynb` - benchmark modeli klasycznych i Hugging Face.
- `notebooks/05_interpretacja_i_wnioski_koncowe.ipynb` - podsumowanie wyników i szkic prezentacji.
- `src/reddit_pbl/` - proste funkcje pomocnicze do cech czasowych, sieciowych i statystyk.
- `scripts/run_core_analyses.py` - uruchamia rdzeniowe analizy i zapisuje wyniki do `outputs/`.
- `scripts/generate_notebooks.py` - odtwarza notebooki z uporządkowanej definicji.
- `outputs/` - zapisane wyniki hipotez i benchmarku.
- `tests/` - testy funkcji pomocniczych.

## Hipotezy

1. **Eskalacja emocjonalna**: prawdopodobieństwo linku negatywnego rośnie, jeśli w poprzednich 24 godzinach ta sama para subredditów miała interakcje z wysokim `LIWC_Anger`.
2. **Złożoność poznawcza**: teksty przy linkach negatywnych mają niższą złożoność językową niż teksty przy linkach pozytywnych.
3. **Model multimodalny**: modele ML potrafią przewidzieć negatywne odniesienie z F1 powyżej 75%, łącząc cechy lingwistyczne, emocjonalne i strukturalne.

## Najważniejsze wyniki

### H1. Eskalacja emocjonalna

W podstawowej operacjonalizacji H1 nie została potwierdzona.
Wysoki wcześniejszy `LIWC_Anger` w oknie 24h wystąpił w 124 rekordach.
Odsetek linków negatywnych był bardzo podobny do pozostałych rekordów:

- bez wysokiego wcześniejszego anger: `7.72%`;
- z wysokim wcześniejszym anger: `8.06%`;
- Fisher p-value: `0.866`;
- odds ratio: `1.049`.

### H2. Złożoność poznawcza

Wyniki są mieszane. Negatywne linki mają niższy `Automated readability index`
i niższą średnią liczbę słów na zdanie, co częściowo wspiera hipotezę.
Jednocześnie mają więcej słów, więcej `LIWC_Conj` i wyższy `LIWC_CogMech`, więc
nie można przyjąć prostej tezy, że negatywne linki są zawsze językowo prostsze.

### H3. Benchmark modeli

Najważniejsza metryka to F1 dla klasy negatywnej, a nie accuracy, ponieważ klasa
negatywna jest mniejszościowa.

| Model | Accuracy | Macro F1 | F1 klasy negatywnej |
| --- | ---: | ---: | ---: |
| TF-IDF word 1-2 + Logistic Regression balanced | 0.849 | 0.647 | 0.380 |
| TF-IDF + Properties + Logistic Regression balanced | 0.839 | 0.641 | 0.374 |
| TF-IDF word 1-2 + Linear SVC balanced | 0.890 | 0.646 | 0.352 |
| HF Content sentiment + Properties + Logistic Regression balanced | 0.742 | 0.566 | 0.290 |
| Properties + Logistic Regression balanced | 0.743 | 0.564 | 0.285 |
| HF `Content_Sentiment` direct baseline | 0.823 | 0.574 | 0.248 |
| HF Content sentiment + Logistic Regression balanced | 0.699 | 0.522 | 0.231 |
| Properties + Random Forest balanced | 0.923 | 0.523 | 0.086 |
| Dummy most frequent | 0.923 | 0.480 | 0.000 |

H3 nie została potwierdzona dla progu F1 > 75%.
Najlepiej lokalnie wypadł klasyczny model TF-IDF + Logistic Regression balanced.
Random Forest ma wysoką accuracy, ale bardzo niski recall i F1 dla klasy negatywnej.

## Uruchamianie

Zainstaluj zależności:

```bash
python -m pip install -r requirements.txt
```

Uruchom testy:

```bash
python -m pytest tests -q
```

Odtwórz rdzeniowe wyniki:

```bash
python scripts/run_core_analyses.py
```

Odtwórz notebooki:

```bash
python scripts/generate_notebooks.py
```

Notebook `04_hipoteza_3_benchmark_modeli.ipynb` zawiera opcjonalną komórkę do
benchmarku embeddingów Hugging Face przez `sentence-transformers`. Najwygodniej
uruchomić ją w Google Colabie z GPU albo w lokalnym środowisku z poprawnie
działającym PyTorch.
