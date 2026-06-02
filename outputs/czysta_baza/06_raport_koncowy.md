# Raport koncowy - badanie hipotez na oczyszczonej bazie

## Dane

- Plik: `database/NajnowszaWersjaBazy1205_prepared.csv`.
- Liczba obserwacji: 49 918.
- Target: `is_negative_link`, czyli binarna wersja `LINK_SENTIMENT`.
- Podzial: `split_chronological`, aby test odzwierciedlal predykcje na pozniejszych danych.

## Decyzje wobec hipotez

- **H1 eskalacja emocjonalna: niepotwierdzona.** negative_rate high anger=0.0726, other=0.0772, Fisher p=1.0000, odds ratio=0.935.
- **H2 zlozonosc poznawcza: czesciowo potwierdzona.** 2/6 cech ma nizsza wartosc dla linkow negatywnych.
- **H3 model multimodalny: niepotwierdzona.** najlepszy model: TF-IDF word 1-2 + Logistic Regression balanced, negative_f1=0.3437.

## Najlepszy model H3

- Model: `TF-IDF word 1-2 + Logistic Regression balanced`.
- Accuracy: 0.856.
- Macro F1: 0.631.
- F1 klasy negatywnej: 0.344.

## Materialy

- Notebooki: `notebooks_czysta_baza/`.
- Tabele i wykresy: `outputs/czysta_baza/`.
