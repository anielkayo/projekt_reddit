# Raport przygotowania danych

## Jakosc i gotowosc

- Liczba obserwacji: 49918
- Liczba kolumn po przygotowaniu: 106
- Braki danych lacznie: 0
- Zduplikowane POST_ID: 0
- Zakres czasu: 2013-12-31 16:39:58 - 2017-04-19 00:15:44
- Linki negatywne: 3854
- Odsetek linkow negatywnych: 0.0772
- Wiersze train: 39934
- Wiersze test: 9984

## Kolumny modelowe

- Target: `is_negative_link`
- Split: `split_chronological`
- Tekst: `combined_text`, `Raw_Title`, `Raw_Content`
- Liczba cech numerycznych: 94
- Liczba cech wskaznikowych: 1
- Liczba cech strukturalnych: 7

## Dopasowanie do hipotez

- H1: uzyj `high_previous_anger_24h`, `prev_pair_mean_anger_24h` i targetu `is_negative_link`.
- H2: uzyj listy `cognitive_complexity_features` oraz porownania wzgledem `is_negative_link`.
- H3: uzyj `combined_text`, cech numerycznych, cech strukturalnych i `split_chronological`.
