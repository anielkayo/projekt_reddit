
## Wnioski badawcze

**H1:** niepotwierdzona. Wysoki poprzedni anger w oknie 24h nie daje istotnego wzrostu prawdopodobienstwa linku negatywnego.

**H2:** czesciowo potwierdzona. Wyniki sugeruja roznice stylistyczne, ale nie prosty wzorzec, w ktorym wszystkie teksty negatywne sa mniej zlozone.

**H3:** niepotwierdzona. Najlepszy model nie osiaga progu F1 = 0.75 dla klasy negatywnej, mimo ze accuracy moze wygladac wysoko przy niezbalansowanej klasie.

## Ograniczenia

- Okno 24h w H1 daje mala liczbe przypadkow z historia tej samej pary.
- `LINK_SENTIMENT` i `Content_Sentiment` nie sa tym samym: pierwsza zmienna opisuje link miedzy spolecznosciami, druga sentyment tresci.
- Podzial chronologiczny jest bardziej realistyczny niz losowy, ale moze obnizac wyniki wzgledem walidacji losowej.
- Modele transformerowe wymagajace pobierania wag zostaly pozostawione jako opcjonalne rozszerzenie.

## Dalsze prace

- Przetestowac H1 dla okien 48h, 7 dni i par nieskierowanych.
- Dla H2 dodac cechy skladniowe lub embeddingowe miary zlozonosci.
- Dla H3 sprawdzic modele kosztoczulne, prog decyzyjny optymalizowany pod F1 oraz embeddingi sentence-transformers.
