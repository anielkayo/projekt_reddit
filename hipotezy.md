1. Hipoteza "Eskalacji Emocjonalnej"
Treść: „Prawdopodobieństwo wystąpienia linku negatywnego między dwoma subredditami wzrasta, jeśli w poprzednich 24 godzinach wystąpiły między nimi interakcje o wysokim natężeniu słownictwa związanego z gniewem (anger)”.   + HuggingFace Transformers wykorzystujemy własne właściwości- samemu wykonać analizę- bardziej zaawansowane (rok 2017r) - nowsze modele i biblioteki pythona 
2. Hipoteza "Złożoności Poznawczej"
Treść: „Teksty towarzyszące negatywnym linkom mają niższą złożoność językową (np. krótsze słowa, mniej spójników logicznych) niż teksty w linkach pozytywnych”.
Uzasadnienie: Często agresja wiąże się z uproszczonym widzeniem świata. Możemy to zweryfikować, korelując sentyment z miarami takimi jak średnia długość słowa lub wskaźnik czytelności, które można wyliczyć z body properties.

3. Hipoteza Multimodalna (Najsilniejsza badawczo)
"Modele uczenia maszynowego (np. Random Forest) potrafią przewidzieć wystąpienie negatywnego odniesienia między dwiema społecznościami ze skutecznością (F1-score) powyżej 75%, łącząc cechy lingwistyczne (czas przyszły, emocje) z cechami strukturalnymi sieci (wspólni sąsiedzi, dotychczasowa historia interakcji).”
