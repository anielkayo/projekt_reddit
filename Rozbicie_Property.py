path = r"C:\Users\macie\Desktop\Projekt AI\soc-redditHyperlinks-body.tsv"

import pandas as pd
import os

# 1. Definiujemy czystą listę 86 nazw kolumn (bez numeracji)
columns_names = [
    "Number of characters", "Number of characters without counting white space", 
    "Fraction of alphabetical characters", "Fraction of digits", "Fraction of uppercase characters", 
    "Fraction of white spaces", "Fraction of special characters, such as comma, exclamation mark, etc.", 
    "Number of words", "Number of unique works", "Number of long words (at least 6 characters)", 
    "Average word length", "Number of unique stopwords", "Fraction of stopwords", 
    "Number of sentences", "Number of long sentences (at least 10 words)", 
    "Average number of characters per sentence", "Average number of words per sentence", 
    "Automated readability index", "Positive sentiment calculated by VADER", 
    "Negative sentiment calculated by VADER", "Compound sentiment calculated by VADER", 
    "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", 
    "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", 
    "LIWC_Past", "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", 
    "LIWC_Negate", "LIWC_Quant", "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family", 
    "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", 
    "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", 
    "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", 
    "LIWC_See", "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual", 
    "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work", 
    "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", "LIWC_Death", 
    "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"
]

# 2. Wczytujemy oryginalny plik TSV
# Używamy sep='\t', ponieważ to plik TSV, a nie CSV
df = pd.read_csv(path, sep='\t')

# 3. Rozdzielamy kolumnę PROPERTIES po przecinku
# Parametr expand=True sprawia, że wynik to nowy DataFrame z 86 kolumnami
properties_split = df['PROPERTIES'].str.split(',', expand=True)

# 4. Przypisujemy nasze nazwy do nowych kolumn
properties_split.columns = columns_names

# 5. Konwertujemy wszystkie rozdzielone dane z tekstu na liczby (float)
# Dzięki temu będziesz mógł robić na nich wykresy i obliczenia matematyczne
properties_split = properties_split.astype(float)

# 6. Łączymy oryginalną bazę (usuwając z niej starą kolumnę PROPERTIES) z nowymi kolumnami
df_final = pd.concat([df.drop('PROPERTIES', axis=1), properties_split], axis=1)

# Sprawdzamy efekt
print("Rozmiar nowej bazy:", df_final.shape)
print(df_final.head())

df_final.to_csv(r'C:\Users\macie\Desktop\Projekt AI\Fixed_Property.csv', index=False, sep=';', encoding='utf-8-sig')
print("Twój plik zapisał się tutaj:", os.getcwd())