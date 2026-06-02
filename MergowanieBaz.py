import pandas as pd
import tkinter as tk
new_base = pd.read_csv(r"C:\Users\macie\Desktop\Projekt AI\reddit_full_sentiment_analyzed_fast.csv")
old_base = pd.read_csv(r"C:\Users\macie\Desktop\Projekt AI\Fixed_Property.csv", sep=';', encoding='utf-8-sig',parse_dates=False)
#print(old_base.columns.tolist())
new_base = new_base[['POST_ID', 'Content_Sentiment','TIMESTAMP','Content_Score','Raw_Content','Raw_Title']]
old_base = old_base[[
    'SOURCE_SUBREDDIT',
    'TARGET_SUBREDDIT',
    'POST_ID',

    'LINK_SENTIMENT',
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
]]
old_base_unique = old_base.drop_duplicates(subset=['POST_ID'])
new_base_unique = new_base.drop_duplicates(subset=['POST_ID'])
sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}


new_base_unique[['Content_Sentiment']] = new_base[['Content_Sentiment']].replace(sentiment_map)

merged = pd.merge(new_base_unique, old_base_unique, on='POST_ID', suffixes=('_stara', '_nowa'))
merged.to_csv(r"C:\Users\macie\Desktop\Projekt AI\NajnowszaWersjaBazy1205.csv", index=False)
#print(merged.head(10))












#|-------------------------------------------------------------
#roznice_content = merged[merged['LINK_SENTIMENT'] != merged['Content_Sentiment']]
#roznice_title = merged[merged['LINK_SENTIMENT'] != merged['Title_Sentiment']]
#old_base_unique.describe()
#print(merged.columns.tolist())
#import seaborn as sns
#import matplotlib.pyplot as plt


#y_label = "LIWC_Death"
#dostepne_kolumny = merged.select_dtypes(include=['float64', 'int64']).columns.tolist()

#def wybierz_kolumny_z_okienka(kolumny):
#    root = tk.Tk()
#    root.title("Wybór danych")
#    # Okno pojawi się na wierzchu
#    root.attributes('-topmost', True)

#    tk.Label(root, text="Wybierz kolumny do macierzy korelacji:\n(Przytrzymaj CTRL, aby zaznaczyć kilka)").pack(pady=10)

    # Tworzymy listę wielokrotnego wyboru
#    listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=70, height=20)
#    for col in kolumny:
#        listbox.insert(tk.END, col)
#    listbox.pack(padx=20, pady=5)

#    wybrane_lista = []

    # Akcja po kliknięciu przycisku
#    def zatwierdz():
#        zaznaczone_indeksy = listbox.curselection()
#        for i in zaznaczone_indeksy:
#            wybrane_lista.append(listbox.get(i))
#        root.destroy() # Zamykamy okienko

#    tk.Button(root, text="Generuj macierz korelacji", command=zatwierdz, bg="lightblue").pack(pady=10)
    
#    root.mainloop() # Zatrzymuje skrypt i czeka na akcję użytkownika
#    return wybrane_lista

# 3. Odpalamy nasze okienko!
#Wybrnae_kolumny = wybierz_kolumny_z_okienka(dostepne_kolumny)

# 4. Generujemy wykres TYLKO wtedy, gdy użytkownik coś wybrał
#if len(Wybrnae_kolumny) > 1:
#    print(f"Wybrano kolumny: {Wybrnae_kolumny}")
#    matryca_korelacji = merged[Wybrnae_kolumny].corr()
#    plt.figure(figsize=(10,8))
#    sns.heatmap(matryca_korelacji, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#    #plt.title("Macierz korelacji")
    #plt.tight_layout()
    #plt.show()
#else:
#    print("Anulowano. Aby zobaczyć macierz korelacji, musisz wybrać minimum 2 kolumny.")




#Wybrnae_kolumny = [
#    "LIWC_Relig", "LIWC_Death", 
#    "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"]

#matryca_korelacji = merged[Wybrnae_kolumny].corr()
#plt.figure(figsize=(10,8))
#sns.heatmap(matryca_korelacji, annot=True, cmap='coolwarm',fmt=".2f",linewidths=0.5)
#plt.title("Macierz korelacji")
#plt.show()
#print(f"Liczba różnic Body {len(roznice_content)}")
#print(f"Liczba różnic Title {len(roznice_title)}")