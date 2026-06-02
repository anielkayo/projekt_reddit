# Poprawiony Notebook Sentymentu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Przygotowac poprawiony notebook na podstawie `PBL_dataset_analysis.ipynb`, ktory przewiduje sentyment linku roznymi metodami i porownuje wyniki modeli.

**Architecture:** Oryginalny notebook zostaje jako material zrodlowy, a nowy notebook `PBL_dataset_analysis_poprawiony.ipynb` powstaje obok niego. Notebook jest samodzielny: wczytuje baze, przygotowuje target, trenuje proste baseline'y i modele ML, zapisuje wyniki oraz pokazuje interpretacje.

**Tech Stack:** Python, pandas, scikit-learn, matplotlib/seaborn, nbformat, opcjonalnie transformers/sentence-transformers w komorkach domyslnie wylaczonych.

---

### Task 1: Utworzenie notebooka

**Files:**
- Create: `PBL_dataset_analysis_poprawiony.ipynb`

- [ ] **Step 1: Zbuduj sekcje markdown**

Dodaj sekcje: cel, wczytanie danych, przygotowanie targetu, baseline'y klasyczne, modele ML, porownanie metryk, przyklady predykcji, opcjonalne modele Hugging Face, wnioski.

- [ ] **Step 2: Dodaj prosty kod**

Kod ma uzywac tych samych glownych bibliotek i podejsc co oryginal: `pandas`, `train_test_split`, `classification_report`, `confusion_matrix`, `DecisionTreeClassifier`, `RandomForestClassifier`, `LogisticRegression`, `LinearSVC`, `TfidfVectorizer` oraz opcjonalnie `transformers`.

- [ ] **Step 3: Dodaj porownanie modeli**

Porownaj co najmniej: Dummy baseline, VADER compound baseline, istniejący `Content_Sentiment` baseline, TF-IDF + Logistic Regression, TF-IDF + Linear SVC, Properties + Logistic Regression, Properties + Decision Tree, Properties + Random Forest, HF sentiment features + Logistic Regression, TF-IDF + Properties + Logistic Regression.

### Task 2: Weryfikacja

**Files:**
- Check: `PBL_dataset_analysis_poprawiony.ipynb`

- [ ] **Step 1: Sprawdz strukture notebooka**

Uruchom walidacje JSON/nbformat i policz komorki markdown oraz code.

- [ ] **Step 2: Uruchom notebook lub smoke-test modelowania**

Notebook powinien przejsc wykonanie bez pobierania modeli z internetu, poniewaz komorki Hugging Face sa domyslnie wylaczone.

- [ ] **Step 3: Sprawdz artefakty wynikowe**

Sprawdz, czy powstaja `outputs/model_comparison_poprawiony.csv` oraz `outputs/predykcje_sentymentu_poprawiony.csv`.
