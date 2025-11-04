import pandas as pd
import sys
import re # Useremo 're' per la ricerca di parole

# --- PERCORSI DEI FILE (corretti dalla tua posizione) ---
ARTICLES_FILE = 'Tesi_HateSpeech/data/processed/articles_titles_processed.csv'
COMMENTS_FILE = 'Tesi_HateSpeech/data/raw/comments_raw.csv'
OUTPUT_FILE = 'Tesi_HateSpeech/data/processed/comments_classified.csv'

# --- CUORE DELL'ANALISI (v3.0 - Keyword) ---
# Queste liste sono basate sui commenti che HAI TROVATO TU.
# Puoi (e devi) espanderle per la tua tesi.

# Parole che indicano odio, stereotipi, attacchi
HATE_KEYWORDS = [
    'straniero', 'stranieri', 'immigrato', 'immigrati', 'clandestino',
    'mandatelo a casa', 'espellerlo', 'casa sua',
    'risorse', 'invasione', 'delinquenti', 'schifo',
    'povera italia', 'vergogna'
]

# Parole che indicano polarizzazione politica o critica al "sistema"
POLITICS_KEYWORDS = [
    'sinistra', 'destra',
    'pd', 'meloni', 'salvini', 'lega',
    'giudici', 'magistratura', 'giustizia',
    'buonisti', 'farlocca', 'leggi',
    'sistema', 'carceri', 'libero', 'liberati'
]

def classify_comment_by_keyword(text):
    """
    Classifica un commento in base alle parole chiave.
    Restituisce 'hate_speech', 'polarizing' o 'other'.
    """
    try:
        comment_lower = str(text).lower()
    except AttributeError:
        return 'other'

    # 1. Controlla prima le parole di odio
    for keyword in HATE_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', comment_lower):
            return 'hate_speech'
            
    # 2. Se non è odio, controlla se è polarizzazione
    for keyword in POLITICS_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', comment_lower):
            return 'polarizing'

    # 3. Se non è nessuno dei due
    return 'other'

def main():
    print("Inizio analisi commenti (v3.0 - Keyword) dei commenti...")
    
    # 1. Caricare i file
    try:
        df_articles = pd.read_csv(ARTICLES_FILE, encoding='utf-8')
        df_comments = pd.read_csv(COMMENTS_FILE, encoding='utf-8')
    except FileNotFoundError as e:
        print(f"Errore: File non trovato. {e}")
        sys.exit(1)

    # 2. Unire (Merge) i due DataFrame
    print("Unione dei commenti con la classificazione dei titoli...")
    df_merged = pd.merge(
        df_comments,
        df_articles[['url', 'title', 'title_classification']],
        left_on='article_url',
        right_on='url',
        how='left'
    )

    # 3. Eseguire l'Analisi con Keyword
    print("Esecuzione dell'analisi (Keyword) su ogni commento...")
    
    df_merged['comment_classification'] = df_merged['comment_text'].apply(classify_comment_by_keyword)
    
    print("Analisi completata.")

    # 4. Salvare i risultati
    try:
        df_merged.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Errore during saving file: {e}")
        sys.exit(1)

    print(f"\nAnalisi completata! Dati salvati in {OUTPUT_FILE}")
    print("--- Anteprima dei Risultati dell'Analisi (Keyword) ---")
    
    # Stampiamo le colonne importanti
    print(df_merged[['title_classification', 'comment_text', 'comment_classification']].to_string())
    
    print("\n--- RIEPILOGO STATISTICO (KEYWORD) ---")
    # Raggruppa per tipo di titolo e conta le classificazioni
    summary = df_merged.groupby('title_classification')['comment_classification'].value_counts(normalize=True).unstack(fill_value=0)
    print("Percentuale di tipo di commento per tipo di titolo:")
    print(summary)


# --- BLOCCO DI AVVIO ---
if __name__ == "__main__":
    main()