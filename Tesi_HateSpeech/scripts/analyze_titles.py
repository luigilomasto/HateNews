import pandas as pd
import sys
import re # Modulo per la ricerca avanzata (RegEx)

# --- PERCORSI DEI FILE ---
INPUT_FILE = 'data/processed/articles_raw.csv'
OUTPUT_FILE = 'data/processed/articles_titles_processed.csv' 

# --- CUORE DELL'ANALISI LINGUISTICA (v3.0) ---
# ORA INCLUDIAMO ANCHE I PLURALI!
CONNOTATED_KEYWORDS = [
    # Etnia/Nazionalità (singolari e plurali)
    'immigrato', 'immigrati',
    'straniero', 'stranieri',
    'clandestino', 'clandestini',
    'nordafricano', 'nordafricani',
    'africano', 'africani',
    'sudamericano', 'sudamericani',
    'extracomunitario', 'extracomunitari',
    'marocchino', 'marocchini',
    'tunisino', 'tunisini',
    'albanese', 'albanesi',
    'rumeno', 'rumeni',
    'nigeriano', 'nigeriani',
    
    # Status Sociale (singolari e plurali)
    'rom',
    'nomade', 'nomadi',
    'zingaro', 'zingari',
    'senzatetto', # Plurale è uguale
    'barbone', 'barboni',
    'profugo', 'profughi',
    'richiedente asilo', 'richiedenti asilo'
]

def classify_title(title):
    """
    Classifica un titolo come 'Connotato' o 'Neutro'
    usando una ricerca "whole word" (parola intera).
    """
    try:
        title_lower = str(title).lower()
    except AttributeError:
        return 'Neutro' 

    for keyword in CONNOTATED_KEYWORDS:
        # Cerca la parola chiave esatta (ignorando la punteggiatura)
        if re.search(r'\b' + re.escape(keyword) + r'\b', title_lower):
            # Se trova ANCHE UNA SOLA keyword, è "Connotato" e ci fermiamo
            return 'Connotato'
    
    # Se il ciclo finisce senza trovare nessuna keyword, è "Neutro"
    return 'Neutro'

def main():
    """
    Funzione principale: carica i dati, applica la 
    classificazione e salva i risultati.
    """
    print(f"Inizio analisi e classificazione dei titoli (v3.0 - Plurali) da {INPUT_FILE}...")
    
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except FileNotFoundError:
        print(f"Errore: File non trovato in {INPUT_FILE}")
        print("Assicurati di aver prima eseguito 'scrape_articles.py'")
        sys.exit(1) 
        
    if 'title' not in df.columns:
        print(f"Errore: La colonna 'title' non è presente nel file {INPUT_FILE}.")
        sys.exit(1)

    df['title_classification'] = df['title'].apply(classify_title)

    try:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Errore durante il salvataggio del file: {e}")
        sys.exit(1)
        
    print(f"\nClassificazione completata! Dati salvati in {OUTPUT_FILE}")
    print("--- Anteprima dei Risultati (Definitiva) ---")
    print(df[['title', 'title_classification']].to_string())


# --- BLOCCO DI AVVIO ---
if __name__ == "__main__":
    main()