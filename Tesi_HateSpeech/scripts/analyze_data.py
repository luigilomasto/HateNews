import pandas as pd
from transformers import pipeline
import sys
import os

# --- CONFIGURAZIONE ---
# Percorsi dei file generati dallo script di download
COMMENTS_FILE = "Tesi_HateSpeech/data/raw/comments_dataset.csv"
OUTPUT_FILE = "Tesi_HateSpeech/data/processed/comments_analyzed.csv"

# Modello approvato dal prof (IMSyPP)
MODEL_NAME = "IMSyPP/hate_speech_it"

def analyze_dataset(filename):
    print(f"\n--- AVVIO ANALISI COMPLETA SU: {filename} ---")
    
    # 1. Caricamento Dati
    if not os.path.exists(filename):
        print(f"❌ Errore: Il file {filename} non esiste.")
        print("   Esegui prima 'python Tesi_HateSpeech/scripts/download_datasets.py'")
        return

    try:
        df = pd.read_csv(filename)
        print(f"✔ Caricate {len(df)} righe dal dataset.")
        # NOTA: Ho rimosso il limite .head(100), ora analizza TUTTO.
    except Exception as e:
        print(f"❌ Errore lettura CSV: {e}")
        return

    # 2. Caricamento Modello IA
    print(f"Caricamento modello IA '{MODEL_NAME}'...")
    try:
        classifier = pipeline("text-classification", model=MODEL_NAME)
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

    # 3. Classificazione
    print("Classificazione in corso (questo processo richiederà alcuni minuti)...")
    results = []
    scores = []
    
    # Mappa delle etichette del modello IMSyPP
    labels_map = {
        "LABEL_0": "Acceptable",
        "LABEL_1": "Inappropriate",
        "LABEL_2": "Offensive",
        "LABEL_3": "Violent"
    }

    total_rows = len(df)
    
    for index, row in df.iterrows():
        text = str(row['text'])[:512] # Taglia testi troppo lunghi per il modello
        try:
            pred = classifier(text)[0]
            label_code = pred['label']
            # Converte LABEL_0 in "Acceptable", ecc.
            label_human = labels_map.get(label_code, label_code)
            
            results.append(label_human)
            scores.append(pred['score'])
        except:
            results.append("Error")
            scores.append(0.0)

        # Barra di avanzamento: stampa ogni 100 commenti
        if index % 100 == 0:
            print(f"   Analizzati {index}/{total_rows} commenti...")

    # Aggiungiamo i risultati al DataFrame
    df['ai_label'] = results
    df['ai_score'] = scores

    # 4. Statistiche Finali
    print("\n--- RISULTATI ANALISI ---")
    # Mostra le percentuali di ogni categoria
    stats = df['ai_label'].value_counts(normalize=True) * 100
    print(stats)
    
    # 5. Salvataggio
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✔ Risultati salvati in: {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_dataset(COMMENTS_FILE)