import os
import pandas as pd
from datasets import load_dataset

# Percorsi di salvataggio
DATA_DIR = "Tesi_HateSpeech/data/raw"
COMMENTS_FILE = os.path.join(DATA_DIR, "comments_dataset.csv")
TITLES_FILE = os.path.join(DATA_DIR, "titles_dataset.csv")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("\n--- DOWNLOAD DATASET (VIA HUGGING FACE) ---\n")

    try:
        # Scarichiamo il dataset "HateCheck" (versione italiana)
        # È un dataset accademico ufficiale per testare i modelli di Hate Speech
        print("Scaricamento dataset 'Paul/hatecheck-italian'...")
        dataset = load_dataset("Paul/hatecheck-italian", split="test")
        
        # Convertiamo in DataFrame (Tabella)
        df = pd.DataFrame(dataset)
        
        # Il dataset ha colonne: 'functionality', 'test_case', 'label_gold', etc.
        # Lo adattiamo al nostro formato standard: 'text', 'label'
        df_clean = pd.DataFrame()
        df_clean['text'] = df['test_case']
        df_clean['label'] = df['label_gold'] # h = hateful, nh = non-hateful
        
        # Mappiamo le etichette per chiarezza
        df_clean['label'] = df_clean['label'].map({'h': 'Hate Speech', 'nh': 'Non Hate'})

        # SALVATAGGIO
        # Usiamo lo stesso dataset sia per "commenti" che per "titoli" 
        # per testare il sistema (visto che mancano titoli reali nel dataset)
        
        # 1. Salviamo come commenti
        df_clean.to_csv(COMMENTS_FILE, index=False)
        print(f"✔ Dataset Commenti salvato: {COMMENTS_FILE} ({len(df_clean)} righe)")
        
        # 2. Salviamo come titoli (ne prendiamo un campione diverso o uguale)
        df_clean.to_csv(TITLES_FILE, index=False)
        print(f"✔ Dataset Titoli salvato: {TITLES_FILE}")
        
        print("\n✔ Download completato con successo.")
        print("Ora puoi eseguire 'analyze_data.py'.")

    except Exception as e:
        print(f"❌ Errore critico: {e}")

if __name__ == "__main__":
    main()