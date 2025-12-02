import os
import pandas as pd
from datasets import load_dataset

# Paths
DATA_DIR = "Tesi_HateSpeech/data/raw"
OUTPUT_FILE = os.path.join(DATA_DIR, "comments_dataset.csv")

def main():
    print("\n--- DOWNLOAD DATASET EVALITA: HASPEEDE 2018 + 2020 ---\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    # --- HASPEEDE 2018 ---
    print("Scaricamento HaSpeeDe 2018...")
    ds2018_train = load_dataset("valeriobiscione/haspeede", split="train")
    ds2018_test = load_dataset("valeriobiscione/haspeede", split="test")

    df2018 = pd.concat([pd.DataFrame(ds2018_train), pd.DataFrame(ds2018_test)])
    df2018['source'] = '2018'

    # --- HASPEEDE 2020 (HaSpeeDe 2) ---
    print("Scaricamento HaSpeeDe 2020...")
    ds2020_train = load_dataset("valeriobiscione/haspeede2", split="train")
    ds2020_test = load_dataset("valeriobiscione/haspeede2", split="test")

    df2020 = pd.concat([pd.DataFrame(ds2020_train), pd.DataFrame(ds2020_test)])
    df2020['source'] = '2020'

    # --- UNIONE DEI DATASET ---
    df_all = pd.concat([df2018, df2020], ignore_index=True)

    # Rinomina le colonne per uniformità
    df_all = df_all[['text', 'label', 'source']]
    df_all['label'] = df_all['label'].map({0: "Non Hate", 1: "Hate Speech"})

    # Salvataggio finale
    df_all.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✔ Dataset EVALITA unificato salvato in:\n{OUTPUT_FILE}")
    print(f"Totale righe: {len(df_all)}")
    print("\nOra puoi eseguire: analyze_data.py")

if __name__ == "__main__":
    main()
