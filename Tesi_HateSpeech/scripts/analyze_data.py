import pandas as pd
from transformers import pipeline
import os

# Input/Output
INPUT_FILE = "Tesi_HateSpeech/data/raw/comments_dataset.csv"
OUTPUT_FILE = "Tesi_HateSpeech/data/processed/comments_analyzed.csv"
MODEL_NAME = "IMSyPP/hate_speech_it"

def analyze_dataset():
    print("\n--- ANALISI CON MODELLO IMSyPP ---\n")

    if not os.path.exists(INPUT_FILE):
        print("❌ Errore: dataset non trovato. Esegui prima download_datasets.py")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"✔ Caricate {len(df)} righe.")

    # Caricamento modello
    classifier = pipeline("text-classification", model=MODEL_NAME)

    labels_map = {
        "LABEL_0": "Acceptable",
        "LABEL_1": "Inappropriate",
        "LABEL_2": "Offensive",
        "LABEL_3": "Violent"
    }

    ai_labels = []
    ai_scores = []

    print("Classificazione in corso...")

    for idx, row in df.iterrows():
        text = str(row['text'])[:512]
        pred = classifier(text)[0]

        ai_labels.append(labels_map[pred['label']])
        ai_scores.append(pred['score'])

        if idx % 200 == 0:
            print(f"   Analizzati {idx}/{len(df)}...")

    df['ai_label'] = ai_labels
    df['ai_score'] = ai_scores

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✔ File analizzato salvato in:\n{OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_dataset()
