import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE = "Tesi_HateSpeech/data/processed/comments_analyzed.csv"
IMG_DIR = "Tesi_HateSpeech/data/processed/plots"

def main():
    print("\n--- GENERAZIONE GRAFICI (ITALIANO) ---")

    if not os.path.exists(INPUT_FILE):
        print("❌ File non trovato. Esegui analyze_data.py")
        return

    df = pd.read_csv(INPUT_FILE)
    os.makedirs(IMG_DIR, exist_ok=True)

    # Traduzione etichette IA
    translate = {
        "Acceptable": "Accettabile",
        "Inappropriate": "Inappropriato",
        "Offensive": "Offensivo",
        "Violent": "Violento"
    }

    df['ai_label'] = df['ai_label'].map(translate)

    sns.set_theme(style="whitegrid")

    order = ["Accettabile", "Inappropriato", "Offensivo", "Violento"]

    # --- Grafico distribuzione generale ---
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x="ai_label", order=order, palette="viridis")
    plt.title("Distribuzione delle Categorie (Modello IMSyPP)")
    plt.xlabel("Categoria")
    plt.ylabel("Numero di Commenti")

    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percent = height / total * 100
        ax.text(p.get_x() + p.get_width()/2, height + 20,
                f"{percent:.1f}%", ha='center', fontweight='bold')

    path = os.path.join(IMG_DIR, "distribuzione.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"✔ Grafico salvato: {path}")

    print("\nFINITO!")

if __name__ == "__main__":
    main()
