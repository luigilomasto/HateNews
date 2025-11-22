import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
INPUT_FILE = "Tesi_HateSpeech/data/processed/comments_analyzed.csv"
IMG_DIR = "Tesi_HateSpeech/data/processed/plots"

def main():
    print("GENERAZIONE GRAFICI...")
    
    # 1. Carica i dati
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("Errore: File dati non trovato.")
        return

    # Crea cartella immagini
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    # Imposta lo stile
    sns.set_theme(style="whitegrid")

    # --- GRAFICO 1: DISTRIBUZIONE TOTALE (BAR PLOT) ---
    plt.figure(figsize=(10, 6))
    
    # Ordine delle categorie per logica
    order = ["Acceptable", "Inappropriate", "Offensive", "Violent"]
    
    # Conta le occorrenze
    ax = sns.countplot(x='ai_label', data=df, order=order, palette="viridis")
    
    plt.title("Distribuzione delle Categorie di Hate Speech (Modello IMSyPP)", fontsize=14)
    plt.xlabel("Categoria", fontsize=12)
    plt.ylabel("Numero di Commenti", fontsize=12)
    
    # Aggiungi le percentuali sopra le barre
    total = len(df)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_height() + 20
        ax.annotate(percentage, (x, y), size=12, weight='bold')

    # Salva
    save_path1 = os.path.join(IMG_DIR, "grafico_distribuzione.png")
    plt.savefig(save_path1, dpi=300)
    print(f"✔ Grafico 1 salvato: {save_path1}")
    plt.close()

    # --- GRAFICO 2: CONFRONTO HATE SPEECH REALE vs AI ---
    # Questo grafico mostra: Dei commenti che erano VERAMENTE odio, come li ha classificati l'IA?
    if 'original_label' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Prendiamo solo quelli che erano etichettati come "Hate Speech" nel dataset originale
        hate_only = df[df['original_label'] == 'Hate Speech']
        
        ax2 = sns.countplot(x='ai_label', data=hate_only, order=order, palette="magma")
        
        plt.title("Come l'IA classifica il VERO Hate Speech", fontsize=14)
        plt.xlabel("Classificazione Modello IMSyPP", fontsize=12)
        plt.ylabel("Conteggio", fontsize=12)
        
        save_path2 = os.path.join(IMG_DIR, "grafico_confronto_hate.png")
        plt.savefig(save_path2, dpi=300)
        print(f"✔ Grafico 2 salvato: {save_path2}")
        plt.close()

    print("\nFINITO! Trovi le immagini nella cartella 'data/processed/plots'")

if __name__ == "__main__":
    main()