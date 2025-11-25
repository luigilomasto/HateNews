import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Percorsi
INPUT_FILE = "Tesi_HateSpeech/data/processed/comments_analyzed.csv"
IMG_DIR = "Tesi_HateSpeech/data/processed/plots"

def main():
    print("GENERAZIONE GRAFICI (IN ITALIANO)...")
    
    # 1. Carica i dati
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("Errore: File dati non trovato.")
        return

    # Crea cartella immagini
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    # --- TRADUZIONE ETICHETTE (Novità) ---
    # Mappiamo le etichette inglesi in italiano corretto
    translation_map = {
        "Acceptable": "Accettabile",
        "Inappropriate": "Inappropriato",
        "Offensive": "Offensivo",
        "Violent": "Violento"
    }
    
    # Applica la traduzione alla colonna 'ai_label'
    # Se trova un valore non presente nella mappa, lo lascia com'è
    df['ai_label'] = df['ai_label'].map(translation_map).fillna(df['ai_label'])

    # Imposta lo stile grafico
    sns.set_theme(style="whitegrid")

    # --- GRAFICO 1: DISTRIBUZIONE TOTALE (BAR PLOT) ---
    plt.figure(figsize=(10, 6))
    
    # Ordine delle categorie in italiano
    order = ["Accettabile", "Inappropriato", "Offensivo", "Violento"]
    
    # Crea il grafico
    ax = sns.countplot(x='ai_label', data=df, order=order, palette="viridis")
    
    plt.title("Distribuzione delle Categorie di Hate Speech (Modello IMSyPP)", fontsize=14)
    plt.xlabel("Categoria", fontsize=12)
    plt.ylabel("Numero di Commenti", fontsize=12)
    
    # Aggiungi le percentuali sopra le barre
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height): height = 0 # Gestione casi vuoti
        percentage = '{:.1f}%'.format(100 * height/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = height + 20
        ax.text(x, y, percentage, ha='center', size=12, weight='bold')

    # Salva
    save_path1 = os.path.join(IMG_DIR, "grafico_distribuzione.png")
    plt.savefig(save_path1, dpi=300)
    print(f"✔ Grafico 1 salvato (Italiano): {save_path1}")
    plt.close()

    # --- GRAFICO 2: CONFRONTO HATE SPEECH REALE vs AI ---
    if 'original_label' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Prendiamo solo quelli che erano etichettati come "Hate Speech" nel dataset originale
        hate_only = df[df['original_label'] == 'Hate Speech']
        
        ax2 = sns.countplot(x='ai_label', data=hate_only, order=order, palette="magma")
        
        plt.title("Come l'IA classifica il VERO Hate Speech", fontsize=14)
        plt.xlabel("Classificazione Modello IMSyPP", fontsize=12)
        plt.ylabel("Conteggio", fontsize=12)
        
        # Aggiunge etichette anche qui
        total_hate = len(hate_only)
        for p in ax2.patches:
            height = p.get_height()
            if pd.isna(height): height = 0
            percentage = '{:.1f}%'.format(100 * height/total_hate)
            x = p.get_x() + p.get_width() / 2 - 0.05
            y = height + 5
            ax2.text(x, y, percentage, ha='center', size=10)

        save_path2 = os.path.join(IMG_DIR, "grafico_confronto_hate.png")
        plt.savefig(save_path2, dpi=300)
        print(f"✔ Grafico 2 salvato (Italiano): {save_path2}")
        plt.close()

    print("\nFINITO! I grafici ora sono in italiano corretto.")

if __name__ == "__main__":
    main()