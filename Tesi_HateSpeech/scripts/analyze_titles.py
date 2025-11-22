import pandas as pd

df = pd.read_csv("data/raw/haspeede2_titles.csv")

# Conta quanti titoli hate / non-hate
counts = df['label'].value_counts(normalize=True) * 100

print("\n--- ANALISI TITOLI ---")
print(counts)
