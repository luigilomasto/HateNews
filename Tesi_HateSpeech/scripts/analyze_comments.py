import pandas as pd

df = pd.read_csv("data/raw/haspeede1_comments.csv")

counts = df['label'].value_counts(normalize=True) * 100

print("\n--- ANALISI COMMENTI ---")
print(counts)
