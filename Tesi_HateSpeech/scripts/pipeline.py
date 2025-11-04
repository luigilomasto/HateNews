import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import os
import re
import matplotlib.pyplot as plt

# -----------------------------
# Percorsi file
# -----------------------------
RAW_ARTICLES_FILE = "data/raw/article_urls.txt"
RAW_COMMENTS_FILE = "data/raw/comments.csv"
ARTICLES_PROCESSED = "data/processed/articles_raw.csv"
TITLES_PROCESSED = "data/processed/articles_titles_processed.csv"
ENGAGEMENTS_FILE = "data/processed/engagements_by_article.csv"
COMMENTS_CLASSIFIED = "data/processed/comments_classified.csv"

# -----------------------------
# Parole chiave
# -----------------------------
TITLE_KEYWORDS = ["immigrato", "nordafricano", "rom", "extracomunitario"]
HATE_WORDS = ["odio", "razzista", "straniero", "stupido"]

# -----------------------------
# 1. Scarica articoli
# -----------------------------
def fetch_article(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title_tag = soup.find("h1")
        if not title_tag: return None
        title = title_tag.get_text(strip=True)
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text(strip=True) for p in paragraphs])
        return {"url": url, "title": title, "content": content}
    except Exception as e:
        print(f"Errore con {url}: {e}")
        return None

def scrape_articles():
    urls = [line.strip() for line in open(RAW_ARTICLES_FILE)]
    articles = []
    for url in urls:
        data = fetch_article(url)
        if data: articles.append(data)
        time.sleep(1)
    df = pd.DataFrame(articles)
    os.makedirs(os.path.dirname(ARTICLES_PROCESSED), exist_ok=True)
    df.to_csv(ARTICLES_PROCESSED, index=False)
    print(f"Salvati {len(articles)} articoli in {ARTICLES_PROCESSED}")
    return df

# -----------------------------
# 2. Analizza titoli
# -----------------------------
def classify_title(title):
    for kw in TITLE_KEYWORDS:
        if re.search(rf"\b{kw}\b", title, re.IGNORECASE):
            return "connotato"
    return "neutro"

def analyze_titles(df):
    df['title_type'] = df['title'].apply(classify_title)
    df.to_csv(TITLES_PROCESSED, index=False)
    print(f"Titoli classificati in {TITLES_PROCESSED}")
    return df

# -----------------------------
# 3. Simula interazioni social
# -----------------------------
def simulate_engagement(title_type):
    if title_type == "connotato":
        return {
            "likes": random.randint(100, 500),
            "shares": random.randint(50, 100),
            "comments": random.randint(20, 50)
        }
    else:
        return {
            "likes": random.randint(0, 300),
            "shares": random.randint(0, 50),
            "comments": random.randint(0, 30)
        }

def collect_engagements(df):
    engagements = []
    for _, row in df.iterrows():
        data = simulate_engagement(row["title_type"])
        data["url"] = row["url"]
        data["title"] = row["title"]
        data["title_type"] = row["title_type"]
        engagements.append(data)
    df_eng = pd.DataFrame(engagements)
    df_eng.to_csv(ENGAGEMENTS_FILE, index=False)
    print(f"Dati di engagement salvati in {ENGAGEMENTS_FILE}")
    return df_eng

# -----------------------------
# 4. Classifica commenti
# -----------------------------
def classify_comment(text):
    for w in HATE_WORDS:
        if w in str(text).lower():
            return "hate_speech"
    return "neutro"

def analyze_comments():
    df = pd.read_csv(RAW_COMMENTS_FILE)
    df['classification'] = df['comment'].apply(classify_comment)
    df.to_csv(COMMENTS_CLASSIFIED, index=False)
    print(f"Commenti classificati in {COMMENTS_CLASSIFIED}")
    return df

# -----------------------------
# 5. Visualizza risultati
# -----------------------------
def visualize_engagements(df):
    plt.figure(figsize=(10,6))
    df.groupby("title_type")[["likes","shares","comments"]].sum().plot(kind="bar")
    plt.ylabel("Totale interazioni")
    plt.title("Interazioni per tipo di titolo (neutro vs connotato)")
    plt.xticks(rotation=0)
    plt.show()

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("1. Scaricando articoli...")
    articles_df = scrape_articles()
    print("2. Analizzando titoli...")
    titles_df = analyze_titles(articles_df)
    print("3. Simulando engagement...")
    engagements_df = collect_engagements(titles_df)
    print("4. Analizzando commenti...")
    comments_df = analyze_comments()
    print("5. Visualizzando risultati...")
    visualize_engagements(engagements_df)

if __name__ == "__main__":
    main()
