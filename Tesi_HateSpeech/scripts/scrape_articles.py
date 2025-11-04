import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse # Per analizzare l'URL

# --- PERCORSI DEI FILE ---
INPUT_FILE = 'data/processed/articles_raw.csv'
OUTPUT_FILE = 'data/processed/articles_raw.csv' # Leggiamo e scriviamo sullo stesso file

# --- HEADER PER SIMULARE UN BROWSER ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_selectors(url):
    """
    Restituisce i selettori HTML (tag e classe) giusti
    per il sito web specificato nell'URL.
    """
    
    domain = urlparse(url).netloc

    if "repubblica.it" in domain:
        # --- Selettori per REPUBBLICA (Confermati!) ---
        title_tag = 'h1'
        title_class = 'story__title' 
        content_tag = 'div'
        content_class = 'story__content'

    elif "parmatoday.it" in domain:
        # --- Selettori per PARMATODAY (Confermati!) ---
        title_tag = 'h1'
        title_class = 'l-entry__title' 
        content_tag = 'div'
        content_class = 'c-entry'
        
    elif "lagazzettadelmezzogiorno.it" in domain:
        # --- Selettori per LA GAZZETTA (Confermati!) ---
        title_tag = 'h1'
        title_class = 'titolo_articolo' 
        content_tag = 'div'
        content_class = 'testo_articolo'
        
    else:
        print(f"Attenzione: Selettori non configurati per il dominio {domain}")
        return None, None, None, None

    return title_tag, title_class, content_tag, content_class


def scrape_article_details(url):
    """
    Visita una singola URL, ottiene i selettori giusti
    e estrae titolo e contenuto.
    """
    
    title_tag, title_class, content_tag, content_class = get_selectors(url)
    
    if title_tag is None:
        return "DOMINIO NON CONFIGURATO", "DOMINIO NON CONFIGURATO"

    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Errore nel caricare {url}: Status {response.status_code}")
            return "ERRORE HTTP", "ERRORE HTTP"

        soup = BeautifulSoup(response.text, 'html.parser')

        # 2. Trova il titolo
        title_element = soup.find(title_tag, class_=title_class)
        if title_element:
            title = title_element.get_text(strip=True)
        else:
            title = "TITOLO NON TROVATO" 

        # 3. Trova il contenuto
        content_element = soup.find(content_tag, class_=content_class)
        
        # --- QUI C'È IL FIX 2 ---
        if content_element:
            # Metodo più robusto: prende tutto il testo dal contenitore,
            # usando uno spazio come separatore per unire i pezzi.
            content = content_element.get_text(strip=True, separator=' ')
        else:
            content = "CONTENUTO NON TROVATO"
        # --- FINE FIX 2 ---
        
        return title, content

    except Exception as e:
        print(f"Errore generico during scraping di {url}: {e}")
        return "ERRORE SCRAPING", "ERRORE SCRAPING"

def main():
    """
    Funzione principale: carica il CSV, itera sui link,
    esegue lo scraping e salva il file aggiornato.
    """
    try:
        # --- QUI C'È IL FIX 1 ---
        # Specifichiamo il 'dtype' (tipo di dato) per le colonne
        # 'object' è il modo in cui pandas definisce il "testo"
        df = pd.read_csv(
            INPUT_FILE, 
            encoding='utf-8', 
            dtype={'title': 'object', 'content': 'object'}
        )
        # --- FINE FIX 1 ---
        
    except FileNotFoundError:
        print(f"Errore: File non trovato in {INPUT_FILE}")
        return
    
    # Riempiamo eventuali 'NaN' (Not a Number) con stringhe vuote
    # Questo serve a pulire le colonne che pandas ha letto come vuote
    df.fillna('', inplace=True)

    print("Inizio scraping degli articoli (versione corretta)...")

    for index, row in df.iterrows():
        url = row['url']
        
        # Saltiamo lo scraping solo se la colonna 'title' è GIA PIENA
        # e non è un vecchio errore
        error_conditions = ["TITOLO NON TROVATO", "CONTENUTO NON TROVATO", "ERRORE HTTP", "ERRORE SCRAPING", "DOMINIO NON CONFIGURATO"]
        
        if row['title'] != '' and row['title'] not in error_conditions:
            print(f"Skipping (già processato): {url}")
            continue

        print(f"Processing: {url}")
        
        title, content = scrape_article_details(url)
        
        df.at[index, 'title'] = title
        df.at[index, 'content'] = content
        
        time.sleep(1) 

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\nScraping completato! Dati salvati in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()