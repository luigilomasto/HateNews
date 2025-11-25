import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import time
import os

# --- CONFIGURAZIONE PAGINA (Deve essere la prima istruzione) ---
st.set_page_config(
    page_title="Rilevatore Hate Speech | Tesi S.Tavolo",
    page_icon="⚖️",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZZATO (CORREZIONE COLORI TESTO) ---
st.markdown("""
    <style>
    /* Sfondo generale */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Colore TESTO NERO per tutti i titoli e paragrafi */
    h1, h2, h3, h4, h5, h6, p, li, label {
        color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Stile dei bottoni */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4e73df;
        color: white !important; /* Testo bottone bianco su sfondo blu */
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2e59d9;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Card per i risultati */
    .result-card {
        padding: 25px;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;    
        }
    /* Titoli */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        sans-serif;
    }
            
    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARICAMENTO MODELLO (Con Cache) ---
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="IMSyPP/hate_speech_it", return_all_scores=True)

# Carichiamo il modello all'avvio
model_loaded = False
try:
    # Non mostriamo lo spinner se è già in cache
    with st.spinner("Inizializzazione del motore di Intelligenza Artificiale..."):
        classifier = load_model()
        model_loaded = True
except Exception as e:
    st.error(f"Errore critico nel caricamento del modello: {e}")

# --- MENU LATERALE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2082/2082883.png", width=80) 
    st.title("Navigazione")
    
    page = st.radio(
        "Seleziona una sezione:",
        ["🏠 Analisi in Tempo Reale", "📊 Risultati della Ricerca", "ℹ️ Informazioni sul Progetto"],
        index=0
    )
    
    st.markdown("---")
    st.info(
        """
        **Tesi di Laurea**
        *Dal titolo al flame: come gli articoli online alimentano l’hate speech sui social network”*
        
        **Studente:** Stella Tavolo
        
        **Relatore:** [Delfina Malandrino]
        
        **Modello:** IMSyPP (BERT)
        **Anno Accademico:** 2024/2025
        """
    )

# --- PAGINA 1: ANALISI LIVE (HOME) ---
if page == "🏠 Analisi in Tempo Reale":
    
    # Header con colonne per centrare
    c1, c2, c3 = st.columns([1, 10, 1])
    with c2:
        st.title("Analisi Automatica del Linguaggio")
        st.markdown("""
        Questa interfaccia permette di testare in tempo reale il modello di **Deep Learning** sviluppato per la tesi. 
        Il sistema è in grado di rilevare sfumature linguistiche complesse, distinguendo tra opinioni legittime, linguaggio offensivo e incitamento all'odio.
        """)
        st.markdown("---")

    col_input, col_space, col_result = st.columns([1.2, 0.1, 1])
    
    with col_input:
        st.subheader("📝 Input")
        st.markdown("Inserisci qui sotto un titolo di giornale o un commento social:")
        user_input = st.text_area("", height=150, placeholder="Es: Non sono d'accordo con questa legge... oppure... Questi criminali vanno cacciati!", label_visibility="collapsed")
        
        analyze_btn = st.button("🔍 ANALIZZA IL TESTO")
        
        st.markdown("### 💡 Esempi da provare:")
        if st.button("Test 1: Opinione legittima"):
            st.info("Copia e incolla: *'Non sono d'accordo con la gestione dei flussi migratori, serve più controllo.'*")
            
        if st.button("Test 2: Hate Speech"):
            st.info("Copia e incolla: *'Questi invasori devono tornare a casa loro, sono solo criminali.'*")

    # Logica di Analisi
    if analyze_btn and model_loaded:
        if user_input.strip():
            # Simulazione caricamento per effetto visivo
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.003)
                progress_bar.progress(i + 1)
            progress_bar.empty()
            
            # Predizione vera
            results = classifier(user_input)[0]
            
            # Mappa Etichette (TRADUZIONE CORRETTA IN ITALIANO)
            labels_map = {
                "LABEL_0": "Accettabile",
                "LABEL_1": "Inappropriato",  
                "LABEL_2": "Offensivo",      
                "LABEL_3": "Violento"        
            }
            
            # Colori per il grafico e per i box (Tutti leggibili)
            colors = {
                "Accettabile": "#2ecc71",    # Verde Smeraldo Luminoso
                "Inappropriato": "#f1c40f",  # Giallo Sole / Oro
                "Offensivo": "#e67e22",      # Arancione Vivo
                "Violento": "#e74c3c"        # Rosso Acceso
            }
            
            # Trova la classe vincente
            data = []
            best_label = ""
            best_score = 0
            
            for r in results:
                label_name = labels_map[r['label']]
                score = r['score']
                data.append({"Categoria": label_name, "Confidenza": score})
                
                if score > best_score:
                    best_score = score
                    best_label = label_name
            
            # --- COLONNA RISULTATI ---
            with col_result:
                st.subheader("📊 Risultato dell'Analisi")
                
                # Box colorato
                # NOTA: Il testo "Classificazione" e la Confidenza sono Neri (#000) per leggibilità.
                # Solo il RISULTATO (es. VIOLENTO) prende il colore acceso.
                st.markdown(
                    f"""
                    <div class="result-card" style="border-left: 10px solid {colors[best_label]}; background-color: #ffffff;">
                        <h4 style="color: #000000; margin:0; text-transform: uppercase; letter-spacing: 1px; font-weight: bold;">Classificazione</h4>
                        <h1 style="color: {colors[best_label]}; margin: 10px 0; font-size: 42px;">{best_label.upper()}</h1>
                        <p style="color: #000000; font-size: 16px; font-weight: 500;">Confidenza del modello: <b>{best_score:.1%}</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Spiegazione del risultato
                if best_label == "Accettabile":
                    st.success("✅ **Analisi:** Il testo rientra nei canoni della comunicazione civile. Non sono stati rilevati elementi di tossicità o odio.")
                elif best_label == "Inappropriato":
                    st.warning("⚠️ **Analisi:** Il testo contiene un linguaggio volgare, sarcastico o maleducato, ma non costituisce tecnicamente hate speech contro un gruppo protetto.")
                elif best_label == "Offensivo":
                    st.error("🚫 **Analisi:** Rilevati insulti diretti o l'uso di stereotipi offensivi volti a denigrare una persona o un gruppo.")
                elif best_label == "Violento":
                    st.error("🚨 **ATTENZIONE:** Rilevato contenuto grave. Il testo contiene incitamento all'odio, minacce fisiche o deumanizzazione.")

            # Grafico a barre sotto l'input
            with col_input:
                st.markdown("#### Dettaglio Probabilità")
                df_chart = pd.DataFrame(data)
                fig = px.bar(
                    df_chart, 
                    x="Confidenza", 
                    y="Categoria", 
                    orientation='h', 
                    text_auto='.1%',
                    color="Categoria",
                    color_discrete_map=colors
                )
                fig.update_layout(
                    showlegend=False, 
                    height=250, 
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False),
                    font=dict(color="black") # Testo del grafico NERO
                )
                fig.update_xaxes(range=[0, 1]) 
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ Inserisci del testo prima di cliccare su Analizza.")

# --- PAGINA 2: RISULTATI DELLA RICERCA ---
elif page == "📊 Risultati della Ricerca":
    st.title("📊 Risultati Sperimentali")
    st.markdown("### Analisi del dataset HaSpeeDe (EVALITA)")
    
    st.write("""
    Nell'ambito del lavoro di tesi, il modello è stato utilizzato per analizzare un dataset di riferimento (**HaSpeeDe**, ~3.700 commenti) 
    per validare la sua efficacia nel distinguere le diverse sfumature dell'odio online.
    """)
    
    # Proviamo a caricare i dati reali se esistono
    try:
        df_real = pd.read_csv("Tesi_HateSpeech/data/processed/comments_analyzed.csv")
        
        # Mappa Etichette anche qui per sicurezza
        translation_map = {
            "Acceptable": "Accettabile",
            "Inappropriate": "Inappropriato",
            "Offensive": "Offensivo",
            "Violent": "Violento"
        }
        # Applichiamo la traduzione se i dati nel file sono ancora in inglese
        df_real['ai_label'] = df_real['ai_label'].map(translation_map).fillna(df_real['ai_label'])
        
        counts = df_real['ai_label'].value_counts(normalize=True) * 100
        df_summary = pd.DataFrame({"Categoria": counts.index, "Percentuale": counts.values})
        
        data_available = True
    except:
        st.warning("⚠️ File dei dati analizzati non trovato. Mostro dati simulati per dimostrazione.")
        # Dati di fallback corretti
        data_summary = {
            "Categoria": ["Accettabile", "Offensivo", "Inappropriato", "Violento"],
            "Percentuale": [58.7, 26.6, 8.3, 6.4]
        }
        df_summary = pd.DataFrame(data_summary)
        data_available = False

    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # Grafico 
        with col1:
         fig_pie = px.pie(
            df_summary, 
            values='Percentuale', 
            names='Categoria', 
            title='Distribuzione delle Categorie nel Dataset',
            color='Categoria',
            color_discrete_map={
                "Accettabile": "#196F3D",    # Verde Scuro
                "Inappropriato": "#D35400",  # Arancione Scuro
                "Offensivo": "#C0392B",      # Rosso Mattone
                "Violento": "#641E16"        # Rosso Scuro
            },
            hole=0.5
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label', textfont=dict(color='white'))
        fig_pie.update_layout(font=dict(color="black"), title_font=dict(color="white")
        ) # Testo grafico nero
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.markdown("#### Interpretazione dei Dati")
        st.info("""
        I risultati confermano che l'hate speech non è un fenomeno monolitico:
        
        * **58.7% - Accettabile:** La maggioranza dei commenti, pur se critici, rimane nei limiti della civiltà.  
        * **8.27% - Inappropriato:** Linguaggio non formale o maleducato, senza odio diretto.  
        * **26.6% - Offensivo:** La forma più diffusa di tossicità, caratterizzata da stereotipi e insulti generici.  
        * **6.4% - Violento:** Commenti altamente pericolosi che richiedono un'attenzione prioritaria nella moderazione.
""")
        if data_available:
            totale = f"{len(df_real):,}".replace(",", ".")
        else:
            totale = "3.690"

        st.markdown(f"""
<div style="
    background-color: #F8F9FA;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    text-align: center;
"> <div style="font-size: 1.1em; color: #333;">
        Totale Commenti Analizzati
    </div>
    <div style="font-size: 2.4em; font-weight: 800; color: black; margin-top:5px;">
        {totale}
    </div>
</div>
<div style="
    background-color: #F8F9FA;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    text-align: center;
">
    <div style="font-size: 1.1em; color: #333;">
        Tossicità Complessiva
    </div>
    <div style="font-size: 2.4em; font-weight: 800; color: black; margin-top:5px;">
        41.3%
    </div>
</div>
""", unsafe_allow_html=True)


        st.markdown(
            "*Tossicità Complessiva:* "
            "<span style='color:black; font-weight:800'>41.3%</span><br>"
            "<span style='font-size:0.85em;'>Somma delle categorie negative</span>",
            unsafe_allow_html=True
        )

        
        
# --- PAGINA 3: INFO SUL PROGETTO ---
elif page == "ℹ️ Informazioni sul Progetto":
    st.title("ℹ️ Il Modello Scientifico")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("""
        ### Il Modello: IMSyPP
        Questa applicazione utilizza il modello **`IMSyPP/hate_speech_it`**, basato sull'architettura **BERT (Bidirectional Encoder Representations from Transformers)**.
        
        È stato sviluppato specificamente per la lingua italiana nell'ambito del progetto europeo *Innovative Monitoring Systems and Prevention Policies of Online Hate Speech*.
        
        ---
        
        ### Riferimenti Bibliografici
        La base teorica di questo classificatore è descritta nello studio:
        
        > *Kralj Novak, P., Scantamburlo, T., Pelicon, A., Cinelli, M., Mozetič, I., & Zollo, F. (2022).*
        > **Handling Disagreement in Hate Speech Modelling.**
        > *Springer International Publishing.*
        """)
    
    with c2:
        st.markdown("### Le 4 Categorie")
        st.success("**🟢 Accettabile**\n\nContenuti neutri, positivi o critiche costruttive.")
        st.warning("**🟡 Inappropriato**\n\nLinguaggio volgare o maleducazione, ma senza discriminazione.")
        st.error("**🟠 Offensivo**\n\nInsulti diretti, stereotipi negativi o linguaggio discriminatorio.")
        st.error("**🔴 Violento**\n\nMinacce fisiche, incitamento alla violenza, deumanizzazione.")