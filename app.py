import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.sparse import hstack
import re
from pathlib import Path

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Détecteur de Rumeurs & Simulateur SIR", 
    page_icon="🕵️‍♂️",
    layout="wide"
)

st.title("🕵️‍♂️ Système d'Alerte Précoce : Rumeurs Twitter")
st.markdown("Ce tableau de bord combine **Machine Learning** (Détection) et **Systèmes Dynamiques SIR** (Simulation d'impact).")

# ==========================================
# 2. CHARGEMENT DES MODÈLES (En cache pour la vitesse)
# ==========================================
@st.cache_resource
def load_assets():
    try:
        # Assure-toi que ces 3 fichiers sont dans le dossier models/.
        model_dir = Path(__file__).resolve().parent / "models"
        model = joblib.load(model_dir / 'model_rumor_detector.pkl')
        tfidf = joblib.load(model_dir / 'tfidf_vectorizer.pkl')
        scaler = joblib.load(model_dir / 'feature_scaler.pkl')
        return model, tfidf, scaler
    except FileNotFoundError:
        st.error("Erreur : Fichiers modèles introuvables. Vérifiez que vous avez bien exporté les fichiers .pkl depuis votre Notebook.")
        st.stop()

model, tfidf, scaler = load_assets()

# ==========================================
# 3. FONCTIONS TECHNIQUES
# ==========================================
# Fonction mathématique pour l'épidémie (SIR)
def deriv_sir(y, t, N, beta, gamma):
    S, I, R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]

# Nettoyage de texte basique (doit correspondre à ce que tu as fait dans le Notebook)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|\@\w+', '', text)
    text = re.sub(r'\d+', 'numero', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ==========================================
# 4. INTERFACE UTILISATEUR (ONGLETS)
# ==========================================
tab1, tab2 = st.tabs(["🔍 Analyse Manuelle & Simulation", "📁 Analyse par lot (Import CSV)"])

# ------------------------------------------
# ONGLET 1 : ANALYSE MANUELLE
# ------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Saisie des Données")
        tweet_input = st.text_area("Contenu du tweet :", "BREAKING: Huge explosion near the central station! 50 dead!")
        
        st.subheader("📊 Métadonnées & Intelligence Collective")
        col_a, col_b = st.columns(2)
        with col_a:
            followers = st.number_input("Nombre de followers :", value=300, min_value=0)
            retweets = st.number_input("Nombre de retweets :", value=4500, min_value=0)
            reactions = st.number_input("Nombre de réponses :", value=150, min_value=0)
        with col_b:
            is_verified = st.toggle("Le compte est certifié ?")
            favorites = st.number_input("Nombre de favoris :", value=2000, min_value=0)
            doubt_score = st.number_input("Score de doute (ex: 15) :", value=15, min_value=0)

        analyze_button = st.button("Lancer l'Analyse", type="primary")

    if analyze_button:
        # Prétraitement
        text_cleaned = clean_text(tweet_input)
        text_tfidf = tfidf.transform([text_cleaned])
        num_features = np.array([[followers, int(is_verified), retweets, favorites, reactions, doubt_score]])
        num_scaled = scaler.transform(num_features)
        final_input = hstack([text_tfidf, num_scaled])
        
        # Prédiction ML
        prob = model.predict_proba(final_input)[0][1]
        is_rumor = prob > 0.5
        
        with col2:
            st.header("🔍 Résultats de l'IA")
            if is_rumor:
                st.error(f"🚨 ALERTE RUMEUR DÉTECTÉE (Probabilité : {prob*100:.1f}%)")
                st.write("Calcul de la propagation en cours...")
                
                # Modélisation Dynamique (SIR)
                beta = min(0.8, 0.1 + (retweets / 5000))
                gamma = min(0.5, 0.05 + (doubt_score / 20))
                
                N = 100000
                t = np.linspace(0, 48, 100) # 48 heures
                res = odeint(deriv_sir, [N-1, 1, 0], t, args=(N, beta, gamma))
                S, I, R = res.T
                
                pic_index = np.argmax(I)
                st.warning(f"💥 **Prévision de crise :** Pic de viralité estimé dans **{int(t[pic_index])} heures** avec **{int(I[pic_index])} propagateurs**.")
                
                # Affichage du graphique
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(t, I, 'r', linewidth=2, label='Propagateurs (Infectés)')
                ax.plot(t, R, 'g', linewidth=2, label='Réfractaires (Démentis/Oubli)')
                ax.fill_between(t, 0, I, color='red', alpha=0.1)
                ax.set_title("Modèle SIR : Impact de la rumeur sur 48h")
                ax.set_xlabel("Heures après publication")
                ax.set_ylabel("Utilisateurs impactés")
                ax.legend()
                st.pyplot(fig)
                
            else:
                st.success(f"✅ Information jugée FIABLE (Probabilité de rumeur : {prob*100:.1f}%)")
                st.info("Aucune modélisation de crise n'est nécessaire pour une information vérifiée.")

# ------------------------------------------
# ONGLET 2 : ANALYSE PAR LOT (CSV)
# ------------------------------------------
with tab2:
    st.header("📁 Modération de masse")
    st.write("Importez un fichier CSV contenant des tweets et leurs métadonnées pour analyser un événement complet.")
    
    # Uploader de fichier
    uploaded_file = st.file_uploader("Glissez votre fichier CSV ici", type="csv")
    
    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        
        # Vérification basique des colonnes requises
        colonnes_requises = ['text', 'followers_count', 'user_verified', 'retweet_count', 'favorite_count', 'reaction_count', 'doubt_score']
        if not all(col in df_new.columns for col in colonnes_requises):
            st.error(f"⚠️ Erreur de format. Le CSV doit contenir au minimum ces colonnes : {', '.join(colonnes_requises)}")
        else:
            st.success(f"Fichier chargé avec succès ! ({len(df_new)} tweets à analyser)")
            
            if st.button("Lancer le scan complet", type="primary"):
                with st.spinner('Le modèle Random Forest analyse les données...'):
                    
                    # 1. Pipeline NLP
                    df_new['clean_text'] = df_new['text'].apply(clean_text)
                    X_text_new = tfidf.transform(df_new['clean_text'])
                    
                    # 2. Pipeline Numérique
                    features_num = df_new[['followers_count', 'user_verified', 'retweet_count', 'favorite_count', 'reaction_count', 'doubt_score']]
                    X_num_new = scaler.transform(features_num)
                    
                    # 3. Fusion et Prédiction
                    X_final_new = hstack([X_text_new, X_num_new])
                    predictions = model.predict(X_final_new)
                    probabilites = model.predict_proba(X_final_new)[:, 1]
                    
                    # 4. Traitement des résultats
                    df_new['Statut IA'] = np.where(predictions == 1, '🚨 Rumeur', '✅ Fiable')
                    df_new['Certitude'] = np.round(probabilites * 100, 1).astype(str) + '%'
                    
                    # 5. Affichage propre
                    nb_rumeurs = sum(predictions == 1)
                    st.warning(f"Bilan de modération : {nb_rumeurs} rumeurs potentielles détectées sur {len(df_new)} tweets.")
                    
                    colonnes_a_afficher = ['text', 'Statut IA', 'Certitude', 'retweet_count', 'doubt_score']
                    st.dataframe(df_new[colonnes_a_afficher], use_container_width=True)