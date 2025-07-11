import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
st.set_page_config(page_title="Émotions App", layout="wide")
st.title("Anlayse des émotions")
st.markdown("""Bienvenue dans cette application d'anlayse des émotions. Saisissez le texte, choisis une méthode et applique l'analyse des émotions.""")

with st.container():
    method = st.selectbox("Selectionnez un modèle", ["Bert", "Bert Lora", "Transformer"])
    text = st.text_area("Saisissez le texte")
    compute_btn = st.button("Analyser le texte")

    if text is not None and compute_btn:
        with st.spinner("Analysing..."):
            payload = {
                "text": text,
                "method": method
            }
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"L'émotion dans ce texte est : **{result['emotion']}**")
                else:
                    st.error(f"Erreur API : {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Impossible de se connecter à l'API.")
