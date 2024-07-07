import streamlit as st
import pandas as pd

def display():
    # URL du fichier Dropbox
    dropbox_url = "https://www.dropbox.com/scl/fi/95kkevx66y0teeyrqrfnp/creditcard.csv?rlkey=svlt3izx0v6qntwca7afrvop1&st=rfbftb3m&dl=1"

    # Chargement des données
    @st.cache_data
    def load_data(url):
        return pd.read_csv(url)

    card_df = load_data(dropbox_url)

    # Filtrage des données
    st.header("Filtrage des données")
    st.write("Données de tête")
    st.write(card_df.head())
    # Description des données 
    st.header("Description des donnéess")
    st.write("Description")
    st.write(card_df.describe())

    # Vérification des valeurs manquantes
    st.header("Vérification des valeurs manquantes")
    st.write("Vérification")
    st.write(card_df.isnull().sum())
