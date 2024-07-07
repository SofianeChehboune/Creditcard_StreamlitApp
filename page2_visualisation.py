import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display():
    # URL du fichier Dropbox
    dropbox_url = "https://www.dropbox.com/scl/fi/95kkevx66y0teeyrqrfnp/creditcard.csv?rlkey=svlt3izx0v6qntwca7afrvop1&st=rfbftb3m&dl=1"

    # Chargement des données
    @st.cache_data
    def load_data(url):
        return pd.read_csv(url)

    card_df = load_data(dropbox_url)

    # Visualisation des données
    st.header("Visualisation des données")
    st.write("Distribution des classes")
    st.write(card_df['Class'].value_counts())

    st.write("Visualisation des transactions")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=card_df, ax=ax)
    st.pyplot(fig)
