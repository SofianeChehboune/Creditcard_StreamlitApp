import streamlit as st

# Titre de l'application
st.title("Analyse et Détection de Fraude par Carte de Crédit")

# Navigation
pages = ["Filtrage des données", "Visualisation des données", "Machine Learning - Random Forest"]
page = st.sidebar.selectbox("Choisissez une page", pages)

if page == "Filtrage des données":
    import page1_filtrage as page1
    page1.display()
elif page == "Visualisation des données":
    import page2_visualisation as page2
    page2.display()
elif page == "Machine Learning - Random Forest":
    import page3_ml as page3
    page3.display()
