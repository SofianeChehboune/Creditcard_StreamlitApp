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
    # page1_filtrage
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
# page2_visualisation
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
# page3_ml
    import streamlit as st
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    def display():
    # URL du fichier Dropbox
    dropbox_url = "https://www.dropbox.com/scl/fi/95kkevx66y0teeyrqrfnp/creditcard.csv?rlkey=svlt3izx0v6qntwca7afrvop1&st=rfbftb3m&dl=1"

    # Chargement des données
    @st.cache_data
    def load_data(url):
        return pd.read_csv(url)

    card_df = load_data(dropbox_url)

    # Machine Learning avec Random Forest
    st.header("Machine Learning - Random Forest")

    # Séparation des données en caractéristiques et cible
    X = card_df.drop('Class', axis=1).dropna()
    y = card_df['Class'].loc[X.index]

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=101)
    model.fit(X_train, y_train)

    # Évaluation du modèle
    y_pred = model.predict(X_test)
    st.write("Rapport de classification")
    st.text(classification_report(y_test, y_pred))

    # Calcul et affichage du score ROC-AUC
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    st.write("Score ROC-AUC")
    st.write(f"ROC-AUC Score: {roc_auc:.4f}")

    # Visualisation de la courbe ROC
    st.write("Courbe ROC")
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)



