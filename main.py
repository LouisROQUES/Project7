### PACKAGES
import streamlit as st
from PIL import Image


st.title('Dashboard prédiction non solvabilité de prêt')
st.subheader('par ROQUES Louis dans le cadre du diplome de Data Scientist')

st.markdown(
    """
    Cette application permet de déterminer un score de probabilité de non solvabilité d'un prêt pour un client.
    Des explications sont apportées par des visualisations pour la compréhension de ce score.
    """
)

image = Image.open('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/photo.jpg')
st.image(image)

image2 = Image.open('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/logo.png')
st.sidebar.image(image2)

client_input = st.number_input('Veuillez renseigner le numéro client SK_ID_CURR', step=1)

# bouton soumettre pour récupérer récuprer l'id client et récupérer le data
if st.button('Soumettre id client'):
    result = client_input
    st.success(result)

    # chargement de la base de données client brute
    from Loading_data.Loading_data import load_data
    raw_data = load_data()

    # chargement de la base de données client preprocessée
    from Loading_data.Loading_data import load_data_preprocess
    data = load_data_preprocess()

    # chargement des informations client brutes
    from Loading_data.Loading_data import get_raw_data
    raw_data_client = get_raw_data(result, raw_data)

    # chargement des informations client preprocessées
    from Loading_data.Loading_data import get_data
    data_client = get_data(result, data)

    # visualisation des principales informations client
    from Visualisation.Visualisation import main_informations

    st.sidebar.header('Informations générales')
    main_informations = main_informations(raw_data_client)
    st.sidebar.write(main_informations)

    # afficher predict_proba (loader modèle puis predict, return predict) => retourner explicabilité par exemple features importance da
    # cette fonction permet de charger le modèle et d'effectuer la prediction du score pour l'octroie de prêt
    from Models.Model import predict_prob
    prediction = predict_prob(data_client)

    # visualisation du score sur une jauge
    from Visualisation.Visualisation import score_vis
    score_vis = score_vis(prediction)
    st.plotly_chart(score_vis)

    #  retourner explicabilité par exemple features importance au niveau prédictione tester package shap ou lime
    # chargement du modèle visualisation
    from Models.Model import load_model
    xgbc_model = load_model()

    # création des shap value pour explication du score
    from Visualisation.Visualisation import shap_explaner
    st.header('Explicabilité du score')
    explainer = shap_explaner(xgbc_model, data_client)
    st.pyplot(explainer)

    # chargement du jeu de données d'entrainement du modèle
    from Loading_data.Loading_data import load_train_set
    train_set = load_train_set()

    # récupération des features importances pour visualisation histogramme
    from Visualisation.Visualisation import features_importances
    most_important_features = features_importances(xgbc_model, train_set)

    # affichage des données client pour les features les plus importantes
    from Visualisation.Visualisation import main_features_informations
    st.sidebar.header('Informations client les plus importantes')
    main_features_informations = main_features_informations(raw_data_client, most_important_features)
    st.sidebar.write(main_features_informations)

    # chargement du data train
    from Loading_data.Loading_data import load_data_train
    data_train = load_data_train()

    # histogrammes des features les plus importantes
    from Visualisation.Visualisation import hist_input
    st.header('Visualisation des informations les plus importantes')
    ax1, ax2, ax3, ax4, ax5 = hist_input(data_train, most_important_features)
    for fig in [ax1, ax2, ax3, ax4, ax5]:
        st.pyplot(fig)
