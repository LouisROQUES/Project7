import shap
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd



def score_vis(prediction):
    """
    Fonction permettant de visualiser le predict_proba sous forme de jauge
    :param prediction: predict proba du model
    :return: gauge chart
    """
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = prediction[0],
        mode = "gauge+number+delta",
        title = {'text': "Score de non solvabilité"},
        gauge = {'axis': {'range': [None, 1]},
                 'bar': {'color': "black"},
                 'steps' : [
                     {'range': [0, 0.4], 'color': "green"},
                     {'range': [0.4, 0.6], 'color': "orange"},
                     {'range': [0.6, 1], 'color': "red"}],
                 'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.99}}))
    return fig

def shap_explaner(my_model, data_processed):
    """
    Fonction permettant de générer des shap values et une visualisation pour l'explicabilité des résultats
    :param my_model: model de prédiction
    :param data_for_prediction: données clients entrant dans le model
    :return: graphique des shap values
    """
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(my_model)
    # Calculate Shap values
    shap_values = explainer.shap_values(data_processed)
    fig = plt.figure()
    shap.summary_plot(shap_values, data_processed)
    return fig

def features_importances(my_model, train_set):
    """
    Fonction permettant de générer les 5 features les plus importantes du model
    :param my_model: model de prediction
    :return: liste des 5 features les plus importantes
    """
    importances = my_model.feature_importances_
    feature_names = list(train_set.head(0))
    feature_importances = pd.Series(importances, index=feature_names)
    fi_model = pd.DataFrame(feature_importances)
    fi_model = fi_model.sort_values(by=0, ascending=False)
    most_important_features = list(fi_model[:5].index)
    return most_important_features

def hist_input(data, most_important_features):
    """
    Fonction permettant de générer les histogrammes des 5 features les plus importantes
    :param data: base de données d'entrainement
    :param most_important_features: liste des features les plus importantes
    :return: histogrammes des features les plus importantes
    """
    ax1 = sns.displot(data, x=data[most_important_features[0]], hue=data["TARGET"], element="step")
    ax2 = sns.displot(data, x=data[most_important_features[1]], hue=data["TARGET"], element="step")
    ax3 = sns.displot(data, x=data[most_important_features[2]], hue=data["TARGET"], element="step")
    ax4 = sns.displot(data, x=data[most_important_features[3]], hue=data["TARGET"], element="step")
    ax5 = sns.displot(data, x=data[most_important_features[4]], hue=data["TARGET"], element="step")
    return  ax1, ax2, ax3, ax4, ax5

def main_informations(raw_data):
    """
    Fonction permettant d'afficher certaines informations sur le client
    :return: dataframe contenant les principale informations client
    """
    main_informations = raw_data[['CODE_GENDER', 'CNT_CHILDREN', 'DAYS_EMPLOYED',
                                  'DAYS_BIRTH','FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                                  'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'
                                  ]]
    main_informations = main_informations.T
    main_informations.columns = ['Informations']
    return main_informations

def main_features_informations(raw_data, most_important_features):
    """
    Fonction permettant d'afficher certaines informations sur le client
    :return: dataframe contenant les principale informations client
    """
    main_features_informations = raw_data[most_important_features]
    main_features_informations = main_features_informations.T
    main_features_informations.columns = ['Informations clé']
    return main_features_informations