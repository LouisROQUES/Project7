
import joblib

def load_model():
    """
    Fonction permettant de charger le modèle de prédiction
    :return: modèle
    """
    xgbc_model = joblib.load('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/gbc_model.pkl')
    return xgbc_model

def predict_prob(X_test):
    """
    Cette fonction sert à prédire ...
    :param X_test: le vecteur de données pour un individu (liste)
    :return: prédiction de type float
    """
    xgbc_model = joblib.load('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/gbc_model.pkl')
    pred = xgbc_model.predict_proba(X_test)[:,1]
    return pred

