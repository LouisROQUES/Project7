import pandas as pd
import numpy as np


def load_data():
    """
    Fonction permettant de charger le jeu données contenant les informations clients
    :return: dataframe contenant les infos clients
    """
    raw_data = pd.read_csv('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv')
    raw_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    raw_data['CREDIT_INCOME_PERCENT'] = raw_data['AMT_CREDIT'] / raw_data['AMT_INCOME_TOTAL']
    raw_data['ANNUITY_INCOME_PERCENT'] = raw_data['AMT_ANNUITY'] / raw_data['AMT_INCOME_TOTAL']
    raw_data['CREDIT_TERM'] = raw_data['AMT_ANNUITY'] / raw_data['AMT_CREDIT']
    raw_data['DAYS_EMPLOYED_PERCENT'] = raw_data['DAYS_EMPLOYED'] / raw_data['DAYS_BIRTH']
    return raw_data

def load_data_preprocess():
    """
    Fonction permettant de charger le jeu données contenant les informations clients preprocessées
    :return: dataframe contenant les infos clients préprocessées
    """
    data = pd.read_csv('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/application_test_preprocessed.csv')
    return data

def get_data(number_input1, data_input):
    """
    Fonction permettant de récupérer les données du client brutes
    :param number_input1: n° client rentré par le chargé de clientèle
    :param data_input: base de données sur les infos clients
    :return: dataframe contenant uniquement les informations du client
    """
    data = data_input[data_input['SK_ID_CURR']==number_input1]
    data = data.drop(['SK_ID_CURR'], axis=1)
    return data

def get_raw_data(number_input1, raw_data_input):
    """
    Fonction permettant de récupérer les données du client préprocessées
    :param number_input1: n° client rentré par le chargé de clientèle
    :param data_input: base de données sur les infos clients
    :return: dataframe contenant uniquement les informations du client préprocessées
    """
    data = raw_data_input[raw_data_input['SK_ID_CURR']==number_input1]
    data = data.drop(['SK_ID_CURR'], axis=1)
    return data

def load_data_train():
    """
    Fonction permettant de charger le jeu données contenant les informations clients d'entrainement
    :return: dataframe contenant les infos clients
    """
    app_train = pd.read_csv('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/Projet+Mise+en+prod+-+home-credit-default-risk/application_train.csv')
    min_count = len(app_train.index) * 0.80
    app_train = app_train.dropna(axis='columns', thresh=min_count)
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    app_train['CREDIT_INCOME_PERCENT'] = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
    app_train['ANNUITY_INCOME_PERCENT'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
    app_train['CREDIT_TERM'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
    app_train['DAYS_EMPLOYED_PERCENT'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']
    return app_train

def load_train_set():
    """
    Fonction permettant de charger le jeu de données d'entrainement du modèle pour visualisation du feature importance
    :return: dataframe contenant le jeu de données d'entrainement du modèle
    """
    train_set = pd.read_csv('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/train_set.csv')
    return train_set