

import pandas as pd
import numpy as np
import json

def load_data():
    """
    Fonction permettant de charger le jeu données contenant les informations clients
    :return: dataframe contenant les infos clients
    """
    app_test = pd.read_csv('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv')
    return app_test

def features_engineering(data):
    """
    Fonction de preprocessing des données client
    :param data: dataframe des données du client
    :return: données préprocessées du client
    """
    data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
    data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    return data

def get_data(number_input1, data_input):
    """
    Fonction permettant de récupérer les données du client
    :param number_input1: n° client rentré par le chargé de clientèle
    :param data_input: base de données sur les infos clients
    :return: dataframe contenant uniquement les informations du client
    """
    data = data_input[data_input['SK_ID_CURR']==number_input1]
    data = data.drop(['SK_ID_CURR'], axis=1)
    return data

def load_num_metadata():
    """
    Fonction permettant de récupérer les valeurs des médianes de chaque feature pour imputation
    :return: un dictionnaire des valeurs médianes
    """
    num_imputation_file = open('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/imputation_file.json')
    data = pd.DataFrame(json.load(num_imputation_file))
    dict_median = data.set_index('Feature').to_dict()['Median']
    return dict_median

def load_cat_metadata():
    """
    Fonction permettant de récupérer les valeurs les plus fréquentes de chaque feature categ pour imputation
    :return: un dictionnaire des valeurs les plus fréquentes
    """
    cat_imputation_file = open('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/cat_imputation_file.json')
    data = pd.DataFrame(json.load(cat_imputation_file))
    dict_most_frequent = data.set_index('Feature').to_dict()['Most_frequent']
    return dict_most_frequent

def load_data_train():
    """
    Fonction permettant de charger le jeu données contenant les informations clients d'entrainement
    :return: dataframe contenant les infos clients
    """
    app_train = pd.read_csv('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/Projet+Mise+en+prod+-+home-credit-default-risk/application_train.csv')
    min_count = len(app_train.index) * 0.80
    app_train = app_train.dropna(axis='columns', thresh=min_count)
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    return app_train

def load_df_minmax():
    df_num_imp = pd.read_csv('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/Projet+Mise+en+prod+-+home-credit-default-risk/df_num_imp.csv')
    return df_num_imp