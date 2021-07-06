
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def align_data(data_train, data_test):
    data_train, data_test = data_train.align(data_test, join='inner', axis=1)
    return data_test

def preprocess (data, df_num_model, dict_median, dict_most_frequent):
    """
    Cette fonction permet le preprocessing des données avant modélisation
    :param data: données du client pouvant contenir des données manquantes
    :param dict_median: médianes des features numériques
    :param dict_most_frequent: most_frequent value des features catégorielles
    :return: dataframe imputé et features mises à l'échelle
    """
    # création d'un df avec variables numériques
    df_num = data.select_dtypes(include=np.number)
    # imputation par la médiane des valeurs manquantes
    for col in df_num:
        df_num[col] = df_num[col].fillna(value=dict_median.get(col))
    # scaling avec MinMaxScaler
    minmax_sc = MinMaxScaler(feature_range=(0, 1))
    minmax_sc = minmax_sc.fit(df_num_model)
    std_df_num = minmax_sc.transform(df_num)
    std_df_num = pd.DataFrame(std_df_num, columns=df_num.columns)
    std_df_num

    # création d'un df avec variables catégorielles
    df_cat =data.select_dtypes(exclude=np.number)
    # imputation par le string le plus frequent
    for col in df_cat:
        df_cat[col] = df_cat[col].fillna(value=dict_most_frequent.get(col))
    # encodage des variables catégorielles
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded_df_cat = pd.DataFrame(onehot_encoder.fit_transform(df_cat),
                                  columns=onehot_encoder.get_feature_names(df_cat.columns))
    encoded_df_cat
    data_preprocess = pd.concat([std_df_num, encoded_df_cat], axis=1)
    return data_preprocess


