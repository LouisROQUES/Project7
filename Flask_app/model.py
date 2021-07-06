# Import dependencies
import pandas as pd
import numpy as np

# Load the dataset in a dataframe object
app_train = pd.read_csv('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/Projet+Mise+en+prod+-+home-credit-default-risk/application_train.csv')
app_test = pd.read_csv('/Users/louisroques/Desktop/Diplome Data Scientist/Projet 7 - Implémentez un modèle de scoring/Dataset/Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv')

# Data Preprocessing
def preprocessing(app_train, app_test):
    ## Ne garder que les variables contenant 80% de valeurs présentes
    min_count = len(app_train.index)*0.80
    app_train_filtered = app_train.dropna(axis='columns', thresh = min_count)
    ## Replace the anomalous values with nan
    app_train_filtered['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
    ## Feature engineering
    app_train_domain = app_train_filtered.copy()
    app_test_domain = app_test.copy()
    app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
    app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
    app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
    app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
    app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
    app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
    ## Alignement des df
    df_num_test = app_test_domain.select_dtypes(include=np.number)
    df_num_test = df_num_test.drop(['SK_ID_CURR'], axis=1)
    df_num_train = app_train_domain.select_dtypes(include=np.number)
    df_num_train = df_num_train.drop(['SK_ID_CURR'], axis=1)
    df_num_train, df_num_test = df_num_train.align(df_num_test, join='inner', axis=1)
    df_cat_test = app_test_domain.select_dtypes(exclude=np.number)
    df_cat_train = app_train_domain.select_dtypes(exclude=np.number)
    df_cat_train, df_cat_test = df_cat_train.align(df_cat_test, join='inner', axis=1)
    ## imputation des valeurs catégorielles
    for col in df_cat_train:
        df_cat_train[col] = df_cat_train[col].fillna(value=dict_most_frequent.get(col))
    for col in df_cat_test:
        df_cat_test[col] = df_cat_test[col].fillna(value=dict_most_frequent.get(col))
    ## imputation des valeurs numériques
    for col in df_num_train:
        df_num_train[col] = df_num_train[col].fillna(value=dict_median.get(col))
    for col in df_num_test:
        df_num_test[col] = df_num_test[col].fillna(value=dict_median.get(col))
    ## OneHotEncoder des variables catégorielles
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(df_cat_train)
    encoded_df_cat_train = pd.DataFrame(onehot_encoder.fit_transform(df_cat_train),
                                      columns=onehot_encoder.get_feature_names(df_cat_train.columns))
    encoded_df_cat_test = pd.DataFrame(onehot_encoder.fit_transform(df_cat_test),
                                      columns=onehot_encoder.get_feature_names(df_cat_test.columns))
    ## concaténation des df imputés
    df_train_imputed = pd.concat([df_num_train, encoded_df_cat_train], axis=1)
    df_test_imputed = pd.concat([df_num_test, encoded_df_cat_test], axis=1)
    ## alignement des df
    df_train_imputed, df_test_imputed = df_train_imputed.align(df_test_imputed, join='inner', axis=1)
    ## mise à l'échelle des df
    minmax_sc = MinMaxScaler(feature_range = (0, 1))
    std_df_train = minmax_sc.fit_transform(df_train_imputed)
    std_df_train = pd.DataFrame(std_df_train, columns = df_train_imputed.columns)
    std_df_test = minmax_sc.transform(df_test_imputed)
    std_df_test = pd.DataFrame(std_df_test, columns = df_test_imputed.columns)
    ## rajout des variables propres aux df
    std_df_train['TARGET'] = app_train_domain['TARGET']
    std_df_test['SK_ID_CURR'] = app_test_domain['SK_ID_CURR']
    return std_df_train


# XGBoost Classifier
x = std_df_train.drop(columns = ['TARGET'])
y = std_df_train['TARGET']
xgbc_model = GradientBoostingClassifier('subsample': 0.95, 'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'max_depth': 1, 'learning_rate': 0.1)
xgbc_model.fit(x, y)

# Save your model
joblib.dump(xgbc_model, 'xgbc_model.pkl')
print("Model dumped!")

# Load the model that you just saved
xgbc_model = joblib.load('xgbc_model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")