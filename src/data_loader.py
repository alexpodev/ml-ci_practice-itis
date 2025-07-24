import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


def load_data(path: str):
    """Загружает данные из CSV и возвращает DataFrame."""
    return pd.read_csv(path)


def impute_categorical_missing_data(df, passed_col, missing_data_cols, bool_cols):
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]
    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    label_encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col].astype(str))
    # Ключевая правка:
    if passed_col in bool_cols:
        y = y.astype(int)
    else:
        y = y.astype(str)
    y = y.fillna('unknown')
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)
    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"The feature '{passed_col}' has been imputed with {round((acc_score * 100), 2)}% accuracy\n")
    X_null = df_null.drop(passed_col, axis=1)
    for col in X_null.columns:
        if X_null[col].dtype == 'object' or X_null[col].dtype == 'category':
            X_null[col] = label_encoder.fit_transform(X_null[col].astype(str))
    for col in other_missing_cols:
        if X_null[col].isnull().sum() > 0:
            col_with_missing_values = X_null[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X_null[col] = imputed_values[:, 0]
    if len(df_null) > 0:
        df_null = df_null.copy()
        pred = rf_classifier.predict(X_null)
        if passed_col in bool_cols:
            df_null.loc[:, passed_col] = pred.astype(bool)
        else:
            df_null.loc[:, passed_col] = pred
    df_combined = pd.concat([df_not_null, df_null])
    return df_combined[passed_col]


def impute_continuous_missing_data(df, passed_col, missing_data_cols):
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]
    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    label_encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col].astype(str))
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)
    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)
    print("MAE =", mean_absolute_error(y_test, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_test, y_pred), "\n")
    print("R2 =", r2_score(y_test, y_pred), "\n")
    X_null = df_null.drop(passed_col, axis=1)
    for col in X_null.columns:
        if X_null[col].dtype == 'object' or X_null[col].dtype == 'category':
            X_null[col] = label_encoder.fit_transform(X_null[col].astype(str))
    for col in other_missing_cols:
        if X_null[col].isnull().sum() > 0:
            col_with_missing_values = X_null[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X_null[col] = imputed_values[:, 0]
    if len(df_null) > 0:
        df_null = df_null.copy()
        df_null.loc[:, passed_col] = rf_regressor.predict(X_null)
    df_combined = pd.concat([df_not_null, df_null])
    return df_combined[passed_col]


def advanced_impute(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Не трогаем целевую переменную
    target_cols = ['num', 'nums']
    missing_data_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and col not in target_cols]
    for col in missing_data_cols:
        print(f"Missing Values {col}: {round((df[col].isnull().sum() / len(df)) * 100, 2)}%")
        if col in bool_cols or col in categorical_cols:
            df[col] = impute_categorical_missing_data(df, col, missing_data_cols, bool_cols)
        elif col in numeric_cols and col not in bool_cols:
            df[col] = impute_continuous_missing_data(df, col, missing_data_cols)
    return df


def preprocess_data(df: pd.DataFrame, for_catboost=True):
    """Предобрабатывает DataFrame.
    При тренировке возвращает (X, y), при inference — (X, None)."""
    df = df.copy()
    df = advanced_impute(df)
    # Категориальные признаки
    cat_cols = ['sex', 'cp', 'restecg', 'slope', 'thal', 'dataset']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    # Булевы признаки
    for col in ['fbs', 'exang']:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    # Удаляем неиспользуемые
    df.drop(columns=['id'], inplace=True, errors='ignore')
    
    # Целевая переменная
    if 'num' in df.columns:
        y = df['num']
        X = df.drop(columns=['num'])
    elif 'nums' in df.columns:
        y = df['nums']
        X = df.drop(columns=['nums'])
    else:
        X = df
        y = None
    return X, y


def load_and_preprocess(path: str, for_catboost=True):
    df = load_data(path)
    return preprocess_data(df, for_catboost=for_catboost)


def load_sample_data(path: str, test_size=0.2, random_state=42, for_catboost=True):
    """Загружает и делит данные на обучающую и тестовую выборки."""
    X, y = load_and_preprocess(path, for_catboost=for_catboost)
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
