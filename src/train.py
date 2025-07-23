import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from src.data_loader import load_sample_data, load_data
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Пути
DATA_PATH = '../data/heart_disease_uci.csv' if __name__ == '__main__' else 'data/heart_disease_uci.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')

# Включить бинарную классификацию?
BINARY_MODE = False  # False — мультикласс, True — 0 vs 1+

os.makedirs(MODEL_DIR, exist_ok=True)

# Анализ распределения классов
raw_df = load_data(DATA_PATH)
if BINARY_MODE:
    y = (raw_df['num'] > 0).astype(int)
    print('Class distribution (binary):')
    print(y.value_counts())
else:
    print('Class distribution (num):')
    print(raw_df['num'].value_counts())

# Загрузка и препроцессинг
if BINARY_MODE:
    X = raw_df.drop(columns=['id', 'num'], errors='ignore')
    y = (raw_df['num'] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    X_train, X_test, y_train, y_test = load_sample_data(DATA_PATH, for_catboost=True)

cat_features = ['sex', 'cp', 'restecg', 'slope', 'thal', 'dataset']
num_features = [col for col in X_train.columns if col not in cat_features]

# Заполняем NaN
for cat in cat_features:
    if cat in X_train.columns:
        X_train[cat] = X_train[cat].fillna('unknown').astype(str)
        X_test[cat] = X_test[cat].fillna('unknown').astype(str)
for num in num_features:
    if num in X_train.columns:
        X_train[num] = X_train[num].fillna(X_train[num].median())
        X_test[num] = X_test[num].fillna(X_train[num].median())

# Препроцессинг: масштабирование числовых + one-hot для категориальных
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(class_weight='balanced', random_state=42))
])

model_params = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [10, 20],
}

grid_search = GridSearchCV(pipeline, model_params, cv=5, scoring='accuracy', error_score='raise', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print('Best params:', grid_search.best_params_)

# Предсказания и метрики
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'✅ Accuracy: {accuracy:.4f}')
print('F1-score (macro):', f1_score(y_test, y_pred, average='macro'))
print('F1-score (micro):', f1_score(y_test, y_pred, average='micro'))
print('F1-score (weighted):', f1_score(y_test, y_pred, average='weighted'))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))

# Сохраняем пайплайн
joblib.dump(best_model, MODEL_PATH)
print(f'✅ Model saved to {MODEL_PATH}')

# Важность признаков (через feature_importances_ модели внутри пайплайна)
importances = best_model.named_steps['model'].feature_importances_
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
print('Feature importances:')
for name, score in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f'{name}: {score:.2f}')
