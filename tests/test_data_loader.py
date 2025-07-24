import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from data_loader import preprocess_data

def test_preprocess_data():
    data = {
        'id': [1, 2],
        'age': [63, 67],
        'sex': ['Male', 'Female'],
        'dataset': ['Cleveland', 'Cleveland'],
        'cp': ['typical angina', 'asymptomatic'],
        'trestbps': [145, 160],
        'chol': [233, 286],
        'fbs': [True, False],
        'restecg': ['lv hypertrophy', 'lv hypertrophy'],
        'thalch': [150, 108],
        'exang': [False, True],
        'oldpeak': [2.3, 1.5],
        'slope': ['downsloping', 'flat'],
        'ca': [0, 3],
        'thal': ['fixed defect', 'normal'],
        'num': [0, 2]
    }

    df = pd.DataFrame(data)
    X, y = preprocess_data(df)

    # Проверим форму
    assert X.shape[0] == 2
    assert y.tolist() == [0, 2]
    # Проверим, что булевые поля действительно булевые
    assert X['fbs'].dtype == bool
    assert X['exang'].dtype == bool
    # Проверим, что категориальные признаки не пустые
    for col in ['sex', 'cp', 'restecg', 'slope', 'thal', 'dataset']:
        assert col in X.columns
        assert X[col].isnull().sum() == 0
    # Проверим, что колонка 'id' удалена
    assert 'id' not in X.columns


def test_preprocess_data_with_nans():
    data = {
        'id': [1, 2, 3],
        'age': [63, None, 55],
        'sex': ['Male', None, 'Female'],
        'dataset': ['Cleveland', 'Cleveland', None],
        'cp': ['typical angina', 'asymptomatic', None],
        'trestbps': [145, 160, None],
        'chol': [233, None, 210],
        'fbs': [True, None, False],
        'restecg': ['lv hypertrophy', None, 'normal'],
        'thalch': [150, 108, None],
        'exang': [False, True, None],
        'oldpeak': [2.3, None, 1.2],
        'slope': ['downsloping', None, 'flat'],
        'ca': [0, 3, None],
        'thal': ['fixed defect', None, 'normal'],
        'num': [0, 2, 1]
    }
    df = pd.DataFrame(data)
    X, y = preprocess_data(df)

    # Проверка: все строки на месте
    assert X.shape[0] == 3
    assert y.tolist() == [0, 2, 1]
    # Проверка: нет NaN в числовых признаках
    num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    for col in num_cols:
        assert X[col].isnull().sum() == 0, f"NaN в числовом признаке {col}"
    # Проверка: нет NaN в категориальных признаках
    cat_cols = ['sex', 'cp', 'restecg', 'slope', 'thal', 'dataset']
    for col in cat_cols:
        assert X[col].isnull().sum() == 0, f"NaN в категориальном признаке {col}"
    # Проверка: булевые признаки действительно булевые
    assert X['fbs'].dtype == bool
    assert X['exang'].dtype == bool
    # Проверка: не появилось лишних признаков
    expected_cols = set(num_cols + cat_cols + ['fbs', 'exang'])
    assert set(X.columns) == expected_cols, f"Лишние признаки: {set(X.columns) - expected_cols}"
    # Проверка: целевая переменная не изменилась
    assert y.equals(df['num'])
