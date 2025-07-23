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
