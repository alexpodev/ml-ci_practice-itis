from src.data_loader import load_sample_data

DATA_PATH = 'data/heart_disease_uci.csv'

def test_data_loading_from_local_csv():
    X_train, X_test, y_train, y_test = load_sample_data(DATA_PATH)

    # Проверка, что данные загружены и не пусты
    assert X_train.shape[0] > 0, "Тренировочная выборка пуста"
    assert X_test.shape[0] > 0, "Тестовая выборка пуста"
    assert X_train.shape[1] == X_test.shape[1], (
        "Количество признаков в train и test не совпадает"
    )
    assert len(y_train) == X_train.shape[0], (
        "Размерности X_train и y_train не совпадают"
    )
    assert len(y_test) == X_test.shape[0], (
        "Размерности X_test и y_test не совпадают"
    )
    # Проверим, что целевая переменная не пустая
    assert y_train.notnull().all(), "В y_train есть пропуски"
    assert y_test.notnull().all(), "В y_test есть пропуски"

    print(
        f"✅ Загружено: {X_train.shape[0]} train, "
        f"{X_test.shape[0]} test, {X_train.shape[1]} признаков"
    )
