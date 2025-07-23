# Heart Disease CI/CD

ML pipeline for heart disease type prediction with automated CI/CD using GitHub Actions.

## 🚀 Возможности

- Обработка данных и обучение модели
- Модульные тесты через `pytest`
- Автоматическая проверка кода (`flake8`)
- CI-пайплайн на GitHub Actions

## 📦 Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## 🏗️ Инициализация структуры проекта

```bash
mkdir -p data/raw notebooks src tests .github/workflows
touch data/.gitkeep
```

```
heart_disease-cicd/
├── data/                  # Данные (heart_disease_uci.csv и т.д.)
├── notebooks/             # Jupyter ноутбуки
├── src/                   # Исходный код (data_loader, model и т.д.)
├── tests/                 # Модульные тесты (pytest)
├── .github/workflows/     # CI pipeline (GitHub Actions)
├── requirements.txt       # Зависимости
└── README.md              # Описание проекта
```
## 🧪 Запуск тестов
```bash
pytest tests/
```
## 🧹 Проверка стиля кода
```bash
flake8 src/ tests/
```
## 📡 CI/CD
GitHub Actions автоматически запускает тесты и линтер при каждом push. См. `.github/workflows/ci.yml`.

