import os
import joblib
import pandas as pd
from datetime import datetime
from src.data_loader import preprocess_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_disease_uci.csv')
PRED_PATH = 'predictions/predictions.csv'
REPORT_PATH = 'predictions/report.html'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

raw_data = pd.read_csv(DATA_PATH).head(5)
X, _ = preprocess_data(raw_data)
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

preds = model.predict(X)
raw_data['Num_Predicted'] = preds
raw_data.to_csv(PRED_PATH, index=False)
print(f"✅ Предсказания сохранены в {PRED_PATH}")

html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset=\"UTF-8\">
    <title>Отчёт о предсказаниях</title>
</head>
<body>
    <h1>❤️ Heart Disease Inference Report</h1>
    <p><strong>Дата:</strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Количество пациентов:</strong> {len(preds)}</p>
    <p><strong>Предсказания:</strong></p>
    {raw_data[['Num_Predicted']].to_html(index=False)}
</body>
</html>
"""

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"📄 Отчёт сохранён в {REPORT_PATH}")
