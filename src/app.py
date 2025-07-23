from fastapi.responses import FileResponse
from src.data_loader import preprocess_data
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Схема входных данных для heart disease
class HeartDiseaseFeatures(BaseModel):
    age: int
    sex: str
    cp: str
    trestbps: int
    chol: int
    fbs: bool
    restecg: str
    thalch: int
    exang: bool
    oldpeak: float
    slope: str
    ca: int
    thal: str

app = FastAPI(title="Heart Disease Type Predictor")

# Загружаем модель при старте
model = joblib.load("models/model.joblib")

@app.post("/predict")
def predict(data: HeartDiseaseFeatures):
    df_raw = pd.DataFrame([data.dict()])
    try:
        X, _ = preprocess_data(df_raw)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing error: {e}")
    X_aligned = X.reindex(columns=model.feature_names_in_, fill_value=0)
    try:
        pred = model.predict(X_aligned)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    return {"nums": int(pred)}

@app.get("/report")
def get_report():
    return FileResponse("predictions/report.html", media_type="text/html")
