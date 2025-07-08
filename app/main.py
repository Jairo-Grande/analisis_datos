from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import os
from typing import Optional
from fastapi import Query

LOG_DIR = os.path.dirname(__file__)
#PATH PARA LAS PREDICCIONES.
LOG_PATH = os.path.join(os.path.dirname(__file__), "predictions_log.csv")

# Cargar el modelo
model = joblib.load('app/modelo_xgb_optimizado.pkl')



app = FastAPI()

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: CustomerData):
    return predict_churn(customer.dict())


def predict_churn(data: dict):
    # Convertir dict a DataFrame (una fila)
    df = pd.DataFrame([data])
    
    # Hacer predicción
    proba = model.predict_proba(df)[0][1]  # probabilidad de churn (clase 1)
    pred = model.predict(df)[0]           # clase predicha: 0 o 1
    
    # Resultado
    result = {
        "churn_probability": float(round(proba, 4)),
        "prediction": int(pred)
    }

    # Guardar la predicción con los datos
    save_prediction(data, result)

    return result


@app.get("/sample")
def get_sample_input():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1025.3
    }


@app.post("/batch-predict")
def batch_predict(data: list[CustomerData]):
    results = [predict_churn(item.dict()) for item in data]
    return {"results": results}


def save_prediction(data: dict, result: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    row = {
        **data,
        "prediction": result["prediction"],
        "churn_probability": result["churn_probability"],
        "timestamp": datetime.utcnow().isoformat()
    }

    df_row = pd.DataFrame([row])

    if not os.path.exists(LOG_PATH):
        df_row.to_csv(LOG_PATH, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode='a', header=False, index=False)


#HISTORICO DE MODIFICADOS. clientes que ya se les hizo la prediccion.
@app.get("/history/filter")
def filter_history(
    prediction: Optional[int] = Query(None, ge=0, le=1),
    min_probability: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_probability: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    return filter_prediction_history(prediction, min_probability, max_probability)

def filter_prediction_history(
    prediction: int = None,
    min_probability: float = None,
    max_probability: float = None
):
    if not os.path.exists(LOG_PATH):
        return []

    df = pd.read_csv(LOG_PATH)

    # Aplicar filtros si existen
    if prediction is not None:
        df = df[df["prediction"] == prediction]

    if min_probability is not None:
        df = df[df["churn_probability"] >= min_probability]

    if max_probability is not None:
        df = df[df["churn_probability"] <= max_probability]

    return df.to_dict(orient="records")