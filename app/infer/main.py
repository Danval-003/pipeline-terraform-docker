from fastapi import FastAPI, Body
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd
import threading

# Importa la función de prueba en TEST separada en tu librería
# (asegúrate de tener test_model en oxigen_pipeline.model como te pasé)
from oxigen_pipeline.model import test_model

app = FastAPI(title="Infer API")
MODEL_PATH = Path("/mnt/models/model.pkl")
REPORTS    = Path("/mnt/reports")
REPORTS.mkdir(parents=True, exist_ok=True)

_model = {"obj": None}
_lock = threading.Lock()

def _load_model():
    with _lock:
        if MODEL_PATH.exists():
            _model["obj"] = joblib.load(MODEL_PATH)

@app.on_event("startup")
def startup():
    _load_model()

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok" if _model["obj"] is not None else "model-not-loaded"

class PredictDTO(BaseModel):
    records: list[dict]  # filas estilo JSON -> DataFrame

@app.post("/predict")
def predict(payload: PredictDTO):
    if _model["obj"] is None:
        return {"error": "model not loaded"}
    df = pd.DataFrame(payload.records)
    yhat = _model["obj"].predict(df)
    return {"predictions": yhat.tolist()}

@app.post("/reload")
def reload_model():
    _load_model()
    return {"status": "reloaded", "loaded": _model["obj"] is not None}

# ===== Evaluación en TEST (separada del entrenamiento) =====
@app.post("/evaluate_test")
def evaluate_test(payload: dict = Body(...)):
    """
    Espera:
      - test_path: ruta parquet con el split de test (incluye target)
      - target_col: nombre de la columna objetivo
      - (opcional) shap_max_display: int
    Devuelve métricas y genera /mnt/reports/test_report.html
    """
    if _model["obj"] is None:
        return {"error": "model not loaded"}

    test_path = Path(payload["test_path"])
    target_col = payload.get("target_col")
    shap_max_display = int(payload.get("shap_max_display", 25))

    if not target_col:
        raise ValueError("Falta target_col en payload")

    df_te = pd.read_parquet(test_path)
    X_test, y_test = df_te.drop(columns=[target_col]), df_te[target_col]

    test_report = REPORTS / "test_report.html"
    metrics = test_model(
        _model["obj"],
        X_test, y_test,
        test_report_path=str(test_report),
        shap_max_display=shap_max_display,
        # background_X opcional: podrías pasar una muestra de train si la tienes montada
    )

    return {"status": "ok", "metrics": metrics, "test_report": str(test_report)}
