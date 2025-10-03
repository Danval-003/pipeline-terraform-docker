from fastapi import FastAPI, Body
from fastapi.responses import PlainTextResponse, HTMLResponse
from pathlib import Path
import pandas as pd
import joblib

# Importa las funciones NUEVAS separadas en tu librería
# (asegúrate de tener train_model en oxigen_pipeline.model como te pasé)
from oxigen_pipeline.model import train_model

app = FastAPI(title="Train API")
DATA_DIR = Path("/mnt/data")
MODELS   = Path("/mnt/models")
REPORTS  = Path("/mnt/reports")
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

VAL_REPORT = REPORTS / "val_report.html"
DIAGRAM    = REPORTS / "pipeline_model_diagram.html"
MODEL_OUT  = MODELS / "model.pkl"

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.get("/report", response_class=HTMLResponse)
def report():
    return VAL_REPORT.read_text(encoding="utf-8") if VAL_REPORT.exists() else "<p>Sin reporte</p>"

@app.post("/train")
def train(payload: dict = Body(...)):
    """
    Espera:
      - model_name: "XGBoost" | "RandomForest" | "LightGBM" | "GBM"
      - train_path, val_path (parquet con la columna target incluida)
      - target_col
    NO usa test aquí (queda para infer).
    """
    model_name = payload.get("model_name", "XGBoost")
    target_col = payload.get("target_col")
    if not target_col:
        raise ValueError("Falta target_col en payload")

    train_path = Path(payload["train_path"])
    val_path   = Path(payload["val_path"])

    df_tr = pd.read_parquet(train_path)
    df_va = pd.read_parquet(val_path)

    X_train, y_train = df_tr.drop(columns=[target_col]), df_tr[target_col]
    X_val,   y_val   = df_va.drop(columns=[target_col]), df_va[target_col]

    best_model, best_params, metrics = train_model(
        X_train, y_train, X_val, y_val,
        model_name=model_name,
        val_report_path=str(VAL_REPORT),
        diagram_path=str(DIAGRAM),
    )

    # Guardado atómico del pipeline completo (pre + modelo)
    tmp = MODELS / "model.pkl.tmp"
    joblib.dump(best_model, tmp)
    tmp.replace(MODEL_OUT)

    return {
        "status": "ok",
        "best_params": best_params,
        "metrics": metrics,                 # {"val_r2": ...}
        "val_report": str(VAL_REPORT),
        "diagram": str(DIAGRAM),
        "model_path": str(MODEL_OUT),
    }
