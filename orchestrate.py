import os, sys, time, json
from typing import Any, Dict, Optional
import requests

# URLs de los servicios (podés sobreescribir con variables de entorno)
PREPROC_URL = os.getenv("PREPROC_URL", "http://localhost:8080")
TRAIN_URL   = os.getenv("TRAIN_URL",   "http://localhost:8081")
INFER_URL   = os.getenv("INFER_URL",   "http://localhost:8082")

def _wait_healthy(url: str, path: str = "/health", timeout_s: int = 180, interval_s: float = 2.0):
    """Espera a que el endpoint /health devuelva ok/healthy."""
    health = f"{url.rstrip('/')}{path}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(health, timeout=3)
            if r.ok and r.text.strip().lower().startswith(("ok", "healthy", "model-not-loaded")):
                print(f"[ok] {health}")
                return
        except Exception:
            pass
        time.sleep(interval_s)
    raise TimeoutError(f"Service not healthy: {health}")

def _post_json(url: str, path: str, payload: Dict[str, Any], timeout_s: int = 1800) -> Dict[str, Any]:
    """POST JSON con manejo simple de errores y respuesta."""
    full = f"{url.rstrip('/')}{path}"
    r = requests.post(full, json=payload, timeout=timeout_s)
    try:
        r.raise_for_status()
    except Exception as e:
        body = r.text[:500].replace("\n", " ")
        raise RuntimeError(f"POST {full} -> {r.status_code} {body}") from e
    if r.headers.get("content-type", "").startswith("application/json"):
        return r.json()
    return {"raw": r.text}

def run_all(preproc_cfg: Dict[str, Any], *, model_name: str, target_col: str,
            predict_sample: Optional[Dict[str, Any]] = None, shap_max_display: int = 25):
    """
    Flujo:
      1) PREPROC /run -> genera train/val/test y devuelve paths
      2) TRAIN   /train (con train+val)
      3) INFER   /reload
      4) INFER   /evaluate_test (con test)
      5) (opc)   /predict (con sample)
    """
    # ========= 0) Salud =========
    print("== Esperando servicios ==")
    _wait_healthy(PREPROC_URL)
    _wait_healthy(TRAIN_URL)
    _wait_healthy(INFER_URL)

    # ========= 1) Preprocesamiento =========
    print("== 1) Preprocesamiento ==")
    
    preproc_cfg.setdefault("test_size", 0.20)
    preproc_cfg.setdefault("val_size",  0.20)
    preproc_cfg.setdefault("random_state", 42)
    print(f"[preproc] cfg={preproc_cfg} url={PREPROC_URL}")
    pre = _post_json(PREPROC_URL, "/run", preproc_cfg)
    print(f"[preproc] reporte={pre.get('report')}")
    paths = pre.get("paths") or {}
    train_path = paths.get("train_path")
    val_path   = paths.get("val_path")
    test_path  = paths.get("test_path")
    if not (train_path and val_path and test_path):
        raise RuntimeError(f"Preproc no devolvió paths completos: {paths}")

    # ========= 2) Entrenamiento (solo train+val) =========
    print("== 2) Entrenamiento (train+val) ==")
    tr_payload = {
        "model_name": model_name,
        "target_col": target_col,
        "train_path": train_path,
        "val_path":   val_path,
    }
    tr = _post_json(TRAIN_URL, "/train", tr_payload)
    print(f"[train] val_metrics={tr.get('metrics')} model={tr.get('model_path')}")
    print(f"[train] val_report={tr.get('val_report')} diagram={tr.get('diagram')}")

    # ========= 3) Recarga de modelo en infer =========
    print("== 3) Recarga en infer ==")
    _post_json(INFER_URL, "/reload", {})
    print("[infer] reload ok")

    # ========= 4) Evaluación en TEST desde infer =========
    print("== 4) Evaluación en TEST ==")
    ev_payload = {"test_path": test_path, "target_col": target_col, "shap_max_display": shap_max_display}
    ev = _post_json(INFER_URL, "/evaluate_test", ev_payload)
    print(f"[infer/evaluate_test] test_metrics={ev.get('metrics')} test_report={ev.get('test_report')}")

    # ========= 5) (Opcional) Predicción con muestra nueva =========
    if predict_sample:
        print("== 5) /predict (input nuevo) ==")
        res = _post_json(INFER_URL, "/predict", {"records": [predict_sample]})
        print(f"[infer/predict] {res}")

    # Resumen final útil
    return {
        "train": {
            "val_metrics": tr.get("metrics"),
            "val_report": tr.get("val_report"),
            "diagram": tr.get("diagram"),
            "model_path": tr.get("model_path"),
        },
        "test": {
            "metrics": ev.get("metrics"),
            "test_report": ev.get("test_report"),
        },
        "paths": paths,
    }

if __name__ == "__main__":
    # Defaults de ejemplo: ajustá según tu PipelineConfig real
    # Se pueden sobreescribir con archivos/argv.
    preproc_cfg = {
        "input_path": "/mnt/data/raw.csv",
        "target_col": "AQI"
    }
    model_name = "XGBoost"
    target_col = "AQI"
    sample = None  # p.ej. {"f1": 1.0, "f2": "A"}

    # CLI:
    #   python orchestrate.py [preproc.json] [model_name] [target_col] [sample.json]
    if len(sys.argv) >= 2:
        preproc_cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    if len(sys.argv) >= 3:
        model_name = sys.argv[2]
    if len(sys.argv) >= 4:
        target_col = sys.argv[3]
    if len(sys.argv) >= 5:
        sample = json.load(open(sys.argv[4], "r", encoding="utf-8"))

    out = run_all(preproc_cfg, model_name=model_name, target_col=target_col, predict_sample=sample)
    print("\n== RESUMEN ==")
    print(json.dumps(out, indent=2))
