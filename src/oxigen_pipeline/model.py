import io, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone, estimator_html_repr
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# =========================
# Helpers
# =========================

def _detect_feature_types(X: pd.DataFrame):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num, cat

def _make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = _detect_feature_types(X_train)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.2
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", ohe, categorical_features),
    ])
    return pre

def _get_model_and_space(model_name: str):
    spaces = {
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
            }
        ),
        "XGBoost": (
            XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, tree_method="hist"),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [3, 6, 10],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            }
        ),
        "LightGBM": (
            LGBMRegressor(random_state=42),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [-1, 10, 20],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__num_leaves": [31, 50, 100],
            }
        ),
        "GBM": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 5, 7],
            }
        ),
    }
    if model_name not in spaces:
        raise ValueError(f"Modelo {model_name} no reconocido. Opciones: {list(spaces.keys())}")
    return spaces[model_name]

def _save_pipeline_diagram_html(estimator, path: str = "pipeline_model_diagram.html"):
    try:
        html_code = estimator_html_repr(estimator)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_code)
    except Exception:
        pass

def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

# =========================
# API pública
# =========================

def train_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
    model_name: str,
    val_report_path: str | None = "val_report.html",
    diagram_path: str | None = "pipeline_model_diagram.html",
):
    """
    Entrena y selecciona hiperparámetros SOLO con TRAIN (+ valida en VAL).
    NO toca TEST (se evalúa aparte con test_model).

    Devuelve:
      - best_pipeline (preprocesador + modelo)
      - best_params (del search)
      - metrics: {"val_r2": ...}
    """
    base_model, param_grid = _get_model_and_space(model_name)
    pre = _make_preprocessor(X_train)

    # pipeline base para búsqueda
    pipe = Pipeline([("pre", pre), ("model", base_model)])
    _save_pipeline_diagram_html(pipe, path=diagram_path or "pipeline_model_diagram.html")

    # búsqueda en CV (sin eval_set)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=5,
        cv=3,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    # Mejor pipeline por defecto
    best_pipeline = search.best_estimator_

    # Refit especial para XGBoost con early stopping usando VAL transformado,
    # SIN tocar TEST, y reconstruyendo un pipeline consistente.
    if model_name == "XGBoost":
        pre_fitted = clone(pre).fit(X_train, y_train)
        X_tr_tx = pre_fitted.transform(X_train)
        X_va_tx = pre_fitted.transform(X_val)

        best_params_core = {
            k.split("__", 1)[1]: v
            for k, v in search.best_params_.items()
            if k.startswith("model__")
        }
        xgb_final = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            **best_params_core,
            early_stopping_rounds=50,
        )
        xgb_final.fit(X_tr_tx, y_train, eval_set=[(X_va_tx, y_val)], verbose=False)
        best_pipeline = Pipeline([("pre", pre_fitted), ("model", xgb_final)])

        _save_pipeline_diagram_html(best_pipeline, path=diagram_path or "pipeline_model_diagram.html")

    # Métrica de validación
    y_val_pred = best_pipeline.predict(X_val)
    val_r2 = r2_score(y_val, y_val_pred)

    # Reporte breve (solo validación)
    if val_report_path:
        html = [
            "<html><head><title>Validación</title></head><body>",
            f"<h1>Modelo: {model_name}</h1>",
            f"<p>Best Params: {search.best_params_}</p>",
            f"<p>Validación R²: {val_r2:.4f}</p>",
            "</body></html>",
        ]
        with open(val_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

    return best_pipeline, search.best_params_, {"val_r2": val_r2}


def test_model(
    model_pipeline: Pipeline,
    X_test: pd.DataFrame, y_test: pd.Series,
    test_report_path: str | None = "test_report.html",
    shap_max_display: int = 25,
    background_X: pd.DataFrame | None = None,
):
    """
    Evalúa SOLO en TEST y (opcional) genera SHAP.
    'model_pipeline' debe ser el Pipeline devuelto por train_model (pre + model).
    """
    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # Intento SHAP opcional
    shap_img_b64 = None
    try:
        import shap  # import local para no fallar si no está en runtime
        pre = model_pipeline.named_steps["pre"]
        mdl = model_pipeline.named_steps["model"]

        X_test_tx = pre.transform(X_test)
        X_test_tx = _to_dense(X_test_tx)

        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(X_test_tx.shape[1])]

        is_tree = isinstance(mdl, (RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor))
        if background_X is not None:
            bg_tx = pre.transform(background_X)
            bg_tx = _to_dense(bg_tx)
        else:
            # usar una pequeña muestra de test como background para no depender de TRAIN aquí
            n_bg = min(200, X_test_tx.shape[0])
            bg_tx = X_test_tx[:n_bg]

        if is_tree:
            explainer = shap.TreeExplainer(mdl, data=bg_tx, feature_perturbation="interventional")
            sv = explainer(X_test_tx, check_additivity=False)
        else:
            explainer = shap.Explainer(mdl, bg_tx)
            sv = explainer(X_test_tx)

        plt.figure(figsize=(8, 6))
        X_test_df = pd.DataFrame(X_test_tx, columns=feature_names)
        shap.summary_plot(sv, X_test_df, show=False, max_display=shap_max_display)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        shap_img_b64 = base64.b64encode(buf.read()).decode("ascii")
    except Exception as e:
        # SHAP es opcional; si falla, seguimos igual y lo anotamos en el reporte
        shap_img_b64 = f"__ERROR__: {e}"

    # Reporte TEST
    if test_report_path:
        parts = [
            "<html><head><title>Test</title></head><body>",
            "<h1>Evaluación en TEST</h1>",
            f"<p>Test MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}</p>",
        ]
        if shap_img_b64:
            if shap_img_b64.startswith("__ERROR__"):
                parts.append(f"<p>⚠️ No se pudo generar SHAP: {shap_img_b64}</p>")
            else:
                parts.append("<h3>SHAP summary</h3>")
                parts.append(f'<img alt="SHAP summary" style="max-width:100%;height:auto" src="data:image/png;base64,{shap_img_b64}">')
        parts.append("</body></html>")
        with open(test_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

    return {"mse": mse, "mae": mae, "r2": r2}


def predict(model_pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predicción genérica para inputs nuevos (producción)."""
    return model_pipeline.predict(X)
