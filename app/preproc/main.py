from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from pathlib import Path
import pandas as pd
import shutil

# Tu librería
from oxigen_pipeline.types_ import PipelineConfig  # dataclass
from oxigen_pipeline.pipeline import run_data_pipeline

app = FastAPI(title="Preproc API")

DATA_DIR = Path("/mnt/data")
REPORTS  = Path("/mnt/reports")
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ------- Payload flexible (API-owned types) -------
class PreprocPayload(BaseModel):
    # Acepta ambos nombres y los normaliza
    input_path: Optional[str] = None
    data_path: Optional[str] = None

    target_col: Optional[str] = None
    target_column: Optional[str] = None

    test_size: float = 0.20
    val_size: float  = 0.20
    random_state: int = 42

    # opcionales de tu dataclass
    drop_duplicates: bool = True
    id_columns: Optional[List[str]] = None
    require_target_positive: bool = True
    iqr_clip: bool = False
    iqr_factor: float = 3.0
    cast_features_to: str = "float32"
    cast_target_to: str = "float32"
    non_negative_columns: Optional[List[str]] = None

    @field_validator("test_size", "val_size")
    @classmethod
    def _check_fracs(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("test_size/val_size deben estar en (0, 1)")
        return v

    # Normaliza a la dataclass de la librería
    def to_lib_config(self) -> PipelineConfig:
        dp = self.data_path or self.input_path
        tc = self.target_column or self.target_col
        if not dp:
            raise HTTPException(400, "Falta data_path (o input_path)")
        if not tc:
            raise HTTPException(400, "Falta target_column (o target_col)")
        return PipelineConfig(
            data_path=dp,
            target_column=tc,
            test_size=self.test_size,
            val_size=self.val_size,
            drop_duplicates=self.drop_duplicates,
            id_columns=self.id_columns,
            require_target_positive=self.require_target_positive,
            iqr_clip=self.iqr_clip,
            iqr_factor=self.iqr_factor,
            cast_features_to=self.cast_features_to,
            cast_target_to=self.cast_target_to,
            random_state=self.random_state,
            non_negative_columns=self.non_negative_columns,
        )

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.get("/pipeline", response_class=HTMLResponse)
def get_pipeline_diagram():
    p = REPORTS / "pipeline_diagram.html"
    return p.read_text(encoding="utf-8") if p.exists() else "<p>Sin diagrama. Ejecuta POST /run</p>"

@app.post("/run")
def run(cfg: PreprocPayload = Body(...)):
    lib_cfg = cfg.to_lib_config()
    print(f"[preproc] lib_cfg={lib_cfg}")

    X_train, X_val, X_test, y_train, y_val, y_test = run_data_pipeline(lib_cfg)

    # Mover el diagrama evitando EXDEV (cross-device); usa copy2 + unlink
    src = Path("pipeline_diagram.html")
    if src.exists():
        dst = REPORTS / "pipeline_diagram.html"
        shutil.copy2(src, dst)  # conserva metadatos
        try:
            src.unlink(missing_ok=True)
        except Exception:
            pass

    # Persistir splits con el nombre real de target
    target = lib_cfg.target_column
    paths = {}
    def _save(name, X, y):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        df[target] = y
        out = DATA_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        paths[f"{name}_path"] = str(out)

    _save("train", X_train, y_train)
    _save("val",   X_val,   y_val)
    _save("test",  X_test,  y_test)

    return {"status": "ok", "paths": paths, "report": str(REPORTS / "pipeline_diagram.html")}
