from sklearn.base import estimator_html_repr
from .types_ import PipelineConfig
from .FilterAndClear import CleanTransformer, SplitTransformer
from .extract import ClassifyTypesTransformer, extract_csv
from sklearn.pipeline import Pipeline

def run_data_pipeline(config: PipelineConfig):
    print(f"run_data_pipeline with config: {config}")
    df_raw = extract_csv(config)
    pipe = Pipeline(steps=[
        ("classify", ClassifyTypesTransformer(config)),
        ("clean",    CleanTransformer(config)),
        ("split",    SplitTransformer(config)),
    ])
    html_code = estimator_html_repr(pipe)

    with open("pipeline_diagram.html", "w", encoding="utf-8") as f:
        f.write(html_code)

    print("Archivo guardado: pipeline_diagram.html (ábrelo en tu navegador)")

    X_train, X_val, X_test, y_train, y_val, y_test = pipe.fit_transform(df_raw)
    return X_train, X_val, X_test, y_train, y_val, y_test