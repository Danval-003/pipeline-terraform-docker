# src/aqi_pipeline/cli.py
import argparse
import os
import pandas as pd
import pickle
from .pipeline import PipelineConfig, run_data_pipeline
from .model import train_and_evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Run data pipeline.")
    parser.add_argument("--data-path", required=True, help="Path to input CSV.")
    parser.add_argument("--target", default="AQI", required=True, help="Target column name.")
    parser.add_argument("--out-dir", default="./data", help="Output directory.")
    parser.add_argument("--model", default="RandomForest", help="Model to use.")

    args = parser.parse_args()

    if (args.data_path is None) or (args.data_path.strip() == ""):
        raise ValueError("Data path must be specified.")

    cfg = PipelineConfig(
        data_path=args.data_path,
        target_column=args.target,
        test_size=0.2,
        val_size=0.1,
        drop_duplicates=True,
        id_columns=[],
        require_target_positive=True,
        iqr_clip=False,
        iqr_factor=3.0,
        cast_features_to="float32",
        cast_target_to="float32",
        random_state=42,
    )

    cfg.non_negative_columns = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "AQI"]

    X_train, X_val, X_test, y_train, y_val, y_test = run_data_pipeline(cfg)
    best_model, best_params, metrics = train_and_evaluate_model(
        X_train, X_val, X_test, y_train, y_val, y_test, 
        html_output_path=os.path.join(args.out_dir, "report.html"),
        model_name=args.model
    )

    # Guardar el modelo entrenado 
    with open(os.path.join(args.out_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
        print(f"Best model saved to {os.path.join(args.out_dir, 'best_model.pkl')}")

    print("Best Model Parameters:", best_params)
    print("Evaluation Metrics:", metrics)
    

    os.makedirs(args.out_dir, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(args.out_dir, "processed_train.csv"), index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(args.out_dir, "processed_val.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(args.out_dir, "processed_test.csv"), index=False)

if __name__ == "__main__":
    main()
