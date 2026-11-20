import logging
from pathlib import Path

import pandas as pd
import yaml

from src.datapreparation_2 import Preprocessor
from src.datatraining_1 import ModelTraining


logging.basicConfig(level=logging.INFO)


def main():
    print("Main starting...", flush=True)

    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / "src" / "config.yaml"

    print("Loading config...", flush=True)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["param_grid"] = {
        "ridge_tuned": {
            "regressor__alpha": [0.1, 1.0, 10.0],
        },
        "lasso_tuned": {
            "regressor__alpha": [0.001, 0.01, 0.1, 1.0],
        },
    }

    print("Loading CSV...", flush=True)
    df = pd.read_csv(config["data_path"])
    df.columns = df.columns.astype(str).str.strip()
    print("CSV loaded.", flush=True)

    target_column = config["target_column"]
    drop_columns = config.get("drop_columns", ["ID", "URL"])

    numerical_features = [
        column
        for column in config["numerical_features"]
        if column in df.columns
        and column != target_column
        and column not in drop_columns
    ]
    categorical_features = [
        column
        for column in config["nominal_features"]
        if column in df.columns
        and column != target_column
        and column not in drop_columns
    ]
    ordinal_features = [
        column
        for column in config.get("ordinal_features", [])
        if column in df.columns
        and column != target_column
        and column not in drop_columns
    ]

    print("Creating preprocessor...", flush=True)
    data_prep = Preprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        ordinal_features=ordinal_features,
    )

    config["drop_columns"] = drop_columns

    print("Creating model trainer...", flush=True)
    model_training = ModelTraining(
        config=config,
        preprocessor=data_prep.preprocessor,
    )

    print("Splitting data...", flush=True)
    x_train, x_val, x_test, y_train, y_val, y_test = (
        model_training.split_data(df)
    )
    print("Data split.", flush=True)

    print("Training baseline models...", flush=True)
    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_baseline_models(
            x_train,
            y_train,
            x_val,
            y_val,
        )
    )

    print("Training tuned models...", flush=True)
    tuned_models, tuned_metrics = (
        model_training.train_and_evaluate_tuned_models(
            x_train,
            y_train,
            x_val,
            y_val,
        )
    )

    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    if not all_metrics:
        raise ValueError("No model metrics were returned.")

    best_model_name = max(
        all_metrics,
        key=lambda name: all_metrics[name]["r2"],
    )
    best_model = all_models[best_model_name]

    print("Best model:", best_model_name, flush=True)

    final_metrics = model_training.evaluate_final_model(
        best_model,
        x_test,
        y_test,
        best_model_name,
    )
    print("Final test metrics:", final_metrics, flush=True)


if __name__ == "__main__":
    main()