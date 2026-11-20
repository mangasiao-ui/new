import logging
from math import sqrt

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

class ModelTraining:
    def __init__(
    self,
    config: dict,
    preprocessor: ColumnTransformer,
) -> None:
    
        self.config = config
        self.preprocessor = preprocessor

    def split_data(self, df: pd.DataFrame):
        logging.info("Starting data splitting.")

        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]

        x_train, x_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.config["test_size"],
            random_state=42,
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=self.config["val_size"],
            random_state=42,
        )

        logging.info("Data splitting completed successfully.")
        return x_train, x_val, x_test, y_train, y_val, y_test

    def evaluate_model(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "Model",
    ) -> dict[str, float]:
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mse,
            "r2": r2_score(y, predictions),
            "rmse": sqrt(mse),
        }

        logging.info("%s metrics: %s", model_name, metrics)
        return metrics

    def train_and_evaluate_baseline_models(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[dict[str, Pipeline], dict[str, dict[str, float]]]:
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
        }

        pipelines = {}
        metrics = {}

        for model_name, estimator in models.items():
            pipeline = Pipeline([
                ("preprocessor", self.preprocessor),
                ("regressor", estimator),
            ])
            pipeline.fit(x_train, y_train)

            pipelines[model_name] = pipeline
            metrics[model_name] = self.evaluate_model(
                pipeline, x_val, y_val, model_name
            )

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[dict[str, Pipeline], dict[str, dict[str, float]]]:
        models = {
            "ridge_tuned": Ridge(),
            "lasso_tuned": Lasso(),
        }
        param_grids = self.config["param_grid"]

        tuned_models = {}
        tuned_metrics = {}

        for model_name, estimator in models.items():
            pipeline = Pipeline([
                ("preprocessor", self.preprocessor),
                ("regressor", estimator),
            ])

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[model_name],
                cv=self.config.get("cv", 5),
                scoring=self.config.get(
                    "scoring", "neg_mean_squared_error"
                ),
                n_jobs=-1,
            )
            grid_search.fit(x_train, y_train)

            tuned_models[model_name] = grid_search.best_estimator_
            tuned_metrics[model_name] = self.evaluate_model(
                grid_search.best_estimator_,
                x_val,
                y_val,
                f"{model_name} (tuned)",
            )

        return tuned_models, tuned_metrics

    def evaluate_final_model(
        self,
        model: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
    ) -> dict[str, float]:
        return self.evaluate_model(model, x_test, y_test, model_name)