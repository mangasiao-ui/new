from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class Preprocessor:
    def __init__(
        self,
        categorical_features,
        numerical_features,
        ordinal_features=None,
    ):
        self.ordinal_features = ordinal_features or []
        transformers = []

        if numerical_features:
            transformers.append((
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numerical_features,
            ))

        if categorical_features:
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ))

        if self.ordinal_features:
            transformers.append((
                "ordinal",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]),
                self.ordinal_features,
            ))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )