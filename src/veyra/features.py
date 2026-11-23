"""Conservative, train-fitted preprocessing shared by both classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

# Static policy, not a feature selection learned from validation or full-data EDA.
# id_30/id_31/id_33 are OS/browser/screen attributes, not row identifiers.
NUMERIC_FEATURES = ("TransactionAmt", "dist1", "dist2")
CATEGORICAL_FEATURES = (
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    *(f"M{index}" for index in range(1, 10)),
    "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
)
ALLOWED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def exclusion_reason(column: str) -> str:
    if column == "isFraud":
        return "target; never a feature"
    if column == "TransactionDT":
        return "absolute time; routing only, not a predictor"
    if column in {"TransactionID", "card1", "card2", "card3", "card5", "addr1", "addr2"}:
        return "row identifier or identifier-like card/address code"
    if column.startswith(("C", "D", "V")) and column[1:].isdigit():
        return "anonymous count/delta/aggregate: past-only availability unverified"
    return "not in the reviewed allowlist; provenance/availability unverified"


class FeaturePolicy(TransformerMixin, BaseEstimator):
    """Drop everything outside the explicit allowlist; normalize without learning.

    The *selected schema* and cardinality diagnostics are captured on fit only.
    Statistics, scaling and category grouping are learned by subsequent steps.
    Unknown extra prediction columns are ignored; missing expected columns fail.
    """

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise ValueError("Features require a DataFrame with unique column names")
        if "TransactionAmt" not in X:
            raise ValueError("Missing required feature: TransactionAmt")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.selected_columns_ = [name for name in ALLOWED_FEATURES if name in X]
        self.excluded_columns_ = {name: exclusion_reason(name) for name in X if name not in self.selected_columns_}
        self.fit_rows_ = len(X)
        normalized = self.transform(X)
        self.cardinalities_ = {
            name: int(normalized[name].nunique())
            for name in self.selected_columns_ if name in CATEGORICAL_FEATURES
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_columns_")
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise ValueError("Features require a DataFrame with unique column names")
        missing = set(self.selected_columns_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing fitted feature columns: {sorted(missing)}")
        result = pd.DataFrame(index=X.index)
        for name in self.selected_columns_:
            if name in NUMERIC_FEATURES:
                numeric = pd.to_numeric(X[name], errors="raise").replace([np.inf, -np.inf], np.nan)
                result[name] = numeric.astype(np.float32)
            else:
                # Prefix real values so a literal missing-token string cannot collide.
                result[name] = ("value:" + X[name].astype("string")).fillna("missing:").astype(object)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_columns_")
        return np.asarray(self.selected_columns_, dtype=object)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in NUMERIC_FEATURES if name in frame]


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in CATEGORICAL_FEATURES if name in frame]


def float32_csr(matrix):
    return sparse.csr_matrix(matrix, dtype=np.float32)


def build_preprocessor(*, max_categories: int = 64, min_frequency: int = 20) -> Pipeline:
    if max_categories < 2 or min_frequency < 1:
        raise ValueError("max_categories must be >= 2 and min_frequency >= 1")
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    columns = ColumnTransformer([
        ("numeric", numeric, numeric_columns),
        ("categorical", OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
            max_categories=max_categories, sparse_output=True, dtype=np.float32,
        ), categorical_columns),
    ], remainder="drop", sparse_threshold=1.0)
    return Pipeline([
        ("policy", FeaturePolicy()), ("columns", columns),
        ("sparse", FunctionTransformer(float32_csr, accept_sparse=True)),
    ])
