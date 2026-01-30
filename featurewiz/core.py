"""Minimal Featurewiz-inspired feature selection routine.

The implementation follows the two main ideas of Featurewiz:

1. Remove highly correlated features (SULOV-style pruning).
2. Rank remaining features via XGBoost importance scores.

Only the subset of functionality required by this project is provided.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def _prepare_dataframe(data: pd.DataFrame | str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, str):
        return pd.read_csv(data)
    raise TypeError("Unsupported data type for featurewiz input.")


def _correlation_pruning(df: pd.DataFrame, corr_limit: float) -> pd.Index:
    if df.shape[1] <= 1:
        return df.columns
    corr = df.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = {col for col in upper.columns if (upper[col] > corr_limit).any()}
    return df.columns.difference(list(to_drop), sort=False)


def _xgb_rank_features(X: pd.DataFrame, y: np.ndarray) -> Sequence[str]:
    if X.empty:
        return []
    num_classes = int(np.unique(y).size)
    objective = "binary:logistic" if num_classes <= 2 else "multi:softprob"
    
    # Handle small datasets - use fewer estimators and simpler model
    n_estimators = min(100, max(10, X.shape[0] * 2))
    max_depth = min(3, max(1, X.shape[0] // 2))
    
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "max_depth": max_depth,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": objective,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": 42,
    }
    
    # For binary classification, set base_score to avoid XGBoost error
    if num_classes <= 2:
        model_params["base_score"] = 0.5
    
    model = XGBClassifier(**model_params)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    selected = [feat for feat, score in importances.items() if score > 0]
    if not selected:
        selected = list(importances.index[: min(10, len(importances))])
    return selected


def featurewiz(
    data: pd.DataFrame | str,
    target: str,
    corr_limit: float = 0.7,
    verbose: int = 0,
    skip_sulov: bool = False,
    skip_xgboost: bool = False,
    **_: object,
) -> Tuple[Sequence[str], pd.DataFrame]:
    """Approximate Featurewiz selection and return selected feature names and dataframe.

    Parameters
    ----------
    data:
        Input DataFrame or path to CSV containing the dataset.
    target:
        Name of the target column.
    corr_limit:
        Maximum allowed correlation between retained features.
    verbose:
        Verbosity flag (currently unused, present for API compatibility).
    skip_sulov:
        When True, correlation pruning is skipped.
    skip_xgboost:
        When True, correlation filtering is the only selection step.
    **_:
        Additional keyword arguments are accepted for signature compatibility
        but ignored by this lightweight implementation.
    """
    df = _prepare_dataframe(data)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not present in dataframe.")

    y_raw = df[target]
    feature_df = df.drop(columns=[target])
    numeric_df = feature_df.select_dtypes(include=[np.number]).fillna(0.0)

    if numeric_df.empty:
        raise ValueError("No numeric features available for selection.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw.astype(str))

    if skip_sulov:
        reduced_cols = numeric_df.columns
    else:
        reduced_cols = _correlation_pruning(numeric_df, corr_limit=corr_limit)
    reduced_df = numeric_df[reduced_cols]

    if skip_xgboost:
        selected_features = list(reduced_df.columns)
    else:
        selected_features = list(_xgb_rank_features(reduced_df, y_encoded))

    result_df = df[selected_features + [target]].copy()
    return selected_features, result_df

