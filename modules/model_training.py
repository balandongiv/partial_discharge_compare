"""
Model Training Module for Partial Discharge Classification Pipeline

This module implements Phase 2 of the PD classification pipeline, responsible for training
a diverse set of machine learning models with hyperparameter optimization. It applies
cross-validation, grid search, and model selection to identify the best performing
classifiers for partial discharge detection.

Step-by-Step Process:
1. Data Loading and Preparation:
   - Loads selected features from baseline track (features/4_feature_selection/tracks/baseline_all_feats/)
   - Combines features from all stations into unified dataset
   - Generates dummy labels based on file naming patterns (1 = PD present, 0 = no PD)
   - Separates feature matrix (X) and target labels (y)

2. Data Validation and Split:
   - Validates minimum class requirements for cross-validation
   - Checks for sufficient samples per class (minimum 2 samples per class)
   - Performs stratified train-test split (75% train, 25% test)
   - Ensures balanced representation across classes

3. Cross-Validation Setup:
   - Configures StratifiedKFold with appropriate number of splits
   - Adjusts CV splits based on minimum class count (max 3 splits)
   - Handles edge cases with insufficient data for CV
   - Falls back to no-CV training when necessary

4. Model Zoo Definition:
   The module trains four different model types:

   Logistic Regression (logreg):
   - Pipeline with StandardScaler preprocessing
   - Grid search over C values: [0.1, 1, 10]
   - Maximum 200 iterations for convergence

   Support Vector Machine (svm):
   - Pipeline with StandardScaler preprocessing
   - Grid search over C values: [0.1, 1] and kernels: ['rbf', 'linear']
   - Probability estimates enabled for ROC-AUC calculation

   Random Forest (rf):
   - 200 estimators with random state for reproducibility
   - Grid search over max_depth: [None, 5, 10]
   - Handles both deep and shallow tree configurations

   XGBoost (xgb):
   - 200 estimators with controlled parameters
   - Grid search over max_depth: [3, 4]
   - Optimized for speed with hist tree method

5. Hyperparameter Optimization:
   - Applies GridSearchCV for systematic parameter exploration
   - Uses cross-validation for robust model evaluation
   - Selects best parameters based on CV performance
   - Handles single-parameter scenarios gracefully

6. Model Training and Evaluation:
   - Trains each model with optimal hyperparameters
   - Evaluates performance on held-out test set
   - Calculates accuracy scores for model comparison
   - Handles both CV and no-CV training scenarios

7. Model Persistence:
   - Saves trained models to models/tuned_gridsearch/ directory
   - Uses joblib for efficient model serialization
   - Maintains model traceability and reproducibility

8. Results Compilation:
   - Collects performance scores for all models
   - Logs training progress and final results
   - Returns comprehensive results dictionary

Training Strategy:
- Stratified sampling ensures balanced class representation
- Cross-validation provides robust performance estimates
- Grid search explores parameter space systematically
- Multiple algorithms capture different learning patterns

Configuration Parameters:
- station_id: Optional specific station to train on (uses all if None)
- input_path: Path to processed data
- output_path: Root directory for model outputs

Dependencies:
- joblib: Model serialization and persistence
- numpy: Numerical operations and array handling
- pandas: Data manipulation and CSV I/O
- sklearn: Machine learning algorithms and utilities
- xgboost: Gradient boosting classifier
- pathlib: Cross-platform path handling

Output Structure:
- models/tuned_gridsearch/<model_name>.joblib
- Trained models ready for evaluation and deployment
- Performance metrics for model comparison
"""

from __future__ import annotations

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import json
import platform
import shutil
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .logger import get_logger
import config as app_config


def _load_selected_features(station_id: Optional[str] = None, track: Optional[str] = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load selected features from a specific track or default location.
    
    Parameters
    ----------
    station_id
        Optional station ID to filter by.
    track
        Optional track name (e.g., "mljar_internal", "baseline_all_feats", "featurewiz_corr_xgb").
        If None, uses the default selected_features.parquet.
    """
    if track:
        selected_path = Path("features") / "4_feature_selection" / "tracks" / track / "selected_features.parquet"
    else:
        selected_path = Path("features") / "4_feature_selection" / "selected_features.parquet"
    
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected features file not found at {selected_path}. Run feature selection first.")
    df = pd.read_parquet(selected_path)
    if df.empty:
        raise ValueError("Selected features dataframe is empty.")
    if station_id is not None:
        df = df[df["station_id"].astype(str) == str(station_id)]
        if df.empty:
            raise ValueError(f"No rows available for station {station_id}.")
    target = df["faultAnnotation"].fillna("unknown").astype(str)
    meta_cols = [c for c in ["station_id", "measurement_id", "faultAnnotation", "signal_length"] if c in df.columns]
    feature_df = df.drop(columns=[c for c in meta_cols if c in df.columns])
    return feature_df.reset_index(drop=True), target.reset_index(drop=True), df[meta_cols].reset_index(drop=True)


def _compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray, average: str = "macro") -> float:
    classes = np.unique(y_true)
    if classes.size <= 1 or y_proba.ndim == 1:
        return float("nan")
    try:
        if classes.size == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average=average))
    except ValueError:
        return float("nan")


def run(station_id: Optional[str], input_path: Path, output_path: Path, run_config: Optional[dict] = None) -> dict:
    """Train predefined models with cross-validation and persist metrics and artefacts.
    
    Parameters
    ----------
    station_id
        Optional station ID to filter by.
    input_path
        Path to input data (currently unused, kept for API compatibility).
    output_path
        Root directory for outputs (currently unused, kept for API compatibility).
    run_config
        Optional configuration dict. Can contain:
        - "track": track name (e.g., "mljar_internal", "baseline_all_feats") to use specific feature selection track.
                   If not specified, uses default selected_features.parquet.
    """
    log = get_logger(__name__, log_dir=output_path / "reports")
    
    # Get track name from config, default to None (uses default path)
    track = (run_config or {}).get("track")
    track_display = track if track else "default"
    log.info("Starting model training stage (track: %s)", track_display)

    feature_df, target_series, meta_df = _load_selected_features(station_id=station_id, track=track)
    feature_names = list(feature_df.columns)
    X = feature_df.astype(np.float32).to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target_series)

    classes, class_counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("Need at least two classes for model training.")
    min_class_count = class_counts.min()

    desired_folds = 10 if min_class_count >= 10 else 5
    n_splits = min(desired_folds, int(min_class_count))

    if n_splits < 2:
        log.warning("Insufficient samples per class for StratifiedKFold; falling back to StratifiedShuffleSplit")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        cv_indices = list(splitter.split(X, y))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_indices = list(splitter.split(X, y))
    log.info("Using %d validation split(s)", len(cv_indices))

    models: dict[str, Pipeline] = {
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "ann": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        max_iter=300,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    results_records = []
    metrics_rows = []
    
    # Use track-specific directory for results if track is specified
    if track:
        preds_dir = Path("results") / track
        models_dir = Path("models") / "trained" / track
    else:
        preds_dir = Path("results")
        models_dir = Path("models") / "trained"
    
    preds_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    log.info("Results will be saved to: %s", preds_dir)
    log.info("Models will be saved to: %s", models_dir)

    for model_name, estimator in models.items():
        log.info("Training model: %s", model_name)
        y_pred = np.empty_like(y)
        y_proba = np.zeros((y.shape[0], len(classes)), dtype=np.float32)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_indices, start=1):
            est = clone(estimator)
            est.fit(X[train_idx], y[train_idx])
            preds = est.predict(X[val_idx])
            y_pred[val_idx] = preds

            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X[val_idx])
            else:
                proba = np.zeros((len(val_idx), len(classes)), dtype=np.float32)
            y_proba[val_idx] = proba

            precision, recall, f1, _ = precision_recall_fscore_support(
                y[val_idx],
                preds,
                average="macro",
                zero_division=0,
            )
            accuracy = accuracy_score(y[val_idx], preds)
            roc_auc = _compute_roc_auc(y[val_idx], proba)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc,
                }
            )
            log.debug(
                "%s fold %d -> acc=%.3f precision=%.3f recall=%.3f f1=%.3f roc_auc=%s",
                model_name,
                fold_idx,
                accuracy,
                precision,
                recall,
                f1,
                "nan" if np.isnan(roc_auc) else f"{roc_auc:.3f}",
            )

        metrics_df = pd.DataFrame(fold_metrics)
        summary_row = {
            "model": model_name,
            "accuracy_mean": metrics_df["accuracy"].mean(),
            "accuracy_std": metrics_df["accuracy"].std(ddof=0),
            "precision_mean": metrics_df["precision"].mean(),
            "precision_std": metrics_df["precision"].std(ddof=0),
            "recall_mean": metrics_df["recall"].mean(),
            "recall_std": metrics_df["recall"].std(ddof=0),
            "f1_mean": metrics_df["f1"].mean(),
            "f1_std": metrics_df["f1"].std(ddof=0),
            "roc_auc_mean": metrics_df["roc_auc"].mean(),
            "roc_auc_std": metrics_df["roc_auc"].std(ddof=0),
        }
        results_records.append(summary_row)

        metrics_df["model"] = model_name
        metrics_rows.append(metrics_df)

        pred_df = meta_df.copy()
        if "faultAnnotation" in pred_df.columns:
            pred_df = pred_df.rename(columns={"faultAnnotation": "true_label"})
        pred_df["true_label"] = target_series.values
        pred_df["pred_label"] = label_encoder.inverse_transform(y_pred)
        pred_df["pred_encoded"] = y_pred
        for idx, label in enumerate(label_encoder.classes_):
            pred_df[f"prob_{label}"] = y_proba[:, idx]
        pred_path = preds_dir / f"predictions_{model_name}.parquet"
        pred_df.to_parquet(pred_path, index=False)
        log.info("Stored cross-validated predictions -> %s", pred_path)

        final_estimator = clone(estimator)
        final_estimator.fit(X, y)
        artifact = {
            "model": final_estimator,
            "feature_names": feature_names,
            "classes": label_encoder.classes_,
        }
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(artifact, model_path)
        log.info("Persisted trained model -> %s", model_path)

    metrics_summary_df = pd.DataFrame(results_records)
    metrics_summary_path = preds_dir / "model_metrics.csv"
    metrics_summary_df.to_csv(metrics_summary_path, index=False)
    log.info("Model metrics summary -> %s", metrics_summary_path)

    detailed_metrics_df = pd.concat(metrics_rows, ignore_index=True)
    detailed_metrics_path = preds_dir / "model_metrics_detailed.csv"
    detailed_metrics_df.to_csv(detailed_metrics_path, index=False)
    log.info("Detailed fold metrics -> %s", detailed_metrics_path)

    best_model_name: Optional[str] = None
    best_model_metrics: Dict[str, float] = {}
    if not metrics_summary_df.empty:
        ordered = metrics_summary_df.sort_values(["f1_mean", "roc_auc_mean"], ascending=False, ignore_index=True)
        best_row = ordered.iloc[0]
        best_model_name = str(best_row["model"])
        best_model_metrics = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in best_row.items()}

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = app_config.REPORTS_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path: Optional[Path] = snapshot_dir / f"config_snapshot_{timestamp}.yaml"
    try:
        shutil.copy(app_config.CONFIG_PATH, snapshot_path)
    except OSError:
        log.exception("Unable to copy config snapshot to %s", snapshot_path)
        snapshot_path = None

    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn_version,
        "timestamp_utc": timestamp,
    }

    training_meta = {
        "models": list(models.keys()),
        "feature_count": len(feature_names),
        "class_labels": label_encoder.classes_.tolist(),
        "splits": len(cv_indices),
        "best_model": {
            "name": best_model_name,
            "metrics": best_model_metrics,
        },
        "config_snapshot": str(snapshot_path) if snapshot_path else None,
        "environment": env_info,
    }
    meta_path = preds_dir / "training_summary.json"
    meta_path.write_text(json.dumps(training_meta, indent=2), encoding="utf-8")

    log.info("Model training stage completed")
    return training_meta


