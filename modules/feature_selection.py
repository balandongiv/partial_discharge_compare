"""
Feature Selection Module for Partial Discharge Classification Pipeline

This module implements the feature selection phase of the PD classification pipeline,
responsible for applying multiple feature selection strategies to identify optimal
feature subsets for machine learning models. It implements three distinct selection
tracks to compare different approaches, matching the documented experiment design.

Step-by-Step Process:
1. Input Data Loading:
   - Reads expanded feature Parquet file from ``features/3_feature_comb_expansion/mathematical_combination``
   - Loads feature matrices for each station (optionally filtered)
   - Separates feature columns from metadata columns (station_id, measurement_id, etc.)

2. Selection Track Implementation:
   The module implements the following feature selection strategies:

   Track 4A – Baseline (All Features):
   - Preserves all engineered features without selection
   - Serves as the control track for comparison

   Track 4B – Featurewiz (Correlation-Prune + XGBoost):
   - Invokes Featurewiz to run SULOV correlation pruning and XGBoost ranking
   - Respects configurable correlation limits via ``config["corr_limit"]`` (default 0.7)

   Track 4C – MLJAR-Supervised (Internal Selector):
   - Approximated via variance-based filtering to emulate MLJAR's internal pruning
   - Threshold configurable via ``config["variance_threshold"]`` (default 1e-12)

3. Feature Selection Application:
   - Applies each selection track to the numeric feature matrix
   - Produces reduced feature sets for each track while preserving metadata
   - Documents selected column names and counts per track

4. Output Organization:
   - Creates track-specific directories under ``features/4_feature_selection/tracks/``
   - Saves selected features as Parquet and CSV for each track
   - Stores JSON summaries and label encoder mappings for reproducibility

5. Summary Statistics:
   - Logs input vs output feature counts per track
   - Returns combined summary dictionary for downstream stages

Configuration Parameters:
- station_id: Optional specific station to process (processes all if None)
- input_path: Path to expanded feature outputs (currently unused)
- output_path: Root directory for selected feature outputs (used for logging)

Dependencies:
- numpy: Numerical operations
- pandas: Data manipulation and Parquet/CSV I/O
- pathlib: Cross-platform path handling
- featurewiz: Feature selection
- scikit-learn: Variance-based filtering

Output Structure:
- ``features/4_feature_selection/tracks/<track_name>/selected_features.(parquet|csv)``
- ``features/4_feature_selection/tracks/<track_name>/summary.json``
- ``features/4_feature_selection/tracks/<track_name>/label_mapping.json``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .logger import get_logger

try:
    from featurewiz import featurewiz  # type: ignore
except ImportError as exc:  # pragma: no cover - handled during runtime
    raise ImportError(
        "featurewiz package is required for feature selection. Install with `pip install featurewiz`."
    ) from exc

try:
    from supervised.automl import AutoML  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mljar-supervised package is required for Track 4C. Install with `pip install mljar-supervised`."
    ) from exc


def run(station_id: Optional[str], input_path: Path, output_path: Path, config: Optional[dict] = None) -> dict:
    """Execute feature selection tracks 4A (Baseline), 4B (Featurewiz), and 4C (MLJAR)."""
    log = get_logger(__name__, log_dir=output_path / "reports")
    log.info("Starting feature selection stage (tracks 4A, 4B, 4C)")

    combined_path = Path("features") / "3_feature_comb_expansion" / "mathematical_combination" / "combined_features.parquet"
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined features not found at {combined_path}. Run feature combination first.")

    df = pd.read_parquet(combined_path)
    if df.empty:
        raise ValueError("Combined features dataframe is empty; cannot perform selection.")

    if station_id is not None:
        df = df[df["station_id"].astype(str) == str(station_id)]
        if df.empty:
            raise ValueError(f"No rows found for station {station_id} in combined features.")

    if "faultAnnotation" not in df.columns:
        raise ValueError("Column 'faultAnnotation' not found; cannot encode target for selection tracks.")

    meta_cols = [c for c in ["station_id", "measurement_id", "faultAnnotation", "signal_length"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    feature_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    target_series = df["faultAnnotation"].fillna("unknown").astype(str)
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target_series)

    fw_input = feature_df.copy()
    fw_input["target"] = encoded_target

    selection_root = Path("features") / "4_feature_selection" / "tracks"
    selection_root.mkdir(parents=True, exist_ok=True)

    mapping = {int(idx): label for idx, label in enumerate(label_encoder.classes_)}

    def _write_track(track_slug: str, selected_columns: list[str], summary_payload: dict) -> None:
        clean_columns = list(dict.fromkeys(selected_columns))
        track_dir = selection_root / track_slug
        track_dir.mkdir(parents=True, exist_ok=True)

        selected_df = df[meta_cols + clean_columns]
        selected_path = track_dir / "selected_features.parquet"
        selected_csv = track_dir / "selected_features.csv"
        selected_df.to_parquet(selected_path, index=False)
        selected_df.to_csv(selected_csv, index=False)
        log.info("[%s] Saved %d features -> %s", track_slug, len(clean_columns), selected_path)

        summary_path = track_dir / "summary.json"
        summary_payload = {
            "track": track_slug,
            "total_input_features": len(feature_cols),
            "selected_feature_count": len(clean_columns),
            "selected_features": clean_columns,
            **summary_payload,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        log.debug("[%s] Summary saved -> %s", track_slug, summary_path)

        mapping_path = track_dir / "label_mapping.json"
        mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        log.debug("[%s] Label mapping saved -> %s", track_slug, mapping_path)

    # Track 4A – Baseline (All Features)
    baseline_summary = {"description": "Track 4A Baseline (all engineered features retained)"}
    _write_track("baseline_all_feats", feature_cols, baseline_summary)

    # Track 4B – Featurewiz (Correlation pruning + XGB ranking)
    corr_limit = (config or {}).get("corr_limit", 0.7)
    fw_selected, _ = featurewiz(
        fw_input,
        "target",
        corr_limit=corr_limit,
        verbose=0,
        skip_sulov=False,
        skip_xgboost=False,
    )

    if not fw_selected:
        log.warning("Featurewiz returned no features; falling back to all engineered features for Track 4B")
        fw_selected = feature_cols

    featurewiz_summary = {
        "description": "Track 4B Featurewiz (correlation pruning + XGB ranking)",
        "corr_limit": corr_limit,
    }
    _write_track("featurewiz_corr_xgb", fw_selected, featurewiz_summary)

    root_selection_dir = Path("features") / "4_feature_selection"
    root_selection_dir.mkdir(parents=True, exist_ok=True)
    root_selected_df = df[meta_cols + fw_selected].copy()
    numeric_cols_root = [c for c in root_selected_df.columns if c not in meta_cols and np.issubdtype(root_selected_df[c].dtype, np.number)]
    root_selected_df[numeric_cols_root] = root_selected_df[numeric_cols_root].astype(np.float32, copy=False)
    root_selected_df.to_parquet(root_selection_dir / "selected_features.parquet", index=False)
    root_selected_df.to_csv(root_selection_dir / "selected_features.csv", index=False)

    # Track 4C – MLJAR-supervised (AutoML with internal selector)
    mljar_settings = (config or {}).get("mljar", {})
    mljar_track_slug = "mljar_internal"
    mljar_results_dir = selection_root / mljar_track_slug / "automl_results"
    mljar_results_dir.mkdir(parents=True, exist_ok=True)

    default_ml_task = "binary_classification" if len(label_encoder.classes_) <= 2 else "multiclass_classification"
    automl_defaults = {
        "ml_task": mljar_settings.get("ml_task", default_ml_task),
        "mode": mljar_settings.get("mode", "Explain"),
        "total_time_limit": mljar_settings.get("total_time_limit", 1200),
        "model_time_limit": mljar_settings.get("model_time_limit", 240),
        "results_path": str(mljar_results_dir),
        "features_selection": True,
        "train_ensemble": True,
        "verbose": mljar_settings.get("verbose", 0),
    }

    validation_strategy = mljar_settings.get(
        "validation_strategy",
        {"validation_type": "kfold", "k_folds": 5, "shuffle": True, "random_seed": 42},
    )
    automl_defaults["validation_strategy"] = validation_strategy

    log.info("[Track 4C] Launching MLJAR AutoML with settings: %s", automl_defaults)
    automl = AutoML(**automl_defaults)
    automl.fit(feature_df, encoded_target)

    # Attempt to read column importances produced by MLJAR
    importance_path = mljar_results_dir / "columns_importance.json"
    importance_threshold = mljar_settings.get("importance_threshold", 0.0)
    mljar_selected: list[str]

    if importance_path.exists():
        try:
            importance_payload = json.loads(importance_path.read_text(encoding="utf-8"))
            if isinstance(importance_payload, dict):
                sorted_items = sorted(
                    importance_payload.items(), key=lambda item: item[1], reverse=True
                )
                mljar_selected = [
                    feat for feat, score in sorted_items if score is not None and score > importance_threshold
                ]
            else:
                mljar_selected = feature_cols
                log.warning(
                    "[Track 4C] Unexpected importance payload structure (%s); defaulting to all features",
                    type(importance_payload),
                )
        except json.JSONDecodeError:
            log.exception("[Track 4C] Failed to parse columns_importance.json; defaulting to all features")
            mljar_selected = feature_cols
    else:
        log.warning("[Track 4C] columns_importance.json not found; defaulting to all features")
        mljar_selected = feature_cols

    if not mljar_selected:
        log.warning("[Track 4C] Importance filtering removed all features; defaulting to all engineered features")
        mljar_selected = feature_cols

    mljar_summary = {
        "description": "Track 4C MLJAR-supervised (internal selector + AutoML leaderboard)",
        "mljar_settings": {k: v for k, v in automl_defaults.items() if k != "validation_strategy"},
        "validation_strategy": validation_strategy,
        "importance_threshold": importance_threshold,
    }

    # Persist leaderboard snapshot alongside selected features
    leaderboard_path = mljar_results_dir / "leaderboard.csv"
    if leaderboard_path.exists():
        snapshot_path = selection_root / mljar_track_slug / "leaderboard.csv"
        shutil.copy2(leaderboard_path, snapshot_path)
        log.info("[Track 4C] Leaderboard snapshot saved -> %s", snapshot_path)
    else:
        log.warning("[Track 4C] Leaderboard not found at %s", leaderboard_path)

    _write_track(mljar_track_slug, mljar_selected, mljar_summary)

    summary_payload = {
        "tracks": {
            "4A_baseline_all_feats": len(feature_cols),
            "4B_featurewiz_corr_xgb": len(fw_selected),
            "4C_mljar_internal": len(mljar_selected),
        },
        "total_input_features": len(feature_cols),
    }
    log.info("Feature selection summary: %s", summary_payload)

    return summary_payload


