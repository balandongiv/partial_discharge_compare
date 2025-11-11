from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules import (
    data_cleaning,
    evaluation_reporting,
    feature_engineering,
    feature_expansion,
    feature_selection,
    model_training,
)

ROOT = Path.cwd()
PROJECT_ROOT = ROOT / "partial_discharge_project"
DATASET_ROOT = ROOT / "dataset"


@pytest.fixture(scope="session")
def pipeline_outputs():
    PROJECT_ROOT.mkdir(exist_ok=True)
    summaries = {}
    summaries["cleaning"] = data_cleaning.run(None, DATASET_ROOT, PROJECT_ROOT, config={"fs": 1_000_000.0})
    summaries["feature_engineering"] = feature_engineering.run(None, PROJECT_ROOT, PROJECT_ROOT, config={"fs": 1_000_000.0})
    summaries["feature_expansion"] = feature_expansion.run(None, PROJECT_ROOT, PROJECT_ROOT)
    summaries["feature_selection"] = feature_selection.run(None, PROJECT_ROOT, PROJECT_ROOT)
    summaries["model_training"] = model_training.run(None, PROJECT_ROOT, PROJECT_ROOT)
    summaries["evaluation_reporting"] = evaluation_reporting.run(None, PROJECT_ROOT, PROJECT_ROOT)
    return summaries


def test_preprocessing_outputs(pipeline_outputs):
    station_dirs = list(PROJECT_ROOT.glob("station_*/data_clean/standard_denoising_normalisation/cleaned_windows.parquet"))
    assert station_dirs, "Cleaned parquet files not generated per station"
    for parquet_path in station_dirs:
        df = pd.read_parquet(parquet_path)
        assert "faultAnnotation" in df.columns
        assert "signal_key" in df.columns
        assert "signal_length" in df.columns
        npz_path = parquet_path.with_name("cleaned_windows.npz")
        assert npz_path.exists(), "Cleaned signal archive missing"
        store = np.load(npz_path)
        first_row = df.iloc[0]
        key = str(first_row["signal_key"])
        assert key in store.files, f"Signal key {key} missing from archive"
        assert store[key].shape[0] == int(first_row["signal_length"])


def test_feature_engineering_outputs(pipeline_outputs):
    base_path = ROOT / "features" / "2_feature_engineering" / "classic_stats" / "base_features.parquet"
    assert base_path.exists(), "Base features parquet missing"
    df = pd.read_parquet(base_path)
    assert "faultAnnotation" in df.columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
    assert len(numeric_cols) > 0, "No numeric features produced"


def test_feature_combination_outputs(pipeline_outputs):
    combined_path = ROOT / "features" / "3_feature_comb_expansion" / "mathematical_combination" / "combined_features.parquet"
    assert combined_path.exists(), "Combined features parquet missing"
    df = pd.read_parquet(combined_path)
    original_cols = {"station_id", "measurement_id", "faultAnnotation"}
    assert original_cols.issubset(df.columns)
    doc_path = ROOT / "features" / "3_feature_comb_expansion" / "mathematical_combination" / "feature_combinations.md"
    assert doc_path.exists(), "Feature combinations documentation missing"


def test_feature_selection_outputs(pipeline_outputs):
    base_dir = ROOT / "features" / "4_feature_selection" / "tracks" / "featurewiz_corr_xgb"
    selected_path = base_dir / "selected_features.parquet"
    summary_path = base_dir / "featurewiz_summary.json"
    assert selected_path.exists(), "Selected features parquet missing"
    assert summary_path.exists(), "Featurewiz summary missing"
    df = pd.read_parquet(selected_path)
    assert "faultAnnotation" in df.columns


def test_model_training_outputs(pipeline_outputs):
    metrics_path = ROOT / "results" / "model_metrics.csv"
    assert metrics_path.exists(), "Model metrics CSV missing"
    metrics_df = pd.read_csv(metrics_path)
    expected_models = {"svm", "random_forest", "decision_tree", "knn", "ann"}
    assert expected_models.issubset(set(metrics_df["model"].unique()))
    for model in expected_models:
        model_file = ROOT / "models" / "trained" / f"{model}.joblib"
        pred_file = ROOT / "results" / f"predictions_{model}.parquet"
        assert model_file.exists(), f"Trained model artifact missing for {model}"
        assert pred_file.exists(), f"Prediction file missing for {model}"


def test_reporting_outputs(pipeline_outputs):
    report_path = ROOT / "reports" / "pipeline_report.md"
    assert report_path.exists(), "Pipeline report markdown missing"
    confusion_plots = list((ROOT / "reports").glob("confusion_*.png"))
    assert confusion_plots, "Confusion matrix plots not generated"

