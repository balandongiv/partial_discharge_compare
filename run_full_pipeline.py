#!/usr/bin/env python
"""
=============================================================================
PARTIAL DISCHARGE CLASSIFICATION PIPELINE - SINGLE-CLICK EXECUTION
=============================================================================

This script executes the complete PD classification pipeline from raw data 
to final results with a single click/run. It automatically:

1. Discovers raw data from dataset/contactless_pd_detection/
2. Cleans and preprocesses signals (Phase 0)
3. Extracts base features (Phase 1) - Classic stats, Time-frequency (FFT, STFT)
4. Expands features with mathematical combinations (Phase 1b) - Pairwise ops, PolynomialFeatures, feature-engine
5. Selects optimal features using multiple tracks (Phase 2):
   - Track 4B: Featurewiz (SULOV + XGBoost)
   - Track 4C: MLJAR-supervised (internal selector)
6. Trains ML models: SVM, RF, DT, ANN, KNN (Phase 2b)
7. Generates evaluation reports with visualizations (Phase 3)

CHECKPOINT/RESUME: Automatically saves progress after each phase and resumes
from the last completed step if interrupted. No need to restart from scratch!

OUTPUT LOCATIONS:
-----------------
- Preprocessed data:  outputs/preprocessing/station_<ID>/data_clean/
- Base features:      features/2_feature_engineering/classic_stats/
- Expanded features:  features/3_feature_comb_expansion/mathematical_combination/
- Selected features:  features/4_feature_selection/
- Trained models:     models/trained/track_4b/, models/trained/track_4c/
- Predictions:        results/track_4b/, results/track_4c/
- Final reports:      reports/track_4b/, reports/track_4c/

USAGE:
------
Simply run this file:
    python run_full_pipeline.py

Or from IDE: Click Run/Execute

Author: Auto-generated for PD Classification Pipeline
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.signal import butter, filtfilt, stft
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving figures
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from feature_engine.creation import MathFeatures
    HAS_FEATURE_ENGINE = True
except ImportError:
    HAS_FEATURE_ENGINE = False
    print("[WARNING] feature_engine not installed. Mathematical combinations will be limited.")

try:
    from featurewiz import featurewiz
    HAS_FEATUREWIZ = True
except ImportError:
    HAS_FEATUREWIZ = False
    print("[WARNING] featurewiz not installed. Feature selection will use variance-based fallback.")

try:
    from supervised.automl import AutoML
    HAS_MLJAR = True
except ImportError:
    HAS_MLJAR = False
    print("[WARNING] mljar-supervised not installed. MLJAR feature selection track will be skipped.")


# =============================================================================
# CONFIGURATION
# =============================================================================

PIPELINE_PATHS_FILE = Path(__file__).parent.resolve() / "run_full_pipeline.yaml"


def _normalize_user_path(value: str) -> Path:
    """Normalize a user-provided path string into a Path."""
    cleaned = value.strip().strip('"').strip("'")
    return Path(cleaned).expanduser()


def _prompt_for_path(prompt: str, default: Path, must_exist: bool) -> Path:
    """Prompt the user for a path, falling back to a default value."""
    while True:
        response = input(f"{prompt} [{default}]: ").strip()
        if not response:
            candidate = default
        else:
            candidate = _normalize_user_path(response)

        if must_exist and not candidate.exists():
            print(f"[ERROR] Path not found: {candidate}")
            continue
        return candidate


def _load_pipeline_paths(config_path: Path, project_root: Path) -> Dict[str, Path]:
    """Load dataset/output roots from YAML, prompting the user on first run."""
    data: Dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except yaml.YAMLError as exc:
            print(f"[WARNING] Failed to parse {config_path}: {exc}")

    dataset_root_raw = str(data.get("dataset_root", "")).strip()
    output_root_raw = str(data.get("output_root", "")).strip()

    needs_prompt = not dataset_root_raw or not output_root_raw
    if needs_prompt:
        print("\nFirst run: please provide dataset and output locations.")
        default_dataset = project_root / "dataset"
        default_output = project_root
        dataset_root = _prompt_for_path("Dataset root folder", default_dataset, must_exist=True)
        output_root = _prompt_for_path("Output base folder", default_output, must_exist=False)
        output_root.mkdir(parents=True, exist_ok=True)

        config_payload = {
            "dataset_root": dataset_root.resolve().as_posix(),
            "output_root": output_root.resolve().as_posix(),
        }
        config_path.write_text(
            yaml.safe_dump(config_payload, sort_keys=False),
            encoding="utf-8",
        )
        print(f"[INFO] Saved pipeline paths to {config_path}")
    else:
        dataset_root = _normalize_user_path(dataset_root_raw)
        output_root = _normalize_user_path(output_root_raw)

    return {
        "dataset_root": dataset_root.resolve(),
        "output_root": output_root.resolve(),
    }


class PipelineConfig:
    """Central configuration for the pipeline."""

    def __init__(self, dataset_root: Path, output_root: Path, project_root: Optional[Path] = None) -> None:
        # Paths
        self.PROJECT_ROOT = project_root or Path(__file__).parent.resolve()
        self.DATASET_ROOT = dataset_root
        self.RAW_DATA_DIR = self.DATASET_ROOT / "contactless_pd_detection"
        self.ANNOTATION_FILE = self.DATASET_ROOT / "inferred_annotation.csv"

        # Output directories (relative to output_root)
        self.OUTPUT_BASE = output_root
        self.OUTPUT_ROOT = self.OUTPUT_BASE / "outputs" / "preprocessing"
        self.FEATURES_ROOT = self.OUTPUT_BASE / "features"
        self.MODELS_DIR = self.OUTPUT_BASE / "models" / "trained"
        self.RESULTS_DIR = self.OUTPUT_BASE / "results"
        self.REPORTS_DIR = self.OUTPUT_BASE / "reports"
        self.CHECKPOINT_FILE = self.OUTPUT_BASE / "pipeline_checkpoint.json"

        # Signal processing parameters
        self.SAMPLING_FREQ = 1_000_000.0  # 1 MHz
        self.BANDPASS_LOW = 1e3           # 1 kHz
        self.BANDPASS_HIGH = 450_000.0    # 450 kHz (45% of fs)

        # Feature selection
        self.CORR_LIMIT = 0.7  # Correlation limit for featurewiz

        # Model training
        self.RANDOM_STATE = 42
        self.N_JOBS = -1  # Use all cores

        # Models to train (SVM, RF, DT, ANN, KNN only - as requested)
        self.MODELS_TO_TRAIN = ["svm", "random_forest", "decision_tree", "knn", "ann"]


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging with console and file output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("PDPipeline")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)
    
    # File handler
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================

def _infer_completed_phases(config: PipelineConfig) -> List[str]:
    """Infer completed phases from existing output artifacts."""
    phases: List[str] = []
    if (config.OUTPUT_ROOT.exists() and
        any(config.OUTPUT_ROOT.glob("station_*/data_clean/standard_denoising_normalisation/cleaned_windows.parquet"))):
        phases.append("preprocessing")
    
    if (config.FEATURES_ROOT / "2_feature_engineering" / "classic_stats" / "base_features.parquet").exists():
        phases.append("feature_engineering")
    
    if (config.FEATURES_ROOT / "3_feature_comb_expansion" / "mathematical_combination" / "combined_features.parquet").exists():
        phases.append("feature_expansion")
    
    # Feature selection tracks
    if (config.FEATURES_ROOT / "4_feature_selection" / "tracks" / "featurewiz_corr_xgb" / "selected_features.parquet").exists():
        phases.append("feature_selection_4b")
    
    if (config.FEATURES_ROOT / "4_feature_selection" / "tracks" / "mljar_internal" / "selected_features.parquet").exists():
        phases.append("feature_selection_4c")
    
    if ((config.RESULTS_DIR / "track_4b" / "model_metrics.csv").exists() or
        (config.RESULTS_DIR / "track_4c" / "model_metrics.csv").exists() or
        (config.RESULTS_DIR / "model_metrics.csv").exists()):
        phases.append("model_training")
    
    if ((config.REPORTS_DIR / "track_4b" / "pipeline_report.md").exists() or
        (config.REPORTS_DIR / "track_4c" / "pipeline_report.md").exists() or
        (config.REPORTS_DIR / "pipeline_report.md").exists()):
        phases.append("evaluation")
    
    return phases


def save_checkpoint(config: PipelineConfig, phase: str, status: str = "completed") -> None:
    """Save pipeline checkpoint to resume from later."""
    checkpoint_data = {
        "last_phase": phase,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "phases_completed": _infer_completed_phases(config),
    }
    
    config.CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_FILE.write_text(json.dumps(checkpoint_data, indent=2), encoding="utf-8")


def load_checkpoint(config: PipelineConfig) -> Dict[str, Any]:
    """Load pipeline checkpoint to determine where to resume."""
    if not config.CHECKPOINT_FILE.exists():
        inferred = _infer_completed_phases(config)
        if inferred:
            checkpoint_data = {
                "last_phase": inferred[-1],
                "status": "inferred",
                "timestamp": datetime.now().isoformat(),
                "phases_completed": inferred,
            }
            config.CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            config.CHECKPOINT_FILE.write_text(json.dumps(checkpoint_data, indent=2), encoding="utf-8")
            return checkpoint_data
        return {"last_phase": None, "phases_completed": []}
    
    try:
        data = json.loads(config.CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return {
            "last_phase": data.get("last_phase"),
            "phases_completed": data.get("phases_completed", []),
            "timestamp": data.get("timestamp")
        }
    except Exception:
        return {"last_phase": None, "phases_completed": []}


def should_skip_phase(phase_name: str, checkpoint: Dict[str, Any]) -> bool:
    """Determine if a phase should be skipped based on checkpoint."""
    return phase_name in checkpoint.get("phases_completed", [])


# =============================================================================
# PHASE 0: DATA DISCOVERY & PREPROCESSING
# =============================================================================

def discover_stations(raw_data_dir: Path) -> List[str]:
    """Discover all station IDs from the dataset directory."""
    station_ids = []
    if not raw_data_dir.exists():
        return station_ids
    
    for p in raw_data_dir.iterdir():
        if p.is_dir() and p.name.startswith("station_"):
            station_ids.append(p.name.split("station_")[-1])
    
    return sorted(station_ids)


def bandpass_filter(x: np.ndarray, low_hz: float, high_hz: float, fs: float) -> np.ndarray:
    """Apply bandpass Butterworth filter."""
    nyq = 0.5 * fs
    low = max(1e-9, min(low_hz, nyq * 0.99)) / nyq
    high = max(1e-9, min(high_hz, nyq * 0.99)) / nyq
    
    if not (0.0 < low < high < 1.0):
        return x
    
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, x)


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Apply z-score normalization."""
    mean = float(np.mean(x))
    std = float(np.std(x)) or 1.0
    return (x - mean) / std


def run_preprocessing(config: PipelineConfig, logger: logging.Logger) -> Dict[str, int]:
    """
    Phase 0: Clean raw signals and save preprocessed data.
    
    Input:  dataset/contactless_pd_detection/station_<ID>/*.npy
    Output: outputs/preprocessing/station_<ID>/data_clean/standard_denoising_normalisation/
    """
    logger.info("=" * 60)
    logger.info("PHASE 0: DATA PREPROCESSING")
    logger.info("=" * 60)
    
    # Load annotations
    annotations: Optional[pd.DataFrame] = None
    if config.ANNOTATION_FILE.exists():
        annotations = pd.read_csv(
            config.ANNOTATION_FILE,
            usecols=["idStation", "idMeasurement", "faultAnnotation"],
        )
        annotations["idStation"] = annotations["idStation"].astype(int)
        annotations["idMeasurement"] = annotations["idMeasurement"].astype(int)
        annotations["faultAnnotation"] = annotations["faultAnnotation"].astype(str)
        logger.info(f"Loaded {len(annotations)} annotation records")
    else:
        logger.warning(f"Annotation file not found: {config.ANNOTATION_FILE}")
    
    # Discover stations
    station_ids = discover_stations(config.RAW_DATA_DIR)
    if not station_ids:
        logger.error("No stations found in dataset directory!")
        return {}
    
    logger.info(f"Discovered {len(station_ids)} stations: {station_ids}")
    
    summary: Dict[str, int] = {}
    
    for sid in station_ids:
        station_raw = config.RAW_DATA_DIR / f"station_{sid}"
        out_dir = config.OUTPUT_ROOT / f"station_{sid}" / "data_clean" / "standard_denoising_normalisation"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        npy_files = list(station_raw.glob("*.npy"))
        logger.info(f"Processing station {sid}: {len(npy_files)} files")
        
        rows: List[Dict[str, Any]] = []
        count = 0
        
        for npy_file in sorted(npy_files):
            try:
                # Load raw signal
                x = np.load(npy_file)
                if x.ndim > 1:
                    x = x.squeeze()
                
                # Apply preprocessing
                x = bandpass_filter(x, config.BANDPASS_LOW, config.BANDPASS_HIGH, config.SAMPLING_FREQ)
                x = zscore_normalize(x)
                x = x.astype(np.float32, copy=False)
                
                key = npy_file.stem
                count += 1
                
                # Save each cleaned signal as its own .npy file to avoid keeping
                # all signals in memory at once (prevents OOM for large stations)
                clean_path = out_dir / f"{key}.npy"
                np.save(clean_path, x)
                
                rows.append({
                    "station_id": int(sid),
                    "measurement_id": int(key),
                    "signal_key": str(key),
                    "signal_length": int(x.shape[0]),
                    "signal_path": str(clean_path),
                })
            except Exception as err:
                logger.error(f"Failed to process {npy_file}: {err}")
        
        # Save metadata with labels
        if rows:
            df = pd.DataFrame(rows)
            if annotations is not None:
                df = df.merge(
                    annotations,
                    how="left",
                    left_on=["station_id", "measurement_id"],
                    right_on=["idStation", "idMeasurement"],
                )
                df = df.drop(columns=["idStation", "idMeasurement"], errors="ignore")
            
            if "faultAnnotation" not in df.columns:
                df["faultAnnotation"] = "unknown"
            df["faultAnnotation"] = df["faultAnnotation"].fillna("unknown").astype(str)
            
            parquet_path = out_dir / "cleaned_windows.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Station {sid}: cleaned {count} files -> {parquet_path}")
        
        summary[sid] = count
    
    logger.info(f"Preprocessing complete. Total files processed: {sum(summary.values())}")
    return summary


# =============================================================================
# PHASE 1: FEATURE ENGINEERING
# =============================================================================

def compute_entropy_features(x: np.ndarray) -> Dict[str, float]:
    """Compute entropy-based features."""
    hist, _ = np.histogram(x, bins=64, density=True)
    hist = hist + 1e-12
    shannon_entropy = float(stats.entropy(hist))
    
    spectral = np.abs(np.fft.rfft(x))
    spectral = spectral / (spectral.sum() + 1e-12)
    spectral_entropy = float(stats.entropy(spectral + 1e-12))
    
    return {
        "shannon_entropy": shannon_entropy,
        "spectral_entropy": spectral_entropy,
    }


def higuchi_fractal_dimension(x: np.ndarray, kmax: int = 5) -> float:
    """Compute Higuchi fractal dimension."""
    n = len(x)
    if n < 2:
        return 0.0
    
    lk = []
    logk = []
    
    for k in range(1, kmax + 1):
        lk_sum = 0.0
        count = 0
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            diff = np.abs(np.diff(x[idx])).sum()
            norm = (n - 1) / (len(idx) * k)
            lk_sum += diff * norm
            count += 1
        if count > 0 and lk_sum > 0:
            lk.append(np.log(lk_sum / count))
            logk.append(np.log(1.0 / k))
    
    if len(lk) < 2:
        return 0.0
    
    slope, _ = np.polyfit(logk, lk, 1)
    return float(-slope)


def compute_prpd_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute Phase-Resolved Partial Discharge features."""
    if fs <= 0:
        fs = 1.0
    
    mains_freq = 50.0
    idx = np.arange(len(x))
    phase = (idx / fs) * mains_freq * 2 * np.pi
    phase = np.mod(phase, 2 * np.pi)
    abs_x = np.abs(x)
    total = len(x)
    
    features: Dict[str, float] = {}
    edges = np.linspace(0.0, 2 * np.pi, 5)
    
    for quadrant in range(4):
        mask = (phase >= edges[quadrant]) & (phase < edges[quadrant + 1])
        quadrant_values = abs_x[mask]
        mean_val = float(np.mean(quadrant_values)) if quadrant_values.size else 0.0
        density = float(quadrant_values.size / total) if total else 0.0
        features[f"prpd_q{quadrant + 1}_mean"] = mean_val
        features[f"prpd_q{quadrant + 1}_density"] = density
    
    positive = np.sum(x > 0)
    negative = max(np.sum(x < 0), 1)
    features["prpd_polarity_ratio"] = float(positive / negative)
    features["prpd_peak_amplitude"] = float(np.max(abs_x)) if total else 0.0
    features["prpd_energy"] = float(np.sum(abs_x ** 2) / total) if total else 0.0
    
    return features


def compute_basic_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute all basic features for a signal."""
    x = np.asarray(x, dtype=np.float32)
    
    if x.size == 0:
        return {key: 0.0 for key in [
            "rms", "skewness", "kurtosis", "crest_factor", "variance",
            "fft_mean", "fft_max", "stft_mean", "stft_max",
            "shannon_entropy", "spectral_entropy", "fractal_dimension",
            "prpd_q1_mean", "prpd_q2_mean", "prpd_q3_mean", "prpd_q4_mean",
            "prpd_q1_density", "prpd_q2_density", "prpd_q3_density", "prpd_q4_density",
            "prpd_polarity_ratio", "prpd_peak_amplitude", "prpd_energy",
        ]}
    
    # Time-domain statistics
    rms = float(np.sqrt(np.mean(np.square(x))))
    skew = float(np.nan_to_num(stats.skew(x, bias=False), nan=0.0))
    kurt = float(np.nan_to_num(stats.kurtosis(x, bias=False), nan=0.0))
    peak = float(np.max(np.abs(x)))
    crest = float(peak / (rms if rms > 0 else 1.0))
    var = float(np.var(x))
    
    # FFT summary
    fft_vals = np.abs(np.fft.rfft(x))
    fft_mean = float(np.mean(fft_vals)) if fft_vals.size else 0.0
    fft_max = float(np.max(fft_vals)) if fft_vals.size else 0.0
    
    # STFT summary
    _, _, Zxx = stft(x, fs=fs, nperseg=min(256, max(8, len(x))))
    mag = np.abs(Zxx)
    stft_mean = float(np.mean(mag)) if mag.size else 0.0
    stft_max = float(np.max(mag)) if mag.size else 0.0
    
    # Entropy features
    entropy_feats = compute_entropy_features(x)
    
    # Fractal dimension
    fractal_dim = higuchi_fractal_dimension(x)
    
    # PRPD features
    prpd_feats = compute_prpd_features(x, fs)
    
    return {
        "rms": rms,
        "skewness": skew,
        "kurtosis": kurt,
        "crest_factor": crest,
        "variance": var,
        "fft_mean": fft_mean,
        "fft_max": fft_max,
        "stft_mean": stft_mean,
        "stft_max": stft_max,
        "shannon_entropy": entropy_feats["shannon_entropy"],
        "spectral_entropy": entropy_feats["spectral_entropy"],
        "fractal_dimension": fractal_dim,
        **prpd_feats,
    }


def run_feature_engineering(config: PipelineConfig, logger: logging.Logger) -> Dict[str, int]:
    """
    Phase 1: Extract features from preprocessed signals.
    
    Input:  outputs/preprocessing/station_<ID>/data_clean/standard_denoising_normalisation/
    Output: features/2_feature_engineering/classic_stats/base_features.parquet
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    # Output directories
    engineering_dir = config.FEATURES_ROOT / "2_feature_engineering"
    classic_stats_dir = engineering_dir / "classic_stats"
    stations_dir = classic_stats_dir / "stations"
    classic_stats_dir.mkdir(parents=True, exist_ok=True)
    stations_dir.mkdir(parents=True, exist_ok=True)
    
    summary: Dict[str, int] = {}
    combined_rows: List[pd.DataFrame] = []
    
    # Find all preprocessed stations
    stations = sorted(config.OUTPUT_ROOT.glob("station_*/"))
    logger.info(f"Found {len(stations)} preprocessed stations")
    
    for station_dir in stations:
        clean_dir = station_dir / "data_clean" / "standard_denoising_normalisation"
        parquet_file = clean_dir / "cleaned_windows.parquet"
        
        if not parquet_file.exists():
            logger.warning(f"Missing parquet for {station_dir.name}")
            continue
        
        clean_df = pd.read_parquet(parquet_file)
        
        if clean_df.empty:
            logger.warning(f"No cleaned windows for {station_dir.name}")
            continue
        
        logger.info(f"Extracting features from {station_dir.name}: {len(clean_df)} signals")
        
        rows: List[Dict[str, Any]] = []
        for record in clean_df.itertuples(index=False):
            signal_key = str(getattr(record, "signal_key", getattr(record, "measurement_id")))
            signal_path = getattr(record, "signal_path", None)
            
            if not signal_path:
                logger.warning(f"Missing signal_path for key {signal_key} in {station_dir.name}")
                continue
            
            signal_file = Path(signal_path)
            if not signal_file.exists():
                logger.warning(f"Signal file not found: {signal_file}")
                continue
            
            try:
                signal = np.load(signal_file)
                signal = np.asarray(signal, dtype=np.float32)
            except Exception as err:
                logger.warning(f"Failed to load signal {signal_file}: {err}")
                continue
            
            if signal.size == 0:
                continue
            
            feats = compute_basic_features(signal, config.SAMPLING_FREQ)
            feats["station_id"] = str(record.station_id)
            feats["measurement_id"] = int(record.measurement_id)
            feats["faultAnnotation"] = str(getattr(record, "faultAnnotation", "unknown"))
            feats["signal_length"] = int(getattr(record, "signal_length", signal.size))
            rows.append(feats)
        
        if not rows:
            continue
        
        station_df = pd.DataFrame(rows)
        station_df["faultAnnotation"] = station_df["faultAnnotation"].fillna("unknown").astype(str)
        
        # Convert numeric columns to float32
        numeric_cols = [c for c in station_df.columns 
                       if c not in {"station_id", "measurement_id", "faultAnnotation"} 
                       and np.issubdtype(station_df[c].dtype, np.number)]
        station_df[numeric_cols] = station_df[numeric_cols].astype(np.float32, copy=False)
        
        # Save per-station features
        station_out = stations_dir / f"{station_dir.name}.parquet"
        station_df.to_parquet(station_out, index=False)
        summary[station_dir.name] = len(station_df)
        combined_rows.append(station_df)
        
        logger.info(f"  -> Extracted {len(station_df)} feature rows")
    
    # Combine all stations
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_df = combined_df.sort_values(["station_id", "measurement_id"]).reset_index(drop=True)
        
        numeric_cols = [c for c in combined_df.columns 
                       if c not in ["station_id", "measurement_id", "faultAnnotation"] 
                       and np.issubdtype(combined_df[c].dtype, np.number)]
        combined_df[numeric_cols] = combined_df[numeric_cols].astype(np.float32, copy=False)
        
        base_parquet = classic_stats_dir / "base_features.parquet"
        base_csv = classic_stats_dir / "base_features.csv"
        combined_df.to_parquet(base_parquet, index=False)
        combined_df.to_csv(base_csv, index=False)
        
        logger.info(f"Combined base features saved: {len(combined_df)} rows, {len(numeric_cols)} features")
        logger.info(f"  -> {base_parquet}")
    else:
        logger.warning("No feature rows generated!")
    
    return summary


# =============================================================================
# PHASE 1b: FEATURE EXPANSION
# =============================================================================

def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:     
    """Safe division avoiding division by zero."""
    out = np.zeros_like(a, dtype=np.float32)
    mask = np.abs(b) > 1e-9
    np.divide(a, b, out=out, where=mask)
    return out


def run_feature_expansion(config: PipelineConfig, logger: logging.Logger) -> Dict[str, int]:
    """
    Phase 1b: Expand features with mathematical combinations.
    
    Input:  features/2_feature_engineering/classic_stats/base_features.parquet
    Output: features/3_feature_comb_expansion/mathematical_combination/combined_features.parquet
    """
    logger.info("=" * 60)
    logger.info("PHASE 1b: FEATURE EXPANSION")
    logger.info("=" * 60)
    
    base_path = config.FEATURES_ROOT / "2_feature_engineering" / "classic_stats" / "base_features.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Base features not found: {base_path}")
    
    base_df = pd.read_parquet(base_path)
    if base_df.empty:
        raise ValueError("Base features dataframe is empty!")
    
    logger.info(f"Loaded base features: {len(base_df)} rows")
    
    meta_cols = ["station_id", "measurement_id", "faultAnnotation", "signal_length"]
    meta_cols = [c for c in meta_cols if c in base_df.columns]
    feature_cols = [c for c in base_df.columns if c not in meta_cols]
    
    numeric_df = base_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    
    logger.info(f"Base features: {len(feature_cols)} columns")
    
    combination_df = pd.DataFrame(index=base_df.index)
    
    # Pairwise arithmetic combinations
    logger.info("Computing pairwise arithmetic combinations...")
    for col_i, col_j in combinations(feature_cols, 2):
        # Addition
        combination_df[f"{col_i}_plus_{col_j}"] = (numeric_df[col_i] + numeric_df[col_j]).astype(np.float32)
        # Subtraction
        combination_df[f"{col_i}_minus_{col_j}"] = (numeric_df[col_i] - numeric_df[col_j]).astype(np.float32)
        # Division
        combination_df[f"{col_i}_div_{col_j}"] = safe_divide(
            numeric_df[col_i].to_numpy(), 
            numeric_df[col_j].to_numpy()
        ).astype(np.float32)
        # Absolute difference
        combination_df[f"{col_i}_abs_diff_{col_j}"] = np.abs(numeric_df[col_i] - numeric_df[col_j]).astype(np.float32)
        # Multiplication
        combination_df[f"{col_i}_times_{col_j}"] = (numeric_df[col_i] * numeric_df[col_j]).astype(np.float32)
    
    # Polynomial interactions
    logger.info("Computing polynomial interactions...")
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_data = poly.fit_transform(numeric_df)
    poly_names = poly.get_feature_names_out(feature_cols)
    
    interaction_start = len(feature_cols)
    for idx, name in enumerate(poly_names[interaction_start:], start=interaction_start):
        terms = name.split(" ")
        if len(terms) != 2:
            continue
        col_i, col_j = terms
        mult_name = f"{col_i}_mult_{col_j}"
        combination_df[mult_name] = poly_data[:, idx].astype(np.float32)
    
    # Feature-engine MathFeatures (if available)
    if HAS_FEATURE_ENGINE:
        logger.info("Computing feature-engine mathematical combinations...")
        math_operations = ["sum", "prod", "mean"]
        # Collect all new columns in a dictionary to avoid DataFrame fragmentation
        mc_columns = {}
        for col_i, col_j in combinations(feature_cols, 2):
            try:
                math_combiner = MathFeatures(
                    variables=[col_i, col_j],
                    func=math_operations,
                    new_variables_names=[
                        f"sum_{col_i}_{col_j}",
                        f"prod_{col_i}_{col_j}",
                        f"mean_{col_i}_{col_j}",
                    ],
                    missing_values="ignore",
                    drop_original=False,
                )
                pair_df = numeric_df[[col_i, col_j]].copy()
                mc_df = math_combiner.fit_transform(pair_df)
                new_cols = [c for c in mc_df.columns if c not in pair_df.columns]
                for original_name in new_cols:
                    mc_name = f"mc_{original_name}"
                    mc_columns[mc_name] = mc_df[original_name].astype(np.float32)
            except Exception:
                pass  # Skip if combination fails
        
        # Concatenate all collected columns at once to avoid fragmentation
        if mc_columns:
            mc_df_new = pd.DataFrame(mc_columns, index=base_df.index)
            combination_df = pd.concat([combination_df, mc_df_new], axis=1)
    
    # Remove duplicates
    combination_df = combination_df.loc[:, ~combination_df.columns.duplicated()]
    
    # Combine with original features
    combined_features = pd.concat([numeric_df, combination_df], axis=1)
    final_df = pd.concat([base_df[meta_cols], combined_features], axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    
    # Save
    expansion_root = config.FEATURES_ROOT / "3_feature_comb_expansion"
    math_dir = expansion_root / "mathematical_combination"
    math_dir.mkdir(parents=True, exist_ok=True)
    
    combined_path = math_dir / "combined_features.parquet"
    combined_csv = math_dir / "combined_features.csv"
    final_df.to_parquet(combined_path, index=False)
    final_df.to_csv(combined_csv, index=False)
    
    total_features = final_df.shape[1] - len(meta_cols)
    logger.info(f"Feature expansion complete:")
    logger.info(f"  Base features: {len(feature_cols)}")
    logger.info(f"  Combination features: {combination_df.shape[1]}")
    logger.info(f"  Total features: {total_features}")
    logger.info(f"  -> {combined_path}")
    
    return {
        "base_features": len(feature_cols),
        "combination_features": combination_df.shape[1],
        "total_features": total_features,
    }


# =============================================================================
# PHASE 2: FEATURE SELECTION (Tracks 4B: Featurewiz, 4C: MLJAR)
# =============================================================================

def run_feature_selection(config: PipelineConfig, logger: logging.Logger) -> Dict[str, Any]:
    """
    Phase 2: Select optimal features using multiple tracks.
    
    Track 4B: Featurewiz (SULOV + XGBoost ranking)
    Track 4C: MLJAR-supervised (internal selector)
    
    Input:  features/3_feature_comb_expansion/mathematical_combination/combined_features.parquet
    Output: features/4_feature_selection/tracks/<track_name>/selected_features.parquet
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: FEATURE SELECTION (Tracks 4B & 4C)")
    logger.info("=" * 60)
    
    combined_path = config.FEATURES_ROOT / "3_feature_comb_expansion" / "mathematical_combination" / "combined_features.parquet"
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined features not found: {combined_path}")
    
    df = pd.read_parquet(combined_path)
    if df.empty:
        raise ValueError("Combined features dataframe is empty!")
    
    logger.info(f"Loaded combined features: {len(df)} rows")
    
    if "faultAnnotation" not in df.columns:
        raise ValueError("Column 'faultAnnotation' not found!")
    
    meta_cols = [c for c in ["station_id", "measurement_id", "faultAnnotation", "signal_length"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    feature_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    
    target_series = df["faultAnnotation"].fillna("unknown").astype(str)
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target_series)
    
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Classes: {list(label_encoder.classes_)}")
    
    selection_root = config.FEATURES_ROOT / "4_feature_selection"
    selection_root.mkdir(parents=True, exist_ok=True)
    tracks_dir = selection_root / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    
    mapping = {int(idx): label for idx, label in enumerate(label_encoder.classes_)}
    all_results = {}
    
    # ========================================================================
    # TRACK 4B: Featurewiz (Correlation pruning + XGBoost)
    # ========================================================================
    logger.info("")
    logger.info("--- Track 4B: Featurewiz (SULOV + XGBoost) ---")
    track_4b_dir = tracks_dir / "featurewiz_corr_xgb"
    track_4b_dir.mkdir(parents=True, exist_ok=True)
    
    if (track_4b_dir / "selected_features.parquet").exists():
        logger.info("Track 4B already completed, loading existing results...")
        fw_selected_df = pd.read_parquet(track_4b_dir / "selected_features.parquet")
        fw_selected = [c for c in fw_selected_df.columns if c not in meta_cols]
    else:
        if HAS_FEATUREWIZ:
            logger.info(f"Running Featurewiz with corr_limit={config.CORR_LIMIT}...")
            
            fw_input = feature_df.copy()
            fw_input["target"] = encoded_target
            
            try:
                fw_selected, _ = featurewiz(
                    fw_input,
                    "target",
                    corr_limit=config.CORR_LIMIT,
                    verbose=0,
                    skip_sulov=False,
                    skip_xgboost=False,
                )
                
                if not fw_selected or len(fw_selected) == 0:
                    logger.warning("Featurewiz returned no features; using all features")
                    fw_selected = feature_cols
                else:
                    # Remove 'target' if it was included
                    fw_selected = [f for f in fw_selected if f != "target"]
                    logger.info(f"Featurewiz selected {len(fw_selected)} features")
            except Exception as e:
                logger.warning(f"Featurewiz failed: {e}. Using variance-based selection.")
                variances = feature_df.var()
                threshold = variances.quantile(0.1)
                fw_selected = variances[variances > threshold].index.tolist()
        else:
            logger.warning("Featurewiz not available. Using variance-based selection...")
            variances = feature_df.var()
            threshold = variances.quantile(0.1)
            fw_selected = variances[variances > threshold].index.tolist()
        
        # Save Track 4B results
        selected_df_4b = df[meta_cols + fw_selected].copy()
        numeric_cols = [c for c in selected_df_4b.columns if c not in meta_cols and np.issubdtype(selected_df_4b[c].dtype, np.number)]
        selected_df_4b[numeric_cols] = selected_df_4b[numeric_cols].astype(np.float32, copy=False)
        
        selected_df_4b.to_parquet(track_4b_dir / "selected_features.parquet", index=False)
        selected_df_4b.to_csv(track_4b_dir / "selected_features.csv", index=False)
        
        summary_4b = {
            "track": "4B",
            "method": "featurewiz" if HAS_FEATUREWIZ else "variance_threshold",
            "total_input_features": len(feature_cols),
            "selected_feature_count": len(fw_selected),
            "selected_features": fw_selected,
        }
        (track_4b_dir / "summary.json").write_text(json.dumps(summary_4b, indent=2), encoding="utf-8")
        (track_4b_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        
        logger.info(f"Track 4B complete: {len(fw_selected)} features selected")
        logger.info(f"  -> {track_4b_dir / 'selected_features.parquet'}")
    
    all_results["track_4b"] = {
        "method": "featurewiz" if HAS_FEATUREWIZ else "variance_threshold",
        "selected_count": len(fw_selected) if 'fw_selected' in locals() else len([c for c in pd.read_parquet(track_4b_dir / "selected_features.parquet").columns if c not in meta_cols])
    }
    
    # ========================================================================
    # TRACK 4C: MLJAR-supervised (Internal selector)
    # ========================================================================
    logger.info("")
    logger.info("--- Track 4C: MLJAR-supervised (Internal selector) ---")
    track_4c_dir = tracks_dir / "mljar_internal"
    track_4c_dir.mkdir(parents=True, exist_ok=True)
    mljar_results_dir = track_4c_dir / "automl_results"
    mljar_results_dir.mkdir(parents=True, exist_ok=True)
    
    if (track_4c_dir / "selected_features.parquet").exists():
        logger.info("Track 4C already completed, loading existing results...")
        mljar_selected_df = pd.read_parquet(track_4c_dir / "selected_features.parquet")
        mljar_selected = [c for c in mljar_selected_df.columns if c not in meta_cols]
    else:
        if HAS_MLJAR:
            logger.info("Running MLJAR AutoML with feature selection enabled...")
            
            # Prepare data for MLJAR
            mljar_df = feature_df.copy()
            mljar_df["target"] = target_series.values  # Use string labels for MLJAR
            
            # Determine task type
            n_classes = len(label_encoder.classes_)
            ml_task = "binary_classification" if n_classes == 2 else "multiclass_classification"
            
            try:
                automl = AutoML(
                    ml_task=ml_task,
                    mode="Explain",  # Fast mode with feature selection
                    total_time_limit=600,  # 10 minutes max
                    model_time_limit=120,  # 2 minutes per model
                    results_path=str(mljar_results_dir),
                    features_selection=True,  # Enable feature selection
                    train_ensemble=True,
                    verbose=0,
                    validation_strategy={
                        "validation_type": "kfold",
                        "k_folds": min(5, max(2, n_classes)),
                        "shuffle": True,
                        "random_seed": config.RANDOM_STATE,
                    }
                )
                
                automl.fit(mljar_df, target_col="target")
                
                # Get feature importance from MLJAR
                try:
                    leaderboard = automl.get_leaderboard()
                    if not leaderboard.empty:
                        # Get features from best model
                        best_model = automl.get_best_model()
                        if hasattr(best_model, "get_feature_importance"):
                            importance_df = best_model.get_feature_importance()
                            if importance_df is not None and not importance_df.empty:
                                # Select features with importance > 0
                                important_features = importance_df[importance_df["importance"] > 0]["feature"].tolist()
                                mljar_selected = [f for f in important_features if f in feature_cols]
                            else:
                                # Fallback: use all features
                                mljar_selected = feature_cols
                        else:
                            # Fallback: use variance threshold
                            variances = feature_df.var()
                            threshold = variances.quantile(0.1)
                            mljar_selected = variances[variances > threshold].index.tolist()
                    else:
                        # Fallback: variance threshold
                        variances = feature_df.var()
                        threshold = variances.quantile(0.1)
                        mljar_selected = variances[variances > threshold].index.tolist()
                except Exception as e:
                    logger.warning(f"Could not extract feature importance from MLJAR: {e}. Using variance threshold.")
                    variances = feature_df.var()
                    threshold = variances.quantile(0.1)
                    mljar_selected = variances[variances > threshold].index.tolist()
                
                logger.info(f"MLJAR selected {len(mljar_selected)} features")
                
            except Exception as e:
                logger.warning(f"MLJAR AutoML failed: {e}. Using variance-based selection.")
                variances = feature_df.var()
                threshold = variances.quantile(0.1)
                mljar_selected = variances[variances > threshold].index.tolist()
        else:
            logger.warning("MLJAR-supervised not available. Using variance-based selection...")
            variances = feature_df.var()
            threshold = variances.quantile(0.1)
            mljar_selected = variances[variances > threshold].index.tolist()
        
        # Save Track 4C results
        selected_df_4c = df[meta_cols + mljar_selected].copy()
        numeric_cols = [c for c in selected_df_4c.columns if c not in meta_cols and np.issubdtype(selected_df_4c[c].dtype, np.number)]
        selected_df_4c[numeric_cols] = selected_df_4c[numeric_cols].astype(np.float32, copy=False)
        
        selected_df_4c.to_parquet(track_4c_dir / "selected_features.parquet", index=False)
        selected_df_4c.to_csv(track_4c_dir / "selected_features.csv", index=False)
        
        summary_4c = {
            "track": "4C",
            "method": "mljar_supervised" if HAS_MLJAR else "variance_threshold",
            "total_input_features": len(feature_cols),
            "selected_feature_count": len(mljar_selected),
            "selected_features": mljar_selected,
        }
        (track_4c_dir / "summary.json").write_text(json.dumps(summary_4c, indent=2), encoding="utf-8")
        (track_4c_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        
        logger.info(f"Track 4C complete: {len(mljar_selected)} features selected")
        logger.info(f"  -> {track_4c_dir / 'selected_features.parquet'}")
    
    all_results["track_4c"] = {
        "method": "mljar_supervised" if HAS_MLJAR else "variance_threshold",
        "selected_count": len(mljar_selected) if 'mljar_selected' in locals() else len([c for c in pd.read_parquet(track_4c_dir / "selected_features.parquet").columns if c not in meta_cols])
    }
    
    logger.info("")
    logger.info("Feature selection complete for all tracks:")
    logger.info(f"  Track 4B (Featurewiz): {all_results['track_4b']['selected_count']} features")
    logger.info(f"  Track 4C (MLJAR): {all_results['track_4c']['selected_count']} features")
    
    return all_results


# =============================================================================
# PHASE 2b: MODEL TRAINING (SVM, RF, DT, ANN, KNN only)
# =============================================================================

def compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray, average: str = "macro") -> float:
    """Compute ROC-AUC score safely."""
    classes = np.unique(y_true)
    if classes.size <= 1 or y_proba.ndim == 1:
        return float("nan")
    try:
        if classes.size == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average=average))
    except ValueError:
        return float("nan")


def run_model_training(config: PipelineConfig, logger: logging.Logger) -> Dict[str, Any]:
    """
    Phase 2b: Train ML models (SVM, RF, DT, ANN, KNN only).
    
    Trains a full model set for each available feature-selection track:
      - Track 4B: Featurewiz-selected features
      - Track 4C: MLJAR-selected features
    
    Per-track metrics and artefacts are saved separately for comparison.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2b: MODEL TRAINING")
    logger.info("=" * 60)
    
    # Ensure results subfolders exist
    (config.RESULTS_DIR / "featurewiz").mkdir(parents=True, exist_ok=True)
    
    # Helper to train models for a single feature-selection track
    def _train_for_track(
        track_name: str,
        selected_path: Path,
        results_dir: Path,
        models_dir: Path,
    ) -> Dict[str, Any]:
        logger.info("")
        logger.info(f"=== MODEL TRAINING for {track_name} ===")
        logger.info(f"Selected features path: {selected_path}")
        
        if not selected_path.exists():
            raise FileNotFoundError(f"Selected features not found for {track_name}: {selected_path}")
        
        df = pd.read_parquet(selected_path)
        if df.empty:
            raise ValueError(f"Selected features dataframe is empty for {track_name}!")
        
        logger.info(f"[{track_name}] Loaded selected features: {len(df)} rows")
        
        target_series = df["faultAnnotation"].fillna("unknown").astype(str)
        meta_cols = [c for c in ["station_id", "measurement_id", "faultAnnotation", "signal_length"] if c in df.columns]
        feature_df = df.drop(columns=[c for c in meta_cols if c in df.columns])
        feature_names = list(feature_df.columns)
        
        X = feature_df.astype(np.float32).to_numpy()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(target_series)
        
        classes, class_counts = np.unique(y, return_counts=True)
        if classes.size < 2:
            raise ValueError(f"Need at least two classes for model training ({track_name})!")
        
        min_class_count = class_counts.min()
        logger.info(f"[{track_name}] Classes: {list(label_encoder.classes_)}")
        logger.info(f"[{track_name}] Class counts: {dict(zip(label_encoder.classes_, class_counts))}")
        logger.info(f"[{track_name}] Features: {len(feature_names)}")
        
        # Configure cross-validation
        desired_folds = 10 if min_class_count >= 10 else 5
        n_splits = min(desired_folds, int(min_class_count))
        
        if n_splits < 2:
            logger.warning(f"[{track_name}] Insufficient samples; using StratifiedShuffleSplit")
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=config.RANDOM_STATE)
            cv_indices = list(splitter.split(X, y))
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
            cv_indices = list(splitter.split(X, y))
        
        logger.info(f"[{track_name}] Using {len(cv_indices)}-fold cross-validation")
        
        # Define models (SVM, RF, DT, ANN, KNN only - as requested)
        models: Dict[str, Any] = {
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=config.RANDOM_STATE)),
            ]),
            "random_forest": RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS
            ),
            "decision_tree": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            "knn": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]),
            "ann": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=300,
                    random_state=config.RANDOM_STATE,
                )),
            ]),
        }
        
        # Output directories (track-specific)
        results_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        results_records: List[Dict[str, Any]] = []
        metrics_rows: List[pd.DataFrame] = []
        meta_df = df[meta_cols].reset_index(drop=True)
        
        for model_name, estimator in models.items():
            if model_name not in config.MODELS_TO_TRAIN:
                continue
            
            logger.info(f"[{track_name}] Training: {model_name}")
            
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
                    y[val_idx], preds, average="macro", zero_division=0
                )
                accuracy = accuracy_score(y[val_idx], preds)
                roc_auc = compute_roc_auc(y[val_idx], proba)
                
                fold_metrics.append({
                    "fold": fold_idx,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc,
                })
            
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
            
            logger.info(f"[{track_name}]  -> Accuracy: {summary_row['accuracy_mean']:.4f} ± {summary_row['accuracy_std']:.4f}")
            logger.info(f"[{track_name}]  -> F1 Score: {summary_row['f1_mean']:.4f} ± {summary_row['f1_std']:.4f}")
            
            # Save predictions (track-specific)
            pred_df = meta_df.copy()
            if "faultAnnotation" in pred_df.columns:
                pred_df = pred_df.rename(columns={"faultAnnotation": "true_label"})
            pred_df["true_label"] = target_series.values
            pred_df["pred_label"] = label_encoder.inverse_transform(y_pred)
            pred_df["pred_encoded"] = y_pred
            for idx, label in enumerate(label_encoder.classes_):
                pred_df[f"prob_{label}"] = y_proba[:, idx]
            
            pred_path = results_dir / f"predictions_{model_name}.parquet"
            pred_df.to_parquet(pred_path, index=False)
            
            # Save trained model (track-specific)
            final_estimator = clone(estimator)
            final_estimator.fit(X, y)
            artifact = {
                "model": final_estimator,
                "feature_names": feature_names,
                "classes": label_encoder.classes_,
            }
            model_path = models_dir / f"{model_name}.joblib"
            joblib.dump(artifact, model_path)
            logger.info(f"[{track_name}]  -> Model saved: {model_path}")
        
        # Save metrics summary (track-specific)
        metrics_summary_df = pd.DataFrame(results_records)
        metrics_summary_path = results_dir / "model_metrics.csv"
        metrics_summary_df.to_csv(metrics_summary_path, index=False)
        
        detailed_metrics_df = pd.concat(metrics_rows, ignore_index=True)
        detailed_metrics_path = results_dir / "model_metrics_detailed.csv"
        detailed_metrics_df.to_csv(detailed_metrics_path, index=False)
        
        # Find best model for this track
        best_model_name = None
        best_model_metrics: Dict[str, Any] = {}
        if not metrics_summary_df.empty:
            ordered = metrics_summary_df.sort_values(["f1_mean", "roc_auc_mean"], ascending=False, ignore_index=True)
            best_row = ordered.iloc[0]
            best_model_name = str(best_row["model"])
            best_model_metrics = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in best_row.items()}
        
        # Save training summary (track-specific)
        training_meta = {
            "track": track_name,
            "models": list(models.keys()),
            "feature_count": len(feature_names),
            "class_labels": label_encoder.classes_.tolist(),
            "splits": len(cv_indices),
            "best_model": {
                "name": best_model_name,
                "metrics": best_model_metrics,
            },
            "results_dir": str(results_dir),
            "models_dir": str(models_dir),
            "timestamp": datetime.now().isoformat(),
        }
        meta_path = results_dir / "training_summary.json"
        meta_path.write_text(json.dumps(training_meta, indent=2), encoding="utf-8")
        
        logger.info(f"[{track_name}] Model training complete!")
        logger.info(f"[{track_name}]  -> Best model: {best_model_name}")
        logger.info(f"[{track_name}]  -> Metrics saved: {metrics_summary_path}")
        
        return training_meta
    
    # Train for each available track
    tracks_meta: Dict[str, Any] = {}
    
    # Track 4B (Featurewiz) - stored in track-specific folders
    track_4b_path = config.FEATURES_ROOT / "4_feature_selection" / "tracks" / "featurewiz_corr_xgb" / "selected_features.parquet"
    if track_4b_path.exists():
        tracks_meta["track_4b"] = _train_for_track(
            "track_4b",
            track_4b_path,
            config.RESULTS_DIR / "track_4b",
            config.MODELS_DIR / "track_4b",
        )
    else:
        logger.warning("Selected features for Track 4B not found; skipping training for Track 4B.")
    
    # Track 4C (MLJAR) - secondary, uses track-specific subdirectories
    track_4c_path = config.FEATURES_ROOT / "4_feature_selection" / "tracks" / "mljar_internal" / "selected_features.parquet"
    if track_4c_path.exists():
        tracks_meta["track_4c"] = _train_for_track(
            "track_4c",
            track_4c_path,
            config.RESULTS_DIR / "track_4c",
            config.MODELS_DIR / "track_4c",
        )
    else:
        logger.warning("Selected features for Track 4C not found; skipping training for Track 4C.")
    
    if not tracks_meta:
        raise RuntimeError("No training was performed: no selected features found for any track.")
    
    # Choose primary track for summary reporting (prefer 4B if available)
    primary_track = "track_4b" if "track_4b" in tracks_meta else next(iter(tracks_meta.keys()))
    primary_meta = tracks_meta[primary_track]
    
    overall_meta = {
        "primary_track": primary_track,
        "models": primary_meta.get("models", []),
        "feature_count": primary_meta.get("feature_count", 0),
        "class_labels": primary_meta.get("class_labels", []),
        "splits": primary_meta.get("splits", 0),
        "best_model": primary_meta.get("best_model", {}),
        "tracks": tracks_meta,
        "timestamp": datetime.now().isoformat(),
    }
    
    logger.info("Model training complete for all tracks.")
    logger.info(f"  -> Primary track: {primary_track}")
    logger.info(f"  -> Best model (primary): {overall_meta['best_model'].get('name')}")
    
    return overall_meta


# =============================================================================
# PHASE 3: EVALUATION & REPORTING
# =============================================================================

def format_metric(mean: float, std: float) -> str:
    """Format metric as mean ± std."""
    if np.isnan(mean):
        return "nan"
    return f"{mean:.3f} ± {std:.3f}"


def run_evaluation_reporting(config: PipelineConfig, logger: logging.Logger) -> Dict[str, Any]:
    """
    Phase 3: Generate evaluation reports and visualizations.
    
    Input:  results/<track>/model_metrics.csv, results/<track>/predictions_*.parquet
    Output: reports/<track>/pipeline_report.md, reports/<track>/confusion_*.png, reports/<track>/roc_*.png
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: EVALUATION & REPORTING")
    logger.info("=" * 60)
    
    def _run_report_for_track(track_name: str, results_dir: Path, reports_dir: Path) -> Optional[Dict[str, Any]]:
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = results_dir / "model_metrics.csv"
        if not metrics_path.exists():
            logger.warning(f"[{track_name}] Model metrics not found: {metrics_path}")
            return None
        
        metrics_df = pd.read_csv(metrics_path)
        logger.info(f"[{track_name}] Loaded metrics for {len(metrics_df)} models")
        
        # Load training metadata
        training_meta_path = results_dir / "training_summary.json"
        training_meta = {}
        if training_meta_path.exists():
            training_meta = json.loads(training_meta_path.read_text(encoding="utf-8"))
        
        best_model_info = training_meta.get("best_model", {})
        best_model_name = best_model_info.get("name")
        best_model_metrics = best_model_info.get("metrics", {})
        
        # Generate plots for each model
        prediction_files = sorted(results_dir.glob("predictions_*.parquet"))
        figures_info = []
        class_balance = None
        
        for pred_path in prediction_files:
            model_name = pred_path.stem.replace("predictions_", "", 1)
            df = pd.read_parquet(pred_path)
            y_true = df["true_label"].astype(str)
            y_pred = df["pred_label"].astype(str)
            labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
            
            if class_balance is None:
                counts = y_true.value_counts()
                class_balance = {label: int(counts.get(label, 0)) for label in labels}
            
            # Confusion Matrix
            if plt is not None:
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
                ax_cm.set_title(f"Confusion Matrix - {model_name}")
                fig_cm.tight_layout()
                cm_path = reports_dir / f"confusion_{model_name}.png"
                fig_cm.savefig(cm_path, dpi=200)
                plt.close(fig_cm)
                logger.info(f"[{track_name}] Saved confusion matrix: {cm_path}")
                
                # ROC Curve
                prob_cols = [c for c in df.columns if c.startswith("prob_")]
                roc_path = None
                if prob_cols and len(prob_cols) >= 2:
                    class_labels = [col.replace("prob_", "", 1) for col in prob_cols]
                    proba = df[prob_cols].to_numpy()
                    y_binarized = label_binarize(y_true, classes=class_labels)
                    
                    if y_binarized.ndim > 1 and y_binarized.shape[1] >= 2:
                        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                        
                        if len(class_labels) == 2:
                            pos_idx = 1
                            fpr, tpr, _ = roc_curve(y_binarized[:, pos_idx], proba[:, pos_idx])
                            roc_auc_val = auc(fpr, tpr)
                            ax_roc.plot(fpr, tpr, label=f"{class_labels[pos_idx]} (AUC={roc_auc_val:.3f})")
                        else:
                            for idx, label in enumerate(class_labels[:proba.shape[1]]):
                                fpr, tpr, _ = roc_curve(y_binarized[:, idx], proba[:, idx])
                                roc_auc_val = auc(fpr, tpr)
                                ax_roc.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_val:.3f})")
                        
                        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
                        ax_roc.set_xlabel("False Positive Rate")
                        ax_roc.set_ylabel("True Positive Rate")
                        ax_roc.set_title(f"ROC Curves - {model_name}")
                        ax_roc.legend(loc="lower right", fontsize="small")
                        ax_roc.grid(True, linestyle="--", alpha=0.4)
                        fig_roc.tight_layout()
                        roc_path = reports_dir / f"roc_{model_name}.png"
                        fig_roc.savefig(roc_path, dpi=200)
                        plt.close(fig_roc)
                        logger.info(f"[{track_name}] Saved ROC curve: {roc_path}")
                
                figures_info.append({"model": model_name, "confusion": cm_path, "roc": roc_path})
        
        # Model comparison chart
        if plt is not None and not metrics_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(metrics_df["model"], metrics_df["f1_mean"], yerr=metrics_df["f1_std"], capsize=4, color='steelblue')
            ax.set_ylabel("F1 Score (mean)")
            ax.set_xlabel("Model")
            ax.set_title("Model F1 Score Comparison")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=45, ha='right')
            fig.tight_layout()
            comparison_path = reports_dir / "model_f1_comparison.png"
            fig.savefig(comparison_path, dpi=200)
            plt.close(fig)
            logger.info(f"[{track_name}] Saved comparison chart: {comparison_path}")
        
        # Generate Markdown report
        report_lines = [
            f"# Partial Discharge Classification Report ({track_name})",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Performance Summary",
            "",
            "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        
        for _, row in metrics_df.iterrows():
            report_lines.append(
                f"| {row['model']} | "
                f"{format_metric(row['accuracy_mean'], row['accuracy_std'])} | "
                f"{format_metric(row['precision_mean'], row['precision_std'])} | "
                f"{format_metric(row['recall_mean'], row['recall_std'])} | "
                f"{format_metric(row['f1_mean'], row['f1_std'])} | "
                f"{format_metric(row['roc_auc_mean'], row['roc_auc_std'])} |"
            )
        
        if class_balance:
            report_lines.extend([
                "",
                "## Class Distribution",
                "",
                "| Label | Count |",
                "| --- | --- |",
            ])
            for label, count in class_balance.items():
                report_lines.append(f"| {label} | {count} |")
        
        if best_model_name:
            report_lines.extend([
                "",
                "## Best Performing Model",
                "",
                f"- **Model:** {best_model_name}",
            ])
            if isinstance(best_model_metrics, dict):
                f1_val = best_model_metrics.get("f1_mean")
                roc_val = best_model_metrics.get("roc_auc_mean")
                if isinstance(f1_val, (int, float)):
                    report_lines.append(f"- **F1 Score (mean):** {float(f1_val):.4f}")
                if isinstance(roc_val, (int, float)):
                    report_lines.append(f"- **ROC-AUC (mean):** {float(roc_val):.4f}")
        
        report_lines.extend([
            "",
            "## Model Comparison",
            "",
            "![Model F1 Comparison](model_f1_comparison.png)",
            "",
            "## Model-Level Diagnostics",
        ])
        
        for info in figures_info:
            report_lines.append(f"### {info['model']}")
            report_lines.append("")
            report_lines.append(f"![Confusion Matrix](confusion_{info['model']}.png)")
            if info.get("roc"):
                report_lines.append("")
                report_lines.append(f"![ROC Curves](roc_{info['model']}.png)")
            report_lines.append("")
        
        # Save report
        report_path = reports_dir / "pipeline_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.info(f"[{track_name}] Markdown report saved: {report_path}")
        
        # Copy metrics to reports
        shutil.copy(metrics_path, reports_dir / "model_metrics.csv")
        
        logger.info(f"[{track_name}] Evaluation and reporting complete!")
        
        return {
            "track": track_name,
            "num_models": len(figures_info),
            "report": str(report_path),
            "best_model": best_model_name,
        }
    
    reports_root = config.REPORTS_DIR
    reports_root.mkdir(parents=True, exist_ok=True)
    
    track_reports: Dict[str, Any] = {}
    
    # Track 4B (Featurewiz) - prefer track folder, fallback to legacy root results
    track_4b_results = config.RESULTS_DIR / "track_4b"
    if not (track_4b_results / "model_metrics.csv").exists() and (config.RESULTS_DIR / "model_metrics.csv").exists():
        track_4b_results = config.RESULTS_DIR
    report_4b = _run_report_for_track("track_4b", track_4b_results, reports_root / "track_4b")
    if report_4b:
        track_reports["track_4b"] = report_4b
    
    # Track 4C (MLJAR)
    track_4c_results = config.RESULTS_DIR / "track_4c"
    report_4c = _run_report_for_track("track_4c", track_4c_results, reports_root / "track_4c")
    if report_4c:
        track_reports["track_4c"] = report_4c
    
    if not track_reports:
        raise RuntimeError("No reports generated: no model metrics found for any track.")
    
    logger.info("Evaluation and reporting complete for all available tracks.")
    
    return {
        "tracks": track_reports,
    }


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def run_full_pipeline():
    """
    Execute the complete PD classification pipeline from raw data to results.
    
    This is the main entry point - just run this file!
    Supports checkpoint/resume functionality to continue from last completed step.
    """
    project_root = Path(__file__).parent.resolve()
    paths = _load_pipeline_paths(PIPELINE_PATHS_FILE, project_root)
    config = PipelineConfig(
        dataset_root=paths["dataset_root"],
        output_root=paths["output_root"],
        project_root=project_root,
    )
    
    # Setup logging
    logger = setup_logging(config.REPORTS_DIR)
    
    print("\n" + "=" * 70)
    print("  PARTIAL DISCHARGE CLASSIFICATION PIPELINE")
    print("  Single-Click Full Execution (with Checkpoint/Resume)")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    logger.info("Pipeline started")
    logger.info(f"Project root: {config.PROJECT_ROOT}")
    logger.info(f"Dataset path: {config.DATASET_ROOT}")
    logger.info(f"Output base: {config.OUTPUT_BASE}")
    
    # Load checkpoint to determine where to resume
    checkpoint = load_checkpoint(config)
    if checkpoint.get("phases_completed"):
        logger.info("")
        logger.info("=" * 60)
        logger.info("CHECKPOINT DETECTED - RESUMING FROM LAST STEP")
        logger.info("=" * 60)
        logger.info(f"Completed phases: {', '.join(checkpoint['phases_completed'])}")
        if checkpoint.get("timestamp"):
            logger.info(f"Last checkpoint: {checkpoint['timestamp']}")
        logger.info("")
    
    results = {}
    
    try:
        # Phase 0: Preprocessing
        if should_skip_phase("preprocessing", checkpoint):
            logger.info("=" * 60)
            logger.info("PHASE 0: DATA PREPROCESSING (SKIPPED - Already completed)")
            logger.info("=" * 60)
            logger.info("Loading existing preprocessing results...")
            # Load summary from existing files
            stations = sorted(config.OUTPUT_ROOT.glob("station_*/"))
            summary = {s.name.split("station_")[-1]: 0 for s in stations}
            for station_dir in stations:
                parquet_file = station_dir / "data_clean" / "standard_denoising_normalisation" / "cleaned_windows.parquet"
                if parquet_file.exists():
                    df = pd.read_parquet(parquet_file)
                    sid = station_dir.name.split("station_")[-1]
                    summary[sid] = len(df)
            results["preprocessing"] = summary
            logger.info(f"Found {len(summary)} preprocessed stations")
        else:
            logger.info("")
            preprocess_result = run_preprocessing(config, logger)
            results["preprocessing"] = preprocess_result
            save_checkpoint(config, "preprocessing", "completed")
        
        # Phase 1: Feature Engineering
        if should_skip_phase("feature_engineering", checkpoint):
            logger.info("=" * 60)
            logger.info("PHASE 1: FEATURE ENGINEERING (SKIPPED - Already completed)")
            logger.info("=" * 60)
            logger.info("Loading existing feature engineering results...")
            base_path = config.FEATURES_ROOT / "2_feature_engineering" / "classic_stats" / "base_features.parquet"
            if base_path.exists():
                df = pd.read_parquet(base_path)
                results["feature_engineering"] = {"total_rows": len(df), "total_features": len(df.columns)}
                logger.info(f"Found {len(df)} feature rows")
        else:
            logger.info("")
            feature_result = run_feature_engineering(config, logger)
            results["feature_engineering"] = feature_result
            save_checkpoint(config, "feature_engineering", "completed")
        
        # Phase 1b: Feature Expansion
        if should_skip_phase("feature_expansion", checkpoint):
            logger.info("=" * 60)
            logger.info("PHASE 1b: FEATURE EXPANSION (SKIPPED - Already completed)")
            logger.info("=" * 60)
            logger.info("Loading existing feature expansion results...")
            expansion_path = config.FEATURES_ROOT / "3_feature_comb_expansion" / "mathematical_combination" / "combined_features.parquet"
            if expansion_path.exists():
                df = pd.read_parquet(expansion_path)
                meta_cols = ["station_id", "measurement_id", "faultAnnotation", "signal_length"]
                feature_cols = [c for c in df.columns if c not in meta_cols]
                results["feature_expansion"] = {
                    "total_features": len(feature_cols),
                    "base_features": 0,  # Would need to recalculate
                    "combination_features": len(feature_cols)
                }
                logger.info(f"Found {len(feature_cols)} expanded features")
        else:
            logger.info("")
            expansion_result = run_feature_expansion(config, logger)
            results["feature_expansion"] = expansion_result
            save_checkpoint(config, "feature_expansion", "completed")
        
        # Phase 2: Feature Selection (Tracks 4B & 4C)
        track_4b_completed = should_skip_phase("feature_selection_4b", checkpoint)
        track_4c_completed = should_skip_phase("feature_selection_4c", checkpoint)
        
        if track_4b_completed and track_4c_completed:
            logger.info("=" * 60)
            logger.info("PHASE 2: FEATURE SELECTION (SKIPPED - Both tracks already completed)")
            logger.info("=" * 60)
            logger.info("Loading existing feature selection results...")
            # Load summaries from both tracks
            track_4b_dir = config.FEATURES_ROOT / "4_feature_selection" / "tracks" / "featurewiz_corr_xgb"
            track_4c_dir = config.FEATURES_ROOT / "4_feature_selection" / "tracks" / "mljar_internal"
            results["feature_selection"] = {}
            if (track_4b_dir / "summary.json").exists():
                summary_4b = json.loads((track_4b_dir / "summary.json").read_text(encoding="utf-8"))
                results["feature_selection"]["track_4b"] = summary_4b
            if (track_4c_dir / "summary.json").exists():
                summary_4c = json.loads((track_4c_dir / "summary.json").read_text(encoding="utf-8"))
                results["feature_selection"]["track_4c"] = summary_4c
            logger.info("Both feature selection tracks found")
        else:
            logger.info("")
            selection_result = run_feature_selection(config, logger)
            results["feature_selection"] = selection_result
            save_checkpoint(config, "feature_selection", "completed")
        
        # Phase 2b: Model Training (SVM, RF, DT, ANN, KNN)
        if should_skip_phase("model_training", checkpoint):
            logger.info("=" * 60)
            logger.info("PHASE 2b: MODEL TRAINING (SKIPPED - Already completed)")
            logger.info("=" * 60)
            logger.info("Loading existing model training results...")
            training_meta = {}
            track_4b_meta_path = config.RESULTS_DIR / "track_4b" / "training_summary.json"
            track_4c_meta_path = config.RESULTS_DIR / "track_4c" / "training_summary.json"
            legacy_meta_path = config.RESULTS_DIR / "training_summary.json"
            if track_4b_meta_path.exists():
                training_meta["track_4b"] = json.loads(track_4b_meta_path.read_text(encoding="utf-8"))
            if track_4c_meta_path.exists():
                training_meta["track_4c"] = json.loads(track_4c_meta_path.read_text(encoding="utf-8"))
            if not training_meta and legacy_meta_path.exists():
                training_meta["track_4b"] = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
            if training_meta:
                results["model_training"] = training_meta
                logger.info(f"Found trained models for tracks: {list(training_meta.keys())}")
        else:
            logger.info("")
            training_result = run_model_training(config, logger)
            results["model_training"] = training_result
            save_checkpoint(config, "model_training", "completed")
        
        # Phase 3: Evaluation & Reporting
        if should_skip_phase("evaluation", checkpoint):
            logger.info("=" * 60)
            logger.info("PHASE 3: EVALUATION & REPORTING (SKIPPED - Already completed)")
            logger.info("=" * 60)
            logger.info("Loading existing evaluation results...")
            report_4b = config.REPORTS_DIR / "track_4b" / "pipeline_report.md"
            report_4c = config.REPORTS_DIR / "track_4c" / "pipeline_report.md"
            reports_found = []
            if report_4b.exists():
                reports_found.append(str(report_4b))
            if report_4c.exists():
                reports_found.append(str(report_4c))
            if reports_found:
                results["evaluation"] = {
                    "reports": reports_found,
                    "status": "completed"
                }
                logger.info(f"Found existing reports: {reports_found}")
        else:
            logger.info("")
            report_result = run_evaluation_reporting(config, logger)
            results["evaluation"] = report_result
            save_checkpoint(config, "evaluation", "completed")
        
        elapsed = time.time() - start_time
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info("")
        logger.info("OUTPUT LOCATIONS:")
        logger.info(f"  - Preprocessed data: {config.OUTPUT_ROOT}")
        logger.info(f"  - Base features:     {config.FEATURES_ROOT / '2_feature_engineering' / 'classic_stats'}")
        logger.info(f"  - Expanded features: {config.FEATURES_ROOT / '3_feature_comb_expansion'}")
        logger.info(f"  - Selected features: {config.FEATURES_ROOT / '4_feature_selection'}")
        logger.info(f"  - Trained models:    {config.MODELS_DIR / 'track_4b'}")
        logger.info(f"                      {config.MODELS_DIR / 'track_4c'}")
        logger.info(f"  - Predictions:       {config.RESULTS_DIR / 'track_4b'}")
        logger.info(f"                      {config.RESULTS_DIR / 'track_4c'}")
        logger.info(f"  - Final reports:     {config.REPORTS_DIR / 'track_4b'}")
        logger.info(f"                      {config.REPORTS_DIR / 'track_4c'}")
        logger.info("")
        
        if training_result.get("best_model"):
            best = training_result["best_model"]
            logger.info(f"BEST MODEL: {best.get('name')}")
            metrics = best.get("metrics", {})
            if metrics:
                logger.info(f"  - F1 Score:  {metrics.get('f1_mean', 'N/A'):.4f}")
                logger.info(f"  - Accuracy:  {metrics.get('accuracy_mean', 'N/A'):.4f}")
                logger.info(f"  - ROC-AUC:   {metrics.get('roc_auc_mean', 'N/A'):.4f}")
        
        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Reports: {config.REPORTS_DIR / 'track_4b'}")
        print(f"           {config.REPORTS_DIR / 'track_4c'}")
        print("=" * 70 + "\n")
        
        # Save final summary
        summary_path = config.REPORTS_DIR / "pipeline_summary.json"
        model_training = results.get("model_training", {})
        if "models" in model_training:
            models_trained = len(model_training.get("models", []))
            best_model = model_training.get("best_model", {})
        else:
            models_trained = 0
            best_model = {}
            for track_meta in model_training.values():
                if isinstance(track_meta, dict):
                    models_trained = max(models_trained, len(track_meta.get("models", [])))
                    if not best_model and track_meta.get("best_model"):
                        best_model = track_meta.get("best_model", {})
        summary_path.write_text(json.dumps({
            "status": "success",
            "execution_time_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
            "results": {
                "stations_processed": len(results.get("preprocessing", {})),
                "features_extracted": results.get("feature_expansion", {}).get("total_features", 0),
                "feature_selection_tracks": {
                    "track_4b": results.get("feature_selection", {}).get("track_4b", {}),
                    "track_4c": results.get("feature_selection", {}).get("track_4c", {}),
                },
                "models_trained": models_trained,
                "best_model": best_model,
            }
        }, indent=2), encoding="utf-8")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        print(f"\n[ERROR] Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_full_pipeline()
