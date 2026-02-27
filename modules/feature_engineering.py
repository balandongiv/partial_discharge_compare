"""
Feature Engineering Module for Partial Discharge Classification Pipeline

This module implements Phase 1 of the PD classification pipeline, responsible for extracting
classical statistical and spectral features from cleaned partial discharge signals. It computes
time-domain, frequency-domain, and time-frequency features to characterize PD signal patterns.

Step-by-Step Process:
1. Data Discovery:
   - Scans partial_discharge_project/station_<ID>/data_clean/z_score_normalisation/
   - Discovers all cleaned .npy signal files for each station
   - Builds processing queue for feature extraction

2. Signal Loading:
   - Loads cleaned PD signals from .npy files
   - Validates signal integrity and handles loading errors
   - Processes signals station by station for organized output

3. Time-Domain Feature Extraction:
   - RMS (Root Mean Square): Signal energy measure
   - Skewness: Third-order moment indicating signal asymmetry
   - Kurtosis: Fourth-order moment indicating signal peakedness
   - Peak Value: Maximum absolute amplitude
   - Crest Factor: Ratio of peak to RMS value
   - Variance: Signal variability measure

4. Frequency-Domain Feature Extraction:
   - FFT Analysis: Computes real FFT of signals
   - FFT Mean: Average magnitude across frequency spectrum
   - FFT Max: Maximum magnitude in frequency domain
   - Spectral characteristics for PD pattern recognition

5. Time-Frequency Analysis:
   - STFT (Short-Time Fourier Transform): Time-frequency representation
   - STFT Mean: Average magnitude across time-frequency plane
   - STFT Max: Maximum magnitude in time-frequency domain

6. Feature Organization:
   - Combines all extracted features into structured dictionary
   - Adds metadata: station_id and original filename
   - Maintains traceability between signals and features

7. Output Generation:
   - Creates features/2_feature_engineering/classic_stats/ directory
   - Saves features as CSV files per station
   - Logs processing progress and feature counts

Feature Categories:
- Statistical Features: RMS, variance, skewness, kurtosis, crest factor
- Spectral Features: FFT mean/max, frequency domain characteristics
- Time-Frequency Features: STFT mean/max, transient analysis

Configuration Parameters:
- fs: Sampling frequency (default: 1,000,000 Hz)
- station_id: Optional specific station to process (processes all if None)
- input_path: Path to cleaned data (partial_discharge_project/)
- output_path: Root directory for feature outputs

Dependencies:
- numpy: Array operations and mathematical functions
- pandas: Data manipulation and CSV export
- scipy.stats: Statistical moment calculations
- scipy.signal: STFT computation and signal processing
- pathlib: Cross-platform path handling
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import stft

from .utils.logger import get_logger


def _compute_entropy_features(x: np.ndarray) -> dict[str, float]:
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


def _higuchi_fractal_dimension(x: np.ndarray, kmax: int = 5) -> float:
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


def _compute_prpd_features(x: np.ndarray, fs: float) -> dict[str, float]:
    if fs <= 0:
        fs = 1.0
    mains_freq = 50.0
    idx = np.arange(len(x))
    phase = (idx / fs) * mains_freq * 2 * np.pi
    phase = np.mod(phase, 2 * np.pi)
    abs_x = np.abs(x)
    total = len(x)
    features: dict[str, float] = {}
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


FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "rms": {
        "source": "classic_stats",
        "domain": "Time",
        "description": "Root Mean Square amplitude",
        "meaning": "Energy intensity of the PD pulse",
    },
    "skewness": {
        "source": "classic_stats",
        "domain": "Time",
        "description": "Third central moment",
        "meaning": "Asymmetry of the discharge waveform",
    },
    "kurtosis": {
        "source": "classic_stats",
        "domain": "Time",
        "description": "Fourth central moment",
        "meaning": "Pulse sharpness and impulsiveness",
    },
    "crest_factor": {
        "source": "classic_stats",
        "domain": "Time",
        "description": "Peak-to-RMS ratio",
        "meaning": "Presence of sharp discharge spikes",
    },
    "variance": {
        "source": "classic_stats",
        "domain": "Time",
        "description": "Amplitude variance",
        "meaning": "Overall variability within the window",
    },
    "fft_mean": {
        "source": "spectral",
        "domain": "Frequency",
        "description": "Mean magnitude of FFT spectrum",
        "meaning": "Average spectral energy distribution",
    },
    "fft_max": {
        "source": "spectral",
        "domain": "Frequency",
        "description": "Maximum FFT magnitude",
        "meaning": "Dominant frequency component strength",
    },
    "stft_mean": {
        "source": "time_frequency",
        "domain": "Time-Frequency",
        "description": "Mean STFT magnitude",
        "meaning": "Average energy across time-frequency bins",
    },
    "stft_max": {
        "source": "time_frequency",
        "domain": "Time-Frequency",
        "description": "Max STFT magnitude",
        "meaning": "Peak localized energy in time-frequency plane",
    },
    "shannon_entropy": {
        "source": "entropy",
        "domain": "Time",
        "description": "Shannon entropy of amplitude histogram",
        "meaning": "Signal complexity and dispersion",
    },
    "spectral_entropy": {
        "source": "entropy",
        "domain": "Frequency",
        "description": "Entropy of normalized FFT magnitudes",
        "meaning": "Spectral flatness indicating broadband activity",
    },
    "fractal_dimension": {
        "source": "fractal",
        "domain": "Time",
        "description": "Higuchi fractal dimension",
        "meaning": "Signal roughness and structural complexity",
    },
    "prpd_q1_mean": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Mean magnitude in quadrant 1",
        "meaning": "Discharge intensity near 0° phase",
    },
    "prpd_q2_mean": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Mean magnitude in quadrant 2",
        "meaning": "Discharge intensity near 90° phase",
    },
    "prpd_q3_mean": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Mean magnitude in quadrant 3",
        "meaning": "Discharge intensity near 180° phase",
    },
    "prpd_q4_mean": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Mean magnitude in quadrant 4",
        "meaning": "Discharge intensity near 270° phase",
    },
    "prpd_q1_density": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Sample density in quadrant 1",
        "meaning": "Pulse occurrence near 0° phase",
    },
    "prpd_q2_density": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Sample density in quadrant 2",
        "meaning": "Pulse occurrence near 90° phase",
    },
    "prpd_q3_density": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Sample density in quadrant 3",
        "meaning": "Pulse occurrence near 180° phase",
    },
    "prpd_q4_density": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Sample density in quadrant 4",
        "meaning": "Pulse occurrence near 270° phase",
    },
    "prpd_polarity_ratio": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Positive-to-negative pulse ratio",
        "meaning": "Dominant discharge polarity",
    },
    "prpd_peak_amplitude": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Maximum PRPD amplitude",
        "meaning": "Strongest discharge magnitude",
    },
    "prpd_energy": {
        "source": "prpd",
        "domain": "Phase-resolved",
        "description": "Energy of phase-resolved discharge pattern",
        "meaning": "Cumulative discharge energy across cycle",
    },
}


def _write_feature_summary(output_dir: Path) -> None:
    summary_path = output_dir / "feature_summary.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature Summary",
        "",
        "| Feature Name | Source Module | Domain | Description | Meaning |",
        "| --- | --- | --- | --- | --- |",
    ]
    for feature in sorted(FEATURE_METADATA):
        meta = FEATURE_METADATA[feature]
        lines.append(
            f"| {feature} | {meta['source']} | {meta['domain']} | {meta['description']} | {meta['meaning']} |"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
def _compute_basic_features(x: np.ndarray, fs: float) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return {
            "rms": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "crest_factor": 0.0,
            "variance": 0.0,
            "fft_mean": 0.0,
            "fft_max": 0.0,
            "stft_mean": 0.0,
            "stft_max": 0.0,
            "shannon_entropy": 0.0,
            "spectral_entropy": 0.0,
            "fractal_dimension": 0.0,
            "prpd_q1_mean": 0.0,
            "prpd_q2_mean": 0.0,
            "prpd_q3_mean": 0.0,
            "prpd_q4_mean": 0.0,
            "prpd_q1_density": 0.0,
            "prpd_q2_density": 0.0,
            "prpd_q3_density": 0.0,
            "prpd_q4_density": 0.0,
            "prpd_polarity_ratio": 0.0,
            "prpd_peak_amplitude": 0.0,
            "prpd_energy": 0.0,
        }

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

    entropy_feats = _compute_entropy_features(x)
    fractal_dimension = _higuchi_fractal_dimension(x)
    prpd_features = _compute_prpd_features(x, fs)

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
        "fractal_dimension": fractal_dimension,
        **prpd_features,
    }


def run(station_id: Optional[str], input_path: Path, output_path: Path, config: Optional[dict] = None) -> dict:
    """Extract classic statistical and spectral features into CSV files.

    Input signals are expected under
    ``partial_discharge_project/station_<ID>/data_clean/standard_denoising_normalisation/*.npy``.

    CSV files are saved under ``features/2_feature_engineering/classic_stats/``.
    """
    cfg = config or {}
    fs = float(cfg.get("fs", 1_000_000.0))

    log = get_logger(__name__, log_dir=output_path / "reports")
    log.info("Starting feature engineering stage")

    stations = [output_path / f"station_{station_id}"] if station_id else sorted(output_path.glob("station_*/"))

    root_features_dir = Path("features")
    engineering_dir = root_features_dir / "2_feature_engineering"
    classic_stats_dir = engineering_dir / "classic_stats"
    stations_dir = classic_stats_dir / "stations"
    classic_stats_dir.mkdir(parents=True, exist_ok=True)
    stations_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    combined_rows: list[pd.DataFrame] = []

    for station_dir in stations:
        clean_dir = station_dir / "data_clean" / "standard_denoising_normalisation"
        parquet_file = clean_dir / "cleaned_windows.parquet"
        if not parquet_file.exists():
            log.warning("Missing cleaned parquet for %s", station_dir.name)
            continue
        clean_df = pd.read_parquet(parquet_file)
        npz_file = clean_dir / "cleaned_windows.npz"
        signals_store: dict[str, np.ndarray] | None = None
        if npz_file.exists():
            signals_store = np.load(npz_file)
        if clean_df.empty:
            log.warning("No cleaned windows present for %s", station_dir.name)
            continue

        rows: list[dict[str, object]] = []
        for record in clean_df.itertuples(index=False):
            try:
                signal = np.asarray(record.signal, dtype=np.float32)
            except AttributeError:
                signal = None
            if signal is None or signal.size == 0:
                signal_key = str(getattr(record, "signal_key", getattr(record, "measurement_id")))
                if signals_store is None:
                    raise FileNotFoundError(f"Missing signal data for key {signal_key} under {npz_file}")
                try:
                    signal = np.asarray(signals_store[signal_key], dtype=np.float32)
                except KeyError as exc:  # pragma: no cover - unexpected missing key
                    raise KeyError(f"Signal key {signal_key} not found in {npz_file}") from exc
            if signal.size == 0:
                continue
            feats = _compute_basic_features(signal, fs)
            feats["station_id"] = str(record.station_id)
            feats["measurement_id"] = int(record.measurement_id)
            feats["faultAnnotation"] = str(getattr(record, "faultAnnotation", "unknown"))
            feats["signal_length"] = int(getattr(record, "signal_length", signal.size))
            rows.append(feats)

        if not rows:
            continue

        station_df = pd.DataFrame(rows)
        station_df["faultAnnotation"] = station_df["faultAnnotation"].fillna("unknown").astype(str)
        numeric_cols = [c for c in station_df.columns if c not in {"station_id", "measurement_id", "faultAnnotation"} and np.issubdtype(station_df[c].dtype, np.number)]
        station_df[numeric_cols] = station_df[numeric_cols].astype(np.float32, copy=False)
        station_name = station_dir.name
        station_out = stations_dir / f"{station_name}.parquet"
        station_df.to_parquet(station_out, index=False)
        summary[station_name] = len(station_df)
        combined_rows.append(station_df)
        log.info("Station %s: extracted %d feature rows -> %s", station_name, len(station_df), station_out)

    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_df = combined_df.sort_values(["station_id", "measurement_id"]).reset_index(drop=True)
        numeric_cols = [c for c in combined_df.columns if c not in ["station_id", "measurement_id", "faultAnnotation"] and np.issubdtype(combined_df[c].dtype, np.number)]
        combined_df[numeric_cols] = combined_df[numeric_cols].astype(np.float32, copy=False)
        base_parquet = classic_stats_dir / "base_features.parquet"
        base_csv = classic_stats_dir / "base_features.csv"
        combined_df.to_parquet(base_parquet, index=False)
        combined_df.to_csv(base_csv, index=False)
        log.info("Base features combined -> %s", base_parquet)
        _write_feature_summary(engineering_dir)
    else:
        log.warning("No feature rows generated; check preprocessing output")

    log.info("Feature engineering stage completed")
    return summary


