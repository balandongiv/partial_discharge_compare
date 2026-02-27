"""
Data Cleaning Module for Partial Discharge Classification Pipeline

This module implements Phase 0 of the PD classification pipeline, responsible for cleaning
and preprocessing raw partial discharge measurement signals. It performs signal filtering,
normalization, and saves cleaned data in a standardized format for downstream feature extraction.

Step-by-Step Process:
1. Discovery Phase:
   - Scans dataset/contactless_pd_detection/station_<ID>/ directories
   - Discovers all available station IDs containing raw .npy signal files
   - Builds a list of stations to process

2. Signal Loading:
   - Loads raw PD signals from .npy files for each station
   - Handles multi-dimensional arrays by squeezing to 1D
   - Validates file integrity and handles loading errors gracefully

3. Signal Filtering:
   - Applies band-pass Butterworth filter (4th order) to remove noise
   - Default frequency range: 1 kHz to 45% of sampling frequency
   - Clamps filter frequencies to valid digital frequency range (0, 1)
   - Uses zero-phase filtering (filtfilt) to avoid phase distortion

4. Signal Normalization:
   - Applies z-score normalization (zero mean, unit variance)
   - Handles edge case where standard deviation is zero
   - Ensures signals are standardized for consistent feature extraction

5. Output Management:
   - Creates standardized directory structure: partial_discharge_project/station_<ID>/data_clean/standard_denoising_normalisation/
   - Saves cleaned signals with original filenames
   - Maintains traceability between raw and cleaned data

6. Logging and Summary:
   - Logs processing progress for each station
   - Tracks number of successfully processed files per station
   - Returns summary statistics for pipeline monitoring

Configuration Parameters:
- fs: Sampling frequency (default: 1,000,000 Hz)
- band_hz: Frequency band tuple (low, high) for filtering
- station_id: Optional specific station to process (processes all if None)

Dependencies:
- numpy: Array operations and file I/O
- scipy.signal: Butterworth filter design and application
- pathlib: Cross-platform path handling
- logging: Progress tracking and error reporting
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .utils.logger import get_logger
from .utils.discovery import discover_station_ids


def _bandpass_filter(x: np.ndarray, low_hz: float, high_hz: float, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    # Clamp to valid digital frequencies (0, 1)
    low = max(1e-9, min(low_hz, nyq * 0.99)) / nyq
    high = max(1e-9, min(high_hz, nyq * 0.99)) / nyq
    if not (0.0 < low < high < 1.0):
        # If invalid, return as-is (filter disabled)
        return x
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, x)


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x)) or 1.0
    return (x - mean) / std


def run(station_id: Optional[str], input_path: Path, output_path: Path, config: Optional[dict] = None) -> dict:
    """Clean raw ``.npy`` signals and save to standard path.

    Parameters
    ----------
    input_path:
        Root folder containing ``dataset/contactless_pd_detection/station_<ID>/*.npy``.
    output_path:
        Root folder ``partial_discharge_project`` where cleaned data will be saved.
    config:
        Optional configuration dict with keys ``fs``, ``band_hz``.

    Returns
    -------
    dict
        Summary dict with counts per station.
    """
    cfg = config or {}
    fs = float(cfg.get("fs", 1_000_000.0))
    band_low, band_high = (cfg.get("band_hz", (1e3, fs * 0.45)))

    log = get_logger(__name__, log_dir=output_path / "reports")
    log.info("Starting data cleaning stage")

    dataset_root = input_path
    annotations: Optional[pd.DataFrame] = None
    ann_file = dataset_root / "inferred_annotation.csv"
    if ann_file.exists():
        annotations = pd.read_csv(
            ann_file,
            usecols=["idStation", "idMeasurement", "faultAnnotation"],
        )
        annotations["idStation"] = annotations["idStation"].astype(int)
        annotations["idMeasurement"] = annotations["idMeasurement"].astype(int)
        annotations["faultAnnotation"] = annotations["faultAnnotation"].astype(str)
        log.info("Loaded %d annotation records", len(annotations))
    else:
        log.warning("Annotation file %s not found; labels will be missing", ann_file)

    station_ids = [station_id] if station_id else discover_station_ids(dataset_root)
    if not station_ids:
        log.warning("No stations found under dataset/contactless_pd_detection")
    summary: dict[str, int] = {}

    for sid in station_ids:
        station_raw = dataset_root / "contactless_pd_detection" / f"station_{sid}"
        out_dir = output_path / f"station_{sid}" / "data_clean" / "standard_denoising_normalisation"
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        aggregated: dict[str, np.ndarray] = {}
        rows: list[dict[str, object]] = []
        for npy_file in sorted(station_raw.glob("*.npy")):
            try:
                x = np.load(npy_file)
                if x.ndim > 1:
                    x = x.squeeze()
                x = _bandpass_filter(x, band_low, band_high, fs)
                x = _zscore(x)
                # ensure float32 for space efficiency
                x = x.astype(np.float32, copy=False)
                key = npy_file.stem
                aggregated[key] = x
                count += 1
                rows.append(
                    {
                        "station_id": int(sid),
                        "measurement_id": int(key),
                        "signal_key": str(key),
                        "signal_length": int(x.shape[0]),
                    }
                )
            except Exception as err:  # pragma: no cover - log unexpected file issues
                log.error(f"Failed cleaning {npy_file}: {err}")
        # write single aggregated npz per station
        out_path = out_dir / "cleaned_windows.npz"
        if aggregated:
            np.savez_compressed(out_path, **aggregated)
        # remove any legacy per-window npz files in the directory
        for f in out_dir.glob("*.npz"):
            if f.name != "cleaned_windows.npz":
                try:
                    f.unlink()
                except Exception:
                    pass

        # Persist dataframe with metadata + labels for downstream stages
        if rows:
            df = pd.DataFrame(rows)
            if annotations is not None:
                df = df.merge(
                    annotations,
                    how="left",
                    left_on=["station_id", "measurement_id"],
                    right_on=["idStation", "idMeasurement"],
                )
                df = df.drop(columns=["idStation", "idMeasurement"])
            if "faultAnnotation" not in df.columns:
                df["faultAnnotation"] = "unknown"
            df["faultAnnotation"] = df["faultAnnotation"].fillna("unknown").astype(str)
            if "signal_length" not in df.columns:
                df["signal_length"] = df["signal_key"].map(lambda key: int(aggregated[str(key)].shape[0]))
            df["signal_path"] = str(out_path)
            parquet_path = out_dir / "cleaned_windows.parquet"
            df.to_parquet(parquet_path, index=False)
            log.info("Station %s: cleaned data saved -> %s", sid, parquet_path)
        summary[sid] = count
        log.info(f"Station {sid}: cleaned {count} files -> {out_path}")

    log.info("Data cleaning stage completed")
    return summary


