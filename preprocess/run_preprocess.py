from __future__ import annotations

"""Orchestrate preprocessing of raw sessions."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import math

from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

import numpy as np

import config
from preprocess import discovery, io, cleaning, augmentation, windowing

logger = logging.getLogger(__name__)


@dataclass
class StationProcessingResult:
    """Container for per-station preprocessing outputs."""

    station_id: str
    records: List[Dict[str, Any]]
    arrays: Dict[str, np.ndarray]
    stats: Dict[str, Any]
    source: str


class CheckpointManager:
    """Persist processed station identifiers to allow resumable preprocessing."""

    def __init__(self, path: Path, interval: int) -> None:
        self.path = path
        self.interval = max(1, interval)
        self.completed: set[str] = set()
        self._dirty = False
        if path.exists():
            try:
                self.completed = set(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                logger.warning("Checkpoint file %s is corrupted; continuing without it.", path)

    def should_skip(self, key: str, resume: bool, force: bool) -> bool:
        return resume and not force and key in self.completed

    def mark_done(self, key: str) -> None:
        if key in self.completed:
            return
        self.completed.add(key)
        self._dirty = True
        if len(self.completed) % self.interval == 0:
            self.flush()

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(self.completed), indent=2), encoding="utf-8")
        self._dirty = False


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32, order="C")


def _window_stats(window: np.ndarray) -> Dict[str, float]:
    arr = _ensure_float32(window)
    return {
        "nan": float(np.isnan(arr).any()),
        "zero_std": float(np.std(arr) == 0.0),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _station_base_dir(station_id: str) -> Path:
    return config.ROOT_DIR / station_id / "data_clean" / "standard_denoising_normalisation"


def _store_station_outputs(result: StationProcessingResult) -> None:
    base_dir = _station_base_dir(result.station_id)
    base_dir.mkdir(parents=True, exist_ok=True)
    if result.records:
        df = pd.DataFrame(result.records)
        float_cols = [c for c in df.columns if c not in {"station_id", "measurement_id", "sensor", "faultAnnotation", "source"} and df[c].dtype != object]
        df[float_cols] = df[float_cols].astype(np.float32, copy=False)
        df.to_parquet(base_dir / "cleaned_windows.parquet", index=False)
        np.savez(base_dir / "cleaned_windows.npz", **{k: v for k, v in result.arrays.items()})


def _quality_report_from_disk() -> None:
    report_dir = config.REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "data_quality_report.md"

    rows: List[Dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    total_windows = 0
    nan_windows = 0
    zero_std_windows = 0
    global_min = math.inf
    global_max = -math.inf

    for station_dir in sorted((config.ROOT_DIR).glob("station_*")):
        parquet_path = station_dir / "data_clean" / "standard_denoising_normalisation" / "cleaned_windows.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        station_id = station_dir.name
        total_windows += len(df)
        if "faultAnnotation" in df.columns:
            label_counts.update(df["faultAnnotation"].fillna("unlabeled").astype(str))
        signal_lengths = None
        if "signal_length" in df.columns:
            signal_lengths = df["signal_length"].to_numpy(dtype=np.float32, copy=False)
        for record in df.itertuples(index=False):
            signal = getattr(record, "signal", None)
            if isinstance(signal, (list, np.ndarray)):
                arr = _ensure_float32(signal)
                nan_windows += int(np.isnan(arr).any())
                zero_std_windows += int(np.std(arr) == 0.0)
                global_min = min(global_min, float(np.min(arr)))
                global_max = max(global_max, float(np.max(arr)))
        rows.append(
            {
                "station": station_id,
                "windows": len(df),
                "mean_signal_len": float(np.mean(signal_lengths)) if signal_lengths is not None else float("nan"),
            }
        )

    if not rows:
        report_path.write_text("# Data Quality Report\n\nNo processed stations available.\n", encoding="utf-8")
        return

    report_lines = [
        "# Data Quality Report",
        "",
        f"- Total stations: {len(rows)}",
        f"- Total windows: {total_windows}",
        f"- Windows containing NaNs: {nan_windows}",
        f"- Windows with zero variance: {zero_std_windows}",
        f"- Signal value range: [{global_min:.4f}, {global_max:.4f}]",
        "",
        "## Label Balance",
        "",
        "| Label | Count |",
        "| --- | --- |",
    ]
    for label, count in sorted(label_counts.items()):
        report_lines.append(f"| {label} | {count} |")

    report_lines.extend(
        [
            "",
            "## Station Summary",
            "",
            "| Station | Windows | Mean Signal Length |",
            "| --- | --- | --- |",
        ]
    )
    for row in rows:
        report_lines.append(f"| {row['station']} | {row['windows']} | {row['mean_signal_len']:.1f} |")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def _segment_and_store(
    station_id: str,
    sensor: str,
    signal: np.ndarray,
    fs: float,
    label_source: Optional[str],
    force: bool,
    augment: bool,
    adv_denoise: bool,
) -> Dict[str, Any]:
    window_id_base = f"{station_id}_{sensor}"
    bp = cleaning.bandpass_filter(
        signal,
        config.CONFIG.preprocessing_options.bandpass_hz[0],
        config.CONFIG.preprocessing_options.bandpass_hz[1],
        fs,
    )
    cleaning.save_cleaned_signal(bp, station_id, "standard_denoising_normalisation", f"{window_id_base}_bp")

    if adv_denoise:
        den = cleaning.advanced_denoise(bp)
        cleaning.save_cleaned_signal(den, station_id, "advanced_denoising/VMD", f"{window_id_base}_den")
    else:
        den = bp

    norm = cleaning.zscore_normalize(den).astype(np.float32, copy=False)
    cleaning.save_cleaned_signal(norm, station_id, "standard_denoising_normalisation", f"{window_id_base}_norm")

    if augment:
        aug = augmentation.add_jitter(augmentation.time_warp(norm).astype(np.float32, copy=False)).astype(np.float32, copy=False)
    else:
        aug = norm

    windows = windowing.segment_signal(aug, config.CONFIG.preprocessing_options.window_length_ms, fs)
    labels = windowing.load_window_labels(label_source, len(windows))

    out_dir = config.PROCESSED_DIR / station_id / sensor
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    label_counter: Counter[str] = Counter()
    sensor_counter: Counter[str] = Counter()
    min_val = math.inf
    max_val = -math.inf

    for idx, (window, label) in enumerate(zip(windows, labels)):
        arr = _ensure_float32(window)
        min_val = min(min_val, float(np.min(arr)))
        max_val = max(max_val, float(np.max(arr)))

        window_path = out_dir / f"{window_id_base}_{idx}.npy"
        if force or not window_path.exists():
            np.save(window_path, arr)
            if label is not None:
                window_path.with_suffix(".label").write_text(str(label), encoding="utf-8")

        label_key = "unlabeled" if label is None else str(label)
        label_counter[label_key] += 1
        sensor_counter[sensor] += 1

        measurement_id = len(records)
        records.append(
            {
                "station_id": station_id,
                "measurement_id": measurement_id,
                "sensor": sensor,
                "faultAnnotation": label_key,
                "signal_length": int(arr.size),
                "signal": arr.tolist(),
                "source": "session" if label_source else "station",
            }
        )
        arrays[f"{sensor}_{measurement_id}"] = arr

    return {
        "records": records,
        "arrays": arrays,
        "label_counts": label_counter,
        "sensor_counts": sensor_counter,
        "windows": len(records),
        "min_value": min_val if min_val is not math.inf else 0.0,
        "max_value": max_val if max_val is not -math.inf else 0.0,
    }


def process_session(
    session: discovery.Session,
    force: bool,
    adv_denoise: bool,
    augment: bool,
) -> Optional[StationProcessingResult]:
    if not session.sensor_files:
        return None

    station_id = session.cable_id
    fs = 1.0

    all_records: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    label_counts: Counter[str] = Counter()
    sensor_counts: Counter[str] = Counter()
    min_val = math.inf
    max_val = -math.inf

    for sensor, path in session.sensor_files.items():
        signal_path = Path(path)
        if signal_path.suffix == ".csv":
            signal = io.load_pd_csv(signal_path)
        else:
            signal = io.load_pd_hdf5(signal_path)
        stats = _segment_and_store(station_id, sensor, signal, fs, session.label_file, force, augment, adv_denoise)
        all_records.extend(stats["records"])
        arrays.update(stats["arrays"])
        label_counts.update(stats["label_counts"])
        sensor_counts.update(stats["sensor_counts"])
        min_val = min(min_val, stats["min_value"])
        max_val = max(max_val, stats["max_value"])

    stats_summary = {
        "windows": sum(sensor_counts.values()),
        "label_counts": {k: int(v) for k, v in label_counts.items()},
        "station_counts": {station_id: {k: int(v) for k, v in sensor_counts.items()}},
        "min_value": min_val if min_val is not math.inf else 0.0,
        "max_value": max_val if max_val is not -math.inf else 0.0,
    }
    return StationProcessingResult(station_id=station_id, records=all_records, arrays=arrays, stats=stats_summary, source="session")


def process_npy_file(
    record: discovery.FileRecord,
    force: bool,
    adv_denoise: bool,
    augment: bool,
) -> Optional[StationProcessingResult]:
    station_id = record.station_id
    signal = io.load_pd_npy(record.file_path)
    fs = 1.0
    stats = _segment_and_store(station_id, "RAW", signal, fs, None, force, augment, adv_denoise)
    stats_summary = {
        "windows": stats["windows"],
        "label_counts": {k: int(v) for k, v in stats["label_counts"].items()},
        "station_counts": {station_id: {k: int(v) for k, v in stats["sensor_counts"].items()}},
        "min_value": stats["min_value"],
        "max_value": stats["max_value"],
    }
    return StationProcessingResult(
        station_id=station_id,
        records=stats["records"],
        arrays=stats["arrays"],
        stats=stats_summary,
        source="station",
    )


def run(
    dataset: str,
    force: bool,
    adv_denoise: bool,
    augment: bool,
    jobs: int = 1,
    resume: bool = False,
) -> None:
    """Run preprocessing for ``dataset`` with resumable checkpoints and quality reporting."""
    resume = resume or config.CONFIG.runtime.resume
    checkpoint = CheckpointManager(config.REPORTS_DIR / "preprocess_state.json", config.CONFIG.runtime.checkpoint_interval)
    adv_denoise = adv_denoise or config.CONFIG.preprocessing_options.advanced_denoise[0]
    augment = augment or config.CONFIG.preprocessing_options.augment[0]

    records = discovery.discover_npy_files(dataset)
    if records:
        tasks = [rec for rec in records if not checkpoint.should_skip(rec.station_id, resume, force)]
        logger.info("Processing %d station file(s)", len(tasks))
        results = Parallel(n_jobs=jobs)(
            delayed(process_npy_file)(rec, force, adv_denoise, augment)
            for rec in tqdm(tasks, desc="stations")
        )
        for result in results:
            if result is None:
                continue
            _store_station_outputs(result)
            checkpoint.mark_done(result.station_id)
        checkpoint.flush()
        _quality_report_from_disk()
        return

    sessions = discovery.discover_sessions(dataset)
    tasks = [sess for sess in sessions if not checkpoint.should_skip(sess.cable_id, resume, force)]
    logger.info("Processing %d session(s)", len(tasks))
    results = Parallel(n_jobs=jobs)(
        delayed(process_session)(session, force, adv_denoise, augment)
        for session in tqdm(tasks, desc="sessions")
    )
    for result in results:
        if result is None:
            continue
        _store_station_outputs(result)
        checkpoint.mark_done(result.station_id)
    checkpoint.flush()
    _quality_report_from_disk()


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess PD datasets")
    parser.add_argument("dataset", help="Dataset name under raw data dir")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--advanced-denoise", action="store_true", help="Use advanced denoising")
    parser.add_argument("--augment", action="store_true", help="Augment training data")
    parser.add_argument("--jobs", type=int, default=config.CONFIG.project.jobs, help="Parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume preprocessing from the last checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.dataset, args.force, args.advanced_denoise, args.augment, args.jobs, args.resume)
