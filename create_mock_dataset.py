"""
Create a mock raw dataset compatible with run_full_pipeline.py.

This script generates a dataset that mirrors the real layout:
<dataset_root>/
  contactless_pd_detection/
    station_<ID>/
      <measurement_id>.npy
  inferred_annotation.csv

Each .npy is a 1D int8 signal and the annotation CSV includes:
idStation, idMeasurement, faultAnnotation, timeStamp
"""

from __future__ import annotations

import argparse
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_SIGNAL_LENGTH = 800_000
DEFAULT_SAMPLING_FREQ = 1_000_000
DEFAULT_STATIONS = 2
DEFAULT_SAMPLES_PER_CLASS = 6
DEFAULT_START_STATION_ID = 90001
DEFAULT_SEED = 42


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Create a mock raw dataset compatible with run_full_pipeline.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("mock_dataset"),
        help="Destination root folder for the mock dataset.",
    )
    parser.add_argument(
        "--stations",
        type=int,
        default=DEFAULT_STATIONS,
        help="Number of station folders to generate.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="Samples per class (fault=1 and fault=0) per station.",
    )
    parser.add_argument(
        "--signal-length",
        type=int,
        default=DEFAULT_SIGNAL_LENGTH,
        help="Signal length in samples (1D array).",
    )
    parser.add_argument(
        "--sampling-freq",
        type=int,
        default=DEFAULT_SAMPLING_FREQ,
        help="Sampling frequency used for synthetic signal generation.",
    )
    parser.add_argument(
        "--start-station-id",
        type=int,
        default=DEFAULT_START_STATION_ID,
        help="Starting numeric station id; increments by 1 per station.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output folder if it already exists.",
    )
    return parser


def _ensure_output_root(output_root: Path, force: bool) -> None:
    """Prepare the output directory."""
    if output_root.exists() and force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _generate_pulse(
    length: int,
    fs: int,
    rng: np.random.Generator,
    amplitude_range: Tuple[float, float],
    freq_range: Tuple[float, float],
    width_range: Tuple[float, float],
) -> np.ndarray:
    """Generate a damped sinusoid pulse."""
    pulse_width = rng.uniform(*width_range)
    pulse_len = max(4, int(pulse_width * fs))
    pulse_len = min(pulse_len, length // 4)
    t = np.arange(pulse_len) / fs

    amplitude = rng.uniform(*amplitude_range)
    freq = rng.uniform(*freq_range)
    decay = np.exp(-t * rng.uniform(4e5, 1.2e6))
    oscillation = np.sin(2 * np.pi * freq * t)
    pulse = amplitude * decay * oscillation

    pulse_signal = np.zeros(length, dtype=np.float32)
    start = rng.integers(0, max(1, length - pulse_len))
    pulse_signal[start:start + pulse_len] += pulse.astype(np.float32)
    return pulse_signal


def generate_signal(
    length: int,
    fs: int,
    rng: np.random.Generator,
    fault: bool,
) -> np.ndarray:
    """Generate a synthetic PD-like signal."""
    signal = rng.normal(0.0, 2.0, size=length).astype(np.float32)

    if fault:
        num_pulses = rng.integers(3, 7)
        amplitude_range = (30.0, 80.0)
        freq_range = (80_000.0, 4_000_000.0)
        width_range = (0.00005, 0.0004)
    else:
        num_pulses = rng.integers(0, 3)
        amplitude_range = (5.0, 25.0)
        freq_range = (50_000.0, 2_000_000.0)
        width_range = (0.00003, 0.0002)

    for _ in range(int(num_pulses)):
        signal += _generate_pulse(length, fs, rng, amplitude_range, freq_range, width_range)

    # Clamp to int8 range to match real dataset dtype.
    signal = np.clip(np.rint(signal), -127, 127).astype(np.int8)
    return signal


def create_mock_dataset(
    output_root: Path,
    stations: int,
    samples_per_class: int,
    signal_length: int,
    sampling_freq: int,
    start_station_id: int,
    seed: int,
    force: bool,
) -> None:
    """
    Create a mock dataset compatible with the full pipeline.

    Args:
        output_root: Root folder for the dataset.
        stations: Number of stations to generate.
        samples_per_class: Samples per class per station.
        signal_length: Length of each signal.
        sampling_freq: Sampling frequency (Hz).
        start_station_id: First numeric station id.
        seed: Random seed for reproducibility.
        force: Whether to overwrite existing output folder.
    """
    _ensure_output_root(output_root, force)

    rng = np.random.default_rng(seed)
    random.seed(seed)

    contactless_root = output_root / "contactless_pd_detection"
    contactless_root.mkdir(parents=True, exist_ok=True)
    annotation_path = output_root / "inferred_annotation.csv"

    annotations: List[Dict[str, object]] = []
    base_time = datetime(2017, 11, 11, 0, 0, 0)

    for station_idx in range(stations):
        station_id = start_station_id + station_idx
        station_dir = contactless_root / f"station_{station_id}"
        station_dir.mkdir(parents=True, exist_ok=True)

        total_samples = samples_per_class * 2
        for i in range(total_samples):
            fault = i < samples_per_class
            measurement_id = station_id * 1000 + (i + 1)

            signal = generate_signal(signal_length, sampling_freq, rng, fault=fault)
            np.save(station_dir / f"{measurement_id}.npy", signal)

            annotations.append({
                "idStation": int(station_id),
                "idMeasurement": int(measurement_id),
                "faultAnnotation": int(1 if fault else 0),
                "timeStamp": (base_time + timedelta(hours=len(annotations))).strftime("%Y-%m-%d %H:%M:%S"),
            })

    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv(annotation_path, index=False)

    print("[OK] Mock dataset created.")
    print(f"  Root: {output_root.resolve()}")
    print(f"  Stations: {stations}")
    print(f"  Samples per class per station: {samples_per_class}")
    print(f"  Total signals: {len(annotations)}")
    print(f"  Annotation file: {annotation_path}")


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    create_mock_dataset(
        output_root=args.output_root,
        stations=args.stations,
        samples_per_class=args.samples_per_class,
        signal_length=args.signal_length,
        sampling_freq=args.sampling_freq,
        start_station_id=args.start_station_id,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
