from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_station_annotations(
    annotations_path: Path,
    station_id: str,
    fault_column: str = "faultAnnotation",
) -> pd.DataFrame:
    """Load annotations for a specific station.

    Args:
        annotations_path: Path to the annotations CSV.
        station_id: Target station identifier.
        fault_column: Name of the annotation column containing binary labels.

    Returns:
        DataFrame filtered to the requested station with string `idMeasurement`.
    """
    df = pd.read_csv(
        annotations_path,
        dtype={
            "idStation": str,
            "idMeasurement": str,
            fault_column: "Int64",
        },
    )
    station_df = df[df["idStation"] == station_id].copy()
    if station_df.empty:
        raise ValueError(
            f"No records found for station {station_id} in {annotations_path}"
        )
    if station_df[fault_column].isna().any():
        raise ValueError(
            f"Found missing values in {fault_column} for station {station_id}"
        )
    station_df["idMeasurement"] = station_df["idMeasurement"].astype(str)
    station_df[fault_column] = station_df[fault_column].astype(int)
    return station_df


def choose_measurements(
    station_df: pd.DataFrame,
    fault_column: str,
    label: int,
    n_samples: int,
) -> pd.Series:
    """Select `n_samples` measurement IDs for a given label."""
    label_df = station_df[station_df[fault_column] == label]
    available = len(label_df)
    if available < n_samples:
        raise ValueError(
            f"Requested {n_samples} samples for label {label}, "
            f"but only {available} available."
        )
    selection = label_df.sample(n=n_samples, random_state=42)["idMeasurement"]
    logger.info(
        "Selected %s measurement ids for label %s (from %s available).",
        n_samples,
        label,
        available,
    )
    return selection


def copy_measurements(
    measurement_ids: Iterable[str],
    source_dir: Path,
    destination_dir: Path,
) -> None:
    """Copy measurement files from source to destination."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for measurement_id in measurement_ids:
        source_file = source_dir / f"{measurement_id}.npy"
        if not source_file.exists():
            raise FileNotFoundError(f"Missing source file: {source_file}")
        target_file = destination_dir / source_file.name
        shutil.copy2(source_file, target_file)
        copied += 1
    logger.info("Copied %s files into %s", copied, destination_dir)


def run(
    source_dir: Path,
    annotations_path: Path,
    destination_root: Path,
    station_id: str,
    samples_label0: int,
    samples_label1: int,
    fault_column: str = "faultAnnotation",
) -> None:
    """Sample and copy measurement files for a binary classification dataset."""
    configure_logging()
    logger.info("Loading annotations for station %s", station_id)
    station_df = load_station_annotations(
        annotations_path=annotations_path,
        station_id=station_id,
        fault_column=fault_column,
    )

    selected = {
        0: choose_measurements(
            station_df=station_df,
            fault_column=fault_column,
            label=0,
            n_samples=samples_label0,
        ),
        1: choose_measurements(
            station_df=station_df,
            fault_column=fault_column,
            label=1,
            n_samples=samples_label1,
        ),
    }

    target_dir = destination_root / f"station_{station_id}"
    combined_ids = pd.concat(
        [
            selected[0].reset_index(drop=True),
            selected[1].reset_index(drop=True),
        ]
    )
    logger.info(
        "Copying %s total files to %s",
        combined_ids.size,
        target_dir,
    )
    copy_measurements(
        measurement_ids=combined_ids,
        source_dir=source_dir,
        destination_dir=target_dir,
    )
    logger.info("Completed copying samples for station %s", station_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample 5000 measurements per label and copy to dataset directory."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing raw .npy measurement files.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to inferred_annotation.csv.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        required=True,
        help="Root dataset directory where station folder will be created.",
    )
    parser.add_argument(
        "--station-id",
        type=str,
        default="52008",
        help="Station identifier to sample.",
    )
    parser.add_argument(
        "--samples-label0",
        type=int,
        default=5000,
        help="Number of samples to copy for label 0.",
    )
    parser.add_argument(
        "--samples-label1",
        type=int,
        default=5000,
        help="Number of samples to copy for label 1.",
    )
    parser.add_argument(
        "--fault-column",
        type=str,
        default="faultAnnotation",
        help="Annotation column name containing binary labels.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source_dir=args.source,
        annotations_path=args.annotations,
        destination_root=args.destination,
        station_id=args.station_id,
        samples_label0=args.samples_label0,
        samples_label1=args.samples_label1,
        fault_column=args.fault_column,
    )

