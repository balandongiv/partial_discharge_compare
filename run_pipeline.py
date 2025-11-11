from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from modules.data_cleaning import run as clean_run
from modules.feature_engineering import run as feats_run
from modules.feature_expansion import run as expand_run
from modules.feature_selection import run as select_run
from modules.model_training import run as train_run
from modules.evaluation_reporting import run as eval_run
from modules.logger import get_logger


def run_tests(kexpr: str) -> bool:
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/test_pipeline.py", "-k", kexpr], capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    return proc.returncode == 0


def _discover_station_ids(dataset_root: Path) -> list[str]:
    ids = []
    base = dataset_root / "contactless_pd_detection"
    if base.exists():
        for p in base.iterdir():
            if p.is_dir() and p.name.startswith("station_"):
                ids.append(p.name.split("station_")[-1])
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", help="Station ID to target (optional)")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fs", type=float, default=1_000_000.0)
    args = parser.parse_args()

    root = Path.cwd()
    dataset_root = root / "dataset"
    project_root = root / "partial_discharge_project"
    project_root.mkdir(exist_ok=True)
    logger = get_logger(__name__, log_dir=project_root / "reports")

    # Determine station list
    station_ids = [args.station] if args.station else _discover_station_ids(dataset_root)
    logger.info(f"Stations detected: {station_ids}")

    target_station = args.station

    clean_run(target_station, dataset_root, project_root, config={"fs": args.fs})
    if not run_tests("test_preprocessing_outputs"):
        logger.error("Data cleaning tests failed. Aborting.")
        sys.exit(1)

    feats_run(target_station, project_root, project_root, config={"fs": args.fs})
    if not run_tests("test_feature_engineering_outputs"):
        logger.error("Feature engineering tests failed. Aborting.")
        sys.exit(1)

    expand_run(target_station, project_root, project_root)
    if not run_tests("test_feature_combination_outputs"):
        logger.error("Feature expansion tests failed. Aborting.")
        sys.exit(1)

    select_run(target_station, project_root, project_root)
    if not run_tests("test_feature_selection_outputs"):
        logger.error("Feature selection tests failed. Aborting.")
        sys.exit(1)

    train_run(target_station, project_root, project_root)
    if not run_tests("test_model_training_outputs"):
        logger.error("Model training tests failed. Aborting.")
        sys.exit(1)

    eval_run(target_station, project_root, project_root)
    if not run_tests("test_reporting_outputs"):
        logger.error("Evaluation/reporting tests failed. Aborting.")
        sys.exit(1)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()


