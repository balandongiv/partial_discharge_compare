from __future__ import annotations

"""Entry point for the PD classification pipeline."""

from argparse import ArgumentParser
from datetime import datetime
import logging
import subprocess
import sys
from pathlib import Path

from preprocess import run_preprocess

import config


def parse_args() -> ArgumentParser:
    """Parse command line arguments."""
    parser = ArgumentParser(description="PD classification pipeline")
    parser.add_argument(
        "--stage",
        default="full-run",
        choices=["full-run", "preprocess", "extract", "analyze", "report"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--jobs", type=int, default=config.CONFIG.project.jobs, help="Parallel workers")
    parser.add_argument("--advanced-denoise", action="store_true", help="Enable advanced denoising")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    return parser


def setup_logging(stage: str) -> Path:
    """Initialise the root logger to ``logs/``.

    Parameters
    ----------
    stage
        Pipeline stage name.

    Returns
    -------
    Path
        Path to the created log file.
    """
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOGS_DIR / f"{timestamp}_{stage}.log"
    logging.basicConfig(
        level=getattr(logging, config.CONFIG.runtime.verbosity),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return log_file


def main(argv: list[str] | None = None) -> None:
    """Entry CLI for the project."""
    parser = parse_args()
    args = parser.parse_args(argv)

    log_file = setup_logging(args.stage)
    logger = logging.getLogger(__name__)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = "unknown_commit"
    logger.info("--- Experiment Start ---")
    logger.info("Config: %s", config.CONFIG_PATH.resolve())
    logger.info("Git commit: %s", commit)
    logger.info("Stage: %s", args.stage)
    logger.info("Flags: %s", vars(args))
    logger.info("------------------------")

    # Stage 0: Preprocessing
    if args.stage in {"preprocess", "full-run"}:
        for ds in config.CONFIG.datasets:
            logger.info("Processing dataset: %s", ds.path)
            run_preprocess.run(
                ds.path,
                args.force,
                args.advanced_denoise,
                args.augment,
                args.jobs,
            )
        if args.stage == "preprocess":
            return

    # Configuration for subsequent stages
    pipeline_config = {"fs": 40_000_000.0}
    project_root = config.ROOT_DIR

    # Stage 1: Feature Extraction
    if args.stage in {"extract", "full-run"}:
        logger.info("Starting Feature Engineering...")
        from modules import feature_engineering
        feature_engineering.run(None, project_root, project_root, config=pipeline_config)
        
        logger.info("Starting Feature Expansion...")
        from modules import feature_expansion
        feature_expansion.run(None, project_root, project_root)
        
        if args.stage == "extract":
            return

    # Stage 2: Analysis & Modeling
    if args.stage in {"analyze", "full-run"}:
        logger.info("Starting Feature Selection...")
        from modules import feature_selection
        feature_selection.run(None, project_root, project_root, config=config.CONFIG)

        logger.info("Starting Model Training...")
        from modules import model_training
        model_training.run(None, project_root, project_root)
        
        if args.stage == "analyze":
            return

    # Stage 3: Reporting
    if args.stage in {"report", "full-run"}:
        logger.info("Starting Evaluation & Reporting...")
        from modules import evaluation_reporting
        evaluation_reporting.run(None, project_root, project_root)
        
    logger.info("Pipeline Execution Completed.")


if __name__ == "__main__":
    main()
