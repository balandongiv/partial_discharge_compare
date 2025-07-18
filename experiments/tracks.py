from __future__ import annotations

"""Utilities to run predefined experiment tracks."""

import logging
from dataclasses import dataclass

from preprocess import run_preprocess
from features import extract_from_clean

logger = logging.getLogger(__name__)


@dataclass
class ExperimentTrack:
    """Configuration for a single experiment run."""

    name: str
    advanced_denoise: bool = False
    augment: bool = False
    wavelet_feats: bool = False

    def run(self, dataset: str, jobs: int = 1, force: bool = False) -> None:
        """Execute preprocessing and feature extraction for ``dataset``."""
        logger.info("Running %s", self.name)
        run_preprocess.run(
            dataset,
            force=force,
            adv_denoise=self.advanced_denoise,
            augment=self.augment,
            jobs=jobs,
        )
        extract_from_clean.run()
        logger.info("%s complete", self.name)


EXPERIMENTS: dict[str, ExperimentTrack] = {
    "exp1_featurewiz": ExperimentTrack(
        name="Featurewiz Track",
        advanced_denoise=False,
        augment=False,
        wavelet_feats=False,
    ),
    "exp2_mljar": ExperimentTrack(
        name="MLJAR-Supervised",
        advanced_denoise=False,
        augment=False,
        wavelet_feats=False,
    ),
    "exp3_borutashap": ExperimentTrack(
        name="Advanced Denoising + BorutaShap",
        advanced_denoise=True,
        augment=False,
        wavelet_feats=False,
    ),
    "exp4_catboost": ExperimentTrack(
        name="Data Augmentation + CatBoost",
        advanced_denoise=False,
        augment=True,
        wavelet_feats=False,
    ),
    "exp5_wavelet_cnn": ExperimentTrack(
        name="Wavelet Image CNN",
        advanced_denoise=False,
        augment=False,
        wavelet_feats=True,
    ),
}


def run_experiment(name: str, dataset: str, jobs: int = 1, force: bool = False) -> None:
    """Run the experiment specified by ``name``."""
    track = EXPERIMENTS[name]
    track.run(dataset=dataset, jobs=jobs, force=force)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run predefined experiment tracks")
    parser.add_argument("experiment", choices=EXPERIMENTS.keys())
    parser.add_argument("dataset")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(args.experiment, dataset=args.dataset, jobs=args.jobs, force=args.force)
