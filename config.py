"""Configuration loader for the PD classification pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project path configuration."""

    root_dir: str = Field(..., alias="root_dir")
    raw_data_dir: str
    processed_dir: str
    features_dir: str
    models_dir: str
    reports_dir: str
    logs_dir: str
    random_state: int
    jobs: int


class RuntimeConfig(BaseModel):
    """Runtime behaviour switches."""

    checkpoint_interval: int
    resume: bool
    verbosity: str
    save_confusion_matrices: bool
    save_roc_curves: bool
    model_registry_format: str


class DatasetConfig(BaseModel):
    """Configuration entry for a dataset."""

    path: str
    label_mapping: Optional[Dict[str, int]] = None


class PreprocessingOptions(BaseModel):
    """Preprocessing related options."""

    advanced_denoise: List[bool]
    augment: List[bool]
    wavelet_feats: List[bool]
    window_length_ms: int
    bandpass_hz: List[int]


class LibraryFeatureBlock(BaseModel):
    """Configuration for a group of features from a library."""

    enabled: bool = True
    selected_features: List[str] = Field(default_factory=list)


class MNEFeatureBlock(LibraryFeatureBlock):
    freq_bands: Dict[str, List[float]] = Field(default_factory=dict)


class FeatureCatalog(BaseModel):
    """Root feature catalog configuration."""

    enable_all: bool = True
    mne_features: MNEFeatureBlock
    librosa: LibraryFeatureBlock
    custom: LibraryFeatureBlock


class Config(BaseModel):
    """Root configuration model."""

    project: ProjectConfig
    runtime: RuntimeConfig
    datasets: List[DatasetConfig] = Field(default_factory=list)
    preprocessing_options: PreprocessingOptions
    features: FeatureCatalog
    scaling_methods: List[str]
    feature_selections: List[str]
    model_types: List[str]
    hyperparameter_tuning: Dict[str, Any]
    cv_strategy: Dict[str, Any]
    experiments: Dict[str, Any]


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path = CONFIG_PATH) -> Config:
    """Load and validate ``config.yaml``.

    Parameters
    ----------
    path
        Path to the configuration YAML file.

    Returns
    -------
    Config
        Parsed configuration instance.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return Config(**data)


CONFIG = load_config(CONFIG_PATH)

ROOT_DIR = Path(CONFIG.project.root_dir).resolve()
RAW_DIR = ROOT_DIR / CONFIG.project.raw_data_dir
PROCESSED_DIR = ROOT_DIR / CONFIG.project.processed_dir
FEATURES_DIR = ROOT_DIR / CONFIG.project.features_dir
MODELS_DIR = ROOT_DIR / CONFIG.project.models_dir
REPORTS_DIR = ROOT_DIR / CONFIG.project.reports_dir
LOGS_DIR = ROOT_DIR / CONFIG.project.logs_dir
