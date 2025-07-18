# Partial Discharge Compare

This project contains a modular pipeline for partial discharge (PD) signal classification.
Stages of the pipeline are invoked via `ml/main.py` and configured through `config.yaml`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run a specific stage with:

```bash
python -m ml.main --stage preprocess
```

Use `--stage full-run` to execute all stages sequentially. Flags such as
`--advanced-denoise` and `--augment` toggle optional preprocessing steps.
Pass `--jobs N` to enable parallel processing across N workers.

## Directory Structure

```
project_pd/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── features/
│   ├── models/
│   └── reports/
└── logs/
```

Synthetic test signals are located under `unitest/fixtures/` for unit testing.

## Pre-processing utilities

Running the preprocessing stage will read the datasets listed in
`config.yaml`, apply cleaning and optional augmentation, and write
windowed arrays under `data/processed/`.

```bash
python -m main --stage preprocess --advanced-denoise --augment
```

Key modules:

- `preprocess/cleaning.py` – filtering and denoising helpers (e.g. `bandpass_filter`)
- `preprocess/augmentation.py` – optional `time_warp` and `add_jitter`
- `preprocess/windowing.py` – segmentation and label loading
- `preprocess/run_preprocess.py` – orchestrates the full workflow

A short walkthrough is provided in `notebooks/cleaning_tutorial.ipynb`.

### Developer workflow

`notebooks/new_developer_workflow.py` is a compact walkthrough of the main
modules.  It loads a sample signal, cleans it with `denoise_signal`, extracts a
set of statistical features via `FeatureExtractor`, applies a toy pairwise
distance step and finally runs feature selection.  The sample data is loaded
from `unitest/data/748987.npy` via a path resolved from the project root so the
script runs from any working directory.  Execute it line by line to understand
the flow; every stage emits INFO logs including how many features were
generated and how many remain after selection.  The workflow attempts to use
`featurewiz` for the final selection step. If that package fails to import, a
warning is logged and all features are retained. Use these logs as a starting
point for extending your own experiments.

### Tests

Run all unit tests with:

```bash
python unitest/run_all_test.py
```

### Experiment runner

Predefined experiment tracks are available via `experiments.tracks`. Execute all steps for a given dataset using:

```bash
python -m experiments.tracks run_experiment exp1_featurewiz dataset/contactless_pd_detection
```

Each track toggles options such as advanced denoising or augmentation as described in the code.
