# Partial Discharge Compare

Single-click, end-to-end pipeline for partial discharge (PD) signal classification.
Run one script to go from raw `.npy` signals to trained models and reports, with
automatic checkpoint/resume.

## What This Project Does

The pipeline in `run_full_pipeline.py` executes all phases:

1. Discover raw data under `dataset/contactless_pd_detection/`
2. Preprocess signals (bandpass + z-score) and write cleaned `.npy` files
3. Extract base features (classic stats + time/frequency features)
4. Expand features with mathematical combinations
5. Feature selection (Track 4B: Featurewiz, Track 4C: MLJAR)
6. Train models (SVM, Random Forest, Decision Tree, KNN, ANN)
7. Generate evaluation reports and plots

It writes a checkpoint after each phase and resumes if interrupted.

## Requirements

- Python 3.9+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

Optional extras used by the pipeline:

- `feature_engine` for richer feature combinations
- `featurewiz` for Track 4B feature selection (SULOV + XGBoost)
- `mljar-supervised` for Track 4C feature selection

If any optional package is missing, the pipeline logs a warning and falls back.

## Quick Start

1. Place your dataset in `dataset/` so the structure looks like:

```
dataset/
├── contactless_pd_detection/
│   ├── station_52008/
│   │   ├── 123.npy
│   │   └── 124.npy
│   └── station_52009/
│       └── ...
└── inferred_annotation.csv
```

2. Run the pipeline:

```bash
python run_full_pipeline.py
```

On first run, the script prompts for:

- Dataset root (defaults to `dataset/`)
- Output base (defaults to project root)

It saves these paths in `run_full_pipeline.yaml` for future runs.

## Output Locations

All output paths are relative to the selected output base:

- Preprocessed data: `outputs/preprocessing/station_<ID>/data_clean/standard_denoising_normalisation/`
- Base features: `features/2_feature_engineering/classic_stats/base_features.parquet`
- Expanded features: `features/3_feature_comb_expansion/mathematical_combination/combined_features.parquet`
- Selected features: `features/4_feature_selection/`
- Trained models: `models/trained/track_4b/`, `models/trained/track_4c/`
- Predictions + metrics: `results/track_4b/`, `results/track_4c/`
- Final reports: `reports/track_4b/pipeline_report.md`, `reports/track_4c/pipeline_report.md`
- Checkpoint: `pipeline_checkpoint.json`

## Checkpoint / Resume

If the run is interrupted, rerun:

```bash
python run_full_pipeline.py
```

The pipeline detects completed phases and skips them automatically.

## Notes

- The annotation file `dataset/inferred_annotation.csv` is used to attach
  `faultAnnotation` labels. If missing, labels are set to `unknown`.
- Outputs are stored as Parquet and NumPy files for easy inspection or reuse.


