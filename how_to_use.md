# How to Use the Partial Discharge Classification Pipeline

This project delivers a modular workflow for cleaning partial discharge (PD) signals, engineering and expanding features, selecting feature subsets, training models, and producing evaluation artifacts. The pipeline is organized into independently executable stages, so you can run the full process or jump to the step you need.

---

## 1. Environment Setup
- **Python:** 3.10 or newer.
- **Install dependencies:**
  ```
  python -m pip install -r requirements.txt
  ```
- (Optional) Create and activate a virtual environment before installing.

---

## 2. Data Preparation
1. Place raw PD recordings under `dataset/raw_pd/<cable_id>/...` or use the provided example data in `dataset/contactless_pd_detection`.
2. If Phase 0 (preprocessing) has been run, cleaned windows live under `partial_discharge_project/station_<ID>/data_clean/...`.
3. Review and adjust configuration parameters in `config.yaml` or per-experiment overrides in `experiment_configs.yaml`.

---

## 3. Stage-Based CLI with `main.py`

`main.py` exposes the pipeline via the `--stage` flag:

| Stage        | Command Example                              | Description                                                  |
|--------------|----------------------------------------------|--------------------------------------------------------------|
| preprocess   | `python main.py --stage preprocess`          | Clean raw signals, perform windowing, save normalized arrays |
| extract      | `python main.py --stage extract --force`     | Compute classic statistical features per window              |
| analyze      | `python main.py --stage analyze`             | Run feature expansion (pairwise ops + polynomial + MathFeatures) and feature selection tracks (Baseline, Featurewiz, MLJAR AutoML) |
| report       | `python main.py --stage report`              | Generate plots/reports (if implemented)                      |
| full-run     | `python main.py --stage full-run --force`    | Run all stages sequentially (Phase 0 → Phase 3)              |

**Common flags**
- `--force`: overwrite existing outputs for the stage (especially useful for `extract`).
- `--jobs N`: parallel workers where supported.
- `--advanced-denoise`, `--augment`, `--wavelet-feats`, `--optuna`: enable optional processing and hyper-parameter features.

> **Note:** `main.py` logs the current Git commit. If Git is unavailable, comment out or guard the `subprocess.check_output(["git", "rev-parse", "HEAD"])` line.

---

## 4. Running Module Functions Directly

Each phase is available via a `run(...)` function for scripting or notebooks:

```python
from pathlib import Path
from modules.feature_engineering import run as feats_run
from modules.feature_expansion import run as expand_run
from modules.feature_selection import run as select_run
from modules.model_training import run as train_run
from modules.evaluation_reporting import run as report_run

project_root = Path("partial_discharge_project")

# Classic statistical features (Phase 1)
feats_run(station_id=None, input_path=project_root, output_path=project_root, config={"fs": 1_000_000})

# Feature expansion (pairwise arithmetic + polynomial + feature-engine MathFeatures)
expand_run(None, project_root, project_root)

# Feature selection tracks (4A Baseline, 4B Featurewiz, 4C MLJAR AutoML)
select_run(None, project_root, project_root, config={"mljar": {"total_time_limit": 1200}})

# Optional downstream stages
train_run(None, project_root, project_root)
report_run(None, project_root, project_root)
```

- Pass a specific station ID to target a single dataset (e.g., `"52008"`).
- Track 4C (MLJAR) time limits can be increased via `config={"mljar": {"total_time_limit": ..., "model_time_limit": ...}}`.

---

## 5. One-Command End-to-End Pipeline

`run_pipeline.py` orchestrates the entire workflow with intermediate tests:

```
python run_pipeline.py --station 52008
```

Pipeline order:
1. Data cleaning (`modules.data_cleaning`)
2. Feature engineering (`modules.feature_engineering`)
3. Feature expansion (`modules.feature_expansion`)
4. Feature selection (`modules.feature_selection`)
5. Model training (`modules.model_training`)
6. Evaluation & reporting (`modules.evaluation_reporting`)

After each stage, targeted `pytest` suites (from `tests/test_pipeline.py`) validate the outputs. Any failure halts the process.

---

## 6. Output Locations

| Stage                     | Path                                                                                   | Contents                                                                          |
|---------------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Preprocessing             | `partial_discharge_project/station_<ID>/data_clean/...`                               | Cleaned signals, metadata, `.npz`/`.parquet` windows                              |
| Feature engineering       | `features/2_feature_engineering/classic_stats/`                                       | Per-station `.parquet` + combined `base_features.(parquet|csv)`                   |
| Feature expansion         | `features/3_feature_comb_expansion/mathematical_combination/`                         | `combined_features.(parquet|csv)` and `feature_combinations.md` documentation     |
| Feature selection tracks  | `features/4_feature_selection/tracks/<track_slug>/`                                   | Track-specific selected features, summaries, and label mappings                   |
| MLJAR artifacts (Track 4C)| `features/4_feature_selection/tracks/mljar_internal/automl_results/`, `leaderboard.csv` | AutoML logs, configuration, best-model leaderboard                                |
| Models & scalers          | `models/`                                                                             | Trained estimators (`*.joblib`) and preprocessing pipelines                       |
| Reports                   | `reports/`                                                                            | Confusion matrices, comparative plots, pipeline summaries                         |

---

## 7. Testing & Validation
- Run all tests: `python -m pytest`
- Stage-specific tests are triggered automatically by `run_pipeline.py`. Use them directly if needed:
  ```
  python -m pytest tests/test_pipeline.py -k test_feature_engineering_outputs
  ```
- Review logs stored under `partial_discharge_project/reports` and pipeline logs under `partial_discharge_project/logs`.

---

## 8. Tips & Troubleshooting
- **Performance warnings during feature expansion:** Large numbers of pairwise features trigger pandas fragmentation warnings—safe to ignore, but consider refactoring for performance if it becomes an issue.
- **MLJAR timeouts:** If Track 4C defaults back to all features, increase `total_time_limit` (`>= 900s` recommended for large feature sets).
- **Dependency issues:** `feature-engine` and `mljar-supervised` are required. Install manually via `python -m pip install feature-engine mljar-supervised` if necessary.
- **Custom configurations:** Modify `config.yaml` for dataset paths, model grids, and runtime options. `experiment_configs.yaml` contains pre-defined experiment setups.
- **Existing outputs:** Use `--force` or delete the `features/...` folders before re-running feature-related stages.

---

With these steps, you can operate the PD classification pipeline end-to-end or select individual components. Customize stages, integrate additional features, and extend model experimentation through the provided modular architecture. Happy experimenting!

