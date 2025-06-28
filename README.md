# Partial Discharge Compare

This repository contains a small machine learning workflow built around `ml_flow/`.

## Configuration

All parameters are provided via `ml_flow/config.yaml`.  The most important option
is `datasets`, which is a list of dataset definitions.  Each dataset entry
supports the following fields:

- `name`: a descriptive name for the dataset. When set to `iris` with no
  `path`, the builtin Iris dataset is loaded.
- `path`: optional path to a CSV file. If provided, the file is loaded instead of
  the builtin dataset loader.
- `label_mapping`: optional mapping of original target labels to new labels.

Other configuration options:

- `model_types`: list of models to evaluate.
- `scaling_methods`: feature scaling strategies.
- `feature_selections`: feature selection methods (`null` means no selection).
- `hyperparameter_tuning`: enable Optuna based tuning when `true`.
- `n_trials_optuna`: number of Optuna trials.
- `cv_folds`: number of cross‑validation folds.
- `test_size`: fraction of the data used for the test split.
- `val_size`: fraction of the data used for the validation split.
- `random_state`: random seed for reproducible splits.
- `target_column`: name of the target column.

## Running

```
python ml_flow/main.py --config ml_flow/config.yaml
```

The script iterates over all datasets and configurations and stores evaluation
results under `ml_flow/`.
