# Partial Discharge Compare

This repository contains a simple machine learning workflow implemented in the `ml_flow` package. It demonstrates loading a dataset, optional feature selection, hyper-parameter tuning and training multiple models.

## Running the workflow

```bash
python ml_flow/main.py --config ml_flow/config.json
```

Edit `config.json` to control dataset, models and other options.

## Checkpointing and Resuming

`train_model` now supports saving intermediate checkpoints when training XGBoost models. Two new configuration keys are available:

- `checkpoint_interval`: save a checkpoint every _N_ boosting rounds. Set to `null` to disable.
- `resume_training`: if `true`, the trainer will resume from the latest checkpoint in the `model` directory when training an XGBoost model.

Example excerpt in `config.json`:

```json
{
  "checkpoint_interval": 10,
  "resume_training": true
}
```

Checkpoints are stored in `model` with filenames like `XGBoost_checkpoint_10.json`. To start training from scratch, delete existing checkpoint files or set `resume_training` to `false`.
