"""Tests for experiment configurations."""

from experiments import EXPERIMENTS, ExperimentTrack


def test_experiment_count() -> None:
    assert len(EXPERIMENTS) == 5


def test_experiment_instances() -> None:
    for exp in EXPERIMENTS.values():
        assert isinstance(exp, ExperimentTrack)
