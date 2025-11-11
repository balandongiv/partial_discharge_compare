"""Lightweight local fallback implementation of the ``featurewiz`` API.

This module provides a minimal drop-in replacement for the external
``featurewiz`` package so the pipeline can execute in environments
where the original dependency is unavailable. It approximates the
Featurewiz workflow using correlation filtering followed by XGBoost
feature importance ranking.
"""

from .core import featurewiz

__all__ = ["featurewiz"]

