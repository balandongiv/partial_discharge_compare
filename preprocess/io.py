from __future__ import annotations

"""I/O helpers for partial discharge signals."""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd


def load_pd_csv(path: str | Path | None) -> np.ndarray:
    """Load a CSV file containing a single time-series."""
    if path is None:
        return np.array([])
    data = pd.read_csv(path, header=None).iloc[:, 0].to_numpy()
    return data.astype(np.float32)


def load_pd_hdf5(path: str | Path | None) -> np.ndarray:
    """Load an HDF5 file containing a dataset named ``data``."""
    if path is None:
        return np.array([])
    with h5py.File(path, "r") as fh:
        key = list(fh.keys())[0]
        data = np.asarray(fh[key])
    return data.astype(np.float32).ravel()


def load_pd_npy(path: str | Path | None) -> np.ndarray:
    """Load a NumPy ``.npy`` file as a 1D array."""
    if path is None:
        return np.array([])
    data = np.load(path)
    return data.astype(np.float32).ravel()
