from __future__ import annotations

import numpy as np

from vdbbench.groundtruth.base import GroundTruthSource


class PrecomputedGroundTruth(GroundTruthSource):
    """Lane A: ground truth shipped with an ann-benchmarks HDF5 dataset."""

    def __init__(self, neighbors: np.ndarray):
        self._neighbors = neighbors

    def neighbors(self, queries: np.ndarray, k: int) -> np.ndarray:
        if k > self._neighbors.shape[1]:
            raise ValueError(
                f"Requested k={k} exceeds precomputed neighbors width "
                f"{self._neighbors.shape[1]}"
            )
        return self._neighbors[:, :k]
