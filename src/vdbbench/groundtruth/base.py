from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GroundTruthSource(ABC):
    """Source of exact nearest-neighbor ground truth."""

    @abstractmethod
    def neighbors(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return exact GT neighbor ids, shape (n_queries, k)."""
