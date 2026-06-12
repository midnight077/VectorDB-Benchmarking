from __future__ import annotations

from abc import ABC, abstractmethod

from vdbbench.core.types import Dataset


class DatasetSource(ABC):
    """Source of corpus + query vectors (and optionally ground truth)."""

    @abstractmethod
    def load(self) -> Dataset:
        """Returns Dataset(train_vectors, test_vectors, dim, metric, gt_neighbors)."""
