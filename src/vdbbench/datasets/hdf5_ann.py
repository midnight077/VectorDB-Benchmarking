from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from vdbbench.core.types import Dataset, Metric
from vdbbench.datasets.base import DatasetSource

# ann-benchmarks `distance` attr / filename suffix -> internal Metric.
_METRIC_MAP = {
    "euclidean": Metric.L2,
    "l2": Metric.L2,
    "angular": Metric.COSINE,
    "cosine": Metric.COSINE,
    "dot": Metric.INNER_PRODUCT,
    "ip": Metric.INNER_PRODUCT,
}


class HDF5AnnDataset(DatasetSource):
    """Lane A loader: an ann-benchmarks-format HDF5 file (train/test/neighbors)."""

    def __init__(self, path: str | Path, metric: str | None = None):
        self.path = Path(path)
        self.metric_override = metric

    def load(self) -> Dataset:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.path}. "
                "Run scripts/download_dataset.py to fetch the dev dataset."
            )

        with h5py.File(self.path, "r") as f:
            train = f["train"][:].astype(np.float32)
            test = f["test"][:].astype(np.float32)
            neighbors = f["neighbors"][:].astype(np.int64) if "neighbors" in f else None
            metric_name = self.metric_override or f.attrs.get("distance")

        if metric_name is None:
            raise ValueError(
                f"Metric not specified and not found in {self.path} attrs; "
                "pass `metric` explicitly via config."
            )
        metric = _METRIC_MAP.get(str(metric_name).lower())
        if metric is None:
            raise ValueError(f"Unknown metric '{metric_name}' in {self.path}")

        return Dataset(
            train_vectors=train,
            test_vectors=test,
            dim=train.shape[1],
            metric=metric,
            gt_neighbors=neighbors,
        )
