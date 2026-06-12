from __future__ import annotations

import os
import time

import faiss
import numpy as np
import psutil

from vdbbench.backends.base import VectorDBAdapter
from vdbbench.core.types import BuildResult, Metric

_METRIC_MAP = {
    Metric.L2: faiss.METRIC_L2,
    Metric.INNER_PRODUCT: faiss.METRIC_INNER_PRODUCT,
    # Cosine requires vectors to be L2-normalized upstream (core/normalize.py, Phase 2+).
    Metric.COSINE: faiss.METRIC_INNER_PRODUCT,
}


class FaissFlatAdapter(VectorDBAdapter):
    """Exact brute-force baseline (faiss.IndexFlat) -- the recall@k == 1.0 reference."""

    name = "faiss_flat"

    def __init__(self):
        self.index: faiss.Index | None = None
        self.dim: int | None = None
        self.metric: Metric | None = None
        self._rss_before: int = 0

    def setup(self, dim: int, metric: Metric, build_params: dict) -> None:
        self.dim = dim
        self.metric = metric
        self._rss_before = psutil.Process(os.getpid()).memory_info().rss
        self.index = faiss.IndexFlat(dim, _METRIC_MAP[metric])

    def build_index(self, vectors: np.ndarray, ids: np.ndarray) -> BuildResult:
        # IndexFlat has no add_with_ids; it assigns positional ids 0..n-1, which
        # is exactly the ann-benchmarks ground-truth id space.
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        start = time.perf_counter()
        self.index.add(vectors)
        build_time_s = time.perf_counter() - start
        return BuildResult(build_time_s=build_time_s)

    def set_search_params(self, search_params: dict) -> None:
        pass  # Flat index has no search-time parameters.

    def search(self, query: np.ndarray, k: int) -> list[int]:
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        _, ids = self.index.search(query, k)
        return ids[0].tolist()

    def index_size_bytes(self) -> int:
        return faiss.serialize_index(self.index).nbytes

    def memory_rss_bytes(self) -> int:
        rss_now = psutil.Process(os.getpid()).memory_info().rss
        return max(rss_now - self._rss_before, 0)

    def teardown(self) -> None:
        self.index = None
