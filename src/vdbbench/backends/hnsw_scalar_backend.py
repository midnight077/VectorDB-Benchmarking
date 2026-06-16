from __future__ import annotations

import os
import tempfile
import time

import _vdbhnsw
import numpy as np
import psutil

from vdbbench.backends.base import VectorDBAdapter
from vdbbench.core.types import BuildResult, Metric

# Cosine is handled via L2-normalized vectors + inner product (PROJECT_PLAN.md §2),
# so COSINE and INNER_PRODUCT both map to the IP distance kernel.
_CLASS_MAP = {
    Metric.L2: _vdbhnsw.HnswScalarL2,
    Metric.INNER_PRODUCT: _vdbhnsw.HnswScalarIP,
    Metric.COSINE: _vdbhnsw.HnswScalarIP,
}


class HnswScalarAdapter(VectorDBAdapter):
    """Custom from-scratch HNSW: scalar distances, serial build, single-thread search.

    See CUSTOM_HNSW_PLAN.md Phase A. Build params: M, ef_construction.
    Search param: ef_search, set via set_search_params without rebuilding.
    """

    name = "hnsw_scalar"

    def __init__(self):
        self.index = None
        self.dim: int | None = None
        self.metric: Metric | None = None
        self._rss_before: int = 0

    def setup(self, dim: int, metric: Metric, build_params: dict) -> None:
        self.dim = dim
        self.metric = metric
        self._rss_before = psutil.Process(os.getpid()).memory_info().rss
        m = build_params.get("M", 16)
        ef_construction = build_params.get("ef_construction", 200)
        cls = _CLASS_MAP[metric]
        self.index = cls(dim=dim, M=m, ef_construction=ef_construction, seed=42)

    def build_index(self, vectors: np.ndarray, ids: np.ndarray) -> BuildResult:
        # Positional ids 0..n-1 match the ann-benchmarks ground-truth id space.
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        start = time.perf_counter()
        self.index.add_points(vectors, parallel=False, num_threads=1)
        build_time_s = time.perf_counter() - start
        return BuildResult(
            build_time_s=build_time_s,
            extra={"index_memory_bytes": self.index.index_memory_bytes()},
        )

    def set_search_params(self, search_params: dict) -> None:
        if "ef_search" in search_params:
            self.index.set_ef(search_params["ef_search"])

    def search(self, query: np.ndarray, k: int) -> list[int]:
        query = np.ascontiguousarray(query, dtype=np.float32)
        return self.index.search(query, k)

    def index_size_bytes(self) -> int:
        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as f:
            path = f.name
        try:
            self.index.save(path)
            return os.path.getsize(path)
        finally:
            os.unlink(path)

    def memory_rss_bytes(self) -> int:
        rss_now = psutil.Process(os.getpid()).memory_info().rss
        return max(rss_now - self._rss_before, 0)

    def teardown(self) -> None:
        self.index = None
