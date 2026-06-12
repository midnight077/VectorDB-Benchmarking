from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vdbbench.core.types import BuildResult, Metric


class VectorDBAdapter(ABC):
    """Central abstraction: one adapter per (backend, index family)."""

    name: str

    @abstractmethod
    def setup(self, dim: int, metric: Metric, build_params: dict) -> None:
        """Create collection/table and configure the index with build-time params."""

    @abstractmethod
    def build_index(self, vectors: np.ndarray, ids: np.ndarray) -> BuildResult:
        """Insert vectors and build the index. Returns build_time_s and any DB-reported stats."""

    @abstractmethod
    def set_search_params(self, search_params: dict) -> None:
        """Set ef_search / nprobe etc. WITHOUT rebuilding the index."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[int]:
        """Single serial query -> list of neighbor ids. This is the ONLY timed call."""

    @abstractmethod
    def index_size_bytes(self) -> int: ...

    @abstractmethod
    def memory_rss_bytes(self) -> int:
        """In-process RSS delta for libs; server PID RSS for Qdrant/Postgres/Milvus."""

    @abstractmethod
    def teardown(self) -> None:
        """Drop collection/table; release resources between sweep points."""
