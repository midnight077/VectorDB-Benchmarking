from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Metric(str, Enum):
    """Distance/similarity metric, shared by datasets, ground truth, and adapters."""

    L2 = "l2"
    COSINE = "cosine"
    INNER_PRODUCT = "ip"


@dataclass
class Dataset:
    """In-memory representation of a benchmark dataset (corpus + queries)."""

    train_vectors: np.ndarray
    test_vectors: np.ndarray
    dim: int
    metric: Metric
    gt_neighbors: np.ndarray | None = None


@dataclass
class BuildResult:
    """Outcome of VectorDBAdapter.build_index."""

    build_time_s: float
    extra: dict = field(default_factory=dict)


@dataclass
class RunResult:
    """One measured operating point: a single (backend, build config, search config, k) row."""

    run_id: str
    timestamp: str
    lane: str
    dataset: str
    backend: str
    index_type: str
    metric: str
    dim: int
    build_params: dict
    search_params: dict
    k: int
    recall_at_k: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    serial_qps: float
    build_time_s: float
    peak_rss_mb: float
    index_disk_mb: float
    n_queries: int
    n_warmup: int
    n_repeats: int
