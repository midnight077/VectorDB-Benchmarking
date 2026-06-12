from __future__ import annotations

import numpy as np


def recall_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """recall@k = |retrieved_ids ∩ gt_ids| / k, averaged over queries.

    retrieved_ids: (n_queries, >=k) ids returned by the adapter.
    gt_ids: (n_queries, >=k) exact ground-truth neighbor ids.
    """
    n_queries = retrieved_ids.shape[0]
    total = 0.0
    for i in range(n_queries):
        retrieved_set = set(retrieved_ids[i, :k].tolist())
        gt_set = set(gt_ids[i, :k].tolist())
        total += len(retrieved_set & gt_set) / k
    return total / n_queries


def latency_percentiles(latencies_ms: np.ndarray | list[float]) -> dict[str, float]:
    """p50/p95/p99/mean (ms) over a set of timed query latencies."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }


def serial_qps(mean_ms: float) -> float:
    """Serial throughput = 1 / mean latency (not true concurrent QPS)."""
    return 1000.0 / mean_ms if mean_ms > 0 else float("inf")
