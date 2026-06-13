from __future__ import annotations

import numpy as np

from vdbbench.backends.base import VectorDBAdapter


def run_warmup(adapter: VectorDBAdapter, queries: np.ndarray, k: int, n_warmup: int) -> None:
    """Run n_warmup queries, cycling through the query set; discard timings."""
    n_queries = queries.shape[0]
    for i in range(n_warmup):
        adapter.search(queries[i % n_queries], k)
