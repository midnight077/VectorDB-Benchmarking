from __future__ import annotations

import argparse
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from vdbbench.backends.base import VectorDBAdapter
from vdbbench.backends.faiss_backend import FaissFlatAdapter, FaissHNSWAdapter
from vdbbench.backends.hnsw_fast_backend import HnswFastAdapter
from vdbbench.backends.hnsw_scalar_backend import HnswScalarAdapter
from vdbbench.config import BackendConfig, Config, load_config
from vdbbench.core.metrics import latency_percentiles, recall_at_k, serial_qps
from vdbbench.core.types import RunResult
from vdbbench.datasets.hdf5_ann import HDF5AnnDataset
from vdbbench.groundtruth.precomputed import PrecomputedGroundTruth
from vdbbench.harness.warmup import run_warmup
from vdbbench.storage.store import SQLiteStore

ADAPTERS: dict[str, type[VectorDBAdapter]] = {
    "faiss_flat": FaissFlatAdapter,
    "faiss_hnsw": FaissHNSWAdapter,
    "hnsw_scalar": HnswScalarAdapter,
    "hnsw_fast": HnswFastAdapter,
}


def run_backend(config: Config, backend_cfg: BackendConfig) -> list[RunResult]:
    """Build the index once, then measure and emit one RunResult per configured k."""
    dataset = HDF5AnnDataset(config.dataset.path, metric=config.dataset.metric).load()
    if dataset.gt_neighbors is None:
        raise ValueError(f"Dataset {config.dataset.name} has no ground-truth neighbors")
    gt_source = PrecomputedGroundTruth(dataset.gt_neighbors)

    adapter = ADAPTERS[backend_cfg.name]()
    adapter.setup(dataset.dim, dataset.metric, backend_cfg.build_params)

    ids = np.arange(dataset.train_vectors.shape[0], dtype=np.int64)
    build_result = adapter.build_index(dataset.train_vectors, ids)
    adapter.set_search_params(backend_cfg.search_params)

    k_max = max(config.k_values)
    n_queries = dataset.test_vectors.shape[0]

    # Warmup: discard timings.
    run_warmup(adapter, dataset.test_vectors, k_max, config.n_warmup)

    # Measure: time only adapter.search().
    latencies_ms = np.empty(config.n_repeats * n_queries, dtype=np.float64)
    retrieved = np.empty((n_queries, k_max), dtype=np.int64)
    idx = 0
    for rep in range(config.n_repeats):
        for i in range(10):
            start = time.perf_counter()
            result_ids = adapter.search(dataset.test_vectors[i], k_max)
            latencies_ms[idx] = (time.perf_counter() - start) * 1000.0
            idx += 1
            if rep == 0:
                retrieved[i] = result_ids

    perc = latency_percentiles(latencies_ms)
    qps = serial_qps(perc["mean_ms"])
    gt = gt_source.neighbors(dataset.test_vectors, k_max)
    build_time_s = build_result.build_time_s
    peak_rss_mb = adapter.memory_rss_bytes() / (1024 * 1024)
    index_disk_mb = adapter.index_size_bytes() / (1024 * 1024)

    timestamp = datetime.now(timezone.utc).isoformat()
    results = []
    for k in config.k_values:
        results.append(
            RunResult(
                run_id=uuid.uuid4().hex,
                timestamp=timestamp,
                lane=config.lane,
                dataset=config.dataset.name,
                backend=adapter.name,
                index_type=backend_cfg.index_type,
                metric=dataset.metric.value,
                dim=dataset.dim,
                build_params=backend_cfg.build_params,
                search_params=backend_cfg.search_params,
                k=k,
                recall_at_k=recall_at_k(retrieved, gt, k),
                p50_ms=perc["p50_ms"],
                p95_ms=perc["p95_ms"],
                p99_ms=perc["p99_ms"],
                mean_ms=perc["mean_ms"],
                serial_qps=qps,
                build_time_s=build_time_s,
                peak_rss_mb=peak_rss_mb,
                index_disk_mb=index_disk_mb,
                n_queries=n_queries,
                n_warmup=config.n_warmup,
                n_repeats=config.n_repeats,
            )
        )

    adapter.teardown()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vdbbench harness")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    store = SQLiteStore(config.results.path)

    for backend_cfg in config.backends:
        for result in run_backend(config, backend_cfg):
            store.save(result)
            print(
                f"{result.backend} ({result.index_type}) k={result.k}: "
                f"recall@{result.k}={result.recall_at_k:.4f} "
                f"p50={result.p50_ms:.3f}ms p95={result.p95_ms:.3f}ms p99={result.p99_ms:.3f}ms "
                f"mean={result.mean_ms:.3f}ms serial_qps={result.serial_qps:.1f} "
                f"build_time={result.build_time_s:.2f}s "
                f"rss={result.peak_rss_mb:.1f}MB disk={result.index_disk_mb:.1f}MB"
            )

    store.close()


if __name__ == "__main__":
    main()
