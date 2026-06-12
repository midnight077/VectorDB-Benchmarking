# Vector Database Benchmarking Harness — Build Plan

> Paste this into Claude Code as the project brief. Build it **phase by phase** (Phase 0 → 6). Do not skip ahead: each phase has acceptance criteria that must pass before starting the next. The goal is a CPU-only, single-query (serial) benchmark that compares vector databases on ANN recall and retrieval performance, with a pluggable ground-truth source and an interactive CLI.

---

## 1. Goal & scope

Build a modular benchmarking harness that indexes a dataset into multiple vector databases and measures **database-only** metrics:

- `recall@k` (vs exact ground truth)
- index build time
- search latency: `p50 / p95 / p99` and mean
- serial throughput = `1 / latency` (labeled **serial QPS**; true concurrent QPS is out of scope for now)
- peak process memory (RSS)
- on-disk index size

**Hard constraints**
- Runs on a high-performance **CPU** host (>40 GB RAM). No GPU.
- **Single query at a time, serially.** No concurrency in this version (it is a planned later phase — keep the search interface clean so it can be added without refactoring).
- Embedding model `bge-m3` is hosted remotely at an Ollama endpoint (default `http://192.168.3.143:11434`). It is **only** used in Lane B (see §3).

**Backends**
- **FAISS** — pure Python library, no server. Also provides the exact (`Flat`) baseline.
- **Qdrant** — run the **native binary** over localhost (NOT `qdrant-client` local/`:memory:` mode, which is an unrepresentative pure-Python reimplementation).
- **pgvector** — native PostgreSQL install with the `vector` extension.
- **Milvus** — **Docker exception**: run Milvus Standalone via docker-compose. (Milvus Lite is FLAT-only and pure-Python, so it cannot do an ANN sweep — Docker is required for a fair comparison.)

**Primary index family:** standardize on **HNSW** across all backends for the headline comparison. IVF is an optional secondary family. Comparing across index families is comparing algorithms, not databases — keep them separate.

---

## 2. Decisions baked in

- **Pluggable ground truth, Lane A is the default.** (See §3.)
- **Pre-embed all query vectors before timing.** The measured path is the DB search call only — no embedding latency inside it.
- **Distance metric must be consistent** across: ground-truth computation, index configuration, and any normalization. The adapter maps the dataset metric to each DB's metric name. For cosine, L2-normalize once centrally and use inner product.
- **Two-level sweep.** Build-time params (HNSW `M`, `ef_construction`; IVF `nlist`) require an index rebuild; search-time params (`ef_search`, `nprobe`) do not. Outer loop rebuilds (measures build time / RSS / disk size), inner loop sweeps search params to trace the recall/latency curve cheaply.
- **RSS is measured per-backend with the correct method and labeled as not-strictly-comparable.** In-process libraries (FAISS) → this process's RSS delta. Server engines (Qdrant binary, Postgres, Milvus) → RSS of the server PID(s) via `psutil`. Disk size and build time *are* directly comparable.

---

## 3. The two lanes (pluggable ground truth)

The harness must support two interchangeable sources of vectors + ground truth, selected by config.

**Lane A — precomputed ANN dataset (DEFAULT).**
Load an `ann-benchmarks`-format HDF5 file containing `train` (corpus vectors), `test` (query vectors), and `neighbors` (exact GT ids). `bge-m3` is **not** used. Dimension and metric are fixed by the dataset.
- Dev/smoke default: `fashion-mnist-784-euclidean` (60k, small, fast).
- Real benchmark: `glove-100-angular` (cosine) and/or `sift-128-euclidean` (L2).

**Lane B — own corpus + bge-m3.**
Load a raw text corpus, embed it with `bge-m3` via the Ollama endpoint (dense vector, 1024-dim, cosine), cache embeddings to disk, then **compute exact ground truth yourself** with a brute-force FAISS `Flat` search. More compute, realistic to the actual RAG stack. Ollama returns bge-m3's dense vector only (not sparse/ColBERT), so this is dense-only ANN.

Both lanes feed the **same** downstream interface: `(corpus_vectors, query_vectors, gt_neighbor_ids, dim, metric)`.

---

## 4. Module / package structure

```
vdbbench/
├── README.md
├── pyproject.toml                 # deps + console entrypoint
├── config/
│   ├── default.yaml               # dataset, lane, backends, k values, warmup, repeats
│   └── sweeps/                    # per-backend build/search param grids
├── data/                          # datasets, cached embeddings, GT  (gitignored)
├── results/                       # sqlite db / parquet / plots        (gitignored)
├── infra/
│   ├── milvus/docker-compose.yml  # the Docker exception
│   ├── qdrant/run_qdrant.sh       # download + run native binary
│   └── postgres/setup.sql         # CREATE EXTENSION vector; table DDL
└── src/vdbbench/
    ├── __init__.py
    ├── cli.py                     # interactive menu (Phase 6)
    ├── config.py                  # pydantic config load + validate
    ├── core/
    │   ├── types.py               # Metric enum, dataclasses (Dataset, RunResult, BuildResult)
    │   ├── normalize.py           # L2 normalize; map Metric -> per-DB metric name
    │   └── metrics.py             # recall@k, latency percentiles, serial qps
    ├── datasets/
    │   ├── base.py                # DatasetSource ABC
    │   ├── hdf5_ann.py            # Lane A loader
    │   └── corpus.py              # Lane B raw-text loader
    ├── groundtruth/
    │   ├── base.py                # GroundTruthSource ABC
    │   ├── precomputed.py         # Lane A: read neighbors from HDF5
    │   └── bruteforce.py          # Lane B: FAISS Flat exact KNN
    ├── embeddings/
    │   ├── base.py                # EmbeddingClient ABC
    │   ├── ollama_bge_m3.py       # /api/embed, batching, retry, timeout
    │   └── cache.py               # disk cache keyed by content hash
    ├── backends/
    │   ├── base.py                # VectorDBAdapter ABC  <-- key interface
    │   ├── faiss_backend.py
    │   ├── qdrant_backend.py
    │   ├── pgvector_backend.py
    │   └── milvus_backend.py
    ├── harness/
    │   ├── runner.py              # orchestrate build + warmup + measure
    │   ├── sweep.py               # two-level sweep loop
    │   ├── resources.py           # RSS (in-proc vs PID), disk size, timers
    │   └── warmup.py
    ├── storage/
    │   ├── schema.py              # results row schema
    │   └── store.py               # SQLite/Parquet persistence
    └── reporting/
        ├── pareto.py             # recall-vs-serial-QPS frontier plots
        └── tables.py             # summary tables
```

---

## 5. Key interfaces (implement these signatures)

### Vector DB adapter — the central abstraction
```python
# backends/base.py
from abc import ABC, abstractmethod
import numpy as np
from vdbbench.core.types import Metric, BuildResult

class VectorDBAdapter(ABC):
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
```

### Ground-truth source
```python
# groundtruth/base.py
class GroundTruthSource(ABC):
    @abstractmethod
    def neighbors(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return exact GT neighbor ids, shape (n_queries, k)."""
```

### Dataset source
```python
# datasets/base.py
class DatasetSource(ABC):
    @abstractmethod
    def load(self) -> "Dataset":
        """Returns Dataset(train_vectors, test_vectors, dim, metric, gt_or_none)."""
```

---

## 6. Metrics & results schema

`recall@k = |retrieved_ids ∩ gt_ids| / k`, averaged over the query set, at each configured `k` (default `k ∈ {1, 10, 100}`; GT must contain ≥ max(k) neighbors).

Persist **one row per measured operating point** (one backend × one build config × one search config):

```
run_id, timestamp, lane, dataset, backend, index_type, metric, dim,
build_params(json), search_params(json), k,
recall_at_k, p50_ms, p95_ms, p99_ms, mean_ms, serial_qps,
build_time_s, peak_rss_mb, index_disk_mb,
n_queries, n_warmup, n_repeats
```

Storage: SQLite (queryable) or Parquet. Reporting reads from storage only, so runs accumulate across sessions.

---

## 7. Measurement methodology (do exactly this)

1. Build the index for one build-param config; record `build_time_s`, `peak_rss_mb`, `index_disk_mb`.
2. For each search-param config (no rebuild):
   a. **Warmup:** run `n_warmup` queries (default 50), discard timings.
   b. **Measure:** run the fixed query set `n_repeats` times (default 3), one query at a time, timing only `adapter.search()`.
   c. Compute `recall@k` from retrieved ids vs GT.
   d. Compute `p50/p95/p99/mean` over all timed queries; `serial_qps = 1000 / mean_ms`.
   e. Persist a row.
3. Keep warm cache (steady state). Pin the process to fixed cores (`psutil` CPU affinity / `taskset`) for stable timings.
4. **Benchmark one backend at a time** — never run two DB servers under load simultaneously.

---

## 8. Sweep configuration

`config/sweeps/<backend>.yaml` defines the grid. Map equivalent knobs across backends:

| Family | Build params (rebuild) | Search params (no rebuild) |
|---|---|---|
| HNSW | `M`, `ef_construction` | `ef_search` |
| IVF  | `nlist` | `nprobe` |

Per-backend knob names to wire up:
- **FAISS:** `IndexHNSWFlat(M, efConstruction)`, set `efSearch`; IVF via `IndexIVFFlat(nlist)`, set `nprobe`.
- **Qdrant:** HNSW `m`, `ef_construct` (collection config); `hnsw_ef` at search.
- **pgvector:** HNSW `m`, `ef_construction`; `hnsw.ef_search` (SET per session). IVFFlat `lists`; `ivfflat.probes`.
- **Milvus:** `HNSW` with `M`, `efConstruction`; `ef` at search. `IVF_FLAT` with `nlist`; `nprobe`.

The sweep is the deliverable: each backend produces a set of (recall, serial_qps) points; the **recall-vs-serial-QPS Pareto frontier** is the headline plot.

---

## 9. Reporting

- `reporting/pareto.py`: for each dataset, plot recall@k (x) vs serial QPS (y) with one frontier line per backend (matplotlib). Up-and-to-the-right is better.
- `reporting/tables.py`: summary tables of build time, peak RSS, on-disk size per backend at matched recall (e.g., the search config closest to recall@10 = 0.95).
- Output PNGs + a markdown/CSV summary into `results/`.

---

## 10. Phased roadmap (build in this order)

**Phase 0 — Scaffold + exact baseline.**
Config system, `core/types`, `core/metrics`, Lane A HDF5 loader, `precomputed` GT, FAISS `Flat` adapter.
*Acceptance:* on `fashion-mnist`, FAISS Flat returns `recall@10 == 1.0`; a results row is written.

**Phase 1 — One real ANN backend end-to-end.**
FAISS HNSW adapter, warmup + percentiles + serial QPS, storage layer.
*Acceptance:* produces a row with `recall@10 < 1.0` and sane `p50/p95/p99`; rerun appends, doesn't overwrite.

**Phase 2 — Generalize + add backends.**
Lock the `VectorDBAdapter` ABC. Add Qdrant (native binary), pgvector (native Postgres), Milvus (docker-compose). Metric mapping + normalization centralized.
*Acceptance:* all four backends produce comparable rows on the same dataset/metric/k at one config each.

**Phase 3 — Two-level sweep + resource metrics.**
`harness/sweep.py`, `harness/resources.py` (per-backend RSS, disk size, build time).
*Acceptance:* a sweep yields multiple (recall, serial_qps) points per backend, each with build time / RSS / disk size.

**Phase 4 — Reporting.**
Pareto plots + summary tables.
*Acceptance:* one PNG per dataset with a frontier per backend; a summary table at matched recall.

**Phase 5 — Lane B (bge-m3).**
Corpus loader, Ollama bge-m3 client (batching, retry, timeout), embedding disk cache, brute-force FAISS GT.
*Acceptance:* a small corpus runs through Lane B end-to-end; embeddings cached and reused across all backends; GT recall sanity-checks against FAISS Flat = 1.0.

**Phase 6 — Interactive CLI.**
Menu over the programmatic API: select lane/dataset → select backends → build/index → run recall → run latency → run full sweep → generate plots → export results.
*Acceptance:* every action is runnable from the menu and also callable programmatically (menu is a thin shell).

**Out of scope now (later):** concurrency / load generator for true concurrent QPS. Keep `search()` single-query so a concurrent runner can wrap it later without changing adapters.

---

## 11. Setup / logistics

**Python deps:** `numpy`, `h5py` (HDF5 datasets), `faiss-cpu`, `qdrant-client`, `psycopg[binary]` + `pgvector`, `pymilvus`, `psutil`, `pydantic`, `pyyaml`, `pandas`, `pyarrow`, `matplotlib`, `requests` (Ollama), `tqdm`. Use a venv.

**Infra (native, except Milvus):**
- *Qdrant:* download the release binary (or `cargo build --release --bin qdrant`), run on `localhost:6333`. `infra/qdrant/run_qdrant.sh`.
- *pgvector:* install PostgreSQL natively, `CREATE EXTENSION vector;`, create the table/index. `infra/postgres/setup.sql`.
- *Milvus:* `docker compose -f infra/milvus/docker-compose.yml up -d` (standalone = milvus + etcd + minio). This is the one Docker dependency.
- *bge-m3:* Ollama endpoint reachable at `http://192.168.3.143:11434` from the benchmark host (Lane B only). Verify with a single `/api/embed` call before bulk embedding.

**Host:** high-perf CPU, >40 GB RAM. Benchmark one backend at a time so memory isn't contended. Datasets, embeddings, GT, and indexes go under `data/` (gitignored).

**Reproducibility:** fixed seeds, fixed query set, `n_repeats ≥ 3`, report variance. Pin CPU affinity for timing stability.

---
