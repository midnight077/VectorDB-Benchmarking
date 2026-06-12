# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`vdbbench` is a CPU-only, single-query (serial) benchmarking harness for vector databases
(FAISS now; Qdrant/pgvector/Milvus planned). The full design and phased roadmap live in
`PROJECT_PLAN.md` — read it before making architectural changes, since each phase has an
explicit acceptance criterion that later phases depend on.

**Current status: Phase 0 only** (per `PROJECT_PLAN.md` §10). Implemented: config system,
`core/types`, `core/metrics`, Lane A HDF5 loader, `precomputed` ground truth, FAISS `Flat`
adapter, SQLite results store. Not yet implemented: HNSW/IVF adapters, sweeps, resource
metrics beyond basic RSS/disk, reporting/plots, Lane B (bge-m3), CLI.

A full, beginner-oriented file-by-file walkthrough of the codebase exists at
`docs/CODEBASE.md` — use it instead of re-deriving how individual modules work.

## Commands

Setup (editable install, src-layout package):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download a dataset (ann-benchmarks HDF5 format; defaults to `fashion-mnist-784-euclidean`,
the small dev/smoke dataset):
```bash
python scripts/download_dataset.py [dataset-name]
```

Run a benchmark (writes one row per `(backend, k)` to SQLite at `results/results.db`):
```bash
vdbbench-run --config config/default.yaml
```
This is the `vdbbench.harness.runner:main` console entry point defined in `pyproject.toml`.

There is currently **no test suite, no linter config, and no separate build step** —
do not invent commands like `pytest`, `ruff`, or `make` unless you add the corresponding
config/tooling yourself.

## Architecture

### Data flow (harness/runner.py is the orchestrator)

```
Config (config/*.yaml, validated by config.py)
  -> Dataset (datasets/hdf5_ann.py: HDF5AnnDataset.load())
  -> GroundTruthSource (groundtruth/precomputed.py: wraps Dataset.gt_neighbors)
  -> VectorDBAdapter (backends/*.py: setup -> build_index -> set_search_params -> search)
  -> core/metrics.py (recall_at_k, latency_percentiles, serial_qps)
  -> RunResult (core/types.py)
  -> SQLiteStore (storage/store.py, append-only)
```

`run_backend()` in `src/vdbbench/harness/runner.py` builds the index **once**, then for each
configured `k` in `config.k_values` emits a separate `RunResult` row. Adding a new metric/k
value does not require a rebuild; adding a new build-param sweep point does (this distinction
is the basis for the Phase 3 two-level sweep that doesn't exist yet).

### The plugin pattern: three ABCs

Three abstract base classes define the extension points. Any new dataset format, ground-truth
strategy, or database backend must implement one of these (`src/vdbbench/*/base.py`):

- **`VectorDBAdapter`** (`backends/base.py`) — the central abstraction. 7 abstract methods:
  `setup`, `build_index`, `set_search_params`, `search`, `index_size_bytes`,
  `memory_rss_bytes`, `teardown`. `search()` is the *only* method that gets timed — keep it
  free of side work. New backends must be registered in the `ADAPTERS` dict in
  `harness/runner.py` (keyed by the name used in `config.backends[].name`).
- **`DatasetSource`** (`datasets/base.py`) — `load() -> Dataset`. `HDF5AnnDataset` is the
  Lane A implementation.
- **`GroundTruthSource`** (`groundtruth/base.py`) — `neighbors(queries, k) -> np.ndarray`.
  `PrecomputedGroundTruth` is the Lane A implementation (reads `neighbors` straight from the
  HDF5 file).

### Lane A vs Lane B

Per `PROJECT_PLAN.md` §3, the harness supports two interchangeable sources of
`(corpus_vectors, query_vectors, gt_neighbor_ids, dim, metric)`:

- **Lane A (default, implemented)** — precomputed `ann-benchmarks` HDF5 datasets
  (`train`/`test`/`neighbors`/`attrs["distance"]`). No embedding involved.
- **Lane B (Phase 5, not implemented)** — own text corpus embedded via `bge-m3` over Ollama,
  with ground truth computed by brute-force FAISS `Flat`. Both lanes must feed the same
  downstream `Dataset` shape so the harness/metrics/storage layers don't need to know which
  lane produced the data.

### Metric handling

`core/types.Metric` (`L2`, `COSINE`, `INNER_PRODUCT`) is the canonical metric enum used
everywhere. Each backend adapter maps it to its own representation (e.g.
`backends/faiss_backend.py` maps `COSINE -> faiss.METRIC_INNER_PRODUCT`, since cosine is
handled via L2-normalized vectors + inner product per `PROJECT_PLAN.md` §2). When adding a
backend, keep this mapping local to the adapter rather than leaking DB-specific constants
into `core`.

### Results schema

`RunResult` (`core/types.py`) has 23 fields mirrored exactly by `CREATE_TABLE_SQL` in
`storage/schema.py` — one row per `(backend, build_params, search_params, k)` operating
point, with `build_params`/`search_params` stored as JSON text. `SQLiteStore.save()` is
append-only (`run_id` is a fresh UUID per run), so repeated runs accumulate rather than
overwrite — reporting (Phase 4, not yet built) is expected to read the full history from
`results/results.db`.

### Measurement methodology (do not deviate without checking PROJECT_PLAN.md §7)

For each backend: build index once (records `build_time_s`, RSS delta, serialized index
size) -> run `n_warmup` queries cycling through the query set (timings discarded) -> run
`n_repeats * n_queries` queries at `k_max = max(config.k_values)`, timing only
`adapter.search()` -> compute `recall@k` per configured `k` from the retrieved ids captured
on the first repeat (`rep == 0`) vs ground truth.
