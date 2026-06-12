# vdbbench

CPU-only, single-query (serial) benchmarking harness for vector databases.
See `PROJECT_PLAN.md` for the full design and phased roadmap.

## Status

Phase 0: scaffold, config system, Lane A (ann-benchmarks HDF5) loader,
precomputed ground truth, FAISS `Flat` exact baseline.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Download the dev dataset (fashion-mnist-784-euclidean)

```bash
python scripts/download_dataset.py
```

## Run

```bash
vdbbench-run --config config/default.yaml
```
