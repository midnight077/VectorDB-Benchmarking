# vdbbench Codebase Documentation

This document explains **every file** in the `vdbbench` project: what it does,
what each function and class is for, and how all the pieces connect into a
working program. It assumes **no prior Python experience** — every
Python-specific concept (decorators, type hints, classes, etc.) is explained
the first time it appears, with a quick-reference glossary in [Section 2](#2-python-primer-read-this-first).

If you want the *design rationale* (why HNSW, why these databases, what
Phase 1-6 will add), see `PROJECT_PLAN.md` in the repo root. This document is
about the *code that exists right now* (Phase 0).

---

## Table of contents

1. [The big picture](#1-the-big-picture)
2. [Python primer (read this first)](#2-python-primer-read-this-first)
3. [Project layout](#3-project-layout)
4. [Configuration files](#4-configuration-files)
5. [`src/vdbbench/__init__.py`](#5-srcvdbbench__init__py)
6. [The `core` package](#6-the-core-package)
7. [`src/vdbbench/config.py`](#7-srcvdbbenchconfigpy)
8. [The `datasets` package](#8-the-datasets-package)
9. [The `groundtruth` package](#9-the-groundtruth-package)
10. [The `backends` package](#10-the-backends-package)
11. [The `storage` package](#11-the-storage-package)
12. [The `harness` package](#12-the-harness-package)
13. [`scripts/download_dataset.py`](#13-scriptsdownload_datasetpy)
14. [End-to-end walkthrough](#14-end-to-end-walkthrough)
15. [Where to go next](#15-where-to-go-next)

---

## 1. The big picture

`vdbbench` measures how well different **vector databases** (FAISS for now;
Qdrant, pgvector, Milvus later) find the "nearest neighbors" of a query
vector, and how that compares to the *exact* (brute-force, 100% correct)
answer.

For Phase 0, the program does exactly one thing end-to-end:

1. Read a **config file** (`config/default.yaml`) describing what dataset and
   which database "adapter" to test.
2. **Load a dataset**: a file full of vectors (the "corpus" to search) plus a
   separate set of "query" vectors, plus the *correct* answer for each query
   (precomputed by the dataset's authors).
3. Build an **exact search index** using FAISS (`faiss_flat`) — this is a
   brute-force index that always returns the mathematically correct nearest
   neighbors.
4. Run every query vector through the index, **timing each search**.
5. Compare the returned neighbor IDs to the "correct answer" IDs and compute
   **recall@k** (what fraction of the true top-k neighbors did we actually
   get back?).
6. Compute latency statistics (p50/p95/p99/mean) and a "serial queries per
   second" number.
7. Write one row of results into a **SQLite database file**
   (`results/results.db`).

Every later phase (Phase 1-6) reuses these same pieces — they just add more
*adapters* (real ANN indexes like HNSW, and real database servers), more
*sweeps* (try many parameter combinations), and reporting/plots.

---

## 2. Python primer (read this first)

This section explains every Python language feature used in the codebase.
Skim it now; refer back to it as you read the file walkthroughs.

### 2.1 Files, modules, and packages

- Every `.py` file is called a **module**. You can `import` it from other
  files.
- A **package** is a folder that contains a (possibly empty) file named
  `__init__.py`. That file's presence tells Python "treat this folder as an
  importable package". For example, `src/vdbbench/core/__init__.py` (empty)
  is what makes `from vdbbench.core.types import Metric` work.
- Dotted paths like `vdbbench.core.types` mean: package `vdbbench` → sub-package
  `core` → module `types.py`.
- `src/vdbbench/__init__.py` (non-empty here — see [Section 5](#5-srcvdbbench__init__py))
  is the top of the package; it's the thing `pyproject.toml` tells Python how
  to install.

### 2.2 Imports

```python
import os                                  # import a whole module, use as os.getpid()
from pathlib import Path                   # import one name directly, use as Path(...)
from vdbbench.core.types import Metric     # import from our own package
```

### 2.3 `from __future__ import annotations`

Every file in this project starts with this exact line. It changes how
Python treats **type hints** (see next section): instead of evaluating them
immediately when the file is loaded, Python keeps them as plain text and only
looks at them if a tool (like Pydantic) explicitly asks. Practically, this
lets the code use modern, readable syntax like `int | None` (see below) and
let a class refer to itself in its own method signatures, regardless of the
exact Python version. You can treat this line as boilerplate that should
always be there and ignore it otherwise.

### 2.4 Type hints

```python
def recall_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
```

- `retrieved_ids: np.ndarray` means "the `retrieved_ids` parameter is
  *expected* to be a NumPy array".
- `-> float` means "this function is expected to return a `float`".
- **Type hints are documentation + tooling aids.** Plain Python does **not**
  enforce them at runtime — nothing crashes if you pass the "wrong" type to a
  normal function. The one place hints *are* actively enforced in this
  codebase is **Pydantic models** (see [2.9](#29-pydantic-models)), which
  validate data loaded from YAML/JSON against the declared types.

Common hint syntax you'll see:

| Syntax | Meaning |
|---|---|
| `int`, `str`, `float`, `bool` | the obvious built-in types |
| `np.ndarray` | a NumPy array (see [2.10](#210-numpy-arrays)) |
| `list[int]` | a list containing only `int`s |
| `dict[str, float]` | a dict whose keys are `str` and values are `float` |
| `X \| None` | "a value of type `X`, **or** `None`" — Python's modern way to write "optional" |
| `type[Foo]` | "the *class* `Foo` itself (or a subclass), not an *instance* of it" — see [10.2](#102-backendsfaiss_backendpy) |

### 2.5 Classes, `self`, and `__init__`

```python
class FaissFlatAdapter:
    def __init__(self):
        self.index = None   # an "instance attribute"

    def teardown(self):
        self.index = None
```

- `class Foo:` defines a new type. Creating an object of that type —
  `Foo()` — is called **instantiating** the class.
- `__init__` is the **constructor**: a special method Python calls
  automatically when you write `Foo()`. It's where you set up the object's
  initial state.
- `self` is the first parameter of every method and refers to "this specific
  object". `self.index = None` stores a value named `index` *on this
  object*; it's called an **instance attribute**, and every object created
  from the class gets its own independent copy.
- A value written directly under `class Foo:` *without* `self.` (e.g.
  `name = "faiss_flat"` in [`FaissFlatAdapter`](#102-backendsfaiss_backendpy)) is a
  **class attribute** — it belongs to the class itself and is shared as a
  default for all instances.

### 2.6 Decorators

A **decorator** is a marker written as `@something` placed directly above a
function or class definition. It means "before defining this thing, run it
through `something`, which can inspect, modify, or wrap it." This codebase
uses two decorators:

- **`@dataclass`** — auto-generates boilerplate for "plain data" classes (see
  [2.7](#27-dataclasses)).
- **`@abstractmethod`** — marks a method that subclasses *must* implement
  (see [2.8](#28-abstract-base-classes-abc-and-abstractmethod)).

### 2.7 Dataclasses

Writing a class that's just a labeled bundle of fields normally requires a
hand-written constructor:

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

The `@dataclass` decorator (from the built-in `dataclasses` module) generates
this `__init__` (plus a readable `__repr__` for printing, and `__eq__` for
comparing) automatically from type-hinted field declarations:

```python
@dataclass
class Point:
    x: int
    y: int
```

`Point(3, 4)` now works, `print(Point(3, 4))` shows `Point(x=3, y=4)`, and
`Point(3, 4) == Point(3, 4)` is `True`. This codebase uses `@dataclass` for
[`Dataset`, `BuildResult`, and `RunResult`](#61-coretypespy) — they're just
typed bundles of values passed between components.

One wrinkle: `field(default_factory=dict)` (used for `BuildResult.extra`). If
you wrote `extra: dict = {}` directly, **every instance would share the same
dict object** (a well-known Python pitfall — mutable default arguments).
`field(default_factory=dict)` tells Python "call `dict()` fresh for each new
instance instead".

### 2.8 Abstract Base Classes (`ABC`) and `@abstractmethod`

This is the most important pattern in the codebase — it's how `vdbbench`
defines "plug-in" interfaces that every database adapter must follow.

```python
from abc import ABC, abstractmethod

class VectorDBAdapter(ABC):
    @abstractmethod
    def search(self, query, k) -> list[int]:
        """Single serial query -> list of neighbor ids."""
```

- `ABC` (Abstract Base Class, from the built-in `abc` module) is a special
  base class. Inheriting from it — `class VectorDBAdapter(ABC):` — turns the
  class into a **contract/template** rather than something you can use
  directly.
- `@abstractmethod` marks a method as "every concrete subclass *must*
  override this with a real implementation".
- **You cannot create an instance of `VectorDBAdapter` itself** —
  `VectorDBAdapter()` raises a `TypeError`. And if a subclass (like
  `FaissFlatAdapter`) forgets to implement even one `@abstractmethod`,
  *creating that subclass* also raises a `TypeError`.

This is Python's equivalent of an **interface** in languages like Java/C#/Go:
it specifies *what* methods must exist and their signatures, without saying
*how* they work. `FaissFlatAdapter` is one *implementation* of the
`VectorDBAdapter` interface; later, `QdrantAdapter`, `PgVectorAdapter`, etc.
will be others — and the rest of the codebase (the harness) only ever talks
to the *interface*, so it doesn't care which one it's given.

A method body that's just `...` (three dots, called an **Ellipsis**) is a
placeholder meaning "no implementation here" — used to keep one-line abstract
method declarations compact, e.g.:

```python
@abstractmethod
def index_size_bytes(self) -> int: ...
```

### 2.9 Pydantic models

[`config.py`](#7-srcvdbbenchconfigpy) defines its settings as subclasses of
`pydantic.BaseModel` instead of `@dataclass`. Pydantic does two extra things
dataclasses don't:

1. **Validation**: when you construct `Config(**raw)` from a plain
   dict/YAML, Pydantic checks every field against its type hint and raises a
   clear error if something doesn't match (e.g., a string where a number was
   expected) or is missing.
2. **Defaults via `Field(...)`**: similar to `field(default_factory=...)` for
   dataclasses, `Field(default_factory=dict)` / `Field(default_factory=lambda: [10])`
   gives each `Config` instance its own fresh `dict`/`list` rather than
   sharing one.

### 2.10 NumPy arrays

`numpy` (imported as `np`) is the standard library for working with large
grids of numbers efficiently. Things you'll see:

- `np.ndarray` — the array type itself.
- `arr.astype(np.float32)` — convert the array's numbers to 32-bit floats
  (FAISS requires `float32`).
- `arr[:, :k]` — "all rows, but only the first `k` columns" (2D slicing).
- `arr.shape` — a tuple of dimensions, e.g. `(10000, 784)` = 10,000 rows of
  784 numbers each.
- `arr.tolist()` — convert a NumPy array into a plain Python `list`.
- `np.arange(n)` — an array `[0, 1, 2, ..., n-1]`.
- `np.percentile(arr, 95)` — the 95th percentile value.

### 2.11 f-strings

`f"text {expression}"` is a **formatted string literal**: anything inside
`{}` is evaluated and inserted into the string. E.g.
`f"recall@{result.k}={result.recall_at_k:.4f}"` produces
`"recall@10=1.0000"` (the `:.4f` part means "format as a fixed-point number
with 4 digits after the decimal point").

### 2.12 The `with` statement (context managers)

```python
with open(path) as f:
    data = f.read()
# file is automatically closed here, even if an error occurred above
```

`with` guarantees cleanup code runs when the indented block ends — used here
for opening HDF5 files (`h5py.File`) and plain files.

### 2.13 Conditional (`if/else`) expressions and `or`

- `A if condition else B` — a one-line "if/else" that *evaluates to a value*.
  E.g. `neighbors = f["neighbors"][:].astype(np.int64) if "neighbors" in f else None`
  means "if the HDF5 file has a `neighbors` dataset, load and convert it;
  otherwise use `None`".
- `A or B` — Python's "or" short-circuits: if `A` is *truthy* (not `None`,
  not empty, not zero/`False`), the expression evaluates to `A`; otherwise it
  evaluates to `B`. E.g.
  `metric_name = self.metric_override or f.attrs.get("distance")` means "use
  the explicit override if one was given, otherwise fall back to whatever the
  dataset file says".

### 2.14 `**` dict-unpacking

`Config(**raw)` — if `raw` is a dict like `{"lane": "A", "dataset": {...}}`,
`**raw` "unpacks" it so each key/value becomes a keyword argument, equivalent
to writing `Config(lane="A", dataset={...})`.

### 2.15 Leading underscores (`_name`)

A name starting with `_` (e.g. `_neighbors`, `_METRIC_MAP`, `_rss_before`) is
a **convention**, not a language rule: it signals "internal detail of this
class/module — don't rely on it from outside". Python doesn't actually
prevent access.

### 2.16 `if __name__ == "__main__":`

A standard idiom meaning "only run this code when the file is executed
directly (e.g. `python runner.py`), not when another file `import`s it".

### 2.17 Other standard-library pieces used

| Used in | What it is |
|---|---|
| `pathlib.Path` | Object-oriented file paths; `/` joins paths, `.exists()`, `.parent`, `.mkdir(parents=True, exist_ok=True)` |
| `argparse` | Parses command-line flags like `--config config/default.yaml` |
| `uuid.uuid4().hex` | Generates a random unique ID string |
| `datetime.now(timezone.utc).isoformat()` | Current UTC timestamp as text, e.g. `"2026-06-12T06:38:17+00:00"` |
| `time.perf_counter()` | A high-resolution clock, used to time operations |
| `dataclasses.asdict(obj)` | Converts a `@dataclass` instance into a plain `dict` |
| `json.dumps(obj)` | Converts a Python dict/list into a JSON text string |
| `sqlite3` | Built-in interface to a SQLite database file |
| `os.getpid()` / `psutil` | Get the current process's ID / memory usage |

---

## 3. Project layout

```
Rag_benchmarking/
├── pyproject.toml              # package metadata, dependencies, console script
├── README.md
├── PROJECT_PLAN.md             # full design doc / phased roadmap
├── config/
│   └── default.yaml            # Phase 0 run configuration
├── data/                       # downloaded datasets (gitignored)
├── results/                    # results.db (SQLite, gitignored)
├── scripts/
│   └── download_dataset.py     # fetches an ann-benchmarks HDF5 file
└── src/vdbbench/
    ├── __init__.py
    ├── config.py                # Pydantic config schema + YAML loader
    ├── core/
    │   ├── __init__.py
    │   ├── types.py             # Metric enum, Dataset/BuildResult/RunResult
    │   └── metrics.py           # recall@k, latency percentiles, serial QPS
    ├── datasets/
    │   ├── __init__.py
    │   ├── base.py               # DatasetSource interface
    │   └── hdf5_ann.py           # Lane A: ann-benchmarks HDF5 loader
    ├── groundtruth/
    │   ├── __init__.py
    │   ├── base.py               # GroundTruthSource interface
    │   └── precomputed.py        # Lane A: GT shipped with the HDF5 file
    ├── backends/
    │   ├── __init__.py
    │   ├── base.py               # VectorDBAdapter interface (the central abstraction)
    │   └── faiss_backend.py      # FaissFlatAdapter (exact baseline)
    ├── storage/
    │   ├── __init__.py
    │   ├── schema.py              # SQLite table definition
    │   └── store.py               # SQLiteStore (append-only results writer)
    └── harness/
        ├── __init__.py
        └── runner.py              # orchestrates everything; CLI entry point
```

`__init__.py` files inside `core/`, `datasets/`, `groundtruth/`, `backends/`,
`harness/`, and `storage/` are **empty** — their only job is to mark the
folder as an importable package (see [2.1](#21-files-modules-and-packages)).

---

## 4. Configuration files

### 4.1 `pyproject.toml`

This is the standard Python project manifest. The key parts:

- `[project] dependencies = [...]` — the third-party libraries this project
  needs (NumPy, h5py, faiss-cpu, psutil, pydantic, pyyaml, pandas, pyarrow,
  tqdm, requests). Running `pip install -e .` installs all of these.
- `[project.optional-dependencies]` — extra dependency groups for later
  phases (`qdrant`, `pgvector`, `milvus`, `viz`) that aren't needed yet and
  so aren't installed by default.
- `[project.scripts] vdbbench-run = "vdbbench.harness.runner:main"` — this
  creates a command-line program called `vdbbench-run`. When you type
  `vdbbench-run` in the terminal (inside the virtualenv), Python runs the
  `main()` function from `src/vdbbench/harness/runner.py`.
- `[tool.setuptools.packages.find] where = ["src"]` — tells the build tool
  that the actual Python package lives under `src/`, not the repo root (this
  is the common "src layout").

### 4.2 `config/default.yaml`

```yaml
lane: A

dataset:
  name: fashion-mnist-784-euclidean
  path: data/fashion-mnist-784-euclidean.hdf5
  metric: euclidean

k_values: [10]

n_warmup: 50
n_repeats: 3

backends:
  - name: faiss_flat
    index_type: flat
    build_params: {}
    search_params: {}

results:
  backend: sqlite
  path: results/results.db
```

This YAML file is parsed by [`config.py`](#7-srcvdbbenchconfigpy) into a
`Config` object. Field-by-field:

| Key | Meaning |
|---|---|
| `lane: A` | "Lane A" = use a precomputed ann-benchmarks dataset (vs. "Lane B" = embed your own text corpus, a later phase) |
| `dataset.name` | A human-readable label, stored in results rows |
| `dataset.path` | Where the `.hdf5` dataset file lives on disk |
| `dataset.metric` | Distance metric (`euclidean`, `angular`, ...); overrides whatever the HDF5 file itself says, if set |
| `k_values` | Which `k` values to compute `recall@k` for — one results row is written per value. Currently just `[10]` |
| `n_warmup` | Number of "throwaway" queries run before timing starts, to warm up caches |
| `n_repeats` | How many times to run the full query set while timing |
| `backends` | A list of database adapters to test — currently just `faiss_flat` |
| `results.path` | Path to the SQLite database file results get appended to |

---

## 5. `src/vdbbench/__init__.py`

```python
__version__ = "0.1.0"
```

This is the top-level file of the `vdbbench` package. `__version__` is a
conventional variable name that other tools (and `pip`) can read to find out
the installed version of the package. It currently has no other behaviour —
it exists primarily so that `src/vdbbench/` is recognized as the root package
referenced by `pyproject.toml`.

---

## 6. The `core` package

The `core` package holds the foundational, dependency-free pieces shared by
*every* other part of the program: the shared **data shapes** (`types.py`)
and the **math** for scoring a run (`metrics.py`). Nothing in `core` knows
anything about FAISS, HDF5, SQLite, etc. — that separation is intentional, so
these definitions can be imported anywhere without pulling in heavy
dependencies or creating circular imports.

### 6.1 `core/types.py`

This file defines the shared "nouns" of the system — the data structures
that flow between datasets, ground truth, adapters, and storage.

#### `class Metric(str, Enum)`

```python
class Metric(str, Enum):
    L2 = "l2"
    COSINE = "cosine"
    INNER_PRODUCT = "ip"
```

An **`Enum`** (enumeration) defines a fixed, named set of possible values —
here, the three distance/similarity metrics the harness understands:

- `L2` — Euclidean ("straight-line") distance. Smaller = more similar.
- `COSINE` — cosine similarity (angle between vectors). Used after
  L2-normalizing vectors.
- `INNER_PRODUCT` — raw dot product.

By also inheriting from `str`, each member *is* a string: `Metric.L2 == "l2"`
is `True`, and `Metric.L2.value` is `"l2"`. This matters because:

- Config files and HDF5 dataset attributes describe metrics as plain strings
  (`"euclidean"`, `"angular"`, ...).
- `Metric` lets the rest of the code work with a small, type-checked set of
  values instead of arbitrary strings, while still being easy to convert
  to/from the strings that come from YAML/HDF5/SQLite.

`Metric` is used by: [`Dataset`](#dataclass-class-dataset) (what metric the data uses),
[`VectorDBAdapter.setup`](#101-backendsbasepy) (which metric to configure the
index for), and [`faiss_backend.py`](#102-backendsfaiss_backendpy) (mapping to FAISS's
own metric constants).

#### `@dataclass class Dataset`

```python
@dataclass
class Dataset:
    train_vectors: np.ndarray
    test_vectors: np.ndarray
    dim: int
    metric: Metric
    gt_neighbors: np.ndarray | None = None
```

The in-memory representation of "everything you need to run a benchmark":

- `train_vectors` — the **corpus**: every vector that gets inserted into the
  database/index (shape `(n_corpus, dim)`).
- `test_vectors` — the **queries**: vectors to search for (shape
  `(n_queries, dim)`).
- `dim` — the dimensionality of each vector (e.g. `784` for fashion-mnist).
- `metric` — a [`Metric`](#class-metricstr-enum) value describing how
  distance/similarity is computed for this dataset.
- `gt_neighbors` — the **ground truth**: for each query, the IDs of its true
  nearest neighbors in `train_vectors`, shape `(n_queries, k_gt)`. `None` if
  the dataset doesn't ship this (Lane B will compute it separately via
  brute-force FAISS).

Produced by any [`DatasetSource.load()`](#81-datasetsbasepy) (currently
[`HDF5AnnDataset`](#82-datasetshdf5_annpy)).

#### `@dataclass class BuildResult`

```python
@dataclass
class BuildResult:
    build_time_s: float
    extra: dict = field(default_factory=dict)
```

The return value of [`VectorDBAdapter.build_index()`](#101-backendsbasepy):

- `build_time_s` — wall-clock seconds taken to insert all vectors and build
  the index.
- `extra` — an open-ended dict for any backend-specific stats an adapter
  wants to report later (e.g. a server's own reported index size). Empty for
  Phase 0's `faiss_flat`. Uses `field(default_factory=dict)` so every
  `BuildResult` gets its *own* empty dict (see [2.7](#27-dataclasses)).

#### `@dataclass class RunResult`

```python
@dataclass
class RunResult:
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
```

This is **one row of the results table** — everything measured for one
`(backend, build config, search config, k)` combination ("one operating
point"). Every field maps 1:1 to a column in the SQLite `results` table (see
[`storage/schema.py`](#111-storageschemapy)):

| Field | Meaning |
|---|---|
| `run_id` | Random unique ID for this row (so reruns don't collide) |
| `timestamp` | When the run happened (UTC, ISO-8601 text) |
| `lane` | `"A"` or `"B"` (which dataset lane was used) |
| `dataset` | Dataset name, e.g. `"fashion-mnist-784-euclidean"` |
| `backend` | Adapter name, e.g. `"faiss_flat"` |
| `index_type` | Index family, e.g. `"flat"`, `"hnsw"` |
| `metric` | The metric used, as a string (`"l2"`, `"cosine"`, `"ip"`) |
| `dim` | Vector dimensionality |
| `build_params` | The index-build-time parameters used (as a `dict`, stored as JSON text) |
| `search_params` | The search-time parameters used (as a `dict`, stored as JSON text) |
| `k` | The `k` in `recall@k` for *this row* |
| `recall_at_k` | The recall score, 0.0–1.0 |
| `p50_ms` / `p95_ms` / `p99_ms` / `mean_ms` | Latency percentiles/mean across all timed queries, in milliseconds |
| `serial_qps` | `1000 / mean_ms` — single-threaded "queries per second" |
| `build_time_s` | Seconds to build the index |
| `peak_rss_mb` | Memory used by the adapter, in megabytes |
| `index_disk_mb` | On-disk size of the index, in megabytes |
| `n_queries` | How many query vectors were used |
| `n_warmup` | How many warmup queries were run beforehand |
| `n_repeats` | How many times the full query set was repeated for timing |

### 6.2 `core/metrics.py`

Pure math functions — no I/O, no classes, just NumPy. These compute the
scoring/statistics fields of [`RunResult`](#dataclass-class-runresult).

#### `recall_at_k(retrieved_ids, gt_ids, k)`

```python
def recall_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    n_queries = retrieved_ids.shape[0]
    total = 0.0
    for i in range(n_queries):
        retrieved_set = set(retrieved_ids[i, :k].tolist())
        gt_set = set(gt_ids[i, :k].tolist())
        total += len(retrieved_set & gt_set) / k
    return total / n_queries
```

For each query `i` (looping `i` from `0` to `n_queries - 1`):

1. Take that query's top-`k` *retrieved* neighbor IDs and turn them into a
   Python `set` (an unordered collection with no duplicates).
2. Take that query's top-`k` *ground-truth* neighbor IDs, also as a `set`.
3. `retrieved_set & gt_set` — the `&` operator on two sets computes their
   **intersection**: the IDs that appear in *both*. `len(...)` counts how
   many.
4. Divide by `k` — the fraction of the true top-`k` that we actually found —
   and add that to a running `total`.

Finally, `total / n_queries` averages this fraction across all queries. A
perfect (exact) search returns `1.0`; if a search backend only finds half the
true neighbors on average, this returns `0.5`.

This is the literal definition from `PROJECT_PLAN.md` §6:
`recall@k = |retrieved_ids ∩ gt_ids| / k`, averaged over the query set.

#### `latency_percentiles(latencies_ms)`

```python
def latency_percentiles(latencies_ms: np.ndarray | list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }
```

- Input: a list/array of per-query timings, in milliseconds.
- `np.asarray(..., dtype=np.float64)` — convert whatever was passed in into a
  NumPy array of 64-bit floats (so `np.percentile`/`np.mean` work regardless
  of whether the caller passed a Python `list` or an existing array).
- `np.percentile(arr, 50)` — the median: the value below which 50% of
  latencies fall. Similarly `95`/`99` for the "tail latency" — how slow the
  *slowest* 5% / 1% of queries were.
- `np.mean(arr)` — the average.
- Returns a `dict` with the four numbers, keyed exactly as the corresponding
  [`RunResult`](#dataclass-class-runresult) fields (`p50_ms`, `p95_ms`,
  `p99_ms`, `mean_ms`) — this lets the caller do
  `RunResult(..., **latency_percentiles(...))`-style unpacking conveniently
  (though [`runner.py`](#121-harnessrunnerpy) currently spells out each field
  individually for clarity).

#### `serial_qps(mean_ms)`

```python
def serial_qps(mean_ms: float) -> float:
    return 1000.0 / mean_ms if mean_ms > 0 else float("inf")
```

Converts an average latency (in milliseconds) into a throughput number:
"if every query took exactly `mean_ms` milliseconds and we ran them one after
another with zero overhead, how many could we do per second?"
`1000.0 / mean_ms` converts milliseconds to "per second". The
`if mean_ms > 0 else float("inf")` guards against dividing by zero — if the
mean latency were `0`, throughput is treated as infinite rather than crashing
(`float("inf")` is Python's representation of ∞).

As `PROJECT_PLAN.md` notes, this is called **serial QPS** specifically because
it's `1 / latency` for *one query at a time* — not a measurement of true
concurrent throughput (that's a later phase).

---

## 7. `src/vdbbench/config.py`

This file defines the **shape of `config/default.yaml`** as a set of
[Pydantic models](#29-pydantic-models) (classes inheriting from
`pydantic.BaseModel`), plus a function to load and validate that YAML file.

```python
class DatasetConfig(BaseModel):
    name: str
    path: str
    metric: str | None = None
```

- `DatasetConfig` mirrors the `dataset:` block in the YAML. `metric` is
  optional (`str | None = None`) — if the YAML doesn't set it, the loader in
  [`hdf5_ann.py`](#82-datasetshdf5_annpy) falls back to reading the metric
  from the HDF5 file's own metadata.

```python
class BackendConfig(BaseModel):
    name: str
    index_type: str = "flat"
    build_params: dict = Field(default_factory=dict)
    search_params: dict = Field(default_factory=dict)
```

- One entry in the `backends:` list. `name` (e.g. `"faiss_flat"`) is used to
  look up the actual adapter class in
  [`runner.ADAPTERS`](#121-harnessrunnerpy). `build_params`/`search_params`
  are open-ended dicts of index parameters (e.g. HNSW's `M`/`ef_construction`
  in later phases) — empty for Phase 0's exact-search `faiss_flat`.
- `Field(default_factory=dict)` — same "fresh empty dict per instance"
  reasoning as [`BuildResult.extra`](#dataclass-class-buildresult) (see
  [2.7](#27-dataclasses)/[2.9](#29-pydantic-models)).

```python
class ResultsConfig(BaseModel):
    backend: str = "sqlite"
    path: str = "results/results.db"
```

- Where results get written. `backend` is currently always `"sqlite"`
  (`Parquet` is a possible future option per `PROJECT_PLAN.md` §6, not yet
  implemented).

```python
class Config(BaseModel):
    lane: str = "A"
    dataset: DatasetConfig
    k_values: list[int] = Field(default_factory=lambda: [10])
    n_warmup: int = 50
    n_repeats: int = 3
    backends: list[BackendConfig]
    results: ResultsConfig = Field(default_factory=ResultsConfig)
```

The top-level config object — one `Config` instance represents the *entire*
parsed `default.yaml`. Note `dataset: DatasetConfig` and
`backends: list[BackendConfig]` — Pydantic recursively validates *nested*
models too, so the `dataset:` block in the YAML becomes a real
`DatasetConfig` object, and each entry under `backends:` becomes a
`BackendConfig` object.

`Field(default_factory=lambda: [10])` — a `lambda` is an anonymous,
inline function; `lambda: [10]` is a zero-argument function that returns the
list `[10]`. So if `k_values` is omitted from the YAML, each `Config` gets its
own fresh `[10]` list (rather than one `[10]` list shared by every `Config`
instance ever created).

```python
def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
```

- `open(path)` opens the YAML file (the `with` block — see
  [2.12](#212-the-with-statement-context-managers) — closes it automatically).
- `yaml.safe_load(f)` parses the YAML text into plain Python data structures —
  nested `dict`s and `list`s. `raw` at this point is just a `dict` like
  `{"lane": "A", "dataset": {"name": ..., ...}, "backends": [...], ...}`,
  with **no validation or type-checking yet**.
- `Config(**raw)` — see [2.14](#214--dict-unpacking): unpacks the dict into
  keyword arguments for `Config`'s constructor. **This is the validation
  step** — Pydantic checks every field against the type hints declared above
  and raises a descriptive error if, say, `n_repeats` were a string instead
  of an integer, or `dataset` were missing entirely.

---

## 8. The `datasets` package

"Datasets" are sources of vectors-to-search and vectors-to-search-for. The
package defines a generic interface (`base.py`) and one concrete
implementation for "Lane A" (`hdf5_ann.py`).

### 8.1 `datasets/base.py`

```python
class DatasetSource(ABC):
    @abstractmethod
    def load(self) -> Dataset:
        """Returns Dataset(train_vectors, test_vectors, dim, metric, gt_neighbors)."""
```

An [ABC](#28-abstract-base-classes-abc-and-abstractmethod) with one required
method: `load()`, which must return a [`Dataset`](#dataclass-class-dataset). This is the
"contract" that any dataset source — the current HDF5-based Lane A loader, or
a future Lane B "raw text corpus + bge-m3 embeddings" loader — must satisfy.
Because the rest of the harness only calls `.load()` and reads the resulting
`Dataset` fields, it doesn't need to know or care *how* the vectors were
obtained.

### 8.2 `datasets/hdf5_ann.py`

Implements `DatasetSource` for **Lane A**: "precomputed ann-benchmarks-format
HDF5 file". [ann-benchmarks](http://ann-benchmarks.com/) is a well-known
benchmark suite that distributes datasets as `.hdf5` files containing:

- a `train` array (the corpus),
- a `test` array (the queries),
- a `neighbors` array (precomputed ground-truth neighbor IDs for each query),
- and a `distance` attribute (a string like `"euclidean"` or `"angular"`
  describing the metric the ground truth was computed with).

```python
_METRIC_MAP = {
    "euclidean": Metric.L2,
    "l2": Metric.L2,
    "angular": Metric.COSINE,
    "cosine": Metric.COSINE,
    "dot": Metric.INNER_PRODUCT,
    "ip": Metric.INNER_PRODUCT,
}
```

A module-level dictionary (created once when the file is first imported)
translating the various strings ann-benchmarks/config might use into the
internal [`Metric`](#class-metricstr-enum) enum. `"angular"` (ann-benchmarks'
term for cosine-distance datasets) maps to `Metric.COSINE`.

```python
class HDF5AnnDataset(DatasetSource):
    def __init__(self, path: str | Path, metric: str | None = None):
        self.path = Path(path)
        self.metric_override = metric
```

- Inherits from `DatasetSource`, so it must implement `load()` (and does,
  below) — this is what makes it a valid, instantiable
  [`DatasetSource`](#81-datasetsbasepy).
- The constructor just stores the file path (converted to a `pathlib.Path`
  for convenient path operations) and an optional metric override string
  (from `config.dataset.metric`).

```python
    def load(self) -> Dataset:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.path}. "
                "Run scripts/download_dataset.py to fetch the dev dataset."
            )
```

If the `.hdf5` file isn't present on disk, immediately fail with a helpful
message pointing at [`scripts/download_dataset.py`](#13-scriptsdownload_datasetpy)
rather than letting `h5py` raise a more cryptic error later.

```python
        with h5py.File(self.path, "r") as f:
            train = f["train"][:].astype(np.float32)
            test = f["test"][:].astype(np.float32)
            neighbors = f["neighbors"][:].astype(np.int64) if "neighbors" in f else None
            metric_name = self.metric_override or f.attrs.get("distance")
```

- `h5py.File(path, "r")` opens the HDF5 file for **r**eading; the `with`
  block closes it automatically afterward.
- `f["train"]` accesses the dataset named `"train"` inside the file (HDF5
  files are like a mini filesystem of named arrays). `[:]` reads the *entire*
  array into memory as a NumPy array. `.astype(np.float32)` ensures it's
  32-bit floats (the precision FAISS expects).
- `neighbors = ... if "neighbors" in f else None` — a
  [conditional expression](#213-conditional-ifelse-expressions-and-or): only
  load the `"neighbors"` array if the file actually contains one (Lane B
  datasets built later might not).
- `metric_name = self.metric_override or f.attrs.get("distance")` — use the
  config's explicit `metric:` override if given; otherwise read the
  HDF5 file's own `distance` attribute (`f.attrs` is the file's metadata
  dictionary).

```python
        if metric_name is None:
            raise ValueError(
                f"Metric not specified and not found in {self.path} attrs; "
                "pass `metric` explicitly via config."
            )
        metric = _METRIC_MAP.get(str(metric_name).lower())
        if metric is None:
            raise ValueError(f"Unknown metric '{metric_name}' in {self.path}")
```

Two validation checks:
1. If *neither* the config nor the file specifies a metric, fail with a clear
   message.
2. `_METRIC_MAP.get(...)` looks up the (lowercased) metric string in the
   translation table above; `dict.get()` returns `None` if the key isn't
   found (instead of raising an error), which lets us produce a friendlier
   "Unknown metric '...'" message for typos/unsupported metrics.

```python
        return Dataset(
            train_vectors=train,
            test_vectors=test,
            dim=train.shape[1],
            metric=metric,
            gt_neighbors=neighbors,
        )
```

Finally, package everything into a [`Dataset`](#dataclass-class-dataset). `dim=train.shape[1]`
— `train.shape` is `(n_corpus, dim)`, so `[1]` is the vector dimensionality
(e.g. `784`).

---

## 9. The `groundtruth` package

"Ground truth" = the *correct* answer to "what are this query's true nearest
neighbors?" — used to score how good (or bad) an approximate search backend
is.

### 9.1 `groundtruth/base.py`

```python
class GroundTruthSource(ABC):
    @abstractmethod
    def neighbors(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return exact GT neighbor ids, shape (n_queries, k)."""
```

Another [ABC](#28-abstract-base-classes-abc-and-abstractmethod): any ground
truth source must provide `neighbors(queries, k)` returning an array of
shape `(n_queries, k)`. Currently there's one implementation
([`PrecomputedGroundTruth`](#92-groundtruthprecomputedpy)) that ignores the
`queries` argument because the answers were already computed by the dataset's
authors; a future Lane B implementation
(`groundtruth/bruteforce.py`, per `PROJECT_PLAN.md`) will *use* `queries` to
run a real brute-force FAISS search and compute ground truth on the fly.
Defining the interface with `queries` as a parameter now means both
implementations can be swapped in without changing any calling code.

### 9.2 `groundtruth/precomputed.py`

```python
class PrecomputedGroundTruth(GroundTruthSource):
    def __init__(self, neighbors: np.ndarray):
        self._neighbors = neighbors

    def neighbors(self, queries: np.ndarray, k: int) -> np.ndarray:
        if k > self._neighbors.shape[1]:
            raise ValueError(
                f"Requested k={k} exceeds precomputed neighbors width "
                f"{self._neighbors.shape[1]}"
            )
        return self._neighbors[:, :k]
```

- Constructor: stores the `gt_neighbors` array that came straight from the
  HDF5 file (via [`Dataset.gt_neighbors`](#dataclass-class-dataset)) as `self._neighbors`
  (shape `(n_queries, k_gt)`, where `k_gt` is however many neighbors the
  dataset authors precomputed — typically 100).
- `neighbors(queries, k)`:
  - First checks that the requested `k` isn't larger than what's available
    (`self._neighbors.shape[1]` = `k_gt`) — you can't ask for "top-100" if
    only the top-10 were precomputed.
  - `self._neighbors[:, :k]` — [NumPy slicing](#210-numpy-arrays): "all rows,
    first `k` columns" — i.e., truncate each query's precomputed neighbor
    list down to the requested `k`.
  - The `queries` parameter is unused here (required only to satisfy the
    [`GroundTruthSource`](#91-groundtruthbasepy) interface) — precomputed
    ground truth doesn't need to look at the actual query vectors, just slice
    the existing answer table.

---

## 10. The `backends` package

This is **the central abstraction of the whole project** (per
`PROJECT_PLAN.md` §5): a uniform interface that every vector database — FAISS,
Qdrant, pgvector, Milvus — implements, so the harness can run identical
benchmarks against any of them.

### 10.1 `backends/base.py`

```python
class VectorDBAdapter(ABC):
    name: str

    @abstractmethod
    def setup(self, dim: int, metric: Metric, build_params: dict) -> None: ...

    @abstractmethod
    def build_index(self, vectors: np.ndarray, ids: np.ndarray) -> BuildResult: ...

    @abstractmethod
    def set_search_params(self, search_params: dict) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[int]: ...

    @abstractmethod
    def index_size_bytes(self) -> int: ...

    @abstractmethod
    def memory_rss_bytes(self) -> int: ...

    @abstractmethod
    def teardown(self) -> None: ...
```

`name: str` is a **class-level type annotation with no value** — it documents
"every subclass must define a `name` attribute" (a short identifier like
`"faiss_flat"`) without giving it a default. Each of the seven methods is
[`@abstractmethod`](#28-abstract-base-classes-abc-and-abstractmethod), so any
concrete adapter class — [`FaissFlatAdapter`](#102-backendsfaiss_backendpy) today,
Qdrant/pgvector/Milvus adapters later — *must* implement all seven, or Python
refuses to let you create it. What each method means, in the order the
harness calls them:

| Method | When called | Purpose |
|---|---|---|
| `setup(dim, metric, build_params)` | once, before building | Create the collection/table/index structure with the given dimensionality, metric, and build-time parameters (e.g. HNSW's `M`/`ef_construction`) — but don't insert data yet |
| `build_index(vectors, ids)` | once, after `setup` | Insert all corpus vectors (with their IDs) and build the index; return a [`BuildResult`](#dataclass-class-buildresult) with the time taken |
| `set_search_params(search_params)` | once, after building (and again between sweep points, in later phases) | Adjust search-time parameters (e.g. `ef_search`, `nprobe`) **without rebuilding** the index |
| `search(query, k)` | once **per query**, the *only timed operation* | Run a single nearest-neighbor search for one query vector and return the top-`k` neighbor IDs as a plain `list[int]` |
| `index_size_bytes()` | after building, to record `index_disk_mb` | Report how large the built index is on disk |
| `memory_rss_bytes()` | after building, to record `peak_rss_mb` | Report how much memory the adapter is using (see [10.2](#102-backendsfaiss_backendpy) for how this is computed for in-process libraries vs. server processes) |
| `teardown()` | once, at the end | Release resources / drop the collection so the next adapter/config starts clean |

Because `search()` takes one query and returns one answer, and is the *only*
method the harness times, the interface is explicitly designed for "single
query at a time, serially" (per `PROJECT_PLAN.md` §1) — a future concurrent
benchmark runner could call `search()` from multiple threads without changing
any adapter.

### 10.2 `backends/faiss_backend.py`

The first (and so far only) concrete adapter:
[`FaissFlatAdapter`](#class-faissflatadaptervectordbadapter), implementing
[`VectorDBAdapter`](#101-backendsbasepy) using FAISS's `IndexFlat` — an
**exact, brute-force** index (compares the query against *every* corpus
vector). This is intentionally the "ground truth reference" adapter: with an
exact index, `recall@k` should always be `1.0`.

```python
_METRIC_MAP = {
    Metric.L2: faiss.METRIC_L2,
    Metric.INNER_PRODUCT: faiss.METRIC_INNER_PRODUCT,
    Metric.COSINE: faiss.METRIC_INNER_PRODUCT,
}
```

Translates `vdbbench`'s [`Metric`](#class-metricstr-enum) enum into FAISS's
own metric constants (`faiss.METRIC_L2`, `faiss.METRIC_INNER_PRODUCT`).
`Metric.COSINE` also maps to `faiss.METRIC_INNER_PRODUCT` because FAISS has no
separate "cosine" mode — cosine similarity is computed as an inner product
*after* vectors have been L2-normalized (a normalization step planned for
`core/normalize.py` in a later phase; not needed for Phase 0's
Euclidean fashion-mnist dataset).

#### `class FaissFlatAdapter(VectorDBAdapter)`

```python
class FaissFlatAdapter(VectorDBAdapter):
    name = "faiss_flat"

    def __init__(self):
        self.index: faiss.Index | None = None
        self.dim: int | None = None
        self.metric: Metric | None = None
        self._rss_before: int = 0
```

- `name = "faiss_flat"` — the class-level attribute required by
  [`VectorDBAdapter`](#101-backendsbasepy); this string ends up in the
  `backend` column of every [`RunResult`](#dataclass-class-runresult) row.
- The constructor initializes everything to "empty"/`None`. `self.index`
  will hold the actual FAISS index object once `setup()` runs.
  `self._rss_before` will record the process's memory usage *before* the
  index is built, so `memory_rss_bytes()` can later report the *increase*.

```python
    def setup(self, dim: int, metric: Metric, build_params: dict) -> None:
        self.dim = dim
        self.metric = metric
        self._rss_before = psutil.Process(os.getpid()).memory_info().rss
        self.index = faiss.IndexFlat(dim, _METRIC_MAP[metric])
```

- Stores `dim`/`metric` for later use (`build_params` is unused — `IndexFlat`
  has no build-time parameters to configure).
- `os.getpid()` gets the current process's ID; `psutil.Process(pid)` wraps it
  for inspection; `.memory_info().rss` is the process's current **Resident Set
  Size** — how much physical RAM it's actually using right now, in bytes.
  This snapshot, taken *before* any vectors are loaded, is the "before"
  baseline.
- `faiss.IndexFlat(dim, _METRIC_MAP[metric])` creates the actual exact-search
  index: a structure that will hold `dim`-dimensional vectors and compare new
  queries against all of them using the given metric.

```python
    def build_index(self, vectors: np.ndarray, ids: np.ndarray) -> BuildResult:
        # IndexFlat has no add_with_ids; it assigns positional ids 0..n-1, which
        # is exactly the ann-benchmarks ground-truth id space.
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        start = time.perf_counter()
        self.index.add(vectors)
        build_time_s = time.perf_counter() - start
        return BuildResult(build_time_s=build_time_s)
```

- `np.ascontiguousarray(vectors, dtype=np.float32)` — ensures the array is
  laid out in memory the way FAISS (a C++ library under the hood) requires:
  contiguous (no gaps/strides) and 32-bit floats. `h5py` arrays are usually
  already like this, but this is a defensive guarantee.
- `time.perf_counter()` — a high-resolution clock; called once before and
  once after `self.index.add(vectors)` (which inserts every corpus vector and
  builds the index), and the difference is the build time in seconds.
- The `ids` parameter is accepted (to satisfy the
  [`VectorDBAdapter`](#101-backendsbasepy) interface) but not directly used:
  FAISS's plain `IndexFlat.add()` always assigns vectors positions `0, 1, 2,
  ...` in insertion order, and `search()` returns those positions as the
  "IDs". Since `runner.py` always passes `ids = np.arange(n)` (i.e. exactly
  `0..n-1`, matching ann-benchmarks' own ID convention — see
  [12.1](#121-harnessrunnerpy)), FAISS's positional indices line up perfectly
  with the ground-truth ID space without any extra mapping.
- Returns a [`BuildResult`](#dataclass-class-buildresult) with just the
  timing (`extra` stays as its default empty dict).

```python
    def set_search_params(self, search_params: dict) -> None:
        pass  # Flat index has no search-time parameters.
```

`pass` is Python's "do nothing" statement — required because a function body
can't be empty. `IndexFlat` has no tunable search parameters (unlike HNSW's
`ef_search`), so this method is a no-op but must still exist to satisfy the
interface.

```python
    def search(self, query: np.ndarray, k: int) -> list[int]:
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        _, ids = self.index.search(query, k)
        return ids[0].tolist()
```

This is **the one timed operation** in the whole benchmark.

- `query.reshape(1, -1)` — FAISS's `search()` expects a *batch* of queries
  (a 2D array, shape `(n, dim)`), even for a single query. `.reshape(1, -1)`
  turns a 1D vector of length `dim` into a 2D array of shape `(1, dim)` — `-1`
  means "figure out this dimension automatically" (here, it'll be `dim`).
- `self.index.search(query, k)` returns a tuple `(distances, ids)`, each of
  shape `(1, k)` (one row per query — here just one). `_` is a conventional
  name for "a value we're receiving but intentionally not using" — we don't
  need the distances, only the IDs.
- `ids[0]` — the first (only) row, shape `(k,)`. `.tolist()` converts it from
  a NumPy array to a plain Python `list[int]`, matching the
  [`VectorDBAdapter.search`](#101-backendsbasepy) return type.

```python
    def index_size_bytes(self) -> int:
        return faiss.serialize_index(self.index).nbytes
```

`faiss.serialize_index(self.index)` converts the in-memory FAISS index into
the same byte representation `faiss.write_index()` would write to disk
(returned as a NumPy byte array); `.nbytes` is the length of that array in
bytes — i.e., exactly how big the index *would be* on disk, without actually
writing a file. (This same approach will work unchanged for HNSW/IVF indexes
in later phases.)

```python
    def memory_rss_bytes(self) -> int:
        rss_now = psutil.Process(os.getpid()).memory_info().rss
        return max(rss_now - self._rss_before, 0)
```

Takes a *new* RSS measurement and subtracts the "before" baseline captured in
`setup()` — i.e., "how much additional memory has this process consumed since
right before we started building the index?" `max(..., 0)` guards against a
(possible but unlikely) negative number if memory usage happened to *decrease*
in the meantime (e.g. due to garbage collection), since a negative "memory
used" wouldn't make sense.

This is the "in-process RSS delta" approach `PROJECT_PLAN.md` §2 specifies for
library-based backends like FAISS — as opposed to server-based backends
(Qdrant/Postgres/Milvus in later phases), which will instead measure the RSS
of their *server* process(es) via `psutil`.

```python
    def teardown(self) -> None:
        self.index = None
```

Drops the reference to the FAISS index object. Once nothing else refers to
it, Python's garbage collector frees the memory it used (including the
underlying C++ memory FAISS allocated). This prepares the adapter for a clean
re-`setup()` with a different configuration in a future sweep.

---

## 11. The `storage` package

Persists [`RunResult`](#dataclass-class-runresult) rows to a SQLite database
file so results accumulate across runs and can be queried/reported on later
(`PROJECT_PLAN.md` §6, §9).

### 11.1 `storage/schema.py`

```python
RESULTS_TABLE = "results"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    ...
    n_repeats INTEGER NOT NULL
)
"""
```

- `RESULTS_TABLE = "results"` — the table name, defined once and reused (by
  both `schema.py` and `store.py`) so it can't accidentally drift out of sync
  if renamed.
- `CREATE_TABLE_SQL` is an **f-string** (see [2.11](#211-f-strings)) containing
  a SQL `CREATE TABLE IF NOT EXISTS` statement. `{RESULTS_TABLE}` is
  substituted with `results`. `IF NOT EXISTS` means running this repeatedly
  (e.g., every time the program starts) is safe — it won't error or wipe
  existing data if the table is already there.
- The columns are a 1:1 mirror of [`RunResult`](#dataclass-class-runresult)'s
  fields. `run_id TEXT PRIMARY KEY` makes `run_id` the table's unique
  identifier (and since [`runner.py`](#121-harnessrunnerpy) generates a fresh
  random UUID per row, every insert is guaranteed unique — reruns *append*,
  they never overwrite). `build_params`/`search_params` are stored as `TEXT`
  because SQLite has no native "dictionary"/JSON column type — `store.py`
  converts the Python `dict`s to JSON text before inserting (see below).

### 11.2 `storage/store.py`

```python
class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(CREATE_TABLE_SQL)
        self._conn.commit()
```

- `self.path.parent.mkdir(parents=True, exist_ok=True)` — `self.path.parent`
  is the *containing directory* of the database file (e.g. `results/` for
  `results/results.db`). `.mkdir(parents=True, exist_ok=True)` creates that
  directory (and any missing parent directories) if it doesn't exist yet, and
  does nothing (rather than erroring) if it already does.
- `sqlite3.connect(self.path)` — opens (creating if necessary) the SQLite
  database file and returns a connection object, stored as `self._conn`.
- `self._conn.execute(CREATE_TABLE_SQL)` — runs the `CREATE TABLE IF NOT
  EXISTS` statement from [`schema.py`](#111-storageschemapy), ensuring the
  `results` table exists.
- `.commit()` — SQLite groups changes into transactions; `commit()` makes the
  change permanent on disk.

```python
    def save(self, result: RunResult) -> None:
        row = dataclasses.asdict(result)
        row["build_params"] = json.dumps(row["build_params"])
        row["search_params"] = json.dumps(row["search_params"])

        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        self._conn.execute(
            f"INSERT INTO {RESULTS_TABLE} ({cols}) VALUES ({placeholders})",
            row,
        )
        self._conn.commit()
```

Step by step, for a single [`RunResult`](#dataclass-class-runresult) object:

1. `dataclasses.asdict(result)` — converts the dataclass instance into a
   plain Python `dict` mapping field names to values, e.g.
   `{"run_id": "...", "timestamp": "...", ..., "build_params": {}, ...}`.
2. `row["build_params"] = json.dumps(row["build_params"])` —
   `build_params`/`search_params` are still Python `dict`s (e.g. `{}` or
   `{"ef_search": 64}`); `json.dumps(...)` converts each into its JSON text
   representation (e.g. `"{}"` or `'{"ef_search": 64}'`) so they can be stored
   in a `TEXT` column.
3. `cols = ", ".join(row.keys())` — joins all the dict's keys (column names)
   into a comma-separated string: `"run_id, timestamp, lane, ..."`.
4. `placeholders = ", ".join(f":{k}" for k in row.keys())` — this is a
   **generator expression**: `f":{k}" for k in row.keys()` produces
   `":run_id"`, `":timestamp"`, etc. (one per key), and `", ".join(...)`
   combines them into `":run_id, :timestamp, :lane, ..."`. These `:name`
   placeholders are SQLite's **named parameters** — a safe way to insert
   values into SQL without manually building a string (which would risk SQL
   injection bugs).
5. `self._conn.execute(f"INSERT INTO {RESULTS_TABLE} ({cols}) VALUES
   ({placeholders})", row)` — runs e.g.
   `INSERT INTO results (run_id, timestamp, ...) VALUES (:run_id, :timestamp, ...)`,
   and `row` (the dict) supplies the actual values for each `:name`
   placeholder — `sqlite3` matches dict keys to placeholder names
   automatically.
6. `.commit()` — saves the new row to disk.

```python
    def close(self) -> None:
        self._conn.close()
```

Closes the database connection/file handle — called once at the very end of a
run (in [`runner.main()`](#121-harnessrunnerpy)).

---

## 12. The `harness` package

The **orchestrator**: ties together config, datasets, ground truth, adapters,
metrics, and storage into one runnable program. This is also the
[`vdbbench-run`](#41-pyprojecttoml) command-line entry point.

### 12.1 `harness/runner.py`

```python
ADAPTERS: dict[str, type[VectorDBAdapter]] = {
    "faiss_flat": FaissFlatAdapter,
}
```

A **registry**: maps the `name` string from a config's `backends:` entry
(e.g. `"faiss_flat"`) to the *adapter class itself*. The type hint
`dict[str, type[VectorDBAdapter]]` reads as: "a dict from strings to *classes
that are `VectorDBAdapter` subclasses*" (not instances — see
[2.4](#24-type-hints)). `ADAPTERS["faiss_flat"]` evaluates to the
`FaissFlatAdapter` class; `ADAPTERS["faiss_flat"]()` (note the `()`) creates
("instantiates") an *object* of that class. As later phases add
`QdrantAdapter`, `PgVectorAdapter`, etc., they get one more line each here —
the rest of the code below never needs to change.

#### `run_backend(config, backend_cfg)`

This function runs **one full benchmark** for one entry in `config.backends`
and returns a list of [`RunResult`](#dataclass-class-runresult)s (one per
`k` in `config.k_values`).

```python
    dataset = HDF5AnnDataset(config.dataset.path, metric=config.dataset.metric).load()
    if dataset.gt_neighbors is None:
        raise ValueError(f"Dataset {config.dataset.name} has no ground-truth neighbors")
    gt_source = PrecomputedGroundTruth(dataset.gt_neighbors)
```

- Creates an [`HDF5AnnDataset`](#82-datasetshdf5_annpy) for the configured
  path/metric and immediately calls `.load()` to get a
  [`Dataset`](#dataclass-class-dataset) object — `dataset.train_vectors`,
  `dataset.test_vectors`, `dataset.dim`, `dataset.metric`,
  `dataset.gt_neighbors`.
- If the dataset has no precomputed ground truth, Lane A can't proceed (Lane
  B's brute-force ground truth is a later phase) — fail clearly.
- Wraps the ground-truth array in [`PrecomputedGroundTruth`](#92-groundtruthprecomputedpy),
  giving us a `gt_source.neighbors(queries, k)` we can call later.

```python
    adapter = ADAPTERS[backend_cfg.name]()
    adapter.setup(dataset.dim, dataset.metric, backend_cfg.build_params)

    ids = np.arange(dataset.train_vectors.shape[0], dtype=np.int64)
    build_result = adapter.build_index(dataset.train_vectors, ids)
    adapter.set_search_params(backend_cfg.search_params)
```

- Looks up and instantiates the adapter class (e.g. `FaissFlatAdapter()`),
  then calls the three [`VectorDBAdapter`](#101-backendsbasepy) setup steps in
  order, exactly as the interface prescribes:
  1. `setup(dim, metric, build_params)` — configure, don't insert yet.
  2. `build_index(train_vectors, ids)` — insert everything and build the
     index; `ids = np.arange(n)` is `[0, 1, 2, ..., n-1]`, matching the
     ann-benchmarks ID convention (see the comment in
     [`faiss_backend.build_index`](#102-backendsfaiss_backendpy)).
  3. `set_search_params(search_params)` — apply any search-time tuning (a
     no-op for `faiss_flat`, meaningful for HNSW's `ef_search` later).

```python
    k_max = max(config.k_values)
    n_queries = dataset.test_vectors.shape[0]

    # Warmup: discard timings.
    for i in range(config.n_warmup):
        adapter.search(dataset.test_vectors[i % n_queries], k_max)
```

- `k_max = max(config.k_values)` — since one `search()` call returns the
  top-`k` for *whatever* `k` you ask, and we ultimately want
  `recall_at_k` for every value in `config.k_values` (currently just
  `[10]`), we run each search once asking for the *largest* needed `k`
  (`k_max`), and compute smaller-`k` recall by slicing the result afterward
  (see further below). This means each query is only actually searched once
  per repeat, regardless of how many `k` values are configured.
- The **warmup loop**: runs `config.n_warmup` (default 50) searches and
  throws away the results/timings entirely. This is standard benchmarking
  practice — it lets caches, memory allocators, etc. reach a "steady state"
  before real measurements begin (`PROJECT_PLAN.md` §7 step 3a).
  `dataset.test_vectors[i % n_queries]` — `%` is the remainder/modulo
  operator; `i % n_queries` cycles back to `0` once `i` reaches
  `n_queries`, so warmup safely works even if `n_warmup > n_queries`.

```python
    # Measure: time only adapter.search().
    latencies_ms = np.empty(config.n_repeats * n_queries, dtype=np.float64)
    retrieved = np.empty((n_queries, k_max), dtype=np.int64)
    idx = 0
    for rep in range(config.n_repeats):
        for i in range(n_queries):
            start = time.perf_counter()
            result_ids = adapter.search(dataset.test_vectors[i], k_max)
            latencies_ms[idx] = (time.perf_counter() - start) * 1000.0
            idx += 1
            if rep == 0:
                retrieved[i] = result_ids
```

The core measurement loop:

- `np.empty(shape, dtype=...)` pre-allocates an array of the given shape
  *without* initializing its contents (faster than starting from zeros, since
  every slot will be overwritten anyway).
  - `latencies_ms` — one slot per `(repeat, query)` pair —
    `config.n_repeats * n_queries` total — to hold every individual timing.
  - `retrieved` — shape `(n_queries, k_max)` — holds the neighbor IDs returned
    for each query (only needs to be recorded once, not once per repeat,
    since a deterministic exact index returns the same answer every time).
- **Nested loop**: the outer loop runs `config.n_repeats` (default 3) full
  passes over the query set; the inner loop iterates every query.
- For each query: record `time.perf_counter()` immediately before and after
  the *single* call `adapter.search(dataset.test_vectors[i], k_max)` — this
  is the **only timed operation**, matching `PROJECT_PLAN.md`'s "only
  `adapter.search()` is timed" requirement. The difference, multiplied by
  `1000.0`, converts seconds to milliseconds and is stored in
  `latencies_ms[idx]`.
- `if rep == 0: retrieved[i] = result_ids` — only on the *first* repeat, save
  the returned neighbor IDs for later recall computation (repeats exist purely
  to get stable latency statistics; the search results themselves don't
  change between repeats for a deterministic index).

```python
    perc = latency_percentiles(latencies_ms)
    qps = serial_qps(perc["mean_ms"])
    gt = gt_source.neighbors(dataset.test_vectors, k_max)
    build_time_s = build_result.build_time_s
    peak_rss_mb = adapter.memory_rss_bytes() / (1024 * 1024)
    index_disk_mb = adapter.index_size_bytes() / (1024 * 1024)
```

After the loop, compute everything that's the *same* for every
[`RunResult`](#dataclass-class-runresult) row this `run_backend` call will
produce:

- `perc` — `{"p50_ms": ..., "p95_ms": ..., "p99_ms": ..., "mean_ms": ...}`
  from [`latency_percentiles`](#latency_percentileslatencies_ms) over *all*
  `n_repeats * n_queries` timings.
- `qps` — [`serial_qps`](#serial_qpsmean_ms) derived from the mean latency.
- `gt` — the ground-truth neighbor IDs for all queries, truncated to `k_max`
  columns, via [`PrecomputedGroundTruth.neighbors`](#92-groundtruthprecomputedpy).
- `peak_rss_mb` / `index_disk_mb` — convert the adapter's byte-based
  measurements ([`memory_rss_bytes()`](#102-backendsfaiss_backendpy),
  [`index_size_bytes()`](#102-backendsfaiss_backendpy)) into megabytes by dividing by
  `1024 * 1024`.

```python
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
```

- `timestamp` — one shared UTC timestamp (as ISO-8601 text, e.g.
  `"2026-06-12T06:38:17.179216+00:00"`) for every row produced by this call.
- For **each** `k` in `config.k_values` (just `[10]` in Phase 0, but this loop
  is what lets later phases set `k_values: [1, 10, 100]` and get three rows
  for the price of one search pass):
  - `run_id=uuid.uuid4().hex` — a fresh random unique ID per row (so the
    `run_id TEXT PRIMARY KEY` constraint in [`schema.py`](#111-storageschemapy)
    is always satisfied, even across repeated runs).
  - `recall_at_k=recall_at_k(retrieved, gt, k)` — calls the
    [`core/metrics.recall_at_k`](#recall_at_kretrieved_ids-gt_ids-k) function,
    slicing both `retrieved` and `gt` down to this row's `k` (recall, this
    function itself does `[:, :k]` internally — see
    [6.2](#62-coremetricspy)).
  - All the other fields are either copied straight from `config`/
    `backend_cfg`/`dataset`/`adapter`, or are the shared `perc`/`qps`/
    `build_time_s`/`peak_rss_mb`/`index_disk_mb` values computed once above.
  - `dataset.metric.value` — recall from [2.4](#24-type-hints)/
    [6.1](#61-coretypespy) that `Metric` is a `str` `Enum`; `.value` gets the
    plain string (`"l2"`) to store in the `metric TEXT` column.

```python
    adapter.teardown()
    return results
```

Releases the adapter's resources (the last
[`VectorDBAdapter`](#101-backendsbasepy) lifecycle step) and returns the list
of `RunResult`s to the caller.

#### `main()`

The actual command-line entry point — what runs when you type `vdbbench-run`.

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vdbbench harness")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
```

`argparse` is Python's standard library for parsing command-line flags. This
sets up one optional flag, `--config`, defaulting to `"config/default.yaml"`.
Running `vdbbench-run --config config/my_other.yaml` would set
`args.config` to that path instead.

```python
    config = load_config(args.config)
    store = SQLiteStore(config.results.path)
```

Loads and validates the config file (see
[`config.load_config`](#7-srcvdbbenchconfigpy)), then opens (creating if
necessary) the SQLite results database at the path it specifies.

```python
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
```

- **Outer loop**: for every backend listed in `config.backends` (just one,
  `faiss_flat`, in Phase 0; later phases will list several).
- **Inner loop**: for every [`RunResult`](#dataclass-class-runresult) that
  [`run_backend`](#run_backendconfig-backend_cfg) produces (one per `k` in
  `config.k_values`):
  - `store.save(result)` — write it to SQLite immediately (so partial
    progress isn't lost if a later backend crashes).
  - Print a human-readable one-line summary to the terminal. The `:.4f`,
    `:.3f`, `:.1f`, `:.2f` are [f-string](#211-f-strings) format specifiers —
    "format this number with N digits after the decimal point".
- Finally, `store.close()` closes the database connection.

```python
if __name__ == "__main__":
    main()
```

The [standard idiom](#216-if-__name__--__main__) — lets you also run this
file directly via `python -m vdbbench.harness.runner` or
`python src/vdbbench/harness/runner.py`, in addition to the installed
`vdbbench-run` command (which calls `main()` directly, per
`pyproject.toml`'s `[project.scripts]`).

---

## 13. `scripts/download_dataset.py`

A small standalone utility (not part of the `vdbbench` package — it's run
directly with `python`, not imported) that downloads an ann-benchmarks
`.hdf5` dataset file into `data/`.

```python
BASE_URL = "http://ann-benchmarks.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
```

- `BASE_URL` — where ann-benchmarks hosts its dataset files.
- `__file__` is a special variable equal to the path of the *current* script.
  `.resolve()` makes it absolute; `.parent.parent` goes up two directories
  (from `scripts/download_dataset.py` → `scripts/` → repo root); `/ "data"`
  ([`Path`](#217-other-standard-library-pieces-used)'s `/` operator joins path
  components) gives the repo's `data/` directory regardless of where the
  script is run from.

```python
def download(name: str) -> Path:
    url = f"{BASE_URL}/{name}.hdf5"
    dest = DATA_DIR / f"{name}.hdf5"
    if dest.exists():
        print(f"Already present: {dest}")
        return dest
```

Builds the full download URL (e.g.
`http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5`) and the local
destination path. If the file's already there, skip downloading entirely.

```python
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest
```

- `requests.get(url, stream=True, ...)` — starts an HTTP download.
  `stream=True` means "don't load the whole response into memory at once" —
  important since these files are ~100-200+ MB.
- `resp.raise_for_status()` — raises an error if the server responded with an
  error status (e.g. 404 Not Found), instead of silently saving an error page
  as if it were the dataset.
- `total = int(resp.headers.get("content-length", 0))` — reads the expected
  file size from the response headers (for the progress bar); `0` if the
  server doesn't report it.
- `open(dest, "wb")` — opens the destination file for **w**riting in
  **b**inary mode. Note this `with` statement has **two** context managers
  separated by a comma — both `f` (the file) and `bar` (the progress bar) are
  set up together and both get cleaned up automatically at the end of the
  block.
- `tqdm(total=total, unit="B", unit_scale=True)` — `tqdm` is a library that
  renders a progress bar; `unit="B"`/`unit_scale=True` makes it display sizes
  like "150MB" instead of raw byte counts.
- `for chunk in resp.iter_content(chunk_size=1 << 20):` — downloads the file
  in 1 MiB chunks (`1 << 20` is bit-shift notation for `2^20 = 1,048,576`
  bytes). For each chunk: write it to the file (`f.write(chunk)`) and advance
  the progress bar by that many bytes (`bar.update(len(chunk))`).

```python
if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "fashion-mnist-784-euclidean"
    download(dataset_name)
```

- `sys.argv` is the list of command-line arguments; `sys.argv[0]` is always
  the script name itself, so `sys.argv[1]` is the *first argument the user
  typed* (e.g. `python download_dataset.py sift-128-euclidean` →
  `sys.argv[1] == "sift-128-euclidean"`).
- The [conditional expression](#213-conditional-ifelse-expressions-and-or)
  picks that argument if one was given, otherwise defaults to
  `"fashion-mnist-784-euclidean"` (the dev/smoke dataset).

---

## 14. End-to-end walkthrough

Putting it all together — here's exactly what happens when you run
`vdbbench-run --config config/default.yaml`:

1. **`main()`** ([runner.py](#121-harnessrunnerpy)) parses `--config`, calls
   [`load_config`](#7-srcvdbbenchconfigpy), which reads `config/default.yaml`
   and returns a validated [`Config`](#7-srcvdbbenchconfigpy) object.
2. `main()` opens [`SQLiteStore("results/results.db")`](#112-storagestorepy),
   creating the `results` table if it doesn't exist
   ([schema.py](#111-storageschemapy)).
3. For the one entry in `config.backends` (`faiss_flat`), `main()` calls
   [`run_backend(config, backend_cfg)`](#run_backendconfig-backend_cfg):
   - [`HDF5AnnDataset(...).load()`](#82-datasetshdf5_annpy) opens
     `data/fashion-mnist-784-euclidean.hdf5`, reads `train` (60,000×784),
     `test` (10,000×784), and `neighbors`, determines the metric is
     `Metric.L2`, and returns a [`Dataset`](#dataclass-class-dataset).
   - [`PrecomputedGroundTruth(dataset.gt_neighbors)`](#92-groundtruthprecomputedpy)
     wraps the ground-truth array.
   - `ADAPTERS["faiss_flat"]()` creates a
     [`FaissFlatAdapter`](#102-backendsfaiss_backendpy).
   - `adapter.setup(784, Metric.L2, {})` creates a `faiss.IndexFlat(784,
     faiss.METRIC_L2)` and records the baseline process memory.
   - `adapter.build_index(train_vectors, ids)` inserts all 60,000 vectors
     (`ids = [0..59999]`) and times it (~0.07s).
   - `adapter.set_search_params({})` — no-op for Flat.
   - 50 warmup searches run and are discarded.
   - 3 × 10,000 = 30,000 timed searches run, each calling
     `adapter.search(query, k_max=10)`; the first 10,000 results are saved as
     `retrieved`; every individual latency is recorded.
   - [`latency_percentiles`](#latency_percentileslatencies_ms) and
     [`serial_qps`](#serial_qpsmean_ms) summarize the 30,000 timings.
   - [`recall_at_k(retrieved, gt, 10)`](#recall_at_kretrieved_ids-gt_ids-k)
     compares `retrieved` to the ground truth — for an exact index, this is
     `1.0`.
   - One [`RunResult`](#dataclass-class-runresult) is built (since
     `k_values = [10]` has one entry) with all of the above plus
     `peak_rss_mb` and `index_disk_mb` from the adapter.
   - `adapter.teardown()` frees the FAISS index.
4. Back in `main()`, that single `RunResult` is passed to
   `store.save(result)` ([store.py](#112-storagestorepy)), which converts it
   to a dict, JSON-encodes `build_params`/`search_params`, and `INSERT`s a row
   into the `results` table.
5. `main()` prints a one-line summary, e.g.:
   ```
   faiss_flat (flat) k=10: recall@10=1.0000 p50=20.476ms p95=28.002ms p99=32.014ms mean=20.976ms serial_qps=47.7 build_time=0.07s rss=185.4MB disk=179.4MB
   ```
6. `store.close()` closes the database file.

The result: `results/results.db` now contains one row proving that an exact
FAISS index achieves perfect recall on fashion-mnist-784-euclidean, alongside
its latency/throughput/memory/disk characteristics — the baseline every future
*approximate* adapter (HNSW, Qdrant, pgvector, Milvus) will be compared
against.

---

## 15. Where to go next

This document covers **Phase 0** exactly as it exists in the repo today. Per
`PROJECT_PLAN.md`'s roadmap, future phases will extend (not replace) these
same pieces:

- **Phase 1** adds a `FaissHNSWAdapter` (a second
  [`VectorDBAdapter`](#101-backendsbasepy) implementation, registered in
  [`ADAPTERS`](#121-harnessrunnerpy)) — recall should drop below `1.0` and
  vary with HNSW's `ef_search` parameter.
- **Phase 2** adds Qdrant/pgvector/Milvus adapters (more `VectorDBAdapter`
  implementations) and `core/normalize.py` (L2 normalization for cosine
  metrics).
- **Phase 3** adds `harness/sweep.py` (loops `run_backend`-like logic over a
  grid of `build_params`/`search_params`) and `harness/resources.py`
  (per-backend RSS for server processes).
- **Phase 4** adds `reporting/pareto.py` and `reporting/tables.py`, which
  *read* from `results/results.db` (via the same
  [`SQLiteStore`](#112-storagestorepy)/[schema](#111-storageschemapy)) to
  produce plots and summary tables.
- **Phase 5** adds a `datasets/corpus.py` + `embeddings/` package (Lane B) and
  `groundtruth/bruteforce.py` — both implementing the same
  [`DatasetSource`](#81-datasetsbasepy) /
  [`GroundTruthSource`](#91-groundtruthbasepy) interfaces documented above.
- **Phase 6** adds `cli.py`, an interactive menu that calls the same
  `run_backend`/`SQLiteStore`/reporting functions.

Because every extension point is one of the [ABC](#28-abstract-base-classes-abc-and-abstractmethod)
interfaces described in this document
([`DatasetSource`](#81-datasetsbasepy),
[`GroundTruthSource`](#91-groundtruthbasepy),
[`VectorDBAdapter`](#101-backendsbasepy)), understanding *this* document is
enough to understand how every future phase plugs in.
