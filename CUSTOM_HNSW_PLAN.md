# Custom C++ HNSW + pybind11 — Build Plan

> Paste into Claude Code. This is an inter-phase sub-project that sits **between Phase 1 and Phase 2** of `PROJECT_PLAN.md`. It adds two new in-process backends that plug into the existing `VectorDBAdapter` ABC (see `PROJECT_PLAN.md` §5) and run through the existing harness/sweep/reporting unchanged. Build **A → B → C → D** in order; each gate must pass before the next.

---

## 1. Goal

Implement HNSW from scratch in C++ with a pybind11 wrapper, in **two versions**, both registered as harness backends:

- **`hnsw_scalar`** — reference: scalar distance functions, **serial** build, single-thread search. No SIMD, no OpenMP.
- **`hnsw_fast`** — optimized: **VCL SIMD** distance kernels + **OpenMP parallel** build. Single-thread search (same as scalar).

The point is to benchmark naive vs optimized for: **build time**, **search latency (p50/p95/p99)**, **recall@k** (should match), and **index size** (should match) — all through the harness you already have.

### Fixed decisions
- Wrapper: **pybind11**.
- SIMD: **VCL** (Agner Fog's Vector Class Library v2, header-only, C++17). In scope for v1.
- OpenMP: **build only**. Search stays single-threaded for both versions — single-query serial latency is the harness's headline metric, and a single HNSW greedy traversal has sequential data dependencies, so parallelizing it would only add thread-spawn overhead and *slow down* latency. Parallelism goes in the build (parallel insertion).

### Critical framing
- The two-version build-time comparison **conflates SIMD and OpenMP** (both change between scalar and fast). Acceptable per scope. *Optional:* a third variant `hnsw_simd_serial` (SIMD distance + serial build) would isolate the two — build it only if asked.
- Expect `hnsw_fast` to be **slower than FAISS/Qdrant** unless SIMD + cache layout are good. That's the expected baseline for a from-scratch impl, not a failure — the interesting result is the size of the gap and where it comes from.

---

## 2. Design: maximize shared code, keep the perf comparison honest

The two versions must share the **entire algorithm** (graph structure, level assignment, search, neighbor selection/heuristic). They differ in exactly two places:

1. **Distance kernel** — scalar vs VCL SIMD. Select at **compile time** (template/functor) so each is fully inlined; runtime function-pointer dispatch would tax the scalar baseline with indirection and make the comparison dishonest.
2. **Build path** — serial insertion loop vs OpenMP-parallel insertion loop with locking.

Recommended structure: a templated core parameterized on a distance functor, instantiated per `(variant × metric)`. The Python backend picks the right class from the metric it gets in `setup()`.

```cpp
// Distance functors encode metric + scalar/SIMD; inlined at compile time.
struct ScalarL2  { static float dist(const float* a, const float* b, int d); };
struct ScalarIP  { static float dist(const float* a, const float* b, int d); }; // returns -dot (smaller = closer)
struct SimdL2    { static float dist(const float* a, const float* b, int d); }; // VCL Vec8f/Vec16f
struct SimdIP    { static float dist(const float* a, const float* b, int d); };

template <class Dist>
class HnswIndex {
public:
  HnswIndex(int dim, int M, int ef_construction, uint64_t seed);
  void add_points(const float* data, int n, bool parallel, int num_threads); // serial or OpenMP
  void set_ef(int ef_search);
  void search(const float* q, int k, int* out_ids) const;  // single query, single thread
  size_t index_memory_bytes() const;
  void save(const std::string& path) const;                // for on-disk size metric
  void load(const std::string& path);
};
```

- `hnsw_scalar` ⇒ `HnswIndex<ScalarL2|ScalarIP>` built with `parallel=false`.
- `hnsw_fast`   ⇒ `HnswIndex<SimdL2|SimdIP>` built with `parallel=true`.

pybind11 (`src/bindings.cpp`) exposes named classes per combination, e.g. `HnswScalarL2`, `HnswScalarIP`, `HnswFastL2`, `HnswFastIP`. Wrap `add_points`/`search` in `py::gil_scoped_release` (good hygiene; required for the later concurrency phase). Accept `py::array_t<float, c_style | forcecast>`; build takes `(N, dim)`, search takes `(dim,)` and returns a `list[int]` of ids in the dataset's id space (row index 0..N-1).

---

## 3. Repo additions

```
native/hnsw/
├── CMakeLists.txt
├── include/hnsw/
│   ├── hnsw_index.hpp          # templated core (shared algorithm)
│   ├── distances_scalar.hpp    # ScalarL2, ScalarIP
│   ├── distances_simd.hpp      # SimdL2, SimdIP (VCL)
│   ├── visited_pool.hpp        # epoch-counter visited set for search
│   └── locks.hpp               # striped lock pool for parallel build
├── src/bindings.cpp            # pybind11 module -> _vdbhnsw
├── tests/test_hnsw.cpp         # standalone: recall vs brute force; SIMD vs scalar
└── third_party/vcl/            # vendored VCL v2 headers

# Python side
pyproject.toml                  # scikit-build-core + CMake + pybind11; `pip install -e .` builds the ext
src/vdbbench/backends/hnsw_scalar_backend.py   # thin VectorDBAdapter -> _vdbhnsw.HnswScalar*
src/vdbbench/backends/hnsw_fast_backend.py     # thin VectorDBAdapter -> _vdbhnsw.HnswFast*
```

Build toolchain: C++17, `-O3 -march=native`, OpenMP (`-fopenmp`), VCL headers on the include path. Use scikit-build-core so the extension builds on `pip install -e .`.

---

## 4. SIMD specifics (VCL)

- Use `Vec8f` (AVX2/FMA, 8 floats) as the portable default; use `Vec16f` (AVX-512) if the host has it (`grep avx512f /proc/cpuinfo`). Pick via compile flag; `-march=native` on the HPC host lets VCL select.
- **L2:** accumulate `(a-b)^2` in a `Vec8f`, `horizontal_add` at the end. **IP/cosine:** accumulate `a*b`, return negative (HNSW expects smaller = closer); for cosine, vectors are L2-normalized centrally (harness already handles normalization) so IP == cosine.
- **Remainder handling is mandatory.** Dims that aren't multiples of 8/16 (e.g. glove-100 → 100 = 12×8 + 4) must process the tail via `load_partial`/`store_partial` or a scalar epilogue. fashion-mnist (784), SIFT (128), bge-m3 (1024) are clean; glove-100 is not — test it explicitly.
- float32 throughout (all datasets are float32).

---

## 5. Parallel build specifics (OpenMP) — the hard part

This is where the bugs live. Build the **serial** version (Phase A) fully correct first; only add OpenMP in Phase C.

- Parallelize the **insertion loop**: `#pragma omp parallel for` over points, each thread inserts one point.
- **Locking:** use a **striped lock pool** (e.g. 4096 mutexes indexed by `node_id % 4096`) rather than one mutex per node (cheaper memory at 1M nodes). Lock a node's stripe while modifying its neighbor list. When adding bidirectional links, acquire locks in a **consistent order** (by node id) to avoid deadlock. A separate global lock guards entry-point and `max_level` updates.
- **RNG:** level assignment is random — give each thread a thread-local RNG seeded deterministically (`seed ^ point_id`). Serial build is fully deterministic; parallel build is **not** (insertion order varies with scheduling) → recall varies slightly run to run. Document the tolerance.
- Build and search never run concurrently here (build completes before any search), so search remains lock-free.
- During development, build the parallel version with **ThreadSanitizer** (`-fsanitize=thread`) and run a stress test (large N, high thread count, repeated builds) to confirm no races and stable recall.

---

## 6. Search specifics (both versions)

- Single-query, single-thread. Use an **epoch-counter visited set** (`visited_pool.hpp`): an array of `uint32` version tags, bumped per query, to avoid reallocating a visited set each search.
- `ef_search` settable via `set_ef()` with **no rebuild** (maps to the adapter's `set_search_params`).

---

## 7. Harness integration

Both backends implement the existing `VectorDBAdapter` ABC unchanged:

- `setup(dim, metric, build_params)` → pick the C++ class by `(variant, metric)`; store `M`, `ef_construction`.
- `build_index(vectors, ids)` → `add_points` (fast: `parallel=True, num_threads=N`); return `BuildResult(build_time_s=...)`.
- `set_search_params({"ef_search": v})` → `set_ef(v)`, no rebuild.
- `search(query, k)` → C++ single-query → `list[int]`.
- `index_size_bytes()` → call `save()` to a temp file and measure file size (consistent with other backends' on-disk metric); also expose the exact in-memory `index_memory_bytes()`.
- `memory_rss_bytes()` → in-process RSS delta, same method as the FAISS backend.
- `teardown()` → free the index.

Sweep grid (`config/sweeps/hnsw_scalar.yaml`, `hnsw_fast.yaml`): build `M ∈ {8,16,32}`, `ef_construction ∈ {100,200}`; search `ef_search ∈ {20,40,80,160,320}`. Same grid for both so curves overlay.

---

## 8. Validation plan (gates correctness before any perf claim)

1. **Standalone C++ recall** (`tests/test_hnsw.cpp`): build on a small set, compare against brute-force exact — `recall@10` should be high (e.g. >0.95) at reasonable `ef`.
2. **SIMD vs scalar distance:** unit-test `SimdL2/SimdIP` against `ScalarL2/ScalarIP` on random vectors; allow small tolerance (reduction order differs), e.g. relative diff < 1e-3. **Test a non-multiple-of-8 dim (100).**
3. **`hnsw_scalar` vs reference:** through the harness on fashion-mnist, recall curve should track **hnswlib** (the closest reference impl) and FAISS HNSW at matched `M`/`ef_construction`/`ef_search` within a few percent. Large divergence = bug.
4. **`hnsw_fast` vs `hnsw_scalar`:** same params, recall within parallel-build nondeterminism tolerance (≈1–2%).
5. **Determinism:** scalar serial build with fixed seed → identical recall across runs. Parallel → document variance.

---

## 9. Phases

**Phase A — `hnsw_scalar` (scalar, serial), end-to-end.**
Templated core + scalar L2/IP + serial build + single-thread search + visited pool. CMake/pybind11/scikit-build wiring so `pip install -e .` builds `_vdbhnsw`. `save/load`. Standalone C++ recall test. Python `hnsw_scalar` backend registered in the harness.
*Acceptance:* `pip install -e .` builds clean; standalone test passes; through the harness on fashion-mnist, `recall@10 < 1.0` and tracks FAISS HNSW / hnswlib within a few %; a full results row is written; serial build is deterministic across runs.

**Phase B — VCL SIMD distance kernels.**
Add `SimdL2`/`SimdIP` (`Vec8f`, optional `Vec16f`) with remainder handling. Unit tests vs scalar. Instantiate the SIMD-distance classes (build still serial at this point).
*Acceptance:* SIMD-vs-scalar distance tests pass including dim=100; a SIMD-distance index matches scalar recall on fashion-mnist; search latency improves vs scalar.

**Phase C — OpenMP parallel build + locking ⇒ completes `hnsw_fast`.**
Add the parallel insertion path (striped locks, ordered acquisition, thread-local RNG). Pair SIMD distance + parallel build = `hnsw_fast`. ThreadSanitizer build + stress test.
*Acceptance:* TSan clean; parallel-build recall matches serial within tolerance; build time drops with thread count (report a thread-count scaling table); no crashes under repeated/large builds.

**Phase D — Register `hnsw_fast` + comparison.**
Register the `hnsw_fast` backend; run **both** through the existing sweep and reporting.
*Acceptance:* one Pareto plot showing `hnsw_scalar`, `hnsw_fast`, and FAISS HNSW overlaid; a comparison table of build_time / p50/p95/p99 / recall@10 / index_size for scalar vs fast at matched configs. Recall and index size match between the two; fast wins on build time and latency.

---
