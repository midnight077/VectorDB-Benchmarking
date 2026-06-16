# `native/hnsw` — Code Walkthrough

This document explains **every file under `native/hnsw/`**: what it does, what
each function/class is for, and how all the pieces fit together into a single
compiled Python module called `_vdbhnsw`.

Unlike the rest of `vdbbench` (which is Python — see `docs/CODEBASE.md`), this
directory is **C++**. It assumes **no prior C++ experience**. Every
C++-specific concept (headers, namespaces, templates, pointers, etc.) is
explained the first time it appears, with a primer in
[Section 2](#2-c-primer-read-this-first) you can refer back to.

If you want the *design rationale* — why this exists, what "Phase A" / "Phase
B" mean in the code comments, what's still to come — see
`CUSTOM_HNSW_PLAN.md` in the repo root. This document is about the *code that
exists right now*.

**Scope:** only files inside `native/hnsw/` are covered (per the request that
produced this doc). The Python files that *use* this code
(`src/vdbbench/backends/hnsw_scalar_backend.py`,
`src/vdbbench/backends/hnsw_fast_backend.py`) are mentioned only as context for
how the compiled module is consumed — see `docs/CODEBASE.md` for those.

---

## Table of contents

1. [The big picture](#1-the-big-picture)
2. [C++ primer (read this first)](#2-c-primer-read-this-first)
3. [Project layout](#3-project-layout)
4. [`CMakeLists.txt`](#4-cmakeliststxt)
5. [`include/hnsw/distances_scalar.hpp`](#5-includehnswdistances_scalarhpp)
6. [`include/hnsw/distances_simd.hpp`](#6-includehnswdistances_simdhpp)
7. [`include/hnsw/visited_pool.hpp`](#7-includehnswvisited_poolhpp)
8. [`include/hnsw/hnsw_index.hpp`](#8-includehnswhnsw_indexhpp) — the core algorithm
9. [`src/bindings.cpp`](#9-srcbindingscpp) — the Python bridge
10. [`tests/test_distances.cpp`](#10-teststest_distancescpp)
11. [`tests/test_hnsw.cpp`](#11-teststest_hnswcpp)
12. [`third_party/vcl/`](#12-third_partyvcl)
13. [End-to-end walkthroughs](#13-end-to-end-walkthroughs)
14. [Where to go next](#14-where-to-go-next)

---

## 1. The big picture

### What is HNSW?

**HNSW** stands for **Hierarchical Navigable Small World** graph. It's an
algorithm for *approximate nearest neighbor search*: given a big collection of
vectors (e.g. 1 million 128-dimensional embeddings) and a new "query" vector,
quickly find the handful of stored vectors that are closest to it — without
comparing the query against all 1 million (that would be "exact" search, but
too slow at scale).

HNSW does this by building a **graph** where each stored vector is a "node",
and nodes are connected to their approximate neighbors. Crucially, it builds
**multiple layers** of this graph, like a multi-level highway system:

- The **top layer** has very few nodes, sparsely connected — like a highway
  that lets you cover a lot of distance quickly.
- Each layer below has more nodes — like progressively more local roads.
- **Layer 0** (the bottom) contains *every* node, densely connected.

A search starts at a fixed **entry point** in the top layer, greedily walks
toward the query (always moving to whichever neighbor is closest), drops down
a layer once it can't get any closer, and repeats — until it reaches layer 0,
where it does a wider "beam search" to collect the final best candidates.

Building the graph means: for every new point, decide which layer it belongs
to (randomly, with higher layers being exponentially rarer), then connect it
to its approximate neighbors at each of its layers (using the same
greedy-search-then-beam-search machinery used at query time).

### What does `native/hnsw/` build?

This directory is a small, **self-contained C++ project** that compiles into
one Python extension module: **`_vdbhnsw`**. That module exposes four C++
classes to Python:

| Python class    | Distance kernel                  | Use case (from `CUSTOM_HNSW_PLAN.md`) |
|------------------|-----------------------------------|----------------------------------------|
| `HnswScalarL2`   | plain-loop squared L2             | `hnsw_scalar` backend, L2 metric        |
| `HnswScalarIP`   | plain-loop negative inner product | `hnsw_scalar` backend, cosine/IP metric |
| `HnswFastL2`     | SIMD (vectorized) squared L2      | `hnsw_fast` backend, L2 metric          |
| `HnswFastIP`     | SIMD negative inner product       | `hnsw_fast` backend, cosine/IP metric   |

All four classes share **the exact same algorithm** (graph structure, search,
neighbor selection). The only thing that differs between "Scalar" and "Fast"
is the **distance function** — one element at a time vs. 8/16 elements at
once using special CPU instructions (SIMD — explained in
[Section 6](#6-includehnswdistances_simdhpp)). This is deliberate: it lets the
benchmark compare "naive C++" vs. "SIMD-optimized C++" while keeping
everything else identical.

From Python, this all looks like:

```python
import _vdbhnsw

index = _vdbhnsw.HnswScalarL2(dim=128, M=16, ef_construction=200, seed=42)
index.add_points(vectors)        # build the graph from an (N, 128) float32 array
index.set_ef(100)                 # search-time tuning knob, no rebuild needed
ids = index.search(query, k=10)   # returns a list of 10 row-indices into `vectors`
```

The Python-side adapter classes (`HnswScalarAdapter` etc.) just translate
between `vdbbench`'s generic benchmarking interface and these four calls —
that translation layer lives outside `native/hnsw/` and isn't covered here.

### How the files relate

```
native/hnsw/
├── CMakeLists.txt              <- build recipe: "how to compile all this"
├── include/hnsw/
│   ├── distances_scalar.hpp    <- ScalarL2, ScalarIP   (plain-loop distance math)
│   ├── distances_simd.hpp      <- SimdL2, SimdIP       (SIMD distance math)
│   ├── visited_pool.hpp        <- VisitedPool          (search bookkeeping helper)
│   └── hnsw_index.hpp           <- HnswIndex<Dist>      (THE algorithm, generic over distance)
├── src/
│   └── bindings.cpp             <- glues HnswIndex<...> to Python via pybind11
├── tests/
│   ├── test_distances.cpp       <- checks SIMD math == scalar math
│   └── test_hnsw.cpp            <- checks the algorithm finds correct-ish neighbors
└── third_party/vcl/             <- vendored SIMD helper library (not our code)
```

`hnsw_index.hpp` is the heart of the project — it implements the actual HNSW
algorithm, but it's written *generically* so it doesn't know or care whether
distances are computed by `distances_scalar.hpp` or `distances_simd.hpp`. That
"plug in a different distance calculator" trick is done with C++ **templates**,
explained next.

---

## 2. C++ primer (read this first)

This section explains every C++ language feature used in this codebase, with
short examples. Skim it now; refer back to it as you read the file
walkthroughs. Where helpful, comparisons to Python are included.

### 2.1 Source files: `.hpp` and `.cpp`

- **`.hpp`** files are "header" files. They contain *declarations and
  definitions* meant to be `#include`d into other files — roughly Python's
  equivalent of a module you `import` from, except the *text* of the file
  gets pasted in wherever it's included (handled by the **preprocessor**,
  before real compilation starts).
- **`.cpp`** files are "source" files that get compiled into actual machine
  code. `src/bindings.cpp` is the only `.cpp` file that becomes part of the
  Python module; the `.cpp` files under `tests/` become separate standalone
  programs.
- **`#pragma once`** at the top of every `.hpp` file here is an "include
  guard": it tells the compiler "if this file gets `#include`d more than once
  (which happens easily with headers), only process it the first time." Without
  it, re-including the same header could cause duplicate-definition errors.

### 2.2 `#include`

```cpp
#include <vector>              // a "system"/standard-library header
#include "hnsw/visited_pool.hpp"  // a header from this project
```

Angle brackets `<...>` mean "look in the standard library / system include
paths". Quotes `"..."` mean "look relative to this project's include
directories" (configured in `CMakeLists.txt`, [Section 4](#4-cmakeliststxt)).
Conceptually this is like Python's `import`, except — again — it's literal
text substitution before compilation.

### 2.3 Namespaces

```cpp
namespace hnsw {
    struct ScalarL2 { ... };
}  // namespace hnsw
```

A **namespace** is a named container for names (types, functions), so two
libraries can both define something called `Index` without colliding. Code
outside the namespace refers to it as `hnsw::ScalarL2`. It's similar to how in
Python `vdbbench.core.types.Metric` and `some_other_package.types.Metric`
don't collide because they live in different packages. Every file in this
project wraps its contents in `namespace hnsw { ... }`.

### 2.4 `struct` vs `class`

In C++, `struct` and `class` are almost the same thing — the only difference
is that `struct` members are `public` by default, while `class` members are
`private` by default. This codebase uses `struct` for small "bag of
behavior" types (like `ScalarL2`, which has no data, just a function) and
`class` for types with internal state and an interface (like `HnswIndex`,
`VisitedPool`).

### 2.5 Templates — "generic" types/functions

```cpp
template <class Dist>
class HnswIndex {
    // ... can use `Dist::dist(...)` anywhere in here ...
};
```

A **template** lets you write code once and have the compiler stamp out a
specialized copy for each type you use it with. Here, `Dist` is a
*placeholder* for "whatever distance-calculator type you give me" —
`ScalarL2`, `ScalarIP`, `SimdL2`, or `SimdIP` (from
[Section 5](#5-includehnswdistances_scalarhpp) and
[6](#6-includehnswdistances_simdhpp)).

When code elsewhere writes `HnswIndex<ScalarL2>`, the compiler generates a
complete, independent version of `HnswIndex` where every `Dist::dist(...)`
call has been replaced with `ScalarL2::dist(...)` — and because this happens
at *compile time*, the compiler can **inline** that function call (paste its
body directly in), so there's zero runtime overhead compared to having
hardcoded `ScalarL2` everywhere. This is the mechanism that lets
`src/bindings.cpp` produce four different Python classes
(`HnswScalarL2`, `HnswScalarIP`, `HnswFastL2`, `HnswFastIP`) from **one**
implementation of the algorithm.

The closest Python analogy is generic functions/duck typing, except in C++
this is resolved entirely before the program runs, producing four separate
compiled algorithms baked with four different distance functions.

### 2.6 Static methods on a "policy" type

```cpp
struct ScalarL2 {
    static float dist(const float* a, const float* b, int d) { ... }
};
```

`static` here means "this function doesn't need an *instance* of `ScalarL2`
to call — call it as `ScalarL2::dist(a, b, d)`, like a plain function attached
to a namespace." `ScalarL2`/`SimdL2`/etc. never get *constructed* — they exist
purely so they can be passed as template arguments (`Dist`) and provide a
`dist` function.

### 2.7 Pointers and references

- **Pointer** (`*`): a variable holding the *memory address* of some data.
  `const float* a` means "a pointer to (read-only) `float` data — i.e. the
  address where a vector's numbers start in memory." `a[i]` reads the `i`-th
  `float` starting from that address. Pointers are how this code passes large
  arrays around without copying them.
- **Reference** (`&`): an alias for an existing variable — "another name for
  the same storage", with no separate copy. `std::vector<uint32_t>& tags_`
  means "a reference to *someone else's* vector; changes through this name
  change their vector too." Python doesn't really have an equivalent — the
  closest mental model is that *all* Python variables are references to
  objects, whereas in C++ you choose whether to copy or refer.
- **`const`**: "this data won't be modified through this name." `const float*`
  = "pointer to data I promise not to change." A function marked `const`
  after its parameter list (e.g. `size_t size() const`) promises not to modify
  the object it's called on.

### 2.8 The standard library containers used here

- **`std::vector<T>`** — a growable array of `T`, roughly like a Python
  `list` but constrained to one element type and stored contiguously in
  memory. `.size()` = number of elements currently stored. `.capacity()` =
  how much memory is currently *allocated* (often more than `.size()`, since
  vectors over-allocate to amortize growth). `.resize(n)` changes the logical
  size (zero-filling new elements for numeric types); `.reserve(n)` /
  `.assign(n, v)` are used for pre-allocating/resetting.
- **`std::pair<A, B>`** — a simple 2-tuple, `.first` and `.second`. Used here
  as `std::pair<float, uint32_t>` = "(distance, node id)". Comparing two
  `pair`s with `<` compares `.first` first, then `.second` as a tie-breaker —
  this is relied upon throughout the algorithm to compare candidates by
  distance.
- **`std::priority_queue<T>`** — a heap: efficiently keeps track of the
  "biggest" (by default) element. `.top()` peeks at it, `.push()`/`.emplace()`
  insert, `.pop()` removes the top. By default it's a **max-heap** (top =
  largest). Passing `std::greater<T>` as a custom comparator flips it into a
  **min-heap** (top = smallest). This codebase uses both — see
  `MaxHeap`/`MinHeap` in [Section 8](#8-includehnswhnsw_indexhpp).
- **C++17 structured bindings**: `for (const auto& [dist_to_point, cand_id] :
  sorted)` unpacks each `std::pair<float, uint32_t>` into two named variables
  `dist_to_point` and `cand_id` — like Python's `for a, b in pairs:`.

### 2.9 Casts

You'll see things like `static_cast<size_t>(n)` and
`reinterpret_cast<const char*>(&magic)`.

- **`static_cast<T>(x)`** — a normal, checked type conversion, e.g.
  `int` → `size_t` (an unsigned integer type used for sizes/counts) or `int` →
  `double`. Used a lot here purely to satisfy the compiler's strictness about
  mixing signed/unsigned integer types.
- **`reinterpret_cast<const char*>(&x)`** — "treat the memory where `x` lives
  as raw bytes (`char`s)", used only for writing/reading binary files
  (`save`/`load` in [Section 8](#8-includehnswhnsw_indexhpp)). This has no
  Python equivalent — Python never lets you look at an object's raw memory
  layout like this.

### 2.10 Preprocessor conditionals

```cpp
#if defined(__AVX512F__)
using SimdF = Vec16f;
#else
using SimdF = Vec8f;
#endif
```

`#if`/`#else`/`#endif` are resolved by the *preprocessor*, before real
compilation — so this is a **compile-time** choice baked into the binary based
on what CPU features the compiler was told to target (`-march=native`, see
[Section 4](#4-cmakeliststxt)), not a runtime `if`. `using SimdF = Vec16f;` is
a *type alias* — like writing `SimdF = Vec16f` to mean "everywhere I write
`SimdF`, substitute the type `Vec16f`."

### 2.11 `mutable`

```cpp
mutable VisitedPool visited_pool_;
```

Normally, a `const` method (like `search() const`) can't modify any of the
object's fields. `mutable` is an escape hatch for fields that represent
"internal bookkeeping/cache" rather than "real, externally-visible state" —
it lets `search() const` still update `visited_pool_`'s internal counters.

---

## 3. Project layout

```
native/hnsw/
├── CMakeLists.txt
├── include/hnsw/
│   ├── distances_scalar.hpp
│   ├── distances_simd.hpp
│   ├── visited_pool.hpp
│   └── hnsw_index.hpp
├── src/
│   └── bindings.cpp
├── tests/
│   ├── test_distances.cpp
│   └── test_hnsw.cpp
├── third_party/vcl/        (vendored SIMD library, see Section 12)
└── build-test/              (generated by CMake when building the standalone
                               tests — not source code, safe to delete/ignore;
                               not covered in this document)
```

- **`include/hnsw/`** — all the actual algorithm code, as headers only (no
  separate `.cpp` files). Because templates need their full definition visible
  at every place they're used, header-only is the simplest way to organize a
  small templated C++ library like this.
- **`src/bindings.cpp`** — the *only* `.cpp` file that gets compiled into the
  Python module. It `#include`s the headers above and wraps them for Python.
- **`tests/`** — two standalone C++ programs (each with their own `main()`),
  built separately from the Python extension, for checking correctness
  without going through Python at all.
- **`third_party/vcl/`** — a copy of someone else's open-source library
  (Vector Class Library), used by `distances_simd.hpp`.

---

## 4. `CMakeLists.txt`

This file is **not C++** — it's a script for **CMake**, the tool that figures
out *how to compile* all the C++ files into the final product(s). Think of it
as the "build recipe": which files to compile, which compiler flags to use,
where to find dependencies, and where to put the output.

Walking through it top to bottom:

```cmake
cmake_minimum_required(VERSION 3.18)
project(vdbhnsw LANGUAGES CXX)
```
Declares the minimum CMake version needed and names the overall project
`vdbhnsw`, written in C++ (`CXX`).

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
```
- Use the **C++17** language standard (this is what enables structured
  bindings like `[dist_to_point, cand_id]` from
  [Section 2.8](#28-the-standard-library-containers-used-here)), and require
  it strictly (fail rather than silently fall back to an older standard).
- "Position independent code" is a requirement for building **shared
  libraries** (like the `.so` file Python loads) — every function's machine
  code must work no matter where in memory it ends up loaded.

```cmake
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
```
If no build type was specified, default to `Release` (optimized, as opposed
to `Debug`), and for `Release` builds pass `-O3` to the compiler — "optimize
aggressively for speed."

```cmake
# Tune for the build host's instruction set (AVX2/FMA or AVX-512) so the VCL
# distance kernels in distances_simd.hpp pick Vec8f/Vec16f accordingly.
add_compile_options(-march=native)
```
`-march=native` tells the compiler "generate code that uses whatever CPU
instruction extensions *this specific machine* supports" (e.g. AVX2,
AVX-512). This is what makes the `#if defined(__AVX512F__)` check in
`distances_simd.hpp` ([Section 6](#6-includehnswdistances_simdhpp)) come out
true or false — the compiler defines `__AVX512F__` automatically when
`-march=native` detects AVX-512 support on the build machine. A consequence:
**a binary built on one machine may not run (or may not use the fast path) on
a different CPU** — acceptable for this benchmarking project, which is built
and run on the same host.

```cmake
find_package(pybind11 CONFIG REQUIRED)
```
Locates the **pybind11** library (the C++↔Python bridge used in
`src/bindings.cpp`, [Section 9](#9-srcbindingscpp)) — `REQUIRED` means "fail
the configure step if it's not found."

```cmake
set(VCL_INCLUDE_DIR third_party/vcl)
```
Just a variable holding a path, for reuse below.

```cmake
pybind11_add_module(_vdbhnsw src/bindings.cpp)
target_include_directories(_vdbhnsw PRIVATE include ${VCL_INCLUDE_DIR})
```
This is the **main event**: `pybind11_add_module` is a helper (provided by
pybind11) that says "compile `src/bindings.cpp` (which transitively pulls in
all of `include/hnsw/*.hpp` and the VCL headers) into a Python extension
module named `_vdbhnsw`." `target_include_directories` tells the compiler
where `#include "hnsw/..."` and `#include "vectorclass.h"` should look —
`include/` and `third_party/vcl/` respectively. `PRIVATE` means "only this
target needs these include paths" (irrelevant here since there's only one
real target, but it's the conventional default).

```cmake
# Land the extension module at the wheel root so `import _vdbhnsw` works as a
# top-level module alongside the `vdbbench` package.
install(TARGETS _vdbhnsw LIBRARY DESTINATION .)
```
When the Python package is installed (`pip install -e .`), this places the
compiled `_vdbhnsw*.so` file at the root of the installed package, so
`import _vdbhnsw` works as a plain top-level import (matching what
`hnsw_scalar_backend.py` does: `import _vdbhnsw`).

```cmake
option(BUILD_TESTS "Build standalone C++ tests" OFF)
if(BUILD_TESTS)
  add_executable(test_hnsw tests/test_hnsw.cpp)
  target_include_directories(test_hnsw PRIVATE include ${VCL_INCLUDE_DIR})
  target_compile_options(test_hnsw PRIVATE -O3)

  add_executable(test_distances tests/test_distances.cpp)
  target_include_directories(test_distances PRIVATE include ${VCL_INCLUDE_DIR})
  target_compile_options(test_distances PRIVATE -O3)
endif()
```
`option(BUILD_TESTS ... OFF)` defines a switch, **off by default**, so that a
normal `pip install -e .` (which only needs `_vdbhnsw`) doesn't waste time
building the test programs. If you explicitly turn it on:

```bash
cmake -S native/hnsw -B native/hnsw/build-test -DBUILD_TESTS=ON
cmake --build native/hnsw/build-test
```

...two extra standalone executables get built: `test_hnsw` and
`test_distances` (covered in [Sections 10](#10-teststest_distancescpp) and
[11](#11-teststest_hnswcpp)), each with the same include paths and `-O3`.
This is exactly the `build-test/` directory you see sitting in the repo —
it's CMake's output from having run those two commands.

---

## 5. `include/hnsw/distances_scalar.hpp`

This is the simplest file in the project: it defines two tiny "distance
calculator" types, each with one function, written as plain loops (no SIMD).
These are the building blocks for the `hnsw_scalar` backend
(`HnswScalarL2` / `HnswScalarIP` in `src/bindings.cpp`).

```cpp
#pragma once

namespace hnsw {

// Squared L2 distance. Smaller = closer.
struct ScalarL2 {
    static float dist(const float* a, const float* b, int d) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
};
```

**`ScalarL2::dist(a, b, d)`** computes the **squared Euclidean (L2)
distance** between two `d`-dimensional vectors `a` and `b`: for each
dimension `i`, take the difference `a[i] - b[i]`, square it, and sum all `d`
of those squares. (It's "squared" L2 — no final square root — because for
the purposes of *ranking* which points are closest, comparing squared
distances gives the same ordering as comparing true distances, and skipping
the square root is faster.) **Smaller return value = the two vectors are
closer together** — this "smaller = closer" convention is what every distance
type in this project must follow, because the HNSW algorithm in
`hnsw_index.hpp` is written assuming it.

```cpp
// Negative inner product, so smaller = closer (matches HNSW's L2 convention).
// For cosine, vectors are L2-normalized centrally by the harness, so IP == cosine.
struct ScalarIP {
    static float dist(const float* a, const float* b, int d) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += a[i] * b[i];
        }
        return -sum;
    }
};

}  // namespace hnsw
```

**`ScalarIP::dist(a, b, d)`** computes the **inner product** (a.k.a. dot
product) of `a` and `b` — sum of `a[i] * b[i]` over all dimensions — and
**negates it**. Why negate? A *larger* inner product means the vectors point
in *more similar* directions (more "similar" = intuitively should mean
"closer"), but the HNSW algorithm everywhere expects "smaller number = closer
point". Negating the inner product flips its sense to match: the most similar
pair gets the most negative (smallest) value.

The comment about cosine: **cosine similarity** between two vectors equals
their inner product *if both vectors have been scaled to length 1* (L2-
normalized). This project's broader harness normalizes vectors before they
ever reach this code (see `CUSTOM_HNSW_PLAN.md` / `PROJECT_PLAN.md` §2), so
`ScalarIP` doubles as both the "inner product" metric and the "cosine" metric
— there's no separate cosine implementation needed here.

Both types are `struct`s with only a `static` function and no data — they
exist purely to be passed as the `Dist` template parameter to `HnswIndex<Dist>`
(see [Section 2.5](#25-templates--generic-typesfunctions) and
[Section 8](#8-includehnswhnsw_indexhpp)).

---

## 6. `include/hnsw/distances_simd.hpp`

This file defines `SimdL2` and `SimdIP` — the **SIMD-accelerated** versions
of the same two distance functions from Section 5, used by the `hnsw_fast`
backend (`HnswFastL2` / `HnswFastIP`). They compute the *same mathematical
result* as `ScalarL2`/`ScalarIP` (up to tiny floating-point rounding
differences — verified by `tests/test_distances.cpp`,
[Section 10](#10-teststest_distancescpp)), but much faster.

### What is SIMD, briefly

**SIMD** = "Single Instruction, Multiple Data". Modern CPUs have special
registers and instructions that can operate on **several numbers at once** —
e.g. one instruction subtracts 8 pairs of `float`s simultaneously, instead of
needing 8 separate subtract instructions. `distances_simd.hpp` uses a library
called **VCL** (Vector Class Library, see [Section 12](#12-third_partyvcl)) to
write this without hand-coding raw CPU instructions — VCL provides C++ types
like `Vec8f` ("a vector of 8 `float`s") with overloaded operators (`+`, `-`,
`*`) and helper functions that compile down to the right SIMD instructions.

### The code

```cpp
#pragma once

#include "vectorclass.h"

namespace hnsw {

// Vector width selected at compile time: Vec16f (AVX-512) if the target
// supports AVX512F, otherwise Vec8f (AVX2/FMA). Picked via -march=native.
#if defined(__AVX512F__)
using SimdF = Vec16f;
#else
using SimdF = Vec8f;
#endif
```

`SimdF` is a **type alias** chosen at compile time (see
[Section 2.10](#210-preprocessor-conditionals)): if the machine compiling this
supports the AVX-512 instruction set, `SimdF` means "a vector of 16 `float`s"
(`Vec16f`); otherwise it falls back to "a vector of 8 `float`s" (`Vec8f`,
AVX2). Everything below is written purely in terms of `SimdF`, so it
automatically gets wider (processes more numbers per step) on machines that
support it.

```cpp
// Squared L2 distance via VCL SIMD. Smaller = closer. Same convention as
// ScalarL2 (distances_scalar.hpp); reduction order differs so results match
// only to within float rounding.
struct SimdL2 {
    static float dist(const float* a, const float* b, int d) {
        constexpr int W = SimdF::size();
        SimdF acc(0.0f);
        int i = 0;
        for (; i + W <= d; i += W) {
            SimdF va = SimdF().load(a + i);
            SimdF vb = SimdF().load(b + i);
            SimdF diff = va - vb;
            acc = mul_add(diff, diff, acc);
        }
        float sum = horizontal_add(acc);

        const int rem = d - i;
        if (rem > 0) {
            // load_partial zero-fills the remaining lanes, so the squared
            // difference in those lanes is 0 and doesn't perturb the sum.
            SimdF va = SimdF().load_partial(rem, a + i);
            SimdF vb = SimdF().load_partial(rem, b + i);
            SimdF diff = va - vb;
            sum += horizontal_add(diff * diff);
        }
        return sum;
    }
};
```

Step by step, for `SimdL2::dist(a, b, d)`:

1. **`constexpr int W = SimdF::size();`** — `W` is the number of `float`s
   `SimdF` handles per operation (8 or 16), known at compile time
   (`constexpr`).
2. **`SimdF acc(0.0f);`** — `acc` is a SIMD "accumulator" register, initialized
   so all 8 (or 16) of its lanes start at `0.0`.
3. **Main loop** (`for (; i + W <= d; i += W)`) — processes the vector `W`
   elements at a time, as long as a full chunk of `W` remains:
   - `SimdF().load(a + i)` reads `W` consecutive `float`s starting at `a[i]`
     into a SIMD register `va` (and similarly `vb` from `b`). This is the SIMD
     equivalent of `a[i], a[i+1], ..., a[i+W-1]` all at once.
   - `SimdF diff = va - vb;` — one instruction computes all `W` differences
     `a[i+j] - b[i+j]` simultaneously.
   - `acc = mul_add(diff, diff, acc);` — a **fused multiply-add**: computes
     `acc = diff * diff + acc` for all `W` lanes in one step. This is exactly
     the squared-difference-then-accumulate from `ScalarL2`, just done `W`
     elements at a time.
4. **`float sum = horizontal_add(acc);`** — after the loop, `acc` holds `W`
   *partial sums* (one per lane). `horizontal_add` adds all `W` lanes together
   into a single `float`.
5. **Remainder handling** (`rem = d - i`): if `d` isn't a multiple of `W`,
   there are `rem` leftover elements (e.g. for `d=100` and `W=8`, after 12
   full chunks of 8, `rem = 4`). `load_partial(rem, ptr)` loads only those
   `rem` real values and fills the remaining lanes of the SIMD register with
   zero — so squaring those zero-filled lanes contributes `0` to the sum and
   doesn't corrupt the result. The comment explicitly flags this as the
   important "tail" case that `tests/test_distances.cpp` exercises with
   `dim=100`.

The result is mathematically the same sum as `ScalarL2`, just computed by
adding numbers together in a **different order** (in groups of `W`, then
combined) — floating-point addition isn't perfectly associative, so the final
bit pattern can differ very slightly (this is the "reduction order differs"
note in the comment, and is why the test in Section 10 checks for *approximate*
equality, not exact).

```cpp
// Negative inner product via VCL SIMD, so smaller = closer (matches HNSW's
// L2 convention). For cosine, vectors are L2-normalized centrally by the
// harness, so IP == cosine.
struct SimdIP {
    static float dist(const float* a, const float* b, int d) {
        constexpr int W = SimdF::size();
        SimdF acc(0.0f);
        int i = 0;
        for (; i + W <= d; i += W) {
            SimdF va = SimdF().load(a + i);
            SimdF vb = SimdF().load(b + i);
            acc = mul_add(va, vb, acc);
        }
        float sum = horizontal_add(acc);

        const int rem = d - i;
        if (rem > 0) {
            SimdF va = SimdF().load_partial(rem, a + i);
            SimdF vb = SimdF().load_partial(rem, b + i);
            sum += horizontal_add(va * vb);
        }
        return -sum;
    }
};

}  // namespace hnsw
```

`SimdIP::dist` is the SIMD twin of `ScalarIP`: same structure as `SimdL2`, but
accumulates `a[i] * b[i]` directly (`mul_add(va, vb, acc)` computes `acc += va
* vb`, the SIMD dot product) instead of squared differences, then negates the
final sum — for the same "smaller = closer" reason explained in Section 5.

---

## 7. `include/hnsw/visited_pool.hpp`

During a search, HNSW needs to remember **which nodes it has already looked
at**, so it doesn't re-examine the same node over and over (which could cause
wasted work or even infinite loops). The obvious approach — a big array of
`true`/`false` flags, reset before every search — has a hidden cost: with an
index of millions of points and millions of searches, *clearing* (or
reallocating) a multi-million-entry array on every single search adds up.

`VisitedPool` solves this with a classic trick: an **epoch counter**.

```cpp
#pragma once

#include <cstdint>
#include <vector>

namespace hnsw {

// Epoch-counter visited set: avoids reallocating/clearing a bitset per query.
// Each id's "visited" state is encoded by whether tags_[id] == current tag.
class VisitedPool {
public:
    explicit VisitedPool(size_t size = 0) : tags_(size, 0), cur_tag_(0) {}
```

Instead of `true`/`false`, `tags_` is a `std::vector<uint32_t>` — one
"version number" (`uint32_t`, an unsigned 32-bit integer) per point in the
index, all starting at `0`. `cur_tag_` is a single global counter, also
starting at `0`.

`explicit` on the constructor means: this constructor can't be used for
*implicit* conversions (e.g. you can't accidentally write something like
`VisitedPool v = 100;` and have it silently construct a `VisitedPool` of size
100 — you must write `VisitedPool(100)`). It's a general C++ safety habit for
single-argument constructors.

The `: tags_(size, 0), cur_tag_(0)` part is a **member initializer list** —
before the (empty) constructor body `{}` runs, `tags_` is initialized to a
vector of `size` elements all set to `0`, and `cur_tag_` is set to `0`.

```cpp
    void resize(size_t size) {
        tags_.assign(size, 0);
        cur_tag_ = 0;
    }

    size_t capacity_bytes() const { return tags_.capacity() * sizeof(uint32_t); }
```

- **`resize(size)`** — re-creates `tags_` as `size` zeros and resets
  `cur_tag_` to `0`. Called whenever the index grows (new points added via
  `add_points`, see [Section 8.6](#86-add_points--building-the-index)), since
  every point needs its own tag slot.
- **`capacity_bytes()`** — how much memory `tags_` currently occupies (used by
  `HnswIndex::index_memory_bytes()`, [Section 8.12](#812-size-dim-and-index_memory_bytes)).

```cpp
    // A lightweight view bound to the current epoch tag.
    class VisitedList {
    public:
        VisitedList(std::vector<uint32_t>& tags, uint32_t tag) : tags_(tags), tag_(tag) {}

        bool is_visited(uint32_t id) const { return tags_[id] == tag_; }
        void set_visited(uint32_t id) { tags_[id] = tag_; }

    private:
        std::vector<uint32_t>& tags_;
        uint32_t tag_;
    };
```

`VisitedList` is a tiny helper *class nested inside* `VisitedPool`. An
instance of it represents "the visited-state for one specific search". It
holds:
- `tags_` — a **reference** (`&`, see
  [Section 2.7](#27-pointers-and-references)) to `VisitedPool`'s `tags_`
  vector — i.e. *the same underlying array*, not a copy.
- `tag_` — the specific tag value that means "visited, for this search".

`is_visited(id)` is then simply: "does this node's tag match *this search's*
tag?" `set_visited(id)` marks a node as visited *for this search* by writing
this search's tag into its slot.

```cpp
    // Bump the epoch and hand back a view for a fresh query. On overflow,
    // reset all tags to 0 and restart numbering at 1.
    VisitedList get() {
        ++cur_tag_;
        if (cur_tag_ == 0) {
            std::fill(tags_.begin(), tags_.end(), 0);
            cur_tag_ = 1;
        }
        return VisitedList(tags_, cur_tag_);
    }

private:
    std::vector<uint32_t> tags_;
    uint32_t cur_tag_;
};

}  // namespace hnsw
```

**`get()`** is called once at the start of every search
([Section 8.11](#811-search--the-public-entry-point-for-querying)) and once
per point being inserted during build
([Section 8.10](#810-add_point--inserting-one-point)). It:

1. Increments `cur_tag_` — this is the entire "reset": the *previous* epoch's
   tag is now stale, so every `tags_[id]` from before (which still holds the
   *old* tag value) will no longer equal the *new* `cur_tag_`, meaning every
   node is "not visited" again — **without touching the array at all**. This
   is the whole point of the trick: resetting is **O(1)** instead of O(number
   of points).
2. **Overflow check**: `uint32_t` can only count up to about 4.29 billion
   before wrapping back to `0`. If that happens (`cur_tag_ == 0` after the
   increment), a stale `tags_[id] == 0` from the very first construction could
   falsely match. So on wraparound, it does pay the O(n) cost *once* —
   `std::fill` resets every slot back to `0` — and restarts numbering at `1`
   (never `0`, so the "freshly constructed/reset" state of `0` never
   collides with a real epoch).
3. Returns a `VisitedList` bound to the new `cur_tag_`.

---

## 8. `include/hnsw/hnsw_index.hpp`

This is **the core of the entire project** — a templated class,
`HnswIndex<Dist>`, that implements the full HNSW algorithm: building the
multi-layer graph and searching it. Everything else in `native/hnsw/` exists
to support, expose, or test this one file.

### 8.1 Overview & class declaration

```cpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnsw/visited_pool.hpp"

namespace hnsw {

// Templated HNSW core shared by all variants (scalar/SIMD, serial/parallel build).
// `Dist::dist(a, b, d)` must return a "smaller = closer" distance (squared L2 for
// L2, negative inner product for IP/cosine).
//
// Phase A: serial build, single-thread search. `add_points`'s `parallel`/
// `num_threads` arguments are accepted (for ABI stability with later phases) but
// ignored here.
template <class Dist>
class HnswIndex {
public:
    ...
private:
    ...
};

}  // namespace hnsw
```

The standard-library headers included here:
- `<vector>`, `<queue>` — `std::vector` and `std::priority_queue` (heaps),
  see [Section 2.8](#28-the-standard-library-containers-used-here).
- `<random>` — random-number generation (`std::mt19937_64`,
  `std::uniform_real_distribution`), used for assigning each point a random
  "top layer".
- `<algorithm>` — generic algorithms like `std::sort`, `std::min`, `std::max`,
  `std::reverse`, `std::fill`.
- `<fstream>` — file streams, for `save`/`load`
  ([Section 8.13](#813-save-and-load)).
- `<cstring>` — `std::memcpy` (fast bulk memory copy), used in
  `add_points`.
- `<stdexcept>` — `std::runtime_error`, thrown when `save`/`load` hit I/O
  problems.
- `"hnsw/visited_pool.hpp"` — the `VisitedPool` class from Section 7.

`HnswIndex<Dist>` is a **template** ([Section 2.5](#25-templates--generic-typesfunctions)):
`Dist` is one of `ScalarL2`, `ScalarIP`, `SimdL2`, `SimdIP`. The class-level
comment is an important contract: *whatever* `Dist` is, calling
`Dist::dist(a, b, d)` must return a number where **smaller means "closer"** —
every algorithm below (greedy descent, beam search, neighbor selection) is
written assuming that contract.

The comment about "Phase A" / `parallel`/`num_threads` refers to
`CUSTOM_HNSW_PLAN.md`: a future phase will add a multi-threaded (OpenMP)
build path. The current code already accepts those parameters (so the Python
binding's function signature won't need to change later) but ignores them —
building is always single-threaded for now.

### 8.2 Member variables (fields) at a glance

These are declared at the very bottom of the class (in the `private:`
section), but it's easier to understand the methods if you know what state
they operate on first:

| Field | Type | Meaning |
|---|---|---|
| `dim_` | `int` | Dimensionality of each vector (e.g. 128). |
| `M_` | `int` | Target number of graph neighbors per node, at every layer except layer 0. |
| `M_max0_` | `int` | Like `M_`, but for layer 0 — always `2 * M_`. Layer 0 contains *every* point, so it's given a denser connection budget. |
| `ef_construction_` | `int` | Beam width (how many candidates to consider) while *building* the graph — bigger = better-quality graph, slower build. |
| `ef_search_` | `int` | Beam width while *searching* — settable at runtime via `set_ef()`, no rebuild needed. |
| `mL_` | `double` | A normalization constant used to randomly pick each new point's top layer (`1 / ln(max(M_, 2))`). |
| `entry_point_` | `uint32_t` | The id of the node where every search/insertion starts (always a node that exists in the topmost layer). |
| `max_level_` | `int` | The highest layer that currently has any nodes in it. `-1` means "the index is empty." |
| `data_` | `std::vector<float>` | All vectors, stored back-to-back ("row-major"): point `id`'s coordinates start at `data_[id * dim_]`. |
| `levels_` | `std::vector<int>` | `levels_[id]` = the highest layer that point `id` belongs to. |
| `links_` | `std::vector<std::vector<std::vector<uint32_t>>>` | The graph itself: `links_[id][level]` is the list of neighbor ids for point `id` at `level`. |
| `visited_pool_` | `VisitedPool` (mutable) | The Section 7 helper, reused across searches/insertions. |
| `rng_` | `std::mt19937_64` | The random-number generator used for `random_level()` — seeded once in the constructor, so a given `seed` always produces the same graph. |

The triply-nested `links_` type is worth pausing on:
`std::vector<std::vector<std::vector<uint32_t>>>` reads as "for each point id
(outer vector), for each layer that point belongs to (middle vector), a list
of neighbor ids (inner vector of `uint32_t`)." So `links_[5][0]` is "the list
of layer-0 neighbors of point 5", and `links_[5].size()` is "how many layers
point 5 belongs to" — i.e. `levels_[5] + 1` (layers are numbered from `0`).

### 8.3 Constructor

```cpp
    HnswIndex(int dim, int M, int ef_construction, uint64_t seed = 42)
        : dim_(dim),
          M_(M),
          M_max0_(2 * M),
          ef_construction_(ef_construction),
          ef_search_(10),
          mL_(1.0 / std::log(static_cast<double>(std::max(M, 2)))),
          entry_point_(0),
          max_level_(-1),
          rng_(seed) {}
```

Again, this `: a(...), b(...), ...` syntax is a **member initializer list** —
each named field is set to the given expression, in the order the fields are
*declared* in the class (not the order written here), before the (empty)
constructor body `{}` runs.

- `dim_`, `M_`, `ef_construction_` — copied directly from the constructor
  arguments.
- `M_max0_(2 * M)` — layer 0's neighbor budget is fixed at double `M`. This
  matches the well-known `hnswlib` reference implementation's choice (denser
  bottom layer, since *all* points live there but only a fraction reach higher
  layers).
- `ef_search_(10)` — a sensible default search beam width, overridable any
  time via `set_ef()` ([Section 8.6](#86-add_points--building-the-index))
  without rebuilding the graph.
- `mL_(1.0 / std::log(static_cast<double>(std::max(M, 2))))` — computes
  `1 / ln(max(M, 2))`. `std::max(M, 2)` guards against `M = 1` or `M = 0`,
  which would make `ln(M) <= 0` (undefined or zero — division by it would be
  invalid). This value is the scale parameter for `random_level()`
  ([Section 8.5](#85-helper-get_data-and-random_level)) — it controls how
  quickly the number of nodes shrinks as you go up a layer (roughly, each
  layer has about `1/M` as many nodes as the layer below it).
- `entry_point_(0)` — a placeholder value. It only becomes *meaningful* once
  the first point is added (`add_point`,
  [Section 8.10](#810-add_point--inserting-one-point)), which explicitly sets it.
- `max_level_(-1)` — the sentinel value meaning "the index has zero points so
  far." Both `add_point` and `search` check `max_level_ < 0` / `size() == 0`
  to handle the empty-index case.
- `rng_(seed)` — `std::mt19937_64` is a high-quality pseudo-random number
  generator (the "Mersenne Twister"), seeded with `seed` (default `42`).
  Because it's seeded explicitly (rather than from the system clock), **two
  indexes built with the same `seed` and the same input data produce
  identical graphs** — this determinism is checked by
  `tests/test_hnsw.cpp` ([Section 11](#11-teststest_hnswcpp)).

### 8.4 Helper types: `Candidate`, `MaxHeap`, `MinHeap`

These type aliases (declared in the `private:` section, but used throughout)
are the basic currency of the search algorithms:

```cpp
    using Candidate = std::pair<float, uint32_t>;
    // Max-heap by distance: top() = furthest.
    using MaxHeap = std::priority_queue<Candidate>;
    // Min-heap by distance: top() = nearest.
    using MinHeap = std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>>;
```

- **`Candidate`** = `std::pair<float, uint32_t>` = "(distance, node id)".
  Comparing two `Candidate`s with `<` compares the **distance** first
  ([Section 2.8](#28-the-standard-library-containers-used-here)) — so
  `Candidate`s sort by distance automatically.
- **`MaxHeap`** = `std::priority_queue<Candidate>` — by default, a
  `priority_queue`'s `.top()` returns the **largest** element. For
  `Candidate`s, "largest" means "**furthest** away" (largest distance). This
  is used to hold "the best `ef` candidates found so far" — keeping `.top()`
  as the *worst of the kept set* lets the algorithm cheaply check "is this new
  candidate better than my current worst?" and evict the worst with `.pop()`
  if so.
- **`MinHeap`** = same `Candidate` type, but with the comparator flipped via
  `std::greater<Candidate>`, so `.top()` returns the **smallest** distance —
  i.e. the **nearest** not-yet-explored candidate. This is used as the
  "frontier" of nodes still waiting to be explored, always exploring the
  nearest one next.

### 8.5 Helper: `get_data` and `random_level`

```cpp
    const float* get_data(uint32_t id) const {
        return data_.data() + static_cast<size_t>(id) * static_cast<size_t>(dim_);
    }
```

Recall `data_` stores **every point's vector concatenated into one flat
array**: point `0`'s `dim_` numbers, then point `1`'s `dim_` numbers, and so
on ("row-major" layout). `data_.data()` returns a raw pointer to the start of
that whole array. `get_data(id)` does the **pointer arithmetic** to find where
point `id`'s vector begins: skip past `id * dim_` `float`s. The result is a
`const float*` — a pointer to `dim_` consecutive `float`s — exactly what
`Dist::dist(a, b, d)` expects for its `a`/`b` arguments.

```cpp
    int random_level() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = -std::log(dist(rng_)) * mL_;
        return static_cast<int>(r);
    }
```

This is the standard HNSW formula for picking a new point's **top layer**:

1. `dist(rng_)` draws a uniform random number in `(0, 1)`.
2. `-std::log(...)` of a uniform-(0,1) value, via a technique called *inverse
   transform sampling*, produces a number following an **exponential
   distribution** (most outputs are near `0`, with a long tail of rare larger
   values).
3. Multiplying by `mL_` (= `1 / ln(M)`, from the constructor) scales that
   distribution so that, on average, only about `1/M` of points end up at
   layer ≥ 1, only about `1/M²` at layer ≥ 2, and so on.
4. `static_cast<int>(r)` truncates to an integer — the point's assigned top
   layer (`0`, `1`, `2`, ...; `0` is by far the most common).

This is exactly the shape described in
[Section 1](#1-the-big-picture): a few nodes reach high layers (the "highway
system"), and the number shrinks geometrically as you go up.

---

### 8.6 `add_points()` — building the index

```cpp
    // Append n points (row-major, n x dim_) and build their graph connections.
    void add_points(const float* data, int n, bool parallel = false, int num_threads = 1) {
        (void)parallel;
        (void)num_threads;

        size_t start_id = data_.size() / static_cast<size_t>(dim_);
        data_.resize((start_id + static_cast<size_t>(n)) * static_cast<size_t>(dim_));
        std::memcpy(data_.data() + start_id * static_cast<size_t>(dim_), data,
                    static_cast<size_t>(n) * static_cast<size_t>(dim_) * sizeof(float));

        levels_.resize(start_id + static_cast<size_t>(n));
        links_.resize(start_id + static_cast<size_t>(n));
        visited_pool_.resize(start_id + static_cast<size_t>(n));

        for (int i = 0; i < n; ++i) {
            add_point(static_cast<uint32_t>(start_id + static_cast<size_t>(i)));
        }
    }

    // Search-time parameter; takes effect on the next search(), no rebuild.
    void set_ef(int ef_search) { ef_search_ = ef_search; }
```

This is the public entry point for building (or extending) the index. It's
called once from Python with the entire corpus as one big `(N, dim)` array.

- **`(void)parallel; (void)num_threads;`** — a standard C++ idiom meaning "I
  am deliberately not using this parameter; please don't warn me about it."
  These two parameters exist (per the class-level comment in
  [8.1](#81-overview--class-declaration)) so the function's *signature* won't
  need to change when a future phase adds real parallel building — for now
  they're accepted and silently ignored, and the build is always
  single-threaded.
- **`start_id = data_.size() / dim_`** — `data_` already holds
  `start_id * dim_` floats from any *previous* call to `add_points` (it's `0`
  the first time). Dividing by `dim_` recovers "how many points are already in
  the index" — i.e., the id the *first new point* will get.
- **`data_.resize(...)`** — grows the flat vector to also fit the `n` new
  points' `dim_` floats each.
- **`std::memcpy(dest, src, num_bytes)`** — a raw bulk memory copy (much
  faster than copying element-by-element in a loop): copies the caller's `n *
  dim_` floats into the freshly-grown tail of `data_`, starting right after
  the existing points.
- **`levels_.resize(...)`, `links_.resize(...)`, `visited_pool_.resize(...)`**
  — grow the three other per-point bookkeeping structures to the new total
  count. The new entries in `links_` start as empty
  `std::vector<std::vector<uint32_t>>`s (no edges yet) — `add_point` fills
  them in.
- **The final loop** — for each new point, in order, call `add_point(id)`
  with `id = start_id, start_id+1, ..., start_id+n-1`. This sequential loop is
  where **all** of the graph-building logic actually happens — see
  [8.10](#810-add_point--inserting-one-point).

`set_ef(int ef_search)` is a one-line setter for `ef_search_`. The comment
explains why this is interesting at all: `search()`
([8.11](#811-search--the-public-entry-point-for-querying)) reads `ef_search_`
fresh every time it runs, so changing it via `set_ef` immediately affects the
*next* search — there's no index data to rebuild, because `ef_search_` only
controls how wide a beam search to do, not the graph structure itself. This is
exactly the "search param, no rebuild" knob described in
`CUSTOM_HNSW_PLAN.md` §6 / §7.

### 8.7 `search_layer()` — beam search at a single layer

This is the workhorse search routine — both `add_point` (while building) and
`search` (while querying) use it. It implements a **beam search**: explore
outward from a starting node, always expanding the closest not-yet-explored
node, while keeping track of the best `ef` nodes seen so far.

```cpp
    // Beam search at a single layer. Returns up to `ef` nearest-to-`point` ids
    // visited from `entry`, as a max-heap (top = furthest of the kept set).
    MaxHeap search_layer(const float* point, uint32_t entry, int ef, int level,
                          VisitedPool::VisitedList& visited) const {
        MaxHeap top_candidates;
        MinHeap candidate_set;

        float d = Dist::dist(point, get_data(entry), dim_);
        top_candidates.emplace(d, entry);
        candidate_set.emplace(d, entry);
        visited.set_visited(entry);
```

**Parameters**: `point` = the vector we're searching near (a query, or a
new point being inserted); `entry` = node id to start from; `ef` = how many
"best results" to track (the beam width); `level` = which layer's edges
(`links_[...][level]`) to follow; `visited` = this search's
`VisitedPool::VisitedList` (Section 7), shared across all layers of one
search so a node is never processed twice.

Setup: compute `entry`'s distance to `point`, and seed *both* heaps with it —
`top_candidates` (the "best results so far", a max-heap so `.top()` is the
*worst* of the kept set) and `candidate_set` (the "frontier to explore", a
min-heap so `.top()` is the *nearest* unexplored node). Mark `entry` visited.

```cpp
        while (!candidate_set.empty()) {
            Candidate current = candidate_set.top();
            if (current.first > top_candidates.top().first &&
                top_candidates.size() >= static_cast<size_t>(ef)) {
                break;
            }
            candidate_set.pop();
```

Main loop: as long as there's something left to explore, look at the nearest
unexplored node (`candidate_set.top()`, without removing it yet).

**The stopping condition** — `current.first > top_candidates.top().first &&
top_candidates.size() >= ef`: if the *nearest remaining unexplored* node
(`current`) is *already farther* than the *worst node we're currently keeping*
(`top_candidates.top()`), and we already have a full set of `ef` results, then
**nothing left to explore can possibly improve our results** — every other
candidate in the frontier is at least as far as `current`. So: stop early.
Otherwise, `current` might lead somewhere useful — `.pop()` it off the
frontier and expand it.

```cpp
            for (uint32_t e : links_[current.second][static_cast<size_t>(level)]) {
                if (!visited.is_visited(e)) {
                    visited.set_visited(e);
                    float dist_e = Dist::dist(point, get_data(e), dim_);
                    if (top_candidates.size() < static_cast<size_t>(ef) ||
                        dist_e < top_candidates.top().first) {
                        candidate_set.emplace(dist_e, e);
                        top_candidates.emplace(dist_e, e);
                        if (top_candidates.size() > static_cast<size_t>(ef)) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }
        return top_candidates;
    }
```

**Expansion**: `links_[current.second][level]` is `current`'s list of graph
neighbors at this layer (`current.second` is the node id — recall `Candidate =
(distance, id)`). For each neighbor `e`:

- If `e` has already been visited (in this search), skip it.
- Otherwise, mark it visited and compute its distance to `point`.
- **Is `e` worth keeping?** Either we don't have `ef` results yet
  (`top_candidates.size() < ef`), or `e` is closer than our current worst
  (`dist_e < top_candidates.top().first`). If so:
  - Add `e` to `candidate_set` — its neighbors might be worth exploring later.
  - Add `e` to `top_candidates` — it's one of our current best.
  - If that made `top_candidates` exceed `ef` entries, `.pop()` removes the
    *worst* one (guaranteed by the max-heap property: `.top()`/`.pop()` always
    act on the largest-distance entry) — keeping `top_candidates` at exactly
    `ef` from then on.

When the loop ends (frontier exhausted, or the early-stop condition fired),
`top_candidates` holds **up to `ef`** of the closest nodes to `point` reachable
(via unvisited edges) from `entry` at this `level`, as a max-heap (the
*furthest* of these is at `.top()`).

### 8.8 `select_neighbors_heuristic()` — diversity-aware pruning

When a node has more *candidate* neighbors than its degree budget (`M` or
`M_max0_`), something has to decide which ones become real graph edges.
Simply keeping "the `M` closest" tends to produce clusters where a node's
neighbors are all bunched in the same direction — which hurts the graph's
ability to "jump across" to other regions during search. This heuristic
(from the original HNSW paper, by Malkov & Yashunin — the comment's "RMY"/
hnswlib's `getNeighborsByHeuristic2`) instead favors **diverse directions**.

```cpp
    // Robert-Malkov-Yashunin heuristic: prefer candidates not dominated by an
    // already-selected, closer neighbor ("diverse" neighbor set). If there are
    // <= M candidates, keep them all (matches hnswlib's getNeighborsByHeuristic2).
    std::vector<uint32_t> select_neighbors_heuristic(MaxHeap candidates, size_t M) const {
        if (candidates.size() <= M) {
            std::vector<uint32_t> result;
            result.reserve(candidates.size());
            while (!candidates.empty()) {
                result.push_back(candidates.top().second);
                candidates.pop();
            }
            return result;
        }
```

`candidates` is a `MaxHeap` of `(distance to "the point", candidate id)` pairs
— note it's taken **by value** (a copy of the caller's heap; the original is
left untouched). `M` is the degree budget.

**Easy case**: if there aren't even more than `M` candidates, there's nothing
to prune — drain the whole heap into a plain `std::vector<uint32_t>` of ids
(the order doesn't matter here, since *all* candidates are kept regardless).

```cpp
        std::vector<Candidate> sorted;
        sorted.reserve(candidates.size());
        while (!candidates.empty()) {
            sorted.push_back(candidates.top());
            candidates.pop();
        }
        std::reverse(sorted.begin(), sorted.end());  // ascending by distance to `point`

        std::vector<uint32_t> result;
        result.reserve(M);
        for (const auto& [dist_to_point, cand_id] : sorted) {
            if (result.size() >= M) break;
            bool good = true;
            for (uint32_t r : result) {
                if (Dist::dist(get_data(cand_id), get_data(r), dim_) < dist_to_point) {
                    good = false;
                    break;
                }
            }
            if (good) result.push_back(cand_id);
        }
        return result;
    }
```

**Pruning case** (more than `M` candidates):

1. Drain the max-heap into `sorted` — popping a max-heap repeatedly yields
   entries in *descending* distance order (furthest first).
2. `std::reverse(...)` flips `sorted` to *ascending* order — **closest first**
   — as the comment says.
3. Walk `sorted` from closest to farthest. Each entry is unpacked via a
   **structured binding** (`const auto& [dist_to_point, cand_id] : sorted`,
   see [Section 2.8](#28-the-standard-library-containers-used-here)) into
   `dist_to_point` (how far `cand_id` is from "the point" we're choosing
   neighbors *for*) and `cand_id` (the candidate's id).
4. **Stop once `result` has `M` entries.**
5. **The diversity check**: for every neighbor `r` *already* in `result`,
   compute the distance between `cand_id` and `r`. If that's **smaller** than
   `dist_to_point` — i.e., `cand_id` is *closer to an already-chosen neighbor
   `r`* than it is to the point we're connecting — then `cand_id` is
   considered "dominated" by `r`: it doesn't offer a meaningfully different
   direction, so it's skipped (`good = false`).
6. If `cand_id` survives every such check, it's added to `result`.

Because some candidates can get skipped this way, **`result` may end up with
fewer than `M` entries** even when more than `M` candidates were available —
the function comment explicitly notes this matches `hnswlib`'s behavior. The
output is the final set of graph-edge targets for one node at one layer.

### 8.9 `connect_and_prune()` — keeping links (roughly) bidirectional

HNSW edges are meant to be (approximately) symmetric: if node `A` decides node
`B` is one of its neighbors, `B` should generally also list `A` as a neighbor
— otherwise a search arriving at `B` could never "discover" `A` through this
edge. `connect_and_prune` performs that reverse-direction update, including
re-pruning if `B` is already at its degree limit.

```cpp
    // Add `new_id` to `neighbor_id`'s adjacency at `level`, pruning with the
    // heuristic if that exceeds the level's degree cap.
    void connect_and_prune(uint32_t new_id, uint32_t neighbor_id, int level) {
        auto& neighbor_links = links_[neighbor_id][static_cast<size_t>(level)];
        for (uint32_t existing : neighbor_links) {
            if (existing == new_id) return;
        }

        const size_t max_conn = (level == 0) ? static_cast<size_t>(M_max0_) : static_cast<size_t>(M_);
        if (neighbor_links.size() < max_conn) {
            neighbor_links.push_back(new_id);
            return;
        }

        const float* neighbor_point = get_data(neighbor_id);
        MaxHeap candidates;
        candidates.emplace(Dist::dist(neighbor_point, get_data(new_id), dim_), new_id);
        for (uint32_t x : neighbor_links) {
            candidates.emplace(Dist::dist(neighbor_point, get_data(x), dim_), x);
        }
        neighbor_links = select_neighbors_heuristic(candidates, max_conn);
    }
```

- **`auto& neighbor_links = links_[neighbor_id][level];`** — `neighbor_links`
  is a *reference* ([2.7](#27-pointers-and-references)) to `neighbor_id`'s
  actual adjacency list at this layer; any change to `neighbor_links` is a
  change to `links_` itself, not a copy.
- **Dedup check**: if `new_id` is already in `neighbor_links`, do nothing and
  return — no duplicate edges.
- **`max_conn`** — the degree cap for this layer (`M_max0_` at layer 0, `M_`
  elsewhere — same rule as everywhere else).
- **Room available** (`neighbor_links.size() < max_conn`): simplest case —
  just append `new_id` to `neighbor_id`'s list and return.
- **No room** — `neighbor_id` is already at its cap. We must decide which
  `max_conn` neighbors `neighbor_id` keeps, *now including* `new_id` as a
  candidate too:
  - Build a fresh `MaxHeap candidates` containing `(distance from
    neighbor_id to X, X)` for `X = new_id` **and** for every node `X`
    currently in `neighbor_links`.
  - Run it through `select_neighbors_heuristic` (Section 8.8) with budget
    `max_conn`.
  - Overwrite `neighbor_links` with the result.

  Two outcomes are possible here: either `new_id` displaces one of
  `neighbor_id`'s previous neighbors, or the heuristic decides `new_id` itself
  is too "dominated" by `neighbor_id`'s existing neighbors and **doesn't**
  get added at all — the edge `id → neighbor_id` (added by the caller,
  [8.10](#810-add_point--inserting-one-point)) can end up one-directional in
  that case. This is normal/expected in HNSW.

### 8.10 `add_point()` — inserting one point

This is where everything from Sections 8.5–8.9 comes together: given a new
point's id (whose vector is already sitting in `data_`, placed there by
`add_points`), wire it into the graph.

```cpp
    void add_point(uint32_t id) {
        const float* point = get_data(id);
        const int new_level = random_level();
        levels_[id] = new_level;
        links_[id].resize(static_cast<size_t>(new_level) + 1);

        if (max_level_ < 0) {
            entry_point_ = id;
            max_level_ = new_level;
            return;
        }
```

1. `point` = pointer to this point's vector.
2. `new_level = random_level()` — roll the dice (Section 8.5) for which
   layers this point will live on: `0` through `new_level`.
3. Record `levels_[id]` and allocate `links_[id]` with one (currently-empty)
   adjacency list per layer `0..new_level`.
4. **Bootstrap case** — if the index was empty (`max_level_ < 0`, the
   constructor's sentinel), there's nothing to connect to yet: this point
   simply *becomes* the graph (`entry_point_ = id`, `max_level_ = new_level`),
   and we're done.

```cpp
        uint32_t cur_ep = entry_point_;
        const int cur_max_level = max_level_;

        // Greedily descend from the top level down to new_level + 1.
        if (new_level < cur_max_level) {
            float cur_dist = Dist::dist(point, get_data(cur_ep), dim_);
            for (int level = cur_max_level; level > new_level; --level) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (uint32_t e : links_[cur_ep][static_cast<size_t>(level)]) {
                        float d = Dist::dist(point, get_data(e), dim_);
                        if (d < cur_dist) {
                            cur_dist = d;
                            cur_ep = e;
                            changed = true;
                        }
                    }
                }
            }
        }
```

**Phase 1 — greedy descent through layers the new point won't belong to.**
`cur_ep` starts at the graph's current entry point, `cur_max_level` at the
graph's current top layer. If `new_level < cur_max_level`, the new point
doesn't exist on layers `new_level+1 .. cur_max_level` — but we still need a
*good starting position* for the real work at `new_level`. For each such
layer, from the top down:

- Compute `cur_ep`'s distance to `point` (`cur_dist`).
- Repeat: scan all of `cur_ep`'s neighbors at this `level`; if any neighbor
  `e` is *closer* to `point` than `cur_ep` currently is, "move" there
  (`cur_dist = d; cur_ep = e; changed = true`) and scan again from the new
  position. Stop (`changed = false`) once no neighbor improves on the current
  position.

This is a simple **greedy hill-climb** (not a full beam search — `ef`-width
exploration starts in Phase 2) — cheap, and it gets `cur_ep` into roughly the
right neighborhood by the time we reach `new_level`. (If `new_level >=
cur_max_level`, this whole block is skipped — `cur_ep` simply stays at
`entry_point_`.)

```cpp
        // Beam search + connect at each level from min(new_level, cur_max_level) down to 0.
        const int top = std::min(new_level, cur_max_level);
        for (int level = top; level >= 0; --level) {
            auto visited = visited_pool_.get();
            MaxHeap candidates = search_layer(point, cur_ep, ef_construction_, level, visited);
            if (!candidates.empty()) {
                cur_ep = candidates.top().second;
            }

            const size_t max_conn = (level == 0) ? static_cast<size_t>(M_max0_) : static_cast<size_t>(M_);
            std::vector<uint32_t> selected = select_neighbors_heuristic(candidates, max_conn);
            links_[id][static_cast<size_t>(level)] = selected;

            for (uint32_t e : selected) {
                connect_and_prune(id, e, level);
            }
        }
```

**Phase 2 — for every layer the new point actually belongs to** (from
`min(new_level, cur_max_level)` down to `0` — capped by `cur_max_level`
because layers above that don't exist yet to search):

1. Get a fresh `VisitedList` for this layer (`visited_pool_.get()`, Section
   7) — a new "epoch", so nothing from a previous layer's search is treated as
   visited here.
2. `search_layer(point, cur_ep, ef_construction_, level, visited)` — a full
   beam search (Section 8.7) at this layer, with beam width
   `ef_construction_` (the "build-time" beam — typically wider than
   `ef_search_`, for a higher-quality graph). Returns up to `ef_construction_`
   nearby existing nodes.
3. `cur_ep = candidates.top().second` — peek at one of the returned
   candidates (the heap's `.top()`, i.e. the *furthest* of the kept set, is
   the cheapest to read without disturbing the heap) and use it as the
   starting point for *next* layer's search. Since the next layer's
   `search_layer` does its own beam search outward from there, any node from
   this layer's reasonably-close result set is a fine jumping-off point.
4. Compute this layer's degree cap `max_conn` (`M_max0_` at level 0, `M_`
   above).
5. `select_neighbors_heuristic(candidates, max_conn)` (Section 8.8) — turn the
   beam-search results into the actual (diversity-pruned) set of neighbor ids
   for `id` at this layer, store it as `links_[id][level]`.
6. For each selected neighbor `e`, call `connect_and_prune(id, e, level)`
   (Section 8.9) to add the reverse edge `e → id` (with re-pruning if `e` is
   full).

```cpp
        if (new_level > cur_max_level) {
            max_level_ = new_level;
            entry_point_ = id;
        }
    }
```

**Finally**: if this new point reaches *higher* than anything previously in
the graph, it becomes the new global entry point and `max_level_` is raised
to match. (If `new_level <= cur_max_level`, the existing entry point and
`max_level_` are left as-is.)

---

### 8.11 `search()` — the public entry point for querying

This is the function exposed to Python as `index.search(query, k)`. It's
marked `const` — searching never modifies the graph (only the `mutable`
`visited_pool_`, see [Section 2.11](#211-mutable)).

```cpp
    // Single query, single thread. Writes k ids (dataset row indices) into out_ids,
    // padding with -1 if the index holds fewer than k points.
    void search(const float* query, int k, int* out_ids) const {
        for (int i = 0; i < k; ++i) out_ids[i] = -1;
        if (size() == 0) return;

        uint32_t cur_ep = entry_point_;
        float cur_dist = Dist::dist(query, get_data(cur_ep), dim_);
        for (int level = max_level_; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (uint32_t e : links_[cur_ep][level]) {
                    float d = Dist::dist(query, get_data(e), dim_);
                    if (d < cur_dist) {
                        cur_dist = d;
                        cur_ep = e;
                        changed = true;
                    }
                }
            }
        }
```

1. **Output convention**: fill all `k` slots of the caller-provided `out_ids`
   array with `-1` first — a sentinel meaning "no result here" — then handle
   the empty-index case (`size() == 0`) by returning immediately (all `-1`s).
   This matters if the index holds *fewer than `k`* points total.
2. **Greedy descent through the upper layers** (`for level = max_level_ downto
   1`): exactly the same hill-climbing pattern as Phase 1 of `add_point`
   ([8.10](#810-add_point--inserting-one-point)) — start at `entry_point_`,
   and at each layer from the top down to layer `1`, repeatedly jump to a
   strictly-closer neighbor (scanning `links_[cur_ep][level]`) until no
   neighbor improves on the current position, then move down to the next
   layer carrying `cur_ep`/`cur_dist` forward. This is the "drive down the
   highway, take progressively more local exits" step from
   [Section 1](#1-the-big-picture) — cheap, since each layer has few nodes and
   few edges.

```cpp
        int ef = std::max(ef_search_, k);
        auto visited = visited_pool_.get();
        MaxHeap top_candidates = search_layer(query, cur_ep, ef, 0, visited);

        while (top_candidates.size() > static_cast<size_t>(k)) {
            top_candidates.pop();
        }

        std::vector<Candidate> result;
        result.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            result.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::sort(result.begin(), result.end());

        for (size_t i = 0; i < result.size(); ++i) {
            out_ids[i] = static_cast<int>(result[i].second);
        }
    }
```

3. **Beam search at layer 0**: `ef = max(ef_search_, k)` — the beam must be at
   least `k` wide, since we can't return more results than we collect (if a
   user sets `ef_search_` smaller than `k`, `k` wins). Get a fresh
   `VisitedList` (a new "epoch", Section 7) and run `search_layer`
   ([8.7](#87-search_layer--beam-search-at-a-single-layer)) at `level=0` with
   this `ef`, starting from the `cur_ep` found by the descent above. This
   returns up to `ef` candidates as a max-heap (furthest of the kept set at
   `.top()`).
4. **Trim down to exactly `k`**: repeatedly `.pop()` (removes the furthest)
   while there are more than `k` candidates — handles the case `ef > k`.
5. **Sort by distance**: draining a max-heap with `.pop()` yields
   *descending*-distance order; instead, the code copies everything into a
   `std::vector<Candidate> result` (still in descending order at this point)
   and then calls `std::sort` — `Candidate = std::pair<float, uint32_t>`'s
   default ordering compares `.first` (distance) first, so after sorting,
   `result` is in **ascending** distance order: closest match first.
6. **Write the output**: copy each `result[i].second` (the node id, i.e. the
   row index into the original dataset) into `out_ids[i]`. If fewer than `k`
   points existed in total, `result.size() < k`, and the remaining `out_ids`
   slots keep their initial `-1`.

### 8.12 `size()`, `dim()`, and `index_memory_bytes()`

```cpp
    size_t size() const { return levels_.size(); }
    int dim() const { return dim_; }
```

Trivial accessors: `size()` returns how many points have been added
(`levels_` has exactly one entry per point); `dim()` returns the configured
vector dimensionality.

```cpp
    // Rough resident-size estimate: data + per-level neighbor lists + bookkeeping.
    size_t index_memory_bytes() const {
        size_t total = sizeof(*this);
        total += data_.capacity() * sizeof(float);
        total += levels_.capacity() * sizeof(int);
        total += links_.capacity() * sizeof(std::vector<std::vector<uint32_t>>);
        for (const auto& per_level : links_) {
            total += per_level.capacity() * sizeof(std::vector<uint32_t>);
            for (const auto& neighbors : per_level) {
                total += neighbors.capacity() * sizeof(uint32_t);
            }
        }
        total += visited_pool_.capacity_bytes();
        return total;
    }
```

This estimates the index's total in-memory footprint (used by the Python
adapter as an `index_memory_bytes` metric). It walks every piece of storage
the index owns and sums it up:

- **`sizeof(*this)`** — the size of the fixed-size part of the `HnswIndex`
  object itself: the scalar fields (`dim_`, `M_`, ..., `rng_`) plus, for each
  `std::vector` member, just its small fixed-size "header" (typically a
  pointer + size + capacity — a few words), **not** the heap memory it points
  to (that's accounted for separately below).
- **`data_.capacity() * sizeof(float)`** — the actual vector data storage.
  `.capacity()` (allocated) rather than `.size()` (used) is deliberately
  chosen, because `.capacity()` reflects what's *actually resident in RAM*
  right now (a `std::vector` often over-allocates when it grows).
- **`levels_.capacity() * sizeof(int)`** — the per-point level array.
- **`links_.capacity() * sizeof(std::vector<std::vector<uint32_t>>)`** — the
  header storage for the *outer* vector (one "per-point adjacency lists"
  header per point slot).
- **The nested loop** — for every point's `per_level` (= `links_[id]`, a
  `vector<vector<uint32_t>>`), add its header storage; then for every layer's
  `neighbors` (= `links_[id][level]`, a `vector<uint32_t>`), add the actual
  neighbor-id storage (`.capacity() * sizeof(uint32_t)`). Together these two
  lines walk the *entire graph* and sum up every adjacency list's memory.
- **`visited_pool_.capacity_bytes()`** — the tag array from Section 7.

The function comment is honest that this is a *rough* estimate (it doesn't
model allocator overhead/fragmentation), but it captures all the
order-of-magnitude-relevant pieces: raw vectors, the graph itself, and
search-time scratch space.

### 8.13 `save()` and `load()`

These write/read the entire index to/from a binary file — used by the Python
adapter to measure **on-disk index size** (write to a temp file, check its
size) and, more generally, to persist a built index.

First, two constants used by both:

```cpp
    static constexpr uint32_t kMagic = 0x57534E48;  // "HNSW"
    static constexpr uint32_t kVersion = 1;
```

`kMagic` is a "magic number" — a fixed value written at the start of every
file this code produces, purely so `load()` can sanity-check "is this even an
HNSW index file?" (`0x57534E48` is the hex encoding of the ASCII bytes for
`"HNSW"`). `kVersion` lets future code detect/reject files written by an
incompatible older or newer version of this format.

```cpp
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("hnsw: failed to open file for writing: " + path);

        const uint32_t magic = kMagic;
        const uint32_t version = kVersion;
        const uint64_t n = size();

        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        out.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
        out.write(reinterpret_cast<const char*>(&M_max0_), sizeof(M_max0_));
        out.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(ef_construction_));
        out.write(reinterpret_cast<const char*>(&ef_search_), sizeof(ef_search_));
        out.write(reinterpret_cast<const char*>(&mL_), sizeof(mL_));
        out.write(reinterpret_cast<const char*>(&entry_point_), sizeof(entry_point_));
        out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));

        out.write(reinterpret_cast<const char*>(data_.data()),
                  static_cast<std::streamsize>(data_.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(levels_.data()),
                  static_cast<std::streamsize>(levels_.size() * sizeof(int)));

        for (uint64_t id = 0; id < n; ++id) {
            for (int level = 0; level <= levels_[id]; ++level) {
                const auto& neighbors = links_[id][static_cast<size_t>(level)];
                const uint32_t cnt = static_cast<uint32_t>(neighbors.size());
                out.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
                if (cnt > 0) {
                    out.write(reinterpret_cast<const char*>(neighbors.data()),
                              static_cast<std::streamsize>(cnt * sizeof(uint32_t)));
                }
            }
        }
        if (!out) throw std::runtime_error("hnsw: write error: " + path);
    }
```

- **`std::ofstream out(path, std::ios::binary)`** — opens `path` for writing
  in *binary* mode (as opposed to text mode, which on some platforms would
  translate line-ending bytes — irrelevant/harmful for raw binary data).
  `if (!out)` checks whether opening failed (bad path, permissions, etc.); if
  so, `throw std::runtime_error(...)` raises a C++ exception with a
  descriptive message — pybind11 automatically turns this into a Python
  exception on the calling side.
- **`reinterpret_cast<const char*>(&magic)`** — "give me a pointer to
  `magic`'s raw bytes" ([Section 2.9](#29-casts)). `ofstream::write` only
  understands `const char*` + a byte count, so every value written needs this
  cast, paired with `sizeof(...)` for the byte count.
- **The header** is written field by field, always in the same order: magic,
  version, then every scalar configuration field (`dim_`, `M_`, `M_max0_`,
  `ef_construction_`, `ef_search_`, `mL_`, `entry_point_`, `max_level_`), then
  the point count `n` (`= size()`).
- **Bulk arrays**: `data_` (all vectors, concatenated) and `levels_` are each
  written with a *single* `.write()` call covering their entire contiguous
  storage (`.data()` pointer + `size() * sizeof(element)` bytes) —
  `std::streamsize` is just the integer type `ofstream::write` expects for a
  byte count.
- **The graph**: for every point `id` (`0..n-1`) and every layer it belongs to
  (`0..levels_[id]`), write a 4-byte neighbor count `cnt`, then (if `cnt > 0`)
  that many neighbor ids in one `.write()`. The `if (cnt > 0)` guard just
  avoids an unnecessary call for empty adjacency lists.
- **Final check**: `if (!out) throw ...` after *all* writes — cheaper than
  checking the stream's error flag after every single `.write()`, and still
  catches any I/O failure that happened along the way (e.g. disk full).

```cpp
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("hnsw: failed to open file for reading: " + path);

        uint32_t magic = 0, version = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != kMagic || version != kVersion) {
            throw std::runtime_error("hnsw: bad file header: " + path);
        }

        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
        in.read(reinterpret_cast<char*>(&M_), sizeof(M_));
        in.read(reinterpret_cast<char*>(&M_max0_), sizeof(M_max0_));
        in.read(reinterpret_cast<char*>(&ef_construction_), sizeof(ef_construction_));
        in.read(reinterpret_cast<char*>(&ef_search_), sizeof(ef_search_));
        in.read(reinterpret_cast<char*>(&mL_), sizeof(mL_));
        in.read(reinterpret_cast<char*>(&entry_point_), sizeof(entry_point_));
        in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
        in.read(reinterpret_cast<char*>(&n), sizeof(n));

        data_.resize(n * static_cast<uint64_t>(dim_));
        in.read(reinterpret_cast<char*>(data_.data()),
                static_cast<std::streamsize>(data_.size() * sizeof(float)));

        levels_.resize(n);
        in.read(reinterpret_cast<char*>(levels_.data()),
                static_cast<std::streamsize>(levels_.size() * sizeof(int)));

        links_.assign(n, {});
        for (uint64_t id = 0; id < n; ++id) {
            links_[id].resize(static_cast<size_t>(levels_[id] + 1));
            for (int level = 0; level <= levels_[id]; ++level) {
                uint32_t cnt = 0;
                in.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
                auto& neighbors = links_[id][static_cast<size_t>(level)];
                neighbors.resize(cnt);
                if (cnt > 0) {
                    in.read(reinterpret_cast<char*>(neighbors.data()),
                            static_cast<std::streamsize>(cnt * sizeof(uint32_t)));
                }
            }
        }

        visited_pool_.resize(n);

        if (!in) throw std::runtime_error("hnsw: read error: " + path);
    }
```

`load()` is the exact mirror of `save()`, reading back the same fields in the
same order — note `load()` is **not** `const`: it's overwriting this object's
own state.

- Open the file for binary reading; fail loudly if it can't be opened.
- Read `magic` and `version` and **validate** them against `kMagic`/`kVersion`
  — if either doesn't match, throw immediately. This rejects, e.g., a
  non-HNSW file or one written by an incompatible version, before any further
  (potentially nonsensical) reads happen.
- Read all the scalar fields directly into the corresponding member variables
  (`&dim_`, `&M_`, etc.) and into a local `n` for the point count.
- `data_.resize(n * dim_)` then one bulk read of that many floats; similarly
  `levels_.resize(n)` then one bulk read of `n` ints.
- **Rebuild the graph**: `links_.assign(n, {})` resets `links_` to `n` empty
  `vector<vector<uint32_t>>` entries (`{}` = empty vector). For each point
  `id`, resize its outer list to `levels_[id] + 1` layers, then for each
  layer read the 4-byte count `cnt`, `resize` that layer's neighbor list to
  `cnt`, and (if `cnt > 0`) bulk-read `cnt` ids into it.
- `visited_pool_.resize(n)` — re-create the tag array sized to the loaded
  point count. (Its *contents* are pure search-time scratch space — Section 7
  — so they don't need to be saved/restored, only correctly sized.)
- Final `if (!in) throw ...` — same end-of-operation error check as `save()`.

Together, `save`/`load` are what let
`HnswScalarAdapter.index_size_bytes()` (Python side) measure on-disk size by
writing to a temp file and calling `os.path.getsize()` on it, and what
`tests/test_hnsw.cpp`'s "save/load roundtrip" check
([Section 11](#11-teststest_hnswcpp)) verifies produces a functionally
identical index.

This completes the walkthrough of `HnswIndex<Dist>` — every public method
(`add_points`, `set_ef`, `search`, `size`, `dim`, `index_memory_bytes`,
`save`, `load`) and every private helper (`get_data`, `random_level`,
`search_layer`, `select_neighbors_heuristic`, `connect_and_prune`,
`add_point`) has now been covered.

---

## 9. `src/bindings.cpp`

Everything so far has been "pure C++" with no idea that Python exists. This
file is the **bridge**: it takes the templated `HnswIndex<Dist>` class
(Section 8), instantiates it for each of the four `Dist` types (Sections 5–6),
and exposes each instantiation as a Python class inside the compiled
`_vdbhnsw` module — using a library called **pybind11**.

```cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "hnsw/distances_scalar.hpp"
#include "hnsw/distances_simd.hpp"
#include "hnsw/hnsw_index.hpp"

namespace py = pybind11;
```

- **`pybind11/pybind11.h`** — the core of pybind11: lets C++ types/functions be
  registered as Python classes/functions.
- **`pybind11/numpy.h`** — adds `py::array_t<...>`, a type representing a
  NumPy array, with access to its shape/dtype/raw data pointer.
- **`pybind11/stl.h`** — adds automatic conversions between C++ standard
  containers (`std::vector<int>`, `std::string`, ...) and their Python
  equivalents (`list`, `str`, ...) wherever they appear as function
  arguments/return values.
- **`<stdexcept>`** — `std::runtime_error`, used for input-validation errors
  below (pybind11 automatically converts an uncaught C++ exception into a
  Python exception of a corresponding type).
- The three `#include "hnsw/...hpp"` lines pull in everything from Sections
  5, 6, and 8 — this is the **only** `.cpp` file that brings the whole project
  together.
- **`namespace py = pybind11;`** — a *namespace alias*: lets the rest of the
  file write the shorter `py::` instead of `pybind11::`.

### 9.1 `bind_index<Dist>` — registering one class

```cpp
namespace {

template <class Dist>
void bind_index(py::module_& m, const char* name) {
    using Index = hnsw::HnswIndex<Dist>;

    py::class_<Index>(m, name)
        .def(py::init<int, int, int, uint64_t>(), py::arg("dim"), py::arg("M"),
             py::arg("ef_construction"), py::arg("seed") = 42)
```

- **`namespace { ... }`** — an *unnamed* (anonymous) namespace. Anything
  inside it is only visible within this `.cpp` file — the C++ equivalent of
  "private to this module" (so other compilation units can't accidentally
  link against `bind_index` or collide names with it). Since `bindings.cpp` is
  the only file that needs these helpers, everything here is wrapped in one.
- **`template <class Dist> void bind_index(py::module_& m, const char* name)`**
  — a template function ([Section 2.5](#25-templates--generic-typesfunctions)):
  given a distance-functor type `Dist` and the desired Python class name
  (e.g. `"HnswScalarL2"`), registers a fully-wired Python class for
  `HnswIndex<Dist>` inside module `m`. Called once per `(variant, metric)`
  combination at the bottom of this file.
- **`using Index = hnsw::HnswIndex<Dist>;`** — a local shorthand so the rest of
  the function can write `Index` instead of the longer
  `hnsw::HnswIndex<Dist>`.
- **`py::class_<Index>(m, name)`** — declares a new Python class named `name`
  in module `m`, backed by the C++ type `Index`. Everything chained off this
  with `.def(...)` adds one method/constructor to that Python class.
- **`.def(py::init<int, int, int, uint64_t>(), py::arg("dim"), py::arg("M"),
  py::arg("ef_construction"), py::arg("seed") = 42)`** — exposes `Index`'s
  constructor ([8.3](#83-constructor)) as Python's `__init__`. `py::init<int,
  int, int, uint64_t>()` declares the constructor's parameter types in order.
  `py::arg("dim")`, `py::arg("M")`, etc. give those positions Python-visible
  *names*, so callers can use keyword arguments —
  `HnswScalarL2(dim=128, M=16, ef_construction=200, seed=42)` — and
  `py::arg("seed") = 42` gives `seed` a default value (mirroring the C++
  default).

### 9.2 `add_points` — wrapping a NumPy array as a raw pointer

```cpp
        .def(
            "add_points",
            [](Index& self, py::array_t<float, py::array::c_style | py::array::forcecast> data,
               bool parallel, int num_threads) {
                auto buf = data.request();
                if (buf.ndim != 2) throw std::runtime_error("add_points: data must be 2D (n, dim)");
                if (static_cast<int>(buf.shape[1]) != self.dim()) {
                    throw std::runtime_error("add_points: dim mismatch");
                }
                const int n = static_cast<int>(buf.shape[0]);
                const float* ptr = static_cast<const float*>(buf.ptr);
                py::gil_scoped_release release;
                self.add_points(ptr, n, parallel, num_threads);
            },
            py::arg("data"), py::arg("parallel") = false, py::arg("num_threads") = 1)
```

`HnswIndex::add_points` ([8.6](#86-add_points--building-the-index)) takes a
raw `const float* data, int n` — but a Python caller passes a NumPy array.
This `.def("add_points", ...)` provides a **lambda** (an anonymous, inline
function — `[](...) { ... }`) that bridges the two, instead of directly
exposing the C++ method:

- **`py::array_t<float, py::array::c_style | py::array::forcecast>`** — the
  parameter type for `data`: "a NumPy array of `float32`". The two flags:
  `c_style` requires C-contiguous (row-major) memory layout — the layout
  `get_data()`/pointer-arithmetic in `hnsw_index.hpp` assumes; `forcecast`
  tells pybind11 "if the array passed in doesn't already satisfy that (wrong
  dtype, non-contiguous, etc.), make a converted/copied array that does,
  rather than raising an error." Together: callers can pass almost any
  array-like and get a clean `(n, dim)` `float32` C-contiguous array here.
- **`data.request()`** — returns a `py::buffer_info` struct (`buf`) describing
  the array: its shape, strides, and a raw `void*` to its data.
- **Validation**: `buf.ndim != 2` checks the array is 2-D
  (`(n_points, dim)`); `buf.shape[1] != self.dim()` checks the second
  dimension matches this index's configured dimensionality. Either failure
  `throw`s a `std::runtime_error`, which pybind11 turns into a Python
  exception — much friendlier than undefined behavior from reading the wrong
  amount of memory.
- **`const int n = buf.shape[0]`**, **`const float* ptr =
  static_cast<const float*>(buf.ptr)`** — extract the point count and a raw
  pointer to the array's underlying `float` data.
- **`py::gil_scoped_release release;`** — releases Python's **Global
  Interpreter Lock (GIL)** for the rest of this scope. Background: normally
  only one thread can execute Python bytecode at a time (the GIL enforces
  this); `add_points` can take a long time for a large index, and it never
  touches any Python objects while running, so releasing the GIL lets *other*
  Python threads make progress concurrently. `release` is a local variable
  whose **constructor releases the GIL** and whose **destructor re-acquires
  it** when `release` goes out of scope — this "do/undo via
  constructor/destructor" pattern is called **RAII** (Resource Acquisition Is
  Initialization), a core C++ idiom for "guaranteed cleanup."
- **`self.add_points(ptr, n, parallel, num_threads);`** — finally, the real
  C++ call.
- **`py::arg("data"), py::arg("parallel") = false, py::arg("num_threads") = 1`**
  — names and defaults for the Python signature:
  `add_points(data, parallel=False, num_threads=1)`.

### 9.3 Simple passthroughs: `set_ef`, `size`, `dim`, `index_memory_bytes`, `save`, `load`

```cpp
        .def("set_ef", &Index::set_ef, py::arg("ef_search"))
```
...and, further down:
```cpp
        .def("size", &Index::size)
        .def("dim", &Index::dim)
        .def("index_memory_bytes", &Index::index_memory_bytes)
        .def("save", &Index::save, py::arg("path"))
        .def("load", &Index::load, py::arg("path"));
```

These methods (Sections [8.6](#86-add_points--building-the-index),
[8.12](#812-size-dim-and-index_memory_bytes),
[8.13](#813-save-and-load)) take/return only simple types (`int`, `size_t`,
`std::string`, `void`) that pybind11 already knows how to convert
automatically — so no lambda wrapper is needed. `&Index::set_ef` etc. are
**pointers to member functions** — "a reference to this specific method,
to be invoked on whatever C++ object backs the Python instance `self`
calling it." `py::arg("ef_search")` / `py::arg("path")` just give the single
parameter a name for Python keyword-argument use
(`index.set_ef(ef_search=100)`, `index.save(path="...")`).

### 9.4 `search` — wrapping a NumPy array, returning a list

```cpp
        .def(
            "search",
            [](const Index& self, py::array_t<float, py::array::c_style | py::array::forcecast> query,
               int k) {
                auto buf = query.request();
                if (buf.ndim != 1 || static_cast<int>(buf.shape[0]) != self.dim()) {
                    throw std::runtime_error("search: query must be 1D with length == dim");
                }
                std::vector<int> out(static_cast<size_t>(k));
                const float* ptr = static_cast<const float*>(buf.ptr);
                {
                    py::gil_scoped_release release;
                    self.search(ptr, k, out.data());
                }
                return out;
            },
            py::arg("query"), py::arg("k"))
```

The same pattern as `add_points`, mirrored for `search`
([8.11](#811-search--the-public-entry-point-for-querying)):

- `query` is validated as **1-D** (`buf.ndim != 1`) with length exactly
  `self.dim()` — a single query vector (as opposed to `add_points`'s 2-D
  batch).
- `std::vector<int> out(k)` — allocates the output buffer; pybind11/stl.h will
  convert this to a Python `list[int]` automatically on return.
- **The extra `{ }` braces** around `py::gil_scoped_release release; ...`
  create a *nested scope*: the GIL is released only for the duration of the
  actual `self.search(...)` call, then automatically **re-acquired** the
  moment `release` is destroyed at the closing `}` — *before* `return out;`
  runs. This matters because converting `out` (a C++ `std::vector<int>`) into
  a Python `list` to return requires the GIL (it allocates Python objects).
- `return out;` — pybind11/stl.h converts `std::vector<int>` → `list[int]`.

### 9.5 The module entry point

```cpp
}  // namespace

PYBIND11_MODULE(_vdbhnsw, m) {
    m.doc() = "Custom HNSW (vdbbench native extension)";

    bind_index<hnsw::ScalarL2>(m, "HnswScalarL2");
    bind_index<hnsw::ScalarIP>(m, "HnswScalarIP");

    // Phase B: SIMD distance kernels, still built serially (no OpenMP yet).
    bind_index<hnsw::SimdL2>(m, "HnswFastL2");
    bind_index<hnsw::SimdIP>(m, "HnswFastIP");
}
```

`PYBIND11_MODULE(_vdbhnsw, m) { ... }` is a macro that expands into the actual
C function Python's import machinery looks for when you `import _vdbhnsw`.
The first argument, **`_vdbhnsw`, must exactly match** the module name CMake
builds (`pybind11_add_module(_vdbhnsw src/bindings.cpp)` in
[Section 4](#4-cmakeliststxt)) — this is the link between the build system and
the importable module name. `m` is the module object under construction;
`m.doc() = "..."` sets its docstring (`_vdbhnsw.__doc__` in Python).

The four `bind_index<...>(m, "...")` calls are the entire payoff of the
template design from [Section 2.5](#25-templates--generic-typesfunctions):
**one** definition of `bind_index` (and, transitively, of `HnswIndex<Dist>`
and `search_layer`/`add_point`/etc.) gets compiled into **four** independent,
fully-specialized classes — `HnswScalarL2`, `HnswScalarIP`, `HnswFastL2`,
`HnswFastIP` — each with `Dist::dist` inlined directly into the algorithm's
hot loops, with zero indirection or runtime dispatch overhead.

The "Phase B" comment is a pointer back to `CUSTOM_HNSW_PLAN.md`: "Fast" here
refers only to the **SIMD distance kernel** (Section 6) — at this point, the
*build* for `HnswFastL2`/`HnswFastIP` is still single-threaded, same as the
scalar versions. A later phase ("Phase C") is planned to add an OpenMP
parallel build path specifically for these two.

---

## 10. `tests/test_distances.cpp`

This is a **standalone program** (it has its own `main()` — it's not part of
the Python module) that checks the SIMD distance functions
(`SimdL2`/`SimdIP`, Section 6) compute essentially the same results as the
plain-loop versions (`ScalarL2`/`ScalarIP`, Section 5). It's built only when
`BUILD_TESTS=ON` ([Section 4](#4-cmakeliststxt)):

```bash
cmake -S native/hnsw -B native/hnsw/build-test -DBUILD_TESTS=ON
cmake --build native/hnsw/build-test
./native/hnsw/build-test/test_distances
```

```cpp
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hnsw/distances_scalar.hpp"
#include "hnsw/distances_simd.hpp"

using namespace hnsw;
```

`using namespace hnsw;` is a **using directive**: it brings every name from
`namespace hnsw { ... }` (Sections 5–6) into this file's scope unqualified —
so the code can write `ScalarL2` instead of `hnsw::ScalarL2`. This is
convenient for a small standalone program like this; in larger
projects/headers it's generally avoided because it risks name collisions
(not a concern in a single self-contained `.cpp` file like this).

### 10.1 Test infrastructure: `check`, `random_vectors`, `l2_normalize`

```cpp
namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        std::printf("  PASS: %s\n", msg);
        ++g_passed;
    } else {
        std::printf("  FAIL: %s\n", msg);
        ++g_failed;
    }
}
```

There's no external testing framework here (no equivalent of Python's
`pytest`) — just two global counters (`g_passed`, `g_failed`, made
file-private by the anonymous namespace, [9.1](#91-bind_indexdist--registering-one-class))
and a `check(condition, message)` helper that prints `PASS`/`FAIL` plus the
message, and bumps the corresponding counter. `main()` (Section 10.3) reports
the final tally and uses it as the process's success/failure signal.

```cpp
std::vector<float> random_vectors(int n, int dim, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(static_cast<size_t>(n) * dim);
    for (auto& x : v) x = dist(gen);
    return v;
}

void l2_normalize(std::vector<float>& data, int n, int dim) {
    for (int i = 0; i < n; ++i) {
        float* row = &data[static_cast<size_t>(i) * dim];
        float norm = 0.0f;
        for (int d = 0; d < dim; ++d) norm += row[d] * row[d];
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (int d = 0; d < dim; ++d) row[d] /= norm;
        }
    }
}
```

- **`random_vectors(n, dim, seed)`** — returns a flat `std::vector<float>` of
  `n * dim` random values in `[-1, 1]`, generated with `std::mt19937` (a
  Mersenne Twister RNG, same family as `std::mt19937_64` in
  [8.3](#83-constructor)) seeded with `seed`. `for (auto& x : v) x =
  dist(gen);` iterates over every element *by reference* (`auto&`) and
  overwrites it with a fresh random draw. Because the seed is fixed, calling
  this with the same `(n, dim, seed)` always produces the same data — needed
  for reproducible tests.
- **`l2_normalize(data, n, dim)`** — for each of the `n` vectors (each `dim`
  consecutive floats, accessed via `&data[i * dim]` — a pointer to the start
  of row `i`), compute its Euclidean length (`norm = sqrt(sum of squares)`)
  and divide every component by it, turning it into a **unit vector** (length
  1). This is exactly the normalization the broader harness applies before
  using the `IP`/cosine distance kernels (Sections 5 and 6's "For cosine,
  vectors are L2-normalized centrally..." comments) — so this test reproduces
  that condition for `SimdIP`/`ScalarIP`.

### 10.2 `max_relative_diff` and `run_dim`

```cpp
// Max over `pairs` of |simd - scalar| / max(|scalar|, |simd|, 1.0).
//
// The floor of 1.0 matters for IP/cosine: with L2-normalized vectors the
// true inner product concentrates near 0 in high dimensions (two random unit
// vectors in R^1024 are nearly orthogonal), so a plain relative error
// |simd-scalar|/|scalar| blows up even though the *absolute* difference is a
// tiny float-rounding artifact (~1e-4). Cosine values are bounded to [-1,1],
// so the floor turns this into an absolute-error check in that regime while
// still giving a true relative check for L2, whose squared distances are
// bounded away from zero for distinct random vectors.
template <class Scalar, class Simd>
float max_relative_diff(const std::vector<float>& data, int n, int dim,
                         const std::vector<float>& queries, int nq) {
    float max_rel = 0.0f;
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = &queries[static_cast<size_t>(qi) * dim];
        for (int i = 0; i < n; ++i) {
            const float* p = &data[static_cast<size_t>(i) * dim];
            float ds = Scalar::dist(q, p, dim);
            float dv = Simd::dist(q, p, dim);
            float denom = std::max({std::fabs(ds), std::fabs(dv), 1.0f});
            float rel = std::fabs(dv - ds) / denom;
            max_rel = std::max(max_rel, rel);
        }
    }
    return max_rel;
}
```

`max_relative_diff<Scalar, Simd>` is itself a **template**
([Section 2.5](#25-templates--generic-typesfunctions)) over *which pair* of
distance types to compare (`<ScalarL2, SimdL2>` or `<ScalarIP, SimdIP>`). For
every `(query, data point)` pair, it computes the distance both ways (`ds` =
scalar, `dv` = SIMD) and measures how different they are, relative to their
magnitude — then returns the **worst** (maximum) such difference across all
pairs.

The lengthy comment explains a subtlety: why divide by `max(|ds|, |dv|, 1.0)`
instead of just `|ds|`? For the IP/cosine kernels on **L2-normalized**
vectors, the true inner product of two random unit vectors in a
high-dimensional space is typically *very close to 0* (random directions in
high dimensions are nearly perpendicular). A relative error formula
`|dv-ds|/|ds|` would then divide a tiny floating-point rounding difference
(~`1e-4`) by an even tinier `|ds|`, producing a huge "relative error" that
doesn't actually reflect a real problem. Flooring the denominator at `1.0`
means: when both `ds` and `dv` are small (cosine values are always in
`[-1,1]`), the check effectively becomes "is the *absolute* difference small?"
— while for L2 (whose squared distances *aren't* close to zero for distinct
random vectors), the floor rarely kicks in and the check remains a true
*relative* one.

```cpp
// Runs the L2 and IP comparisons for one dimension, printing the max
// relative diff for each and checking it against `tol`.
void run_dim(int dim, float tol) {
    const int n = 200;
    const int nq = 50;

    std::printf("-- dim=%d --\n", dim);

    auto data_l2 = random_vectors(n, dim, /*seed=*/100 + static_cast<uint32_t>(dim));
    auto queries_l2 = random_vectors(nq, dim, /*seed=*/200 + static_cast<uint32_t>(dim));
    float rel_l2 = max_relative_diff<ScalarL2, SimdL2>(data_l2, n, dim, queries_l2, nq);
    std::printf("  L2: max relative diff = %.3e\n", rel_l2);
    char msg_l2[128];
    std::snprintf(msg_l2, sizeof(msg_l2), "dim=%d SimdL2 vs ScalarL2 relative diff < %.0e", dim,
                  static_cast<double>(tol));
    check(rel_l2 < tol, msg_l2);

    // IP / cosine: harness L2-normalizes vectors centrally before handing
    // them to the index, so exercise IP on normalized vectors too.
    auto data_ip = random_vectors(n, dim, /*seed=*/300 + static_cast<uint32_t>(dim));
    auto queries_ip = random_vectors(nq, dim, /*seed=*/400 + static_cast<uint32_t>(dim));
    l2_normalize(data_ip, n, dim);
    l2_normalize(queries_ip, nq, dim);
    float rel_ip = max_relative_diff<ScalarIP, SimdIP>(data_ip, n, dim, queries_ip, nq);
    std::printf("  IP: max relative diff = %.3e\n", rel_ip);
    char msg_ip[128];
    std::snprintf(msg_ip, sizeof(msg_ip), "dim=%d SimdIP vs ScalarIP relative diff < %.0e", dim,
                  static_cast<double>(tol));
    check(rel_ip < tol, msg_ip);
}

}  // namespace
```

`run_dim(dim, tol)` runs both checks (L2 and IP) for one specific
dimensionality:

- **L2 check**: generate `n=200` data vectors and `nq=50` query vectors
  (seeded deterministically from `dim`, so every dimension gets its own
  reproducible dataset), compute `max_relative_diff<ScalarL2, SimdL2>`, print
  it, and `check(rel_l2 < tol, ...)`.
- **IP check**: generate a *separate* set of random vectors (different
  seeds), **L2-normalize** both data and queries (per the comment, mirroring
  real usage where the harness normalizes for cosine), compute
  `max_relative_diff<ScalarIP, SimdIP>`, print, and check.
- **`char msg_l2[128]; std::snprintf(msg_l2, sizeof(msg_l2), "...", dim,
  (double)tol);`** — builds a formatted message string for `check()`.
  `snprintf` is the "safe" C-style string-formatting function: it writes at
  most `sizeof(msg_l2)` bytes (including the terminator) into `msg_l2`,
  preventing buffer overflows even if the formatted text would otherwise be
  longer.

### 10.3 `main()` — the dimension matrix

```cpp
int main() {
    std::printf("SimdF::size() = %d (vector width in floats)\n\n", SimdF::size());

    const float tol = 1e-3f;

    // 1, 7: smaller than the vector width, pure remainder path.
    // 8, 16: exactly one/two full vector widths (Vec8f), no remainder.
    // 100: 12*8 + 4 -- the canonical "not a multiple of 8 or 16" case
    //      (glove-100 dimensionality). Most important case.
    // 128, 256: SIFT-like, multiples of both 8 and 16.
    // 784: fashion-mnist, 784 = 98*8 (multiple of 8, not of 16).
    // 1024: bge-m3, multiple of both.
    const int dims[] = {1, 7, 8, 16, 100, 128, 256, 784, 1024};

    for (int dim : dims) {
        run_dim(dim, tol);
    }

    std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
```

- Prints `SimdF::size()` — the SIMD vector width in floats (8 for `Vec8f`/AVX2,
  16 for `Vec16f`/AVX-512 — whichever was selected at compile time, Section 6)
  — so the test output records which path is actually being exercised on this
  machine.
- `tol = 1e-3f` (`0.001`) — the shared tolerance for every dimension's checks.
- **`dims[]`** — a hand-picked list of dimensionalities, each chosen for what
  it exercises relative to the SIMD width `W` (8 or 16), as the comment spells
  out:
  - `1, 7` — smaller than `W`: the *entire* computation goes through the
    `load_partial`/remainder path (Section 6) — no full-width SIMD iterations
    at all.
  - `8, 16` — exactly one or two full `Vec8f` widths, **no remainder**.
  - `100` (= `12*8 + 4`) — has *both* full-width iterations *and* a non-trivial
    4-element remainder, and matches the real `glove-100` dataset's
    dimensionality — called out as the **most important** case.
  - `128, 256` — like the `SIFT` dataset; multiples of both 8 and 16, so no
    remainder under either `Vec8f` or `Vec16f`.
  - `784` (= `98*8`) — `fashion-mnist`'s dimensionality; a multiple of 8 but
    *not* of 16, so it has a remainder only if `Vec16f` (AVX-512) is active.
  - `1024` — `bge-m3`'s (Lane B's embedding model) dimensionality; multiple of
    both, fully clean.
- The loop runs `run_dim` for every dimension in the list.
- Finally prints the pass/fail tally and **returns an exit code**: `0` (the
  conventional "success") only if `g_failed == 0`, otherwise `1`. This is what
  a shell or CI script checks via `$?` after running `./test_distances`.

---

## 11. `tests/test_hnsw.cpp`

This is the second standalone test program — it checks the **algorithm
itself** (Section 8) is correct: does it find approximately the right
neighbors compared to brute-force, is it deterministic, and does
save/load round-trip correctly? Built and run the same way as
`test_distances` ([Section 10](#10-teststest_distancescpp)):

```bash
cmake -S native/hnsw -B native/hnsw/build-test -DBUILD_TESTS=ON
cmake --build native/hnsw/build-test
./native/hnsw/build-test/test_hnsw
```

```cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "hnsw/distances_scalar.hpp"
#include "hnsw/hnsw_index.hpp"

using namespace hnsw;
```

New includes compared to `test_distances.cpp`: **`<chrono>`** (C++'s timing
library, for measuring build/search durations) and **`<set>`**
(`std::set<int>`, used below for fast membership checks in the recall
calculation). Note this file only includes `distances_scalar.hpp`, not
`distances_simd.hpp` — these tests exercise the algorithm via the scalar
distance kernels only (SIMD-vs-scalar correctness is `test_distances.cpp`'s
job).

`g_passed`/`g_failed`, `check()`, `random_vectors()`, and `l2_normalize()` are
all **redefined here identically** to Section 10.1 — each test program is a
fully independent, self-contained file with no shared "test utilities" header.

### 11.1 `brute_force_knn` and `recall_at_k`

```cpp
// Exact brute-force top-k for ground truth.
template <class Dist>
std::vector<std::vector<int>> brute_force_knn(const std::vector<float>& data, int n, int dim,
                                               const std::vector<float>& queries, int nq, int k) {
    std::vector<std::vector<int>> result(nq);
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = &queries[static_cast<size_t>(qi) * dim];
        std::vector<std::pair<float, int>> dists(n);
        for (int i = 0; i < n; ++i) {
            const float* p = &data[static_cast<size_t>(i) * dim];
            dists[i] = {Dist::dist(q, p, dim), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        result[qi].resize(k);
        for (int j = 0; j < k; ++j) result[qi][j] = dists[j].second;
    }
    return result;
}
```

`brute_force_knn<Dist>` computes the **exact** (ground-truth) top-`k` nearest
neighbors for every query, by brute force: for query `qi`, compute
`Dist::dist` against *every one* of the `n` data points, storing
`(distance, index)` pairs in `dists`. `std::partial_sort(dists.begin(),
dists.begin() + k, dists.end())` is like `std::sort` but cheaper when you only
need the smallest `k`: afterward, `dists[0..k-1]` are guaranteed to be the `k`
smallest elements, **in sorted order** — but `dists[k..]` are left in
unspecified order (saving work). The function then extracts just the ids
(`.second`) of those first `k` entries into `result[qi]`. The overall return
type `std::vector<std::vector<int>>` is "for each query, a list of `k` ids" —
the same shape `HnswIndex::search` produces per query.

```cpp
double recall_at_k(const std::vector<std::vector<int>>& retrieved,
                    const std::vector<std::vector<int>>& gt, int k) {
    double total = 0.0;
    for (size_t i = 0; i < retrieved.size(); ++i) {
        std::set<int> gt_set(gt[i].begin(), gt[i].begin() + k);
        int hits = 0;
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(retrieved[i][j])) ++hits;
        }
        total += static_cast<double>(hits) / k;
    }
    return total / static_cast<double>(retrieved.size());
}
```

`recall_at_k` is the standard **recall@k** metric (the same concept the Python
harness computes in `core/metrics.py`, reimplemented standalone here for the
C++ test): for each query `i`, build a `std::set<int>` of its ground-truth
top-`k` ids (`std::set` gives `O(log k)` membership checks via `.count()`),
then count how many of the *retrieved* top-`k` ids (`retrieved[i][j]`) are
*in* that ground-truth set (`hits`). Each query's score is `hits / k` (a
fraction between 0 and 1); the function returns the **average** of these
per-query scores across all queries.

### 11.2 `recall_test<Dist>` — build, search, measure

```cpp
template <class Dist>
double recall_test(const char* label, const std::vector<float>& data, int n, int dim,
                    const std::vector<float>& queries, int nq, int k, int M, int ef_construction,
                    int ef_search) {
    HnswIndex<Dist> index(dim, M, ef_construction, /*seed=*/42);

    auto t0 = std::chrono::high_resolution_clock::now();
    index.add_points(data.data(), n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();

    index.set_ef(ef_search);

    auto gt = brute_force_knn<Dist>(data, n, dim, queries, nq, k);

    std::vector<std::vector<int>> retrieved(nq, std::vector<int>(k));
    auto s0 = std::chrono::high_resolution_clock::now();
    for (int qi = 0; qi < nq; ++qi) {
        index.search(&queries[static_cast<size_t>(qi) * dim], k, retrieved[qi].data());
    }
    auto s1 = std::chrono::high_resolution_clock::now();
    double search_us = std::chrono::duration<double, std::micro>(s1 - s0).count() / nq;

    double recall = recall_at_k(retrieved, gt, k);
    std::printf("[%s] N=%d dim=%d M=%d efC=%d efS=%d build=%.3fs avg_search=%.1fus recall@%d=%.4f\n",
                 label, n, dim, M, ef_construction, ef_search, build_s, search_us, k, recall);
    return recall;
}
```

This function runs one **end-to-end test configuration**:

1. **Build**: construct `HnswIndex<Dist> index(dim, M, ef_construction,
   seed=42)` (fixed seed — Section 8.3), then call `add_points` (Section 8.6)
   on the whole dataset at once. `std::chrono::high_resolution_clock::now()`
   captures timestamps before (`t0`) and after (`t1`); `std::chrono::
   duration<double>(t1 - t0).count()` converts the elapsed time into a `double`
   number of **seconds**.
2. `index.set_ef(ef_search)` — set the search-time beam width (Section 8.6).
3. **Ground truth**: `brute_force_knn<Dist>` (Section 11.1) computes the exact
   answer for every query.
4. **Search**: for each query, call `index.search(...)`
   ([8.11](#811-search--the-public-entry-point-for-querying)), writing into
   `retrieved[qi]` (pre-allocated as `nq` vectors of `k` ints each via
   `std::vector<std::vector<int>> retrieved(nq, std::vector<int>(k))`). Timed
   the same way as the build, but `std::chrono::duration<double,
   std::micro>(...)` requests the result in **microseconds**, then `/ nq`
   gives the *average per-query* search time.
5. `recall = recall_at_k(retrieved, gt, k)` (Section 11.1).
6. Prints a one-line summary via `printf` — e.g.
   `[ScalarL2] N=10000 dim=64 M=16 efC=200 efS=100 build=0.842s avg_search=45.2us recall@10=0.9820`
   — and returns `recall` so `main()` can `check()` it against a threshold.

### 11.3 `main()` — three test blocks

```cpp
int main() {
    const int dim = 64;
    const int n = 10000;
    const int nq = 200;
    const int k = 10;
    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 100;

    std::printf("== Recall vs brute-force ==\n");

    auto data_l2 = random_vectors(n, dim, /*seed=*/1);
    auto queries_l2 = random_vectors(nq, dim, /*seed=*/2);
    double recall_l2 = recall_test<ScalarL2>("ScalarL2", data_l2, n, dim, queries_l2, nq, k, M,
                                              ef_construction, ef_search);
    check(recall_l2 > 0.95, "ScalarL2 recall@10 > 0.95");

    auto data_ip = random_vectors(n, dim, /*seed=*/3);
    auto queries_ip = random_vectors(nq, dim, /*seed=*/4);
    l2_normalize(data_ip, n, dim);
    l2_normalize(queries_ip, nq, dim);
    double recall_ip = recall_test<ScalarIP>("ScalarIP", data_ip, n, dim, queries_ip, nq, k, M,
                                              ef_construction, ef_search);
    check(recall_ip > 0.95, "ScalarIP recall@10 > 0.95");
```

**Fixed configuration** for the whole test: `dim=64` (an arbitrary but
realistic embedding size), `n=10000` points, `nq=200` queries, `k=10`
(recall@10), and the HNSW build/search parameters `M=16, ef_construction=200,
ef_search=100` — reasonable, "default-ish" values.

**Block 1 — Recall vs brute-force**:
- **L2**: random data/queries (seeds `1`/`2`), run `recall_test<ScalarL2>`,
  then `check(recall_l2 > 0.95, ...)`.
- **IP/cosine**: random data/queries (seeds `3`/`4`), **L2-normalized**
  (Section 10.1's `l2_normalize`, matching real cosine usage), run
  `recall_test<ScalarIP>`, then `check(recall_ip > 0.95, ...)`.

The `> 0.95` threshold matches the acceptance criterion in
`CUSTOM_HNSW_PLAN.md` §8 ("`recall@10` should be high (e.g. >0.95) at
reasonable `ef`") — i.e., this test is the automated check for "is the from-
scratch HNSW implementation actually finding approximately-correct nearest
neighbors?"

```cpp
    std::printf("\n== Determinism (serial build, fixed seed) ==\n");
    {
        const int small_n = 2000;
        auto data = random_vectors(small_n, dim, /*seed=*/5);
        auto queries = random_vectors(20, dim, /*seed=*/6);

        HnswIndex<ScalarL2> idx_a(dim, M, ef_construction, /*seed=*/42);
        idx_a.add_points(data.data(), small_n);
        idx_a.set_ef(ef_search);

        HnswIndex<ScalarL2> idx_b(dim, M, ef_construction, /*seed=*/42);
        idx_b.add_points(data.data(), small_n);
        idx_b.set_ef(ef_search);

        bool identical = true;
        for (int qi = 0; qi < 20; ++qi) {
            std::vector<int> ra(k), rb(k);
            idx_a.search(&queries[static_cast<size_t>(qi) * dim], k, ra.data());
            idx_b.search(&queries[static_cast<size_t>(qi) * dim], k, rb.data());
            if (ra != rb) {
                identical = false;
                break;
            }
        }
        check(identical, "two serial builds with the same seed give identical search results");
    }
```

**Block 2 — Determinism**: builds **two separate** `HnswIndex<ScalarL2>`
instances (`idx_a`, `idx_b`), each constructed with the **same** `seed=42`
and fed the **same** `data` (a smaller dataset, `small_n=2000`, for speed).
The extra `{ }` braces around this block create a local scope, so `data`,
`idx_a`, etc. don't leak into the rest of `main()`. For each of 20 queries, it
runs `search` on *both* indexes into separate output vectors `ra`/`rb`, and
compares them with `!=` (C++ compares `std::vector`s element-by-element —
`ra != rb` is true if they differ in size or in any element). If *any* query's
results differ, `identical` becomes `false` and the loop `break`s early.
`check(identical, ...)` then verifies: **same seed + same data + the current
(serial, single-threaded) build process ⇒ bit-for-bit identical graphs ⇒
identical search results.** This is the determinism guarantee promised by
fixing `rng_`'s seed in the constructor ([8.3](#83-constructor)) — important
because a *future* parallel-build path is expected to break this guarantee
(per `CUSTOM_HNSW_PLAN.md` §5: "Serial build is fully deterministic; parallel
build is not").

```cpp
    std::printf("\n== Save / load roundtrip ==\n");
    {
        const int small_n = 2000;
        auto data = random_vectors(small_n, dim, /*seed=*/7);
        auto queries = random_vectors(20, dim, /*seed=*/8);

        HnswIndex<ScalarL2> idx(dim, M, ef_construction, /*seed=*/42);
        idx.add_points(data.data(), small_n);
        idx.set_ef(ef_search);

        const std::string path = "/tmp/vdbhnsw_test_save.bin";
        idx.save(path);

        HnswIndex<ScalarL2> loaded(dim, M, ef_construction, /*seed=*/0);
        loaded.load(path);
        loaded.set_ef(ef_search);

        check(loaded.size() == idx.size(), "loaded index has the same number of points");

        bool identical = true;
        for (int qi = 0; qi < 20; ++qi) {
            std::vector<int> ra(k), rb(k);
            idx.search(&queries[static_cast<size_t>(qi) * dim], k, ra.data());
            loaded.search(&queries[static_cast<size_t>(qi) * dim], k, rb.data());
            if (ra != rb) {
                identical = false;
                break;
            }
        }
        check(identical, "loaded index gives identical search results to the original");
    }

    std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
```

**Block 3 — Save/load roundtrip**: builds one index `idx` (`small_n=2000`,
seed `7`; 20 queries, seed `8`), calls `idx.save("/tmp/vdbhnsw_test_save.bin")`
([8.13](#813-save-and-load)). Then constructs a **brand-new**
`HnswIndex<ScalarL2> loaded(dim, M, ef_construction, seed=0)` — note: a
*different and irrelevant* seed (`0` vs `42`), since `load()` is about to
overwrite *all* of `loaded`'s internal state anyway — and calls
`loaded.load(path)`. It then checks:

- `loaded.size() == idx.size()` — the loaded index has the same point count.
- For all 20 queries, `idx.search(...)` and `loaded.search(...)` produce
  identical result lists.

If both checks pass, `save`/`load` faithfully preserve everything that affects
search behavior — config, vectors, and the graph structure.

**The very end**: prints the overall `g_passed`/`g_failed` tally and returns
`0` (success) only if nothing failed, exactly like `test_distances.cpp`
(Section 10.3).

---

## 12. `third_party/vcl/`

```
third_party/vcl/
├── LICENSE
├── README.md
├── instrset.h
├── vectorclass.h        <- the umbrella header distances_simd.hpp includes
├── vectorf128.h / vectorf256.h / vectorf256e.h / vectorf512.h / vectorf512e.h
├── vectori128.h / vectori256.h / vectori256e.h / vectori512.h / vectori512e.h / vectori512s.h / vectori512se.h
├── vectorfp16.h / vectorfp16e.h
├── vector_convert.h
└── vectormath_common.h / vectormath_exp.h / vectormath_hyp.h / vectormath_lib.h / vectormath_trig.h
```

This directory is a **vendored copy of someone else's library** — Agner Fog's
**Vector Class Library (VCL) version 2**, a header-only C++ library that wraps
raw SIMD CPU instructions (AVX2, AVX-512, etc.) in convenient C++ types like
`Vec8f` ("8 packed `float`s") and `Vec16f` ("16 packed `float`s"), with
overloaded arithmetic operators (`+`, `-`, `*`) and helper functions
(`.load()`, `.load_partial()`, `mul_add()`, `horizontal_add()`, ...). It is
**not code written for this project** and isn't walked through function-by-
function here — only enough context to know what it's for:

- **`vectorclass.h`** — the umbrella header; `distances_simd.hpp`
  ([Section 6](#6-includehnswdistances_simdhpp)) does
  `#include "vectorclass.h"` and gets everything else transitively.
- **`instrset.h`** — detects, from compiler-defined macros (the ones
  `-march=native` causes to be set, [Section 4](#4-cmakeliststxt)), which
  instruction sets are available, and defines the corresponding feature macros
  (e.g. `__AVX512F__`) that `distances_simd.hpp`'s `#if defined(__AVX512F__)`
  checks.
- **`vectorf*.h` / `vectori*.h`** — define the actual floating-point
  (`vectorf`) and integer (`vectori`) vector types at different widths (128 /
  256 / 512-bit registers ↔ 4/8/16 `float`s or `int`s), with an `e`/`se` suffix
  meaning "emulated fallback" for when the CPU lacks the native instruction.
- **`vectorfp16*.h`, `vector_convert.h`, `vectormath_*.h`** — half-precision
  float support, type conversions, and vectorized transcendental math
  (`exp`, `log`, `sin`, ...). **None of these are used by this project** —
  `distances_simd.hpp` only needs `Vec8f`/`Vec16f`, basic arithmetic, `load`,
  `load_partial`, `mul_add`, and `horizontal_add` — but they come along
  because the library is vendored as a whole, header-only.
- **`LICENSE` / `README.md`** — the library's own license (governing this
  vendored copy) and documentation, unrelated to `vdbbench`.

---

## 13. End-to-end walkthroughs

With every file covered individually, here's how a call from Python actually
flows through all of them. (The Python-side adapter code itself —
`HnswScalarAdapter`/`HnswFastAdapter` — is outside this document's scope, but
is shown here just to anchor where each C++ piece is reached *from*.)

### 13.1 Building an index

```python
index = _vdbhnsw.HnswScalarL2(dim=128, M=16, ef_construction=200, seed=42)
index.add_points(vectors)   # vectors: (N, 128) float32 NumPy array
```

1. `HnswScalarL2(...)` → pybind11's generated `__init__`
   ([9.1](#91-bind_indexdist--registering-one-class)) → `HnswIndex<ScalarL2>`'s
   constructor ([8.3](#83-constructor)) — stores `dim_, M_, M_max0_,
   ef_construction_, ef_search_=10, mL_`, sets `entry_point_=0,
   max_level_=-1`, seeds `rng_`.
2. `index.add_points(vectors)` → the `add_points` **lambda**
   ([9.2](#92-add_points--wrapping-a-numpy-array-as-a-raw-pointer)) — validates
   `vectors` is `(N, 128)` `float32`, releases the GIL, and calls
   `HnswIndex::add_points(ptr, N, ...)`
   ([8.6](#86-add_points--building-the-index)).
3. `add_points` copies all `N × 128` floats into `data_`, grows `levels_`,
   `links_`, `visited_pool_` to size `N`, then for `id = 0..N-1` calls
   `add_point(id)` ([8.10](#810-add_point--inserting-one-point)):
   - `id = 0`: `max_level_ < 0` ⇒ becomes `entry_point_` and defines
     `max_level_`. Done.
   - `id ≥ 1`: `random_level()` ([8.5](#85-helper-get_data-and-random_level))
     picks `new_level`. If `new_level < max_level_`, greedily hill-climb down
     from `entry_point_` through layers `max_level_ .. new_level+1`. Then, for
     each layer `min(new_level, max_level_) .. 0`:
     - `search_layer` ([8.7](#87-search_layer--beam-search-at-a-single-layer))
       finds up to `ef_construction_` nearby existing points.
     - `select_neighbors_heuristic`
       ([8.8](#88-select_neighbors_heuristic--diversity-aware-pruning)) picks
       a diverse subset (capped at `M_max0_` for layer 0, `M_` otherwise) as
       `links_[id][level]`.
     - `connect_and_prune`
       ([8.9](#89-connect_and_prune--keeping-links-roughly-bidirectional)) adds
       the reverse edge `neighbor → id` for each selected neighbor, re-pruning
       that neighbor's list if it's now over capacity.
     - If `new_level > max_level_`, `id` becomes the new `entry_point_` and
       `max_level_` is raised.

### 13.2 Searching

```python
index.set_ef(100)
ids = index.search(query, k=10)   # query: (128,) float32 NumPy array
```

1. `index.set_ef(100)` → `HnswIndex::set_ef` → `ef_search_ = 100`. No rebuild.
2. `index.search(query, 10)` → the `search` **lambda**
   ([9.4](#94-search--wrapping-a-numpy-array-returning-a-list)) — validates
   `query` is `(128,)` `float32`, allocates `out = [0]*10`, releases the GIL,
   and calls `HnswIndex::search(ptr, 10, out.data())`
   ([8.11](#811-search--the-public-entry-point-for-querying)).
3. `search`:
   - Fills `out_ids` with `-1` (in case fewer than 10 points exist).
   - Starts at `entry_point_`; for layers `max_level_ downto 1`, greedily
     hill-climbs to the nearest neighbor at each layer (same pattern as
     Phase 1 of `add_point`).
   - At layer 0, runs `search_layer` with `ef = max(ef_search_, k) = 100`,
     getting up to 100 candidates.
   - Trims to the closest 10, sorts ascending by distance, and writes their
     ids into `out_ids`.
4. The lambda re-acquires the GIL and converts `out` (`std::vector<int>`) to a
   Python `list[int]`, returned to the caller as `ids`.

### 13.3 Persistence and size metrics

```python
index.save("/tmp/index.bin")           # -> os.path.getsize() for on-disk size
mem_bytes = index.index_memory_bytes() # in-memory estimate
```

- `index.save(path)` → `HnswIndex::save`
  ([8.13](#813-save-and-load)) — writes a header (magic/version/config/`n`),
  then `data_`, `levels_`, and the entire `links_` graph to a binary file.
  `index.load(path)` reverses this exactly. The Python adapter's
  `index_size_bytes()` uses `save()` to a temp file plus `os.path.getsize()`
  to report **on-disk** index size; `index.index_memory_bytes()`
  ([8.12](#812-size-dim-and-index_memory_bytes)) reports an **in-memory**
  estimate directly, without touching disk.

### 13.4 Where `Scalar*` vs `Fast*` diverge

Both flows above are **identical in code path** for `HnswScalarL2` and
`HnswFastL2` — the *only* difference is which `Dist::dist` gets inlined into
`search_layer`/`add_point`/etc. at compile time
([9.5](#95-the-module-entry-point)): `ScalarL2::dist`
([Section 5](#5-includehnswdistances_scalarhpp)) is a plain loop; `SimdL2::dist`
([Section 6](#6-includehnswdistances_simdhpp)) processes 8 or 16 dimensions per
CPU instruction via VCL ([Section 12](#12-third_partyvcl)). Everything about
the *graph* — levels, edges, beam search, neighbor selection, file format — is
shared, unmodified, templated code.

---

## 14. Where to go next

- **`CUSTOM_HNSW_PLAN.md`** (repo root) — the design rationale and phased
  roadmap this code follows. Of particular note for understanding *why* the
  code looks the way it does:
  - The `parallel`/`num_threads` parameters accepted-but-ignored by
    `add_points` ([8.6](#86-add_points--building-the-index)) exist for a
    planned future **OpenMP parallel build** phase ("Phase C"), which would
    also add a `locks.hpp` (striped lock pool) — not present in the codebase
    yet.
  - "Phase A" (scalar, serial) vs "Phase B" (SIMD distance kernels, still
    serial build) explain the comments in `hnsw_index.hpp` and
    `bindings.cpp`.
- **`docs/CODEBASE.md`** — the equivalent walkthrough for the *Python* side of
  `vdbbench`, including `src/vdbbench/backends/hnsw_scalar_backend.py` and
  `hnsw_fast_backend.py`, which are the `VectorDBAdapter` implementations that
  call into the four classes documented here (Section 9).
- **Building and testing this code directly** (outside the normal `pip
  install -e .` flow):
  ```bash
  cmake -S native/hnsw -B native/hnsw/build-test -DBUILD_TESTS=ON
  cmake --build native/hnsw/build-test
  ./native/hnsw/build-test/test_distances   # SIMD vs scalar (Section 10)
  ./native/hnsw/build-test/test_hnsw        # recall / determinism / save-load (Section 11)
  ```
- **A good order to (re-)read the source in**, if you want to go back to the
  actual code with this document as a companion: `distances_scalar.hpp`
  (Section 5) → `visited_pool.hpp` (Section 7) → `hnsw_index.hpp` (Section 8,
  the templated core) → `distances_simd.hpp` (Section 6) →
  `bindings.cpp` (Section 9) → the two `tests/*.cpp` files (Sections 10–11).







