# Error: `_vdbhnsw` extension fails to build on editable-install rebuild

## Symptom

```
Running cmake --build & --install in /home/midnight/.../build/cp312-cp312-linux_x86_64
No such file or directory
CMake Error: Generator: execution of make failed. Make command was: ninja
...
subprocess.CalledProcessError: Command '['cmake', '--build', '.']' returned non-zero exit status 1.
```

Triggered by importing `vdbbench.backends.hnsw_fast_backend`, which does `import _vdbhnsw`.
Because the project is installed with `pip install -e .` via `scikit-build-core`
(`[tool.scikit-build.editable] rebuild = true`), every import of the package re-checks
whether the native extension is stale and reruns `cmake --build .` if so
(`.venv/lib/python3.12/site-packages/_vdbbench_editable.py`).

## Root causes (two, found in sequence)

1. **`ninja` not on `PATH`.** CMake was configured with `CMAKE_GENERATOR=Ninja` /
   `CMAKE_MAKE_PROGRAM=ninja`, but no `ninja` binary existed anywhere on the system or in
   `.venv`. Installing the `ninja` PyPI package (which ships a console-script binary at
   `.venv/bin/ninja`) was not enough by itself, because `.venv` was not activated in the
   shell — `.venv/bin` wasn't on `PATH`, so the `cmake --build` subprocess (which inherits
   the calling shell's environment) still couldn't find it.

2. **`pybind11` not installed in `.venv`.** Once `ninja` was found, CMake re-ran and failed
   at `find_package(pybind11 ...)` in `native/hnsw/CMakeLists.txt:18` — no
   `pybind11Config.cmake` was discoverable. `pybind11` is listed under `[build-system]
   requires` in `pyproject.toml`, but that list is only installed into pip's *isolated build
   environment* for the initial `pip install -e .` — it is not installed into the live
   `.venv` that the editable-rebuild hook uses for subsequent `cmake --build` calls on
   import. So the live venv needed `pybind11` installed directly.

## Fix applied

```bash
source .venv/bin/activate
pip install ninja pybind11
vdbbench-run --config config/default.yaml   # rebuild succeeds, benchmark runs
```

## Is this permanent?

**Partially.** `requirements.txt` in this repo already happens to pin
`ninja==1.13.0` and `pybind11==3.0.4` (it was generated from a `pip freeze` after a prior
fix), so re-running `pip install -r requirements.txt` on a fresh `.venv` would reproduce a
working environment.

However, **`pyproject.toml` itself does not guarantee this**:

- `ninja` is not declared anywhere in `pyproject.toml` (`[build-system]` nor
  `[project] dependencies`).
- `pybind11` is declared only under `[build-system] requires`, which scikit-build-core's
  *editable rebuild hook* does not consult — it just shells out to `cmake --build .` in the
  developer's existing venv.

So anyone running the documented setup steps from `CLAUDE.md`
(`pip install -e .` with no `requirements.txt` step) will hit this exact error again on a
clean machine. To make the fix durable, `ninja` and `pybind11` should be added to
`[project] dependencies` (or a `dev`/`build` extras group) in `pyproject.toml`, not just
relied upon via `requirements.txt`.

**Update:** `ninja` and `pybind11>=2.12` have since been added to `[project] dependencies`
in `pyproject.toml`, so a plain `pip install -e .` on a clean machine now installs both into
the live venv and the editable-rebuild hook works without needing `requirements.txt` at all.

## Follow-up Q&A

### `pyproject.toml` vs `requirements.txt`

| | `pyproject.toml` | `requirements.txt` |
|---|---|---|
| Standard | PEP 518/621 — the standardized project/build metadata file | Not standardized — just a flat list pip understands |
| Purpose | Describes how to build and what the package is: build backend (`scikit_build_core.build`), build-time deps (`[build-system] requires`), package metadata, runtime deps, optional extras, console scripts | A pinned/unpinned dependency list for `pip install -r requirements.txt` — usually a lockfile-like snapshot |
| Scope | Drives `pip install .` / `pip install -e .` — pip reads `[build-system]` to set up an isolated build env, then `[project] dependencies` for runtime deps | Only installs packages into whatever env is active — no notion of "this is a package," no build backend, no entry points |

### Do we need both?

Not strictly — they overlap and can drift, which is exactly what caused this bug
(`requirements.txt` had `ninja`/`pybind11` pinned, but `pyproject.toml`'s `dependencies` did
not, so a fresh `pip install -e .` without `requirements.txt` would still fail). Now that
`pyproject.toml` is the single source of truth for this project's runtime deps,
`requirements.txt` is redundant unless exact-version pinning/reproducibility (e.g. for CI) is
specifically wanted — in which case a generated lockfile (`pip-compile`, or `uv.lock` under
uv) is a better fit than a hand-maintained `requirements.txt`, since it won't silently drift
out of sync with `pyproject.toml`.

### Can we pin exact versions in `pyproject.toml`?

Yes — `[project] dependencies` accepts PEP 440 version specifiers, including exact pins:

```toml
dependencies = [
    "numpy==2.4.6",
    "faiss-cpu==1.14.2",
    "ninja==1.13.0",
    "pybind11==3.0.4",
]
```

Trade-off: pinning everything in `pyproject.toml` means *every* install (including as a
dependency of another project) is locked to those exact versions, which can cause resolver
conflicts if something else in the environment needs a different version of the same
package. The conventional split is: keep `pyproject.toml` deps loose/range-bound (what the
package is *compatible with*), and use a separate lockfile (`requirements.txt` generated via
`pip-compile`, or `uv.lock`) for *exact, reproducible* environments (CI, deployment). This
project currently keeps `pyproject.toml` unpinned for that reason.

### Is `uv` better than plain `python3`/`pip` here?

Likely yes for this project specifically, given the native-extension build (scikit-build-core
+ cmake + ninja) requires frequent reinstalls during iteration:

- Much faster dependency resolution/installation (Rust-based).
- `uv.lock` gives real reproducibility without manually maintaining `requirements.txt`.
- Replaces `venv` + `pip` + `pip-tools` with one tool.
- Caveat: switching to uv does **not** by itself fix the "build deps must be in the live venv,
  not just `[build-system] requires`" issue — that's solved by declaring `ninja`/`pybind11` in
  `[project] dependencies` regardless of which tool installs them.
- uv still won't install system-level tools (a C++ compiler, system `cmake`) — those remain
  the OS package manager's responsibility either way.

## Error: `cmake: not found` pointing at a deleted `uv` build-isolation temp dir

### Symptom

After cloning on a different machine and running `uv sync` / `vdbbench-run`:

```
/bin/sh: 1: /home/nishit/.cache/uv/builds-v0/.tmpM5qAqt/lib/python3.11/site-packages/cmake/data/bin/cmake: not found
FAILED: [code=127] build.ninja .../cmake_install.cmake
ninja: error: rebuilding 'build.ninja': subcommand failed
```

Adding `cmake` to `[project] dependencies` (so it lands in the persistent `.venv`) did **not**
fix this — the same error recurred with a *different* `.tmp*` path each time.

### Root cause

`uv sync` builds the project's wheel inside a **separate, ephemeral build-isolation venv**
per PEP 517/518. That isolation venv is populated only from `[build-system] requires`
(`scikit-build-core`, `pybind11`) — it does **not** include `[project] dependencies`
(`cmake`, `ninja`, etc.), because those are runtime deps of the *installed* package, not
build-time deps of the *build backend*.

Since no `cmake` exists in that isolated env (and none on the system `PATH` either),
scikit-build-core auto-fetches a `cmake` PyPI wheel into the temp build-isolation directory
just to run the configure step. CMake then **bakes the absolute path to that binary** into
`CMakeCache.txt` (`CMAKE_COMMAND`). Once `uv` finishes the build and deletes the temp
isolation venv, that path no longer exists — but `[tool.scikit-build.editable] rebuild =
true` means every subsequent `import _vdbhnsw` re-invokes `cmake --build .` against the
*same stale cache*, so it fails forever, with a new random temp path each time `uv sync`
reruns the build.

Adding `cmake` to `[project] dependencies` didn't help because the *build* step never
consults the project's dependency list — only the isolated build env, which is intentionally
ephemeral by PEP 517 design.

### Fix

Disable build isolation for this package specifically, so the build runs directly inside the
persistent `.venv` (which already has `cmake`/`ninja`/`pybind11` installed as regular
dependencies), and add `scikit-build-core` itself to `[project] dependencies` too (since
with isolation off, the build backend must also be importable from `.venv`):

```toml
[project]
dependencies = [
    ...
    "ninja",
    "pybind11>=2.12",
    "cmake>=3.21",
    "scikit-build-core>=0.10",
]

[tool.uv]
no-build-isolation-package = ["vdbbench"]
```

After `rm -rf build/ .venv && uv lock && uv sync`, `CMakeCache.txt`'s `CMAKE_COMMAND` now
resolves to `<repo>/.venv/lib/python3.11/site-packages/cmake/data/bin/cmake` — a path that
persists for the lifetime of `.venv`, instead of a deleted `uv` build-isolation temp dir.
Verified working: `ninja: no work to do` on a no-op rebuild, and a full benchmark run
completed successfully (`faiss_flat`/`faiss_hnsw`, all `k` values).

Note: `[tool.uv.extra-build-dependencies]` (which uv's own error hint suggests) does not
appear to be functional in `uv 0.11.21` — putting `scikit-build-core` there had no effect;
putting it directly in `[project] dependencies` alongside `no-build-isolation-package` is
what actually worked.

### Is this permanent?

Yes, on a machine with a C++ toolchain and `cmake`-buildable environment available
(`scikit-build-core`/`cmake`/`ninja`/`pybind11` are now ordinary pinned-by-lock project
dependencies, not implicitly fetched build-isolation artifacts). A fresh clone + `uv sync`
reproduces a working build without manual intervention. System-level prerequisites (C/C++
compiler) are still outside uv's/pip's control either way.


Yes, you can pin exact versions in pyproject.toml — [project] dependencies accepts full PEP 440 specifiers, so "numpy==2.4.6" works exactly like in requirements.txt. The reason this project (and most Python packages) avoids pinning there: if your package is ever installed as a dependency of something else, exact pins in dependencies can conflict with whatever versions that other project needs, causing resolver failures. The convention is to keep pyproject.toml ranges loose (compatibility) and use a separate lockfile — requirements.txt/pip-compile output, or uv.lock — for exact reproducible pins in CI/deployment.

#### How to create env with a different name using uv sync ?

- use uv venv .myenv to create it, then uv pip sync/uv pip install -e . into that explicit path instead of uv sync