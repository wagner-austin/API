---
title: ClearGBM Rust-only refactor plan
tags: [ml, cleargbm, refactor, plan]
related: [[cleargbm-histogram-split-path]]
sources:
  - libs/cleargbm/src/cleargbm/
  - libs/cleargbm_rs/src/
  - libs/covenant_ml/src/covenant_ml/backends/cleargbm/
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-20.md
fact_checked: 2026-07-20
confidence: high
---

# ClearGBM Rust-only refactor plan

Executable plan for turning `cleargbm` from a Python-with-optional-Rust library into a Rust-core-with-thin-Python-API package. Every step names concrete files and functions. Nothing here is aspirational — each item is either shipped, in progress, or has a hard dependency spelled out.

## Vision

**One package** (`libs/cleargbm/`) with:
- Rust as the only computational path — no `_default_*` Python implementations, no `use_rust_backend()` opt-in, no `use_python_backend()` for tests.
- Python code strictly for boundary marshalling: TypedDict configs → Rust config dicts, numpy inputs → Rust arrays, tuple results → numpy arrays.
- One wheel: cleargbm-x.y-cp311-abi3-win_amd64.whl (or platform equivalent) that contains both the compiled Rust extension and the Python surface.
- No `libs/cleargbm_rs/` — merged into `libs/cleargbm/`.

## Package structure

**Before** (as of 2026-07-20):

```
libs/cleargbm/                              # poetry package
  pyproject.toml                            # poetry.core.masonry backend
  src/cleargbm/
    _hooks_binning.py                       # Python default + hook indirection
    _hooks_compute.py
    _hooks_ensemble.py
    _hooks_guard.py
    _hooks_histogram.py
    _hooks_infra.py
    _hooks_loss.py
    _hooks_native.py                        # Only Rust-Path; delegates to cleargbm_rs
    _hooks_prediction.py
    _hooks_sigmoid.py
    _rust_adapters.py                       # Wires Rust into per-op hooks (use_rust_backend)
    _rust_native_adapters.py                # Wires Rust into full-loop hook (wire_native_hooks)
    _test_hooks.py                          # Re-exports for tests
    _types_*.py                             # TypedDicts + encode/decode
    buffers.py
    ensemble.py                             # train_gradient_boosting (Python loop, Rust-hooked)
                                            # + train_gradient_boosting_native (single Rust call)
    explain.py                              # Python-only explainer
    histogram.py                            # Python histogram scan (find_best_split_from_histogram)
    losses.py                               # Delegates to _hooks_loss
    parallel.py                             # Python multiprocessing worker pool
    split.py                                # Exact/sorted path (dead at runtime, live in tests)
    tree.py                                 # Python tree builder
    types.py                                # Re-export barrel
  tests/                                    # 555 tests, 100% coverage
  scripts/
    autotune.py
    benchmark.py
    guard.py

libs/cleargbm_rs/                           # maturin package (separate)
  Cargo.toml
  pyproject.toml                            # maturin backend
  src/
    lib.rs
    training/model.rs                       # GradientBoostingModel (no serde)
    training/config.rs                      # GradientBoostingConfig (no serde)
    tree/                                   # Tree + serde
    split/                                  # MonotonicConstraint (no serde), NanDirection (serde ✓)
    histogram/                              # scan + build (serde ✓ on buffer)
    losses/                                 # scalar + vector
    predict/                                # ensemble + single-tree
    binning/                                # bin edges + assignment
    pyo3_module/
      training_fns.rs                       # PyGbmModel (opaque, no methods except train)
      tree_fns.rs                           # PyTree (has to_json / from_json ✓)
      histogram_fns.rs
      loss_fns.rs
      prediction_fns.rs
      binning_fns.rs
      mod.rs                                # #[pymodule]
    types/serde_impl/
      histogram_buffer.rs
      split_config.rs
      tree_node.rs
      tree_node_config.rs
  target/wheels/
    cleargbm_rs-0.1.0-cp311-cp311-win_amd64.whl
```

**After** (target state):

```
libs/cleargbm/                              # one package, maturin backend
  Cargo.toml                                # (moved from libs/cleargbm_rs)
  pyproject.toml                            # build-backend = "maturin"
  Makefile                                  # runs both cargo + pytest
  rust/                                     # (moved from libs/cleargbm_rs/src)
    lib.rs
    training/
    tree/
    split/
    histogram/
    losses/
    predict/
    binning/
    pyo3_module/
    types/
  python/cleargbm/                          # thin surface
    __init__.py                             # imports cleargbm_rs, re-exports public API
    _rust.py                                # Protocol-typed adapters onto the Rust extension
    _test_hooks.py                          # for test DI
    _types_config.py                        # GradientBoostingConfig TypedDict + encode/decode
    _types_model.py                         # GradientBoostingModel TypedDict + encode/decode
    _types_explain.py
    _types_json.py
    types.py                                # re-export barrel
    ensemble.py                             # train_gradient_boosting (aliases native), predict_proba
    explain.py                              # explain_prediction, get_feature_importances (Rust-backed)
  tests/                                    # updated to reflect Rust-only reality
  scripts/
    autotune.py
    benchmark.py                            # updated to expect Rust-only
    guard.py

libs/cleargbm_rs/                           # DELETED
libs/covenant_ml/src/covenant_ml/backends/cleargbm/backend.py
                                            # updated: uses train_gradient_boosting_native
                                            # save/load via PyGbmModel.to_json
                                            # get_feature_importances via PyGbmModel.feature_importances
```

## Files that get DELETED (concrete list)

- `libs/cleargbm/src/cleargbm/_hooks_binning.py` — delete entire file
- `libs/cleargbm/src/cleargbm/_hooks_compute.py` — delete entire file
- `libs/cleargbm/src/cleargbm/_hooks_ensemble.py` — delete entire file
- `libs/cleargbm/src/cleargbm/_hooks_guard.py` — retained (guard-script hooks are a separate concern from math backend)
- `libs/cleargbm/src/cleargbm/_hooks_histogram.py` — delete entire file
- `libs/cleargbm/src/cleargbm/_hooks_infra.py` — audit; delete Python worker-pool code if `train_gradient_boosting_native` handles parallelism internally in Rust
- `libs/cleargbm/src/cleargbm/_hooks_loss.py` — delete entire file
- `libs/cleargbm/src/cleargbm/_hooks_native.py` — merge into `_rust.py`
- `libs/cleargbm/src/cleargbm/_hooks_prediction.py` — delete entire file
- `libs/cleargbm/src/cleargbm/_hooks_sigmoid.py` — delete entire file
- `libs/cleargbm/src/cleargbm/_rust_adapters.py` — delete entire file (per-op adapters obsoleted by native path)
- `libs/cleargbm/src/cleargbm/_rust_native_adapters.py` — merge into `_rust.py`
- `libs/cleargbm/src/cleargbm/histogram.py::find_best_split_from_histogram` — delete function (Rust handles this)
- `libs/cleargbm/src/cleargbm/histogram.py::_compute_split_gain` — delete function
- `libs/cleargbm/src/cleargbm/histogram.py::_check_monotonicity_constraint` — delete function
- `libs/cleargbm/src/cleargbm/histogram.py::_evaluate_nan_direction` — delete function
- `libs/cleargbm/src/cleargbm/parallel.py` — delete entire file
- `libs/cleargbm/src/cleargbm/split.py` — delete entire file
- `libs/cleargbm/src/cleargbm/tree.py::_build_tree_with_histograms` — delete function
- `libs/cleargbm/src/cleargbm/tree.py::_compute_child_histograms` — delete function
- `libs/cleargbm/src/cleargbm/tree.py::build_tree` — delete function (replaced by Rust)
- `libs/cleargbm/src/cleargbm/ensemble.py::train_gradient_boosting` — delete (replaced by `train_gradient_boosting_native`)
- `libs/cleargbm/src/cleargbm/ensemble.py::predict_proba` (Python-loop version) — delete
- `libs/cleargbm/src/cleargbm/ensemble.py::_EarlyStoppingState`, `_update_early_stopping_state`, `_ValidationState`, `_update_validation`, `_SimpleValState`, `_update_simple_validation`, `_compute_loss`, `_add_tree_predictions`, `_create_worker_pool` — delete (early stopping now lives in Rust)
- `libs/cleargbm/src/cleargbm/losses.py::compute_raw_predictions`, `raw_to_proba` — delete (Rust handles)
- `libs/cleargbm/tests/test_tree.py` — delete (Python tree builder is gone)
- `libs/cleargbm/tests/test_parallel.py` — delete (Python parallel worker is gone)
- `libs/cleargbm/tests/test_split.py` — delete (Python exact split is gone)
- `libs/cleargbm/tests/test_ensemble.py::TestEarlyStopping*` — delete
- Every test that specifically asserts against `_default_*` — delete (they were exercising deleted code)

## Files that get CREATED (concrete list)

### Rust side (in cleargbm_rs, before merging)

- `libs/cleargbm_rs/src/training/serde_impl.rs` — new file, manual `Serialize`/`Deserialize` for `GradientBoostingConfig` + `GradientBoostingModel` (no `?` operator per project convention; follow the pattern in `tree/serde_impl.rs`)
- `libs/cleargbm_rs/src/split/serde_impl.rs` — add `impl Serialize for MonotonicConstraint` + `impl<'de> Deserialize<'de> for MonotonicConstraint` (three-variant string enum: "None" / "Increasing" / "Decreasing")
- `libs/cleargbm_rs/src/training/importance.rs` — new file, `pub fn feature_importances(model: &GradientBoostingModel) -> Vec<(String, f64)>` (walks trees, tallies feature_index appearances at internal nodes, normalizes to [0, 1])
- `libs/cleargbm_rs/src/pyo3_module/training_fns.rs` — add `#[pymethods] impl PyGbmModel`:
  - `pub fn to_json(&self) -> PyResult<String>`
  - `#[staticmethod] pub fn from_json(json: &str) -> PyResult<PyGbmModel>`
  - `pub fn feature_importances(&self) -> PyResult<Vec<(String, f64)>>`
  - `pub fn n_trees(&self) -> usize`
  - `pub fn n_classes(&self) -> usize`
- `libs/cleargbm_rs/src/training/tests/model_serde_tests.rs` — new file: roundtrip test on `GradientBoostingModel`
- `libs/cleargbm_rs/src/training/tests/importance_tests.rs` — new file: importance sums to 1.0, monotone-informative feature ranked first, degenerate cases (empty model, single-leaf model)
- `libs/cleargbm_rs/src/pyo3_module/tests/model_json_tests.rs` — new file: end-to-end train → to_json → from_json → predict → same output

### Python side (after merge, in libs/cleargbm/python/cleargbm/)

- `python/cleargbm/_rust.py` — Protocol-typed adapters, imports `cleargbm_rs` at module load, defines:
  - `class PyGbmModelProto(Protocol)` — matches PyO3 surface exactly
  - `train_gradient_boosting(...)` — direct call into `cleargbm_rs.train_gradient_boosting_rs`
  - `predict_proba_model(...)` — direct call into `cleargbm_rs.predict_proba_model_rs`
  - `model_to_json(model)` — thin wrapper on `model.to_json()`
  - `model_from_json(s)` — thin wrapper on `cleargbm_rs.PyGbmModel.from_json(s)`
  - `feature_importances(model)` — thin wrapper on `model.feature_importances()`
- `python/cleargbm/__init__.py` — imports `_rust` at load, re-exports `train_gradient_boosting`, `predict_proba`, `model_to_json`, `model_from_json`, `feature_importances`, `GradientBoostingConfig`, `GradientBoostingModel` (opaque Python type = `PyGbmModel`)

### Guards + tooling (parity with other repos)

- `scripts/guard.py` — matches shape of `libs/covenant_ml/scripts/guard.py`; runs `mcp_shared_py.guard` or equivalent for cleargbm-specific rules
- `Makefile` — `lint` runs `poetry run python -m scripts.guard`, `poetry run ruff check`, `poetry run mypy src tests scripts`, `cargo fmt --check`, `cargo clippy -- -D warnings`; `test` runs `poetry run pytest -n auto --cov=python --cov=scripts --cov-branch --cov-report=term-missing` AND `cargo test --all-features`; `check` = `lint | test`
- `pyproject.toml`:
  - `build-system.requires = ["maturin>=1.4,<2.0"]`
  - `build-system.build-backend = "maturin"`
  - `tool.maturin.features = ["extension-module"]`
  - `tool.maturin.python-source = "python"`
  - `tool.maturin.module-name = "cleargbm_rs"` (extension module) — Python package `cleargbm` imports `cleargbm_rs` internally
  - `project.dependencies` includes `numpy>=2.3.5` (no `cleargbm_rs` — that's the built extension)
- `Cargo.toml` — copied from `libs/cleargbm_rs/Cargo.toml`, package name unchanged (`cleargbm_rs`), lives at `libs/cleargbm/Cargo.toml`

## Phase-by-phase execution order

Each phase leaves `make check` green in cleargbm and covenant_ml, no half-shipped state.

### Phase A — Rust serialization + feature importance (in `libs/cleargbm_rs/`)

Preserves current package split; adds capability. Doesn't touch Python code except tests exercising new PyO3 methods.

Steps:

1. Implement `impl Serialize for MonotonicConstraint` + `impl<'de> Deserialize<'de> for MonotonicConstraint` in `libs/cleargbm_rs/src/split/serde_impl.rs`. Add tests in `libs/cleargbm_rs/src/split/tests/serde_tests.rs`: three variants roundtrip; unknown variant returns error.
2. Implement `impl Serialize for GradientBoostingConfig` + `impl<'de> Deserialize<'de> for GradientBoostingConfig` in a new `libs/cleargbm_rs/src/training/serde_impl.rs`. Serialize all 12 fields; deserialize routes through `GradientBoostingConfig::new(...)` to preserve validation. Tests: field-order-independent roundtrip; missing-required-field error; validation error propagation.
3. Implement `impl Serialize for GradientBoostingModel` + `impl<'de> Deserialize<'de> for GradientBoostingModel` in the same file. Serialize `trees`, `base_prediction`, `learning_rate`, `feature_names`, `n_classes`, `config`. Deserialize constructs via a new `GradientBoostingModel::new_from_serde(...)` constructor that trusts the input (already-trained model, no re-validation).
4. Implement `pub fn feature_importances(&self) -> Vec<(String, f64)>` on `GradientBoostingModel` in new file `libs/cleargbm_rs/src/training/importance.rs`. Split-count importance: for each internal node in each tree, tally `feature_index`; normalize so importances sum to 1.0. Tests: single-feature model has `[1.0]` for that feature; empty model returns all zeros; multi-feature model importances sum to 1.0 within FP epsilon; feature ranking matches expected structure on synthetic-signal data.
5. Add `#[pymethods] impl PyGbmModel` block in `libs/cleargbm_rs/src/pyo3_module/training_fns.rs`:
   - `pub fn to_json(&self) -> PyResult<String>` — `serde_json::to_string(&self.inner).map_err(|e| ser_err(e.to_string()))`
   - `#[staticmethod] pub fn from_json(json: &str) -> PyResult<PyGbmModel>` — `serde_json::from_str(json).map(|m| PyGbmModel { inner: m }).map_err(|e| ser_err(e.to_string()))`
   - `pub fn feature_importances(&self) -> Vec<(String, f64)>` — direct return
   - `pub fn n_trees(&self) -> usize` + `pub fn n_classes(&self) -> usize` — accessors
6. Add integration test `libs/cleargbm_rs/src/pyo3_module/tests/model_json_tests.rs`: end-to-end train small model → to_json → from_json → predict on same X → assert per-sample predictions match within 1e-15.
7. `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features` — must be green.
8. `maturin build --release` from `libs/cleargbm_rs/` — produces new wheel at `target/wheels/cleargbm_rs-0.1.0-cp311-cp311-win_amd64.whl`.
9. Update `libs/cleargbm_rs/README.md` with the new methods.
10. **Phase A gate:** cargo test green, cargo llvm-cov segment coverage = 100%.

### Phase B — Package merge (`libs/cleargbm_rs/` → `libs/cleargbm/`)

At this point Rust is capable. Merge packages so Rust is inseparable from cleargbm.

Steps:

1. Rename `libs/cleargbm/src/` to `libs/cleargbm/python/` (matches maturin `python-source` default).
2. Copy `libs/cleargbm_rs/Cargo.toml` → `libs/cleargbm/Cargo.toml` (package name stays `cleargbm_rs` — that's the extension module name).
3. Copy `libs/cleargbm_rs/src/` → `libs/cleargbm/rust/src/` (adjust Cargo.toml `path` if needed to point at `rust/src/`).
4. Move `libs/cleargbm_rs/Cargo.lock` → `libs/cleargbm/Cargo.lock`.
5. Rewrite `libs/cleargbm/pyproject.toml`:
   - `build-system.requires = ["maturin>=1.4,<2.0"]`
   - `build-system.build-backend = "maturin"`
   - Add `[tool.maturin]` section (features=`["extension-module"]`, `python-source="python"`, `module-name="cleargbm_rs"`)
   - Remove poetry sections; move dependencies to `[project.dependencies]`.
   - Coverage sources move from `["src", "scripts"]` to `["python", "scripts"]`.
6. Rewrite `libs/cleargbm/Makefile` to mirror `libs/cleargbm_rs/Makefile` (rust-lint + rust-test + python-lint + python-test targets; `check = lint | test`).
7. Delete `libs/cleargbm_rs/` entirely.
8. Search-replace `from libs.cleargbm_rs` and `cleargbm_rs` import references across the api monorepo. `covenant_ml/backends/cleargbm/backend.py` uses `import cleargbm_rs` — that stays the same (the extension module name doesn't change; only the SOURCE package that produces it has moved).
9. **Phase B gate:** `cd libs/cleargbm && maturin develop --release` succeeds; `poetry run pytest -q` succeeds (tests still work because Python interface is unchanged); `make check` on cleargbm and covenant_ml both green.

### Phase C — Delete Python fallback

Now that Rust is a hard dep and always available, the fallback comes out.

Steps:

1. Delete `python/cleargbm/_hooks_binning.py`, `_hooks_compute.py`, `_hooks_ensemble.py`, `_hooks_histogram.py`, `_hooks_loss.py`, `_hooks_prediction.py`, `_hooks_sigmoid.py`, `_rust_adapters.py`, `_rust_native_adapters.py`.
2. Create `python/cleargbm/_rust.py` — imports `cleargbm_rs`, exposes Protocol-typed function references, no Any and no cast at the boundary. Uses `__import__("cleargbm_rs.cleargbm_rs")` + `getattr` pattern with Protocol annotations (matches Austin's ASGI/framework-boundary rule).
3. Rewrite `python/cleargbm/losses.py` — direct call into `_rust.py`; no hook indirection.
4. Rewrite `python/cleargbm/histogram.py` — retain only functions the Python API still exports (nothing runtime-critical; may end up empty and get deleted).
5. Delete `python/cleargbm/parallel.py`, `python/cleargbm/split.py`, `python/cleargbm/tree.py`.
6. Rewrite `python/cleargbm/ensemble.py`:
   - `def train_gradient_boosting(...)` → calls `_rust.train_gradient_boosting(...)` — 20 lines of marshalling max, no Python loop.
   - `def predict_proba(model, x)` → calls `_rust.predict_proba_model(model, x)`.
7. Rewrite `python/cleargbm/explain.py`:
   - `def get_feature_importances(model)` → calls `_rust.feature_importances(model)`.
   - `def explain_prediction(model, x_single)` → walk the deserialized JSON model via `model.to_json()` for path attribution; Python is fine here because explanation is called per sample, not per training step. If path attribution becomes a bottleneck, lift it into Rust as a follow-up.
8. Update every remaining `python/cleargbm/_types_*.py`: types stay (config + model JSON schema is stable); encode/decode functions stay (still needed at the Python↔Rust boundary).
9. Delete every test file that specifically exercised the deleted Python code: `test_parallel.py`, `test_split.py`, `test_tree.py`, `test_ensemble.py::TestEarlyStopping*`, all `test_scripts_*` bits that exercised the deleted paths, all `test_*hooks*` files.
10. Rewrite `tests/test_losses.py`, `tests/test_histogram.py`, `tests/test_ensemble.py`, `tests/test_explain.py` to exercise the Python-marshaling boundary (call `train_gradient_boosting`, assert output shape/values; call `feature_importances`, assert sums / rankings).
11. Add integration test `tests/test_rust_boundary.py` — every Protocol declared in `_rust.py` must have a live test that asserts the return type and value shape matches the Protocol declaration.
12. Update `scripts/benchmark.py` to expect Rust-only (delete Python-fallback benchmarks; add uint8-bins-vs-int64-bins comparison as follow-up).
13. Update `scripts/guard.py` to match the shape of `libs/covenant_ml/scripts/guard.py` (Python guard runner over cleargbm's Python surface).
14. Update `pyproject.toml`: coverage `source` becomes `["python", "scripts"]`; adjust `tool.ruff.src` and `tool.mypy.files`.
15. **Phase C gate:** `make check` (cleargbm) green — 100% branch coverage across `python/` and `scripts/`.

### Phase D — covenant_ml wrapper on native path

Steps:

1. Rewrite `libs/covenant_ml/src/covenant_ml/backends/cleargbm/backend.py::_ClearGBMPrepared` to wrap `PyGbmModel` directly (the opaque Rust type). Change:
   - `predict_proba(self, x)` — call `cleargbm._rust.predict_proba_model(self._model, x)`.
   - `save(...)` — `path.write_text(self._model.to_json())`.
   - `load(...)` — `_ClearGBMPrepared(cleargbm_rs.PyGbmModel.from_json(path.read_text()))`.
2. Rewrite `ClearGBMBackend.train(...)` — call `cleargbm.ensemble.train_gradient_boosting(...)` (which now dispatches straight into Rust). Uses `_ClearGBMPrepared` around the returned `PyGbmModel`. No `GradientBoostingModel` (Python TypedDict) needed anymore for storage.
3. Rewrite `ClearGBMBackend.get_feature_importances(...)` — `model.feature_importances()` directly on the PyGbmModel wrapped in `_ClearGBMPrepared`.
4. Delete `try_extract_cleargbm_model` if it's no longer needed by downstream code (search the covenant_ml + covenant-radar-api trees first).
5. Update covenant_ml's `tests/backends/cleargbm/` — `_ClearGBMPrepared` shape changed, save/load semantics changed; tests reflect new API.
6. **Phase D gate:** `make check` (covenant_ml) green.

### Phase E — validation

Steps:

1. Re-run the benchmark script (`docs/BENCHMARK_MANIFEST_2026-07-20.json` is the baseline). Post-refactor manifest at `docs/BENCHMARK_MANIFEST_2026-07-21.json` (or equivalent). Expected: cleargbm_hook and cleargbm_native rows collapse to a single row; fit_time matches the current `cleargbm_native` number (~8s). If it's not there, something regressed and we investigate before shipping.
2. Update `libs/cleargbm/README.md` with the honest post-refactor benchmark table.
3. Update `libs/cleargbm/docs/architecture.md` to reflect the merged package.
4. Delete `docs/rust-core-transition-plan.md` if the transition is now complete (or mark it "COMPLETED 2026-07-XX").
5. Commit each phase separately for review.

## Testing strategy

- **Rust unit tests** cover every branch of every new function. Same `cargo llvm-cov` 100% segment threshold as existing code.
- **Rust integration tests** exercise the full training loop with the new serde methods (train → to_json → from_json → predict roundtrip).
- **Python boundary tests** (`tests/test_rust_boundary.py`) exercise every Protocol in `_rust.py` — real Rust calls, real return values, no fakes.
- **No Rust mocks** — Rust code is tested with real Rust. Python testing DI via `_test_hooks.py` is retained for scenarios where a test wants to inject a fake `_rust` module surface (e.g., "what if `predict_proba_model_rs` raises?").
- **No `_default_*` fallback tests** — those functions are deleted; their tests go with them.
- **Coverage config** — `.coveragerc` / `pyproject.toml` `tool.coverage.run.source = ["python", "scripts"]`. Nothing in the deleted-file set counts toward coverage.

## Guards to keep the pattern from re-emerging

- New guard rule `no-python-math-fallback` in `scripts/guard.py`: fires on any `def _default_*` in `python/cleargbm/` — the fallback shape is banned by construction.
- New guard rule `rust-hook-required-if-imported`: any Python module that imports from `python/cleargbm/_rust.py` must delegate 100% to that module (no local implementation).
- Architecture test `tests/architecture/no_python_fallback.py` — walks `python/cleargbm/` AST, asserts no function names match `_default_*` or the retired `_hooks_*` prefix.

## Rollback plan

- Every phase is one commit; `git revert` a phase if a downstream regression surfaces.
- The prior `libs/cleargbm_rs/` layout stays reproducible from git history for the first two weeks post-merge.
- Post-refactor benchmark manifest (Phase E step 1) is the numerical rollback signal — if a phase regresses quality metrics beyond 3 sigma of the prior manifest, roll that phase back.

## What isn't in this plan (called out to prevent scope creep)

- **Column-major sample_bins layout, uint8 bins, SIMD histogram accumulator, leaf-wise growth** — real perf wins over LightGBM but Rust-side engineering. Separate follow-up plan.
- **Explain-in-Rust** — path-attribution + rule extraction. Python is currently fine here (called per sample, not per training step). Lift only if profiling shows it as a bottleneck.
- **`covenant_ml.ClearGBMBackend.get_default_search_space` / `get_focused_search_space`** — these stay Python. They produce sklearn-style search spaces for hyperparameter tuning; nothing to gain from moving them.

## Where this lives

- Refactor plan: this page + `libs/cleargbm/docs/RUST_ONLY_REFACTOR.md` (to be created; mirrors this for consumption inside the cleargbm repo).
- Motivating benchmark: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-20.md`.
- Correctness baseline: `libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md`.
- Architecture claim being invalidated: `libs/cleargbm/docs/architecture.md` (rewrite as part of Phase E).
- Transition plan being retired: `libs/cleargbm/docs/rust-core-transition-plan.md` (marked complete or deleted as part of Phase E).
