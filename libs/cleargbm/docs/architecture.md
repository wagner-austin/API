# ClearGBM Architecture

ClearGBM is a strict-typed Python API over the Rust gradient-boosting core in [`cleargbm_rs`](../../cleargbm_rs/). All computation happens in Rust; the Python surface validates inputs at the boundary, marshals config into the Rust-side dict shape, and hands opaque model handles back to callers.

There is **no Python fallback** and **no per-primitive hook layer**. The Python side is not a math library.

## Package layout

```
libs/cleargbm/
├── pyproject.toml           # poetry backend; path-deps on ../cleargbm_rs
├── Makefile                 # lint + test + check targets
├── docs/
│   ├── architecture.md      # this file
│   ├── VALIDATION_REPORT_2026-07-20.md
│   ├── BENCHMARK_RESULTS_2026-07-20.md   # pre-refactor baseline
│   ├── BENCHMARK_RESULTS_2026-07-21.md   # post-refactor re-run
│   ├── BENCHMARK_MANIFEST_2026-07-20.json
│   ├── BENCHMARK_MANIFEST_2026-07-21.json
│   ├── HANDOFF_BENCHMARK_AND_VALIDATE.md  # completed handoff
│   └── RUST_ONLY_REFACTOR.md              # completed refactor plan
├── scripts/
│   ├── autotune.py          # small hyperparameter sweep (calls train_gradient_boosting)
│   ├── benchmark.py         # synthetic-data timing suite
│   └── guard.py             # monorepo guard runner
├── src/cleargbm/
│   ├── __init__.py          # empty; consumers import from ensemble/types
│   ├── _rust.py             # Protocol-typed __import__("cleargbm_rs")
│   ├── ensemble.py          # public API: train_gradient_boosting, predict_proba, predict_raw
│   ├── types.py             # re-export barrel for _types_*
│   ├── _types_json.py       # JSONValue + narrow_json_to_* validators
│   ├── _types_tree.py       # TreeNode, DecisionTree TypedDicts (decode target for JSON)
│   ├── _types_model.py      # GradientBoostingConfig, GradientBoostingModel, TrainingProgress
│   ├── _types_buffer.py     # buffer TypedDicts (still referenced by covenant_ml SHAP decode)
│   ├── _types_explain.py    # explanation TypedDicts (still referenced by covenant_ml)
│   ├── _types_tuning.py     # autotune TypedDicts (used by scripts/autotune.py)
│   ├── _hooks_guard.py      # guard script hooks (test DI)
│   └── py.typed             # PEP 561 marker
└── tests/
    ├── conftest.py          # config factory + fixtures (no hook-reset)
    ├── test_ensemble.py     # train + predict boundary tests
    ├── test_types.py        # TypedDict encode/decode roundtrips + require_* error paths
    ├── test_scripts_autotune.py
    ├── test_scripts_benchmark.py
    └── test_scripts_guard.py
```

**Total Python surface:** ~855 statements. 100% branch coverage. 182 tests.

## The Rust boundary

Every call into Rust routes through `cleargbm._rust`, which does exactly one dynamic import at module load and pins each callable to a `Protocol` type so mypy sees precise signatures:

```python
# cleargbm/_rust.py (excerpt)
class _TrainProto(Protocol):
    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64] | None,
        y_val: NDArray[np.int64] | None,
        config: dict[str, int | float | bool | list[int] | None],
        feature_names: list[str],
    ) -> PyGbmModelProto: ...

_native_mod = __import__("cleargbm_rs")
train_gradient_boosting_rs: _TrainProto = _native_mod.train_gradient_boosting_rs
```

This pattern satisfies the workspace's strict-typing rules (`disallow_any_expr`, etc.) without requiring `cast` or `type: ignore`. If `cleargbm_rs` is not installed, `_rust.py` raises `ImportError` at cleargbm import time.

## What the Rust core does

Everything performance-critical. Full list of primitives exposed at `cleargbm_rs`:

- `train_gradient_boosting_rs` — single-call full training loop (binning → boosting → early stopping)
- `predict_proba_model_rs` — batch probability prediction
- `predict_raw_model_rs` — batch raw log-odds prediction
- `py_gbm_model_to_json_rs` / `py_gbm_model_from_json_rs` — model serialization
- `py_gbm_model_feature_importances_rs` — split-count feature importance
- `py_gbm_model_n_trees_rs` / `py_gbm_model_n_classes_rs` — introspection
- Subprimitives (`build_tree_rs`, `build_histogram_rs`, `subtract_histogram_rs`, `predict_tree_rs`, `predict_ensemble_rs`, `precompute_feature_bins_rs`, `bin_samples_rs`, `compute_bin_edges_rs`, `sigmoid_rs`, `sigmoid_array_rs`, `binary_log_loss_*_rs`) — mostly unused now that training runs as a single native call; retained for the autotune script and for direct Rust callers.

For the Rust-side architecture, see [`cleargbm_rs/README.md`](../../cleargbm_rs/README.md).

## What the Python surface does NOT do

The following used to live in cleargbm and were retired in Phase C (2026-07-21):

- Per-primitive hook layer (`_hooks_*.py`) — dispatch table for Python-fallback vs Rust-adapter routing. Retired: Rust is the only path.
- `_rust_adapters.py` / `_rust_native_adapters.py` — the wiring that put Rust callables into the hook table. Retired.
- `parallel.py`, `split.py`, `tree.py`, `histogram.py`, `losses.py`, `explain.py`, `buffers.py` — pure-Python implementations of GBM primitives. Retired: consumers who want an ML computation call Rust.
- `_test_hooks.py` re-export barrel — the hooks no longer exist.
- `train_gradient_boosting` (Python-loop version) — was a co-existing Python-orchestrated alternative to the native call. Retired; the sole `train_gradient_boosting` in `ensemble.py` calls Rust directly.

Total code deleted: ~10,000 lines (source + tests). Total tests: 555 → 182 (no coverage loss — the deleted tests exercised code that no longer exists).

## Testing

- **Test suite:** 182 tests, 100% branch coverage. Enforced by `make check`.
- **What's tested:** TypedDict encode/decode roundtrips + `require_*` error paths (test_types.py); `train_gradient_boosting` + `predict_proba` + `predict_raw` on a small synthetic dataset (test_ensemble.py); the three CLI scripts (autotune/benchmark/guard).
- **What's NOT tested here:** Rust correctness. That lives in `cleargbm_rs`'s own cargo tests (1,485 Rust tests, cargo llvm-cov segment coverage 100%).

## Consumers

The primary consumer today is [`covenant_ml`](../../covenant_ml/) via its `ClearGBMBackend`:

```
libs/covenant_ml/src/covenant_ml/backends/cleargbm/backend.py
  ├── imports: cleargbm.ensemble.train_gradient_boosting, predict_proba
  ├── imports: cleargbm.types.GradientBoostingConfig
  └── imports: cleargbm_rs.py_gbm_model_{to_json,from_json,feature_importances,n_trees}_rs
```

`covenant_ml.explainers.cleargbm_shap` also depends on `cleargbm.types` (via a JSON-decode shim that translates Rust-side model JSON into the Python-side `GradientBoostingModel` TypedDict shape that SHAP's tree walker consumes).

## Related documents

- [`BENCHMARK_RESULTS_2026-07-21.md`](BENCHMARK_RESULTS_2026-07-21.md) — post-refactor performance vs LightGBM
- [`VALIDATION_REPORT_2026-07-20.md`](VALIDATION_REPORT_2026-07-20.md) — correctness audit (bit-for-bit checks, sibling subtraction verification)
- [`RUST_ONLY_REFACTOR.md`](RUST_ONLY_REFACTOR.md) — the completed refactor plan
- [`HANDOFF_BENCHMARK_AND_VALIDATE.md`](HANDOFF_BENCHMARK_AND_VALIDATE.md) — the original scoping handoff
