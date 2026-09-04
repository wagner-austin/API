# ClearGBM

*Gradient Boosting You Can See Through*

A strict-typed Python API over a Rust gradient-boosting core. The Python surface is thin: input validation, config marshalling, JSON persistence. The training loop, split finding, tree construction, and prediction all live in Rust ([`cleargbm_rs`](../cleargbm_rs/)) and run in a single native call.

There is **no Python fallback and no hook indirection layer** — Rust is the only compute path.

## Install

```bash
poetry add cleargbm
```

The wheel imports the sibling `cleargbm_rs` extension at load time; if the extension is not available, `import cleargbm.ensemble` raises `ImportError` at the boundary. There is no silent fallback.

## Quick start

```python
import numpy as np
from cleargbm.ensemble import train_gradient_boosting, predict_proba
from cleargbm.types import GradientBoostingConfig

config: GradientBoostingConfig = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_samples_split": 40,
    "min_samples_leaf": 20,
    "max_features": None,
    "max_bins": 64,
    "subsample": 1.0,
    "random_state": 42,
    "track_contributions": False,
    "monotonic_constraints": None,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "n_jobs": 1,
    "early_stopping_rounds": None,
}

model = train_gradient_boosting(
    x_train=x_train,
    y_train=y_train,
    x_val=None,
    y_val=None,
    config=config,
    feature_names=("f0", "f1", "f2"),
)

proba = predict_proba(model, x_test)
```

For JSON persistence and feature importance, call the module-level functions on `cleargbm_rs`:

```python
import cleargbm_rs as native

json_str = native.py_gbm_model_to_json_rs(model)
restored = native.py_gbm_model_from_json_rs(json_str)
importances: list[tuple[str, float]] = native.py_gbm_model_feature_importances_rs(model)
```

## Public API surface

Just two modules:

- `cleargbm.ensemble` — `train_gradient_boosting`, `predict_proba`, `predict_raw`.
- `cleargbm.types` — TypedDicts (`GradientBoostingConfig`, `GradientBoostingModel`, `DecisionTree`, `TreeNode`, ...) + encode/decode/require_* helpers.

Everything else is private (`_rust`, `_types_*`, `_hooks_guard`) or lives in `cleargbm_rs`.

## Design

**Strict typing.** No `Any`, no `cast`, no `type: ignore`. TypedDicts with encode/decode + `require_*` validation everywhere. The dynamic `__import__("cleargbm_rs")` is pinned to Protocol types in `cleargbm._rust` so mypy sees precise signatures.

**Rust is authoritative.** The Rust core stores training data column-first, builds histograms in Rust, finds splits in Rust, constructs trees in Rust, predicts in Rust. Python is a thin API surface, not a computation layer.

**No hooks, no fallbacks.** Prior to 2026-07-21 the codebase had per-primitive hooks that could be pointed at either a Python default or a Rust binding — a "dependency-injection for math primitives" architecture. That's been retired: there's a single Protocol-typed native module accessor at `cleargbm._rust`, and every call goes straight through it.

## Benchmarks

Latest head-to-head vs LightGBM on `american_bankruptcy.csv` with a company-disjoint split (78,682 rows, 18 features, 6.63% positive, 3 seeds, `n_estimators=200`, `max_depth=6`, `max_bins=64`):

| Model | AUC-ROC | AUC-PR | log-loss | Brier | fit_time |
| ----- | ------- | ------ | -------- | ----- | -------- |
| lightgbm | 0.687 ± 0.021 | 0.138 ± 0.015 | 0.229 | 0.059 | **0.87s ± 0.09s** |
| cleargbm | 0.683 ± 0.019 | 0.142 ± 0.018 | 0.230 | 0.058 | 6.88s ± 0.13s (**8.0× slower**) |

**Quality: statistical tie.** All differences smaller than the seed std.
**Speed: 8× slower than LightGBM.** LightGBM has 8+ years of production tuning (SIMD histogram accumulator, uint8 bins, leaf-wise growth, column-major data layout). ClearGBM has none of those yet.

Full report + methodology: [`docs/BENCHMARK_RESULTS_2026-07-21.md`](docs/BENCHMARK_RESULTS_2026-07-21.md).
Prior baseline (pre-Rust-only refactor): [`docs/BENCHMARK_RESULTS_2026-07-20.md`](docs/BENCHMARK_RESULTS_2026-07-20.md).

## Non-goals

- **Not designed to beat LightGBM on speed.** LightGBM will be faster on production workloads until the perf-fix roadmap ships (see the "future work" section in the 2026-07-21 benchmark report).
- **Not designed to beat LightGBM on accuracy.** On the datasets tested so far, quality is a tie within seed noise.
- **Not a batteries-included framework.** No hyperparameter search harness, no persistence infrastructure beyond `to_json`/`from_json`. Consumers (like [`covenant_ml`](../covenant_ml/)) provide those layers.

The value proposition is: strict-typed Python API + Rust-backed correctness + interpretable model shape.

## Requirements

- Python 3.11+
- numpy 2.3.5+
- `cleargbm_rs` (sibling path dep — build with `maturin build --release` from `libs/cleargbm_rs/`, install the wheel)

## Development

```bash
make lint   # scripts.guard + ruff + mypy
make test   # pytest with 100% branch coverage
make check  # lint + test
```
