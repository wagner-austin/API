# Covenant Radar - Implementation Plan

## Progress Summary

### Original Implementation (Milestones 1-7)

| Milestone | Status | Tests | Coverage |
|-----------|--------|-------|----------|
| 1. covenant_domain | ✅ Complete | 113 | 100% |
| 2. covenant_ml (initial XGBoost scope) | ✅ Complete | 18 | 100% |
| 3. covenant_persistence | ✅ Complete | 96 | 100% |
| 4. Service Shell | ✅ Complete | - | 100% |
| 5. CRUD Endpoints | ✅ Complete | 79 | 100% |
| 6. ML Endpoints | ✅ Complete | 146 | 100% |
| 7. Documentation | ✅ Complete | - | - |

### Expanded Scope (Post-Milestone 7)

| Area | Status | Tests | Coverage |
|------|--------|-------|----------|
| covenant_ml: 5 classifier backends + 4 regressor backends | ✅ Complete | 2069 | 100% |
| covenant_nn: MLP/LSTM classifiers + regressors | ✅ Complete | - | 100% |
| covenant-radar-api: All 7 classifier train-external | ✅ Complete | - | 100% |
| covenant-radar-api: Unified classifier optimize (all 7) | ✅ Complete | - | 100% |
| covenant-radar-api: Classifier explainability (SHAP, permutation, gradient, IG) | ✅ Complete | - | 100% |
| covenant-radar-api: Temporal features (McKinnon PNAS 2024) | ✅ Complete | - | 100% |
| covenant-radar-api: Regression dataset infrastructure | ✅ Complete | - | 100% |
| covenant-radar-api: Regression API endpoints (train, optimize, predict, explain) | ✅ Complete | 2402 | 100% |
| covenant-radar-api: Regression explainability (SHAP, permutation, gradient, IG) | ✅ Complete | - | 100% |
| platform_ml: Regression explainer protocols | ✅ Complete | 296 | 100% |

---

## Overview

Covenant Radar is a loan covenant monitoring and breach prediction system. It stores loan deals and covenant rules, ingests borrower financial data, computes covenant status using deterministic rules, and serves pluggable ML models for predicting future breaches.

The system uses three ML libraries:
- **covenant_ml** — 5 classifier backends (XGBoost, LightGBM, ClearGBM, LogReg, Random Forest) + 4 regressor backends (XGBoost, LightGBM, MLP, LSTM) + datasets, validation, ensemble, calibration, optimization, explainability, temporal features
- **covenant_nn** — 4 PyTorch neural backends (MLP/LSTM classifiers + regressors) + 4 Optuna objectives
- **covenant-radar-api** — FastAPI service wiring libraries to HTTP endpoints with RQ background jobs

**Key Differentiator from Model-Trainer:** This is tabular ML (tree-based + neural), not sequence modeling. Requires a standalone service.

---

## Architecture

### Components

```
libs/
  covenant_domain/       # Pure business logic (TypedDict models, rule engine)
  covenant_ml/           # Pluggable ML framework (classifiers, regressors, optimization)
  covenant_nn/           # PyTorch neural network backends (MLP, LSTM)
  covenant_persistence/  # PostgreSQL repositories
  platform_ml/           # Shared ML protocols, explainer infrastructure, artifact store

services/
  covenant-radar-api/    # FastAPI service (port 8007)
```

### Reused Components

| Existing Library | Usage in Covenant Radar |
|------------------|------------------------|
| `platform_core` | Logging, config, error handling, `DataBankClient`, `json_utils` |
| `platform_workers` | Redis Protocols, RQ harness for training jobs |
| `platform_ml` | `ArtifactStore` for model storage, manifest patterns, explainer protocols |

---

## Strict Typing Standards

All code follows monorepo standards:

```toml
[tool.mypy]
strict = true
disallow_any_unimported = true
disallow_any_expr = true
disallow_any_decorated = true
disallow_any_explicit = true
```

**Banned:**
- `typing.Any`
- `typing.cast`
- `# type: ignore`
- `.pyi` stub files
- `dataclasses`
- `try/except` in core logic (exceptions propagate)

---

## Implementation Milestones

### Milestone 1: covenant_domain Library ✅ COMPLETE

**Files:**
- [x] `libs/covenant_domain/pyproject.toml`
- [x] `libs/covenant_domain/src/covenant_domain/__init__.py`
- [x] `libs/covenant_domain/src/covenant_domain/models.py` — Deal, Covenant, Measurement, CovenantResult TypedDicts
- [x] `libs/covenant_domain/src/covenant_domain/decode.py` — JSON decoders using platform_core.json_utils
- [x] `libs/covenant_domain/src/covenant_domain/encode.py` — TypedDict to JSON encoders
- [x] `libs/covenant_domain/src/covenant_domain/formula_parser.py` — Shunting-yard expression evaluator
- [x] `libs/covenant_domain/src/covenant_domain/rules.py` — Deterministic covenant evaluation
- [x] `libs/covenant_domain/src/covenant_domain/features.py` — Feature extraction for ML

**Verification:** `cd libs/covenant_domain && make check` ✅ 113 tests, 100% coverage

---

### Milestone 2: covenant_ml Library ✅ COMPLETE (expanded significantly)

**Note:** covenant_ml has grown far beyond the original XGBoost-only scope. It is now a full pluggable ML framework.

**Current modules:**
- `backends/` — XGBoost, LightGBM, Random Forest, Logistic Regression, ClearGBM (classifiers + regressors)
- `optimizer/` — Optuna hyperparameter optimization (TPE, grid, random search)
- `ensemble/` — Weighted ensemble optimization (classification + regression)
- `validation/` — Cross-validation runners, splitters, strategies
- `calibration/` — Probability calibration with Platt scaling
- `preprocessing/` — Feature preprocessing pipelines
- `features.py` — Feature engineering for tabular data
- `metrics.py` — Classification and regression metrics
- `trainer.py` — Unified training orchestrator
- `predictor.py` — Prediction interface
- `datasets/` — Dataset loaders (CSV, Parquet, ARFF, time series, NetCDF temporal, regression CSV)
- `datasets/loaders/_netcdf_temporal.py` — McKinnon PNAS 2024 temporal feature extraction (steps 1-3)
- `datasets/loaders/_netcdf_trend_testing.py` — McKinnon rank-trend hypothesis testing (steps 4-7)
- `datasets/loaders/_regression_csv.py` — Regression CSV dataset loader
- `explainers/` — Model explainability (permutation, gradient, integrated gradients, SHAP tree)
- `explainers/registry.py` — Classifier explainer registry
- `explainers/regression_registry.py` — Regression explainer registry with 4 adapters

**Verification:** `cd libs/covenant_ml && make check` ✅ 2069 tests, 100% statement + branch coverage

---

### Milestone 3: covenant_persistence Library ✅ COMPLETE

**Files:**
- [x] `libs/covenant_persistence/src/covenant_persistence/protocols.py` — CursorProtocol, ConnectionProtocol, ConnectCallable
- [x] `libs/covenant_persistence/src/covenant_persistence/repositories.py` — DealRepository, CovenantRepository, MeasurementRepository protocols
- [x] `libs/covenant_persistence/src/covenant_persistence/postgres.py` — PostgreSQL implementations
- [x] `libs/covenant_persistence/src/covenant_persistence/schema.sql` — DDL

**Verification:** `cd libs/covenant_persistence && make check` ✅ 96 tests, 100% coverage

---

### Milestone 4: covenant-radar-api Service Shell ✅ COMPLETE

**Files:**
- [x] `services/covenant-radar-api/pyproject.toml`
- [x] `services/covenant-radar-api/src/covenant_radar_api/main.py` — FastAPI factory
- [x] `services/covenant-radar-api/src/covenant_radar_api/core/config.py` — Settings re-export
- [x] `services/covenant-radar-api/src/covenant_radar_api/core/container.py` — ServiceContainer DI
- [x] `services/covenant-radar-api/src/covenant_radar_api/api/routes/health.py` — /healthz, /readyz
- [x] `services/covenant-radar-api/src/covenant_radar_api/worker/worker_entry.py` — RQ worker entry

**Verification:** `cd services/covenant-radar-api && make check` ✅ 100% coverage

---

### Milestone 5: CRUD API Endpoints ✅ COMPLETE

**Files:**
- [x] `src/covenant_radar_api/api/routes/deals.py`
- [x] `src/covenant_radar_api/api/routes/covenants.py`
- [x] `src/covenant_radar_api/api/routes/measurements.py`
- [x] `src/covenant_radar_api/api/decode.py`

**Verification:** `cd services/covenant-radar-api && make check` ✅ 79 tests, 100% coverage

---

### Milestone 6: Evaluation and ML Endpoints ✅ COMPLETE

**Files:**
- [x] `src/covenant_radar_api/api/routes/evaluate.py`
- [x] `src/covenant_radar_api/api/routes/ml.py`
- [x] `src/covenant_radar_api/worker/train_job.py`
- [x] `src/covenant_radar_api/worker/evaluate_job.py`
- [x] `src/covenant_radar_api/worker_entry.py`
- [x] `src/covenant_radar_api/_test_hooks.py`
- [x] `src/covenant_radar_api/core/_test_hooks.py`

**Verification:** `cd services/covenant-radar-api && make check` ✅ 146 tests, 100% coverage

---

### Milestone 7: Documentation and Demo ✅ COMPLETE

**Files:**
- [x] `services/covenant-radar-api/README.md`
- [x] `docs/services.md` — Added covenant-radar-api entry
- [x] `docs/architecture.md` — Added to all sections

---

## Port Assignment

| Service | Port |
|---------|------|
| covenant-radar-api | 8007 |

---

## Testing Strategy

All tests follow the monorepo pattern:
- 100% statement coverage
- 100% branch coverage
- `fail_under = 100` enforced
- Parallel execution with pytest-xdist

**Unit Tests:** Pure functions in domain/ml libs
**Integration Tests:** Repository tests with test database
**API Tests:** FastAPI TestClient through HTTP boundary

No mocking of core logic - test through actual code paths.

---

## Completed Work: Full Backend Wiring

### Milestone 8: Close Classifier Train-External Gaps ✅ COMPLETE

Wired ClearGBM, LogReg, and RandomForest into the train-external pipeline. All 7 classifier backends now work for external dataset training.

**Key changes:**
- New: `worker/_train_external_parsers.py` — shared config parsers for all 7 backends (eliminated duplication between `decode.py` and `train_external_job.py`)
- Updated: `train_external_job.py` — slimmed to orchestration only, added log/metadata builders and active/meta filenames for 3 new backends
- Updated: `decode.py` — delegates external train parsing to shared module
- Updated: `routes/ml.py` — description mentions all 7 backends

**Verification:** 2146 tests, 100% statement + branch coverage, 0 violations

### Milestone 9: Unified Optimize + Close Classifier Optimize Gaps ✅ COMPLETE

Created LogReg and RandomForest Optuna objectives in covenant_ml. Extended `ClassifierBackend` protocol with `get_default_search_space()`. Replaced 5 per-backend optimize jobs (~3,400 lines) with 1 unified `optimize_job.py` that dispatches through the registry.

**Key changes (covenant_ml):**
- New: `optimizer/objectives/logreg_objective.py`, `random_forest_objective.py`
- Updated: `backends/protocol.py` — added `get_default_search_space()` and `get_focused_search_space()`
- Updated: all 7 backend implementations to implement search space methods

**Key changes (covenant-radar-api):**
- New: `worker/optimize_types.py` — unified progress/result TypedDicts
- New: `worker/optimize_job.py` — single unified optimize job for all 7 classifier backends
- Deleted: `optimize_xgboost_job.py`, `optimize_lightgbm_job.py`, `optimize_mlp_job.py`, `optimize_lstm_job.py`, `optimize_cleargbm_job.py`
- Updated: `decode.py` — replaced 4 backend-specific OptimizeParseResult types with 1 unified type
- Updated: `routes/ml.py` — replaced 4-branch dispatch with single enqueue call

### Milestone 10: Regression Dataset Infrastructure ✅ COMPLETE

Created regression-specific dataset types, registry, CSV loader in covenant_ml. Sourced real public regression datasets (US bankruptcy, financial distress).

**Key changes (covenant_ml):**
- New: `datasets/types.py` — `RegressionDatasetConfig`, `RegressionTargetSpec`, `RegressionDatasetMeta`
- New: `datasets/loaders/_regression_csv.py` — regression CSV dataset loader
- New: `datasets/registry.py` — `RegressionDatasetRegistry` with verified configs
- Updated: `datasets/testing.py` — `FakeRegressionDatasetLoader` using `RegressionDatasetMeta`
- Updated: `datasets/__init__.py` — exports regression types

**Verification:** `cd libs/covenant_ml && make check` ✅ 2069 tests, 100% coverage

### Milestone 11: Regressor API Wiring ✅ COMPLETE

Wired all 4 regressor backends (xgboost_reg, lightgbm_reg, mlp_reg, lstm_reg) through the API with train, optimize, predict, and explain endpoints. Mirrors the full classifier API surface.

**Key changes (platform_ml):**
- New: `explainers/protocol.py` — `RegressorPredictorProtocol`, `RegressionFeatureExplainer` protocol
- New: `explainers/regression_permutation.py` — regression permutation explainer
- Updated: `explainers/types.py` — added `"shap_tree"` to `ExplainerName`

**Key changes (covenant_ml):**
- New: `explainers/regression_registry.py` — `RegressionExplainerRegistry` with 4 adapters (permutation, gradient, IG, shap_tree)
- Updated: `backends/xgboost/regressor.py` — `raw_model` property on `_XGBRegressorPrepared` for SHAP compatibility
- Updated: `backends/lightgbm/regressor.py` — `raw_model` property on `_LGBMRegressorPrepared` for SHAP compatibility

**Key changes (covenant-radar-api):**
- New: `worker/train_external_regression_job.py` — regression training worker
- New: `worker/optimize_regression_job.py` — unified regressor optimize job (all 4 backends)
- New: `worker/optimize_regression_types.py` — regression optimize TypedDicts
- New: `worker/_regression_hooks.py` — regression dataset, regressor, and explainer registry hooks
- New: `worker/explain_regression_job.py` — regression feature importance explanation job
- New: `worker/_optimize_regression_common.py` — shared regression optimize utilities
- Updated: `api/decode.py` — regression train, optimize, predict, and explain request parsing
- Updated: `api/routes/ml.py` — `/ml/train-external-regression`, `/ml/optimize-regression`, `/ml/predict-regression`, `/ml/explain-regression` endpoints

**Regression API endpoints:**

| Endpoint | Description |
|---|---|
| `POST /ml/train-external-regression` | Train regressor on external dataset |
| `POST /ml/optimize-regression` | Optimize regressor hyperparameters |
| `POST /ml/predict-regression` | Predict continuous values |
| `POST /ml/explain-regression` | Compute regression feature importance |

**Regression explainers by backend:**

| Backend | Explainers |
|---|---|
| `xgboost_reg`, `lightgbm_reg` | permutation, shap_tree |
| `mlp_reg`, `lstm_reg` | permutation, gradient, integrated_gradients |

**Verification:** `cd services/covenant-radar-api && make check` ✅ 2402 tests, 100% coverage

### Milestone 12: Documentation Update ✅ COMPLETE

Updated all docs to reflect full backend matrix with all classifier and regressor endpoints wired.

---

## Full Backend Matrix

### Classifiers (7 backends)

| Backend | Package | Train | Optimize | Predict | Explain |
|---------|---------|:-----:|:--------:|:-------:|:-------:|
| xgboost | covenant_ml | ✅ | ✅ | ✅ | ✅ (permutation, shap_tree) |
| lightgbm | covenant_ml | ✅ | ✅ | ✅ | ✅ (permutation, shap_tree) |
| cleargbm | covenant_ml | ✅ | ✅ | ✅ | ✅ (permutation) |
| logreg | covenant_ml | ✅ | ✅ | ✅ | ✅ (permutation) |
| random_forest | covenant_ml | ✅ | ✅ | ✅ | ✅ (permutation) |
| mlp | covenant_nn | ✅ | ✅ | ✅ | ✅ (gradient, IG, permutation) |
| lstm | covenant_nn | ✅ | ✅ | ✅ | ✅ (gradient, IG, permutation) |

### Regressors (4 backends)

| Backend | Package | Train | Optimize | Predict | Explain |
|---------|---------|:-----:|:--------:|:-------:|:-------:|
| xgboost_reg | covenant_ml | ✅ | ✅ | ✅ | ✅ (permutation, shap_tree) |
| lightgbm_reg | covenant_ml | ✅ | ✅ | ✅ | ✅ (permutation, shap_tree) |
| mlp_reg | covenant_nn | ✅ | ✅ | ✅ | ✅ (gradient, IG, permutation) |
| lstm_reg | covenant_nn | ✅ | ✅ | ✅ | ✅ (gradient, IG, permutation) |

---

## Remaining Work

### cleargbm_rs Rust Backend — Full Training Loop Wiring

`cleargbm_rs` (Rust + PyO3) is fully built: 1,454 tests, 100% coverage. Per-hook Rust wiring exists via `use_rust_backend()` (swaps 12 individual hooks). However, the full Rust training loop (`train_gradient_boosting_rs`) and `PyGbmModel` (trained model with `predict_proba`/`predict_raw` in Rust) are not yet wired through to Python. Currently, Python still orchestrates each boosting iteration, crossing the FFI boundary per tree.

**What's needed:**
1. Python stubs for `train_gradient_boosting_rs`, `PyGbmModel`, `predict_proba_model_rs`, `predict_raw_model_rs`
2. Top-level Rust training adapter that calls `train_gradient_boosting_rs` directly (one native call for entire train)
3. Top-level Rust predict adapter using `PyGbmModel.predict_proba()` (one native call for predict)
4. Wire into `covenant_ml`'s ClearGBM backend (or add a `RustClearGBMBackend`)
