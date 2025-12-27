# Unified Pluggable Architecture Plan

## Overview

Refactor covenant-radar-api's worker and runner layers to properly leverage the pluggable registries in covenant_ml. Currently, each backend has its own bespoke job file with duplicated types and code at THREE layers. This plan consolidates to unified workers using the registry-based architecture.

**Goals:**
1. Eliminate code duplication across backend-specific worker jobs
2. Use existing registries from covenant_ml consistently
3. Wire up fine-tuning (currently unused despite existing infrastructure)
4. Make ALL registries selectable (optimizer strategy, validation strategy)
5. Ensure 100% test coverage with strict typing

---

## Current State Analysis

### What Exists in covenant_ml (the library)

| Registry | Location | Registered Items | Protocol |
|----------|----------|------------------|----------|
| Backends | `backends/registry.py` | xgboost, mlp, lstm, lightgbm, cleargbm | `ClassifierBackend` |
| Optimizers | `optimizer/registry.py` | optuna_tpe, random_search, grid_search | `HyperparameterOptimizerProtocol` |
| Explainers | `explainers/registry.py` | permutation, gradient, integrated_gradients, shap_tree | `FeatureExplainer` |
| Fine-tuning | `finetuning/registry.py` | staged, warm_start, iterative_refinement | `FineTuningStrategyProtocol` |
| Validation | `validation/registry.py` | stratified_kfold, group_stratified_kfold, shuffle_split, time_series | `CVSplitterProtocol` |

### What Exists in covenant-radar-api (the service)

**Current 3-layer architecture with duplication:**

```
CLI Layer (UNIFIED - scripts/optimize/main.py)
    └── run_single_with_progress() dispatches by backend name ✓

Runner Layer (5x DUPLICATED - scripts/optimize/runner.py)
    ├── run_xgboost() → _hooks.xgboost_runner()
    ├── run_mlp() → _hooks.mlp_runner()
    ├── run_lightgbm() → _hooks.lightgbm_runner()
    ├── run_lstm() → _hooks.lstm_runner()
    └── run_cleargbm() → _hooks.cleargbm_runner()

Progress Layer (5x DUPLICATED - scripts/optimize/_runners.py)
    ├── _run_xgboost_with_progress()
    ├── _run_mlp_with_progress()
    ├── _run_lightgbm_with_progress()
    ├── _run_lstm_with_progress()
    └── _run_cleargbm_with_progress()

Worker Layer (5x DUPLICATED - src/worker/optimize_*_job.py)
    ├── optimize_xgboost_job.py (573 lines)
    ├── optimize_mlp_job.py (~600 lines)
    ├── optimize_lightgbm_job.py (~550 lines)
    ├── optimize_lstm_job.py (~600 lines)
    └── optimize_cleargbm_job.py (637 lines)
```

**Duplicated types per backend (25 total TypedDicts):**
- `*OptimizeParseResult` (5 variants)
- `*OptimizationResult` (5 variants)
- `*TrialProgressInfo` (5 variants)
- `*PhaseInfo` (5 variants)
- `*LoadingProgressInfo` (5 variants)

**Not wired up (registry infrastructure exists but unused):**
- Fine-tuning strategies (no CLI/API)
- Optimizer strategy selection (hardcoded to Optuna TPE)
- Validation strategy selection (hardcoded to stratified split)

---

## Progress Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Unified Progress Types | PENDING |
| 2 | Unified Optimize Job | PENDING |
| 3 | Optimizer Strategy Integration | PENDING |
| 4 | Validation Strategy Integration | PENDING |
| 5 | Fine-Tuning CLI and Worker | PENDING |
| 6 | CLI Refactor | PENDING |
| 7 | Test Migration | PENDING |
| 8 | Cleanup and Deprecation | PENDING |

---

## Phase 1: Unified Progress Types

Create shared TypedDicts for progress reporting that work across all backends.

### New File: `worker/types.py`

```python
"""Unified types for worker jobs.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
All TypedDicts use total=True for full type safety.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from covenant_ml.datasets.types import LoadPhase
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer.types import (
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types import BackendName

# =============================================================================
# Phase Progress Types
# =============================================================================

OptimizePhase = Literal["loading_data", "feature_engineering", "optimizing", "saving"]


class PhaseProgressInfo(TypedDict, total=True):
    """Progress information when entering a new optimization phase.

    Attributes:
        phase: Current phase of the optimization workflow.
        backend: Name of the ML backend being used.
        dataset: Name of the dataset being processed.
        n_samples: Number of samples (0 during loading).
        n_features: Number of features (0 during loading).
    """

    phase: OptimizePhase
    backend: BackendName
    dataset: str
    n_samples: int
    n_features: int


class LoadingProgressInfo(TypedDict, total=True):
    """Granular progress during dataset loading.

    Attributes:
        dataset: Name of the dataset being loaded.
        phase: Loading sub-phase (scanning, parsing, converting).
        percent_complete: Progress percentage (0.0 to 100.0).
        rows_processed: Number of rows processed so far.
        rows_total: Total rows to process (0 if unknown).
        message: Human-readable progress message.
    """

    dataset: str
    phase: LoadPhase
    percent_complete: float
    rows_processed: int
    rows_total: int
    message: str


class TrialProgressInfo(TypedDict, total=True):
    """Progress information after each optimization trial.

    Attributes:
        backend: Name of the ML backend being optimized.
        trial_number: Current trial number (1-indexed).
        n_trials_total: Total number of trials configured.
        current_value: Objective value from this trial.
        best_value: Best objective value seen so far.
        best_trial: Trial number that achieved best value.
        is_best: Whether this trial is the new best.
    """

    backend: BackendName
    trial_number: int
    n_trials_total: int
    current_value: float
    best_value: float
    best_trial: int
    is_best: bool


# =============================================================================
# Parse Result Types
# =============================================================================


class OptimizeParseResult(TypedDict, total=True):
    """Parsed optimization request (backend-agnostic).

    Attributes:
        backend: ML backend to use for optimization.
        dataset: Dataset name from registry.
        n_trials: Number of optimization trials.
        timeout_seconds: Maximum time in seconds (None for unlimited).
        feature_preset: Feature engineering preset to apply.
        random_state: Random seed for reproducibility.
        optimizer_strategy: Optimization strategy name.
        validation_strategy: Cross-validation strategy name.
    """

    backend: BackendName
    dataset: str
    n_trials: int
    timeout_seconds: int | None
    feature_preset: FeaturePreset
    random_state: int
    optimizer_strategy: Literal["optuna_tpe", "random_search", "grid_search"]
    validation_strategy: Literal[
        "stratified_kfold", "group_stratified_kfold", "shuffle_split", "time_series"
    ]


# =============================================================================
# Result Types
# =============================================================================


class OptimizationResult(TypedDict, total=True):
    """Result of a hyperparameter optimization run (backend-agnostic).

    Attributes:
        backend: ML backend that was optimized.
        status: Completion status.
        dataset: Dataset used for optimization.
        n_samples: Number of samples in dataset.
        n_features: Number of features (after engineering).
        feature_preset: Feature preset that was applied.
        summary: Full optimization summary from covenant_ml.
        duration_seconds: Total wall-clock time.
    """

    backend: BackendName
    status: Literal["complete"]
    dataset: str
    n_samples: int
    n_features: int
    feature_preset: FeaturePreset
    summary: OptimizationSummary
    duration_seconds: float
```

### Encode/Decode Functions

```python
# In worker/types.py (continued)

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
    require_str,
)


def encode_optimization_result(result: OptimizationResult) -> JSONObject:
    """Encode OptimizationResult to JSON-serializable dict.

    Args:
        result: The optimization result to encode.

    Returns:
        JSON-serializable dictionary.
    """
    summary = result["summary"]
    return {
        "backend": result["backend"],
        "status": result["status"],
        "dataset": result["dataset"],
        "n_samples": result["n_samples"],
        "n_features": result["n_features"],
        "feature_preset": result["feature_preset"],
        "duration_seconds": result["duration_seconds"],
        "best_trial_number": summary["best_trial_number"],
        "best_value": summary["best_value"],
        "best_int_params": dict(summary["best_int_params"]),
        "best_float_params": dict(summary["best_float_params"]),
        "best_string_params": dict(summary["best_string_params"]),
        "n_trials_complete": summary["n_trials_complete"],
        "n_trials_pruned": summary["n_trials_pruned"],
        "n_trials_failed": summary["n_trials_failed"],
    }


def decode_optimization_result(data: JSONObject) -> OptimizationResult:
    """Decode OptimizationResult from JSON dict.

    Args:
        data: JSON dictionary to decode.

    Returns:
        Validated OptimizationResult.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    backend_raw = require_str(data, "backend")
    backend = _validate_backend_name(backend_raw)

    status_raw = require_str(data, "status")
    if status_raw != "complete":
        raise JSONTypeError(f"status must be 'complete', got '{status_raw}'")

    # ... full validation implementation
```

---

## Phase 2: Unified Optimize Job

Replace 5 separate `optimize_*_job.py` files with a single `optimize_job.py`.

### New File: `worker/optimize_job.py`

```python
"""Unified hyperparameter optimization job.

Supports all registered backends through the ClassifierRegistry.
Uses HyperparameterOptimizerProtocol for pluggable optimization strategies.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.optimizer.protocol import ObjectiveProtocol
from covenant_ml.optimizer.strategy_protocol import HyperparameterOptimizerProtocol
from covenant_ml.optimizer.types import OptimizationConfig, SearchSpace

from covenant_radar_api.worker._optimize_common import (
    build_optimization_config,
    load_any_dataset,
)
from covenant_radar_api.worker.types import (
    LoadingProgressInfo,
    OptimizationResult,
    OptimizeParseResult,
    PhaseProgressInfo,
    TrialProgressInfo,
)


class PhaseCallbackProtocol(Protocol):
    """Protocol for phase progress callback."""

    def __call__(self, info: PhaseProgressInfo) -> None:
        """Called when entering a new optimization phase."""
        ...


class TrialCallbackProtocol(Protocol):
    """Protocol for trial progress callback."""

    def __call__(self, info: TrialProgressInfo) -> None:
        """Called after each optimization trial."""
        ...


class LoadingCallbackProtocol(Protocol):
    """Protocol for loading progress callback."""

    def __call__(self, info: LoadingProgressInfo) -> None:
        """Called with granular loading progress."""
        ...


def run_optimization(
    config: OptimizeParseResult,
    external_dir: Path,
    output_dir: Path,
    *,
    phase_callback: PhaseCallbackProtocol | None = None,
    trial_callback: TrialCallbackProtocol | None = None,
    loading_callback: LoadingCallbackProtocol | None = None,
) -> OptimizationResult:
    """Run hyperparameter optimization using pluggable backend and strategy.

    Args:
        config: Parsed optimization configuration.
        external_dir: Path to data/external directory with datasets.
        output_dir: Directory to save optimization results.
        phase_callback: Optional callback for phase transitions.
        trial_callback: Optional callback for trial progress.
        loading_callback: Optional callback for loading progress.

    Returns:
        OptimizationResult with best hyperparameters and summary.

    Raises:
        KeyError: If backend or dataset not found in registry.
        ValueError: If configuration is invalid.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    backend_name = config["backend"]
    dataset_name = config["dataset"]

    # Get backend from registry
    backend_registry = hooks.registry_factory()
    backend: ClassifierBackend = backend_registry.get(backend_name)

    # Get optimizer strategy from registry
    optimizer_registry = hooks.optimizer_registry_factory()
    optimizer: HyperparameterOptimizerProtocol = optimizer_registry.get(
        config["optimizer_strategy"]
    )

    # Report loading phase
    if phase_callback is not None:
        phase_callback(PhaseProgressInfo(
            phase="loading_data",
            backend=backend_name,
            dataset=dataset_name,
            n_samples=0,
            n_features=0,
        ))

    # Load dataset
    dataset = load_any_dataset(
        dataset_name,
        external_dir,
        _make_loading_adapter(dataset_name, loading_callback),
    )

    # Report feature engineering phase
    if phase_callback is not None:
        phase_callback(PhaseProgressInfo(
            phase="feature_engineering",
            backend=backend_name,
            dataset=dataset_name,
            n_samples=dataset["meta"]["n_samples"],
            n_features=dataset["meta"]["n_features"],
        ))

    # Get backend-specific objective and search space
    objective, search_space = _create_objective_and_space(
        backend_name,
        dataset,
        config["feature_preset"],
    )

    # Build optimization config
    opt_config = build_optimization_config(
        n_trials=config["n_trials"],
        timeout_seconds=config["timeout_seconds"],
        random_state=config["random_state"],
    )

    # Report optimizing phase
    if phase_callback is not None:
        phase_callback(PhaseProgressInfo(
            phase="optimizing",
            backend=backend_name,
            dataset=dataset_name,
            n_samples=dataset["meta"]["n_samples"],
            n_features=objective.n_features,
        ))

    # Run optimization
    summary = optimizer.optimize(
        x_features=dataset["x"],
        y_labels=dataset["y"],
        feature_names=list(dataset["meta"]["feature_names"]),
        search_space=search_space,
        config=opt_config,
        objective=objective,
        trial_callback=_make_trial_adapter(
            backend_name,
            config["n_trials"],
            trial_callback,
        ),
    )

    # Report saving phase
    if phase_callback is not None:
        phase_callback(PhaseProgressInfo(
            phase="saving",
            backend=backend_name,
            dataset=dataset_name,
            n_samples=dataset["meta"]["n_samples"],
            n_features=objective.n_features,
        ))

    # Save results
    _save_results(output_dir, dataset_name, backend_name, summary)

    return OptimizationResult(
        backend=backend_name,
        status="complete",
        dataset=dataset_name,
        n_samples=dataset["meta"]["n_samples"],
        n_features=objective.n_features,
        feature_preset=config["feature_preset"],
        summary=summary,
        duration_seconds=summary["total_duration_seconds"],
    )
```

---

## Phase 3: Optimizer Strategy Integration

Add hooks for optimizer strategy selection.

### Update: `worker/_test_hooks.py`

```python
# Add to existing _test_hooks.py

from covenant_ml.optimizer.registry import (
    OptimizerStrategyRegistry,
    default_optimizer_registry,
)
from covenant_ml.optimizer.strategy_protocol import (
    HyperparameterOptimizerProtocol,
    OptimizerStrategyName,
)


class OptimizerRegistryFactoryProtocol(Protocol):
    """Protocol for optimizer registry factory function."""

    def __call__(self) -> OptimizerStrategyRegistry:
        """Create an OptimizerStrategyRegistry.

        Returns:
            OptimizerStrategyRegistry instance.
        """
        ...


def _real_optimizer_registry() -> OptimizerStrategyRegistry:
    """Real implementation returning production optimizer registry.

    Returns:
        OptimizerStrategyRegistry with optuna_tpe, random_search, grid_search.
    """
    return default_optimizer_registry()


optimizer_registry_factory: OptimizerRegistryFactoryProtocol = _real_optimizer_registry
```

---

## Phase 4: Validation Strategy Integration

Add hooks for CV strategy selection.

### Update: `worker/_test_hooks.py`

```python
# Add to existing _test_hooks.py

from covenant_ml.validation.registry import (
    CVSplitterRegistry,
    default_cv_registry,
)
from covenant_ml.validation.protocol import (
    CVSplitterProtocol,
    CVStrategyName,
)


class CVRegistryFactoryProtocol(Protocol):
    """Protocol for CV registry factory function."""

    def __call__(self) -> CVSplitterRegistry:
        """Create a CVSplitterRegistry.

        Returns:
            CVSplitterRegistry instance.
        """
        ...


def _real_cv_registry() -> CVSplitterRegistry:
    """Real implementation returning production CV registry.

    Returns:
        CVSplitterRegistry with all validation strategies.
    """
    return default_cv_registry()


cv_registry_factory: CVRegistryFactoryProtocol = _real_cv_registry
```

---

## Phase 5: Fine-Tuning CLI and Worker

Wire up the existing fine-tuning infrastructure.

### New File: `worker/finetune_job.py`

```python
"""Fine-tuning worker job.

Supports all registered fine-tuning strategies through FineTuningRegistry.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, TypedDict

from covenant_ml.finetuning.protocol import FineTuningStrategyProtocol
from covenant_ml.finetuning.types import (
    FineTuningConfig,
    FineTuningResult,
    WarmStartConfig,
)
from covenant_ml.optimizer.types import OptimizationSummary


class FineTuneParseResult(TypedDict, total=True):
    """Parsed fine-tuning request.

    Attributes:
        backend: ML backend to fine-tune.
        dataset: Dataset name from registry.
        strategy: Fine-tuning strategy name.
        prior_result_path: Path to prior optimization result (for warm start).
        random_state: Random seed for reproducibility.
    """

    backend: Literal["xgboost", "mlp", "lstm", "lightgbm", "cleargbm"]
    dataset: str
    strategy: Literal["staged", "warm_start", "iterative_refinement"]
    prior_result_path: str | None
    random_state: int


def run_fine_tuning(
    config: FineTuneParseResult,
    external_dir: Path,
    output_dir: Path,
) -> FineTuningResult:
    """Run fine-tuning using pluggable strategy.

    Args:
        config: Parsed fine-tuning configuration.
        external_dir: Path to data/external directory.
        output_dir: Directory to save results.

    Returns:
        FineTuningResult with optimized parameters.

    Raises:
        KeyError: If strategy not found in registry.
        FileNotFoundError: If prior result path doesn't exist.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    # Get fine-tuning strategy from registry
    finetuning_registry = hooks.finetuning_registry_factory()
    strategy: FineTuningStrategyProtocol = finetuning_registry.get(config["strategy"])

    # Load dataset and build objective (similar to optimize_job)
    # ...

    # Load warm-start config if prior result provided
    warm_start: WarmStartConfig | None = None
    if config["prior_result_path"] is not None:
        warm_start = _load_warm_start(config["prior_result_path"])

    # Run fine-tuning
    result = strategy.fine_tune(
        x_features=dataset["x"],
        y_labels=dataset["y"],
        feature_names=feature_names,
        search_space=search_space,
        config=finetuning_config,
        objective=objective,
        warm_start=warm_start,
    )

    return result
```

### New File: `scripts/finetune/__main__.py`

```python
"""Fine-tuning CLI entry point.

Usage:
    python -m covenant_radar_api.scripts.finetune --backend xgboost --dataset taiwan --strategy staged
    python -m covenant_radar_api.scripts.finetune --backend lightgbm --dataset us --strategy warm_start --prior results/xgboost_result.json
"""

from __future__ import annotations

import argparse
import sys

from covenant_radar_api.worker.finetune_job import run_fine_tuning


def main() -> int:
    """CLI entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune ML models")
    parser.add_argument("--backend", required=True, choices=["xgboost", "mlp", "lstm", "lightgbm", "cleargbm"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--strategy", required=True, choices=["staged", "warm_start", "iterative_refinement"])
    parser.add_argument("--prior", help="Path to prior optimization result for warm start")
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    # ... run fine-tuning
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## Phase 6: CLI Refactor

Update the `scripts/optimize` CLI to use unified job.

### Update: `scripts/optimize/__main__.py`

```python
"""Unified optimization CLI entry point.

Usage:
    python -m covenant_radar_api.scripts.optimize --backend xgboost --dataset taiwan --trials 100
    python -m covenant_radar_api.scripts.optimize --backend all --dataset taiwan --trials 50
    python -m covenant_radar_api.scripts.optimize --backend lightgbm,cleargbm --dataset us --trials 100 --strategy random_search
"""

# Replace backend-specific subcommands with unified --backend flag
# Add --optimizer-strategy flag (default: optuna_tpe)
# Add --validation-strategy flag (default: stratified_kfold)
```

---

## Phase 7: Test Migration

Create comprehensive tests for unified workers.

### New Test Structure

```
tests/
├── test_optimize_job.py           # Unified optimize job tests
├── test_finetune_job.py           # Fine-tuning job tests
├── test_worker_types.py           # TypedDict encode/decode tests
├── conftest.py                    # Fake registries and fixtures
└── fakes/
    ├── __init__.py
    ├── fake_backend.py            # FakeClassifierBackend
    ├── fake_optimizer.py          # FakeHyperparameterOptimizer
    ├── fake_finetuning.py         # FakeFineTuningStrategy
    └── fake_cv_splitter.py        # FakeCVSplitter
```

### Test Requirements

1. **No mocks** - Use fake implementations that implement the protocols
2. **No weak assertions** - Assert specific values, not just "truthy"
3. **100% coverage** - Statements and branches
4. **Test actual code** - Fakes should exercise real code paths

### Example Test: `test_optimize_job.py`

```python
"""Tests for unified optimize job.

Uses fake registries and backends - no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from covenant_radar_api.worker import _test_hooks as hooks
from covenant_radar_api.worker.optimize_job import run_optimization
from covenant_radar_api.worker.types import OptimizeParseResult

from tests.fakes.fake_backend import FakeClassifierRegistry
from tests.fakes.fake_optimizer import FakeOptimizerRegistry


@pytest.fixture(autouse=True)
def reset_hooks() -> None:
    """Reset all hooks to real implementations after each test."""
    yield
    hooks.registry_factory = hooks._real_registry
    hooks.optimizer_registry_factory = hooks._real_optimizer_registry


def test_run_optimization_xgboost(tmp_path: Path) -> None:
    """Test optimization with XGBoost backend."""
    # Arrange
    fake_backend_registry = FakeClassifierRegistry()
    fake_optimizer_registry = FakeOptimizerRegistry()
    hooks.registry_factory = lambda: fake_backend_registry
    hooks.optimizer_registry_factory = lambda: fake_optimizer_registry

    config = OptimizeParseResult(
        backend="xgboost",
        dataset="taiwan",
        n_trials=5,
        timeout_seconds=None,
        feature_preset="none",
        random_state=42,
        optimizer_strategy="optuna_tpe",
        validation_strategy="stratified_kfold",
    )

    # Act
    result = run_optimization(
        config,
        external_dir=tmp_path / "external",
        output_dir=tmp_path / "output",
    )

    # Assert
    assert result["backend"] == "xgboost"
    assert result["status"] == "complete"
    assert result["dataset"] == "taiwan"
    assert result["summary"]["n_trials_complete"] == 5
```

---

## Phase 8: Cleanup and Deprecation

Remove deprecated files after migration is complete.

### Files to Delete

```
worker/
├── optimize_xgboost_job.py    # DEPRECATED -> optimize_job.py
├── optimize_lightgbm_job.py   # DEPRECATED -> optimize_job.py
├── optimize_cleargbm_job.py   # DEPRECATED -> optimize_job.py
├── optimize_mlp_job.py        # DEPRECATED -> optimize_job.py
└── optimize_lstm_job.py       # DEPRECATED -> optimize_job.py
```

### Tests to Delete

```
tests/
├── test_optimize_xgboost_job.py   # DEPRECATED
├── test_optimize_lightgbm_job.py  # DEPRECATED
├── test_optimize_mlp_job.py       # DEPRECATED
└── test_optimize_lstm_job.py      # DEPRECATED
```

---

## Validation Checklist

### Code Quality
- [ ] `make check` passes
- [ ] 100% test coverage (statements and branches)
- [ ] No `Any`, `cast()`, or `type: ignore`
- [ ] All TypedDicts use `total=True`
- [ ] All TypedDicts have encode/decode functions
- [ ] All decode functions use `require_*` validation

### Architecture
- [ ] Single `optimize_job.py` replaces 5 backend-specific files
- [ ] All registries wired through `_test_hooks.py`
- [ ] Fine-tuning CLI and worker implemented
- [ ] No hardcoded backend-specific logic in unified jobs

### Testing
- [ ] No mocks - only fakes implementing protocols
- [ ] No weak assertions
- [ ] Tests use `_test_hooks` for dependency injection
- [ ] All fakes in `tests/fakes/` directory
- [ ] Reset fixtures restore real implementations

---

## File Change Summary

### New Files
| File | Purpose |
|------|---------|
| `worker/types.py` | Unified TypedDicts for progress and results |
| `worker/optimize_job.py` | Unified optimization job |
| `worker/finetune_job.py` | Fine-tuning job |
| `scripts/finetune/__main__.py` | Fine-tuning CLI |
| `tests/test_optimize_job.py` | Unified job tests |
| `tests/test_finetune_job.py` | Fine-tuning tests |
| `tests/test_worker_types.py` | TypedDict encode/decode tests |
| `tests/fakes/fake_backend.py` | Fake ClassifierBackend |
| `tests/fakes/fake_optimizer.py` | Fake HyperparameterOptimizer |
| `tests/fakes/fake_finetuning.py` | Fake FineTuningStrategy |
| `tests/fakes/fake_cv_splitter.py` | Fake CVSplitter |

### Modified Files
| File | Changes |
|------|---------|
| `worker/_test_hooks.py` | Add optimizer, validation, fine-tuning registry hooks |
| `worker/_optimize_common.py` | Refactor for unified job |
| `scripts/optimize/__main__.py` | Replace subcommands with unified flags |

### Deleted Files (After Migration)
| File | Reason |
|------|--------|
| `worker/optimize_xgboost_job.py` | Replaced by unified job |
| `worker/optimize_lightgbm_job.py` | Replaced by unified job |
| `worker/optimize_cleargbm_job.py` | Replaced by unified job |
| `worker/optimize_mlp_job.py` | Replaced by unified job |
| `worker/optimize_lstm_job.py` | Replaced by unified job |
| `tests/test_optimize_xgboost_job.py` | Replaced by unified tests |
| `tests/test_optimize_lightgbm_job.py` | Replaced by unified tests |
| `tests/test_optimize_mlp_job.py` | Replaced by unified tests |
| `tests/test_optimize_lstm_job.py` | Replaced by unified tests |

---

## Implementation Order

1. **Phase 1**: Create `worker/types.py` with unified TypedDicts
2. **Phase 2**: Create `worker/optimize_job.py` (unified job)
3. **Phase 3**: Add optimizer strategy hooks to `_test_hooks.py`
4. **Phase 4**: Add validation strategy hooks to `_test_hooks.py`
5. **Phase 5**: Create fine-tuning job and CLI
6. **Phase 6**: Refactor optimize CLI to use unified job
7. **Phase 7**: Migrate tests to unified structure
8. **Phase 8**: Delete deprecated files

Each phase must pass `make check` before proceeding.

---

*Last updated: December 2025*
