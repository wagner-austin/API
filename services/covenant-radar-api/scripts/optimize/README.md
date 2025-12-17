# Optimization CLI Package

Modular CLI for multi-backend hyperparameter optimization using Optuna TPE (Tree-structured Parzen Estimator).

Supports four ML backends:
- **XGBoost**: Gradient boosting with feature importance
- **MLP**: Neural network with configurable architecture
- **LightGBM**: Fast gradient boosting for large datasets
- **LSTM**: Recurrent network for temporal sequences

## Usage

```bash
# Run as module
poetry run python -m scripts.optimize [options]

# Examples - XGBoost (default)
poetry run python -m scripts.optimize -n 50 -d taiwan -f full
poetry run python -m scripts.optimize -c -n 25  # Compare presets
poetry run python -m scripts.optimize -a -n 100 # All datasets

# Examples - Other backends
poetry run python -m scripts.optimize -b mlp -n 50 -d taiwan
poetry run python -m scripts.optimize -b lightgbm -n 50 -d us
poetry run python -m scripts.optimize -b lstm -n 50 -d polish
```

## Package Structure

```
scripts/optimize/
├── __init__.py      # Package exports
├── __main__.py      # Module entry point
├── cli.py           # Argument parsing and types
├── display.py       # Rich console output formatting
├── history.py       # Run history tracking (JSONL)
├── logging_config.py# Logging setup and suppression
├── main.py          # Main entry point with lifecycle
├── modes.py         # Run modes (single, compare, all-datasets)
├── runner.py        # Core optimization execution
├── state.py         # Lifecycle state management
└── README.md        # This file
```

## Module Responsibilities

### cli.py
Argument parsing with strict types. Defines:
- `BackendName` - Literal type for ML backends (`xgboost`, `mlp`, `lightgbm`, `lstm`)
- `DatasetName` - Literal type for valid datasets
- `FeaturePreset` - Literal type for feature presets
- `OptimizeArgs` - Parsed argument container
- `parse_args()` - Main parser function

### display.py
Rich console output formatting:
- `print_config()` - Display run configuration table
- `print_result()` - Display optimization results with history comparison
- `create_result_table()` - Build results summary table
- `create_hyperparams_table()` - Build hyperparameters table
- `create_history_comparison_table()` - Build progression comparison

### history.py
JSONL-based run history tracking with backend-specific entry types:
- `XGBoostHistoryEntry` - TypedDict for XGBoost run metadata
- `MLPHistoryEntry` - TypedDict for MLP run metadata
- `LightGBMHistoryEntry` - TypedDict for LightGBM run metadata
- `LSTMHistoryEntry` - TypedDict for LSTM run metadata
- `UnifiedHistoryEntry` - Discriminated union of all backend entries
- `OptimizationHistory` - Manager class for load/append/query
- Tracks: timestamp, backend, dataset, preset, AUC, hyperparameters, duration

### logging_config.py
Logging configuration:
- `set_verbose_mode()` - Enable/disable verbose output
- `suppress_verbose_logging()` - Suppress Optuna/worker logs

### main.py
Main entry point with proper lifecycle:
- `main()` - Entry point with signal handling
- `run()` - Core execution logic
- Handles KeyboardInterrupt gracefully

### modes.py
Three run modes with backend-specific execution:
- `run_single_with_progress()` - Single optimization with Rich progress bar (all backends)
- `compare_presets()` - Run all 4 presets, rank by AUC (per backend)
- `run_all_datasets()` - Run on taiwan, us, polish datasets (per backend)

Backend-specific run functions:
- `_run_xgboost_with_progress()` - XGBoost optimization
- `_run_mlp_with_progress()` - MLP neural network optimization
- `_run_lightgbm_with_progress()` - LightGBM optimization
- `_run_lstm_with_progress()` - LSTM optimization

### runner.py
Core optimization execution with backend-specific runners:
- `RunResult` - Bundles result with history context
- `get_project_root()` - Find service root directory
- `run_xgboost()` - Execute XGBoost optimization via hooks
- `run_mlp()` - Execute MLP optimization via hooks
- `run_lightgbm()` - Execute LightGBM optimization via hooks
- `run_lstm()` - Execute LSTM optimization via hooks

### state.py
Lifecycle state management:
- `OptimizationState` - Manages interruption and shutdown callbacks
- `managed_execution()` - Context manager for proper startup/shutdown
- `is_interrupted()` - Check if execution was interrupted
- Signal handler for SIGINT (Ctrl+C)

## History File Format

Runs are stored in `models/optimization_history.jsonl` (one JSON object per line). Each entry has a `backend` discriminator field:

### XGBoost Entry
```json
{
  "backend": "xgboost",
  "timestamp": "2024-01-15T10:30:00Z",
  "dataset": "taiwan",
  "feature_preset": "full",
  "n_trials": 300,
  "n_samples": 6819,
  "n_features": 830,
  "best_val_auc": 0.9423,
  "best_trial_number": 187,
  "best_max_depth": 5,
  "best_n_estimators": 150,
  "best_learning_rate": 0.08,
  "best_reg_alpha": 0.001,
  "best_reg_lambda": 0.1,
  "best_subsample": 0.9,
  "best_colsample_bytree": 0.85,
  "duration_seconds": 342.5
}
```

### MLP Entry
```json
{
  "backend": "mlp",
  "timestamp": "2024-01-15T11:00:00Z",
  "dataset": "taiwan",
  "feature_preset": "full",
  "n_trials": 100,
  "n_samples": 6819,
  "n_features": 830,
  "best_val_auc": 0.9156,
  "best_trial_number": 72,
  "best_n_layers": 2,
  "best_hidden_size": 128,
  "best_learning_rate": 0.001,
  "best_dropout": 0.3,
  "best_batch_size": 64,
  "duration_seconds": 512.3
}
```

### LightGBM Entry
```json
{
  "backend": "lightgbm",
  "timestamp": "2024-01-15T12:00:00Z",
  "dataset": "taiwan",
  "feature_preset": "full",
  "n_trials": 200,
  "n_samples": 6819,
  "n_features": 830,
  "best_val_auc": 0.9389,
  "best_trial_number": 143,
  "best_max_depth": 6,
  "best_n_estimators": 200,
  "best_num_leaves": 31,
  "best_learning_rate": 0.05,
  "best_subsample": 0.85,
  "best_colsample_bytree": 0.9,
  "duration_seconds": 198.7
}
```

### LSTM Entry
```json
{
  "backend": "lstm",
  "timestamp": "2024-01-15T13:00:00Z",
  "dataset": "taiwan",
  "feature_preset": "full",
  "n_trials": 50,
  "n_samples": 6819,
  "n_features": 830,
  "best_val_auc": 0.9012,
  "best_trial_number": 38,
  "best_hidden_size": 64,
  "best_num_layers": 2,
  "best_learning_rate": 0.001,
  "best_dropout": 0.2,
  "best_batch_size": 32,
  "duration_seconds": 892.1
}
```

## Dependency Injection

The package uses `scripts/_test_hooks.py` for dependency injection:
- Production: Hooks set to real implementations at startup
- Tests: Hooks set to fakes for isolated testing

This enables 100% test coverage without mocks.

## Adding New Features

1. **New CLI option**: Add to `cli.py` (`OptimizeArgs`, `parse_args`)
2. **New display format**: Add to `display.py`
3. **New run mode**: Add to `modes.py`, wire in `main.py`
4. **New history field**: Update backend-specific `HistoryEntry` in `history.py`
5. **New backend**: Add entry type in `history.py`, runner in `runner.py`, progress in `modes.py`, display in `display.py`

## Type Safety

All modules follow strict typing:
- No `Any` types
- No `cast()` calls
- No `type: ignore` comments
- TypedDict for structured data
- Literal types for constrained values
