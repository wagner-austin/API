# AMEX Competition Pipeline

Ensemble pipeline for the Kaggle AMEX Default Prediction competition. Trains multiple ML backends with GroupKFold cross-validation, optimizes ensemble weights, and generates submission files.

## Features

- **Multi-backend ensemble**: LightGBM, XGBoost (extensible to MLP, LSTM)
- **GroupKFold CV**: Ensures no customer leakage between train/validation splits
- **AMEX metric optimization**: Ensemble weights optimized for competition metric
- **Competition features**: Rank, diff, and window aggregations

## Usage

```bash
# Run with defaults
poetry run python -m scripts.amex

# Custom configuration
poetry run python -m scripts.amex \
    --backends lightgbm,xgboost \
    --n-folds 5 \
    --n-estimators 1000 \
    --learning-rate 0.05 \
    --aggregation statistics \
    --window-sizes 3,6 \
    --output submission.csv

# Minimal run for testing
poetry run python -m scripts.amex -b lightgbm -k 2 -n 10
```

## CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backends` | `-b` | `lightgbm,xgboost` | Comma-separated backends |
| `--n-folds` | `-k` | `5` | Number of CV folds |
| `--n-estimators` | `-n` | `1000` | Boosting rounds |
| `--learning-rate` | `-l` | `0.05` | Learning rate |
| `--aggregation` | `-a` | `statistics` | Aggregation: last, first, mean, statistics |
| `--window-sizes` | `-w` | `3,6` | Comma-separated window sizes |
| `--no-rank-features` | | False | Disable rank features |
| `--no-diff-features` | | False | Disable diff features |
| `--no-window-features` | | False | Disable window features |
| `--train-dir` | | `data/external/amex_train` | Training data directory |
| `--test-dir` | | `data/external/amex_test` | Test data directory |
| `--output` | `-o` | `data/submissions/amex_submission.csv` | Output CSV path |
| `--random-state` | `-s` | `42` | Random seed |

## Package Structure

```
scripts/amex/
├── __init__.py        # Package exports
├── __main__.py        # CLI entry point with argument parsing
├── _hooks.py          # Dependency injection hooks (production)
├── _test_hooks.py     # Fake implementations for testing
├── pipeline.py        # Core pipeline functions
├── types.py           # TypedDict definitions
└── README.md          # This file
```

## Module Responsibilities

### __main__.py
CLI entry point:
- `ParsedArgs` - TypedDict for parsed arguments
- `parse_args()` - Argument parser with validation
- `main()` - Entry point with setup and pipeline execution

### _hooks.py
Dependency injection for production:
- `ConsoleProtocol` / `_RichConsoleAdapter` - Console output
- `RegistryProtocol` / registry hooks - ML backend registry
- `TimeSeriesLoaderCallable` / loader hooks - Dataset loading
- `EnsembleOptimizerCallable` / optimizer hooks - Weight optimization
- `configure_real_scipy()` - Production scipy setup

### _test_hooks.py
Fake implementations for testing:
- `FakeConsole` - Captures output
- `FakeTimeseriesLoader` - Returns synthetic datasets
- `FakePreparedClassifier` / `FakeBackend` / `FakeRegistry` - ML fakes
- `configure_all_fakes()` - Sets all hooks for testing

### pipeline.py
Core pipeline functions:
- `load_training_data()` - Load dataset via hooks
- `load_test_data()` - Load test dataset without labels
- `train_single_model()` - GroupKFold CV for one backend
- `train_all_models()` - Train all specified backends
- `optimize_ensemble()` - Weight optimization via AMEX metric
- `generate_ensemble_predictions()` - Weighted predictions
- `write_submission()` - Create Kaggle submission.csv
- `run_pipeline()` - Full end-to-end pipeline

### types.py
TypedDict definitions:
- `AMEXPipelineConfig` - Full pipeline configuration
- `ModelOOFResult` - Per-model OOF predictions
- `EnsembleResult` - Ensemble weights and scores
- `PipelineResult` - Full pipeline output

## Pipeline Flow

1. **Load training data** with competition features (rank, diff, window)
2. **Train each backend** with GroupKFold CV (no customer leakage)
3. **Collect OOF predictions** from all models
4. **Optimize ensemble weights** to maximize AMEX metric
5. **Load test data** with same feature engineering
6. **Generate weighted predictions** using optimized weights
7. **Write submission.csv** for Kaggle

## Competition Features

| Feature Type | Output per Feature | Description |
|--------------|-------------------|-------------|
| Base | 1 | Original feature |
| Statistics | 4 | mean, std, min, max per entity |
| Rank | 1 | Per-entity percentile (0-1) |
| Diff | 5 | Row-to-row diffs: mean, std, min, max, last |
| Window | 4 per size | Last N observations: mean, std, min, max |

Example with 100 base features, `aggregation=statistics`, rank, diff, window_sizes=(3,6):
- Base stats: 100 * 4 = 400
- Rank: 100 * 1 = 100
- Diff: 100 * 5 = 500
- Window: 100 * 4 * 2 = 800
- **Total: 1800 features**

## Data Format

### Training Directory
```
train_dir/
├── train.csv    # Time-series: customer_ID, S_2, feature_0, feature_1, ...
└── labels.csv   # Labels: customer_ID, target
```

### Test Directory
```
test_dir/
├── test.csv     # Time-series: customer_ID, S_2, feature_0, feature_1, ...
└── labels.csv   # Optional labels for validation
```

### Output Format
```csv
customer_ID,prediction
C_0000001,0.234567
C_0000002,0.891234
...
```

## Dependency Injection

The package uses hooks for dependency injection:
- **Production**: `configure_real_scipy()` sets real implementations
- **Tests**: `configure_all_fakes()` sets fake implementations

This enables 100% test coverage without mocks.

## Testing

```bash
# Run all tests with coverage
make check

# Run only amex tests
poetry run pytest tests/scripts/amex/ -v

# Run with coverage report
poetry run pytest tests/scripts/amex/ --cov=scripts/amex --cov-report=html
```

### Test Coverage

The package maintains **100% statement and branch coverage**:

```
tests/scripts/amex/
├── __init__.py       # Package marker
├── test_hooks.py     # Dependency injection tests
├── test_main.py      # CLI and entry point tests
├── test_pipeline.py  # Pipeline function tests
└── test_types.py     # TypedDict tests
```

## Type Safety

All modules follow strict typing:
- No `Any` types
- No `cast()` calls
- No `type: ignore` comments
- TypedDict for structured data
- Protocol types for dependency injection
