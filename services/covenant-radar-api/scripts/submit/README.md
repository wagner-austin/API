# Submit CLI Package

Backend-agnostic CLI for Kaggle competition submission pipelines. Trains models on time-series data and generates prediction CSV files.

Supports four ML backends:
- **LightGBM**: Fast gradient boosting (default)
- **XGBoost**: Gradient boosting with feature importance
- **MLP**: Neural network with configurable architecture
- **LSTM**: Recurrent network for temporal sequences

## Usage

```bash
# Run as module
poetry run python -m scripts.submit [options]

# Examples - Default (LightGBM)
poetry run python -m scripts.submit --train-dir data/train --test-dir data/test -o submission.csv
poetry run python -m scripts.submit -n 100 -l 0.05 --num-leaves 31

# Examples - Other backends
poetry run python -m scripts.submit -b xgboost -n 100 -l 0.1
poetry run python -m scripts.submit -b mlp -n 50 -l 0.001
poetry run python -m scripts.submit -b lstm -n 50 -l 0.001

# With feature engineering options
poetry run python -m scripts.submit --no-rank-features --no-diff-features
poetry run python -m scripts.submit -a statistics  # Aggregation: last, first, mean, statistics
```

## CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backend` | `-b` | `lightgbm` | Backend: `lightgbm`, `xgboost`, `mlp`, `lstm` |
| `--n-estimators` | `-n` | `1000` | Boosting rounds (tree) or epochs (neural) |
| `--learning-rate` | `-l` | `0.05` | Learning rate |
| `--num-leaves` | | `31` | Max leaves per tree (LightGBM only) |
| `--max-depth` | | `-1` | Max tree depth (-1 = unlimited) |
| `--aggregation` | `-a` | `statistics` | Time-series aggregation strategy |
| `--no-rank-features` | | False | Disable per-entity rank features |
| `--no-diff-features` | | False | Disable row-to-row diff features |
| `--train-dir` | | `data/external/amex_train` | Training data directory |
| `--test-dir` | | `data/external/amex_test` | Test data directory |
| `--output` | `-o` | `data/submissions/submission.csv` | Output CSV path |
| `--help` | `-h` | | Show help message |

## Package Structure

```
scripts/submit/
├── __init__.py      # Package exports
├── __main__.py      # Module entry point with CLI
├── _hooks.py        # Dependency injection hooks
├── pipeline.py      # Core pipeline functions
└── README.md        # This file
```

## Module Responsibilities

### __main__.py
CLI entry point with argument parsing:
- `ParsedArgs` - TypedDict for parsed arguments
- `parse_args()` - Command-line argument parser
- `main()` - Entry point with setup and execution

### _hooks.py
Dependency injection for testability:
- `ConsoleProtocol` - Protocol for console output
- `get_console()` - Get console via hook
- `get_project_root()` - Get project root path via hook
- `get_registry()` - Get ClassifierRegistry via hook

### pipeline.py
Core pipeline functions (backend agnostic):
- `SubmitConfig` - TypedDict for pipeline configuration
- `TrainResult` - TypedDict for training results
- `PredictionResult` - TypedDict for prediction results
- `build_dataset_config()` - Build time-series dataset config
- `load_training_data()` - Load and aggregate training data
- `train_model()` - Train model using any backend
- `predict()` - Generate predictions with trained model
- `write_submission()` - Write predictions to CSV
- `run_pipeline()` - Full end-to-end pipeline

## Backend Configuration

Each backend uses optimized default configurations:

### LightGBM
```python
LightGBMConfig(
    device="cpu",
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=1000,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    early_stopping_rounds=10,
)
```

### XGBoost
```python
TrainConfig(
    device="cpu",
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    early_stopping_rounds=10,
)
```

### MLP
```python
MLPConfig(
    device="cpu",
    precision="fp32",
    optimizer="adamw",
    hidden_sizes=(256, 128, 64),
    learning_rate=0.001,
    batch_size=256,
    n_epochs=100,
    dropout=0.3,
    early_stopping_patience=10,
)
```

### LSTM
```python
LSTMConfig(
    device="cpu",
    precision="fp32",
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    bidirectional=True,
    sequence_length=13,
    learning_rate=0.001,
    batch_size=256,
    n_epochs=100,
    early_stopping_patience=10,
)
```

## Time-Series Aggregation

The pipeline supports four aggregation strategies for converting multi-observation time-series to single feature vectors:

| Strategy | Description |
|----------|-------------|
| `last` | Use only the most recent observation per entity |
| `first` | Use only the first observation per entity |
| `mean` | Average all observations per entity |
| `statistics` | Compute mean, std, min, max, last for each feature |

## Data Format

### Training Directory
```
train_dir/
├── data.csv      # Time-series features (entity_id, timestamp, features...)
└── labels.csv    # Labels per entity (entity_id, target)
```

### Test Directory
```
test_dir/
├── data.csv      # Time-series features (entity_id, timestamp, features...)
└── labels.csv    # Labels per entity (optional for test)
```

### Output Format
```csv
customer_ID,prediction
entity_0,0.234567
entity_1,0.891234
...
```

## Dependency Injection

The package uses `_hooks.py` for dependency injection:
- Production: Hooks set to real implementations at startup
- Tests: Hooks set to fakes for isolated testing

This enables 100% test coverage without mocks.

## Testing

### Test Coverage

The package maintains **100% statement and branch coverage**. Tests are organized in `tests/submit/`:

```
tests/submit/
├── __init__.py          # Package marker
├── conftest.py          # Shared fixtures and fake factories
├── test_cli.py          # CLI argument parsing tests
├── test_config.py       # Configuration building tests
├── test_hooks.py        # Dependency injection tests
├── test_pipeline.py     # Pipeline function tests
└── test_integration.py  # End-to-end integration tests
```

### Running Tests

```bash
# Run all tests with coverage
make check

# Run only submit tests
poetry run pytest tests/submit/ -v

# Run with coverage report
poetry run pytest tests/submit/ --cov=scripts/submit --cov-report=html
```

## Type Safety

All modules follow strict typing:
- No `Any` types
- No `cast()` calls
- No `type: ignore` comments
- TypedDict for structured data
- Protocol types for dependency injection
