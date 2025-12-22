# CLI Tools

Command-line tools for local development, optimization, and competition pipelines.

---

## Submit CLI

Backend-agnostic CLI for Kaggle competition submission pipelines. Trains models on time-series data and generates prediction CSV files.

**Usage:**

```bash
# Run as module
poetry run python -m scripts.submit [options]

# Default (LightGBM backend)
poetry run python -m scripts.submit --train-dir data/train --test-dir data/test -o submission.csv
poetry run python -m scripts.submit -n 100 -l 0.05 --num-leaves 31

# Other backends
poetry run python -m scripts.submit -b xgboost -n 100 -l 0.1
poetry run python -m scripts.submit -b mlp -n 50 -l 0.001
poetry run python -m scripts.submit -b lstm -n 50 -l 0.001

# Feature engineering options
poetry run python -m scripts.submit --no-rank-features --no-diff-features
poetry run python -m scripts.submit -a statistics  # Aggregation: last, first, mean, statistics
```

**CLI Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backend` | `-b` | `lightgbm` | Backend: `lightgbm`, `xgboost`, `mlp`, `lstm` |
| `--n-estimators` | `-n` | `1000` | Boosting rounds (tree) or epochs (neural) |
| `--learning-rate` | `-l` | `0.05` | Learning rate |
| `--num-leaves` | | `31` | Max leaves per tree (LightGBM only) |
| `--max-depth` | | `-1` | Max tree depth (-1 = unlimited) |
| `--aggregation` | `-a` | `statistics` | Time-series aggregation: `last`, `first`, `mean`, `statistics` |
| `--no-rank-features` | | False | Disable per-entity rank features |
| `--no-diff-features` | | False | Disable row-to-row diff features |
| `--train-dir` | | `data/external/amex_train` | Training data directory |
| `--test-dir` | | `data/external/amex_test` | Test data directory |
| `--output` | `-o` | `data/submissions/submission.csv` | Output CSV path |

**Backend Configurations:**

| Backend | Format | Best For | Default Config |
|---------|--------|----------|----------------|
| `lightgbm` | `.txt` | Fast training, large datasets | 1000 trees, lr=0.05, leaves=31 |
| `xgboost` | `.ubj` | Tabular data, interpretability | 1000 trees, lr=0.05 |
| `mlp` | `.pt` | Non-linear patterns | 256-128-64, lr=0.001, epochs=100 |
| `lstm` | `.pt` | Temporal sequences | hidden=128, layers=2, bidirectional |

**Time-Series Aggregation:**

| Strategy | Description |
|----------|-------------|
| `last` | Use only the most recent observation per entity |
| `first` | Use only the first observation per entity |
| `mean` | Average all observations per entity |
| `statistics` | Compute mean, std, min, max, last for each feature |

**Data Format:**

Training and test directories should contain:
- `data.csv` - Time-series features (entity_id, timestamp, features...)
- `labels.csv` - Labels per entity (entity_id, target)

**Output Format:**

```csv
customer_ID,prediction
entity_0,0.234567
entity_1,0.891234
...
```

For complete documentation, see [scripts/submit/README.md](../../scripts/submit/README.md).

---

## Optimize CLI

For local development and benchmarking, use the optimization CLI directly:

```bash
# Default (taiwan, 300 trials, full features, cuda)
poetry run python -m scripts.optimize

# Quick test
poetry run python -m scripts.optimize -n 10 -d taiwan -f full --device cpu

# Compare feature presets
poetry run python -m scripts.optimize -c -n 50 -d us

# All standard datasets
poetry run python -m scripts.optimize -a -n 100

# Time-series dataset
poetry run python -m scripts.optimize -d kaggle_amex_default -n 50 -b xgboost

# Multiple backends (compare performance)
poetry run python -m scripts.optimize -b lightgbm,xgboost -n 50
poetry run python -m scripts.optimize -b all -n 100  # Run all four backends
```

**CLI Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backend` | `-b` | `xgboost` | Backend(s): `xgboost`, `lightgbm`, `mlp`, `lstm`, or comma-separated list, or `all` |
| `--dataset` | `-d` | `taiwan` | Dataset: `taiwan`, `us`, `polish`, `kaggle_give_me_credit`, `kaggle_amex_default` |
| `--n-trials` | `-n` | `300` | Number of Optuna trials |
| `--feature-preset` | `-f` | `full` | Feature preset: `none`, `log_only`, `ratios_only`, `full` |
| `--device` | | `cuda` | Device: `cuda`, `cpu`, `auto` |
| `--timeout` | `-t` | None | Timeout per optimization in seconds |
| `--compare-presets` | `-c` | False | Compare all feature presets |
| `--all-datasets` | `-a` | False | Run on all standard datasets |
| `--save-model` | `-s` | False | Save best model after optimization |
| `--verbose` | `-v` | False | Enable debug logging |

**Multi-Backend Mode:**

When specifying multiple backends, the CLI runs each backend sequentially and prints a comparison summary:

```bash
# Compare two backends
poetry run python -m scripts.optimize -b lightgbm,xgboost -n 50

# Compare all four backends
poetry run python -m scripts.optimize -b all -n 50 --device cpu
```

For complete documentation, see [scripts/optimize/README.md](../../scripts/optimize/README.md).

---

## AMEX Competition Pipeline

Ensemble pipeline for Kaggle AMEX Default Prediction competition. Trains multiple ML backends with GroupKFold cross-validation, optimizes ensemble weights using AMEX metric, and generates submission files.

**Usage:**

```bash
# Full pipeline with defaults
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

# Minimal test run
poetry run python -m scripts.amex -b lightgbm -k 2 -n 10
```

**CLI Options:**

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

**Pipeline Steps:**

1. Load training data with competition features (rank, diff, window)
2. Train each backend with GroupKFold CV (no customer leakage)
3. Collect OOF predictions from all models
4. Optimize ensemble weights to maximize AMEX metric
5. Load test data with same feature engineering
6. Generate weighted ensemble predictions
7. Write Kaggle submission.csv

**Competition Features:**

| Feature Type | Output per Feature | Description |
|--------------|-------------------|-------------|
| Base | 1 | Original feature |
| Statistics | 4 | mean, std, min, max per entity |
| Rank | 1 | Per-entity percentile (0-1) |
| Diff | 5 | Row-to-row diffs: mean, std, min, max, last |
| Window | 4 per size | Last N observations: mean, std, min, max |

For complete documentation, see [scripts/amex/README.md](../../scripts/amex/README.md).

---

## Dataset Discovery Scanner

Scans external datasets and generates `DatasetConfig` entries.

```bash
# Scan all datasets in data/external
poetry run python -m scripts.discover_datasets

# Scan a custom directory
poetry run python -m scripts.discover_datasets --external-dir /path/to/datasets

# Show detailed info for a specific dataset
poetry run python -m scripts.discover_datasets --detail dataset_name

# Generate DatasetConfig Python code
poetry run python -m scripts.discover_datasets --generate

# Validate discovered configs
poetry run python -m scripts.discover_datasets --validate
```

For complete documentation, see [scripts/discover_datasets/README.md](../../scripts/discover_datasets/README.md).
