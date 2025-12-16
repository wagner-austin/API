Param(
    [string]$ExternalDir = '',
    [string]$OutputDir = '',
    [string]$Device = 'auto',
    [string]$Dataset = 'all',
    [switch]$IncludeMLP
)

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Error 'poetry is required on PATH to run this benchmark.'
    exit 1
}

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }

if ([string]::IsNullOrWhiteSpace($ExternalDir)) {
    $ExternalDir = Join-Path (Join-Path (Join-Path $scriptRoot '..') 'data') 'external'
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Join-Path (Join-Path $scriptRoot '..') 'models') 'grid_search_full'
}

$ExternalDir = (Resolve-Path $ExternalDir).Path
New-Item -Force -ItemType Directory -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path

Write-Host "=== Comprehensive XGBoost Grid Search ===" -ForegroundColor Cyan
Write-Host "Dataset: $Dataset"
Write-Host "Device: $Device"
Write-Host "Output: $OutputDir"
Write-Host ""

$py = @"
import json
import os
import time
import itertools
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.backends.registry import ClassifierRegistry
from covenant_ml.base_trainer import BaseTabularTrainer
from covenant_ml.types import TrainConfig

from covenant_radar_api.worker.train_external_job import _load_dataset
from covenant_radar_api.worker import _test_hooks as hooks

external_dir = Path(os.environ["EXTERNAL_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
dataset_arg = os.environ["DATASET"]
device = os.environ.get("DEVICE", "auto")

output_dir.mkdir(parents=True, exist_ok=True)
results_file = output_dir / "grid_search_full_results.jsonl"

# Feature engineering functions
def add_ratios(X: NDArray[np.float64], n_original: int) -> NDArray[np.float64]:
    """Add ratio features X[i]/X[j] for top pairs."""
    new_features = []
    for i in range(min(n_original, 10)):
        for j in range(min(n_original, 10)):
            if i != j:
                denom = X[:, j].copy()
                denom[denom == 0] = 1e-8
                ratio = X[:, i] / denom
                ratio = np.clip(ratio, -100, 100)
                new_features.append(ratio.reshape(-1, 1))
    if new_features:
        return np.hstack([X] + new_features)
    return X

def add_products(X: NDArray[np.float64], n_original: int) -> NDArray[np.float64]:
    """Add product features X[i]*X[j] for pairs."""
    new_features = []
    for i in range(min(n_original, 8)):
        for j in range(i+1, min(n_original, 8)):
            prod = X[:, i] * X[:, j]
            prod = np.clip(prod, -1e6, 1e6)
            new_features.append(prod.reshape(-1, 1))
    if new_features:
        return np.hstack([X] + new_features)
    return X

def add_log_features(X: NDArray[np.float64], n_original: int) -> NDArray[np.float64]:
    """Add log(1 + |x|) * sign(x) for numerical stability."""
    new_features = []
    for i in range(min(n_original, 10)):
        col = X[:, i]
        log_col = np.sign(col) * np.log1p(np.abs(col))
        new_features.append(log_col.reshape(-1, 1))
    if new_features:
        return np.hstack([X] + new_features)
    return X

def add_squares(X: NDArray[np.float64], n_original: int) -> NDArray[np.float64]:
    """Add squared features X[i]^2."""
    new_features = []
    for i in range(min(n_original, 10)):
        sq = X[:, i] ** 2
        sq = np.clip(sq, 0, 1e6)
        new_features.append(sq.reshape(-1, 1))
    if new_features:
        return np.hstack([X] + new_features)
    return X

def add_all_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Add all engineered features."""
    n_orig = X.shape[1]
    X = add_ratios(X, n_orig)
    X = add_products(X, n_orig)
    X = add_log_features(X, n_orig)
    X = add_squares(X, n_orig)
    return X

# Focused XGBoost hyperparameter grid - based on previous best (d=5, n=50, lr=0.3)
# More granularity around winning values, still comprehensive
xgb_grid = {
    'max_depth': [4, 5, 6, 7, 8],           # 5 values - centered on best
    'n_estimators': [50, 75, 100, 150],     # 4 values - medium to high
    'learning_rate': [0.1, 0.2, 0.3, 0.4],  # 4 values - medium to high
    'reg_alpha': [0.0, 0.5, 1.0, 2.0],      # 4 values - low to medium
    'reg_lambda': [0.5, 1.0, 2.0, 5.0],     # 4 values - low to high
}

feature_modes = ['original', 'engineered']

# Select datasets
if dataset_arg == 'all':
    datasets = ['taiwan', 'us', 'polish']
else:
    datasets = [dataset_arg]

# Get trainer
reg_factory = hooks.registry_factory
registry: ClassifierRegistry = reg_factory()
trainer = BaseTabularTrainer(registry)

# Generate all XGBoost combinations
xgb_keys = list(xgb_grid.keys())
xgb_combinations = list(itertools.product(*[xgb_grid[k] for k in xgb_keys]))

print(f"\nXGBoost hyperparameter grid:")
for k, v in xgb_grid.items():
    print(f"  {k}: {v}")
print(f"\nTotal XGBoost configs per dataset: {len(xgb_combinations)}")
print(f"Feature modes: {feature_modes}")
print(f"Datasets: {datasets}")
print(f"Total runs: {len(xgb_combinations) * len(feature_modes) * len(datasets)}")

# Track global best
global_best = None
global_best_auc = 0.0

for dataset_name in datasets:
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load dataset
    dataset = _load_dataset(dataset_name, external_dir)
    X_original = dataset['x']
    y_labels = dataset['y']
    feature_names_original = dataset['feature_names']

    print(f"Samples: {len(y_labels)}, Original features: {X_original.shape[1]}")
    print(f"Class balance: {np.bincount(y_labels)}")

    # Pre-compute engineered features
    X_engineered = add_all_features(X_original)
    n_engineered = X_engineered.shape[1] - X_original.shape[1]
    feature_names_engineered = list(feature_names_original) + [f"eng_{i}" for i in range(n_engineered)]
    print(f"Engineered features: {X_engineered.shape[1]} (+{n_engineered} new)")

    # Track best for this dataset
    dataset_best = None
    dataset_best_auc = 0.0
    run_count = 0
    total_runs = len(xgb_combinations) * len(feature_modes)

    results_dataset_file = output_dir / f"results_{dataset_name}.jsonl"

    for feat_mode in feature_modes:
        X_use = X_engineered if feat_mode == 'engineered' else X_original
        feat_names = feature_names_engineered if feat_mode == 'engineered' else feature_names_original

        for combo in xgb_combinations:
            run_count += 1
            params = dict(zip(xgb_keys, combo))

            cfg: TrainConfig = {
                'learning_rate': params['learning_rate'],
                'max_depth': params['max_depth'],
                'n_estimators': params['n_estimators'],
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42,
                'device': device,
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'early_stopping_rounds': 15,
            }

            run_dir = output_dir / dataset_name / f"xgb_{feat_mode}_d{params['max_depth']}_n{params['n_estimators']}_lr{params['learning_rate']}_a{params['reg_alpha']}_l{params['reg_lambda']}"
            run_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            try:
                outcome = trainer.train(
                    backend='xgboost',
                    x_features=X_use,
                    y_labels=y_labels,
                    feature_names=feat_names,
                    config=cfg,
                    output_dir=run_dir,
                    progress=None,
                )
                elapsed = time.perf_counter() - t0

                result = {
                    'dataset': dataset_name,
                    'run': run_count,
                    'total': total_runs,
                    'backend': 'xgboost',
                    'features': feat_mode,
                    'n_features': X_use.shape[1],
                    'params': params,
                    'best_val_auc': float(outcome['best_val_auc']),
                    'test_auc': float(outcome['test_metrics']['auc']),
                    'test_ppl': float(outcome['test_metrics']['ppl']),
                    'elapsed_sec': elapsed,
                    'early_stopped': outcome['early_stopped'],
                    'best_round': outcome['best_round'],
                }

                # Write to dataset-specific file and global file
                with open(results_dataset_file, 'a') as f:
                    f.write(json.dumps(result) + '\\n')
                with open(results_file, 'a') as f:
                    f.write(json.dumps(result) + '\\n')

                if result['best_val_auc'] > dataset_best_auc:
                    dataset_best_auc = result['best_val_auc']
                    dataset_best = result

                if result['best_val_auc'] > global_best_auc:
                    global_best_auc = result['best_val_auc']
                    global_best = result

                status = '*BEST*' if result['best_val_auc'] == dataset_best_auc else ''
                print(f"[{run_count}/{total_runs}] {feat_mode[:3]} d={params['max_depth']} n={params['n_estimators']} lr={params['learning_rate']} a={params['reg_alpha']} l={params['reg_lambda']} | AUC={result['best_val_auc']:.4f} test={result['test_auc']:.4f} {status}")

            except Exception as e:
                print(f"[{run_count}/{total_runs}] FAILED: {e}")

    print(f"\n--- Best for {dataset_name}: AUC={dataset_best_auc:.4f} ---")
    if dataset_best:
        print(f"Config: {dataset_best['params']}")
        print(f"Features: {dataset_best['features']}")

print(f"\n{'='*60}")
print("GRID SEARCH COMPLETE")
print(f"{'='*60}")
print(f"\nGlobal best result:")
if global_best:
    print(json.dumps(global_best, indent=2))
print(f"\nResults saved to: {results_file}")
"@

$env:EXTERNAL_DIR = $ExternalDir
$env:OUTPUT_DIR = $OutputDir
$env:DATASET = $Dataset
$env:DEVICE = $Device

$py | poetry run python -

Write-Host "`nGrid search complete. Results in $OutputDir"
