Param(
    [string]$ExternalDir = '',
    [string]$OutputDir = '',
    [string]$Device = 'auto',
    [string]$Dataset = 'taiwan',
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
    $OutputDir = Join-Path (Join-Path (Join-Path $scriptRoot '..') 'models') 'grid_search'
}

$ExternalDir = (Resolve-Path $ExternalDir).Path
New-Item -Force -ItemType Directory -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path

Write-Host "=== XGBoost + MLP Grid Search with Feature Engineering ===" -ForegroundColor Cyan
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
from covenant_ml.types import TrainConfig, MLPConfig

from covenant_radar_api.worker.train_external_job import _load_dataset
from covenant_radar_api.worker import _test_hooks as hooks

external_dir = Path(os.environ["EXTERNAL_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
dataset_name = os.environ["DATASET"]
device = os.environ.get("DEVICE", "auto")
include_mlp = os.environ.get("INCLUDE_MLP", "0") == "1"

output_dir.mkdir(parents=True, exist_ok=True)
results_file = output_dir / "grid_search_results.jsonl"

# Feature engineering functions
def add_ratios(X: NDArray[np.float64], n_original: int) -> NDArray[np.float64]:
    """Add ratio features X[i]/X[j] for all pairs (avoid division by zero)."""
    new_features = []
    for i in range(min(n_original, 10)):  # Limit to first 10 features for ratios
        for j in range(min(n_original, 10)):
            if i != j:
                denom = X[:, j].copy()
                denom[denom == 0] = 1e-8  # Avoid div by zero
                ratio = X[:, i] / denom
                ratio = np.clip(ratio, -100, 100)  # Clip extreme values
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

def add_all_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Add all engineered features."""
    n_orig = X.shape[1]
    X = add_ratios(X, n_orig)
    X = add_products(X, n_orig)
    X = add_log_features(X, n_orig)
    return X

# XGBoost hyperparameter grid
xgb_grid = {
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'reg_alpha': [0.0, 1.0, 5.0],
    'reg_lambda': [1.0, 5.0, 10.0],
}

# MLP configurations to try
mlp_configs = [
    {'hidden_sizes': [64, 32], 'n_epochs': 20, 'learning_rate': 0.001, 'dropout': 0.1},
    {'hidden_sizes': [128, 64, 32], 'n_epochs': 30, 'learning_rate': 0.001, 'dropout': 0.2},
    {'hidden_sizes': [256, 128, 64], 'n_epochs': 50, 'learning_rate': 0.0005, 'dropout': 0.3},
    {'hidden_sizes': [128, 64], 'n_epochs': 30, 'learning_rate': 0.01, 'dropout': 0.1},
]

feature_modes = ['original', 'engineered']

print(f"\\nLoading dataset: {dataset_name}")
dataset = _load_dataset(dataset_name, external_dir)
print(f"Original features: {dataset['x'].shape[1]}")

# Pre-compute engineered features
X_original = dataset['x']
y_labels = dataset['y']
feature_names_original = dataset['feature_names']

X_engineered = add_all_features(X_original)
n_engineered = X_engineered.shape[1] - X_original.shape[1]
feature_names_engineered = list(feature_names_original) + [f"eng_{i}" for i in range(n_engineered)]
print(f"Engineered features: {X_engineered.shape[1]} (+{n_engineered} new)")

# Generate all XGBoost combinations
xgb_keys = list(xgb_grid.keys())
xgb_combinations = list(itertools.product(*[xgb_grid[k] for k in xgb_keys]))
print(f"\\nTotal XGBoost configs: {len(xgb_combinations)}")
print(f"Feature modes: {feature_modes}")
print(f"Total XGBoost runs: {len(xgb_combinations) * len(feature_modes)}")
if include_mlp:
    print(f"MLP configs: {len(mlp_configs)}")
    print(f"Total MLP runs: {len(mlp_configs) * len(feature_modes)}")

best_result = None
best_auc = 0.0
run_count = 0
total_runs = len(xgb_combinations) * len(feature_modes)
if include_mlp:
    total_runs += len(mlp_configs) * len(feature_modes)

print(f"\\n{'='*60}")
print("Starting grid search...")
print(f"{'='*60}\\n")

# Get trainer via registry
reg_factory = hooks.registry_factory
registry: ClassifierRegistry = reg_factory()
trainer = BaseTabularTrainer(registry)

# XGBoost grid search
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
            'early_stopping_rounds': 10,
        }

        run_dir = output_dir / f"xgb_{feat_mode}_d{params['max_depth']}_n{params['n_estimators']}_lr{params['learning_rate']}_a{params['reg_alpha']}_l{params['reg_lambda']}"
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
            }

            with open(results_file, 'a') as f:
                f.write(json.dumps(result) + '\\n')

            if result['best_val_auc'] > best_auc:
                best_auc = result['best_val_auc']
                best_result = result

            status = '*BEST*' if result['best_val_auc'] == best_auc else ''
            print(f"[{run_count}/{total_runs}] XGB {feat_mode[:3]} d={params['max_depth']} n={params['n_estimators']} lr={params['learning_rate']} a={params['reg_alpha']} l={params['reg_lambda']} | AUC={result['best_val_auc']:.4f} {status}")

        except Exception as e:
            print(f"[{run_count}/{total_runs}] FAILED: {e}")

# MLP grid search (if enabled)
if include_mlp:
    for feat_mode in feature_modes:
        X_use = X_engineered if feat_mode == 'engineered' else X_original
        feat_names = feature_names_engineered if feat_mode == 'engineered' else feature_names_original

        for mlp_cfg in mlp_configs:
            run_count += 1

            cfg_mlp: MLPConfig = {
                'learning_rate': mlp_cfg['learning_rate'],
                'batch_size': 512,
                'n_epochs': mlp_cfg['n_epochs'],
                'dropout': mlp_cfg['dropout'],
                'hidden_sizes': tuple(mlp_cfg['hidden_sizes']),
                'precision': 'fp32',
                'optimizer': 'adamw',
                'random_state': 42,
                'early_stopping_patience': 5,
                'device': device,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
            }

            hs_str = '_'.join(map(str, mlp_cfg['hidden_sizes']))
            run_dir = output_dir / f"mlp_{feat_mode}_h{hs_str}_e{mlp_cfg['n_epochs']}_lr{mlp_cfg['learning_rate']}"
            run_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            try:
                outcome = trainer.train(
                    backend='mlp',
                    x_features=X_use,
                    y_labels=y_labels,
                    feature_names=feat_names,
                    config=cfg_mlp,
                    output_dir=run_dir,
                    progress=None,
                )
                elapsed = time.perf_counter() - t0

                result = {
                    'run': run_count,
                    'total': total_runs,
                    'backend': 'mlp',
                    'features': feat_mode,
                    'n_features': X_use.shape[1],
                    'params': mlp_cfg,
                    'best_val_auc': float(outcome['best_val_auc']),
                    'test_auc': float(outcome['test_metrics']['auc']),
                    'test_ppl': float(outcome['test_metrics']['ppl']),
                    'elapsed_sec': elapsed,
                    'early_stopped': outcome['early_stopped'],
                }

                with open(results_file, 'a') as f:
                    f.write(json.dumps(result) + '\\n')

                if result['best_val_auc'] > best_auc:
                    best_auc = result['best_val_auc']
                    best_result = result

                status = '*BEST*' if result['best_val_auc'] == best_auc else ''
                print(f"[{run_count}/{total_runs}] MLP {feat_mode[:3]} h={hs_str} e={mlp_cfg['n_epochs']} | AUC={result['best_val_auc']:.4f} {status}")

            except Exception as e:
                print(f"[{run_count}/{total_runs}] FAILED: {e}")

print(f"\\n{'='*60}")
print("GRID SEARCH COMPLETE")
print(f"{'='*60}")
print(f"\\nBest result:")
print(json.dumps(best_result, indent=2))
print(f"\\nResults saved to: {results_file}")
"@

$env:EXTERNAL_DIR = $ExternalDir
$env:OUTPUT_DIR = $OutputDir
$env:DATASET = $Dataset
$env:DEVICE = $Device
$env:INCLUDE_MLP = if ($IncludeMLP) { "1" } else { "0" }

$py | poetry run python -

Write-Host "`nGrid search complete. Results in $OutputDir"
