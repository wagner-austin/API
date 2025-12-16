Param(
    [string]$ExternalDir = '',
    [string]$OutputDir = '',
    [string]$Device = 'auto',
    [int]$EpochsMLP = 3,
    [int]$BatchSizeMLP = 1024
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
    $OutputDir = Join-Path (Join-Path (Join-Path $scriptRoot '..') 'models') 'benchmark'
}

$ExternalDir = (Resolve-Path $ExternalDir).Path
New-Item -Force -ItemType Directory -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path $OutputDir).Path

$datasets = @('us','taiwan','polish')
$backends = @('xgboost','mlp')

function Invoke-BackendRun {
    param(
        [string]$Dataset,
        [string]$Backend
    )

    if ($Backend -eq 'xgboost') {
        $cfg = @{
            dataset = $Dataset
            learning_rate = 0.3
            max_depth = 3
            n_estimators = 10
            subsample = 1.0
            colsample_bytree = 1.0
            random_state = 42
            device = $Device
        }
    }
    else {
        $cfg = @{
            backend = 'mlp'
            dataset = $Dataset
            learning_rate = 0.01
            batch_size = $BatchSizeMLP
            n_epochs = $EpochsMLP
            dropout = 0.1
            hidden_sizes = @(64, 32)
            precision = 'fp32'
            optimizer = 'adamw'
            random_state = 42
            early_stopping_patience = 2
            device = $Device
        }
    }

    $env:BENCH_CONFIG = ($cfg | ConvertTo-Json -Depth 6 -Compress)
    $env:EXTERNAL_DIR = $ExternalDir
    $env:OUTPUT_DIR = $OutputDir

    $py = @'
import json, os, time
from pathlib import Path
from covenant_radar_api.worker.train_external_job import run_external_training

cfg = json.loads(os.environ["BENCH_CONFIG"])
external_dir = Path(os.environ["EXTERNAL_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"]).joinpath(f"{cfg.get('backend','xgboost')}_{cfg['dataset']}")
output_dir.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
res = run_external_training(json.dumps(cfg), external_dir, output_dir)
elapsed = time.perf_counter() - t0

out = {
  "backend": cfg.get("backend", "xgboost"),
  "dataset": cfg["dataset"],
  "elapsed_sec": elapsed,
  "best_val_auc": float(res["best_val_auc"]),
  "early_stopped": bool(res["early_stopped"]),
  "val_ppl": float(res["val_metrics"]["ppl"]),
  "test_ppl": float(res["test_metrics"]["ppl"]),
  "model_path": res["model_path"],
}
print(json.dumps(out, ensure_ascii=False))
'@

    $jsonLine = $py | poetry run python -
    try {
        $obj = $jsonLine | ConvertFrom-Json
    } catch {
        Write-Error "Failed to parse benchmark output: $jsonLine"
        throw
    }

    $fmt = "backend={0} dataset={1} elapsed_sec={2:n2} best_val_auc={3:n4} early_stopped={4} val_ppl={5:n4} test_ppl={6:n4} model_path={7}"
    Write-Host ($fmt -f $obj.backend, $obj.dataset, $obj.elapsed_sec, $obj.best_val_auc, $obj.early_stopped, $obj.val_ppl, $obj.test_ppl, $obj.model_path)

    $logPath = Join-Path $OutputDir 'benchmark_results.jsonl'
    Add-Content -Path $logPath -Value ($jsonLine -join "`n")
}

foreach ($d in $datasets) {
    foreach ($b in $backends) {
        Write-Host "=== Running backend '$b' on dataset '$d' ==="
        Invoke-BackendRun -Dataset $d -Backend $b
    }
}

Write-Host "Benchmark complete. Results written to $OutputDir"

