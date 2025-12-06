# PowerShell script to run equivalent commands to the Makefile targets
# for Windows users who don't have GNU Make installed

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Show-Help {
    Write-Host "Available commands:"
    Write-Host "  ./scripts/run.ps1 lint   - Run linting checks (ruff, mypy, guards)"
    Write-Host "  ./scripts/run.ps1 test   - Run tests with coverage"
    Write-Host "  ./scripts/run.ps1 check  - Run lint then tests"
    Write-Host "  ./scripts/run.ps1 clean  - Remove build artifacts and caches"
}

function Invoke-Lint {
    Write-Host "Running ruff linter..."
    ruff check $ProjectRoot
    Write-Host "Auto-fixing format with ruff..."
    ruff format $ProjectRoot
    Write-Host "Running type checking with mypy (strict)..."
    mypy --strict $ProjectRoot\api $ProjectRoot\core
    Write-Host "Running repository guards..."
    python -m tools.guard
}

function Invoke-Test {
    Write-Host "Running tests with coverage..."
    python -I -m pytest -vv -p pytest_cov --cov=api --cov=core --cov-branch --cov-report=term-missing
}

function Invoke-Check {
    Invoke-Lint
    Invoke-Test
}

function Invoke-Clean {
    Write-Host "Cleaning build artifacts..."
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\dist
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\build
    Get-ChildItem -Path $ProjectRoot -Filter "*.egg-info" -Recurse -Directory | Remove-Item -Recurse -Force
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\.coverage
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\htmlcov
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\.pytest_cache
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\.ruff_cache
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProjectRoot\.mypy_cache
    
    Get-ChildItem -Path $ProjectRoot -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Path $ProjectRoot -Filter "*.pyc" -Recurse -File | Remove-Item -Force
}

function Invoke-Build { Write-Host "No build target for API service." }

# Execute the requested command
switch ($Command) {
    "lint" { Invoke-Lint }
    "test" { Invoke-Test }
    "check" { Invoke-Check }
    "clean" { Invoke-Clean }
    default { Show-Help }
}
