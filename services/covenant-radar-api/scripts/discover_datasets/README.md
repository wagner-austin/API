# Dataset Discovery Scanner

Scans external datasets and generates `DatasetConfig` entries for the covenant-radar-api.

## Usage

```bash
# Scan all datasets in data/external
poetry run python -m scripts.discover_datasets

# Scan a custom directory
poetry run python -m scripts.discover_datasets --external-dir /path/to/datasets

# Show detailed info for a specific dataset
poetry run python -m scripts.discover_datasets --detail dataset_name

# Generate DatasetConfig Python code
poetry run python -m scripts.discover_datasets --generate

# Validate discovered configs (check target, values, ratios)
poetry run python -m scripts.discover_datasets --validate

# Enable verbose output
poetry run python -m scripts.discover_datasets -v
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--external-dir` | Path to external datasets directory (default: `data/external`) |
| `--detail NAME` | Show detailed info for a specific dataset folder |
| `--generate` | Generate `DatasetConfig` Python code |
| `--validate` | Validate discovered configs (check target, values, ratios) |
| `-v, --verbose` | Enable verbose output |

## Package Structure

```
scripts/discover_datasets/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── _test_hooks.py       # Dependency injection for testing
├── main.py              # CLI argument parsing and orchestration
├── types.py             # TypedDict definitions
├── scanner.py           # Directory scanning and dataset discovery
├── detection.py         # Target column and value detection
├── encoding.py          # File encoding detection
├── parsers/
│   ├── __init__.py      # Parser exports
│   ├── csv.py           # CSV and .data file parsing
│   ├── arff.py          # ARFF file parsing
│   └── excel.py         # Excel (.xlsx, .xls) parsing
└── README.md            # This file
```

## Module Responsibilities

### `main.py`
- CLI argument parsing with `argparse`
- Output formatting and display
- Config code generation
- Orchestrates discovery workflow

### `scanner.py`
- Scans directories for dataset folders
- Identifies data files by format
- Delegates to appropriate parsers
- Aggregates results into `DiscoverySummary`

### `detection.py`
- Target column detection via pattern matching
- Binary classification detection
- Positive/negative value classification
- Exclude column identification (IDs, names, etc.)
- Class ratio calculation

### `encoding.py`
- File encoding detection
- Supports UTF-8, UTF-8-BOM, Latin-1, CP1252

### `parsers/csv.py`
- CSV parsing with automatic delimiter detection (comma, semicolon, tab)
- Space-delimited `.data` file parsing
- Memory-efficient streaming for large files

### `parsers/arff.py`
- ARFF (Attribute-Relation File Format) parsing
- Handles `@RELATION`, `@ATTRIBUTE`, `@DATA` sections
- Supports numeric, nominal, and string attributes

### `parsers/excel.py`
- Excel file parsing via `openpyxl` (.xlsx) and `xlrd` (.xls)
- Protocol-based typing for external libraries
- Sheet selection and data extraction

### `types.py`
- `TargetColumnCandidate`: Detected target column info
- `DiscoveredDataset`: Complete dataset metadata
- `DiscoverySummary`: Aggregated discovery results
- `DetectionStatus`: Literal type for status values

### `_test_hooks.py`
- Dependency injection for `rich.console.Console`
- Enables testing without console output
- Factory function pattern for swappable dependencies

## Supported File Formats

| Extension | Format | Parser |
|-----------|--------|--------|
| `.csv` | Comma/semicolon/tab-separated | `parsers/csv.py` |
| `.data` | Space-delimited (no header) | `parsers/csv.py` |
| `.arff` | ARFF (Weka format) | `parsers/arff.py` |
| `.xlsx` | Excel 2007+ | `parsers/excel.py` |
| `.xls` | Excel 97-2003 | `parsers/excel.py` |

## Target Detection

The scanner detects target columns by matching column names against known patterns:

- Generic: `target`, `class`, `label`, `y`, `outcome`, `status`, `result`
- Binary: `is_fraud`, `fraud`, `is_default`, `default`, `is_churn`, `churn`
- Bankruptcy: `bankrupt?`, `bankrupt`, `bankruptcy`, `distress`
- Loan/credit: `loan_status`, `credit_risk`, `risk_flag`, `good_bad`
- Delinquency: `seriousdlqin2yrs`
- Medical: `diagnosis`, `disease`, `positive`

Positive (bad) class values detected: `1`, `yes`, `true`, `failed`, `bad`, `fraud`, `churn`, `bankrupt`

Negative (good) class values detected: `0`, `no`, `false`, `alive`, `good`, `pass`, `approved`

## Testing

```bash
# Run all tests
poetry run pytest tests/scripts/discover_datasets/ -v

# Run with coverage
poetry run pytest tests/scripts/discover_datasets/ -v --cov=scripts.discover_datasets --cov-report=term-missing
```

## Type Safety

This module follows strict typing rules:
- No `Any` types
- No `cast()` calls
- No `# type: ignore` comments
- No stub files
- Protocol-based typing for external libraries
