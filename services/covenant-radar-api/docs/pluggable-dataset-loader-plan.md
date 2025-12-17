# Pluggable Dataset Loader and Cross-Validation Plan

## Overview

This document outlines the implementation plan for a pluggable dataset loading system and cross-validation support for the covenant-radar-api bankruptcy prediction system.

**Goals:**
1. Support 38+ external datasets with minimal per-dataset code
2. Auto-detect target columns, feature types, and encodings
3. Add pluggable cross-validation strategies: holdout, k-fold, stratified K-fold cross-validation for robust evaluation
4. Integrate cleanly with the pluggable optimizer architecture
5. Generate verified dataset configs from actual data inspection

---

## Progress Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset Types and Registry | ✅ COMPLETE |
| 2 | Generic Loaders (CSV, ARFF) | ✅ COMPLETE |
| 3 | Cross-Validation | ❌ NOT STARTED |
| 4 | Dataset Discovery Script | ❌ NOT STARTED |
| 5 | Service Integration | ✅ COMPLETE |

### What's Done

- `libs/covenant_ml/src/covenant_ml/datasets/` - Full implementation
  - `types.py` - DatasetConfig, LoadedDataset, DatasetMeta, TargetColumnSpec
  - `protocol.py` - DatasetLoaderProtocol
  - `registry.py` - DatasetRegistry, make_default_registry (3 datasets: taiwan, us, polish)
  - `loader.py` - DatasetLoader (unified router)
  - `loaders/csv_loader.py` - CSVLoader
  - `loaders/arff_loader.py` - ARFFLoader
  - `testing.py` - Test utilities
- `scripts/_test_hooks.py` - dataset_loader and dataset_registry_factory hooks
- Service integration - optimize jobs use registry-based loading

### What Remains

1. **Cross-Validation Module** (`libs/covenant_ml/src/covenant_ml/validation/`)
   - Types: CVResult, FoldResult, OOFPredictions, PreprocessingState
   - Preprocessing: StandardScalerPreprocessor with fit/transform separation
   - Splitter: StratifiedKFoldSplitter
   - Runner: CrossValidationRunner with per-fold preprocessing
   - OOF utilities: compute_oof_auc, oof_to_stacking_features

2. **Dataset Discovery Script** (`scripts/discover_datasets/`)
   - Scan data/external/ for datasets
   - Auto-detect target columns
   - Generate verified configs

3. **More Dataset Configs**
   - Currently: 3 datasets (taiwan, us, polish)
   - Goal: 38+ datasets from data/external/

---

**Principles:**
- pluggable
- Strict typing: No `Any`, no casts, no `type: ignore`, no stubs, no `.pyi` files
- Immutable TypedDicts for all configuration and result types
- Protocol-based abstractions for pluggable components
- Test hooks pattern: production sets real implementations, tests set fakes
- 100% test coverage for statements and branches
- No mocks, no weak assertions, no fallbacks, no best-effort handling
- DRY, consistent, modular codebase
- No try/except for recovery - exceptions propagate failures explicitly
- Do all preprocessing inside each fold (scaling, encoding, feature selection). Otherwise you leak
- If you do stacking, generate OOF predictions using CV (predictions for each row from a model that did not train on that row).
- Report both mean and std across folds; std tells you how sensitive you are to the split.

---

## Current State (Updated December 2025)

### Implemented Components

| Component | Location | Status |
|-----------|----------|--------|
| `DatasetConfig` | `covenant_ml/datasets/types.py` | ✅ TypedDict with all config fields |
| `LoadedDataset` | `covenant_ml/datasets/types.py` | ✅ TypedDict with x, y, meta |
| `DatasetRegistry` | `covenant_ml/datasets/registry.py` | ✅ Registry with 3 datasets |
| `DatasetLoader` | `covenant_ml/datasets/loader.py` | ✅ Unified loader (CSV + ARFF) |
| `CSVLoader` | `covenant_ml/datasets/loaders/csv_loader.py` | ✅ Generic CSV loading |
| `ARFFLoader` | `covenant_ml/datasets/loaders/arff_loader.py` | ✅ Generic ARFF loading |
| Test hooks | `scripts/_test_hooks.py` | ✅ dataset_loader, dataset_registry_factory |
| Service integration | `worker/optimize_job.py`, `_optimize_common.py` | ✅ Uses registry-based loading |

### Still Missing

| Component | Issue |
|-----------|-------|
| Cross-validation | No K-fold, only single train/val/test split |
| OOF predictions | No out-of-fold predictions for stacking |
| Preprocessing isolation | No per-fold fit/transform separation |
| Dataset discovery | No script to scan and generate configs |
| More datasets | Only 3 of 38+ datasets configured |

### External Datasets Available

38 datasets in `data/external/`:
- **Large (100K+ samples):** AMEX (5.5M), Lending Club (2.2M), SBA (900K), Home Credit (307K)
- **Medium (10K-100K):** US Bankruptcy (78K), Loan Default (67K), Vehicle Loan (233K)
- **Small (<10K):** Taiwan (6.8K), Polish (13K), German Credit (1K)

Formats: CSV (most), ARFF (Polish), Excel (some), Pickle/NPY (graph datasets)

---

## Implementation Plan

### Phase 1: Dataset Types and Registry

**Location:** `libs/covenant_ml/src/covenant_ml/datasets/`

#### 1.1 Core Types (`datasets/types.py`)

```python
from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray


# File format literals
FileFormat = Literal["csv", "arff", "excel"]

# Encoding literals
FileEncoding = Literal["utf-8", "utf-8-sig", "latin-1", "cp1252"]

# Label type literals (how the target column encodes classes)
LabelType = Literal["binary_int", "binary_str", "multiclass_int", "multiclass_str"]


class TargetColumnSpec(TypedDict, total=True):
    """Specification for the target/label column."""

    column_name: str
    label_type: LabelType
    positive_values: tuple[str | int, ...]  # Values that map to class 1
    negative_values: tuple[str | int, ...]  # Values that map to class 0


class DatasetConfig(TypedDict, total=True):
    """Configuration for loading a single dataset."""

    name: str  # Unique identifier (e.g., "kaggle_company_bankruptcy")
    display_name: str  # Human-readable name
    folder: str  # Subfolder under data/external/
    file_name: str  # Primary data file
    file_format: FileFormat
    encoding: FileEncoding
    target: TargetColumnSpec
    exclude_columns: tuple[str, ...]  # Columns to drop (IDs, dates, names)
    n_samples_expected: int  # For validation
    n_features_expected: int  # For validation
    positive_class_ratio_expected: float  # For validation (e.g., 0.033 for 3.3%)


class DatasetMeta(TypedDict, total=True):
    """Metadata about a loaded dataset."""

    name: str
    n_samples: int
    n_features: int
    n_positive: int
    n_negative: int
    positive_ratio: float
    feature_names: tuple[str, ...]


class LoadedDataset(TypedDict, total=True):
    """A fully loaded and validated dataset ready for ML."""

    meta: DatasetMeta
    x: NDArray[np.float64]  # (n_samples, n_features)
    y: NDArray[np.int64]  # (n_samples,) binary labels


class DatasetValidationError(TypedDict, total=True):
    """Validation error details."""

    dataset_name: str
    error_type: str
    message: str
    expected: str
    actual: str
```

#### 1.2 Dataset Registry (`datasets/registry.py`)

```python
from __future__ import annotations

from typing import Literal

from covenant_ml.datasets.types import DatasetConfig


# All known dataset names as a Literal type for strict typing
KnownDatasetName = Literal[
    # Original datasets
    "taiwan",
    "us",
    "polish",
    # Kaggle bankruptcy/credit datasets
    "kaggle_company_bankruptcy",
    "kaggle_credit_risk",
    "kaggle_credit_default",
    "kaggle_financial_distress",
    "kaggle_us_bankruptcy",
    "kaggle_taiwan_bankruptcy",
    "kaggle_german_credit",
    "kaggle_south_german",
    "kaggle_heloc",
    "kaggle_loan_default",
    "kaggle_vehicle_loan",
    "kaggle_sba_loans",
    "kaggle_credit_card_fraud",
    "kaggle_fico",
    "kaggle_give_me_credit",
    # Add more as validated...
]


class DatasetRegistry:
    """Registry of known dataset configurations.

    Immutable after construction. Thread-safe for reads.
    """

    def __init__(self, configs: tuple[DatasetConfig, ...]) -> None:
        """Initialize with a tuple of dataset configs.

        Args:
            configs: Immutable tuple of DatasetConfig entries.

        Raises:
            ValueError: If duplicate dataset names found.
        """
        self._configs: dict[str, DatasetConfig] = {}
        for cfg in configs:
            name = cfg["name"]
            if name in self._configs:
                raise ValueError(f"Duplicate dataset name: {name}")
            self._configs[name] = cfg

    def get(self, name: str) -> DatasetConfig:
        """Get configuration for a dataset by name.

        Args:
            name: Dataset name (e.g., "kaggle_company_bankruptcy")

        Returns:
            DatasetConfig for the requested dataset.

        Raises:
            KeyError: If dataset not found in registry.
        """
        if name not in self._configs:
            available = ", ".join(sorted(self._configs.keys()))
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")
        return self._configs[name]

    def list_names(self) -> tuple[str, ...]:
        """List all registered dataset names."""
        return tuple(sorted(self._configs.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if dataset is registered."""
        return name in self._configs

    def __len__(self) -> int:
        """Number of registered datasets."""
        return len(self._configs)


def make_default_registry() -> DatasetRegistry:
    """Create registry with all verified dataset configurations."""
    return DatasetRegistry(_VERIFIED_CONFIGS)


# Verified dataset configurations (generated by discovery script, reviewed by human)
_VERIFIED_CONFIGS: tuple[DatasetConfig, ...] = (
    # Taiwan bankruptcy (original)
    DatasetConfig(
        name="taiwan",
        display_name="Taiwan Bankruptcy (Original)",
        folder="taiwan_data",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="Bankrupt?",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=6819,
        n_features_expected=95,
        positive_class_ratio_expected=0.033,
    ),
    # US bankruptcy (original)
    DatasetConfig(
        name="us",
        display_name="US Bankruptcy (Original)",
        folder="us_data",
        file_name="american_bankruptcy.csv",
        file_format="csv",
        encoding="utf-8-sig",
        target=TargetColumnSpec(
            column_name="status_label",
            label_type="binary_str",
            positive_values=("failed",),
            negative_values=("alive",),
        ),
        exclude_columns=("company_name", "year"),
        n_samples_expected=78682,
        n_features_expected=18,
        positive_class_ratio_expected=0.025,
    ),
    # Polish bankruptcy (original)
    DatasetConfig(
        name="polish",
        display_name="Polish Bankruptcy (Original)",
        folder="polish_data",
        file_name="1year.arff",
        file_format="arff",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="class",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=7027,
        n_features_expected=64,
        positive_class_ratio_expected=0.043,
    ),
    # Kaggle company bankruptcy (Taiwan copy)
    DatasetConfig(
        name="kaggle_company_bankruptcy",
        display_name="Kaggle Company Bankruptcy",
        folder="kaggle_company_bankruptcy",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=TargetColumnSpec(
            column_name="Bankrupt?",
            label_type="binary_int",
            positive_values=(1,),
            negative_values=(0,),
        ),
        exclude_columns=(),
        n_samples_expected=6819,
        n_features_expected=95,
        positive_class_ratio_expected=0.033,
    ),
    # Add more verified configs here...
)
```

#### 1.3 Generic Loader Protocol (`datasets/protocol.py`)

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.datasets.types import DatasetConfig, LoadedDataset


class DatasetLoaderProtocol(Protocol):
    """Protocol for dataset loaders."""

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load a dataset from disk.

        Args:
            config: Dataset configuration specifying file, format, target, etc.
            external_dir: Root directory containing dataset folders.

        Returns:
            LoadedDataset with features, labels, and metadata.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If data doesn't match expected format.
        """
        ...


class DatasetValidatorProtocol(Protocol):
    """Protocol for dataset validators."""

    def validate(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> None:
        """Validate a dataset without fully loading it.

        Args:
            config: Dataset configuration to validate.
            external_dir: Root directory containing dataset folders.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            ValueError: If validation fails (wrong shape, missing columns, etc.)
        """
        ...
```

---

### Phase 2: Generic Loaders

**Location:** `libs/covenant_ml/src/covenant_ml/datasets/`

#### 2.1 CSV Loader (`datasets/loaders/csv_loader.py`)

```python
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    LoadedDataset,
    TargetColumnSpec,
)


class CSVLoader:
    """Loads CSV datasets into LoadedDataset format.

    Handles:
    - Multiple encodings (utf-8, utf-8-sig, latin-1)
    - Target column detection and label encoding
    - Column exclusion
    - Numeric conversion with NaN/inf handling
    """

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load CSV dataset.

        Args:
            config: Dataset configuration.
            external_dir: Root directory for datasets.

        Returns:
            LoadedDataset ready for ML.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If columns missing or data invalid.
        """
        file_path = external_dir / config["folder"] / config["file_name"]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Read raw data
        headers, rows = self._read_csv(file_path, config["encoding"])

        # Find target column index
        target_spec = config["target"]
        target_idx = self._find_column_index(headers, target_spec["column_name"])

        # Find columns to exclude
        exclude_set = set(config["exclude_columns"])
        exclude_set.add(target_spec["column_name"])  # Target is not a feature

        # Build feature column indices
        feature_indices: list[int] = []
        feature_names: list[str] = []
        for i, header in enumerate(headers):
            if header not in exclude_set:
                feature_indices.append(i)
                feature_names.append(header)

        # Convert to arrays
        n_samples = len(rows)
        n_features = len(feature_indices)

        x_array = np.zeros((n_samples, n_features), dtype=np.float64)
        y_array = np.zeros(n_samples, dtype=np.int64)

        for row_idx, row in enumerate(rows):
            # Extract features
            for feat_idx, col_idx in enumerate(feature_indices):
                value = row[col_idx] if col_idx < len(row) else ""
                x_array[row_idx, feat_idx] = self._parse_float(value)

            # Extract and encode label
            target_value = row[target_idx] if target_idx < len(row) else ""
            y_array[row_idx] = self._encode_label(target_value, target_spec)

        # Compute metadata
        n_positive = int(np.sum(y_array))
        n_negative = n_samples - n_positive
        positive_ratio = n_positive / n_samples if n_samples > 0 else 0.0

        meta = DatasetMeta(
            name=config["name"],
            n_samples=n_samples,
            n_features=n_features,
            n_positive=n_positive,
            n_negative=n_negative,
            positive_ratio=positive_ratio,
            feature_names=tuple(feature_names),
        )

        return LoadedDataset(meta=meta, x=x_array, y=y_array)

    def _read_csv(
        self,
        file_path: Path,
        encoding: str,
    ) -> tuple[list[str], list[list[str]]]:
        """Read CSV file and return headers and rows."""
        rows: list[list[str]] = []
        headers: list[str] = []

        with open(file_path, encoding=encoding, newline="") as f:
            reader = csv.reader(f)
            for line_values in reader:
                if not headers:
                    headers = [h.strip() for h in line_values]
                    continue
                rows.append(line_values)

        if not rows:
            raise ValueError(f"No data rows found in {file_path}")

        return headers, rows

    def _find_column_index(self, headers: list[str], column_name: str) -> int:
        """Find column index by name (case-insensitive)."""
        column_lower = column_name.lower()
        for idx, header in enumerate(headers):
            if header.lower() == column_lower:
                return idx
        raise ValueError(
            f"Column '{column_name}' not found. Available: {headers}"
        )

    def _parse_float(self, value: str) -> float:
        """Parse string to float, handling missing/invalid values."""
        stripped = value.strip()
        if stripped in ("", "?", "NA", "NaN", "nan", "None", "N/A", "n/a"):
            return 0.0

        # Remove thousands separators
        cleaned = stripped.replace(",", "")

        result = float(cleaned)

        # Replace inf/nan with 0.0
        if not np.isfinite(result):
            return 0.0

        return result

    def _encode_label(
        self,
        value: str,
        spec: TargetColumnSpec,
    ) -> int:
        """Encode label value to 0/1."""
        # Try as-is first
        stripped = value.strip()

        # Check positive values
        for pos_val in spec["positive_values"]:
            if isinstance(pos_val, int):
                # Try numeric comparison
                try:
                    if int(float(stripped)) == pos_val:
                        return 1
                except (ValueError, TypeError):
                    pass
            elif stripped.lower() == str(pos_val).lower():
                return 1

        # Check negative values
        for neg_val in spec["negative_values"]:
            if isinstance(neg_val, int):
                try:
                    if int(float(stripped)) == neg_val:
                        return 0
                except (ValueError, TypeError):
                    pass
            elif stripped.lower() == str(neg_val).lower():
                return 0

        raise ValueError(
            f"Unknown label value: '{stripped}'. "
            f"Expected positive={spec['positive_values']} or negative={spec['negative_values']}"
        )
```

#### 2.2 ARFF Loader (`datasets/loaders/arff_loader.py`)

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    LoadedDataset,
    TargetColumnSpec,
)


class ARFFLoader:
    """Loads ARFF (Weka) datasets into LoadedDataset format.

    ARFF format:
    - @relation <name>
    - @attribute <name> <type>
    - @data
    - comma-separated values
    """

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load ARFF dataset.

        Args:
            config: Dataset configuration.
            external_dir: Root directory for datasets.

        Returns:
            LoadedDataset ready for ML.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If format invalid.
        """
        file_path = external_dir / config["folder"] / config["file_name"]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Parse ARFF
        attributes, data_rows = self._parse_arff(file_path, config["encoding"])

        # Find target column
        target_spec = config["target"]
        target_idx = self._find_attribute_index(attributes, target_spec["column_name"])

        # Build feature indices (exclude target and any exclude_columns)
        exclude_set = set(config["exclude_columns"])
        exclude_set.add(target_spec["column_name"].lower())

        feature_indices: list[int] = []
        feature_names: list[str] = []
        for i, attr_name in enumerate(attributes):
            if attr_name.lower() not in exclude_set:
                feature_indices.append(i)
                feature_names.append(attr_name)

        # Convert to arrays
        n_samples = len(data_rows)
        n_features = len(feature_indices)

        x_array = np.zeros((n_samples, n_features), dtype=np.float64)
        y_array = np.zeros(n_samples, dtype=np.int64)

        for row_idx, row in enumerate(data_rows):
            # Extract features
            for feat_idx, col_idx in enumerate(feature_indices):
                value = row[col_idx] if col_idx < len(row) else ""
                x_array[row_idx, feat_idx] = self._parse_float(value)

            # Extract and encode label
            target_value = row[target_idx] if target_idx < len(row) else ""
            y_array[row_idx] = self._encode_label(target_value, target_spec)

        # Compute metadata
        n_positive = int(np.sum(y_array))
        n_negative = n_samples - n_positive
        positive_ratio = n_positive / n_samples if n_samples > 0 else 0.0

        meta = DatasetMeta(
            name=config["name"],
            n_samples=n_samples,
            n_features=n_features,
            n_positive=n_positive,
            n_negative=n_negative,
            positive_ratio=positive_ratio,
            feature_names=tuple(feature_names),
        )

        return LoadedDataset(meta=meta, x=x_array, y=y_array)

    def _parse_arff(
        self,
        file_path: Path,
        encoding: str,
    ) -> tuple[list[str], list[list[str]]]:
        """Parse ARFF file and return attribute names and data rows."""
        attributes: list[str] = []
        data_rows: list[list[str]] = []
        in_data = False

        with open(file_path, encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue

                if line.lower() == "@data":
                    in_data = True
                    continue

                if not in_data:
                    # Parse attribute definition
                    if line.lower().startswith("@attribute"):
                        parts = line.split()
                        if len(parts) >= 2:
                            attr_name = parts[1]
                            attributes.append(attr_name)
                else:
                    # Parse data row
                    values = line.split(",")
                    data_rows.append([v.strip() for v in values])

        if not data_rows:
            raise ValueError(f"No data rows found in {file_path}")

        return attributes, data_rows

    def _find_attribute_index(self, attributes: list[str], name: str) -> int:
        """Find attribute index by name (case-insensitive)."""
        name_lower = name.lower()
        for idx, attr in enumerate(attributes):
            if attr.lower() == name_lower:
                return idx
        raise ValueError(f"Attribute '{name}' not found. Available: {attributes}")

    def _parse_float(self, value: str) -> float:
        """Parse string to float, handling missing/invalid values."""
        stripped = value.strip()
        if stripped in ("", "?", "NA", "NaN", "nan", "None"):
            return 0.0

        result = float(stripped)
        if not np.isfinite(result):
            return 0.0

        return result

    def _encode_label(self, value: str, spec: TargetColumnSpec) -> int:
        """Encode label value to 0/1."""
        stripped = value.strip()

        for pos_val in spec["positive_values"]:
            if isinstance(pos_val, int):
                try:
                    if int(float(stripped)) == pos_val:
                        return 1
                except (ValueError, TypeError):
                    pass
            elif stripped.lower() == str(pos_val).lower():
                return 1

        for neg_val in spec["negative_values"]:
            if isinstance(neg_val, int):
                try:
                    if int(float(stripped)) == neg_val:
                        return 0
                except (ValueError, TypeError):
                    pass
            elif stripped.lower() == str(neg_val).lower():
                return 0

        raise ValueError(
            f"Unknown label value: '{stripped}'. "
            f"Expected positive={spec['positive_values']} or negative={spec['negative_values']}"
        )
```

#### 2.3 Unified Loader (`datasets/loader.py`)

```python
from __future__ import annotations

from pathlib import Path

from covenant_ml.datasets.loaders.arff_loader import ARFFLoader
from covenant_ml.datasets.loaders.csv_loader import CSVLoader
from covenant_ml.datasets.types import DatasetConfig, LoadedDataset


class DatasetLoader:
    """Unified dataset loader supporting multiple formats.

    Routes loading to format-specific loaders based on config.
    """

    def __init__(self) -> None:
        """Initialize with format-specific loaders."""
        self._csv_loader = CSVLoader()
        self._arff_loader = ARFFLoader()

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load dataset based on format specified in config.

        Args:
            config: Dataset configuration.
            external_dir: Root directory for datasets.

        Returns:
            LoadedDataset ready for ML.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If format unsupported or data invalid.
        """
        file_format = config["file_format"]

        if file_format == "csv":
            return self._csv_loader.load(config, external_dir)
        if file_format == "arff":
            return self._arff_loader.load(config, external_dir)
        if file_format == "excel":
            raise ValueError("Excel format not yet implemented")

        raise ValueError(f"Unsupported file format: {file_format}")


def create_dataset_loader() -> DatasetLoader:
    """Factory function for creating dataset loader."""
    return DatasetLoader()
```

---

### Phase 3: Cross-Validation

**Location:** `libs/covenant_ml/src/covenant_ml/validation/`

**Critical Design Principles:**
1. **All preprocessing inside each fold** - Scaling, encoding, feature selection MUST be fit on training fold only, then applied to validation fold. Otherwise you leak information from validation into training.
2. **OOF predictions for stacking** - Generate out-of-fold predictions where each row's prediction comes from a model that did NOT train on that row.
3. **Report mean ± std** - Standard deviation across folds reveals sensitivity to the split. High std = unstable model.

#### 3.1 Cross-Validation Types (`validation/types.py`)

```python
from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray


CVStrategy = Literal["holdout", "kfold", "stratified_kfold", "repeated_kfold"]


class FoldIndices(TypedDict, total=True):
    """Train/validation indices for a single fold."""

    fold_number: int
    train_indices: NDArray[np.int64]
    val_indices: NDArray[np.int64]


class CVSplitResult(TypedDict, total=True):
    """Result of cross-validation splitting."""

    strategy: CVStrategy
    n_folds: int
    n_samples: int
    folds: tuple[FoldIndices, ...]
    test_indices: NDArray[np.int64] | None  # Holdout test set if applicable


class FoldResult(TypedDict, total=True):
    """Result from evaluating a single fold.

    Includes OOF predictions for the validation set of this fold.
    These can be concatenated across folds to get full OOF predictions.
    """

    fold_number: int
    train_auc: float
    val_auc: float
    n_train: int
    n_val: int
    val_indices: NDArray[np.int64]  # Which rows these predictions are for
    val_predictions: NDArray[np.float64]  # OOF predictions (probabilities)
    val_labels: NDArray[np.int64]  # Actual labels for verification


class OOFPredictions(TypedDict, total=True):
    """Out-of-fold predictions for the entire dataset.

    Each row's prediction comes from a model that did NOT train on that row.
    Used for stacking/meta-learning and unbiased model evaluation.
    """

    predictions: NDArray[np.float64]  # (n_samples,) probabilities
    labels: NDArray[np.int64]  # (n_samples,) actual labels
    fold_assignments: NDArray[np.int64]  # (n_samples,) which fold predicted each row


class CVResult(TypedDict, total=True):
    """Aggregated cross-validation result with statistics.

    Reports mean ± std to show model stability across splits.
    High std indicates the model is sensitive to the train/val split.
    """

    strategy: CVStrategy
    n_folds: int
    fold_results: tuple[FoldResult, ...]

    # Aggregated metrics with uncertainty
    mean_val_auc: float
    std_val_auc: float  # Key metric: shows sensitivity to split
    min_val_auc: float
    max_val_auc: float

    # OOF predictions for stacking
    oof_predictions: OOFPredictions

    # Final test evaluation (if holdout test set was used)
    test_auc: float | None


class PreprocessingState(TypedDict, total=True):
    """State from fitting preprocessing on training data.

    Captured during training, applied to validation/test.
    Prevents data leakage by never fitting on validation data.
    """

    feature_means: NDArray[np.float64]  # (n_features,)
    feature_stds: NDArray[np.float64]  # (n_features,)
    # Add more as needed: encoders, selected features, etc.
```

#### 3.2 Cross-Validation Splitter (`validation/splitter.py`)

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.validation.types import CVSplitResult, CVStrategy, FoldIndices


class StratifiedKFoldSplitter:
    """Stratified K-Fold cross-validation splitter.

    Maintains class proportions in each fold.
    """

    def split(
        self,
        y_labels: NDArray[np.int64],
        n_folds: int,
        random_state: int,
        test_ratio: float = 0.0,
    ) -> CVSplitResult:
        """Split data into stratified K folds.

        Args:
            y_labels: Binary labels (n_samples,)
            n_folds: Number of folds (typically 5 or 10)
            random_state: Random seed for reproducibility
            test_ratio: Fraction to hold out for final test (0.0 = no holdout)

        Returns:
            CVSplitResult with fold indices and optional test indices.

        Raises:
            ValueError: If n_folds < 2 or insufficient samples per class.
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")

        n_samples = len(y_labels)
        rng = np.random.default_rng(random_state)

        # Separate indices by class
        pos_indices = np.where(y_labels == 1)[0]
        neg_indices = np.where(y_labels == 0)[0]

        # Shuffle within each class
        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        # Hold out test set if requested
        test_indices: NDArray[np.int64] | None = None
        if test_ratio > 0.0:
            n_pos_test = max(1, int(len(pos_indices) * test_ratio))
            n_neg_test = max(1, int(len(neg_indices) * test_ratio))

            test_pos = pos_indices[:n_pos_test]
            test_neg = neg_indices[:n_neg_test]
            test_indices = np.concatenate([test_pos, test_neg])

            pos_indices = pos_indices[n_pos_test:]
            neg_indices = neg_indices[n_neg_test:]

        # Check minimum samples per fold
        min_per_fold_pos = len(pos_indices) // n_folds
        min_per_fold_neg = len(neg_indices) // n_folds

        if min_per_fold_pos < 1:
            raise ValueError(
                f"Not enough positive samples ({len(pos_indices)}) for {n_folds} folds"
            )
        if min_per_fold_neg < 1:
            raise ValueError(
                f"Not enough negative samples ({len(neg_indices)}) for {n_folds} folds"
            )

        # Split each class into folds
        pos_folds = np.array_split(pos_indices, n_folds)
        neg_folds = np.array_split(neg_indices, n_folds)

        # Build fold indices
        folds: list[FoldIndices] = []
        for fold_num in range(n_folds):
            # Validation set is current fold
            val_pos = pos_folds[fold_num]
            val_neg = neg_folds[fold_num]
            val_indices = np.concatenate([val_pos, val_neg])

            # Training set is all other folds
            train_pos = np.concatenate([pos_folds[i] for i in range(n_folds) if i != fold_num])
            train_neg = np.concatenate([neg_folds[i] for i in range(n_folds) if i != fold_num])
            train_indices = np.concatenate([train_pos, train_neg])

            folds.append(
                FoldIndices(
                    fold_number=fold_num,
                    train_indices=train_indices,
                    val_indices=val_indices,
                )
            )

        return CVSplitResult(
            strategy="stratified_kfold",
            n_folds=n_folds,
            n_samples=n_samples,
            folds=tuple(folds),
            test_indices=test_indices,
        )


def create_cv_splitter() -> StratifiedKFoldSplitter:
    """Factory function for creating CV splitter."""
    return StratifiedKFoldSplitter()
```

#### 3.3 Preprocessing Protocol (`validation/preprocessing.py`)

```python
from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from covenant_ml.validation.types import PreprocessingState


class PreprocessorProtocol(Protocol):
    """Protocol for preprocessing that prevents data leakage.

    CRITICAL: fit() is called ONLY on training data.
    transform() applies the fitted state to any data.
    """

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
    ) -> PreprocessingState:
        """Fit preprocessing on training data only.

        Args:
            x_train: Training features (n_train, n_features)
            y_train: Training labels (n_train,)

        Returns:
            PreprocessingState capturing fitted parameters.
        """
        ...

    def transform(
        self,
        x: NDArray[np.float64],
        state: PreprocessingState,
    ) -> NDArray[np.float64]:
        """Apply fitted preprocessing to data.

        Args:
            x: Features to transform (n_samples, n_features)
            state: State from fit() call

        Returns:
            Transformed features.
        """
        ...


class StandardScalerPreprocessor:
    """Standard scaling (z-score normalization).

    Fits mean/std on training data, applies to all data.
    Prevents leakage by never computing stats on validation.
    """

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
    ) -> PreprocessingState:
        """Compute mean and std from training data only."""
        # Compute along axis 0 (per feature)
        means = np.mean(x_train, axis=0)
        stds = np.std(x_train, axis=0)

        # Avoid division by zero
        stds = np.where(stds == 0, 1.0, stds)

        return PreprocessingState(
            feature_means=means,
            feature_stds=stds,
        )

    def transform(
        self,
        x: NDArray[np.float64],
        state: PreprocessingState,
    ) -> NDArray[np.float64]:
        """Apply z-score normalization using fitted stats."""
        return (x - state["feature_means"]) / state["feature_stds"]


def create_standard_scaler() -> StandardScalerPreprocessor:
    """Factory for standard scaler preprocessor."""
    return StandardScalerPreprocessor()
```

#### 3.4 Cross-Validation Runner (`validation/runner.py`)

```python
from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from covenant_ml.validation.preprocessing import PreprocessorProtocol
from covenant_ml.validation.types import (
    CVResult,
    CVSplitResult,
    FoldResult,
    OOFPredictions,
    PreprocessingState,
)


class FoldEvaluatorProtocol(Protocol):
    """Protocol for evaluating a single fold.

    IMPORTANT: Receives preprocessed data. Preprocessing is done
    by the runner to ensure fit happens only on training data.
    """

    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        fold_number: int,
    ) -> FoldResult:
        """Evaluate model on a single fold.

        Args:
            x_train: Preprocessed training features
            y_train: Training labels
            x_val: Preprocessed validation features (using train-fitted params)
            y_val: Validation labels
            fold_number: Current fold number

        Returns:
            FoldResult with metrics and OOF predictions.
        """
        ...


class CrossValidationRunner:
    """Runs cross-validation with proper preprocessing and OOF collection.

    Key guarantees:
    1. Preprocessing is fit on training fold only (no leakage)
    2. OOF predictions are collected for all samples
    3. Mean ± std reported to show model stability
    """

    def __init__(self, preprocessor: PreprocessorProtocol) -> None:
        """Initialize with preprocessor.

        Args:
            preprocessor: Preprocessing pipeline to apply per-fold.
        """
        self._preprocessor = preprocessor

    def run(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        cv_split: CVSplitResult,
        evaluator: FoldEvaluatorProtocol,
    ) -> CVResult:
        """Run cross-validation with preprocessing inside each fold.

        Args:
            x_features: Raw feature matrix (n_samples, n_features)
            y_labels: Full labels (n_samples,)
            cv_split: Pre-computed CV split with fold indices
            evaluator: Function to evaluate a single fold

        Returns:
            CVResult with aggregated metrics and OOF predictions.
        """
        n_samples = len(y_labels)
        fold_results: list[FoldResult] = []

        # Collect OOF predictions
        oof_predictions = np.zeros(n_samples, dtype=np.float64)
        oof_fold_assignments = np.zeros(n_samples, dtype=np.int64)

        for fold in cv_split["folds"]:
            fold_num = fold["fold_number"]
            train_idx = fold["train_indices"]
            val_idx = fold["val_indices"]

            # Extract raw data
            x_train_raw = x_features[train_idx]
            y_train = y_labels[train_idx]
            x_val_raw = x_features[val_idx]
            y_val = y_labels[val_idx]

            # CRITICAL: Fit preprocessing on training data ONLY
            preproc_state = self._preprocessor.fit(x_train_raw, y_train)

            # Apply to both train and val using TRAIN-fitted params
            x_train = self._preprocessor.transform(x_train_raw, preproc_state)
            x_val = self._preprocessor.transform(x_val_raw, preproc_state)

            # Evaluate fold
            result = evaluator(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                fold_number=fold_num,
            )
            fold_results.append(result)

            # Collect OOF predictions for this fold's validation set
            oof_predictions[val_idx] = result["val_predictions"]
            oof_fold_assignments[val_idx] = fold_num

        # Aggregate metrics with uncertainty
        val_aucs = [r["val_auc"] for r in fold_results]
        mean_val_auc = float(np.mean(val_aucs))
        std_val_auc = float(np.std(val_aucs))
        min_val_auc = float(np.min(val_aucs))
        max_val_auc = float(np.max(val_aucs))

        # Build OOF result
        oof = OOFPredictions(
            predictions=oof_predictions,
            labels=y_labels.copy(),
            fold_assignments=oof_fold_assignments,
        )

        # Evaluate on holdout test set if available
        test_auc: float | None = None
        if cv_split["test_indices"] is not None:
            # Retrain on ALL CV data, evaluate on test
            # This requires a separate final training step
            # Implementation: fit preprocessor on all non-test, train model, predict test
            pass  # Deferred to FinalModelTrainer

        return CVResult(
            strategy=cv_split["strategy"],
            n_folds=cv_split["n_folds"],
            fold_results=tuple(fold_results),
            mean_val_auc=mean_val_auc,
            std_val_auc=std_val_auc,
            min_val_auc=min_val_auc,
            max_val_auc=max_val_auc,
            oof_predictions=oof,
            test_auc=test_auc,
        )


def create_cv_runner(preprocessor: PreprocessorProtocol) -> CrossValidationRunner:
    """Factory function for creating CV runner."""
    return CrossValidationRunner(preprocessor)
```

#### 3.5 OOF Utilities (`validation/oof.py`)

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.metrics import compute_auc
from covenant_ml.validation.types import OOFPredictions


def compute_oof_auc(oof: OOFPredictions) -> float:
    """Compute AUC from out-of-fold predictions.

    This gives an unbiased estimate of model performance since
    each prediction comes from a model that didn't see that sample.

    Args:
        oof: OOF predictions for all samples.

    Returns:
        AUC score.
    """
    return compute_auc(oof["labels"], oof["predictions"])


def oof_to_stacking_features(oof: OOFPredictions) -> NDArray[np.float64]:
    """Convert OOF predictions to features for stacking.

    Returns predictions as a column vector suitable for
    use as meta-features in a stacking ensemble.

    Args:
        oof: OOF predictions.

    Returns:
        (n_samples, 1) array of predictions as features.
    """
    return oof["predictions"].reshape(-1, 1)


def validate_oof_coverage(oof: OOFPredictions, expected_samples: int) -> None:
    """Validate that OOF predictions cover all samples.

    Raises ValueError if any sample is missing predictions.

    Args:
        oof: OOF predictions to validate.
        expected_samples: Expected number of samples.

    Raises:
        ValueError: If coverage is incomplete.
    """
    n_predictions = len(oof["predictions"])
    if n_predictions != expected_samples:
        raise ValueError(
            f"OOF coverage incomplete: {n_predictions} predictions "
            f"for {expected_samples} samples"
        )

    # Check fold assignments are valid (no -1 or unassigned)
    min_fold = int(np.min(oof["fold_assignments"]))
    if min_fold < 0:
        raise ValueError("Some samples have no fold assignment")
```

---

### Phase 4: Dataset Discovery Script

**Location:** `services/covenant-radar-api/scripts/discover_datasets/`

#### 4.1 Discovery Script (`scripts/discover_datasets/main.py`)

```python
"""Dataset discovery script.

Scans data/external/ and generates verified dataset configurations.

Usage:
    poetry run python -m scripts.discover_datasets

Output:
    - Console report of detected datasets
    - models/discovered_datasets.json with verified configs
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_rich_console

# Target column name candidates (case-insensitive)
TARGET_CANDIDATES: tuple[str, ...] = (
    "target",
    "class",
    "label",
    "bankrupt?",
    "bankrupt",
    "status_label",
    "status",
    "default",
    "loan_status",
    "is_default",
    "y",
    "outcome",
)


def discover_datasets(external_dir: Path) -> list[dict[str, object]]:
    """Discover datasets in external directory.

    Args:
        external_dir: Path to data/external/

    Returns:
        List of discovered dataset info dicts.
    """
    console = get_rich_console()
    discovered: list[dict[str, object]] = []

    for folder in sorted(external_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith("."):
            continue

        console.print(f"\n[bold]{folder.name}/[/bold]")

        # Find data files
        csv_files = list(folder.glob("*.csv"))
        arff_files = list(folder.glob("*.arff"))
        xlsx_files = list(folder.glob("*.xlsx"))

        if not csv_files and not arff_files and not xlsx_files:
            console.print("  [dim]No CSV/ARFF/Excel files found[/dim]")
            continue

        # Try CSV first
        for csv_file in csv_files:
            info = _analyze_csv(csv_file)
            if info:
                discovered.append(info)
                _print_dataset_info(console, info)

        # Then ARFF
        for arff_file in arff_files:
            info = _analyze_arff(arff_file)
            if info:
                discovered.append(info)
                _print_dataset_info(console, info)

    return discovered


def _analyze_csv(file_path: Path) -> dict[str, object] | None:
    """Analyze a CSV file and detect target column."""
    # Try different encodings
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                headers: list[str] = []
                for row in reader:
                    headers = [h.strip() for h in row]
                    break

                # Count rows
                n_rows = sum(1 for _ in reader)

            # Find likely target column
            target_col = _find_target_column(headers)

            return {
                "folder": file_path.parent.name,
                "file": file_path.name,
                "format": "csv",
                "encoding": encoding,
                "n_rows": n_rows,
                "n_cols": len(headers),
                "headers": headers[:10],  # First 10 for display
                "detected_target": target_col,
            }
        except (UnicodeDecodeError, csv.Error):
            continue

    return None


def _analyze_arff(file_path: Path) -> dict[str, object] | None:
    """Analyze an ARFF file."""
    attributes: list[str] = []
    n_rows = 0
    in_data = False

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data:
                if line.lower().startswith("@attribute"):
                    parts = line.split()
                    if len(parts) >= 2:
                        attributes.append(parts[1])
            else:
                if line and not line.startswith("%"):
                    n_rows += 1

    target_col = _find_target_column(attributes)

    return {
        "folder": file_path.parent.name,
        "file": file_path.name,
        "format": "arff",
        "encoding": "utf-8",
        "n_rows": n_rows,
        "n_cols": len(attributes),
        "headers": attributes[:10],
        "detected_target": target_col,
    }


def _find_target_column(headers: list[str]) -> str | None:
    """Find likely target column from headers."""
    headers_lower = [h.lower() for h in headers]

    for candidate in TARGET_CANDIDATES:
        for i, h in enumerate(headers_lower):
            if h == candidate:
                return headers[i]

    # Check for binary columns at end (common pattern)
    if headers:
        last_col = headers[-1].lower()
        if last_col in ("class", "y", "target", "label"):
            return headers[-1]

    return None


def _print_dataset_info(console: object, info: dict[str, object]) -> None:
    """Print dataset info to console."""
    # Use getattr to call print method
    print_fn = getattr(console, "print")
    file_name: str = str(info["file"])
    n_rows: int = int(str(info["n_rows"]))
    n_cols: int = int(str(info["n_cols"]))
    target: str | None = str(info["detected_target"]) if info["detected_target"] else None

    print_fn(f"  {file_name}: {n_rows:,} rows x {n_cols} cols")
    if target:
        print_fn(f"  [green]Detected target: {target}[/green]")
    else:
        print_fn("  [yellow]No target column detected[/yellow]")


def main() -> None:
    """Main entry point."""
    console = get_rich_console()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    external_dir = project_root / "data" / "external"

    if not external_dir.exists():
        console.print(f"[red]External directory not found: {external_dir}[/red]")
        raise SystemExit(1)

    console.print("[bold]Dataset Discovery[/bold]")
    console.print(f"Scanning: {external_dir}")

    discovered = discover_datasets(external_dir)

    # Write output
    output_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "discovered_datasets.json"

    with open(output_path, "w") as f:
        f.write(dump_json_str(discovered))

    console.print(f"\n[green]Discovered {len(discovered)} datasets[/green]")
    console.print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
```

---

### Phase 5: Service Integration

**Location:** `services/covenant-radar-api/`

#### 5.1 Test Hooks (`scripts/_test_hooks.py`)

Update to support pluggable dataset loading:

```python
# Add to existing _test_hooks.py

from pathlib import Path
from typing import Protocol

from covenant_ml.datasets.types import DatasetConfig, LoadedDataset


class DatasetLoaderProtocol(Protocol):
    """Protocol for dataset loader hook."""

    def __call__(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Load dataset from config."""
        ...


# Default hook - set to real implementation at startup
dataset_loader: DatasetLoaderProtocol


def _real_dataset_loader(
    config: DatasetConfig,
    external_dir: Path,
) -> LoadedDataset:
    """Real implementation using covenant_ml.datasets."""
    from covenant_ml.datasets.loader import create_dataset_loader

    loader = create_dataset_loader()
    return loader.load(config, external_dir)


def use_real_dataset_loader() -> None:
    """Set hook to real implementation (called at service startup)."""
    global dataset_loader
    dataset_loader = _real_dataset_loader


def use_fake_dataset_loader(fake: DatasetLoaderProtocol) -> None:
    """Set hook to fake implementation (called in tests)."""
    global dataset_loader
    dataset_loader = fake


# Initialize to real by default
use_real_dataset_loader()
```

#### 5.2 Updated Optimize Job (`worker/optimize_job.py`)

Replace hardcoded `_load_dataset()` with registry-based loading:

```python
# Replace _load_dataset function with:

def _load_dataset(dataset_name: str, external_dir: Path) -> RawDataset:
    """Load dataset using registry and pluggable loader.

    Args:
        dataset_name: Name of dataset in registry
        external_dir: Path to data/external directory

    Returns:
        RawDataset with feature matrix, labels, and metadata

    Raises:
        KeyError: If dataset not in registry
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If data invalid
    """
    from covenant_ml.datasets.registry import make_default_registry

    import scripts._test_hooks as _hooks

    registry = make_default_registry()
    config = registry.get(dataset_name)

    loaded = _hooks.dataset_loader(config, external_dir)

    # Convert to RawDataset format for backward compatibility
    return RawDataset(
        x=loaded["x"],
        y=loaded["y"],
        feature_names=list(loaded["meta"]["feature_names"]),
        n_samples=loaded["meta"]["n_samples"],
        n_features=loaded["meta"]["n_features"],
        n_bankrupt=loaded["meta"]["n_positive"],
        n_healthy=loaded["meta"]["n_negative"],
    )
```

#### 5.3 Updated CLI (`scripts/optimize/cli.py`)

Replace hardcoded DatasetName with dynamic registry lookup:

```python
from __future__ import annotations

from collections.abc import Sequence

from platform_core.logging import get_rich_console


def get_available_datasets() -> tuple[str, ...]:
    """Get list of available dataset names from registry."""
    from covenant_ml.datasets.registry import make_default_registry

    registry = make_default_registry()
    return registry.list_names()


class OptimizeArgs:
    """Parsed command line arguments."""

    dataset: str  # Now accepts any registered dataset name
    n_trials: int
    n_folds: int  # NEW: number of CV folds (0 = no CV, use holdout)
    feature_preset: str
    device: str
    timeout: int | None
    compare_presets: bool
    all_datasets: bool
    verbose: bool

    def __init__(self) -> None:
        """Initialize with defaults."""
        self.dataset = "taiwan"
        self.n_trials = 300
        self.n_folds = 0  # Default: no CV, use single holdout split
        self.feature_preset = "full"
        self.device = "cuda"
        self.timeout = None
        self.compare_presets = False
        self.all_datasets = False
        self.verbose = False


def _parse_dataset(val: str) -> str:
    """Parse and validate dataset value."""
    console = get_rich_console()
    available = get_available_datasets()

    if val in available:
        return val

    console.print(f"[red]Invalid dataset: {val}[/red]")
    console.print(f"Available datasets: {', '.join(available)}")
    raise SystemExit(1)


# ... rest of parse_args updated similarly
```

---

### Phase 6: File Structure Summary

```
libs/covenant_ml/src/covenant_ml/
├── datasets/
│   ├── __init__.py              # Package exports
│   ├── types.py                 # DatasetConfig, LoadedDataset, etc.
│   ├── protocol.py              # DatasetLoaderProtocol
│   ├── registry.py              # DatasetRegistry, make_default_registry
│   ├── loader.py                # DatasetLoader (unified)
│   └── loaders/
│       ├── __init__.py
│       ├── csv_loader.py        # CSVLoader
│       └── arff_loader.py       # ARFFLoader
├── validation/
│   ├── __init__.py              # Package exports
│   ├── types.py                 # CVResult, FoldResult, OOFPredictions, etc.
│   ├── preprocessing.py         # PreprocessorProtocol, StandardScalerPreprocessor
│   ├── splitter.py              # StratifiedKFoldSplitter
│   ├── runner.py                # CrossValidationRunner (with per-fold preprocessing)
│   └── oof.py                   # OOF utilities for stacking

libs/covenant_ml/tests/
├── datasets/
│   ├── __init__.py
│   ├── test_types.py
│   ├── test_registry.py
│   ├── test_loader.py
│   └── loaders/
│       ├── __init__.py
│       ├── test_csv_loader.py
│       └── test_arff_loader.py
├── validation/
│   ├── __init__.py
│   ├── test_types.py
│   ├── test_preprocessing.py   # Tests for StandardScalerPreprocessor
│   ├── test_splitter.py
│   ├── test_runner.py          # Tests for CV runner with preprocessing
│   └── test_oof.py             # Tests for OOF utilities

libs/covenant_ml/testing.py       # ADD: Fake dataset loader for consumers

services/covenant-radar-api/
├── scripts/
│   ├── _test_hooks.py           # UPDATE: Add dataset_loader hook
│   ├── discover_datasets/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── main.py
│   └── optimize/
│       └── cli.py               # UPDATE: Dynamic dataset names
├── src/covenant_radar_api/
│   └── worker/
│       └── optimize_job.py      # UPDATE: Use registry-based loading
└── tests/
    ├── test_optimize_job.py     # UPDATE: Use fake dataset loader
    └── discover_datasets/
        └── test_main.py
```

---

## Testing Strategy

### Test Patterns

1. **No mocks** - Use real implementations with test data
2. **Test hooks** - Production sets real, tests set fakes
3. **Deterministic** - Fixed random seeds everywhere
4. **Strong assertions** - Assert exact values where possible

### Test Data

Create small test datasets in `libs/covenant_ml/tests/fixtures/`:

```
fixtures/
├── small_csv/
│   └── data.csv           # 100 rows, 10 features, known target
├── small_arff/
│   └── data.arff          # 100 rows, 10 features, known target
└── edge_cases/
    ├── empty.csv          # Empty file
    ├── missing_target.csv # No target column
    └── all_nan.csv        # All NaN values
```

### Coverage Requirements

| Component | Statement | Branch |
|-----------|-----------|--------|
| datasets/types.py | 100% | 100% |
| datasets/registry.py | 100% | 100% |
| datasets/loader.py | 100% | 100% |
| datasets/loaders/*.py | 100% | 100% |
| validation/types.py | 100% | 100% |
| validation/preprocessing.py | 100% | 100% |
| validation/splitter.py | 100% | 100% |
| validation/runner.py | 100% | 100% |
| validation/oof.py | 100% | 100% |

---

## Implementation Order

1. **Phase 1: Dataset Types** - Types and registry
2. **Phase 2: CSV Loader** - Most common format
3. **Phase 3: ARFF Loader** - Polish dataset
4. **Phase 4: Unified Loader** - Route by format
5. **Phase 5: CV Types** - CVResult, FoldResult, OOFPredictions, PreprocessingState
6. **Phase 6: Preprocessing** - PreprocessorProtocol, StandardScalerPreprocessor
7. **Phase 7: CV Splitter** - StratifiedKFoldSplitter with test holdout
8. **Phase 8: CV Runner** - CrossValidationRunner with per-fold preprocessing
9. **Phase 9: OOF Utilities** - compute_oof_auc, oof_to_stacking_features
10. **Phase 10: Discovery Script** - Generate configs
11. **Phase 11: Service Integration** - Hooks, CLI, optimize job
12. **Phase 12: Testing** - Full coverage

---

## Validation Checklist

Before considering implementation complete:

### Code Quality
- [ ] `make check` passes in `libs/covenant_ml/`
- [ ] `make check` passes in `services/covenant-radar-api/`
- [ ] 100% statement coverage for new code
- [ ] 100% branch coverage for new code
- [ ] No `Any` types anywhere
- [ ] No `cast()` calls anywhere
- [ ] No `type: ignore` comments anywhere
- [ ] No `.pyi` stub files
- [ ] No mocks in tests
- [ ] All TypedDicts use `total=True` (immutable)
- [ ] All protocols use strict signatures
- [ ] Test hooks pattern for dependency injection

### Cross-Validation Requirements
- [ ] Preprocessing fits on training data ONLY (no leakage)
- [ ] OOF predictions cover all samples
- [ ] Mean ± std reported for all CV runs
- [ ] High std triggers warning (model is unstable)
- [ ] Stacking features available via `oof_to_stacking_features()`

### Dataset Loading
- [ ] Backward compatibility with existing `RawDataset` consumers
- [ ] Discovery script successfully scans all 38 datasets
- [ ] At least 10 datasets have verified configs in registry
- [ ] Target column auto-detection works for common patterns
- [ ] Encoding auto-detection works (utf-8, utf-8-sig, latin-1)

---

## Migration Path

### Backward Compatibility

The existing `load_taiwan_raw()`, `load_us_raw()`, `load_polish_raw()` functions in `seeding/real_data.py` remain unchanged. The new system is additive.

### Gradual Migration

1. Implement new dataset system in `covenant_ml.datasets`
2. Add registry configs for taiwan, us, polish (verified against existing loaders)
3. Update optimize_job to use new system
4. Gradually add more datasets to registry as verified
5. Eventually deprecate hardcoded loaders in real_data.py

---

*Last updated: December 2025*

---

## Appendix: Data Leakage Prevention

### What is Data Leakage?

Data leakage occurs when information from outside the training data is used to create the model. In cross-validation, the most common form is **preprocessing leakage**:

```python
# WRONG: Fits scaler on ALL data (including validation)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leaks validation info into training

# CORRECT: Fits scaler on training data ONLY
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  # Uses train-fitted params
```

### Why It Matters

When preprocessing sees validation data during fitting:
1. Mean/std computed on full dataset (includes validation)
2. Model indirectly "sees" validation distribution during training
3. Validation AUC is artificially inflated
4. Model performs worse on truly unseen data

### Our Solution

The `CrossValidationRunner` enforces correct preprocessing:

```python
for fold in cv_split["folds"]:
    # CRITICAL: Fit on training fold ONLY
    preproc_state = self._preprocessor.fit(x_train_raw, y_train)

    # Apply to both using TRAIN-fitted params
    x_train = self._preprocessor.transform(x_train_raw, preproc_state)
    x_val = self._preprocessor.transform(x_val_raw, preproc_state)
```

This pattern is enforced at the architecture level - the evaluator never sees raw data, only preprocessed data. The preprocessing state is discarded after each fold.
