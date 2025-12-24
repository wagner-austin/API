# Atmospheric Data Integration Plan

## Overview

Integration of GOES satellite and atmospheric data into the covenant_ml framework for tabular machine learning on weather and environmental prediction tasks.

**Goals:**
1. Add NetCDF/xarray dataset loader for GOES-16/17 satellite data
2. Support atmospheric time-series with spatial entity aggregation (lat/lon pixels)
3. Reuse existing infrastructure: backends (XGBoost, LightGBM, MLP, LSTM), feature engineering (ratios, logs, products), Optuna optimization, cross-validation, preprocessing
4. Maintain 100% test coverage with strict typing (no Any, cast, type: ignore, pyi, noqa)

**Non-Goals:**
- Image-based deep learning (CNNs, Vision Transformers)
- Real-time streaming inference
- Backwards compatibility shims or partial implementations

---

## Progress Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Atmospheric Types and Protocols | PENDING |
| 2 | NetCDF Loader Implementation | PENDING |
| 3 | Registry Integration | PENDING |
| 4 | Preprocessing Adaptations | PENDING |
| 5 | Service Integration | PENDING |
| 6 | Testing and Validation | PENDING |

---

## Architecture

### Dependency Graph

```
libs/covenant_ml/
├── datasets/
│   ├── types.py                          # Add AtmosphericDatasetConfig, SpatialSpec
│   ├── registry.py                       # Add AtmosphericDatasetRegistry
│   ├── loader.py                         # Add load_atmospheric() method
│   └── loaders/
│       ├── netcdf_loader.py              # NEW: NetCDF/xarray loader
│       ├── _netcdf_protocols.py          # NEW: xarray/netCDF4 protocols
│       ├── _netcdf_spatial.py            # NEW: Spatial aggregation helpers
│       └── _netcdf_temporal.py           # NEW: Temporal feature helpers
├── testing.py                            # Add atmospheric test utilities
└── _test_hooks.py                        # NEW: Test hooks for xarray mocking

services/covenant-radar-api/
├── src/covenant_radar_api/
│   ├── worker/
│   │   ├── train_atmospheric_job.py      # NEW: Training job for atmospheric data
│   │   └── optimize_atmospheric_job.py   # NEW: Optuna optimization job
│   └── api/routes/
│       └── atmospheric.py                # NEW: API routes for atmospheric training
└── scripts/
    └── fetch_goes/                       # NEW: GOES data fetching utility
        ├── __init__.py
        ├── __main__.py
        ├── _test_hooks.py
        ├── main.py
        ├── types.py
        ├── downloader.py
        └── converter.py
```

---

## Phase 1: Atmospheric Types and Protocols

### 1.1 New Types in `libs/covenant_ml/src/covenant_ml/datasets/types.py`

Add the following TypedDicts after existing type definitions:

```python
# Spatial aggregation strategy for gridded data
SpatialAggregation = Literal[
    "none",       # Keep all pixels as separate samples
    "mean",       # Mean of all pixels in region
    "statistics", # Mean, std, min, max per feature (4x features)
]

# Target type for atmospheric prediction
AtmosphericTargetType = Literal[
    "regression",           # Continuous value prediction (temp, humidity)
    "binary_classification", # Binary outcome (rain/no rain)
    "multiclass",           # Multiple categories (cloud type)
]


class SpatialSpec(TypedDict, total=True):
    """Specification for spatial data handling.

    Defines how to identify spatial coordinates and aggregate pixels
    into samples for tabular ML.

    Attributes:
        lat_column: Column name for latitude coordinate.
        lon_column: Column name for longitude coordinate.
        entity_format: How to construct entity ID from lat/lon.
            "concat" produces "lat_lon" strings.
            "grid" produces grid cell indices.
        grid_resolution_deg: Grid cell size in degrees (for "grid" format).
        spatial_aggregation: How to aggregate multiple pixels per entity.
    """

    lat_column: str
    lon_column: str
    entity_format: Literal["concat", "grid"]
    grid_resolution_deg: float
    spatial_aggregation: SpatialAggregation


class AtmosphericTargetSpec(TypedDict, total=True):
    """Specification for atmospheric prediction target.

    Defines the target variable and how to encode it for ML.

    Attributes:
        column_name: Name of target column in dataset.
        target_type: Type of prediction task.
        positive_threshold: For binary classification, threshold above which
            target is positive. Ignored for regression/multiclass.
        n_classes: Number of classes for multiclass. Ignored for others.
        class_boundaries: Boundaries for discretizing continuous values
            into classes. Tuple of (n_classes - 1) boundary values.
    """

    column_name: str
    target_type: AtmosphericTargetType
    positive_threshold: float
    n_classes: int
    class_boundaries: tuple[float, ...]


class AtmosphericBand(TypedDict, total=True):
    """Specification for a single satellite band or derived product.

    Attributes:
        name: Short name for the band (e.g., "band_7", "cloud_top_height").
        variable_name: Variable name in NetCDF file.
        units: Physical units (e.g., "K", "m", "dimensionless").
        fill_value: Value indicating missing data in source.
        valid_range: Tuple of (min, max) valid values.
        scale_factor: Multiplicative scale factor to apply.
        add_offset: Additive offset to apply after scaling.
    """

    name: str
    variable_name: str
    units: str
    fill_value: float
    valid_range: tuple[float, float]
    scale_factor: float
    add_offset: float


class AtmosphericDatasetConfig(TypedDict, total=True):
    """Configuration for atmospheric/satellite datasets.

    Extends the dataset configuration pattern for gridded NetCDF data
    with spatial coordinates and multiple spectral bands.

    Attributes:
        name: Unique identifier (e.g., "goes16_cloud_properties").
        display_name: Human-readable name for display.
        folder: Subfolder under data/external/.
        file_pattern: Glob pattern for NetCDF files (e.g., "*.nc").
        file_format: Must be "netcdf" for atmospheric data.
        time_column: Variable name for time coordinate.
        spatial: Spatial coordinate specification.
        bands: Tuple of band specifications to extract.
        target: Target variable specification.
        exclude_bands: Band names to exclude from features.
        n_samples_expected: Expected sample count after aggregation.
        n_features_expected: Expected feature count (bands * aggregation multiplier).
        era5_join: Optional ERA5 reanalysis data to join.
    """

    name: str
    display_name: str
    folder: str
    file_pattern: str
    file_format: Literal["netcdf"]
    time_column: str
    spatial: SpatialSpec
    bands: tuple[AtmosphericBand, ...]
    target: AtmosphericTargetSpec
    exclude_bands: tuple[str, ...]
    n_samples_expected: int
    n_features_expected: int
    era5_join: str  # Empty string if no ERA5 join, else path to ERA5 data


class AtmosphericDatasetMeta(TypedDict, total=True):
    """Metadata about a loaded atmospheric dataset.

    Contains summary statistics computed after loading.

    Attributes:
        name: Dataset identifier.
        n_samples: Total number of samples (entities after aggregation).
        n_features: Number of feature columns.
        n_timesteps: Number of unique timestamps in source data.
        n_spatial_entities: Number of unique spatial entities.
        lat_range: Tuple of (min_lat, max_lat).
        lon_range: Tuple of (min_lon, max_lon).
        time_range: Tuple of (start_iso, end_iso) timestamps.
        feature_names: Ordered tuple of feature column names.
        target_stats: Statistics about target variable.
    """

    name: str
    n_samples: int
    n_features: int
    n_timesteps: int
    n_spatial_entities: int
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    time_range: tuple[str, str]
    feature_names: tuple[str, ...]
    target_stats: AtmosphericTargetStats


class AtmosphericTargetStats(TypedDict, total=True):
    """Statistics about the target variable.

    Attributes:
        target_type: Type of prediction task.
        mean: Mean value (for regression).
        std: Standard deviation (for regression).
        min_value: Minimum value.
        max_value: Maximum value.
        n_positive: Count of positive class (for binary).
        n_negative: Count of negative class (for binary).
        class_counts: Tuple of counts per class (for multiclass).
    """

    target_type: AtmosphericTargetType
    mean: float
    std: float
    min_value: float
    max_value: float
    n_positive: int
    n_negative: int
    class_counts: tuple[int, ...]


class LoadedAtmosphericDataset(TypedDict, total=True):
    """A fully loaded atmospheric dataset ready for ML.

    Contains the feature matrix, targets, and metadata.

    Attributes:
        meta: Dataset metadata with statistics.
        x: Feature matrix of shape (n_samples, n_features).
        y: Target array. Shape (n_samples,) for classification,
            (n_samples,) for regression.
        y_dtype: Data type of y array ("int64" or "float64").
    """

    meta: AtmosphericDatasetMeta
    x: NDArray[np.float64]
    y: NDArray[np.float64]  # float64 for both regression and encoded classification
    y_dtype: Literal["int64", "float64"]
```

### 1.2 Encoding and Decoding Functions

Add to `libs/covenant_ml/src/covenant_ml/datasets/types.py`:

```python
def require_atmospheric_band(data: dict[str, object], key: str) -> AtmosphericBand:
    """Validate and extract AtmosphericBand from parsed data.

    Args:
        data: Dictionary containing band specification.
        key: Key name for error messages.

    Returns:
        Validated AtmosphericBand TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{key}.name must be non-empty string")

    variable_name = data.get("variable_name")
    if not isinstance(variable_name, str) or not variable_name:
        raise ValueError(f"{key}.variable_name must be non-empty string")

    units = data.get("units")
    if not isinstance(units, str):
        raise ValueError(f"{key}.units must be string")

    fill_value = data.get("fill_value")
    if not isinstance(fill_value, (int, float)):
        raise ValueError(f"{key}.fill_value must be numeric")

    valid_range = data.get("valid_range")
    if not isinstance(valid_range, (list, tuple)) or len(valid_range) != 2:
        raise ValueError(f"{key}.valid_range must be tuple of 2 floats")
    vr_min = valid_range[0]
    vr_max = valid_range[1]
    if not isinstance(vr_min, (int, float)) or not isinstance(vr_max, (int, float)):
        raise ValueError(f"{key}.valid_range must contain numeric values")

    scale_factor = data.get("scale_factor")
    if not isinstance(scale_factor, (int, float)):
        raise ValueError(f"{key}.scale_factor must be numeric")

    add_offset = data.get("add_offset")
    if not isinstance(add_offset, (int, float)):
        raise ValueError(f"{key}.add_offset must be numeric")

    return AtmosphericBand(
        name=name,
        variable_name=variable_name,
        units=units,
        fill_value=float(fill_value),
        valid_range=(float(vr_min), float(vr_max)),
        scale_factor=float(scale_factor),
        add_offset=float(add_offset),
    )


def require_spatial_spec(data: dict[str, object], key: str) -> SpatialSpec:
    """Validate and extract SpatialSpec from parsed data.

    Args:
        data: Dictionary containing spatial specification.
        key: Key name for error messages.

    Returns:
        Validated SpatialSpec TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    lat_column = data.get("lat_column")
    if not isinstance(lat_column, str) or not lat_column:
        raise ValueError(f"{key}.lat_column must be non-empty string")

    lon_column = data.get("lon_column")
    if not isinstance(lon_column, str) or not lon_column:
        raise ValueError(f"{key}.lon_column must be non-empty string")

    entity_format = data.get("entity_format")
    if entity_format not in ("concat", "grid"):
        raise ValueError(f"{key}.entity_format must be 'concat' or 'grid'")

    grid_resolution = data.get("grid_resolution_deg")
    if not isinstance(grid_resolution, (int, float)):
        raise ValueError(f"{key}.grid_resolution_deg must be numeric")

    spatial_agg = data.get("spatial_aggregation")
    if spatial_agg not in ("none", "mean", "statistics"):
        raise ValueError(f"{key}.spatial_aggregation must be 'none', 'mean', or 'statistics'")

    return SpatialSpec(
        lat_column=lat_column,
        lon_column=lon_column,
        entity_format=entity_format,
        grid_resolution_deg=float(grid_resolution),
        spatial_aggregation=spatial_agg,
    )


def require_atmospheric_target_spec(data: dict[str, object], key: str) -> AtmosphericTargetSpec:
    """Validate and extract AtmosphericTargetSpec from parsed data.

    Args:
        data: Dictionary containing target specification.
        key: Key name for error messages.

    Returns:
        Validated AtmosphericTargetSpec TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    column_name = data.get("column_name")
    if not isinstance(column_name, str) or not column_name:
        raise ValueError(f"{key}.column_name must be non-empty string")

    target_type = data.get("target_type")
    if target_type not in ("regression", "binary_classification", "multiclass"):
        raise ValueError(f"{key}.target_type must be valid AtmosphericTargetType")

    positive_threshold = data.get("positive_threshold", 0.0)
    if not isinstance(positive_threshold, (int, float)):
        raise ValueError(f"{key}.positive_threshold must be numeric")

    n_classes = data.get("n_classes", 2)
    if not isinstance(n_classes, int) or n_classes < 2:
        raise ValueError(f"{key}.n_classes must be integer >= 2")

    class_boundaries = data.get("class_boundaries", ())
    if not isinstance(class_boundaries, (list, tuple)):
        raise ValueError(f"{key}.class_boundaries must be tuple of floats")
    validated_boundaries: list[float] = []
    for i, b in enumerate(class_boundaries):
        if not isinstance(b, (int, float)):
            raise ValueError(f"{key}.class_boundaries[{i}] must be numeric")
        validated_boundaries.append(float(b))

    return AtmosphericTargetSpec(
        column_name=column_name,
        target_type=target_type,
        positive_threshold=float(positive_threshold),
        n_classes=n_classes,
        class_boundaries=tuple(validated_boundaries),
    )


def require_atmospheric_dataset_config(data: dict[str, object]) -> AtmosphericDatasetConfig:
    """Validate and extract AtmosphericDatasetConfig from parsed data.

    Args:
        data: Dictionary from JSON/TOML parsing.

    Returns:
        Validated AtmosphericDatasetConfig TypedDict.

    Raises:
        ValueError: If required fields missing or invalid types.
    """
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be non-empty string")

    display_name = data.get("display_name")
    if not isinstance(display_name, str) or not display_name:
        raise ValueError("display_name must be non-empty string")

    folder = data.get("folder")
    if not isinstance(folder, str) or not folder:
        raise ValueError("folder must be non-empty string")

    file_pattern = data.get("file_pattern")
    if not isinstance(file_pattern, str) or not file_pattern:
        raise ValueError("file_pattern must be non-empty string")

    file_format = data.get("file_format")
    if file_format != "netcdf":
        raise ValueError("file_format must be 'netcdf' for atmospheric datasets")

    time_column = data.get("time_column")
    if not isinstance(time_column, str) or not time_column:
        raise ValueError("time_column must be non-empty string")

    spatial_data = data.get("spatial")
    if not isinstance(spatial_data, dict):
        raise ValueError("spatial must be dictionary")
    spatial = require_spatial_spec(spatial_data, "spatial")

    bands_data = data.get("bands")
    if not isinstance(bands_data, (list, tuple)):
        raise ValueError("bands must be list of band specifications")
    bands: list[AtmosphericBand] = []
    for i, band_data in enumerate(bands_data):
        if not isinstance(band_data, dict):
            raise ValueError(f"bands[{i}] must be dictionary")
        bands.append(require_atmospheric_band(band_data, f"bands[{i}]"))

    target_data = data.get("target")
    if not isinstance(target_data, dict):
        raise ValueError("target must be dictionary")
    target = require_atmospheric_target_spec(target_data, "target")

    exclude_bands = data.get("exclude_bands", ())
    if not isinstance(exclude_bands, (list, tuple)):
        raise ValueError("exclude_bands must be tuple of strings")
    validated_exclude: list[str] = []
    for i, eb in enumerate(exclude_bands):
        if not isinstance(eb, str):
            raise ValueError(f"exclude_bands[{i}] must be string")
        validated_exclude.append(eb)

    n_samples = data.get("n_samples_expected")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples_expected must be positive integer")

    n_features = data.get("n_features_expected")
    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("n_features_expected must be positive integer")

    era5_join = data.get("era5_join", "")
    if not isinstance(era5_join, str):
        raise ValueError("era5_join must be string")

    return AtmosphericDatasetConfig(
        name=name,
        display_name=display_name,
        folder=folder,
        file_pattern=file_pattern,
        file_format="netcdf",
        time_column=time_column,
        spatial=spatial,
        bands=tuple(bands),
        target=target,
        exclude_bands=tuple(validated_exclude),
        n_samples_expected=n_samples,
        n_features_expected=n_features,
        era5_join=era5_join,
    )


def encode_atmospheric_dataset_config(config: AtmosphericDatasetConfig) -> dict[str, object]:
    """Encode AtmosphericDatasetConfig to JSON-serializable dictionary.

    Args:
        config: Validated configuration to encode.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    return {
        "name": config["name"],
        "display_name": config["display_name"],
        "folder": config["folder"],
        "file_pattern": config["file_pattern"],
        "file_format": config["file_format"],
        "time_column": config["time_column"],
        "spatial": {
            "lat_column": config["spatial"]["lat_column"],
            "lon_column": config["spatial"]["lon_column"],
            "entity_format": config["spatial"]["entity_format"],
            "grid_resolution_deg": config["spatial"]["grid_resolution_deg"],
            "spatial_aggregation": config["spatial"]["spatial_aggregation"],
        },
        "bands": [
            {
                "name": b["name"],
                "variable_name": b["variable_name"],
                "units": b["units"],
                "fill_value": b["fill_value"],
                "valid_range": list(b["valid_range"]),
                "scale_factor": b["scale_factor"],
                "add_offset": b["add_offset"],
            }
            for b in config["bands"]
        ],
        "target": {
            "column_name": config["target"]["column_name"],
            "target_type": config["target"]["target_type"],
            "positive_threshold": config["target"]["positive_threshold"],
            "n_classes": config["target"]["n_classes"],
            "class_boundaries": list(config["target"]["class_boundaries"]),
        },
        "exclude_bands": list(config["exclude_bands"]),
        "n_samples_expected": config["n_samples_expected"],
        "n_features_expected": config["n_features_expected"],
        "era5_join": config["era5_join"],
    }
```

---

## Phase 2: NetCDF Loader Implementation

### 2.1 xarray Protocols in `libs/covenant_ml/src/covenant_ml/datasets/loaders/_netcdf_protocols.py`

```python
"""Protocols for xarray and NetCDF operations.

Strict typing for dynamic imports. No Any, no stubs.
Internal module - used by netcdf_loader, not exported publicly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class XarrayDataArrayProtocol(Protocol):
    """Protocol for xarray DataArray operations."""

    @property
    def values(self) -> NDArray[np.float64]:
        """Return underlying numpy array."""
        ...

    @property
    def dims(self) -> tuple[str, ...]:
        """Return dimension names."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape."""
        ...

    def isel(self, indexers: dict[str, int]) -> XarrayDataArrayProtocol:
        """Select by integer index.

        Args:
            indexers: Dictionary mapping dimension name to index.

        Returns:
            Subset DataArray.
        """
        ...

    def sel(self, indexers: dict[str, object]) -> XarrayDataArrayProtocol:
        """Select by coordinate value.

        Args:
            indexers: Dictionary mapping dimension name to value.

        Returns:
            Subset DataArray.
        """
        ...

    def mean(self, dim: str) -> XarrayDataArrayProtocol:
        """Compute mean along dimension.

        Args:
            dim: Dimension name to reduce.

        Returns:
            Reduced DataArray.
        """
        ...

    def std(self, dim: str) -> XarrayDataArrayProtocol:
        """Compute standard deviation along dimension.

        Args:
            dim: Dimension name to reduce.

        Returns:
            Reduced DataArray.
        """
        ...

    def min(self, dim: str) -> XarrayDataArrayProtocol:
        """Compute minimum along dimension.

        Args:
            dim: Dimension name to reduce.

        Returns:
            Reduced DataArray.
        """
        ...

    def max(self, dim: str) -> XarrayDataArrayProtocol:
        """Compute maximum along dimension.

        Args:
            dim: Dimension name to reduce.

        Returns:
            Reduced DataArray.
        """
        ...

    def where(
        self, cond: XarrayDataArrayProtocol, other: float
    ) -> XarrayDataArrayProtocol:
        """Apply condition, replacing False values.

        Args:
            cond: Boolean condition array.
            other: Value to use where condition is False.

        Returns:
            Masked DataArray.
        """
        ...

    def __mul__(self, other: float) -> XarrayDataArrayProtocol:
        """Multiply by scalar."""
        ...

    def __add__(self, other: float) -> XarrayDataArrayProtocol:
        """Add scalar."""
        ...

    def __gt__(self, other: float) -> XarrayDataArrayProtocol:
        """Greater than comparison."""
        ...

    def __lt__(self, other: float) -> XarrayDataArrayProtocol:
        """Less than comparison."""
        ...

    def __and__(self, other: XarrayDataArrayProtocol) -> XarrayDataArrayProtocol:
        """Logical and."""
        ...


class XarrayDatasetProtocol(Protocol):
    """Protocol for xarray Dataset operations."""

    @property
    def dims(self) -> dict[str, int]:
        """Return dimension sizes."""
        ...

    @property
    def coords(self) -> dict[str, XarrayDataArrayProtocol]:
        """Return coordinates."""
        ...

    @property
    def data_vars(self) -> dict[str, XarrayDataArrayProtocol]:
        """Return data variables."""
        ...

    def __getitem__(self, key: str) -> XarrayDataArrayProtocol:
        """Get variable by name."""
        ...

    def close(self) -> None:
        """Close the dataset."""
        ...


class XarrayOpenDatasetProtocol(Protocol):
    """Protocol for xarray.open_dataset function."""

    def __call__(
        self,
        path: str | Path,
        engine: str = "netcdf4",
    ) -> XarrayDatasetProtocol:
        """Open NetCDF file as xarray Dataset.

        Args:
            path: Path to NetCDF file.
            engine: NetCDF engine to use.

        Returns:
            Opened Dataset.
        """
        ...


class XarrayOpenMFDatasetProtocol(Protocol):
    """Protocol for xarray.open_mfdataset function."""

    def __call__(
        self,
        paths: list[str] | list[Path],
        engine: str = "netcdf4",
        combine: str = "by_coords",
    ) -> XarrayDatasetProtocol:
        """Open multiple NetCDF files as single xarray Dataset.

        Args:
            paths: List of paths to NetCDF files.
            engine: NetCDF engine to use.
            combine: How to combine files.

        Returns:
            Combined Dataset.
        """
        ...


def get_xarray_open_dataset() -> XarrayOpenDatasetProtocol:
    """Get xarray.open_dataset function with strict typing.

    Returns:
        Typed open_dataset function.

    Raises:
        ImportError: If xarray not installed.
    """
    xr_mod = __import__("xarray")
    fn: XarrayOpenDatasetProtocol = xr_mod.open_dataset
    return fn


def get_xarray_open_mfdataset() -> XarrayOpenMFDatasetProtocol:
    """Get xarray.open_mfdataset function with strict typing.

    Returns:
        Typed open_mfdataset function.

    Raises:
        ImportError: If xarray not installed.
    """
    xr_mod = __import__("xarray")
    fn: XarrayOpenMFDatasetProtocol = xr_mod.open_mfdataset
    return fn


__all__ = [
    "XarrayDataArrayProtocol",
    "XarrayDatasetProtocol",
    "XarrayOpenDatasetProtocol",
    "XarrayOpenMFDatasetProtocol",
    "get_xarray_open_dataset",
    "get_xarray_open_mfdataset",
]
```

### 2.2 Spatial Helpers in `libs/covenant_ml/src/covenant_ml/datasets/loaders/_netcdf_spatial.py`

```python
"""Spatial aggregation helpers for NetCDF data.

Functions for handling lat/lon coordinates, entity ID generation,
and spatial aggregation of gridded data.

Internal module - used by netcdf_loader, not exported publicly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.types import SpatialSpec


def compute_entity_ids_concat(
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> NDArray[np.str_]:
    """Compute entity IDs by concatenating lat/lon.

    Args:
        lat: Latitude values, shape (n_points,).
        lon: Longitude values, shape (n_points,).

    Returns:
        Entity ID strings of form "lat_lon", shape (n_points,).
    """
    n_points = int(lat.shape[0])
    result: list[str] = []
    for i in range(n_points):
        lat_val = float(lat.flat[i])
        lon_val = float(lon.flat[i])
        entity_id = f"{lat_val:.4f}_{lon_val:.4f}"
        result.append(entity_id)
    return np.array(result, dtype=np.str_)


def compute_entity_ids_grid(
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    resolution_deg: float,
) -> NDArray[np.str_]:
    """Compute entity IDs by grid cell assignment.

    Args:
        lat: Latitude values, shape (n_points,).
        lon: Longitude values, shape (n_points,).
        resolution_deg: Grid cell size in degrees.

    Returns:
        Entity ID strings of form "lat_idx_lon_idx", shape (n_points,).
    """
    n_points = int(lat.shape[0])
    result: list[str] = []
    for i in range(n_points):
        lat_val = float(lat.flat[i])
        lon_val = float(lon.flat[i])
        lat_idx = int(lat_val / resolution_deg)
        lon_idx = int(lon_val / resolution_deg)
        entity_id = f"grid_{lat_idx}_{lon_idx}"
        result.append(entity_id)
    return np.array(result, dtype=np.str_)


def compute_entity_ids(
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    spec: SpatialSpec,
) -> NDArray[np.str_]:
    """Compute entity IDs based on spatial specification.

    Args:
        lat: Latitude values, shape (n_points,).
        lon: Longitude values, shape (n_points,).
        spec: Spatial specification defining entity format.

    Returns:
        Entity ID strings, shape (n_points,).

    Raises:
        ValueError: If entity_format is not recognized.
    """
    entity_format = spec["entity_format"]
    if entity_format == "concat":
        return compute_entity_ids_concat(lat, lon)
    if entity_format == "grid":
        return compute_entity_ids_grid(lat, lon, spec["grid_resolution_deg"])
    raise ValueError(f"Unknown entity_format: {entity_format}")


def aggregate_by_entity_none(
    entity_ids: NDArray[np.str_],
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.str_], NDArray[np.float64], NDArray[np.float64]]:
    """No aggregation - return data as-is.

    Args:
        entity_ids: Entity IDs, shape (n_samples,).
        features: Feature matrix, shape (n_samples, n_features).
        targets: Target values, shape (n_samples,).

    Returns:
        Tuple of (entity_ids, features, targets) unchanged.
    """
    return entity_ids, features, targets


def aggregate_by_entity_mean(
    entity_ids: NDArray[np.str_],
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.str_], NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate features by entity using mean.

    Args:
        entity_ids: Entity IDs, shape (n_samples,).
        features: Feature matrix, shape (n_samples, n_features).
        targets: Target values, shape (n_samples,).

    Returns:
        Tuple of (unique_entity_ids, aggregated_features, aggregated_targets).
        Targets are aggregated using mean for regression, mode for classification.
    """
    unique_entities: list[str] = []
    entity_set: set[str] = set()
    for eid in entity_ids.flat:
        eid_str = str(eid)
        if eid_str not in entity_set:
            entity_set.add(eid_str)
            unique_entities.append(eid_str)

    n_entities = len(unique_entities)
    n_features = int(features.shape[1])

    agg_features: NDArray[np.float64] = np.zeros((n_entities, n_features), dtype=np.float64)
    agg_targets: NDArray[np.float64] = np.zeros(n_entities, dtype=np.float64)

    entity_to_idx: dict[str, int] = {eid: i for i, eid in enumerate(unique_entities)}

    # Accumulate sums and counts
    feature_sums: NDArray[np.float64] = np.zeros((n_entities, n_features), dtype=np.float64)
    target_sums: NDArray[np.float64] = np.zeros(n_entities, dtype=np.float64)
    counts: NDArray[np.int64] = np.zeros(n_entities, dtype=np.int64)

    n_samples = int(entity_ids.shape[0])
    for i in range(n_samples):
        eid_str = str(entity_ids.flat[i])
        idx = entity_to_idx[eid_str]
        feature_sums[idx, :] += features[i, :]
        target_sums[idx] += float(targets.flat[i])
        counts[idx] += 1

    # Compute means
    for idx in range(n_entities):
        count = int(counts.flat[idx])
        if count > 0:
            agg_features[idx, :] = feature_sums[idx, :] / count
            agg_targets[idx] = target_sums[idx] / count

    return np.array(unique_entities, dtype=np.str_), agg_features, agg_targets


def aggregate_by_entity_statistics(
    entity_ids: NDArray[np.str_],
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.str_], NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate features by entity using statistics (mean, std, min, max).

    Args:
        entity_ids: Entity IDs, shape (n_samples,).
        features: Feature matrix, shape (n_samples, n_features).
        targets: Target values, shape (n_samples,).

    Returns:
        Tuple of (unique_entity_ids, aggregated_features, aggregated_targets).
        Features are expanded to 4x (mean, std, min, max per original feature).
    """
    unique_entities: list[str] = []
    entity_set: set[str] = set()
    for eid in entity_ids.flat:
        eid_str = str(eid)
        if eid_str not in entity_set:
            entity_set.add(eid_str)
            unique_entities.append(eid_str)

    n_entities = len(unique_entities)
    n_features = int(features.shape[1])
    n_output_features = n_features * 4  # mean, std, min, max

    agg_features: NDArray[np.float64] = np.zeros((n_entities, n_output_features), dtype=np.float64)
    agg_targets: NDArray[np.float64] = np.zeros(n_entities, dtype=np.float64)

    entity_to_idx: dict[str, int] = {eid: i for i, eid in enumerate(unique_entities)}

    # Group samples by entity
    entity_samples: dict[int, list[int]] = {i: [] for i in range(n_entities)}
    n_samples = int(entity_ids.shape[0])
    for i in range(n_samples):
        eid_str = str(entity_ids.flat[i])
        idx = entity_to_idx[eid_str]
        entity_samples[idx].append(i)

    # Compute statistics per entity
    for idx in range(n_entities):
        sample_indices = entity_samples[idx]
        if not sample_indices:
            continue

        entity_features = features[sample_indices, :]
        entity_targets = targets[sample_indices]

        # Compute statistics for each feature
        for f in range(n_features):
            col = entity_features[:, f]
            agg_features[idx, f * 4 + 0] = float(np.mean(col))
            agg_features[idx, f * 4 + 1] = float(np.std(col))
            agg_features[idx, f * 4 + 2] = float(np.min(col))
            agg_features[idx, f * 4 + 3] = float(np.max(col))

        # Mean target
        agg_targets[idx] = float(np.mean(entity_targets))

    return np.array(unique_entities, dtype=np.str_), agg_features, agg_targets


def aggregate_by_entity(
    entity_ids: NDArray[np.str_],
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    spec: SpatialSpec,
) -> tuple[NDArray[np.str_], NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate samples by entity using specified strategy.

    Args:
        entity_ids: Entity IDs, shape (n_samples,).
        features: Feature matrix, shape (n_samples, n_features).
        targets: Target values, shape (n_samples,).
        spec: Spatial specification with aggregation strategy.

    Returns:
        Tuple of (unique_entity_ids, aggregated_features, aggregated_targets).

    Raises:
        ValueError: If spatial_aggregation is not recognized.
    """
    agg = spec["spatial_aggregation"]
    if agg == "none":
        return aggregate_by_entity_none(entity_ids, features, targets)
    if agg == "mean":
        return aggregate_by_entity_mean(entity_ids, features, targets)
    if agg == "statistics":
        return aggregate_by_entity_statistics(entity_ids, features, targets)
    raise ValueError(f"Unknown spatial_aggregation: {agg}")


__all__ = [
    "aggregate_by_entity",
    "aggregate_by_entity_mean",
    "aggregate_by_entity_none",
    "aggregate_by_entity_statistics",
    "compute_entity_ids",
    "compute_entity_ids_concat",
    "compute_entity_ids_grid",
]
```

### 2.3 NetCDF Loader in `libs/covenant_ml/src/covenant_ml/datasets/loaders/netcdf_loader.py`

```python
"""NetCDF dataset loader for atmospheric/satellite data.

Loads NetCDF files (GOES-16/17, ERA5, etc.) into tabular format
suitable for covenant_ml training pipelines.

Uses xarray for NetCDF access with strict Protocol typing.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.loaders._netcdf_protocols import (
    XarrayDataArrayProtocol,
    XarrayDatasetProtocol,
    get_xarray_open_dataset,
    get_xarray_open_mfdataset,
)
from covenant_ml.datasets.loaders._netcdf_spatial import (
    aggregate_by_entity,
    compute_entity_ids,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import (
    AtmosphericBand,
    AtmosphericDatasetConfig,
    AtmosphericDatasetMeta,
    AtmosphericTargetSpec,
    AtmosphericTargetStats,
    LoadedAtmosphericDataset,
    LoadProgress,
)


def _extract_band_values(
    ds: XarrayDatasetProtocol,
    band: AtmosphericBand,
) -> NDArray[np.float64]:
    """Extract and scale values for a single band.

    Args:
        ds: Opened xarray Dataset.
        band: Band specification.

    Returns:
        Flattened array of scaled values.

    Raises:
        KeyError: If variable not found in dataset.
    """
    var: XarrayDataArrayProtocol = ds[band["variable_name"]]
    values: NDArray[np.float64] = var.values.astype(np.float64)

    # Apply fill value masking
    fill_value = band["fill_value"]
    mask = np.isclose(values, fill_value, rtol=1e-9, atol=1e-9)
    values[mask] = np.nan

    # Apply valid range
    vmin, vmax = band["valid_range"]
    out_of_range = (values < vmin) | (values > vmax)
    values[out_of_range] = np.nan

    # Apply scale and offset
    values = values * band["scale_factor"] + band["add_offset"]

    return values.flatten()


def _extract_coordinates(
    ds: XarrayDatasetProtocol,
    lat_col: str,
    lon_col: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract flattened lat/lon coordinate arrays.

    Args:
        ds: Opened xarray Dataset.
        lat_col: Latitude coordinate name.
        lon_col: Longitude coordinate name.

    Returns:
        Tuple of (lat, lon) arrays, each flattened.

    Raises:
        KeyError: If coordinates not found.
    """
    lat_var: XarrayDataArrayProtocol = ds[lat_col]
    lon_var: XarrayDataArrayProtocol = ds[lon_col]

    lat_values: NDArray[np.float64] = lat_var.values.astype(np.float64)
    lon_values: NDArray[np.float64] = lon_var.values.astype(np.float64)

    # Handle 1D vs 2D coordinates
    if lat_values.ndim == 1 and lon_values.ndim == 1:
        # Meshgrid to get 2D coordinates
        lon_2d, lat_2d = np.meshgrid(lon_values, lat_values)
        return lat_2d.flatten(), lon_2d.flatten()

    return lat_values.flatten(), lon_values.flatten()


def _extract_target(
    ds: XarrayDatasetProtocol,
    spec: AtmosphericTargetSpec,
) -> NDArray[np.float64]:
    """Extract and encode target variable.

    Args:
        ds: Opened xarray Dataset.
        spec: Target specification.

    Returns:
        Flattened target array.

    Raises:
        KeyError: If target variable not found.
    """
    var: XarrayDataArrayProtocol = ds[spec["column_name"]]
    values: NDArray[np.float64] = var.values.astype(np.float64).flatten()

    target_type = spec["target_type"]
    if target_type == "regression":
        return values

    if target_type == "binary_classification":
        threshold = spec["positive_threshold"]
        encoded = np.zeros_like(values)
        encoded[values > threshold] = 1.0
        return encoded

    if target_type == "multiclass":
        boundaries = spec["class_boundaries"]
        encoded = np.zeros_like(values)
        for i, boundary in enumerate(boundaries):
            encoded[values > boundary] = float(i + 1)
        return encoded

    raise ValueError(f"Unknown target_type: {target_type}")


def _compute_target_stats(
    y: NDArray[np.float64],
    spec: AtmosphericTargetSpec,
) -> AtmosphericTargetStats:
    """Compute statistics about target variable.

    Args:
        y: Target array.
        spec: Target specification.

    Returns:
        Target statistics TypedDict.
    """
    target_type = spec["target_type"]
    y_finite = y[np.isfinite(y)]

    mean_val = float(np.mean(y_finite)) if len(y_finite) > 0 else 0.0
    std_val = float(np.std(y_finite)) if len(y_finite) > 0 else 0.0
    min_val = float(np.min(y_finite)) if len(y_finite) > 0 else 0.0
    max_val = float(np.max(y_finite)) if len(y_finite) > 0 else 0.0

    n_positive = 0
    n_negative = 0
    class_counts: list[int] = []

    if target_type == "binary_classification":
        n_positive = int(np.sum(y_finite > 0.5))
        n_negative = int(len(y_finite) - n_positive)
    elif target_type == "multiclass":
        n_classes = spec["n_classes"]
        class_counts = [int(np.sum(np.isclose(y_finite, float(c)))) for c in range(n_classes)]

    return AtmosphericTargetStats(
        target_type=target_type,
        mean=mean_val,
        std=std_val,
        min_value=min_val,
        max_value=max_val,
        n_positive=n_positive,
        n_negative=n_negative,
        class_counts=tuple(class_counts),
    )


def _build_feature_names(
    bands: tuple[AtmosphericBand, ...],
    exclude: tuple[str, ...],
    aggregation: str,
) -> list[str]:
    """Build list of feature names.

    Args:
        bands: Band specifications.
        exclude: Band names to exclude.
        aggregation: Spatial aggregation strategy.

    Returns:
        List of feature column names.
    """
    exclude_set = set(exclude)
    base_names = [b["name"] for b in bands if b["name"] not in exclude_set]

    if aggregation == "statistics":
        result: list[str] = []
        for name in base_names:
            result.extend([
                f"{name}_mean",
                f"{name}_std",
                f"{name}_min",
                f"{name}_max",
            ])
        return result

    return base_names


class NetCDFLoader:
    """Loads NetCDF atmospheric datasets into tabular format.

    Handles:
    - Multiple NetCDF files matching a glob pattern
    - Lat/lon coordinate extraction and entity ID generation
    - Band extraction with fill value handling and scaling
    - Spatial aggregation (none, mean, statistics)
    - Target variable encoding (regression, binary, multiclass)

    Thread-safe for concurrent reads.
    """

    def load(
        self,
        config: AtmosphericDatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedAtmosphericDataset:
        """Load atmospheric dataset from NetCDF files.

        Args:
            config: Atmospheric dataset configuration.
            external_dir: Root directory for datasets.
            progress_callback: Optional callback for progress updates.

        Returns:
            LoadedAtmosphericDataset ready for ML.

        Raises:
            FileNotFoundError: If no files match pattern.
            ValueError: If data is invalid or missing required variables.
        """
        # Find files
        folder_path = external_dir / config["folder"]
        pattern = str(folder_path / config["file_pattern"])
        file_paths = sorted(glob.glob(pattern))

        if not file_paths:
            raise FileNotFoundError(f"No files match pattern: {pattern}")

        # Report progress
        if progress_callback is not None:
            progress_callback(LoadProgress(
                phase="reading",
                bytes_read=0,
                bytes_total=0,
                rows_processed=0,
                rows_total=len(file_paths),
                percent_complete=0.0,
                message=f"Opening {len(file_paths)} NetCDF files...",
            ))

        # Open dataset(s)
        if len(file_paths) == 1:
            open_fn = get_xarray_open_dataset()
            ds = open_fn(file_paths[0], engine="netcdf4")
        else:
            open_mf_fn = get_xarray_open_mfdataset()
            ds = open_mf_fn(file_paths, engine="netcdf4", combine="by_coords")

        # Extract coordinates
        spatial = config["spatial"]
        lat, lon = _extract_coordinates(ds, spatial["lat_column"], spatial["lon_column"])

        # Compute entity IDs
        entity_ids = compute_entity_ids(lat, lon, spatial)

        # Extract bands as features
        bands = config["bands"]
        exclude = config["exclude_bands"]
        exclude_set = set(exclude)

        feature_arrays: list[NDArray[np.float64]] = []
        for band in bands:
            if band["name"] in exclude_set:
                continue
            band_values = _extract_band_values(ds, band)
            feature_arrays.append(band_values)

        if not feature_arrays:
            raise ValueError("No feature bands after exclusion")

        # Stack features
        n_samples_raw = len(feature_arrays[0])
        n_features_raw = len(feature_arrays)
        features_raw: NDArray[np.float64] = np.column_stack(feature_arrays)

        # Extract target
        targets_raw = _extract_target(ds, config["target"])

        # Close dataset
        ds.close()

        # Report progress
        if progress_callback is not None:
            progress_callback(LoadProgress(
                phase="aggregating",
                bytes_read=0,
                bytes_total=0,
                rows_processed=0,
                rows_total=n_samples_raw,
                percent_complete=50.0,
                message=f"Aggregating {n_samples_raw:,} samples...",
            ))

        # Aggregate by entity
        entity_ids_agg, features_agg, targets_agg = aggregate_by_entity(
            entity_ids, features_raw, targets_raw, spatial
        )

        # Replace NaN with 0
        features_clean: NDArray[np.float64] = np.nan_to_num(
            features_agg, nan=0.0, posinf=0.0, neginf=0.0
        )
        targets_clean: NDArray[np.float64] = np.nan_to_num(
            targets_agg, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Build feature names
        feature_names = _build_feature_names(bands, exclude, spatial["spatial_aggregation"])

        # Compute metadata
        n_samples = int(features_clean.shape[0])
        n_features = int(features_clean.shape[1])
        n_entities = len(set(entity_ids_agg))

        # Lat/lon ranges
        lat_min = float(np.min(lat))
        lat_max = float(np.max(lat))
        lon_min = float(np.min(lon))
        lon_max = float(np.max(lon))

        # Target stats
        target_stats = _compute_target_stats(targets_clean, config["target"])

        # Determine y_dtype
        target_type = config["target"]["target_type"]
        y_dtype: str = "float64" if target_type == "regression" else "int64"
        if y_dtype == "int64":
            targets_clean = targets_clean.astype(np.int64).astype(np.float64)

        meta = AtmosphericDatasetMeta(
            name=config["name"],
            n_samples=n_samples,
            n_features=n_features,
            n_timesteps=0,  # Would need time dimension analysis
            n_spatial_entities=n_entities,
            lat_range=(lat_min, lat_max),
            lon_range=(lon_min, lon_max),
            time_range=("", ""),  # Would need time extraction
            feature_names=tuple(feature_names),
            target_stats=target_stats,
        )

        # Report complete
        if progress_callback is not None:
            progress_callback(LoadProgress(
                phase="encoding",
                bytes_read=0,
                bytes_total=0,
                rows_processed=n_samples,
                rows_total=n_samples,
                percent_complete=100.0,
                message=f"Loaded {n_samples:,} samples with {n_features} features",
            ))

        return LoadedAtmosphericDataset(
            meta=meta,
            x=features_clean,
            y=targets_clean,
            y_dtype=y_dtype,
        )


def create_netcdf_loader() -> NetCDFLoader:
    """Factory function for creating NetCDF loader.

    Returns:
        New NetCDFLoader instance.
    """
    return NetCDFLoader()


__all__ = [
    "NetCDFLoader",
    "create_netcdf_loader",
]
```

---

## Phase 3: Registry Integration

### 3.1 Update `libs/covenant_ml/src/covenant_ml/datasets/registry.py`

Add new registry class and verified configs:

```python
class AtmosphericDatasetRegistry:
    """Registry of atmospheric dataset configurations.

    Stores AtmosphericDatasetConfig entries for satellite/weather datasets.
    Immutable after construction. Thread-safe for reads.
    """

    def __init__(self, configs: tuple[AtmosphericDatasetConfig, ...]) -> None:
        """Initialize with a tuple of atmospheric dataset configs.

        Args:
            configs: Immutable tuple of AtmosphericDatasetConfig entries.

        Raises:
            ValueError: If duplicate dataset names found.
        """
        self._configs: dict[str, AtmosphericDatasetConfig] = {}
        for cfg in configs:
            name = cfg["name"]
            if name in self._configs:
                raise ValueError(f"Duplicate dataset name: {name}")
            self._configs[name] = cfg

    def get(self, name: str) -> AtmosphericDatasetConfig:
        """Get atmospheric configuration for a dataset by name.

        Args:
            name: Dataset name (e.g., "goes16_cloud").

        Returns:
            AtmosphericDatasetConfig for the requested dataset.

        Raises:
            KeyError: If dataset not found in registry.
        """
        if name not in self._configs:
            available = ", ".join(sorted(self._configs.keys()))
            raise KeyError(f"Atmospheric dataset '{name}' not found. Available: {available}")
        return self._configs[name]

    def list_names(self) -> tuple[str, ...]:
        """List all registered atmospheric dataset names.

        Returns:
            Sorted tuple of dataset names.
        """
        return tuple(sorted(self._configs.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if atmospheric dataset is registered.

        Args:
            name: Dataset name to check.

        Returns:
            True if dataset is in registry.
        """
        return name in self._configs

    def __len__(self) -> int:
        """Get number of registered atmospheric datasets.

        Returns:
            Number of datasets in registry.
        """
        return len(self._configs)


def make_default_atmospheric_registry() -> AtmosphericDatasetRegistry:
    """Create registry with verified atmospheric dataset configurations.

    Returns:
        AtmosphericDatasetRegistry with production configs.
    """
    return AtmosphericDatasetRegistry(_VERIFIED_ATMOSPHERIC_CONFIGS)


_VERIFIED_ATMOSPHERIC_CONFIGS: tuple[AtmosphericDatasetConfig, ...] = (
    # Placeholder - add verified configs as data is acquired
)
```

### 3.2 Update `libs/covenant_ml/src/covenant_ml/datasets/loader.py`

Add atmospheric loading method:

```python
from covenant_ml.datasets.loaders.netcdf_loader import NetCDFLoader
from covenant_ml.datasets.types import (
    AtmosphericDatasetConfig,
    LoadedAtmosphericDataset,
)


class DatasetLoader:
    """Unified dataset loader supporting multiple formats."""

    def __init__(self) -> None:
        """Initialize with format-specific loaders."""
        self._csv_loader = CSVLoader()
        self._arff_loader = ARFFLoader()
        self._timeseries_csv_loader = TimeSeriesCSVLoader()
        self._netcdf_loader = NetCDFLoader()

    # ... existing methods ...

    def load_atmospheric(
        self,
        config: AtmosphericDatasetConfig,
        external_dir: Path,
        progress_callback: ProgressCallbackProtocol | None = None,
    ) -> LoadedAtmosphericDataset:
        """Load atmospheric dataset from NetCDF files.

        Handles satellite and weather data with spatial coordinates.
        Aggregates features according to the strategy in config.spatial.

        Args:
            config: Atmospheric dataset configuration.
            external_dir: Root directory containing dataset folders.
            progress_callback: Optional callback for loading progress updates.

        Returns:
            LoadedAtmosphericDataset with tabular features ready for ML.

        Raises:
            FileNotFoundError: If no files match pattern.
            ValueError: If data invalid or parsing fails.
        """
        return self._netcdf_loader.load(config, external_dir, progress_callback)
```

---

## Phase 4: Preprocessing Adaptations

### 4.1 Regression Support in Preprocessing

The existing `AutoPreprocessor` works for feature preprocessing. For regression targets, add validation in trainer to handle continuous y values.

Update `libs/covenant_ml/src/covenant_ml/base_trainer.py` to support regression:

```python
class TrainTask(TypedDict, total=True):
    """Training task specification supporting classification and regression."""

    task_type: Literal["binary_classification", "regression"]
    # ... other fields
```

### 4.2 Special Codes for Atmospheric Data

Add atmospheric-specific special codes to `libs/covenant_ml/src/covenant_ml/preprocessing/types.py`:

```python
# Special codes for atmospheric data (in addition to financial codes)
ATMOSPHERIC_SPECIAL_CODES: frozenset[float] = frozenset({
    -999.0,     # Common fill value
    -9999.0,    # NOAA fill value
    9999.0,     # Another common fill
    -32768.0,   # Int16 fill
    32767.0,    # Int16 max fill
    65535.0,    # UInt16 fill
    -1e30,      # Large negative fill
    1e30,       # Large positive fill
})

# Combined set for atmospheric preprocessing
DEFAULT_ATMOSPHERIC_SPECIAL_CODES: frozenset[float] = DEFAULT_SPECIAL_CODES | ATMOSPHERIC_SPECIAL_CODES
```

---

## Phase 5: Service Integration

### 5.1 Worker Job in `services/covenant-radar-api/src/covenant_radar_api/worker/train_atmospheric_job.py`

```python
"""Training job for atmospheric datasets.

Executes training with atmospheric data using existing backend infrastructure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from covenant_ml.backends import default_registry
from covenant_ml.datasets import create_dataset_loader
from covenant_ml.datasets.registry import make_default_atmospheric_registry
from covenant_ml.features import engineer_features, get_feature_config_for_preset
from covenant_ml.preprocessing import create_auto_preprocessor
from covenant_ml.preprocessing.types import DEFAULT_ATMOSPHERIC_SPECIAL_CODES
from covenant_ml.trainer import stratified_split
from covenant_ml.types import BackendName


def run_atmospheric_training(
    dataset_name: str,
    backend_name: BackendName,
    feature_preset: Literal["none", "log_only", "ratios_only", "full"],
    external_dir: Path,
    output_dir: Path,
) -> None:
    """Execute atmospheric model training.

    Args:
        dataset_name: Registered atmospheric dataset name.
        backend_name: ML backend to use.
        feature_preset: Feature engineering preset.
        external_dir: Root directory for datasets.
        output_dir: Directory for model output.

    Raises:
        KeyError: If dataset not in registry.
        ValueError: If training fails.
    """
    # Load dataset
    atmos_registry = make_default_atmospheric_registry()
    config = atmos_registry.get(dataset_name)

    loader = create_dataset_loader()
    dataset = loader.load_atmospheric(config, external_dir)

    x = dataset["x"]
    y = dataset["y"]
    feature_names = list(dataset["meta"]["feature_names"])

    # Feature engineering
    feat_config = get_feature_config_for_preset(feature_preset)
    engineered = engineer_features(x, feature_names, feat_config)
    x_eng = engineered["x"]
    feature_names_eng = engineered["feature_names"]

    # Split data
    splits = stratified_split(
        x_eng,
        y.astype("int64"),  # For stratification
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # Preprocess with atmospheric special codes
    preprocessor = create_auto_preprocessor(
        special_codes=DEFAULT_ATMOSPHERIC_SPECIAL_CODES,
    )
    state = preprocessor.fit(splits["x_train"], splits["y_train"])
    x_train = preprocessor.transform(splits["x_train"], state)
    x_val = preprocessor.transform(splits["x_val"], state)
    x_test = preprocessor.transform(splits["x_test"], state)

    # Train
    registry = default_registry()
    backend = registry.get(backend_name)

    # Backend training call
    # ... implementation follows existing pattern
```

### 5.2 Test Hooks in `services/covenant-radar-api/src/covenant_radar_api/worker/_test_hooks.py`

Add atmospheric-specific hooks:

```python
from typing import Protocol

from covenant_ml.datasets.types import AtmosphericDatasetConfig, LoadedAtmosphericDataset


class AtmosphericLoaderProtocol(Protocol):
    """Protocol for atmospheric data loader."""

    def load(
        self,
        config: AtmosphericDatasetConfig,
        external_dir: Path,
    ) -> LoadedAtmosphericDataset:
        """Load atmospheric dataset."""
        ...


# Injectable loader for testing
atmospheric_loader_hook: AtmosphericLoaderProtocol | None = None
```

---

## Phase 6: Testing and Validation

### 6.1 Unit Tests Required

| Test File | Coverage Target |
|-----------|-----------------|
| `tests/test_atmospheric_types.py` | All require_* and encode_* functions |
| `tests/test_netcdf_protocols.py` | Protocol instantiation and typing |
| `tests/test_netcdf_spatial.py` | All entity ID and aggregation functions |
| `tests/test_netcdf_loader.py` | Full loader with synthetic NetCDF |
| `tests/test_atmospheric_registry.py` | Registry operations |
| `tests/test_atmospheric_preprocessing.py` | Special code detection |

### 6.2 Integration Tests Required

| Test File | Coverage Target |
|-----------|-----------------|
| `tests/test_atmospheric_training_integration.py` | End-to-end training with fake data |
| `tests/test_atmospheric_optuna_integration.py` | Optuna optimization with fake data |

### 6.3 Test Data Generation

Create synthetic NetCDF files for testing:

```python
# In libs/covenant_ml/src/covenant_ml/datasets/testing.py

def create_synthetic_netcdf(
    path: Path,
    n_lat: int = 10,
    n_lon: int = 10,
    n_bands: int = 4,
    seed: int = 42,
) -> None:
    """Create synthetic NetCDF file for testing.

    Args:
        path: Output file path.
        n_lat: Number of latitude points.
        n_lon: Number of longitude points.
        n_bands: Number of bands to generate.
        seed: Random seed for reproducibility.
    """
    # Implementation using netCDF4 library
    ...
```

### 6.4 Validation Checklist

| Requirement | Validation |
|-------------|------------|
| No `Any` types | `make check` mypy strict passes |
| No `cast()` calls | grep codebase, zero results |
| No `type: ignore` | grep codebase, zero results |
| No `.pyi` stubs | file search, zero results |
| No `noqa` comments | grep codebase, zero results |
| 100% statement coverage | pytest --cov with fail_under=100 |
| 100% branch coverage | pytest --cov-branch with fail_under=100 |
| All TypedDicts immutable | total=True on all TypedDicts |
| All encode/decode functions | require_* for every TypedDict |
| Test hooks in services | _test_hooks.py pattern |
| Testing utilities in libs | testing.py exports |
| No mocks in tests | grep for mock/Mock, zero results |
| No try/except for softening | Code review, only edge propagation |
| Google-style docstrings | All public functions documented |

---

## File Change Summary

### New Files

| Path | Description |
|------|-------------|
| `libs/covenant_ml/src/covenant_ml/datasets/loaders/netcdf_loader.py` | NetCDF loader |
| `libs/covenant_ml/src/covenant_ml/datasets/loaders/_netcdf_protocols.py` | xarray protocols |
| `libs/covenant_ml/src/covenant_ml/datasets/loaders/_netcdf_spatial.py` | Spatial helpers |
| `libs/covenant_ml/tests/test_netcdf_loader.py` | Loader tests |
| `libs/covenant_ml/tests/test_netcdf_spatial.py` | Spatial tests |
| `libs/covenant_ml/tests/test_atmospheric_types.py` | Type tests |
| `services/covenant-radar-api/src/covenant_radar_api/worker/train_atmospheric_job.py` | Training job |
| `services/covenant-radar-api/scripts/fetch_goes/` | Data fetching utility |

### Modified Files

| Path | Changes |
|------|---------|
| `libs/covenant_ml/src/covenant_ml/datasets/types.py` | Add atmospheric types |
| `libs/covenant_ml/src/covenant_ml/datasets/registry.py` | Add atmospheric registry |
| `libs/covenant_ml/src/covenant_ml/datasets/loader.py` | Add load_atmospheric() |
| `libs/covenant_ml/src/covenant_ml/preprocessing/types.py` | Add atmospheric special codes |
| `libs/covenant_ml/src/covenant_ml/testing.py` | Add atmospheric test utilities |
| `libs/covenant_ml/pyproject.toml` | Add xarray, netCDF4 dependencies |

---

## Dependencies

Add to `libs/covenant_ml/pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing ...
    "xarray>=2024.1.0",
    "netCDF4>=1.6.0",
    "h5netcdf>=1.3.0",  # Alternative engine
]

[project.optional-dependencies]
atmospheric = [
    "xarray>=2024.1.0",
    "netCDF4>=1.6.0",
]
```

---

## Rollout Plan

1. **Week 1**: Types and protocols (Phase 1)
   - Add all TypedDicts to types.py
   - Add require_* and encode_* functions
   - Add Protocol definitions for xarray
   - Tests for all type functions

2. **Week 2**: Loader implementation (Phase 2)
   - Implement spatial helpers
   - Implement NetCDF loader
   - Create synthetic test data generator
   - Full test coverage for loader

3. **Week 3**: Integration (Phases 3-5)
   - Registry additions
   - Preprocessing adaptations
   - Service worker jobs
   - Integration tests

4. **Week 4**: Validation and documentation (Phase 6)
   - Run full test suite
   - Verify 100% coverage
   - Update API documentation
   - Create example notebooks

---

*Last updated: December 2025*
