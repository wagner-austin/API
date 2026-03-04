# Atmospheric Data Integration Plan

## Overview

Integration of GOES satellite and atmospheric data into the covenant_ml framework for tabular ML on weather and environmental prediction tasks.

**Goals:**
1. Add NetCDF/xarray dataset loader for GOES-16/17 satellite data
2. Support atmospheric time-series with spatial entity aggregation (lat/lon pixels)
3. Reuse existing infrastructure: backends, feature engineering, Optuna optimization, cross-validation, preprocessing
4. Maintain 100% test coverage with strict typing

**Non-Goals:**
- Image-based deep learning (CNNs, Vision Transformers)
- Real-time streaming inference

---

## Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | McKinnon temporal features (_netcdf_temporal.py, types, testing) | ✅ Done (104 tests) |
| 1 | Atmospheric types and protocols | PENDING |
| 2 | NetCDF loader implementation | PENDING |
| 3 | Registry integration | PENDING |
| 4 | Preprocessing adaptations | PENDING |
| 5 | Service integration | PENDING |
| 6 | Testing and validation | PENDING |

---

## Architecture

```
libs/covenant_ml/datasets/
  types.py                         # Add AtmosphericDatasetConfig, SpatialSpec, etc.
  registry.py                      # Add AtmosphericDatasetRegistry
  loaders/
    netcdf_loader.py               # NEW: NetCDF/xarray loader (NetCDFLoader class)
    _netcdf_protocols.py           # NEW: xarray Protocol types (strict typing for dynamic import)
    _netcdf_spatial.py             # NEW: Spatial aggregation helpers (entity IDs, mean/statistics aggregation)
    _netcdf_temporal.py            # DONE: McKinnon temporal features

services/covenant-radar-api/
  worker/
    train_atmospheric_job.py       # NEW: Training job for atmospheric data
  scripts/
    fetch_goes/                    # NEW: GOES data fetching utility
```

---

## Phase 0: McKinnon Temporal Features (DONE)

Implements Karen McKinnon's climate extremes methodology (PNAS 2024). 104 tests across 4 test files.

**Key decisions:**
- Multi-location arrays: `(n_days, n_locations)`, metrics `(n_years, n_locations, n_metrics)`
- Fit/transform pattern: `fit_temporal_features` → `TemporalFeatureState`, `transform_temporal_features` uses it
- 9 heat metrics per location per year: seasonal_max/min, cum/avg/ndays excess hot/cold, ar1
- Pure numpy (no xarray dependency). Protocol wrappers for `__import__` pattern
- OLS Fourier fit for seasonal cycle removal
- `WeatherFeatureExtractor` injects pre-fitted state via `__init__`

---

## Phase 1: Atmospheric Types and Protocols

**Modify:** `libs/covenant_ml/src/covenant_ml/datasets/types.py`

Add TypedDicts:
- `SpatialSpec` — lat/lon columns, entity format (concat/grid), grid resolution, spatial aggregation (none/mean/statistics)
- `AtmosphericTargetSpec` — column name, target type (regression/binary/multiclass), threshold, class boundaries
- `AtmosphericBand` — name, variable name, units, fill value, valid range, scale factor, offset
- `AtmosphericDatasetConfig` — name, folder, file pattern, time column, spatial, bands, target, expected counts
- `AtmosphericDatasetMeta` — n_samples, n_features, n_timesteps, n_spatial_entities, lat/lon ranges, feature names
- `AtmosphericTargetStats` — target type, mean/std/min/max, class counts
- `LoadedAtmosphericDataset` — meta, x, y, y_dtype

Each TypedDict needs `require_*` decode and `encode_*` functions.

---

## Phase 2: NetCDF Loader Implementation

### 2.1 xarray Protocols

**New:** `libs/covenant_ml/src/covenant_ml/datasets/loaders/_netcdf_protocols.py`

Protocol types for `XarrayDataArrayProtocol`, `XarrayDatasetProtocol`, `XarrayOpenDatasetProtocol`, `XarrayOpenMFDatasetProtocol`. Factory functions using `__import__("xarray")` with Protocol type annotation.

### 2.2 Spatial Helpers

**New:** `libs/covenant_ml/src/covenant_ml/datasets/loaders/_netcdf_spatial.py`

Functions: `compute_entity_ids_concat()`, `compute_entity_ids_grid()`, `compute_entity_ids()`, `aggregate_by_entity_none()`, `aggregate_by_entity_mean()`, `aggregate_by_entity_statistics()`, `aggregate_by_entity()`.

### 2.3 NetCDF Loader

**New:** `libs/covenant_ml/src/covenant_ml/datasets/loaders/netcdf_loader.py`

`NetCDFLoader` class with `load()` method. Handles: multiple NetCDF files via glob, lat/lon coordinate extraction, band extraction with fill value masking and scaling, spatial aggregation, target encoding (regression/binary/multiclass), NaN cleanup.

---

## Phase 3: Registry Integration

**Modify:** `libs/covenant_ml/src/covenant_ml/datasets/registry.py`

Add `AtmosphericDatasetRegistry` class (parallel to `DatasetRegistry`) and `make_default_atmospheric_registry()`. Add verified configs as GOES data is acquired.

**Modify:** `libs/covenant_ml/src/covenant_ml/datasets/loader.py`

Add `load_atmospheric()` method to `DatasetLoader`.

---

## Phase 4: Preprocessing Adaptations

**Modify:** `libs/covenant_ml/src/covenant_ml/preprocessing/types.py`

Add `ATMOSPHERIC_SPECIAL_CODES` frozenset (-999.0, -9999.0, 9999.0, -32768.0, 32767.0, 65535.0, -1e30, 1e30) and `DEFAULT_ATMOSPHERIC_SPECIAL_CODES` combining with existing financial codes.

---

## Phase 5: Service Integration

**New:** `services/covenant-radar-api/src/covenant_radar_api/worker/train_atmospheric_job.py`

Training job using atmospheric registry, NetCDF loader, feature engineering, preprocessing with atmospheric special codes, and existing backend infrastructure.

**Modify:** `services/covenant-radar-api/src/covenant_radar_api/worker/_test_hooks.py`

Add `AtmosphericLoaderProtocol` hook.

**New:** `services/covenant-radar-api/scripts/fetch_goes/` — GOES data fetching utility.

---

## Phase 6: Testing and Validation

**Test files needed:**
- `test_atmospheric_types.py` — all require_*/encode_* functions
- `test_netcdf_protocols.py` — protocol instantiation
- `test_netcdf_spatial.py` — entity ID and aggregation functions
- `test_netcdf_loader.py` — full loader with synthetic NetCDF
- `test_atmospheric_registry.py` — registry operations
- `test_atmospheric_preprocessing.py` — special code detection
- `test_atmospheric_training_integration.py` — end-to-end with fake data

**Test data:** `create_synthetic_netcdf()` factory in `testing.py` for generating test NetCDF files.

---

## Dependencies

Add to `libs/covenant_ml/pyproject.toml`:
- `xarray>=2024.1.0`
- `netCDF4>=1.6.0`
- `h5netcdf>=1.3.0` (alternative engine)

---

*Last updated: March 2026*
