# Large Dataset Loading with Progress Reporting

## Status: ✅ COMPLETE

Support for loading 16GB+ datasets with progress reporting across all optimization backends.

---

## What Was Implemented

### covenant_ml Library

| Module | Purpose |
|--------|---------|
| `loaders/chunked_csv_reader.py` | Polars-based chunked CSV reading with progress |
| `loaders/parquet_cache.py` | Parquet caching for fast repeat loads |
| `loaders/timeseries_csv_loader.py` | Time-series loading with Polars-native ops |
| `loaders/_polars_utils.py` | Protocol classes for Polars types |
| `loaders/_polars_aggregation.py` | Groupby aggregation (last/first/mean/statistics) |
| `loaders/_polars_encoding.py` | Categorical detection and encoding |

### Memory Optimization

**Before (caused OOM on 16GB+ datasets):**
```
CSV (16GB) → Polars DataFrame → Python list conversion → ~40-50GB peak
```

**After (memory-efficient):**
```
CSV (16GB) → Polars DataFrame → Polars groupby (in-place) → NumPy → ~10-12GB peak
```

### Parquet Caching

- Cache stored in `.cache/<config_hash>/` under dataset folder
- Auto-invalidates when source file is modified
- First load: Parse CSV → Save to parquet
- Repeat loads: Load from parquet (10-100x faster)

### covenant-radar-api Service

| Component | Status |
|-----------|--------|
| Phase callbacks (all 4 backends) | ✅ |
| Loading progress callbacks | ✅ |
| Rich CLI progress display | ✅ |
| 100% test coverage | ✅ |

---

## Progress Display Flow

```
User runs optimize command
    ↓
Phase: loading_data
    ↓
LoadProgress(phase="reading", percent=25%, rows=1,500,000)
    ↓
LoadProgress(phase="reading", percent=100%, rows=5,531,451)
    ↓
Phase: feature_engineering
    ↓
Phase: optimizing (trial callbacks)
    ↓
Phase: saving
```

---

## Type Flow

```
CLI (display.py)
    ↓
runner.py (phase_callback, loading_progress_callback)
    ↓
_test_hooks.py (PhaseCallbackProtocol)
    ↓
optimize_*_job.py (run_optimization)
    ↓
_optimize_common.py (load_any_dataset_cached)
    ↓
covenant_ml (ProgressCallbackProtocol)
    ↓
chunked_csv_reader.py / parquet_cache.py
```

---

*Completed: December 2025*
