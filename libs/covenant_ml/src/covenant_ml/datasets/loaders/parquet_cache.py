"""Parquet cache for dataset loading using Polars.

Caches processed datasets as parquet files for fast repeated loading.
Cache is invalidated when source file is modified.

Internal module - used by csv_loader and timeseries_csv_loader.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import TracebackType
from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import (
    CategoricalEncoding,
    DatasetMeta,
    LoadedDataset,
    LoadProgress,
)

# Cache directory name (relative to dataset folder)
CACHE_DIR_NAME: str = ".cache"

# Metadata file name in cache
METADATA_FILE_NAME: str = "meta.parquet"

#: Group codes sidecar, written only for datasets whose config names a
#: group_column. Its absence on load means "no groups", so ungrouped caches
#: stay exactly as they were.
GROUPS_FILE_NAME: str = "groups.parquet"

# Features file name in cache
FEATURES_FILE_NAME: str = "features.parquet"

# Labels file name in cache
LABELS_FILE_NAME: str = "labels.parquet"


class CacheInfo(TypedDict, total=True):
    """Information about cache validity.

    Attributes:
        cache_dir: Path to cache directory.
        is_valid: Whether cache exists and is valid.
        source_mtime: Source file modification time.
        cache_mtime: Cache file modification time (0 if no cache).
    """

    cache_dir: Path
    is_valid: bool
    source_mtime: float
    cache_mtime: float


class _PolarsDataFrameProtocol(Protocol):
    """Protocol for Polars DataFrame with required operations."""

    @property
    def columns(self) -> list[str]:
        """Return column names."""
        ...

    @property
    def height(self) -> int:
        """Return number of rows."""
        ...

    def to_numpy(self) -> NDArray[np.float64]:
        """Convert to numpy array."""
        ...

    def write_parquet(self, file: str | Path, compression: str) -> None:
        """Write to parquet file."""
        ...


class _PolarsReadParquetProtocol(Protocol):
    """Protocol for Polars read_parquet function."""

    def __call__(self, source: str | Path) -> _PolarsDataFrameProtocol:
        """Read parquet file into DataFrame."""
        ...


class _PolarsFromDictProtocol(Protocol):
    """Protocol for Polars DataFrame.from_dict constructor."""

    def __call__(
        self,
        data: dict[str, list[float] | list[int] | list[str]],
    ) -> _PolarsDataFrameProtocol:
        """Create DataFrame from dictionary."""
        ...


def _get_polars_read_parquet() -> _PolarsReadParquetProtocol:
    """Get Polars read_parquet function with typing.

    Returns:
        Typed read_parquet function.
    """
    polars_mod = __import__("polars")
    read_fn: _PolarsReadParquetProtocol = polars_mod.read_parquet
    return read_fn


def _get_polars_dataframe() -> _PolarsFromDictProtocol:
    """Get Polars DataFrame constructor with typing.

    Returns:
        Typed DataFrame constructor.
    """
    polars_mod = __import__("polars")
    df_class: _PolarsFromDictProtocol = polars_mod.DataFrame
    return df_class


def _get_lock_dir(cache_dir: Path) -> Path:
    """Get the lock directory path for a cache directory.

    Args:
        cache_dir: Path to the cache directory (.../.cache/<hash>).

    Returns:
        Path to the corresponding lock directory (.../.cache/<hash>.lock).
    """
    return cache_dir.parent / f"{cache_dir.name}.lock"


class _CacheLock:
    """Filesystem lock using an exclusive directory.

    Creates a lock directory to acquire the lock and removes it to release.
    Directory creation is atomic on local filesystems across platforms, which
    makes this a simple, dependency-free cross-platform lock.

    Args:
        cache_dir: Target cache directory for which we coordinate access.
    """

    _lock_dir: Path
    _acquired: bool

    def __init__(self, cache_dir: Path) -> None:
        self._lock_dir = _get_lock_dir(cache_dir)
        self._acquired = False

    def acquire(self, timeout_seconds: float = 30.0, poll_seconds: float = 0.05) -> None:
        """Acquire the lock, waiting up to timeout_seconds.

        Raises:
            TimeoutError: If the lock cannot be acquired within the timeout.
        """
        deadline = time.monotonic() + timeout_seconds
        # Ensure parent exists (e.g., .cache directory)
        self._lock_dir.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._lock_dir.mkdir()
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring cache lock: {self._lock_dir}"
                    ) from None
                time.sleep(poll_seconds)
                continue
            self._acquired = True
            return

    def release(self) -> None:
        """Release the lock if held."""
        if self._acquired:
            # Remove lock directory if it still exists
            if self._lock_dir.exists():
                self._lock_dir.rmdir()
            self._acquired = False

    # Context manager helpers
    def __enter__(self) -> _CacheLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def _compute_config_hash(config_str: str) -> str:
    """Compute hash of configuration for cache key.

    Args:
        config_str: String representation of configuration.

    Returns:
        Hex digest of SHA256 hash (first 16 chars).
    """
    hasher = hashlib.sha256()
    hasher.update(config_str.encode("utf-8"))
    return hasher.hexdigest()[:16]


def get_cache_dir(
    external_dir: Path,
    folder: str,
    config_hash: str,
) -> Path:
    """Get cache directory path for a dataset.

    Args:
        external_dir: Root directory containing dataset folders.
        folder: Dataset folder name.
        config_hash: Hash of configuration for cache key.

    Returns:
        Path to cache directory.
    """
    return external_dir / folder / CACHE_DIR_NAME / config_hash


def check_cache(
    source_path: Path,
    cache_dir: Path,
) -> CacheInfo:
    """Check if valid cache exists for a dataset.

    Cache is valid if:
    - Cache directory exists
    - All required files exist
    - Cache was created after source file was last modified

    Args:
        source_path: Path to source CSV file.
        cache_dir: Path to cache directory.

    Returns:
        CacheInfo with validity status.
    """
    source_mtime = source_path.stat().st_mtime if source_path.exists() else 0.0

    meta_path = cache_dir / METADATA_FILE_NAME
    features_path = cache_dir / FEATURES_FILE_NAME
    labels_path = cache_dir / LABELS_FILE_NAME

    # Check all files exist
    if not all(p.exists() for p in [meta_path, features_path, labels_path]):
        return CacheInfo(
            cache_dir=cache_dir,
            is_valid=False,
            source_mtime=source_mtime,
            cache_mtime=0.0,
        )

    # Check cache is newer than source
    cache_mtime = min(p.stat().st_mtime for p in [meta_path, features_path, labels_path])
    is_valid = cache_mtime > source_mtime

    return CacheInfo(
        cache_dir=cache_dir,
        is_valid=is_valid,
        source_mtime=source_mtime,
        cache_mtime=cache_mtime,
    )


def _report_progress(
    callback: ProgressCallbackProtocol | None,
    progress: LoadProgress,
) -> None:
    """Report progress if callback is provided.

    Args:
        callback: Optional progress callback.
        progress: Progress state to report.
    """
    if callback is not None:
        callback(progress)


class _PolarsDataFrameRowProtocol(Protocol):
    """Protocol for Polars DataFrame with row access."""

    @property
    def columns(self) -> list[str]:
        """Return column names."""
        ...

    @property
    def height(self) -> int:
        """Return number of rows."""
        ...

    def row(self, index: int) -> tuple[str | int | float, ...]:
        """Return single row as tuple."""
        ...

    def to_numpy(self) -> NDArray[np.float64]:
        """Convert to numpy array."""
        ...

    def __getitem__(self, key: str) -> _PolarsSeriesProtocol:
        """Get column by name."""
        ...


class _PolarsSeriesProtocol(Protocol):
    """Protocol for Polars Series."""

    def to_numpy(self) -> NDArray[np.int64]:
        """Convert to numpy array."""
        ...


def _read_parquet_typed(path: Path) -> _PolarsDataFrameRowProtocol:
    """Read parquet file into typed DataFrame.

    Args:
        path: Path to parquet file.

    Returns:
        Typed DataFrame.
    """
    polars_mod = __import__("polars")
    df: _PolarsDataFrameRowProtocol = polars_mod.read_parquet(path)
    return df


def load_from_cache(
    cache_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> LoadedDataset:
    """Load dataset from parquet cache.

    Args:
        cache_dir: Path to cache directory.
        progress_callback: Optional callback for progress updates.

    Returns:
        LoadedDataset loaded from cache.

    Raises:
        FileNotFoundError: If cache files don't exist.
        ValueError: If cache data is invalid.
    """
    meta_path = cache_dir / METADATA_FILE_NAME
    features_path = cache_dir / FEATURES_FILE_NAME
    labels_path = cache_dir / LABELS_FILE_NAME

    _report_progress(
        callback=progress_callback,
        progress=LoadProgress(
            phase="loading_cache",
            bytes_read=0,
            bytes_total=0,
            rows_processed=0,
            rows_total=0,
            percent_complete=0.0,
            message="Loading from cache...",
        ),
    )

    # Load metadata
    meta_df = _read_parquet_typed(meta_path)
    if meta_df.height == 0:
        raise ValueError(f"Empty metadata in cache: {meta_path}")

    # Extract metadata values - read first row
    meta_row = meta_df.row(0)
    meta_cols = meta_df.columns

    # Build dict from columns and row
    meta_dict: dict[str, str | int | float] = {}
    for i, col in enumerate(meta_cols):
        meta_dict[col] = meta_row[i]

    # Reconstruct categorical encodings from stored string
    encodings_str = str(meta_dict["categorical_encodings"])
    categorical_encodings: tuple[CategoricalEncoding, ...] = _parse_encodings(encodings_str)

    # Parse feature names
    feature_names_str = str(meta_dict["feature_names"])
    feature_names: tuple[str, ...] = _parse_string_tuple(feature_names_str)

    n_samples = int(meta_dict["n_samples"])

    meta = DatasetMeta(
        name=str(meta_dict["name"]),
        n_samples=n_samples,
        n_features=int(meta_dict["n_features"]),
        n_positive=int(meta_dict["n_positive"]),
        n_negative=int(meta_dict["n_negative"]),
        positive_ratio=float(meta_dict["positive_ratio"]),
        feature_names=feature_names,
        categorical_encodings=categorical_encodings,
    )

    _report_progress(
        callback=progress_callback,
        progress=LoadProgress(
            phase="loading_cache",
            bytes_read=0,
            bytes_total=0,
            rows_processed=0,
            rows_total=n_samples,
            percent_complete=33.0,
            message="Loading features from cache...",
        ),
    )

    # Load features
    features_df = _read_parquet_typed(features_path)
    x_array: NDArray[np.float64] = features_df.to_numpy().astype(np.float64)

    _report_progress(
        callback=progress_callback,
        progress=LoadProgress(
            phase="loading_cache",
            bytes_read=0,
            bytes_total=0,
            rows_processed=0,
            rows_total=n_samples,
            percent_complete=66.0,
            message="Loading labels from cache...",
        ),
    )

    # Load labels
    labels_df = _read_parquet_typed(labels_path)
    y_array: NDArray[np.int64] = labels_df["y"].to_numpy().astype(np.int64)

    # Load group codes when the sidecar exists (grouped datasets only)
    groups_path = cache_dir / GROUPS_FILE_NAME
    groups_array: NDArray[np.int64] | None = None
    if groups_path.exists():
        groups_df = _read_parquet_typed(groups_path)
        groups_array = groups_df["g"].to_numpy().astype(np.int64)

    _report_progress(
        callback=progress_callback,
        progress=LoadProgress(
            phase="loading_cache",
            bytes_read=0,
            bytes_total=0,
            rows_processed=n_samples,
            rows_total=n_samples,
            percent_complete=100.0,
            message=f"Loaded {n_samples:,} samples from cache",
        ),
    )

    return LoadedDataset(meta=meta, x=x_array, y=y_array, groups=groups_array)


class _PolarsDataFrameWriteProtocol(Protocol):
    """Protocol for Polars DataFrame with write access."""

    def write_parquet(self, file: str | Path, compression: str) -> None:
        """Write to parquet file."""
        ...


def _create_dataframe_typed(
    data: dict[str, list[str] | list[int] | list[float]],
) -> _PolarsDataFrameWriteProtocol:
    """Create typed DataFrame from dictionary.

    Args:
        data: Dictionary of column data.

    Returns:
        Typed DataFrame.
    """
    polars_mod = __import__("polars")
    df: _PolarsDataFrameWriteProtocol = polars_mod.DataFrame(data)
    return df


def save_to_cache(
    dataset: LoadedDataset,
    cache_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> None:
    """Save dataset to parquet cache.

    Creates cache directory if it doesn't exist.

    Args:
        dataset: LoadedDataset to cache.
        cache_dir: Path to cache directory.
        progress_callback: Optional callback for progress updates.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / METADATA_FILE_NAME
    features_path = cache_dir / FEATURES_FILE_NAME
    labels_path = cache_dir / LABELS_FILE_NAME

    meta = dataset["meta"]

    _report_progress(
        callback=progress_callback,
        progress=LoadProgress(
            phase="caching",
            bytes_read=0,
            bytes_total=0,
            rows_processed=0,
            rows_total=meta["n_samples"],
            percent_complete=0.0,
            message="Writing cache...",
        ),
    )

    # Save metadata
    encodings_str = _serialize_encodings(meta["categorical_encodings"])
    feature_names_str = _serialize_string_tuple(meta["feature_names"])

    meta_data: dict[str, list[str] | list[int] | list[float]] = {
        "name": [meta["name"]],
        "n_samples": [meta["n_samples"]],
        "n_features": [meta["n_features"]],
        "n_positive": [meta["n_positive"]],
        "n_negative": [meta["n_negative"]],
        "positive_ratio": [meta["positive_ratio"]],
        "feature_names": [feature_names_str],
        "categorical_encodings": [encodings_str],
    }
    meta_df = _create_dataframe_typed(meta_data)
    meta_df.write_parquet(meta_path, compression="snappy")

    _report_progress(
        callback=progress_callback,
        progress=LoadProgress(
            phase="caching",
            bytes_read=0,
            bytes_total=0,
            rows_processed=0,
            rows_total=meta["n_samples"],
            percent_complete=33.0,
            message="Writing features to cache...",
        ),
    )

    # Save features
    features_data: dict[str, list[str] | list[int] | list[float]] = {}
    for i, col_name in enumerate(meta["feature_names"]):
        col_values: list[float] = dataset["x"][:, i].tolist()
        features_data[col_name] = col_values
    features_df = _create_dataframe_typed(features_data)
    features_df.write_parquet(features_path, compression="snappy")

    _report_progress(
        callback=progress_callback,
        progress=LoadProgress(
            phase="caching",
            bytes_read=0,
            bytes_total=0,
            rows_processed=0,
            rows_total=meta["n_samples"],
            percent_complete=66.0,
            message="Writing labels to cache...",
        ),
    )

    # Save labels - extract values with typed intermediates
    y_numpy: NDArray[np.int64] = dataset["y"]
    y_list: list[int] = []
    for i in range(len(y_numpy)):
        y_value: np.int64 = y_numpy[i]
        y_list.append(int(y_value))

    labels_data: dict[str, list[str] | list[int] | list[float]] = {"y": y_list}
    labels_df = _create_dataframe_typed(labels_data)
    labels_df.write_parquet(labels_path, compression="snappy")

    # Save group codes sidecar for grouped datasets
    groups_numpy = dataset["groups"]
    if groups_numpy is not None:
        g_list: list[int] = []
        for i in range(len(groups_numpy)):
            g_value: np.int64 = groups_numpy[i]
            g_list.append(int(g_value))
        groups_data: dict[str, list[str] | list[int] | list[float]] = {"g": g_list}
        groups_df = _create_dataframe_typed(groups_data)
        groups_df.write_parquet(cache_dir / GROUPS_FILE_NAME, compression="snappy")

    _report_progress(
        callback=progress_callback,
        progress=LoadProgress(
            phase="caching",
            bytes_read=0,
            bytes_total=0,
            rows_processed=meta["n_samples"],
            rows_total=meta["n_samples"],
            percent_complete=100.0,
            message=f"Cached {meta['n_samples']:,} samples",
        ),
    )


def _serialize_encodings(encodings: tuple[CategoricalEncoding, ...]) -> str:
    """Serialize categorical encodings to string.

    Args:
        encodings: Tuple of categorical encodings.

    Returns:
        String representation for storage.
    """
    if not encodings:
        return "[]"

    parts: list[str] = []
    for enc in encodings:
        mapping_parts: list[str] = []
        for val, code in enc["mapping"]:
            # Escape special characters
            escaped_val = val.replace("\\", "\\\\").replace("|", "\\|").replace(",", "\\,")
            mapping_parts.append(f"{escaped_val}:{code}")
        mapping_str = ",".join(mapping_parts)
        parts.append(f"{enc['column_name']}|{enc['n_categories']}|{mapping_str}")

    return ";".join(parts)


def _parse_encodings(encodings_str: str) -> tuple[CategoricalEncoding, ...]:
    """Parse categorical encodings from string.

    Args:
        encodings_str: String representation from storage.

    Returns:
        Tuple of categorical encodings.
    """
    if encodings_str == "[]" or not encodings_str:
        return ()

    result: list[CategoricalEncoding] = []
    for part in encodings_str.split(";"):
        if not part:
            continue

        sections = part.split("|")
        if len(sections) < 3:
            continue

        column_name = sections[0]
        n_categories = int(sections[1])
        mapping_str = "|".join(sections[2:])  # Rejoin in case of escaped pipes

        mapping: list[tuple[str, int]] = []
        if mapping_str:
            # Parse mapping entries, handling escaped characters
            entries = _split_escaped(mapping_str, ",")
            for entry in entries:
                if ":" in entry:
                    val, code_str = entry.rsplit(":", 1)
                    # Unescape special characters
                    val = val.replace("\\,", ",").replace("\\|", "|").replace("\\\\", "\\")
                    mapping.append((val, int(code_str)))

        result.append(
            CategoricalEncoding(
                column_name=column_name,
                mapping=tuple(mapping),
                n_categories=n_categories,
            )
        )

    return tuple(result)


def _split_escaped(text: str, delimiter: str) -> list[str]:
    """Split string by delimiter, respecting escaped delimiters.

    Args:
        text: String to split.
        delimiter: Delimiter character.

    Returns:
        List of split parts.
    """
    result: list[str] = []
    current: list[str] = []
    i = 0

    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            # Escaped character - include both
            current.append(text[i])
            current.append(text[i + 1])
            i += 2
        elif text[i] == delimiter:
            result.append("".join(current))
            current = []
            i += 1
        else:
            current.append(text[i])
            i += 1

    result.append("".join(current))
    return result


def _serialize_string_tuple(values: tuple[str, ...]) -> str:
    """Serialize tuple of strings to string.

    Args:
        values: Tuple of strings.

    Returns:
        String representation for storage.
    """
    if not values:
        return ""

    escaped: list[str] = []
    for val in values:
        esc = val.replace("\\", "\\\\").replace("|", "\\|")
        escaped.append(esc)

    return "|".join(escaped)


def _parse_string_tuple(text: str) -> tuple[str, ...]:
    """Parse tuple of strings from string.

    Args:
        text: String representation from storage.

    Returns:
        Tuple of strings.
    """
    if not text:
        return ()

    parts = _split_escaped(text, "|")
    result: list[str] = []
    for part in parts:
        unescaped = part.replace("\\|", "|").replace("\\\\", "\\")
        result.append(unescaped)

    return tuple(result)


def invalidate_cache(cache_dir: Path) -> None:
    """Remove cache files for a dataset.

    Args:
        cache_dir: Path to cache directory.
    """
    if cache_dir.exists():
        with _CacheLock(cache_dir):
            for file_name in [METADATA_FILE_NAME, FEATURES_FILE_NAME, LABELS_FILE_NAME]:
                file_path = cache_dir / file_name
                if file_path.exists():
                    file_path.unlink()


__all__ = [
    "CACHE_DIR_NAME",
    "CacheInfo",
    "_get_polars_dataframe",
    "_get_polars_read_parquet",
    "check_cache",
    "get_cache_dir",
    "invalidate_cache",
    "load_from_cache",
    "save_to_cache",
]
