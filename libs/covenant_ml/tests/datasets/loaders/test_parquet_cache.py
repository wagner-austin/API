"""Tests for parquet cache module."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.datasets.loaders.parquet_cache import (
    CACHE_DIR_NAME,
    FEATURES_FILE_NAME,
    LABELS_FILE_NAME,
    METADATA_FILE_NAME,
    _CacheLock,
    _compute_config_hash,
    _get_polars_dataframe,
    _get_polars_read_parquet,
    _parse_encodings,
    _parse_string_tuple,
    _serialize_encodings,
    _serialize_string_tuple,
    _split_escaped,
    check_cache,
    get_cache_dir,
    invalidate_cache,
    load_from_cache,
    save_to_cache,
)
from covenant_ml.datasets.types import (
    CategoricalEncoding,
    DatasetMeta,
    LoadedDataset,
    LoadProgress,
)


class _PolarsDataFrameProtocol(Protocol):
    """Protocol for Polars DataFrame write operations in tests."""

    def write_parquet(self, file: Path) -> None:
        """Write DataFrame to parquet file."""
        ...


class _PolarsDataFrameFactoryProtocol(Protocol):
    """Protocol for Polars DataFrame factory callable."""

    def __call__(self, data: dict[str, list[float]]) -> _PolarsDataFrameProtocol:
        """Create a DataFrame from a dictionary."""
        ...


def _make_test_dataset(
    name: str = "test_dataset",
    n_samples: int = 10,
    n_features: int = 3,
) -> LoadedDataset:
    """Create a test dataset for caching tests."""
    x_array: NDArray[np.float64] = np.random.rand(n_samples, n_features).astype(np.float64)
    y_list: list[int] = [0, 1] * (n_samples // 2)
    y_array: NDArray[np.int64] = np.array(y_list, dtype=np.int64)

    feature_names = tuple(f"feature_{i}" for i in range(n_features))
    positive_mask: NDArray[np.bool_] = y_array == 1
    sum_result: np.intp = np.sum(positive_mask)
    n_positive = int(sum_result)
    n_negative = n_samples - n_positive

    meta = DatasetMeta(
        name=name,
        n_samples=n_samples,
        n_features=n_features,
        n_positive=n_positive,
        n_negative=n_negative,
        positive_ratio=n_positive / n_samples,
        feature_names=feature_names,
        categorical_encodings=(),
    )

    return LoadedDataset(meta=meta, x=x_array, y=y_array)


def _make_dataset_with_categoricals() -> LoadedDataset:
    """Create a test dataset with categorical encodings."""
    x_list: list[list[float]] = [
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
    ]
    x_array: NDArray[np.float64] = np.array(x_list, dtype=np.float64)
    y_list: list[int] = [0, 1, 1, 0]
    y_array: NDArray[np.int64] = np.array(y_list, dtype=np.int64)

    encodings: tuple[CategoricalEncoding, ...] = (
        CategoricalEncoding(
            column_name="color",
            mapping=(("red", 0), ("blue", 1)),
            n_categories=2,
        ),
        CategoricalEncoding(
            column_name="size",
            mapping=(("small", 0), ("medium", 1), ("large", 2)),
            n_categories=3,
        ),
    )

    meta = DatasetMeta(
        name="categorical_test",
        n_samples=4,
        n_features=3,
        n_positive=2,
        n_negative=2,
        positive_ratio=0.5,
        feature_names=("color_encoded", "size_encoded", "value"),
        categorical_encodings=encodings,
    )

    return LoadedDataset(meta=meta, x=x_array, y=y_array)


class TestGetCacheDir:
    """Tests for get_cache_dir function."""

    def test_returns_correct_path(self) -> None:
        """Cache dir path includes folder and hash."""
        external_dir = Path("/data/external")
        folder = "my_dataset"
        config_hash = "abc123"

        result = get_cache_dir(external_dir, folder, config_hash)

        expected = external_dir / folder / CACHE_DIR_NAME / config_hash
        assert result == expected

    def test_cache_dir_name_constant(self) -> None:
        """Cache directory uses .cache name."""
        assert CACHE_DIR_NAME == ".cache"


class TestCheckCache:
    """Tests for check_cache function."""

    def test_no_cache_returns_invalid(self) -> None:
        """Non-existent cache returns invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "data.csv"
            source_path.write_text("a,b,c\n1,2,3\n")

            cache_dir = Path(tmpdir) / ".cache" / "test"

            result = check_cache(source_path, cache_dir)

            assert result["is_valid"] is False
            assert result["cache_mtime"] == 0.0

    def test_partial_cache_returns_invalid(self) -> None:
        """Cache missing files returns invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "data.csv"
            source_path.write_text("a,b,c\n1,2,3\n")

            cache_dir = Path(tmpdir) / ".cache" / "test"
            cache_dir.mkdir(parents=True)
            (cache_dir / METADATA_FILE_NAME).write_text("dummy")
            # Missing features and labels files

            result = check_cache(source_path, cache_dir)

            assert result["is_valid"] is False

    def test_stale_cache_returns_invalid(self) -> None:
        """Cache older than source returns invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "test"
            cache_dir.mkdir(parents=True)

            # Create cache files first
            for name in [METADATA_FILE_NAME, FEATURES_FILE_NAME, LABELS_FILE_NAME]:
                (cache_dir / name).write_text("dummy")

            time.sleep(0.1)  # Ensure different mtime

            # Create source file after cache
            source_path = Path(tmpdir) / "data.csv"
            source_path.write_text("a,b,c\n1,2,3\n")

            result = check_cache(source_path, cache_dir)

            assert result["is_valid"] is False

    def test_valid_cache_returns_valid(self) -> None:
        """Cache newer than source returns valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file first
            source_path = Path(tmpdir) / "data.csv"
            source_path.write_text("a,b,c\n1,2,3\n")

            time.sleep(0.1)  # Ensure different mtime

            # Create cache files after source
            cache_dir = Path(tmpdir) / ".cache" / "test"
            cache_dir.mkdir(parents=True)
            for name in [METADATA_FILE_NAME, FEATURES_FILE_NAME, LABELS_FILE_NAME]:
                (cache_dir / name).write_text("dummy")

            result = check_cache(source_path, cache_dir)

            assert result["is_valid"] is True
            assert result["cache_mtime"] > result["source_mtime"]


class TestSaveAndLoadCache:
    """Tests for save_to_cache and load_from_cache functions."""

    def test_save_creates_cache_files(self) -> None:
        """Save creates metadata, features, and labels files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "test"
            dataset = _make_test_dataset()

            save_to_cache(dataset, cache_dir)

            assert (cache_dir / METADATA_FILE_NAME).exists()
            assert (cache_dir / FEATURES_FILE_NAME).exists()
            assert (cache_dir / LABELS_FILE_NAME).exists()

    def test_cache_lock_acquire_and_release_removes_lock_dir(self) -> None:
        """Acquire then release removes the lock directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "locktest"
            lock = _CacheLock(cache_dir)
            lock.acquire(timeout_seconds=1.0, poll_seconds=0.01)
            lock_path = cache_dir.parent / (cache_dir.name + ".lock")
            assert lock_path.exists()
            lock.release()
            assert not lock_path.exists()

    def test_cache_lock_timeout_raises(self) -> None:
        """Acquire times out when lock directory already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "lockbusy"
            lock_path = cache_dir.parent / (cache_dir.name + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.mkdir()
            lock = _CacheLock(cache_dir)
            with pytest.raises(TimeoutError, match="Timed out acquiring cache lock"):
                lock.acquire(timeout_seconds=0.05, poll_seconds=0.01)

    def test_cache_lock_release_when_not_acquired(self) -> None:
        """Release is a no-op when lock was never acquired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "notacquired"
            lock = _CacheLock(cache_dir)
            # Release without acquiring - should not raise
            lock.release()
            # Lock directory should not exist
            lock_path = cache_dir.parent / (cache_dir.name + ".lock")
            assert not lock_path.exists()

    def test_cache_lock_release_when_lock_dir_already_removed(self) -> None:
        """Release handles case where lock directory was already removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "alreadyremoved"
            lock = _CacheLock(cache_dir)
            lock.acquire(timeout_seconds=1.0, poll_seconds=0.01)
            lock_path = cache_dir.parent / (cache_dir.name + ".lock")
            assert lock_path.exists()
            # Externally remove the lock directory
            lock_path.rmdir()
            assert not lock_path.exists()
            # Release should not raise even though lock dir is gone
            lock.release()

    def test_save_and_load_roundtrip(self) -> None:
        """Save and load preserves dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "test"
            original = _make_test_dataset(n_samples=10, n_features=3)

            save_to_cache(original, cache_dir)
            loaded = load_from_cache(cache_dir)

            assert loaded["meta"]["name"] == original["meta"]["name"]
            assert loaded["meta"]["n_samples"] == original["meta"]["n_samples"]
            assert loaded["meta"]["n_features"] == original["meta"]["n_features"]
            assert loaded["meta"]["n_positive"] == original["meta"]["n_positive"]
            assert loaded["meta"]["n_negative"] == original["meta"]["n_negative"]
            assert loaded["meta"]["feature_names"] == original["meta"]["feature_names"]
            np.testing.assert_array_almost_equal(loaded["x"], original["x"])
            np.testing.assert_array_equal(loaded["y"], original["y"])

    def test_save_and_load_with_categoricals(self) -> None:
        """Save and load preserves categorical encodings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "test"
            original = _make_dataset_with_categoricals()

            save_to_cache(original, cache_dir)
            loaded = load_from_cache(cache_dir)

            assert len(loaded["meta"]["categorical_encodings"]) == 2

            color_enc = loaded["meta"]["categorical_encodings"][0]
            assert color_enc["column_name"] == "color"
            assert color_enc["n_categories"] == 2
            assert ("red", 0) in color_enc["mapping"]
            assert ("blue", 1) in color_enc["mapping"]

            size_enc = loaded["meta"]["categorical_encodings"][1]
            assert size_enc["column_name"] == "size"
            assert size_enc["n_categories"] == 3

    def test_save_with_progress_callback(self) -> None:
        """Save calls progress callback with caching phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "test"
            dataset = _make_test_dataset()

            progress_updates: list[LoadProgress] = []

            def capture(progress: LoadProgress) -> None:
                progress_updates.append(progress)

            save_to_cache(dataset, cache_dir, progress_callback=capture)

            # Should have 4 progress updates (start, meta, features, labels)
            assert len(progress_updates) == 4
            # All should be caching phase
            for update in progress_updates:
                assert update["phase"] == "caching"
            # Last should be 100% complete
            assert progress_updates[-1]["percent_complete"] == 100.0

    def test_load_with_progress_callback(self) -> None:
        """Load calls progress callback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "test"
            dataset = _make_test_dataset()
            save_to_cache(dataset, cache_dir)

            progress_updates: list[LoadProgress] = []

            def capture(progress: LoadProgress) -> None:
                progress_updates.append(progress)

            load_from_cache(cache_dir, progress_callback=capture)

            # Should have 4 progress updates (start, features, labels, complete)
            assert len(progress_updates) == 4
            for update in progress_updates:
                assert update["phase"] == "loading_cache"
            # Last should be 100% complete
            assert progress_updates[-1]["percent_complete"] == 100.0
            # Last message should indicate completion with sample count
            assert "10 samples" in progress_updates[-1]["message"]

    def test_load_empty_metadata_raises(self) -> None:
        """Load raises ValueError for empty metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "test"
            cache_dir.mkdir(parents=True)

            # Create empty parquet files using Protocol-typed Polars DataFrame factory
            polars_mod = __import__("polars")
            dataframe_factory: _PolarsDataFrameFactoryProtocol = polars_mod.DataFrame
            empty_dict: dict[str, list[float]] = {}
            empty_df: _PolarsDataFrameProtocol = dataframe_factory(empty_dict)
            empty_df.write_parquet(cache_dir / METADATA_FILE_NAME)
            empty_df.write_parquet(cache_dir / FEATURES_FILE_NAME)
            empty_df.write_parquet(cache_dir / LABELS_FILE_NAME)

            with pytest.raises(ValueError, match="Empty metadata"):
                load_from_cache(cache_dir)


class TestInvalidateCache:
    """Tests for invalidate_cache function."""

    def test_invalidate_removes_files(self) -> None:
        """Invalidate removes all cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "test"
            dataset = _make_test_dataset()
            save_to_cache(dataset, cache_dir)

            assert (cache_dir / METADATA_FILE_NAME).exists()

            invalidate_cache(cache_dir)

            assert not (cache_dir / METADATA_FILE_NAME).exists()
            assert not (cache_dir / FEATURES_FILE_NAME).exists()
            assert not (cache_dir / LABELS_FILE_NAME).exists()

    def test_invalidate_nonexistent_cache(self) -> None:
        """Invalidate on non-existent cache does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "nonexistent"

            # Should not raise
            invalidate_cache(cache_dir)


class TestSerializationHelpers:
    """Tests for encoding serialization helpers."""

    def test_serialize_empty_encodings(self) -> None:
        """Empty encodings serialize to []."""
        result = _serialize_encodings(())
        assert result == "[]"

    def test_parse_empty_encodings(self) -> None:
        """[] parses to empty tuple."""
        result = _parse_encodings("[]")
        assert result == ()

    def test_serialize_parse_roundtrip(self) -> None:
        """Serialize and parse preserves encodings."""
        encodings: tuple[CategoricalEncoding, ...] = (
            CategoricalEncoding(
                column_name="color",
                mapping=(("red", 0), ("blue", 1)),
                n_categories=2,
            ),
        )

        serialized = _serialize_encodings(encodings)
        parsed = _parse_encodings(serialized)

        assert len(parsed) == 1
        assert parsed[0]["column_name"] == "color"
        assert parsed[0]["n_categories"] == 2
        assert ("red", 0) in parsed[0]["mapping"]

    def test_serialize_with_special_chars(self) -> None:
        """Serialize handles special characters in values."""
        encodings: tuple[CategoricalEncoding, ...] = (
            CategoricalEncoding(
                column_name="test",
                mapping=(("a|b", 0), ("c,d", 1), ("e\\f", 2)),
                n_categories=3,
            ),
        )

        serialized = _serialize_encodings(encodings)
        parsed = _parse_encodings(serialized)

        assert ("a|b", 0) in parsed[0]["mapping"]
        assert ("c,d", 1) in parsed[0]["mapping"]
        assert ("e\\f", 2) in parsed[0]["mapping"]

    def test_serialize_empty_string_tuple(self) -> None:
        """Empty tuple serializes to empty string."""
        result = _serialize_string_tuple(())
        assert result == ""

    def test_parse_empty_string_tuple(self) -> None:
        """Empty string parses to empty tuple."""
        result = _parse_string_tuple("")
        assert result == ()

    def test_string_tuple_roundtrip(self) -> None:
        """String tuple serialize/parse roundtrip."""
        values = ("feature_1", "feature_2", "feature_3")

        serialized = _serialize_string_tuple(values)
        parsed = _parse_string_tuple(serialized)

        assert parsed == values

    def test_string_tuple_with_pipes(self) -> None:
        """String tuple handles pipe characters."""
        values = ("a|b", "c|d|e")

        serialized = _serialize_string_tuple(values)
        parsed = _parse_string_tuple(serialized)

        assert parsed == values

    def test_split_escaped_basic(self) -> None:
        """Split escaped handles basic case."""
        result = _split_escaped("a,b,c", ",")
        assert result == ["a", "b", "c"]

    def test_split_escaped_with_escapes(self) -> None:
        """Split escaped respects escape sequences."""
        result = _split_escaped("a\\,b,c", ",")
        assert result == ["a\\,b", "c"]

    def test_compute_config_hash(self) -> None:
        """Config hash returns consistent hash."""
        config_str = '{"dataset": "test", "n_trials": 10}'

        hash1 = _compute_config_hash(config_str)
        hash2 = _compute_config_hash(config_str)

        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA256

    def test_compute_config_hash_different_inputs(self) -> None:
        """Different configs produce different hashes."""
        hash1 = _compute_config_hash("config1")
        hash2 = _compute_config_hash("config2")

        assert hash1 != hash2

    def test_parse_encodings_empty_parts(self) -> None:
        """Parse encodings skips empty parts in serialized string."""
        # Simulate malformed string with empty parts
        result = _parse_encodings(";;")
        assert result == ()

    def test_parse_encodings_short_sections(self) -> None:
        """Parse encodings skips sections with fewer than 3 parts."""
        # Only 2 parts (column_name|n_categories) - missing mapping
        result = _parse_encodings("col|2")
        assert result == ()

    def test_parse_encodings_empty_mapping(self) -> None:
        """Parse encodings handles empty mapping string."""
        # Valid structure but empty mapping
        result = _parse_encodings("col|0|")
        assert len(result) == 1
        assert result[0]["column_name"] == "col"
        assert result[0]["n_categories"] == 0
        assert result[0]["mapping"] == ()

    def test_parse_encodings_entry_without_colon(self) -> None:
        """Parse encodings skips entries without colon separator."""
        # Mapping entry without colon delimiter
        result = _parse_encodings("col|1|invalid_entry")
        assert len(result) == 1
        assert result[0]["mapping"] == ()


class TestInvalidateCachePartial:
    """Tests for partial cache invalidation."""

    def test_invalidate_partial_cache(self) -> None:
        """Invalidate handles missing files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "partial"
            cache_dir.mkdir(parents=True)

            # Only create metadata file, not features/labels
            (cache_dir / METADATA_FILE_NAME).write_text("dummy")

            # Should not raise even though some files don't exist
            invalidate_cache(cache_dir)

            # Metadata should be removed
            assert not (cache_dir / METADATA_FILE_NAME).exists()


class TestPolarsHelpers:
    """Tests for Polars module helper functions."""

    def test_get_polars_read_parquet_returns_callable(self) -> None:
        """Helper returns a callable read_parquet function."""
        read_fn = _get_polars_read_parquet()
        assert callable(read_fn)

    def test_get_polars_dataframe_returns_callable(self) -> None:
        """Helper returns a callable DataFrame constructor."""
        df_fn = _get_polars_dataframe()
        assert callable(df_fn)
