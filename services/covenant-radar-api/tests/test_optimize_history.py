"""Tests for scripts/optimize/history.py - unified history tracking.

Tests the optimization history manager with unified history entries.
Uses strict typing with no Any, casts, or type: ignore.
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams
from covenant_ml.types import BackendName
from platform_core.json_utils import JSONObject, dump_json_str
from scripts.optimize.cli import DatasetName, FeaturePreset
from scripts.optimize.history import (
    HISTORY_FILENAME,
    OptimizationHistory,
    UnifiedHistoryEntry,
    _decode_backend,
    _decode_history_entry,
    result_to_entry,
)

from covenant_radar_api.worker.optimize_types import UnifiedOptimizationResult


def _make_history_entry(
    backend: str = "xgboost",
    dataset: DatasetName = "taiwan",
    feature_preset: FeaturePreset = "full",
    best_val_auc: float = 0.85,
    timestamp: str = "2024-01-01T00:00:00Z",
) -> UnifiedHistoryEntry:
    """Create a unified history entry for testing.

    Args:
        backend: Backend name string.
        dataset: Dataset name.
        feature_preset: Feature preset name.
        best_val_auc: Best validation AUC score.
        timestamp: ISO timestamp string.

    Returns:
        UnifiedHistoryEntry with test values.
    """
    return UnifiedHistoryEntry(
        timestamp=timestamp,
        backend=backend,
        dataset=dataset,
        feature_preset=feature_preset,
        n_trials=50,
        n_samples=1000,
        n_features=100,
        best_val_auc=best_val_auc,
        best_trial_number=25,
        duration_seconds=60.0,
    )


class TestDecodeBackendError:
    """Tests for _decode_backend error handling."""

    def test_invalid_backend_raises_value_error(self) -> None:
        """Test _decode_backend raises ValueError for invalid backend."""
        import pytest

        obj: JSONObject = {"backend": "invalid_backend"}
        with pytest.raises(ValueError, match="Invalid backend: invalid_backend"):
            _decode_backend(obj)

    def test_valid_backends_are_accepted(self) -> None:
        """Test _decode_backend accepts all 7 valid backends."""
        valid_backends = [
            "xgboost",
            "mlp",
            "lightgbm",
            "lstm",
            "cleargbm",
            "logreg",
            "random_forest",
        ]
        for name in valid_backends:
            obj: JSONObject = {"backend": name}
            result = _decode_backend(obj)
            assert result == name


class TestDecodeHistoryEntry:
    """Tests for _decode_history_entry with unified entries."""

    def test_decodes_unified_json_object(self) -> None:
        """Test decoding a valid unified JSON object."""
        obj: JSONObject = {
            "timestamp": "2024-01-01T00:00:00Z",
            "backend": "xgboost",
            "dataset": "taiwan",
            "feature_preset": "full",
            "n_trials": 50,
            "n_samples": 1000,
            "n_features": 100,
            "best_val_auc": 0.85,
            "best_trial_number": 25,
            "duration_seconds": 60.0,
        }

        entry: UnifiedHistoryEntry = _decode_history_entry(obj)

        assert entry["timestamp"] == "2024-01-01T00:00:00Z"
        assert entry["backend"] == "xgboost"
        assert entry["dataset"] == "taiwan"
        assert entry["feature_preset"] == "full"
        assert entry["n_trials"] == 50
        assert entry["best_val_auc"] == 0.85
        assert entry["best_trial_number"] == 25
        assert entry["duration_seconds"] == 60.0

    def test_decodes_mlp_backend_entry(self) -> None:
        """Test decoding a valid entry with MLP backend."""
        obj: JSONObject = {
            "timestamp": "2024-01-01T00:00:00Z",
            "backend": "mlp",
            "dataset": "taiwan",
            "feature_preset": "full",
            "n_trials": 50,
            "n_samples": 1000,
            "n_features": 100,
            "best_val_auc": 0.85,
            "best_trial_number": 25,
            "duration_seconds": 60.0,
        }

        entry: UnifiedHistoryEntry = _decode_history_entry(obj)

        assert entry["backend"] == "mlp"
        assert entry["best_val_auc"] == 0.85

    def test_ignores_extra_fields(self) -> None:
        """Test decoding ignores extra fields from legacy history entries."""
        obj: JSONObject = {
            "timestamp": "2024-01-01T00:00:00Z",
            "backend": "lightgbm",
            "dataset": "taiwan",
            "feature_preset": "full",
            "n_trials": 50,
            "n_samples": 1000,
            "n_features": 100,
            "best_val_auc": 0.85,
            "best_trial_number": 25,
            "duration_seconds": 60.0,
            "best_num_leaves": 31,
            "best_learning_rate": 0.1,
        }

        entry: UnifiedHistoryEntry = _decode_history_entry(obj)

        assert entry["backend"] == "lightgbm"
        assert entry["best_val_auc"] == 0.85


class TestResultToEntry:
    """Tests for result_to_entry conversion function."""

    def test_converts_xgboost_result(self) -> None:
        """Test converting unified XGBoost result to history entry."""
        result: UnifiedOptimizationResult = UnifiedOptimizationResult(
            backend="xgboost",
            status="complete",
            dataset="us",
            n_samples=2000,
            n_features=200,
            feature_preset="log_only",
            n_trials_complete=100,
            n_trials_pruned=10,
            n_trials_failed=0,
            best_trial_number=50,
            best_value=0.90,
            best_int_params=SampledIntParams(max_depth=8, n_estimators=200),
            best_float_params=SampledFloatParams(
                learning_rate=0.05,
                reg_alpha=0.001,
                reg_lambda=0.001,
                subsample=0.9,
                colsample_bytree=0.9,
            ),
            best_string_params=SampledStringParams(),
            duration_seconds=120.0,
        )

        entry: UnifiedHistoryEntry = result_to_entry(result, 120.0)

        assert entry["backend"] == "xgboost"
        assert entry["dataset"] == "us"
        assert entry["feature_preset"] == "log_only"
        assert entry["n_trials"] == 100
        assert entry["best_val_auc"] == 0.90
        assert entry["best_trial_number"] == 50
        assert entry["duration_seconds"] == 120.0
        assert "T" in entry["timestamp"]

    def test_converts_mlp_result(self) -> None:
        """Test converting unified MLP result to history entry."""
        result: UnifiedOptimizationResult = UnifiedOptimizationResult(
            backend="mlp",
            status="complete",
            dataset="taiwan",
            n_samples=1000,
            n_features=100,
            feature_preset="full",
            n_trials_complete=50,
            n_trials_pruned=5,
            n_trials_failed=0,
            best_trial_number=25,
            best_value=0.88,
            best_int_params=SampledIntParams(n_layers=3, hidden_size=128, batch_size=64),
            best_float_params=SampledFloatParams(learning_rate=0.001, dropout=0.2),
            best_string_params=SampledStringParams(),
            duration_seconds=60.0,
        )

        entry: UnifiedHistoryEntry = result_to_entry(result, 60.0)

        assert entry["backend"] == "mlp"
        assert entry["dataset"] == "taiwan"
        assert entry["best_val_auc"] == 0.88
        assert "T" in entry["timestamp"]


class TestOptimizationHistoryBasics:
    """Tests for OptimizationHistory class basic operations."""

    def test_for_output_dir_creates_correct_path(self, tmp_path: Path) -> None:
        """Test for_output_dir creates history at correct path."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        assert history._path == tmp_path / HISTORY_FILENAME

    def test_load_handles_missing_file(self, tmp_path: Path) -> None:
        """Test load handles missing history file gracefully."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        history.load()
        assert history._loaded is True
        assert len(history._entries) == 0

    def test_load_only_runs_once(self, tmp_path: Path) -> None:
        """Test load only processes file once even if called multiple times."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        history.load()
        assert history._loaded is True

        entry = _make_history_entry()
        history_file = tmp_path / HISTORY_FILENAME
        history_file.write_text(dump_json_str(entry, compact=True) + "\n", encoding="utf-8")

        history.load()
        assert len(history._entries) == 0

    def test_load_parses_jsonl_file(self, tmp_path: Path) -> None:
        """Test load correctly parses JSONL history file."""
        history_file = tmp_path / HISTORY_FILENAME
        entry1 = _make_history_entry(best_val_auc=0.80)
        entry2 = _make_history_entry(best_val_auc=0.85)
        content = dump_json_str(entry1, compact=True) + "\n" + dump_json_str(entry2, compact=True)
        history_file.write_text(content, encoding="utf-8")

        history = OptimizationHistory.for_output_dir(tmp_path)
        history.load()

        assert len(history._entries) == 2
        assert history._entries[0]["best_val_auc"] == 0.80
        assert history._entries[1]["best_val_auc"] == 0.85

    def test_load_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test load skips empty lines in JSONL file."""
        history_file = tmp_path / HISTORY_FILENAME
        entry = _make_history_entry()
        content = "\n" + dump_json_str(entry, compact=True) + "\n\n"
        history_file.write_text(content, encoding="utf-8")

        history = OptimizationHistory.for_output_dir(tmp_path)
        history.load()

        assert len(history._entries) == 1


class TestOptimizationHistoryAppend:
    """Tests for OptimizationHistory append operations."""

    def test_append_writes_to_file(self, tmp_path: Path) -> None:
        """Test append persists entry to file."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        entry = _make_history_entry()

        history.append(entry)

        history_file = tmp_path / HISTORY_FILENAME
        assert history_file.exists()
        content = history_file.read_text(encoding="utf-8")
        assert "taiwan" in content
        assert "0.85" in content

    def test_append_calls_load_if_not_loaded(self, tmp_path: Path) -> None:
        """Test append calls load if history not yet loaded."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        entry = _make_history_entry()

        assert history._loaded is False
        history.append(entry)
        assert history._loaded is True


class TestOptimizationHistoryGetPreviousBest:
    """Tests for OptimizationHistory.get_previous_best method."""

    def test_returns_most_recent(self, tmp_path: Path) -> None:
        """Test get_previous_best returns most recent matching entry."""
        history_file = tmp_path / HISTORY_FILENAME
        entry1 = _make_history_entry(best_val_auc=0.80, timestamp="2024-01-01T00:00:00Z")
        entry2 = _make_history_entry(best_val_auc=0.85, timestamp="2024-01-02T00:00:00Z")
        content = dump_json_str(entry1, compact=True) + "\n" + dump_json_str(entry2, compact=True)
        history_file.write_text(content, encoding="utf-8")

        history = OptimizationHistory.for_output_dir(tmp_path)
        backend: BackendName = "xgboost"
        dataset: DatasetName = "taiwan"
        preset: FeaturePreset = "full"
        result: UnifiedHistoryEntry | None = history.get_previous_best(backend, dataset, preset)

        assert result == entry2

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        """Test get_previous_best returns None when no matching entries."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        backend: BackendName = "xgboost"
        dataset: DatasetName = "taiwan"
        preset: FeaturePreset = "full"
        result = history.get_previous_best(backend, dataset, preset)
        assert result is None

    def test_calls_load_if_not_loaded(self, tmp_path: Path) -> None:
        """Test get_previous_best calls load if not yet loaded."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        assert history._loaded is False
        backend: BackendName = "xgboost"
        dataset: DatasetName = "taiwan"
        preset: FeaturePreset = "full"
        history.get_previous_best(backend, dataset, preset)
        assert history._loaded is True


class TestOptimizationHistoryGetAllTimeBest:
    """Tests for OptimizationHistory.get_all_time_best method."""

    def test_returns_highest_auc(self, tmp_path: Path) -> None:
        """Test get_all_time_best returns entry with highest AUC."""
        history_file = tmp_path / HISTORY_FILENAME
        entry1 = _make_history_entry(best_val_auc=0.80, timestamp="2024-01-01T00:00:00Z")
        entry2 = _make_history_entry(best_val_auc=0.90, timestamp="2024-01-02T00:00:00Z")
        entry3 = _make_history_entry(best_val_auc=0.85, timestamp="2024-01-03T00:00:00Z")
        content = (
            dump_json_str(entry1, compact=True)
            + "\n"
            + dump_json_str(entry2, compact=True)
            + "\n"
            + dump_json_str(entry3, compact=True)
        )
        history_file.write_text(content, encoding="utf-8")

        history = OptimizationHistory.for_output_dir(tmp_path)
        backend: BackendName = "xgboost"
        dataset: DatasetName = "taiwan"
        preset: FeaturePreset = "full"
        result: UnifiedHistoryEntry | None = history.get_all_time_best(backend, dataset, preset)

        assert result == entry2

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        """Test get_all_time_best returns None when no matching entries."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        backend: BackendName = "xgboost"
        dataset: DatasetName = "taiwan"
        preset: FeaturePreset = "full"
        result = history.get_all_time_best(backend, dataset, preset)
        assert result is None

    def test_calls_load_if_not_loaded(self, tmp_path: Path) -> None:
        """Test get_all_time_best calls load if not yet loaded."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        assert history._loaded is False
        backend: BackendName = "xgboost"
        dataset: DatasetName = "taiwan"
        preset: FeaturePreset = "full"
        history.get_all_time_best(backend, dataset, preset)
        assert history._loaded is True


class TestOptimizationHistoryGetAllEntries:
    """Tests for OptimizationHistory.get_all_entries method."""

    def test_returns_copy(self, tmp_path: Path) -> None:
        """Test get_all_entries returns a copy of entries."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        entry = _make_history_entry()
        history.append(entry)

        entries1 = history.get_all_entries()
        entries2 = history.get_all_entries()

        assert entries1 is not entries2
        assert len(entries1) == 1

    def test_calls_load_if_not_loaded(self, tmp_path: Path) -> None:
        """Test get_all_entries calls load if not yet loaded."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        assert history._loaded is False
        history.get_all_entries()
        assert history._loaded is True


class TestOptimizationHistoryGetEntriesForBackend:
    """Tests for OptimizationHistory.get_entries_for_backend method."""

    def test_filters_by_backend(self, tmp_path: Path) -> None:
        """Test get_entries_for_backend returns only matching backend entries."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        history.append(_make_history_entry(backend="xgboost"))
        history.append(_make_history_entry(backend="mlp"))

        xgboost_entries = history.get_entries_for_backend("xgboost")
        mlp_entries = history.get_entries_for_backend("mlp")

        assert len(xgboost_entries) == 1
        assert len(mlp_entries) == 1
        assert xgboost_entries[0]["backend"] == "xgboost"
        assert mlp_entries[0]["backend"] == "mlp"

    def test_calls_load_if_not_loaded(self, tmp_path: Path) -> None:
        """Test get_entries_for_backend calls load if not yet loaded."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        assert history._loaded is False
        history.get_entries_for_backend("xgboost")
        assert history._loaded is True


class TestOptimizationHistoryGetEntriesForDataset:
    """Tests for OptimizationHistory.get_entries_for_dataset method."""

    def test_filters_by_dataset(self, tmp_path: Path) -> None:
        """Test get_entries_for_dataset returns only matching dataset."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        history.append(_make_history_entry(dataset="taiwan"))
        history.append(_make_history_entry(dataset="us"))
        history.append(_make_history_entry(dataset="taiwan"))

        taiwan_entries = history.get_entries_for_dataset("taiwan")
        us_entries = history.get_entries_for_dataset("us")

        assert len(taiwan_entries) == 2
        assert len(us_entries) == 1

    def test_calls_load_if_not_loaded(self, tmp_path: Path) -> None:
        """Test get_entries_for_dataset calls load if not yet loaded."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        assert history._loaded is False
        history.get_entries_for_dataset("taiwan")
        assert history._loaded is True


class TestOptimizationHistoryGetProgression:
    """Tests for OptimizationHistory.get_progression method."""

    def test_filters_by_backend_dataset_and_preset(self, tmp_path: Path) -> None:
        """Test get_progression returns only matching backend/dataset/preset."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        history.append(_make_history_entry(dataset="taiwan", feature_preset="full"))
        history.append(_make_history_entry(dataset="taiwan", feature_preset="none"))
        history.append(_make_history_entry(dataset="us", feature_preset="full"))

        backend: BackendName = "xgboost"
        dataset: DatasetName = "taiwan"
        preset: FeaturePreset = "full"
        progression = history.get_progression(backend, dataset, preset)

        assert len(progression) == 1
        assert progression[0]["dataset"] == "taiwan"
        assert progression[0]["feature_preset"] == "full"

    def test_calls_load_if_not_loaded(self, tmp_path: Path) -> None:
        """Test get_progression calls load if not yet loaded."""
        history = OptimizationHistory.for_output_dir(tmp_path)
        assert history._loaded is False
        backend: BackendName = "xgboost"
        dataset: DatasetName = "taiwan"
        preset: FeaturePreset = "full"
        history.get_progression(backend, dataset, preset)
        assert history._loaded is True
