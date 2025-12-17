"""Optimization run history tracking with backend-aware progression comparison.

Stores each optimization run in a JSONL file for tracking AUC progression
over time. Supports all backends (XGBoost, MLP, LightGBM, LSTM) with
backend-specific hyperparameter tracking.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from covenant_ml.types import BackendName
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

from covenant_radar_api.worker.optimize_job import (
    OptimizationResult as XGBoostOptimizationResult,
)
from covenant_radar_api.worker.optimize_lightgbm_job import LightGBMOptimizationResult
from covenant_radar_api.worker.optimize_lstm_job import LSTMOptimizationResult
from covenant_radar_api.worker.optimize_mlp_job import MLPOptimizationResult
from scripts.optimize.cli import DatasetName, FeaturePreset

logger = get_logger(__name__)

# History file name
HISTORY_FILENAME = "optimization_history.jsonl"


# =============================================================================
# Backend-Specific History Entry Types
# =============================================================================


class XGBoostHistoryEntry(TypedDict):
    """XGBoost optimization run history entry."""

    timestamp: str
    backend: BackendName
    dataset: str
    feature_preset: str
    n_trials: int
    n_samples: int
    n_features: int
    best_val_auc: float
    best_trial_number: int
    duration_seconds: float
    # XGBoost-specific hyperparameters
    best_max_depth: int
    best_n_estimators: int
    best_learning_rate: float
    best_reg_alpha: float
    best_reg_lambda: float
    best_subsample: float
    best_colsample_bytree: float


class MLPHistoryEntry(TypedDict):
    """MLP optimization run history entry."""

    timestamp: str
    backend: BackendName
    dataset: str
    feature_preset: str
    n_trials: int
    n_samples: int
    n_features: int
    best_val_auc: float
    best_trial_number: int
    duration_seconds: float
    # MLP-specific hyperparameters
    best_n_layers: int
    best_hidden_size: int
    best_learning_rate: float
    best_dropout: float
    best_batch_size: int


class LightGBMHistoryEntry(TypedDict):
    """LightGBM optimization run history entry."""

    timestamp: str
    backend: BackendName
    dataset: str
    feature_preset: str
    n_trials: int
    n_samples: int
    n_features: int
    best_val_auc: float
    best_trial_number: int
    duration_seconds: float
    # LightGBM-specific hyperparameters
    best_max_depth: int
    best_n_estimators: int
    best_num_leaves: int
    best_learning_rate: float
    best_reg_alpha: float
    best_reg_lambda: float
    best_subsample: float
    best_colsample_bytree: float


class LSTMHistoryEntry(TypedDict):
    """LSTM optimization run history entry."""

    timestamp: str
    backend: BackendName
    dataset: str
    feature_preset: str
    n_trials: int
    n_samples: int
    n_features: int
    best_val_auc: float
    best_trial_number: int
    duration_seconds: float
    # LSTM-specific hyperparameters
    best_hidden_size: int
    best_num_layers: int
    best_learning_rate: float
    best_dropout: float
    best_batch_size: int


# Union type for all history entries
UnifiedHistoryEntry = (
    XGBoostHistoryEntry | MLPHistoryEntry | LightGBMHistoryEntry | LSTMHistoryEntry
)


# =============================================================================
# Decoder Functions
# =============================================================================


def _decode_backend(obj: JSONObject) -> BackendName:
    """Decode backend name from JSON object.

    Args:
        obj (JSONObject): JSON object with backend field.

    Returns:
        BackendName: Validated backend name (xgboost, mlp, lightgbm, lstm).

    Raises:
        ValueError: If backend field contains invalid value.
    """
    backend = require_str(obj, "backend")
    if backend == "xgboost":
        return "xgboost"
    if backend == "mlp":
        return "mlp"
    if backend == "lightgbm":
        return "lightgbm"
    if backend == "lstm":
        return "lstm"
    raise ValueError(f"Invalid backend: {backend}")


def _decode_xgboost_entry(obj: JSONObject) -> XGBoostHistoryEntry:
    """Decode XGBoost history entry from JSON object.

    Args:
        obj (JSONObject): JSON object with XGBoost history fields.

    Returns:
        XGBoostHistoryEntry: Validated XGBoost history entry TypedDict.
    """
    return XGBoostHistoryEntry(
        timestamp=require_str(obj, "timestamp"),
        backend="xgboost",
        dataset=require_str(obj, "dataset"),
        feature_preset=require_str(obj, "feature_preset"),
        n_trials=require_int(obj, "n_trials"),
        n_samples=require_int(obj, "n_samples"),
        n_features=require_int(obj, "n_features"),
        best_val_auc=require_float(obj, "best_val_auc"),
        best_trial_number=require_int(obj, "best_trial_number"),
        duration_seconds=require_float(obj, "duration_seconds"),
        best_max_depth=require_int(obj, "best_max_depth"),
        best_n_estimators=require_int(obj, "best_n_estimators"),
        best_learning_rate=require_float(obj, "best_learning_rate"),
        best_reg_alpha=require_float(obj, "best_reg_alpha"),
        best_reg_lambda=require_float(obj, "best_reg_lambda"),
        best_subsample=require_float(obj, "best_subsample"),
        best_colsample_bytree=require_float(obj, "best_colsample_bytree"),
    )


def _decode_mlp_entry(obj: JSONObject) -> MLPHistoryEntry:
    """Decode MLP history entry from JSON object.

    Args:
        obj (JSONObject): JSON object with MLP history fields.

    Returns:
        MLPHistoryEntry: Validated MLP history entry TypedDict.
    """
    return MLPHistoryEntry(
        timestamp=require_str(obj, "timestamp"),
        backend="mlp",
        dataset=require_str(obj, "dataset"),
        feature_preset=require_str(obj, "feature_preset"),
        n_trials=require_int(obj, "n_trials"),
        n_samples=require_int(obj, "n_samples"),
        n_features=require_int(obj, "n_features"),
        best_val_auc=require_float(obj, "best_val_auc"),
        best_trial_number=require_int(obj, "best_trial_number"),
        duration_seconds=require_float(obj, "duration_seconds"),
        best_n_layers=require_int(obj, "best_n_layers"),
        best_hidden_size=require_int(obj, "best_hidden_size"),
        best_learning_rate=require_float(obj, "best_learning_rate"),
        best_dropout=require_float(obj, "best_dropout"),
        best_batch_size=require_int(obj, "best_batch_size"),
    )


def _decode_lightgbm_entry(obj: JSONObject) -> LightGBMHistoryEntry:
    """Decode LightGBM history entry from JSON object.

    Args:
        obj (JSONObject): JSON object with LightGBM history fields.

    Returns:
        LightGBMHistoryEntry: Validated LightGBM history entry TypedDict.
    """
    return LightGBMHistoryEntry(
        timestamp=require_str(obj, "timestamp"),
        backend="lightgbm",
        dataset=require_str(obj, "dataset"),
        feature_preset=require_str(obj, "feature_preset"),
        n_trials=require_int(obj, "n_trials"),
        n_samples=require_int(obj, "n_samples"),
        n_features=require_int(obj, "n_features"),
        best_val_auc=require_float(obj, "best_val_auc"),
        best_trial_number=require_int(obj, "best_trial_number"),
        duration_seconds=require_float(obj, "duration_seconds"),
        best_max_depth=require_int(obj, "best_max_depth"),
        best_n_estimators=require_int(obj, "best_n_estimators"),
        best_num_leaves=require_int(obj, "best_num_leaves"),
        best_learning_rate=require_float(obj, "best_learning_rate"),
        best_reg_alpha=require_float(obj, "best_reg_alpha"),
        best_reg_lambda=require_float(obj, "best_reg_lambda"),
        best_subsample=require_float(obj, "best_subsample"),
        best_colsample_bytree=require_float(obj, "best_colsample_bytree"),
    )


def _decode_lstm_entry(obj: JSONObject) -> LSTMHistoryEntry:
    """Decode LSTM history entry from JSON object.

    Args:
        obj (JSONObject): JSON object with LSTM history fields.

    Returns:
        LSTMHistoryEntry: Validated LSTM history entry TypedDict.
    """
    return LSTMHistoryEntry(
        timestamp=require_str(obj, "timestamp"),
        backend="lstm",
        dataset=require_str(obj, "dataset"),
        feature_preset=require_str(obj, "feature_preset"),
        n_trials=require_int(obj, "n_trials"),
        n_samples=require_int(obj, "n_samples"),
        n_features=require_int(obj, "n_features"),
        best_val_auc=require_float(obj, "best_val_auc"),
        best_trial_number=require_int(obj, "best_trial_number"),
        duration_seconds=require_float(obj, "duration_seconds"),
        best_hidden_size=require_int(obj, "best_hidden_size"),
        best_num_layers=require_int(obj, "best_num_layers"),
        best_learning_rate=require_float(obj, "best_learning_rate"),
        best_dropout=require_float(obj, "best_dropout"),
        best_batch_size=require_int(obj, "best_batch_size"),
    )


def _decode_history_entry(obj: JSONObject) -> UnifiedHistoryEntry:
    """Decode history entry from JSON object based on backend type.

    Args:
        obj (JSONObject): JSON object with history fields including backend discriminator.

    Returns:
        UnifiedHistoryEntry: Backend-specific history entry TypedDict.
    """
    backend = _decode_backend(obj)
    if backend == "xgboost":
        return _decode_xgboost_entry(obj)
    if backend == "mlp":
        return _decode_mlp_entry(obj)
    if backend == "lightgbm":
        return _decode_lightgbm_entry(obj)
    # backend must be "lstm" here - mypy validates exhaustiveness via return type
    return _decode_lstm_entry(obj)


# =============================================================================
# Result to History Entry Converters
# =============================================================================


def xgboost_result_to_entry(
    result: XGBoostOptimizationResult, elapsed: float
) -> XGBoostHistoryEntry:
    """Convert XGBoost optimization result to history entry.

    Args:
        result (XGBoostOptimizationResult): Typed XGBoost optimization result.
        elapsed (float): Elapsed time in seconds.

    Returns:
        XGBoostHistoryEntry: History entry with current UTC timestamp.
    """
    return XGBoostHistoryEntry(
        timestamp=datetime.now(UTC).isoformat(),
        backend="xgboost",
        dataset=result["dataset"],
        feature_preset=result["feature_preset"],
        n_trials=result["n_trials_complete"],
        n_samples=result["n_samples"],
        n_features=result["n_features"],
        best_val_auc=result["best_val_auc"],
        best_trial_number=result["best_trial_number"],
        duration_seconds=elapsed,
        best_max_depth=result["best_max_depth"],
        best_n_estimators=result["best_n_estimators"],
        best_learning_rate=result["best_learning_rate"],
        best_reg_alpha=result["best_reg_alpha"],
        best_reg_lambda=result["best_reg_lambda"],
        best_subsample=result["best_subsample"],
        best_colsample_bytree=result["best_colsample_bytree"],
    )


def mlp_result_to_entry(result: MLPOptimizationResult, elapsed: float) -> MLPHistoryEntry:
    """Convert MLP optimization result to history entry.

    Args:
        result (MLPOptimizationResult): Typed MLP optimization result.
        elapsed (float): Elapsed time in seconds.

    Returns:
        MLPHistoryEntry: History entry with current UTC timestamp.
    """
    return MLPHistoryEntry(
        timestamp=datetime.now(UTC).isoformat(),
        backend="mlp",
        dataset=result["dataset"],
        feature_preset=result["feature_preset"],
        n_trials=result["n_trials_complete"],
        n_samples=result["n_samples"],
        n_features=result["n_features"],
        best_val_auc=result["best_val_auc"],
        best_trial_number=result["best_trial_number"],
        duration_seconds=elapsed,
        best_n_layers=result["best_n_layers"],
        best_hidden_size=result["best_hidden_size"],
        best_learning_rate=result["best_learning_rate"],
        best_dropout=result["best_dropout"],
        best_batch_size=result["best_batch_size"],
    )


def lightgbm_result_to_entry(
    result: LightGBMOptimizationResult, elapsed: float
) -> LightGBMHistoryEntry:
    """Convert LightGBM optimization result to history entry.

    Args:
        result (LightGBMOptimizationResult): Typed LightGBM optimization result.
        elapsed (float): Elapsed time in seconds.

    Returns:
        LightGBMHistoryEntry: History entry with current UTC timestamp.
    """
    return LightGBMHistoryEntry(
        timestamp=datetime.now(UTC).isoformat(),
        backend="lightgbm",
        dataset=result["dataset"],
        feature_preset=result["feature_preset"],
        n_trials=result["n_trials_complete"],
        n_samples=result["n_samples"],
        n_features=result["n_features"],
        best_val_auc=result["best_val_auc"],
        best_trial_number=result["best_trial_number"],
        duration_seconds=elapsed,
        best_max_depth=result["best_max_depth"],
        best_n_estimators=result["best_n_estimators"],
        best_num_leaves=result["best_num_leaves"],
        best_learning_rate=result["best_learning_rate"],
        best_reg_alpha=result["best_reg_alpha"],
        best_reg_lambda=result["best_reg_lambda"],
        best_subsample=result["best_subsample"],
        best_colsample_bytree=result["best_colsample_bytree"],
    )


def lstm_result_to_entry(result: LSTMOptimizationResult, elapsed: float) -> LSTMHistoryEntry:
    """Convert LSTM optimization result to history entry.

    Args:
        result (LSTMOptimizationResult): Typed LSTM optimization result.
        elapsed (float): Elapsed time in seconds.

    Returns:
        LSTMHistoryEntry: History entry with current UTC timestamp.
    """
    return LSTMHistoryEntry(
        timestamp=datetime.now(UTC).isoformat(),
        backend="lstm",
        dataset=result["dataset"],
        feature_preset=result["feature_preset"],
        n_trials=result["n_trials_complete"],
        n_samples=result["n_samples"],
        n_features=result["n_features"],
        best_val_auc=result["best_val_auc"],
        best_trial_number=result["best_trial_number"],
        duration_seconds=elapsed,
        best_hidden_size=result["best_hidden_size"],
        best_num_layers=result["best_num_layers"],
        best_learning_rate=result["best_learning_rate"],
        best_dropout=result["best_dropout"],
        best_batch_size=result["best_batch_size"],
    )


# =============================================================================
# History Manager
# =============================================================================


class OptimizationHistory:
    """Manager for optimization run history.

    Tracks optimization runs in a JSONL file for progression comparison.
    Each line in the file is a separate JSON object representing one run.
    Supports all backends with backend-specific hyperparameter tracking.
    """

    def __init__(self, history_path: Path) -> None:
        """Initialize history manager.

        Args:
            history_path (Path): Path to the JSONL history file.
        """
        self._path = history_path
        self._entries: list[UnifiedHistoryEntry] = []
        self._loaded = False

    @classmethod
    def for_output_dir(cls, output_dir: Path) -> OptimizationHistory:
        """Create history manager for an output directory.

        Args:
            output_dir (Path): Directory where models and history are stored.

        Returns:
            OptimizationHistory: New history manager instance.
        """
        return cls(output_dir / HISTORY_FILENAME)

    def load(self) -> None:
        """Load history from file.

        Silently handles missing file (empty history).
        """
        if self._loaded:
            return

        self._entries = []

        if not self._path.exists():
            logger.debug("History file not found, starting fresh: %s", self._path)
            self._loaded = True
            return

        content = self._path.read_text(encoding="utf-8")
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            parsed = load_json_str(stripped)
            obj = narrow_json_to_dict(parsed)
            entry = _decode_history_entry(obj)
            self._entries.append(entry)

        logger.debug("Loaded %d history entries from %s", len(self._entries), self._path)
        self._loaded = True

    def append(self, entry: UnifiedHistoryEntry) -> None:
        """Append a new entry to history and persist to file.

        Args:
            entry (UnifiedHistoryEntry): The history entry to append.
        """
        if not self._loaded:
            self.load()

        self._entries.append(entry)

        # Append to file
        line = dump_json_str(entry, compact=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        logger.debug(
            "Appended history entry: backend=%s dataset=%s preset=%s auc=%.4f",
            entry["backend"],
            entry["dataset"],
            entry["feature_preset"],
            entry["best_val_auc"],
        )

    def get_previous_best(
        self,
        backend: BackendName,
        dataset: DatasetName,
        feature_preset: FeaturePreset,
    ) -> UnifiedHistoryEntry | None:
        """Get the most recent entry for a backend/dataset/preset combination.

        Args:
            backend (BackendName): Backend name to filter by.
            dataset (DatasetName): Dataset name to filter by.
            feature_preset (FeaturePreset): Feature preset to filter by.

        Returns:
            UnifiedHistoryEntry | None: Most recent matching entry, or None if no history.
        """
        if not self._loaded:
            self.load()

        matching = [
            e
            for e in self._entries
            if e["backend"] == backend
            and e["dataset"] == dataset
            and e["feature_preset"] == feature_preset
        ]

        if not matching:
            return None

        # Return the last one (most recent)
        return matching[-1]

    def get_all_time_best(
        self,
        backend: BackendName,
        dataset: DatasetName,
        feature_preset: FeaturePreset,
    ) -> UnifiedHistoryEntry | None:
        """Get the all-time best entry for a backend/dataset/preset combination.

        Args:
            backend (BackendName): Backend name to filter by.
            dataset (DatasetName): Dataset name to filter by.
            feature_preset (FeaturePreset): Feature preset to filter by.

        Returns:
            UnifiedHistoryEntry | None: Entry with highest AUC, or None if no history.
        """
        if not self._loaded:
            self.load()

        matching = [
            e
            for e in self._entries
            if e["backend"] == backend
            and e["dataset"] == dataset
            and e["feature_preset"] == feature_preset
        ]

        if not matching:
            return None

        # Find entry with max AUC
        best = matching[0]
        for entry in matching[1:]:
            if entry["best_val_auc"] > best["best_val_auc"]:
                best = entry

        return best

    def get_all_entries(self) -> list[UnifiedHistoryEntry]:
        """Get all history entries.

        Returns:
            list[UnifiedHistoryEntry]: List of all history entries in chronological order.
        """
        if not self._loaded:
            self.load()
        return list(self._entries)

    def get_entries_for_backend(self, backend: BackendName) -> list[UnifiedHistoryEntry]:
        """Get all entries for a specific backend.

        Args:
            backend (BackendName): Backend name to filter by.

        Returns:
            list[UnifiedHistoryEntry]: List of matching entries in chronological order.
        """
        if not self._loaded:
            self.load()
        return [e for e in self._entries if e["backend"] == backend]

    def get_entries_for_dataset(self, dataset: DatasetName) -> list[UnifiedHistoryEntry]:
        """Get all entries for a specific dataset.

        Args:
            dataset (DatasetName): Dataset name to filter by.

        Returns:
            list[UnifiedHistoryEntry]: List of matching entries in chronological order.
        """
        if not self._loaded:
            self.load()
        return [e for e in self._entries if e["dataset"] == dataset]

    def get_progression(
        self,
        backend: BackendName,
        dataset: DatasetName,
        feature_preset: FeaturePreset,
    ) -> list[UnifiedHistoryEntry]:
        """Get the progression of runs for a backend/dataset/preset combination.

        Args:
            backend (BackendName): Backend name to filter by.
            dataset (DatasetName): Dataset name to filter by.
            feature_preset (FeaturePreset): Feature preset to filter by.

        Returns:
            list[UnifiedHistoryEntry]: List of matching entries in chronological order.
        """
        if not self._loaded:
            self.load()
        return [
            e
            for e in self._entries
            if e["backend"] == backend
            and e["dataset"] == dataset
            and e["feature_preset"] == feature_preset
        ]


__all__ = [
    "HISTORY_FILENAME",
    "LSTMHistoryEntry",
    "LightGBMHistoryEntry",
    "MLPHistoryEntry",
    "OptimizationHistory",
    "UnifiedHistoryEntry",
    "XGBoostHistoryEntry",
    "lightgbm_result_to_entry",
    "lstm_result_to_entry",
    "mlp_result_to_entry",
    "xgboost_result_to_entry",
]
