"""Optimization run history tracking with backend-aware progression comparison.

Stores each optimization run in a JSONL file for tracking AUC progression
over time. Supports all backends with a unified history entry that stores
best parameters alongside common fields.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from covenant_ml.types import BackendName
from platform_core.comparability import (
    RunFingerprint,
    decode_run_fingerprint,
    encode_run_fingerprint,
)
from platform_core.config import _test_hooks as config_env
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

from scripts._test_hooks import UnifiedOptimizationResult
from scripts.optimize.cli import DatasetName, FeaturePreset

logger = get_logger(__name__)

# History file name
HISTORY_FILENAME = "optimization_history.jsonl"


# =============================================================================
# Unified History Entry
# =============================================================================


class UnifiedHistoryEntry(TypedDict, total=True):
    """Unified optimization run history entry for any backend.

    Stores common fields plus the best AUC and best trial number. It carries
    no hyperparameters: the sentence claiming they were held as flat ``best_*``
    fields described a shape this entry does not have, and outlived it.
    """

    timestamp: str
    backend: str
    dataset: str
    feature_preset: str
    n_trials: int
    n_samples: int
    n_features: int
    best_val_auc: float
    best_trial_number: int
    duration_seconds: float
    fingerprint: RunFingerprint | None


# =============================================================================
# Decoder Functions
# =============================================================================


def _decode_backend(obj: JSONObject) -> BackendName:
    """Decode backend name from JSON object.

    Args:
        obj: JSON object with backend field.

    Returns:
        Validated backend name.

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
    if backend == "cleargbm":
        return "cleargbm"
    if backend == "logreg":
        return "logreg"
    if backend == "random_forest":
        return "random_forest"
    raise ValueError(f"Invalid backend: {backend}")


def _decode_history_entry(obj: JSONObject) -> UnifiedHistoryEntry:
    """Decode history entry from JSON object.

    Args:
        obj: JSON object with history fields.

    Returns:
        Unified history entry TypedDict.
    """
    return UnifiedHistoryEntry(
        timestamp=require_str(obj, "timestamp"),
        backend=require_str(obj, "backend"),
        dataset=require_str(obj, "dataset"),
        feature_preset=require_str(obj, "feature_preset"),
        n_trials=require_int(obj, "n_trials"),
        n_samples=require_int(obj, "n_samples"),
        n_features=require_int(obj, "n_features"),
        best_val_auc=require_float(obj, "best_val_auc"),
        best_trial_number=require_int(obj, "best_trial_number"),
        duration_seconds=require_float(obj, "duration_seconds"),
        fingerprint=_require_fingerprint_or_null(obj),
    )


def _require_fingerprint_or_null(obj: JSONObject) -> RunFingerprint | None:
    """Read the configuration a row's numbers were produced under.

    Three states, and collapsing any two would make the row assert something
    it does not know:

    * a fingerprint -- this run's configuration was captured;
    * an explicit ``null`` -- this row predates the field. 3,068 rows written
      before 2026-08-28 are in this state, and they are not claiming the
      configuration was unremarkable; they are saying nobody recorded it;
    * a MISSING key, which is neither and raises. A row that simply omits it
      cannot be told from one where the writer forgot, and the whole point of
      the field is that the reader can tell.

    Args:
        obj: The decoded JSON object for one row.

    Returns:
        The fingerprint, or None when the row states it has none.

    Raises:
        JSONTypeError: If the key is absent.
    """
    if "fingerprint" not in obj:
        raise JSONTypeError(
            "Field 'fingerprint' is required. Write null to state that the row predates "
            "it -- an absent key cannot be told from a writer that forgot."
        )
    value = obj["fingerprint"]
    if value is None:
        return None
    return decode_run_fingerprint(value)


def _encode_history_entry(entry: UnifiedHistoryEntry) -> JSONObject:
    """Encode one row for the JSONL file.

    Exists because the row used to be dumped as the TypedDict itself, which
    worked only while every field was a JSON scalar. A ``RunFingerprint``
    carries tuples -- the determinism settings are sorted pairs and the
    package versions are a tuple of records -- and ``json`` would render
    those as bare arrays that ``decode_run_fingerprint`` does not accept.
    Encoding through the contract's own encoder keeps write and read
    symmetrical.

    Args:
        entry: The row to encode.

    Returns:
        A JSON object the decoder above accepts.
    """
    fingerprint = entry["fingerprint"]
    return {
        "timestamp": entry["timestamp"],
        "backend": entry["backend"],
        "dataset": entry["dataset"],
        "feature_preset": entry["feature_preset"],
        "n_trials": entry["n_trials"],
        "n_samples": entry["n_samples"],
        "n_features": entry["n_features"],
        "best_val_auc": entry["best_val_auc"],
        "best_trial_number": entry["best_trial_number"],
        "duration_seconds": entry["duration_seconds"],
        "fingerprint": None if fingerprint is None else encode_run_fingerprint(fingerprint),
    }


# =============================================================================
# Result to History Entry Converter
# =============================================================================


def result_to_entry(
    result: UnifiedOptimizationResult,
    elapsed: float,
    fingerprint: RunFingerprint | None,
) -> UnifiedHistoryEntry:
    """Convert optimization result to history entry.

    Args:
        result: Unified optimization result.
        elapsed: Elapsed time in seconds.
        fingerprint: The configuration these numbers were produced under, or
            None only where a caller genuinely has none. Required as a
            parameter rather than captured here, so this stays a pure
            conversion and the capture happens once per run at the entry
            point that knows what it pinned.

    Returns:
        History entry with current UTC timestamp.
    """
    return UnifiedHistoryEntry(
        timestamp=datetime.now(UTC).isoformat(),
        backend=result["backend"],
        dataset=result["dataset"],
        feature_preset=result["feature_preset"],
        n_trials=result["n_trials_complete"],
        n_samples=result["n_samples"],
        n_features=result["n_features"],
        best_val_auc=result["best_value"],
        best_trial_number=result["best_trial_number"],
        duration_seconds=elapsed,
        fingerprint=fingerprint,
    )


# =============================================================================
# History Manager
# =============================================================================


class OptimizationHistory:
    """Manager for optimization run history.

    Tracks optimization runs in a JSONL file for progression comparison.
    Each line in the file is a separate JSON object representing one run.
    """

    def __init__(self, history_path: Path) -> None:
        """Initialize history manager.

        Args:
            history_path: Path to the JSONL history file.
        """
        self._path = history_path
        self._entries: list[UnifiedHistoryEntry] = []
        self._loaded = False

    @property
    def path(self) -> Path:
        """The JSONL this history reads and appends to.

        Exposed because :mod:`scripts.optimize.run_records` writes the
        workspace ``RunRecord`` for each run BESIDE this file, and under the
        HPC3 farm the name carries a job-name suffix that only
        :meth:`for_output_dir` knows how to build. A caller that guessed the
        name would write records into one shared file while the history it
        describes was correctly split per member.

        Returns:
            The history file path.
        """
        return self._path

    @classmethod
    def for_output_dir(cls, output_dir: Path) -> OptimizationHistory:
        """Create history manager for an output directory.

        Under the HPC3 farm (``HPC3_JOB_NAME`` exported by the batch
        script), the history file is suffixed with the job name so that
        concurrent sweep members — running on different nodes against
        one shared checkout — never append to the same file. BeeGFS
        does not guarantee cross-node append atomicity, so a shared
        JSONL under concurrent writers can interleave into corrupt
        lines. Locally the name is unchanged and history accumulates
        across runs as before.

        Args:
            output_dir: Directory where models and history are stored.

        Returns:
            New history manager instance.
        """
        job_name = config_env.get_env("HPC3_JOB_NAME")
        if job_name is not None and job_name != "":
            stem = Path(HISTORY_FILENAME).stem
            suffix = Path(HISTORY_FILENAME).suffix
            return cls(output_dir / f"{stem}-{job_name}{suffix}")
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
            entry: The history entry to append.
        """
        if not self._loaded:
            self.load()

        self._entries.append(entry)

        # Append to file
        line = dump_json_str(_encode_history_entry(entry), compact=True)
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
            backend: Backend name to filter by.
            dataset: Dataset name to filter by.
            feature_preset: Feature preset to filter by.

        Returns:
            Most recent matching entry, or None if no history.
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
            backend: Backend name to filter by.
            dataset: Dataset name to filter by.
            feature_preset: Feature preset to filter by.

        Returns:
            Entry with highest AUC, or None if no history.
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
            List of all history entries in chronological order.
        """
        if not self._loaded:
            self.load()
        return list(self._entries)

    def get_entries_for_backend(self, backend: BackendName) -> list[UnifiedHistoryEntry]:
        """Get all entries for a specific backend.

        Args:
            backend: Backend name to filter by.

        Returns:
            List of matching entries in chronological order.
        """
        if not self._loaded:
            self.load()
        return [e for e in self._entries if e["backend"] == backend]

    def get_entries_for_dataset(self, dataset: DatasetName) -> list[UnifiedHistoryEntry]:
        """Get all entries for a specific dataset.

        Args:
            dataset: Dataset name to filter by.

        Returns:
            List of matching entries in chronological order.
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
            backend: Backend name to filter by.
            dataset: Dataset name to filter by.
            feature_preset: Feature preset to filter by.

        Returns:
            List of matching entries in chronological order.
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
    "OptimizationHistory",
    "UnifiedHistoryEntry",
    "result_to_entry",
]
