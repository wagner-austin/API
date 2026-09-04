"""Turn an optimisation run into the shape every experiment in this
workspace emits.

WHAT WAS MISSING. ``optimization_history.jsonl`` has carried a
``RunFingerprint`` since 2026-08-28 -- host, packages and image digest --
but it carried it INSIDE :class:`~scripts.optimize.history.UnifiedHistoryEntry`,
a shape private to this package. :func:`platform_core.run_record.compare_run_records`
cannot read it, so a ClearGBM sweep's AUC could not be placed beside any
other experiment's number even though the provenance to justify the
subtraction was sitting right there. ``monorepo_guards``' ``run-record``
rule names exactly this: "a fingerprint inside a private shape cannot be
read by compare_run_records".

WHY BOTH FILES ARE WRITTEN, and neither is derived from the other at read
time. The history JSONL is what the optimiser's own progression display
reads -- previous best, all-time best, per-backend filtering -- and it holds
the search's shape (trial counts, dataset dimensions). The record JSONL
holds the CLAIM, in the vocabulary the comparability layer already checks.
This mirrors :mod:`covenant_ml.benchmarking.provenance`, where the manifest
holds the per-seed detail and the record holds the claim.

WHAT THIS DOES NOT FIX. ``scripts/optimize`` still pins nothing, so every
fingerprint it captures reports its determinism stack as unpinned and says
so honestly. A record that is comparable is not the same as a run that is
reproducible; see ``docs/RESEARCH.md``'s ``cleargbm`` entry, which states
that gap and still states it after this module.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

from platform_core.json_utils import dump_json_str
from platform_core.run_record import (
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
)

from scripts.optimize.history import UnifiedHistoryEntry

#: What these runs ARE, for pairing two of their records.
#:
#: Stable across invocations and independent of the backend, the dataset and
#: the feature preset -- all three vary between runs that are still the same
#: experiment, and all three belong in the label instead. The backend
#: especially: which model wins is the question this experiment exists to
#: answer, so two backends' records MUST pair rather than be held apart by
#: their experiment names.
OPTIMIZATION_EXPERIMENT: Final[str] = "covenant-radar-hyperparameter-search"

#: What the record file is called, beside the history it describes.
#:
#: ``platform_core.run_record.run_record_sidecar`` is deliberately not used
#: here. It appends one fixed suffix to name ONE record for one result file,
#: which is right for a benchmark that writes a manifest per run. This
#: history is a JSONL that grows by one line per run, so one sidecar for the
#: whole file would name a single record for hundreds of runs.
RECORD_SUFFIX: Final[str] = ".runrecords.jsonl"


def optimization_record_path(history_path: Path) -> Path:
    """Name the record file that belongs beside a history file.

    The history's own stem is reused rather than a constant, so the
    ``HPC3_JOB_NAME`` suffix that
    :meth:`~scripts.optimize.history.OptimizationHistory.for_output_dir`
    applies under the farm carries over. Concurrent sweep members write to
    separate history files precisely because BeeGFS does not guarantee
    cross-node append atomicity; records written beside them must split the
    same way or they reintroduce the shared writer the suffix removed.

    Args:
        history_path: The history JSONL these records describe.

    Returns:
        The record path, in the same directory.
    """
    return history_path.with_name(history_path.stem + RECORD_SUFFIX)


def optimization_label(entry: UnifiedHistoryEntry) -> str:
    """Name which run within the experiment this is.

    Args:
        entry: The completed run's history entry.

    Returns:
        The backend, dataset, feature preset and timestamp, joined. The
        first three are what differ between two runs that are still meant to
        be compared; the timestamp is what makes a re-run of the same three
        a distinct run rather than a collision, which matters here because
        this experiment's whole purpose is progression over time.
    """
    return f"{entry['backend']}-{entry['dataset']}-{entry['feature_preset']}-{entry['timestamp']}"


def optimization_observations(entry: UnifiedHistoryEntry) -> tuple[Observation, ...]:
    """Name every number this run exists to produce.

    ``best_trial_number`` is deliberately absent. It is an index INTO the
    search rather than a measurement OF it -- subtracting one run's best
    trial index from another's is not a quantity -- so it goes into the
    payload digest instead, where it still distinguishes two runs without
    pretending to be comparable.

    Args:
        entry: The completed run's history entry.

    Returns:
        The claim (``best_val_auc``), the wall clock, and how many trials
        actually finished. The last is a measurement and not a knob: it is
        the optimiser's completed-trial count, which falls below the
        requested count whenever trials prune or fail.
    """
    return (
        Observation(name="best_val_auc", value=entry["best_val_auc"]),
        Observation(name="duration_seconds", value=entry["duration_seconds"]),
        Observation(name="trials_completed", value=float(entry["n_trials"])),
    )


def optimization_payload_digest(entry: UnifiedHistoryEntry) -> str:
    """Digest the run detail the comparability layer should not have to read.

    The dataset's dimensions and the winning trial's index say a great deal
    about whether two runs are the same run, and nothing that anyone
    subtracts. Digesting them lets two records be checked for identity
    without this layer understanding what a feature or a trial is.

    Digested over the canonical JSON encoding rather than any file, so a
    record built in memory carries the same digest as one built from a row
    read back off disk.

    Args:
        entry: The completed run's history entry.

    Returns:
        The hex digest.
    """
    return sha256(
        dump_json_str(
            {
                "n_samples": entry["n_samples"],
                "n_features": entry["n_features"],
                "best_trial_number": entry["best_trial_number"],
            }
        ).encode("utf-8")
    ).hexdigest()


def optimization_run_record(entry: UnifiedHistoryEntry) -> RunRecord:
    """Build the record for one completed optimisation run.

    Args:
        entry: The completed run's history entry, carrying the fingerprint
            captured when the run finished.

    Returns:
        The record, its observations in canonical order.

    Raises:
        ValueError: When the entry states it has no fingerprint. A record
            REQUIRES one -- that is the whole basis on which two of them may
            be subtracted -- and the 3,068 rows written before 2026-08-28
            genuinely have none. Refusing is the only honest answer: a
            synthesised fingerprint would claim a configuration nobody
            observed, and defaulting to an empty one would make those rows
            compare equal to each other and to nothing real.
    """
    fingerprint = entry["fingerprint"]
    if fingerprint is None:
        raise ValueError(
            f"history entry {optimization_label(entry)!r} states no fingerprint; "
            f"a RunRecord cannot be built for a run whose configuration was "
            f"never captured"
        )
    return run_record(
        experiment=OPTIMIZATION_EXPERIMENT,
        label=optimization_label(entry),
        fingerprint=fingerprint,
        observations=optimization_observations(entry),
        payload_digest=optimization_payload_digest(entry),
    )


def append_optimization_record(history_path: Path, entry: UnifiedHistoryEntry) -> None:
    """Append one run's record to the record file beside its history.

    Args:
        history_path: The history JSONL the record belongs beside.
        entry: The completed run's history entry. Must carry a fingerprint.

    Raises:
        ValueError: When the entry states no fingerprint, from
            :func:`optimization_run_record`.
    """
    record = optimization_run_record(entry)
    line = dump_json_str(encode_run_record(record), compact=True)
    with optimization_record_path(history_path).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


__all__ = [
    "OPTIMIZATION_EXPERIMENT",
    "RECORD_SUFFIX",
    "append_optimization_record",
    "optimization_label",
    "optimization_observations",
    "optimization_payload_digest",
    "optimization_record_path",
    "optimization_run_record",
]
