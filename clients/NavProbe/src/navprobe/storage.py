"""Persist and reload probe records.

Persistence exists because the fresh-process condition cannot be measured any
other way. Two rollouts inside one interpreter share module state, JIT caches,
and allocator history; proving that a rollout survives process restart requires
writing a record in one process and reading it in another.

Only the records that cross a process boundary get a path here:

* the **run record**, which is what a fresh-process comparison exchanges;
* the **trial record**, which is the result worth keeping;
* the **observation record**, which carries values rather than digests and
  exists for the one comparison that cannot be made live — two configurations
  that cannot share a process, such as two MuJoCo-Warp devices.

A comparison, dispersion or divergence record is derived from those in memory
and has a codec but no file, because giving a derived verdict its own file
invites it to drift from the runs it describes.

The representation is :mod:`navprobe.wireformat`. This module is only the
filesystem boundary, and every effect it has goes through
:mod:`navprobe._test_hooks`.
"""

from __future__ import annotations

from pathlib import Path

from navprobe import _test_hooks
from navprobe.codecs.observation import (
    decode_observation_record,
    encode_observation_record,
)
from navprobe.codecs.run import decode_run_record, encode_run_record
from navprobe.codecs.trial import decode_trial_record, encode_trial_record
from navprobe.records import ObservationRecord, RunRecord, TrialRecord


def _write_record_text(path: Path, text: str) -> None:
    """Create a destination's parents and write encoded text to it.

    Shared by every save so that the create-then-write ordering is expressed
    once. A second copy would be free to omit the parent creation, and the
    resulting failure appears only when a caller writes to a directory that
    does not exist yet.

    Args:
        path: Destination file. Missing parent directories are created.
        text: Encoded record text to write.

    Raises:
        OSError: When the destination cannot be created or written.
    """
    _test_hooks.make_parent_dirs(path)
    _test_hooks.write_text(path, text)


def save_run_record(path: Path, record: RunRecord) -> None:
    """Write a run record to disk.

    Args:
        path: Destination file. Missing parent directories are created.
        record: The record to write.

    Raises:
        OSError: When the destination cannot be created or written.
    """
    _write_record_text(path, encode_run_record(record))


def load_run_record(path: Path) -> RunRecord:
    """Read a run record from disk.

    Args:
        path: File to read.

    Returns:
        The decoded record.

    Raises:
        OSError: When the file cannot be read.
        WireFormatError: When the file does not carry a valid run record.
    """
    return decode_run_record(_test_hooks.read_text(path))


def save_trial_record(path: Path, record: TrialRecord) -> None:
    """Write a trial record to disk.

    Args:
        path: Destination file. Missing parent directories are created.
        record: The record to write.

    Raises:
        OSError: When the destination cannot be created or written.
    """
    _write_record_text(path, encode_trial_record(record))


def load_trial_record(path: Path) -> TrialRecord:
    """Read a trial record from disk.

    Args:
        path: File to read.

    Returns:
        The decoded record.

    Raises:
        OSError: When the file cannot be read.
        WireFormatError: When the file does not carry a valid trial record.
    """
    return decode_trial_record(_test_hooks.read_text(path))


def save_observation_record(path: Path, record: ObservationRecord) -> None:
    """Write an observation record to disk.

    Args:
        path: Destination file. Missing parent directories are created.
        record: The record to write.

    Raises:
        OSError: When the destination cannot be created or written.
    """
    _write_record_text(path, encode_observation_record(record))


def load_observation_record(path: Path) -> ObservationRecord:
    """Read an observation record from disk.

    Args:
        path: File to read.

    Returns:
        The decoded record.

    Raises:
        OSError: When the file cannot be read.
        WireFormatError: When the file does not carry a valid observation record.
    """
    return decode_observation_record(_test_hooks.read_text(path))


__all__ = [
    "load_observation_record",
    "load_run_record",
    "load_trial_record",
    "save_observation_record",
    "save_run_record",
    "save_trial_record",
]
