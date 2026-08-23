"""Appending and reading the local record of submitted jobs.

Writes are append-only and flushed per record. A sweep that dies on member
four leaves three complete lines behind, which is the whole point: the three
jobs it already started are running whether or not the process survived to
mention them.

Reading tolerates nothing. A malformed line fails the read rather than being
skipped, because the failure mode this file exists to prevent is a job that
nobody knows about, and silently dropping an unreadable record is exactly
that failure wearing a different hat.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from platform_core.json_utils import dump_json_str, load_json_str

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.ledger import LedgerEntry, decode_ledger_entry, encode_ledger_entry
from hpc3.core import _test_hooks


def append(path: pathlib.Path, entry: LedgerEntry) -> None:
    """Append one entry to the ledger.

    Args:
        path: Ledger file. Created, with its parent directories, if absent.
        entry: Entry to record.
    """
    line = dump_json_str(encode_ledger_entry(entry)) + "\n"
    _test_hooks.append_text(path, line)


def read(path: pathlib.Path, cluster: ClusterFacts) -> list[LedgerEntry]:
    """Read every entry from the ledger.

    Args:
        path: Ledger file.
        cluster: The cluster the workspace selected. Every recorded partition
            is checked against it, so a ledger written for one machine and
            read against another fails loudly rather than reporting every job
            as unaccounted.

    Returns:
        Every recorded entry, oldest first. An absent ledger reads as empty:
        nothing has been submitted from this machine yet, which is a real
        state and not an error.

    Raises:
        JSONTypeError: If a line is not a valid entry. Skipping it would
            hide a job, which is the one outcome this file exists to prevent.
        InvalidJsonError: If a line is not valid JSON.
        AppError: With ``PARTITION_UNKNOWN`` if a record names a partition
            this cluster does not have.
    """
    if not _test_hooks.file_exists(path):
        return []
    text = _test_hooks.read_bytes(path).decode("utf-8")
    return [
        decode_ledger_entry(load_json_str(line), cluster)
        for line in text.splitlines()
        if line.strip() != ""
    ]


def unfinished(entries: Sequence[LedgerEntry], finished_ids: Sequence[str]) -> list[LedgerEntry]:
    """List recorded jobs that accounting has not reported as finished.

    Args:
        entries: Everything the ledger holds.
        finished_ids: Ids accounting reports in a terminal state.

    Returns:
        The entries whose ids are not among the finished ones, oldest first.
        These are the candidates for "submitted, and then what?" -- either
        still running, or lost track of.
    """
    done = set(finished_ids)
    return [entry for entry in entries if entry["job_id"] not in done]


__all__ = ["append", "read", "unfinished"]
