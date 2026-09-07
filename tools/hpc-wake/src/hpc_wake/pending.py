"""Endings observed but not yet announced, held across cycles.

WHY THIS FILE EXISTS AND THE LEDGER'S CLOSURE RECORD DOES NOT SUFFICE. The
closure record answers "has this job been dealt with"; it is append-only and
grows forever, which is right for a permanent fact. This record answers "is
this ending still waiting for its group to settle", which is a temporary
state that SHRINKS -- every record here is destined to leave. Those are
different lifetimes, and ``hpc3.core.ledger`` opens with the sentence
"Writes are append-only", so a rewriting store does not belong in it.

Everything else is lifted rather than forked. The record's payload is
``hpc3.contracts.closure.Closure`` with its existing encoder and decoder;
the serialisation is ``platform_core.json_utils``; the file I/O is the same
``read_bytes`` / ``write_text`` / ``file_exists`` seam in
``hpc3.core._test_hooks`` that ``hpc3.core.ledger`` itself uses. This module
introduces no seam of its own, and ``tests/test_architecture.py`` asserts
that it never does -- a second clock or a second file seam would be two
sources of truth for one question.

Reading tolerates nothing, for the reason the ledger's own reader gives: a
skipped malformed line is an ending that will never be announced, and
silence is the exact failure this bridge exists to remove.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from hpc3.contracts.closure import Closure, decode_closure, encode_closure
from hpc3.core import _test_hooks as hpc3_hooks
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    require_dict,
    require_int,
)
from typing_extensions import TypedDict


class PendingClosure(TypedDict):
    """One ending waiting for its group to settle.

    Attributes:
        closure: The ending itself, exactly as it will be announced and
            later written to the closure record. Carried whole rather than
            reduced to an id so that a cycle which announces it needs no
            second lookup, and so the record survives a ledger the operator
            has since rotated.
        observed_epoch: Unix seconds at which THIS BRIDGE first saw the job
            terminal. Deliberately not ``closure["closed_at"]``, which is
            the same instant in ISO-8601: the settling policy is integer
            arithmetic, and re-parsing a timestamp on every cycle to redo
            arithmetic we could have stored is both slower and a new failure
            mode (a hand-edited file yielding an unparseable string). The
            two are written from one cycle and never compared to each other.
    """

    closure: Closure
    observed_epoch: int


def encode_pending(record: PendingClosure) -> dict[str, JSONValue]:
    """Encode a pending record to a JSON object.

    Args:
        record: Record to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "closure": encode_closure(record["closure"]),
        "observed_epoch": record["observed_epoch"],
    }


def _require_observed_epoch(obj: JSONObject) -> int:
    """Read the observation time, which must be a non-negative integer.

    Args:
        obj: Object being decoded.

    Returns:
        Unix seconds at which the ending was first seen.

    Raises:
        JSONTypeError: If the key is absent, is not an integer, or is
            negative. A negative epoch would make every ripeness comparison
            in :mod:`hpc_wake.settling` answer "aged" immediately, which
            announces a group the instant it is created -- the defect this
            package was changed to remove, wearing the costume of a fix.
    """
    value = require_int(obj, "observed_epoch")
    if value < 0:
        raise JSONTypeError(f"Field 'observed_epoch' must not be negative, got {value}")
    return value


def decode_pending(value: JSONValue) -> PendingClosure:
    """Decode and validate a JSON value into a pending record.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated record.

    Raises:
        JSONTypeError: If the value is not an object, the closure is missing
            or malformed, or the observation time is absent, non-integer or
            negative.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"pending record must be a JSON object, got {type(value).__name__}")
    return PendingClosure(
        closure=decode_closure(require_dict(value, "closure")),
        observed_epoch=_require_observed_epoch(value),
    )


def pending_path(ledger: pathlib.Path) -> pathlib.Path:
    """Locate the pending record belonging to a ledger.

    Derived rather than configured, for the reason
    :func:`hpc3.core.ledger.closure_path` gives: files that must describe
    the same set of jobs should not be separately addressable.

    Args:
        ledger: The ledger file.

    Returns:
        A sibling file named after it.
    """
    return ledger.with_name(ledger.name + ".pending")


def read_pending(path: pathlib.Path) -> list[PendingClosure]:
    """Read every record still waiting to be announced.

    Args:
        path: Pending file.

    Returns:
        Every record, in the order written. An absent file reads as empty:
        nothing is waiting, which is the ordinary state and not an error.

    Raises:
        JSONTypeError: If a line is not a valid record.
        InvalidJsonError: If a line is not valid JSON.
    """
    if not hpc3_hooks.file_exists(path):
        return []
    text = hpc3_hooks.read_bytes(path).decode("utf-8")
    return [decode_pending(load_json_str(line)) for line in text.splitlines() if line.strip() != ""]


def write_pending(path: pathlib.Path, records: Sequence[PendingClosure]) -> None:
    """Replace the pending record with exactly these entries.

    A REPLACEMENT, not an append, and that is the whole reason this store is
    not in ``hpc3.core.ledger``. Records leave this file when their group is
    announced, so the file's contents shrink; an append-only writer could
    only ever mark them, and a marker file that grows without bound is the
    thing the ledger's closure record is already allowed to be.

    Args:
        path: Pending file. Its parent is the ledger's own directory, which
            the caller has necessarily just read from, so it exists.
        records: The complete new contents. An empty sequence writes an
            empty file rather than deleting it -- an absent file and an
            empty one already read alike, and leaving the file in place
            keeps its existence a fact about this bridge having run rather
            than about what it found.
    """
    hpc3_hooks.write_text(path, "".join(dump_json_str(encode_pending(r)) + "\n" for r in records))


__all__ = [
    "PendingClosure",
    "decode_pending",
    "encode_pending",
    "pending_path",
    "read_pending",
    "write_pending",
]
