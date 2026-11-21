"""The record of what each retire freed on its node (MCPs board task 1526ece1).

One JSON object per line beside the retire record, appended and never
rewritten (:mod:`fleet.core.records`), one line for each retire a node
answered and did: its run, its node and the bytes its enclosure held when the
retire removed it (:mod:`fleet.core.retire_script`). A retire a process kept
whole, a node did not answer or a node refused frees nothing and writes
nothing here. It is the program's own measurement of the space retires give
back on each node, which ``python -m fleet.cli.freed_report`` totals.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, require_int, require_str
from typing_extensions import TypedDict


class FreedRecord(TypedDict):
    """The bytes one retire freed.

    Attributes:
        run_id: The dispatch whose enclosure it removed.
        node: The node's workspace name.
        at_unix: When the retire was recorded, whole seconds since the epoch.
        freed_bytes: The apparent sizes of every entry in the enclosure
            that is not a directory, summed before its removal
            (:mod:`fleet.core.retire_script`); 0 for an enclosure an earlier
            attempt already removed.
    """

    run_id: str
    node: str
    at_unix: int
    freed_bytes: int


def encode_freed_record(record: FreedRecord) -> JSONObject:
    """Encode one freed line.

    Args:
        record: The line to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "run_id": record["run_id"],
        "node": record["node"],
        "at_unix": record["at_unix"],
        "freed_bytes": record["freed_bytes"],
    }


def decode_freed_record(value: JSONValue) -> FreedRecord:
    """Decode and validate one freed line.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated line.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or the byte count is negative.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"freed record must be a JSON object, got {type(value).__name__}")
    freed = require_int(value, "freed_bytes")
    if freed < 0:
        raise JSONTypeError(f"freed record's freed_bytes must not be negative, got {freed}")
    return FreedRecord(
        run_id=require_str(value, "run_id"),
        node=require_str(value, "node"),
        at_unix=require_int(value, "at_unix"),
        freed_bytes=freed,
    )


__all__ = ["FreedRecord", "decode_freed_record", "encode_freed_record"]
