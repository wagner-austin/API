"""What this bridge has already announced, so it never announces it twice.

ONE APPEND-ONLY FILE BESIDE THE LEDGER, one line per dispatch announced. It is
the bridge's whole memory, and its shape is deliberately the same as every
other record in this workspace: a JSON object per line, appended and never
rewritten, read whole, and FATAL on a line that does not decode.

WHY A FILE RATHER THAN A CURSOR. The alternative is remembering a position in
the ledger -- a line number or a timestamp -- and it is wrong for the reason
``fleet.core.records`` documents: the ledger is append-only, so a finished
dispatch has BOTH a running row and a terminal row in the file, and rows for
different dispatches interleave. A position in that stream does not answer
"has this run been announced", which is the only question asked here. Naming
the run ids does.

WHY NOT SIMPLY RE-READ THE BOARD. Because a bridge that asked the board what
it had already posted would depend on the board being reachable to decide
whether to post, and would re-announce everything the first time a query
failed. The local record answers with no network at all.

A MALFORMED LINE IS FATAL, NOT SKIPPED, and the cost here is specific: a line
read as absent means the dispatch it names is announced AGAIN, and the reader
of that second post cannot tell it from a genuinely new ending. So the
decoder raises and names the line, exactly as the ledger's does.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from fleet_wake import _test_hooks


class AnnouncedRun(TypedDict):
    """One dispatch this bridge has posted about.

    Attributes:
        run_id: The dispatch's id, as the fleet ledger spells it. The only
            field the bridge reads; the rest exist so a person opening the
            file can see what happened without joining it to the ledger.
        outcome: The terminal outcome that was announced. Recorded because a
            run announced as ``lost`` and later collected as ``passed`` is a
            real sequence, and a reader needs to see which one was posted.
        announced_unix: When the post landed, whole seconds since the epoch.
    """

    run_id: str
    outcome: str
    announced_unix: int


def encode_announced_run(record: AnnouncedRun) -> JSONObject:
    """Encode one position record.

    Args:
        record: The record to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "run_id": record["run_id"],
        "outcome": record["outcome"],
        "announced_unix": record["announced_unix"],
    }


def decode_announced_run(value: JSONValue) -> AnnouncedRun:
    """Decode and validate one position record.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated record.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing
            or mistyped.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"position record must be a JSON object, got {type(value).__name__}")
    return AnnouncedRun(
        run_id=require_str(value, "run_id"),
        outcome=require_str(value, "outcome"),
        announced_unix=require_int(value, "announced_unix"),
    )


def position_path(ledger: pathlib.Path) -> pathlib.Path:
    """Where the position record lives for a given ledger.

    Beside the ledger rather than in a configured location, so a workspace
    carries its bridge's memory with it and moving one moves both.

    Args:
        ledger: The fleet ledger's path.

    Returns:
        The position file's path.
    """
    return ledger.parent / "announced.jsonl"


def read_announced(path: pathlib.Path) -> frozenset[str]:
    """Read every run id this bridge has already posted about.

    Args:
        path: The position file's path.

    Returns:
        The announced run ids. An absent file reads as empty rather than
        raising: a workspace whose bridge has never run has announced
        nothing, and refusing the first cycle for having no history would
        make the bridge impossible to start.

    Raises:
        InvalidJsonError: If a line is not valid JSON at all. Distinct from
            the below and raised by the parser before any narrowing happens.
        JSONTypeError: If a line is valid JSON but not a record -- not an
            object, or missing or mistyping a field. The not-an-object case
            names the line number, because that is the one a person fixes by
            opening the file. NOT skipped -- see the module docstring on what
            a skipped line costs.
    """
    if not _test_hooks.file_exists(path):
        return frozenset()
    announced: set[str] = set()
    for index, line in enumerate(_test_hooks.read_text(path).splitlines(), start=1):
        if line.strip() == "":
            continue
        value = load_json_str(line)
        if not isinstance(value, dict):
            raise JSONTypeError(
                f"{path} line {index} is a {type(value).__name__}, not an object; a position "
                "line that cannot be read means the dispatch it names is announced a second "
                "time, and nobody reading that post can tell it from a new ending"
            )
        announced.add(decode_announced_run(value)["run_id"])
    return frozenset(announced)


def append_announced(path: pathlib.Path, record: AnnouncedRun) -> None:
    """Append one position record.

    Args:
        path: The position file's path.
        record: The dispatch that was just announced.
    """
    _test_hooks.append_text(path, dump_json_str(encode_announced_run(record)))


__all__ = [
    "AnnouncedRun",
    "append_announced",
    "decode_announced_run",
    "encode_announced_run",
    "position_path",
    "read_announced",
]
