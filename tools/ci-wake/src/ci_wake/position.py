"""What this bridge has already announced, so it never announces it twice.

ONE APPEND-ONLY FILE BESIDE THE ENROLMENT RECORD, one line per push
announced. It is the bridge's whole memory, and its shape is deliberately
the same as every other record in this workspace: a JSON object per line,
appended and never rewritten, read whole, and FATAL on a line that does not
decode.

WHY A KEY SET RATHER THAN A CURSOR. The enrolment record is append-only and
a re-pushed sha appears in it more than once, so a position in that stream
does not answer "has this push been announced", which is the only question
asked here. Naming the ``(repo, sha)`` pairs does.

WHY NOT SIMPLY RE-READ THE BOARD. Because a bridge that asked the board what
it had already posted would depend on the board being reachable to decide
whether to post, and would re-announce everything the first time a query
failed. The local record answers with no network at all.

A MALFORMED LINE IS FATAL, NOT SKIPPED, and the cost here is specific: a
line read as absent means the push it names is announced AGAIN, and the
reader of that second post cannot tell it from a genuinely new verdict. So
the decoder raises and names the line, exactly as the enrolment record's
does.
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

from ci_wake import _test_hooks


class AnnouncedPush(TypedDict):
    """One push this bridge has posted about.

    Attributes:
        key: ``repo@sha``, as :func:`ci_wake.enrolment.attempt_key` spells
            it. The only field the bridge reads; the rest exist so a person
            opening the file can see what happened without joining it to
            the enrolment record.
        state: Which announced state closed the row -- ``ripe``,
            ``abandoned`` or ``stalled``. Recorded because a push closed as
            ``stalled`` whose runs later finished is a real sequence, and a
            reader needs to see that the verdict they never received was
            not lost but deliberately not waited for.
        announced_unix: When the post landed, whole seconds since the epoch.
    """

    key: str
    state: str
    announced_unix: int


def encode_announced_push(record: AnnouncedPush) -> JSONObject:
    """Encode one position record.

    Args:
        record: The record to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "key": record["key"],
        "state": record["state"],
        "announced_unix": record["announced_unix"],
    }


def decode_announced_push(value: JSONValue) -> AnnouncedPush:
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
    return AnnouncedPush(
        key=require_str(value, "key"),
        state=require_str(value, "state"),
        announced_unix=require_int(value, "announced_unix"),
    )


def position_path(enrolment: pathlib.Path) -> pathlib.Path:
    """Where the position record lives for a given enrolment record.

    Beside it rather than in a configured location, so the two files that
    must describe the same set of pushes are not separately addressable and
    moving one moves both.

    Args:
        enrolment: The enrolment record's path.

    Returns:
        The position file's path.
    """
    return enrolment.parent / "announced.jsonl"


def read_announced(path: pathlib.Path) -> frozenset[str]:
    """Read every push key this bridge has already posted about.

    Args:
        path: The position file's path.

    Returns:
        The announced keys. An absent file reads as empty rather than
        raising: a machine whose bridge has never run has announced
        nothing, and refusing the first cycle for having no history would
        make the bridge impossible to start.

    Raises:
        InvalidJsonError: If a line is not valid JSON at all.
        JSONTypeError: If a line is valid JSON but not a record. NOT skipped
            -- see the module docstring on what a skipped line costs.
    """
    if not _test_hooks.file_exists(path):
        return frozenset()
    keys: set[str] = set()
    for index, line in enumerate(_test_hooks.read_text(path).splitlines(), start=1):
        if line.strip() == "":
            continue
        value = load_json_str(line)
        if not isinstance(value, dict):
            raise JSONTypeError(
                f"{path} line {index} is a {type(value).__name__}, not an object; a "
                "position line that cannot be read means the push it names is "
                "announced a second time, and nobody reading that post can tell it "
                "from a new verdict"
            )
        keys.add(decode_announced_push(value)["key"])
    return frozenset(keys)


def append_announced(path: pathlib.Path, record: AnnouncedPush) -> None:
    """Append one position record.

    Args:
        path: The position file's path.
        record: The push that was just announced.
    """
    _test_hooks.append_text(path, dump_json_str(encode_announced_push(record)))


__all__ = [
    "AnnouncedPush",
    "append_announced",
    "decode_announced_push",
    "encode_announced_push",
    "position_path",
    "read_announced",
]
