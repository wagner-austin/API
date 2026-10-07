"""The field readers every decoder of a dispatch queue answer is built from.

Split out of :mod:`fleet.contracts.dispatch` when that module reached the
600-line ceiling and the job gained its lease (MCPs board task c1d48330):
what a field of the queue's JSON may hold, and how a wrong one is refused, is
one concern; what a job, a listing or a trail means is the other, and stays
there. Every reader refuses with ``QUEUE_ANSWER_MALFORMED`` and echoes the
answer, because a field of the wrong shape means the tool's contract moved
and the fix is in the MCPs repo, not in this runner.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Final

from platform_core.error_codes_fleet import FleetErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, load_json_str

#: How the tool renders an instant: JavaScript's ``toISOString``, always UTC.
INSTANT: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z")


def malformed(detail: str, *, answer: str) -> AppError[FleetErrorCode]:
    """Build the refusal for an answer that is not the documented shape.

    Args:
        detail: What was wrong, specifically.
        answer: The whole answer, echoed so the reader sees what arrived.

    Returns:
        The error to raise.
    """
    return AppError(
        code=FleetErrorCode.QUEUE_ANSWER_MALFORMED,
        message=(
            f"the dispatch queue answered a shape this runner cannot read: "
            f"{detail}. The tool's contract is JSON with named keys, so this "
            f"means the contract moved and the fix is in the MCPs repo, not "
            f"here. Received: {answer[:400]}"
        ),
    )


def require_str(row: dict[str, JSONValue], key: str, *, answer: str) -> str:
    """Read one string field.

    Args:
        row: The decoded object.
        key: The field name.
        answer: The whole answer, for the error message.

    Returns:
        The value.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when absent or not a string.
    """
    value = row.get(key)
    if not isinstance(value, str):
        raise malformed(f"field {key!r} is {type(value).__name__}, not a string", answer=answer)
    return value


def require_optional_str(row: dict[str, JSONValue], key: str, *, answer: str) -> str | None:
    """Read one nullable string field.

    ``null`` and a missing key are NOT the same here, and the difference is
    checked: the tool renders every absent value as an explicit ``null``
    precisely so a consumer can tell "no node yet" from "the field is gone".

    Args:
        row: The decoded object.
        key: The field name.
        answer: The whole answer, for the error message.

    Returns:
        The value, or None when the field is present and null.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when absent, or present and
            neither a string nor null.
    """
    if key not in row:
        raise malformed(f"field {key!r} is missing", answer=answer)
    value = row[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise malformed(
            f"field {key!r} is {type(value).__name__}, not a string or null", answer=answer
        )
    return value


def require_optional_instant(row: dict[str, JSONValue], key: str, *, answer: str) -> int | None:
    """Read one nullable instant field as whole seconds since the epoch.

    Args:
        row: The decoded object.
        key: The field name.
        answer: The whole answer, for the error message.

    Returns:
        The instant, or None when the field is present and null.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when absent, neither a string nor
            null, or a string that is not a UTC instant as the tool renders one.
    """
    text = require_optional_str(row, key, answer=answer)
    if text is None:
        return None
    if INSTANT.fullmatch(text) is None:
        raise malformed(f"field {key!r} is {text!r}, not a UTC instant", answer=answer)
    return int(datetime.fromisoformat(text).timestamp())


def envelope(answer: str, key: str) -> JSONValue:
    """Pull one named member out of a tool answer.

    Args:
        answer: The tool's whole text.
        key: The member to read.

    Returns:
        Its value.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when the answer is not a JSON
            object or does not carry the member. Not an ``InvalidJsonError``:
            a caller here cannot act on "the JSON was bad" any differently
            than on "the JSON was fine and had the wrong keys", and both mean
            the same thing -- the tool changed.
    """
    body = load_json_str(answer)
    if not isinstance(body, dict):
        raise malformed(f"the answer is {type(body).__name__}, not an object", answer=answer)
    if key not in body:
        raise malformed(f"the answer has no {key!r} member", answer=answer)
    return body[key]


__all__ = [
    "INSTANT",
    "envelope",
    "malformed",
    "require_optional_instant",
    "require_optional_str",
    "require_str",
]
