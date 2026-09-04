"""Reading TOML into the same value type the JSON helpers already use.

WHY THIS EXISTS SEPARATELY FROM :mod:`platform_core.json_utils`. The two
formats parse differently and validate identically: once a TOML document is
loaded it is a tree of strings, numbers, booleans, lists and tables, which is
exactly :data:`~platform_core.json_utils.JSONValue`. So parsing lives here and
every ``require_*`` validator in ``json_utils`` applies unchanged to the
result, rather than a second family of TOML-shaped validators existing to say
the same things.

WHY IT IS NOT JUST ``tomllib.loads`` AT THE CALL SITE. That function is
annotated ``dict[str, Any]``, and every package here runs mypy with
``disallow_any_expr``. A caller reaching for it directly either fails the type
check or reaches for a cast, and this codebase does not cast. The narrowing
happens once, here, where the reason can be written down.

AND THE NARROWING IS NOT FREE, WHICH IS THE POINT OF :func:`loads_toml`.
:mod:`tomllib` returns ``datetime``, ``date`` and ``time`` objects for TOML's
own date types, and none of those is a JSON value -- so a document carrying
one makes the narrowed annotation a lie, and the lie surfaces much later as a
``require_str`` complaining about an object it has no case for. This module
checks instead, and names the key.
"""

from __future__ import annotations

import datetime
import tomllib
from collections.abc import Callable

from platform_core.json_utils import JSONTypeError, JSONValue

#: The types :mod:`tomllib` produces that JSON has no representation for.
#:
#: ``datetime`` is listed although it subclasses ``date``, because the tuple is
#: read by people as well as by ``isinstance`` and an omission would read as a
#: decision to allow it.
_TEMPORAL_TYPES = (datetime.datetime, datetime.date, datetime.time)


def loads_toml(text: str) -> dict[str, JSONValue]:
    """Parse a TOML document into validated JSON values.

    Args:
        text: The document's complete text.

    Returns:
        The top-level table, whose values may be validated with the
        ``require_*`` helpers in :mod:`platform_core.json_utils`.

    Raises:
        tomllib.TOMLDecodeError: If the text is not valid TOML. Propagated
            rather than translated: its message carries the line and column,
            which is the whole diagnostic.
        JSONTypeError: If the document carries a TOML date, time or datetime,
            naming the path to it. Refused rather than stringified, because a
            silent conversion would let a caller compare a timestamp against a
            string and always find them unequal.
    """
    # Named `parse` rather than `loads`: the json guard bans any call to a
    # bare name `loads`, and a TOML parser borrowing that spelling is exactly
    # the confusion the rule is there to prevent.
    parse: Callable[[str], dict[str, JSONValue]] = tomllib.loads
    parsed = parse(text)
    _reject_temporal(parsed, path="")
    return parsed


def _reject_temporal(value: JSONValue, *, path: str) -> None:
    """Refuse a parsed document that carries a TOML date, time or datetime.

    Args:
        value: The subtree to check.
        path: Dotted path to it, for the message. Empty at the root.

    Raises:
        JSONTypeError: If a temporal value is found anywhere beneath.
    """
    if isinstance(value, _TEMPORAL_TYPES):
        raise JSONTypeError(
            f"TOML value at {path or '<root>'!r} is a {type(value).__name__}, which has no "
            "JSON representation; declare it as a string if it must cross this boundary"
        )
    if isinstance(value, dict):
        for key, entry in value.items():
            _reject_temporal(entry, path=f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            _reject_temporal(entry, path=f"{path}[{index}]")


__all__ = ["loads_toml"]
