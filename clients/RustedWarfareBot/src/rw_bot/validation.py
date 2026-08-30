"""Field validators shared by every ``decode_*`` function in this package.

Decoders receive flat scalar payloads — configuration files, CLI input, run
artifacts — and must convert them into TypedDicts with guaranteed field types.
Each ``require_*`` helper narrows exactly one field and raises
:class:`DecodeError` with a traceable code when the field is absent or the
wrong type.

The payload type is spelled out as ``Mapping[str, str | int | float | bool]`` at every
call site rather than hidden behind an alias: every payload this package
decodes is flat and scalar, so the precise union is both accurate and
checkable. A nested payload would need its own decoder rather than a widening
of this one.

These helpers never coerce. ``require_int`` on the string ``"5"`` is an error,
not a silent ``int("5")`` — a payload that disagrees with its schema is a bug
in the producer, and coercion would hide it.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import PurePosixPath, PureWindowsPath

from rw_bot import RwBotError
from rw_bot.platform_id import is_windows

_MISSING = "RW-DECODE-001"
_WRONG_TYPE = "RW-DECODE-002"
_EMPTY_STR = "RW-DECODE-003"
_NOT_POSITIVE = "RW-DECODE-004"
_NOT_ABSOLUTE = "RW-DECODE-005"
_NOT_FINITE = "RW-DECODE-006"


class DecodeError(RwBotError):
    """A payload field was absent or carried the wrong type.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending field.
    """


def _fetch(payload: Mapping[str, str | int | float | bool], field: str) -> str | int | float | bool:
    """Return one field from a payload, or raise if it is absent.

    Args:
        payload: The payload being decoded.
        field: Field name to read.

    Returns:
        The raw field value, not yet narrowed.

    Raises:
        DecodeError: ``RW-DECODE-001`` when the field is absent.
    """
    if field not in payload:
        raise DecodeError(_MISSING, f"required field {field!r} is absent")
    return payload[field]


def require_str(payload: Mapping[str, str | int | float | bool], field: str) -> str:
    """Narrow one field to ``str``.

    Args:
        payload: The payload being decoded.
        field: Field name to read.

    Returns:
        The field value as a ``str``.

    Raises:
        DecodeError: ``RW-DECODE-001`` when absent, ``RW-DECODE-002`` when the
            value is not a ``str``.
    """
    value = _fetch(payload, field)
    if not isinstance(value, str):
        raise DecodeError(_WRONG_TYPE, f"field {field!r} must be str, got {type(value).__name__}")
    return value


def require_non_empty_str(payload: Mapping[str, str | int | float | bool], field: str) -> str:
    """Narrow one field to a ``str`` with at least one non-whitespace character.

    Args:
        payload: The payload being decoded.
        field: Field name to read.

    Returns:
        The field value as a non-empty ``str``.

    Raises:
        DecodeError: ``RW-DECODE-001`` when absent, ``RW-DECODE-002`` when not a
            ``str``, ``RW-DECODE-003`` when blank.
    """
    value = require_str(payload, field)
    if value.strip() == "":
        raise DecodeError(_EMPTY_STR, f"field {field!r} must not be blank")
    return value


def require_absolute_path(
    payload: Mapping[str, str | int | float | bool], field: str, platform: str
) -> str:
    """Narrow one field to a non-blank absolute path for a platform.

    Absoluteness is a correctness requirement here rather than a style
    preference. Every path in a launch configuration is consumed by the game
    process, which runs with the game directory as its working directory, so a
    relative path silently resolves against the pinned game tree instead of the
    caller's location. That reasoning holds on every platform.

    WHAT "ABSOLUTE" MEANS DOES NOT. The two families disagree about the exact
    case that matters here: ``/runs/x`` is a complete path on POSIX and a
    drive-RELATIVE one on Windows, and ``C:\\runs\\x`` is the reverse. Reading
    a path under the wrong family therefore does not merely mis-parse it -- it
    inverts this check, accepting exactly the values it exists to reject. The
    platform is taken as an argument rather than read from the running machine
    so the rule stays a pure function and both families are provable from
    either.

    Args:
        payload: The payload being decoded.
        field: Field name to read.
        platform: A ``sys.platform`` value naming the family to read under.

    Returns:
        The field value as an absolute path.

    Raises:
        DecodeError: ``RW-DECODE-001`` when absent, ``RW-DECODE-002`` when not a
            ``str``, ``RW-DECODE-003`` when blank, ``RW-DECODE-005`` when the
            value is not absolute under that platform's reading.
    """
    value = require_non_empty_str(payload, field)
    reader = PureWindowsPath if is_windows(platform) else PurePosixPath
    if not reader(value).is_absolute():
        raise DecodeError(
            _NOT_ABSOLUTE,
            f"field {field!r} must be an absolute path on {platform}, got {value!r}: "
            "the game process runs with the game directory as its working directory, "
            "so a relative path resolves against the game tree rather than the caller",
        )
    return value


def require_int(payload: Mapping[str, str | int | float | bool], field: str) -> int:
    """Narrow one field to ``int``.

    ``bool`` is rejected even though it subclasses ``int``: a boolean arriving
    where a count is expected is a producer bug, not an integer.

    Args:
        payload: The payload being decoded.
        field: Field name to read.

    Returns:
        The field value as an ``int``.

    Raises:
        DecodeError: ``RW-DECODE-001`` when absent, ``RW-DECODE-002`` when the
            value is not an ``int`` or is a ``bool``.
    """
    value = _fetch(payload, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DecodeError(_WRONG_TYPE, f"field {field!r} must be int, got {type(value).__name__}")
    return value


def require_positive_int(payload: Mapping[str, str | int | float | bool], field: str) -> int:
    """Narrow one field to an ``int`` greater than zero.

    Args:
        payload: The payload being decoded.
        field: Field name to read.

    Returns:
        The field value as a positive ``int``.

    Raises:
        DecodeError: ``RW-DECODE-001`` when absent, ``RW-DECODE-002`` when not an
            ``int``, ``RW-DECODE-004`` when zero or negative.
    """
    value = require_int(payload, field)
    if value <= 0:
        raise DecodeError(_NOT_POSITIVE, f"field {field!r} must be > 0, got {value}")
    return value


def require_finite_float(payload: Mapping[str, str | int | float | bool], field: str) -> float:
    """Narrow one field to a finite ``float``.

    An ``int`` is accepted and widened, because a whole-numbered coordinate is
    written by the producer as ``4250`` rather than ``4250.0`` whenever it has
    no fractional part, and rejecting that would make the schema depend on the
    value. ``bool`` is rejected despite subclassing ``int``, on the same
    reasoning as :func:`require_int`.

    Non-finite values are rejected rather than passed through. JSON has no
    encoding for them, so a ``NaN`` here means the producer emitted something
    the format cannot carry, which is a bug rather than a datum.

    Args:
        payload: The payload being decoded.
        field: Field name to read.

    Returns:
        The field value as a finite ``float``.

    Raises:
        DecodeError: ``RW-DECODE-001`` when absent, ``RW-DECODE-002`` when the
            value is not a number or is a ``bool``, ``RW-DECODE-006`` when it is
            not finite.
    """
    value = _fetch(payload, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecodeError(
            _WRONG_TYPE, f"field {field!r} must be a number, got {type(value).__name__}"
        )
    widened = float(value)
    if not isfinite(widened):
        raise DecodeError(_NOT_FINITE, f"field {field!r} must be finite, got {widened}")
    return widened


def require_bool(payload: Mapping[str, str | int | float | bool], field: str) -> bool:
    """Narrow one field to ``bool``.

    Args:
        payload: The payload being decoded.
        field: Field name to read.

    Returns:
        The field value as a ``bool``.

    Raises:
        DecodeError: ``RW-DECODE-001`` when absent, ``RW-DECODE-002`` when the
            value is not a ``bool``.
    """
    value = _fetch(payload, field)
    if not isinstance(value, bool):
        raise DecodeError(_WRONG_TYPE, f"field {field!r} must be bool, got {type(value).__name__}")
    return value


__all__ = [
    "DecodeError",
    "require_absolute_path",
    "require_bool",
    "require_finite_float",
    "require_int",
    "require_non_empty_str",
    "require_positive_int",
    "require_str",
]
