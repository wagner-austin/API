"""Canonical byte encoding for float observations.

A determinism verdict is only as trustworthy as the encoding underneath it.
Two rollouts that agree must produce identical bytes, and two that differ by
one bit in one float must produce different bytes. That rules out text
formatting, which loses bits, and rules out :func:`hash`, which is salted per
process and would make a fresh-process comparison meaningless.

So floats are encoded through :mod:`struct` at fixed width, little-endian,
with the payload length prefixed. The length prefix is what stops two
different shapes from colliding: without it ``[(1.0, 2.0)]`` and
``[(1.0,), (2.0,)]`` flatten to the same bytes.

NaN is rejected rather than encoded. ``float("nan") != float("nan")``, so a
digest that admitted NaN would report two byte-identical rollouts as unequal
under any value comparison and equal under digest comparison, which is a
contradiction the instrument must not be able to produce.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from navprobe import NavProbeError

#: Little-endian IEEE-754 binary64. Explicit byte order because the native
#: ``@`` prefix would make a digest depend on the machine that produced it.
_FLOAT_FORMAT = "<d"

#: Little-endian unsigned 32-bit, used for the element-count prefix.
_COUNT_FORMAT = "<I"

#: Upper bound on a single encoded row, set by the count prefix width.
MAX_ROW_LENGTH = 0xFFFFFFFF


class CanonicalEncodingError(NavProbeError):
    """A value could not be encoded to canonical bytes.

    Args:
        code: Stable identifier in the ``NP-CANON-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


def _require_not_nan(value: float, subject: str) -> None:
    """Refuse a value that cannot participate in an equality-based verdict.

    The single statement of this package's NaN policy. Both the encoder and the
    observation check call it, so there is one definition of what is admissible
    and one error code for violating it.

    Args:
        value: The value to check.
        subject: What is being checked, used in the error message.

    Raises:
        CanonicalEncodingError: When ``value`` is NaN.
    """
    if math.isnan(value):
        raise CanonicalEncodingError(
            "NP-CANON-001",
            f"{subject} is NaN: it compares unequal to itself, so any verdict "
            "built on it would report a value as differing from an identical "
            "copy of itself.",
        )


def require_encodable(values: Sequence[float]) -> None:
    """Refuse an observation containing a value no verdict can be built on.

    Digest comparison rejects NaN because a NaN digest is meaningless. A
    *numerical* comparison needs the same guarantee for a sharper reason: NaN
    propagates silently through arithmetic, so a NaN spread compares false
    against every threshold and a caller asking "is the spread below tolerance"
    reads it as a pass.

    Args:
        values: The observation to check.

    Raises:
        CanonicalEncodingError: When any element is NaN.
    """
    for index, value in enumerate(values):
        _require_not_nan(value, f"observation element {index}")


def encode_float(value: float) -> bytes:
    """Encode one float to its canonical little-endian binary64 bytes.

    Args:
        value: The value to encode. Infinities are permitted because they are
            ordered and compare equal to themselves.

    Returns:
        Exactly eight bytes.

    Raises:
        CanonicalEncodingError: When ``value`` is NaN, which cannot participate
            in an equality-based verdict.
    """
    _require_not_nan(value, "value")
    return struct.pack(_FLOAT_FORMAT, value)


def _require_encodable_length(count: int, subject: str) -> None:
    """Refuse a payload the 32-bit count prefix cannot describe.

    Shared by every length-prefixed encoder so the bound and its error code are
    stated once. A second copy of this check would be free to drift to a
    different limit, and two encoders disagreeing about the bound is exactly
    the class of defect the prefix exists to prevent.

    Args:
        count: Number of elements or bytes about to be prefixed.
        subject: What is being encoded, used in the error message.

    Raises:
        CanonicalEncodingError: When ``count`` exceeds :data:`MAX_ROW_LENGTH`.
    """
    if count > MAX_ROW_LENGTH:
        raise CanonicalEncodingError(
            "NP-CANON-002",
            f"{subject} of {count} exceeds the {MAX_ROW_LENGTH} limit imposed "
            "by the 32-bit length prefix.",
        )


def encode_row(values: Sequence[float]) -> bytes:
    """Encode a sequence of floats with its length prefixed.

    The prefix makes the encoding injective over shapes: two rows differing
    only in how their elements are grouped encode differently.

    Args:
        values: The values to encode, in order.

    Returns:
        Four bytes of little-endian element count followed by eight bytes per
        element.

    Raises:
        CanonicalEncodingError: When ``values`` is longer than
            :data:`MAX_ROW_LENGTH`, or when any element is NaN.
    """
    count = len(values)
    _require_encodable_length(count, "row of elements")
    parts = [struct.pack(_COUNT_FORMAT, count)]
    parts.extend(encode_float(value) for value in values)
    return b"".join(parts)


def encode_text(value: str) -> bytes:
    """Encode text as its UTF-8 bytes with the byte length prefixed.

    The prefix is what makes a sequence of encoded strings injective. Without
    it, concatenating ``"aab"`` then ``"b"`` produces the same bytes as
    ``"aa"`` then ``"bb"``, so any digest folding a list of strings would
    collide across different lists.

    Args:
        value: The text to encode.

    Returns:
        Four bytes of little-endian byte count followed by the UTF-8 payload.

    Raises:
        CanonicalEncodingError: When the UTF-8 payload is longer than
            :data:`MAX_ROW_LENGTH` bytes.
    """
    payload = value.encode("utf-8")
    _require_encodable_length(len(payload), "text of bytes")
    return struct.pack(_COUNT_FORMAT, len(payload)) + payload


__all__ = [
    "MAX_ROW_LENGTH",
    "CanonicalEncodingError",
    "encode_float",
    "encode_row",
    "encode_text",
    "require_encodable",
]
