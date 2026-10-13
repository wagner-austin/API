"""Domain validation shared by native exchange and saved-record codecs."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)

from .types import DOLLAR


def read_document(path: Path) -> JSONObject:
    """Read a strict JSON object from persisted evidence.

    Args:
        path: Required file.

    Returns:
        Validated JSON object.

    Raises:
        OSError: Evidence cannot be read.
        ValueError: JSON is malformed.
        TypeError: The root is not an object.
    """
    return narrow_json_to_dict(load_json_str(path.read_text("utf-8")))


def require_bounded_int(document: JSONObject, key: str, low: int, high: int) -> int:
    """Read an integer in a closed interval.

    Args:
        document: Decoded JSON object.
        key: Required field name.
        low: Inclusive minimum.
        high: Inclusive maximum.

    Returns:
        The validated integer.

    Raises:
        ValueError: MARKET-001 when outside the interval.
        TypeError: When the field is absent or not an integer.
    """
    value = require_int(document, key)
    if not low <= value <= high:
        raise ValueError(f"[MARKET-001] {key}={value} must lie in [{low}, {high}]")
    return value


def require_optional_price(document: JSONObject, key: str) -> int | None:
    """Read a required nullable price without treating absence as null.

    Args:
        document: Decoded record.
        key: Required field.

    Returns:
        Explicit null or validated price.

    Raises:
        ValueError: MARKET-002 if the field is absent.
    """
    if key not in document:
        raise ValueError(f"[MARKET-002] missing nullable field {key}")
    if document[key] is None:
        return None
    return require_bounded_int(document, key, 0, DOLLAR)


def require_nonempty(document: JSONObject, key: str) -> str:
    """Read a nonempty identifier.

    Args:
        document: Decoded record.
        key: Required field.

    Returns:
        The identifier.

    Raises:
        ValueError: MARKET-003 when empty or whitespace-only.
    """
    value = require_str(document, key)
    if not value.strip():
        raise ValueError(f"[MARKET-003] {key} must not be blank")
    return value


def require_day(document: JSONObject, key: str) -> str:
    """Read a canonical ISO calendar date.

    Args:
        document: Decoded record.
        key: Required field.

    Returns:
        YYYY-MM-DD text.

    Raises:
        ValueError: Invalid date, or MARKET-004 for noncanonical spelling.
    """
    value = require_str(document, key)
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError(f"[MARKET-004] {key} must use YYYY-MM-DD")
    return value


def parse_fixed(text: str, decimals: int) -> int:
    """Decode a nonnegative fixed-point exchange string without float rounding.

    Args:
        text: Exchange numeric string.
        decimals: Maximum fractional precision, between zero and six.

    Returns:
        Integer scaled by ten to the requested precision.

    Raises:
        ValueError: MARKET-005 for invalid precision or numeric spelling.
    """
    if not 0 <= decimals <= 6:
        raise ValueError(f"[MARKET-005] invalid fixed-point precision {decimals}")
    pattern = rf"[0-9]+(?:\.[0-9]{{1,{decimals}}})?" if decimals else r"[0-9]+"
    if re.fullmatch(pattern, text) is None:
        raise ValueError(f"[MARKET-005] invalid fixed-point value {text!r}")
    whole, _, fraction = text.partition(".")
    scale: int = 10**decimals
    return int(whole) * scale + int("0" + fraction.ljust(decimals, "0"))


def parse_timestamp(text: str) -> int:
    """Decode an aware exchange timestamp as epoch milliseconds.

    Args:
        text: ISO timestamp with a timezone.

    Returns:
        Epoch milliseconds.

    Raises:
        ValueError: Invalid timestamp, or MARKET-006 when timezone is missing.
    """
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError(f"[MARKET-006] timestamp needs a timezone: {text}")
    return int(value.timestamp() * 1000)


__all__ = [
    "parse_fixed",
    "parse_timestamp",
    "require_bounded_int",
    "require_day",
    "require_nonempty",
    "require_optional_price",
]
