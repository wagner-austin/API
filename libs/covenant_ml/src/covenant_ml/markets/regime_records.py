"""Validated immutable pre-entry probability movement measurements."""

from __future__ import annotations

from math import isfinite

from platform_core.json_utils import JSONObject, require_float
from typing_extensions import ReadOnly, TypedDict

from .types import DOLLAR
from .validation import require_bounded_int

QUOTE_COUNT = 6
HOUR_MS = 3_600_000


class QuoteRegime(TypedDict):
    """Six required closes or explicit unavailable measurements, in price units."""

    usable_quotes: ReadOnly[int]
    hourly_change_std: ReadOnly[float | None]
    net_change: ReadOnly[float | None]


def encode_quote_regime(record: QuoteRegime) -> JSONObject:
    """Encode validated measurements without replacing unavailable values.

    Args:
        record: Immutable pre-entry measurements.

    Returns:
        JSON quote coverage and nullable measurements.
    """
    return {
        "usable_quotes": record["usable_quotes"],
        "hourly_change_std": record["hourly_change_std"],
        "net_change": record["net_change"],
    }


def _require_measurement(document: JSONObject, key: str, low: float) -> float | None:
    """Validate a required nullable finite price measurement.

    Args:
        document: Untrusted JSON record.
        key: Required measurement name.
        low: Inclusive lower bound; upper bound is one dollar.

    Returns:
        Explicit null or finite bounded measurement.

    Raises:
        KeyError: Required nullable field is missing.
        TypeError: Measurement is not numeric.
        ValueError: REGIME-001 for nonfinite or out-of-range measurement.
    """
    if document[key] is None:
        return None
    value = require_float(document, key)
    if not isfinite(value) or not low <= value <= DOLLAR:
        raise ValueError("[REGIME-001] measurement must be finite and within price bounds")
    return value


def decode_quote_regime(document: JSONObject) -> QuoteRegime:
    """Require complete grid coverage before accepting computed measurements.

    Args:
        document: Untrusted encoded regime measurements.

    Returns:
        Immutable validated measurements.

    Raises:
        ValueError: REGIME-002 for coverage/measurement inconsistency;
            numeric validation errors propagate.
        TypeError: Required fields have incorrect types.
        KeyError: A required nullable measurement is absent.
    """
    count = require_bounded_int(document, "usable_quotes", 0, QUOTE_COUNT)
    variation = _require_measurement(document, "hourly_change_std", 0.0)
    change = _require_measurement(document, "net_change", -float(DOLLAR))
    complete = count == QUOTE_COUNT
    if (variation is not None) != complete or (change is not None) != complete:
        raise ValueError("[REGIME-002] complete quote coverage and measurements must agree")
    return QuoteRegime(usable_quotes=count, hourly_change_std=variation, net_change=change)
