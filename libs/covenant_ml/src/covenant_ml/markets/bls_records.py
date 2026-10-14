"""Immutable published CPI-U annual release identity and decimal measurement."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from math import isfinite

from platform_core.json_utils import JSONObject, require_float
from typing_extensions import ReadOnly, TypedDict

from .validation import parse_timestamp, require_day, require_nonempty


class CPIRelease(TypedDict):
    """One official release's target, publication time, identity and annual CPI-U."""

    target_month: ReadOnly[str]
    published_at: ReadOnly[str]
    release_id: ReadOnly[str]
    annual_change: ReadOnly[float]


def encode_cpi_release(record: CPIRelease) -> JSONObject:
    """Encode a validated published release.

    Args:
        record: Immutable release.

    Returns:
        JSON publication identity and directly reported percentage.
    """
    return {
        "target_month": record["target_month"],
        "published_at": record["published_at"],
        "release_id": record["release_id"],
        "annual_change": record["annual_change"],
    }


def decode_cpi_release(document: JSONObject) -> CPIRelease:
    """Require a dated publication and finite one-decimal annual measurement.

    Args:
        document: Untrusted normalized release.

    Returns:
        Immutable validated release, with UTC publication timestamp.

    Raises:
        ValueError: CPI-001 for publication identity or chronology; CPI-002
            for nonfinite or more-than-one-decimal annual change.
        TypeError: Required fields have incorrect types.
    """
    target = require_nonempty(document, "target_month")
    require_day({"day": target + "-01"}, "day")
    published = require_nonempty(document, "published_at")
    instant = datetime.fromtimestamp(parse_timestamp(published) / 1000, UTC)
    identifier = require_nonempty(document, "release_id")
    if (
        re.fullmatch(r"USDL-[0-9]{2}-[0-9]{4}", identifier) is None
        or identifier[5:7] != str(instant.year)[-2:]
        or instant.strftime("%Y-%m") <= target
    ):
        raise ValueError("[CPI-001] release identifier or publication month is inconsistent")
    value = require_float(document, "annual_change")
    if (
        not isfinite(value)
        or Decimal(str(value)) * 10 != (Decimal(str(value)) * 10).to_integral_value()
    ):
        raise ValueError("[CPI-002] require a finite annual percentage reported to one decimal")
    return CPIRelease(
        target_month=target,
        published_at=instant.isoformat(),
        release_id=identifier,
        annual_change=value,
    )
