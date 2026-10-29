"""Decode exact CIZ CSV projections without imputing absent source values."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from math import isfinite

from platform_core.json_utils import JSONObject, JSONValue, require_int, require_str

from .validation import require_day


def _cell(column: str, text: str) -> JSONValue:
    """Validate source scalar spelling; blank nullable fields stay explicitly null."""
    if column in {"permno", "shrcd", "exchcd"}:
        number = int(text)
        if str(number) != text or (column == "permno" and number <= 0):
            raise ValueError("[CRSP-007] invalid integer identifier or classification")
        return number
    if column in {"dlyret", "dlyprc", "dlyvol", "delret"}:
        if text == "":
            return None
        value = float(text)
        if (
            not isfinite(value)
            or (column in {"dlyret", "delret"} and value < -1)
            or (column == "dlyvol" and value < 0)
        ):
            raise ValueError("[CRSP-007] nonfinite or out-of-domain numeric observation")
        return value
    if column in {"dlyprevdt", "delamtdt", "delnextdt"} and text == "":
        return None
    if column in {
        "namedt",
        "nameendt",
        "dlycaldt",
        "dlyprevdt",
        "caldt",
        "delistingdt",
        "deldlydt",
        "delamtdt",
        "delnextdt",
    }:
        return require_day({"day": text}, "day")
    if not text.strip():
        raise ValueError("[CRSP-007] required source flag is blank")
    return text


def _key(row: JSONObject, relation: str, extracted_day: str) -> tuple[str, ...]:
    """Require relation chronology and return its unique source key."""
    if relation == "calendar":
        day = require_str(row, "caldt")
        if require_str(row, "exchangegroup") != "CIZ" or require_str(row, "tradingflg") not in {
            "Y",
            "N",
        }:
            raise ValueError("[CRSP-007] require the declared CIZ calendar flags")
        key = (day,)
    elif relation == "membership":
        first, last = require_str(row, "namedt"), require_str(row, "nameendt")
        if first > last:
            raise ValueError("[CRSP-007] membership interval is reversed")
        # End dates may be vendor open-ended bounds after extraction.
        day = first
        key = (str(require_int(row, "permno")), first, last)
    else:
        field = "dlycaldt" if relation == "returns" else "delistingdt"
        day = require_str(row, field)
        key = (str(require_int(row, "permno")), day)
        if relation == "returns":
            previous = row["dlyprevdt"]
            if previous is not None and require_str(row, "dlyprevdt") >= day:
                raise ValueError("[CRSP-007] return predecessor must precede its observation")
    if day >= extracted_day:
        raise ValueError("[CRSP-007] observation must precede extraction day")
    return key


def decode_crsp_rows(
    payload: bytes, relation: str, columns: tuple[str, ...], extracted_at: str
) -> tuple[JSONObject, ...]:
    """Require exact projections, valid scalars and unique natural source keys.

    Args:
        payload: Previously checksum-verified CSV bytes.
        relation: One of the four protocol relation roles.
        columns: Exact ordered columns from the frozen protocol decoder.
        extracted_at: Validated extraction timestamp.

    Returns:
        Rows with explicit nulls retained; no missing value is replaced by zero.

    Raises:
        ValueError: CRSP-007 for malformed projection, data or duplicate keys.
    """
    if relation not in {"membership", "returns", "calendar", "delists"}:
        raise ValueError("[CRSP-007] unknown source relation")
    reader = csv.reader(io.StringIO(payload.decode("utf-8")), strict=True)
    if next(reader, None) != list(columns):
        raise ValueError("[CRSP-007] CSV projection differs from frozen ordered columns")
    keys: set[tuple[str, ...]] = set()
    rows: list[JSONObject] = []
    extracted_day = datetime.fromisoformat(extracted_at).date().isoformat()
    for values in reader:
        if len(values) != len(columns):
            raise ValueError("[CRSP-007] CSV row width differs from its projection")
        row: JSONObject = {
            column: _cell(column, value) for column, value in zip(columns, values, strict=True)
        }
        key = _key(row, relation, extracted_day)
        if key in keys:
            raise ValueError("[CRSP-007] duplicate source key")
        keys.add(key)
        rows.append(row)
    return tuple(rows)
