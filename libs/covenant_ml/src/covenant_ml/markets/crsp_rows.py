"""Decode exact CIZ CSV projections without imputing absent source values."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from datetime import datetime
from math import isfinite

from platform_core.json_utils import JSONObject, JSONValue, require_int, require_str

from .crsp_action_schema import CRSP_ACTIONS
from .validation import require_day

_KEY_FIELDS = {
    "membership": ("permno", "namedt", "nameendt"),
    "returns": ("permno", "dlycaldt"),
    "calendar": ("caldt",),
    "delists": ("permno", "delistingdt"),
    **{relation.role: relation.key for relation in CRSP_ACTIONS},
}


def crsp_row_key(row: JSONObject, relation: str) -> tuple[str, ...]:
    """Identify a decoded row using the reader's canonical natural key.

    Args:
        row: A previously decoded source row.
        relation: A study role or fixed corporate-action diagnostic role.

    Returns:
        The exact key fields, preserving date spelling and integer identity.

    Raises:
        KeyError: Unknown relation or absent key field.
        TypeError: A key field differs from its decoded type.
    """
    return tuple(
        str(require_int(row, field))
        if field in {"permno", "disseqnbr"}
        else require_str(row, field)
        for field in _KEY_FIELDS[relation]
    )


def encode_crsp_rows(rows: Sequence[JSONObject], columns: tuple[str, ...]) -> bytes:
    """Encode decoded rows in frozen column order, retaining nulls as empty cells.

    This is also the synthetic database's CSV writer. It does not claim a new
    vendor extraction; a derived cohort must retain its transformation lineage.

    Args:
        rows: Already decoded source rows, in their original order.
        columns: The protocol's ordered projection for this relation.

    Returns:
        UTF-8 CSV with a header and explicit LF record endings.

    Raises:
        KeyError: A row is missing a required column.
    """
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        values: list[JSONValue] = ["" if row[column] is None else row[column] for column in columns]
        writer.writerow(values)
    return stream.getvalue().encode()


def _cell(column: str, text: str) -> JSONValue:
    """Validate source scalar spelling; blank nullable fields stay explicitly null."""
    if column in {
        "permno",
        "shrcd",
        "exchcd",
        "disseqnbr",
        "delpermno",
        "delpermco",
        "dispermno",
        "dispermco",
    }:
        number = int(text)
        if str(number) != text or (column == "permno" and number <= 0):
            raise ValueError("[CRSP-007] invalid integer identifier or classification")
        return number
    if column in {
        "dlyret",
        "dlyprc",
        "dlyvol",
        "delret",
        "deldtprc",
        "delnextprc",
        "deldivamt",
        "disdivamt",
        "disfacpr",
        "disfacshr",
    }:
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
        "disexdt",
        "disdeclaredt",
        "disrecorddt",
        "dispaydt",
    }:
        return require_day({"day": text}, "day")
    if not text.strip():
        raise ValueError("[CRSP-007] required source flag is blank")
    return text


def _key(row: JSONObject, relation: str, extracted_day: str) -> tuple[str, ...]:
    """Require relation chronology and return its unique source key."""
    if relation == "calendar":
        day = require_str(row, "caldt")
        if (require_str(row, "exchangegroup"), require_str(row, "tradingflg")) not in {
            ("CIZ", "Y"),
            ("N/A", "N"),
        }:
            raise ValueError("[CRSP-007] require CIZ/Y sessions or N/A/N closed calendar days")
    elif relation == "membership":
        first, last = require_str(row, "namedt"), require_str(row, "nameendt")
        if first > last:
            raise ValueError("[CRSP-007] membership interval is reversed")
        # End dates may be vendor open-ended bounds after extraction.
        day = first
    else:
        field = _KEY_FIELDS[relation][1]
        day = require_str(row, field)
        if relation == "returns":
            previous = row["dlyprevdt"]
            if previous is not None and require_str(row, "dlyprevdt") >= day:
                raise ValueError("[CRSP-007] return predecessor must precede its observation")
    if day >= extracted_day:
        raise ValueError("[CRSP-007] observation must precede extraction day")
    return crsp_row_key(row, relation)


def decode_crsp_rows(
    payload: bytes, relation: str, columns: tuple[str, ...], extracted_at: str
) -> tuple[JSONObject, ...]:
    """Require exact projections, valid scalars and unique natural source keys.

    Args:
        payload: Previously checksum-verified CSV bytes.
        relation: A study role or fixed corporate-action diagnostic role.
        columns: Exact ordered protocol columns or fixed diagnostic projection.
        extracted_at: Validated extraction timestamp.

    Returns:
        Rows with explicit nulls retained; no missing value is replaced by zero.

    Raises:
        ValueError: CRSP-007 for malformed projection, data or duplicate keys.
            Scalar errors name the relation and column, never private values;
            raw conversion exception chaining is suppressed.
    """
    if relation not in _KEY_FIELDS:
        raise ValueError("[CRSP-007] unknown source relation")
    actions = {item.role: item for item in CRSP_ACTIONS}
    action = actions.get(relation)
    if action is not None and columns != action.columns:
        raise ValueError("[CRSP-007] diagnostic projection differs from its fixed columns")
    reader = csv.reader(io.StringIO(payload.decode("utf-8")), strict=True)
    if next(reader, None) != list(columns):
        raise ValueError("[CRSP-007] CSV projection differs from frozen ordered columns")
    keys: set[tuple[str, ...]] = set()
    rows: list[JSONObject] = []
    extracted_day = datetime.fromisoformat(extracted_at).date().isoformat()
    for values in reader:
        if len(values) != len(columns):
            raise ValueError("[CRSP-007] CSV row width differs from its projection")
        row: JSONObject = {}
        for column, value in zip(columns, values, strict=True):
            if action is not None and column not in action.key and value == "":
                row[column] = None
                continue
            try:
                row[column] = _cell(column, value)
            except ValueError:
                raise ValueError(f"[CRSP-007] invalid {column} scalar in {relation} CSV") from None
        key = _key(row, relation, extracted_day)
        if key in keys:
            raise ValueError("[CRSP-007] duplicate source key")
        keys.add(key)
        rows.append(row)
    return tuple(rows)


__all__ = ["crsp_row_key", "decode_crsp_rows", "encode_crsp_rows"]
