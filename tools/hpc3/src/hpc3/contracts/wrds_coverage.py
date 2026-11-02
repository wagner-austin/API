"""Validated records of an aggregates-only coverage measurement of WRDS relations.

One :class:`CoverageRow` is one line the coverage query writes with ``COPY
... FORMAT CSV``: a relation's row count, distinct securities and date span,
overall and within one declared slice. No row carries dataset contents, only
counts and the earliest and latest date.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

_FIELDS = 8
_ISO_DAY = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


class CoverageRow(TypedDict):
    """One relation's counts and span, overall and within the slice.

    ``first_date`` and ``last_date`` are ISO dates, or empty for a relation
    with no rows.
    """

    library: str
    relation: str
    rows: int
    securities: int
    first_date: str
    last_date: str
    slice_rows: int
    slice_securities: int


class CoverageReceipt(TypedDict):
    """Where and how one coverage measurement was produced."""

    host: str
    scheduler: str
    job_id: str
    query_sha256: str
    coverage_sha256: str
    remote_output: str
    slice: str
    rows: list[CoverageRow]
    scope: str


def _count(value: str) -> int:
    """Read one non-negative CSV count.

    Args:
        value: Decimal digits.

    Returns:
        The count.

    Raises:
        ValueError: ``WRDS_COVERAGE_VALUE`` for anything but decimal digits.
    """
    if not value.isdecimal():
        raise ValueError(f"WRDS_COVERAGE_VALUE: expected a count, got {value!r}")
    return int(value)


def _day(value: str) -> str:
    """Read one ISO date or an empty bound.

    Args:
        value: ``YYYY-MM-DD`` or empty.

    Returns:
        The date as written.

    Raises:
        ValueError: ``WRDS_COVERAGE_VALUE`` for text that is not a calendar
            date written ``YYYY-MM-DD``.
    """
    if value == "":
        return value
    match = _ISO_DAY.fullmatch(value)
    if match is None:
        raise ValueError(f"WRDS_COVERAGE_VALUE: expected an ISO date, got {value!r}")
    year, month, day = (int(part) for part in match.groups())
    if not 1 <= month <= 12 or not 1 <= day <= calendar.monthrange(year, month)[1]:
        raise ValueError(f"WRDS_COVERAGE_VALUE: {value!r} is not a calendar date")
    return value


def _require_consistent(row: CoverageRow) -> CoverageRow:
    """Refuse counts and spans that contradict each other.

    Args:
        row: A decoded row.

    Returns:
        The row.

    Raises:
        ValueError: ``WRDS_COVERAGE_RECORD`` for an empty name, more
            securities than rows, a slice larger than the relation, dates
            present without rows or missing with them, or a span that ends
            before it starts.
    """
    if row["library"] == "" or row["relation"] == "":
        raise ValueError("WRDS_COVERAGE_RECORD: empty library or relation name")
    if row["securities"] > row["rows"] or row["slice_securities"] > row["slice_rows"]:
        raise ValueError("WRDS_COVERAGE_RECORD: more securities than rows")
    if row["slice_rows"] > row["rows"] or row["slice_securities"] > row["securities"]:
        raise ValueError("WRDS_COVERAGE_RECORD: the slice exceeds the relation")
    dated = row["first_date"] != "" and row["last_date"] != ""
    if dated != (row["rows"] > 0) or row["first_date"] > row["last_date"]:
        raise ValueError("WRDS_COVERAGE_RECORD: the date span does not match the rows")
    return row


def decode_coverage_row(fields: Sequence[str]) -> CoverageRow:
    """Validate one coverage line.

    Args:
        fields: The line's CSV fields, in the query's column order.

    Returns:
        The typed row.

    Raises:
        ValueError: ``WRDS_COVERAGE_RECORD`` for a line of the wrong width or
            inconsistent counts, and ``WRDS_COVERAGE_VALUE`` for a malformed
            count or date.
    """
    if len(fields) != _FIELDS:
        raise ValueError(f"WRDS_COVERAGE_RECORD: expected {_FIELDS} fields, got {len(fields)}")
    return _require_consistent(
        CoverageRow(
            library=fields[0],
            relation=fields[1],
            rows=_count(fields[2]),
            securities=_count(fields[3]),
            first_date=_day(fields[4]),
            last_date=_day(fields[5]),
            slice_rows=_count(fields[6]),
            slice_securities=_count(fields[7]),
        )
    )


def encode_coverage_row(row: CoverageRow) -> JSONObject:
    """Encode a row for the receipt.

    Args:
        row: Validated row.

    Returns:
        The JSON object.
    """
    return {
        "library": row["library"],
        "relation": row["relation"],
        "rows": row["rows"],
        "securities": row["securities"],
        "first_date": row["first_date"],
        "last_date": row["last_date"],
        "slice_rows": row["slice_rows"],
        "slice_securities": row["slice_securities"],
    }


def decode_saved_row(value: JSONValue) -> CoverageRow:
    """Validate a row saved in a receipt.

    Args:
        value: Parsed JSON value.

    Returns:
        The typed row.

    Raises:
        JSONTypeError: When a field is missing or of the wrong type.
        ValueError: The same refusals :func:`decode_coverage_row` raises.
    """
    fields = narrow_json_to_dict(value)
    return _require_consistent(
        CoverageRow(
            library=require_str(fields, "library"),
            relation=require_str(fields, "relation"),
            rows=require_int(fields, "rows"),
            securities=require_int(fields, "securities"),
            first_date=_day(require_str(fields, "first_date")),
            last_date=_day(require_str(fields, "last_date")),
            slice_rows=require_int(fields, "slice_rows"),
            slice_securities=require_int(fields, "slice_securities"),
        )
    )


def encode_coverage_receipt(receipt: CoverageReceipt) -> JSONObject:
    """Encode a receipt for the JSON file saved beside its output.

    Args:
        receipt: Validated receipt.

    Returns:
        The JSON object.
    """
    rows: list[JSONValue] = [encode_coverage_row(row) for row in receipt["rows"]]
    return {
        "host": receipt["host"],
        "scheduler": receipt["scheduler"],
        "job_id": receipt["job_id"],
        "query_sha256": receipt["query_sha256"],
        "coverage_sha256": receipt["coverage_sha256"],
        "remote_output": receipt["remote_output"],
        "slice": receipt["slice"],
        "rows": rows,
        "scope": receipt["scope"],
    }


def decode_coverage_receipt(value: JSONValue) -> CoverageReceipt:
    """Validate a saved receipt.

    Args:
        value: Parsed JSON value.

    Returns:
        The typed receipt.

    Raises:
        JSONTypeError: When a field is missing or of the wrong type.
        ValueError: A saved row's refusals.
    """
    fields = narrow_json_to_dict(value)
    return CoverageReceipt(
        host=require_str(fields, "host"),
        scheduler=require_str(fields, "scheduler"),
        job_id=require_str(fields, "job_id"),
        query_sha256=require_str(fields, "query_sha256"),
        coverage_sha256=require_str(fields, "coverage_sha256"),
        remote_output=require_str(fields, "remote_output"),
        slice=require_str(fields, "slice"),
        rows=[decode_saved_row(row) for row in require_list(fields, "rows")],
        scope=require_str(fields, "scope"),
    )


__all__ = [
    "CoverageReceipt",
    "CoverageRow",
    "decode_coverage_receipt",
    "decode_coverage_row",
    "decode_saved_row",
    "encode_coverage_receipt",
    "encode_coverage_row",
]
