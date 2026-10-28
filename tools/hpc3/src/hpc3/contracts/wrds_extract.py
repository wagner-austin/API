"""Bounded CRSP daily-return selection, shared by admission and extraction."""

from __future__ import annotations

import datetime

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict


class CrspSelection(TypedDict):
    """One security and an inclusive historical interval, with a hard row cap."""

    source_table: ReadOnly[str]
    security_id: ReadOnly[int]
    start_date: ReadOnly[str]
    end_date: ReadOnly[str]
    max_rows: ReadOnly[int]


def decode_crsp_selection(value: JSONValue) -> CrspSelection:
    """Validate scope before constructing any database command.

    Args:
        value: Untrusted selection document.

    Returns:
        A canonical, bounded selection.

    Raises:
        ValueError: For unknown fields, tables, invalid dates or resource bounds.
    """
    obj = narrow_json_to_dict(value)
    if set(obj) != {"source_table", "security_id", "start_date", "end_date", "max_rows"}:
        raise ValueError("WRDS_EXTRACT_FIELDS: expected the five selection fields")
    table = require_str(obj, "source_table")
    security = require_int(obj, "security_id")
    start = require_str(obj, "start_date")
    end = require_str(obj, "end_date")
    cap = require_int(obj, "max_rows")
    if table not in {"crsp.dsf", "crsp_a_stock.dsf"}:
        raise ValueError("WRDS_EXTRACT_TABLE: expected a supported CRSP daily table")
    if security < 1 or not 3 <= cap <= 10000:
        raise ValueError("WRDS_EXTRACT_BOUND: positive security and 3..10000 rows required")
    first, last = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    if first.isoformat() != start or last.isoformat() != end or first > last:
        raise ValueError("WRDS_EXTRACT_DATES: expected ordered canonical ISO dates")
    return CrspSelection(
        source_table=table, security_id=security, start_date=start, end_date=end, max_rows=cap
    )


def encode_crsp_selection(value: CrspSelection) -> JSONObject:
    """Encode selection without introducing a second scope representation.

    Args:
        value: Validated selection.

    Returns:
        Its JSON document.
    """
    return {
        "source_table": value["source_table"],
        "security_id": value["security_id"],
        "start_date": value["start_date"],
        "end_date": value["end_date"],
        "max_rows": value["max_rows"],
    }


def extraction_sql(selection: CrspSelection) -> str:
    """Render read-only COPY with a sentinel row that detects truncation.

    Args:
        selection: Requested scope; revalidated at this SQL boundary.

    Returns:
        Exact SQL to preserve beside the resulting CSV. Missing returns are
        preserved; the downstream evaluation adapter decides their validity.
    """
    scope = decode_crsp_selection(encode_crsp_selection(selection))
    return (
        "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;\n"
        "SET LOCAL statement_timeout = '120s';\n"
        "SET LOCAL DateStyle = 'ISO, YMD';\n"
        "COPY (SELECT permno, date, ret FROM "
        + scope["source_table"]
        + f" WHERE permno = {scope['security_id']}"
        + f" AND date BETWEEN DATE '{scope['start_date']}' AND DATE '{scope['end_date']}'"
        + f" ORDER BY date LIMIT {scope['max_rows'] + 1}) TO STDOUT WITH (FORMAT CSV, HEADER);\n"
        "COMMIT;\n"
    )
