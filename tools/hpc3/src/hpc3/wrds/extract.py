"""Bounded extraction inside a verified WRDS attempt; rows stay in private CSV."""

from __future__ import annotations

import csv
import datetime
import io
import pathlib

from platform_core.json_utils import JSONObject, dump_json_str

from hpc3.contracts.wrds_extract import CrspSelection, encode_crsp_selection, extraction_sql
from hpc3.contracts.wrds_sql import psql_argv
from hpc3.wrds import _test_hooks


def extract(selection: CrspSelection) -> None:
    """Save the exact query, rows and metadata, refusing partial success.

    Args:
        selection: Bounded CRSP daily selection.

    Raises:
        ValueError: On failed queries, empty/oversize results or unexpected rows.
        FileExistsError: If this output directory already contains an extract.
        TimeoutExpired: If the database process exceeds its wall deadline.
    """
    sql = extraction_sql(selection)
    environment = _test_hooks.environment()
    directory = pathlib.Path(environment["WRDS_OUTPUT_DIR"])
    query = directory / "extract.sql"
    data = directory / "data.csv"
    started = _test_hooks.now_iso()
    if selection["end_date"] >= started[:10]:
        raise ValueError("WRDS_EXTRACT_FUTURE: end date must precede the extraction day")
    _test_hooks.write(query, sql.encode())
    _test_hooks.write(data, b"")
    code = _test_hooks.execute(psql_argv(str(query), str(data)), directory, environment, 150)
    if code != 0:
        raise ValueError(f"WRDS_EXTRACT_QUERY: psql exited {code}; see outputs/payload.err")
    rows = csv.reader(io.StringIO(_test_hooks.read(data).decode(), newline=""))
    if next(rows, None) != ["permno", "date", "ret"]:
        raise ValueError("WRDS_EXTRACT_HEADER: unexpected COPY columns")
    count = 0
    previous = ""
    for row in rows:
        if len(row) != 3 or row[0] != str(selection["security_id"]):
            raise ValueError("WRDS_EXTRACT_ROW: unexpected columns or security")
        date = datetime.date.fromisoformat(row[1]).isoformat()
        if not selection["start_date"] <= date <= selection["end_date"] or date <= previous:
            raise ValueError("WRDS_EXTRACT_ROW: date outside scope or duplicate/unordered date")
        previous = date
        count += 1
    if not 3 <= count <= selection["max_rows"]:
        raise ValueError(f"WRDS_EXTRACT_ROWS: {count} rows; require 3..{selection['max_rows']}")
    receipt: JSONObject = {
        "selection": encode_crsp_selection(selection),
        "started_at": started,
        "extracted_at": _test_hooks.now_iso(),
        "rows": count,
        "csv_file": data.name,
        "csv_sha256": _test_hooks.digest_file(data),
        "query_file": query.name,
        "query_sha256": _test_hooks.digest_file(query),
    }
    _test_hooks.write(directory / "extraction.json", (dump_json_str(receipt) + "\n").encode())
    _test_hooks.write(
        directory / "metrics.json",
        (dump_json_str([{"name": "extracted_rows", "value": count}]) + "\n").encode(),
    )
