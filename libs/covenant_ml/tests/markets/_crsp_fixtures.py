"""Synthetic four-relation files exercising the real CRSP loading path."""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

from platform_core.json_utils import JSONObject, JSONValue, dump_json_str, require_list, require_str

from .test_crsp_protocol import protocol_document


def source_rows() -> dict[str, list[JSONObject]]:
    """Generate explicit calendar flags and three stocks with hand-computable features."""
    start, end = date(2018, 12, 1), date(2020, 3, 31)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    calendar: list[JSONObject] = [
        {
            "caldt": day.isoformat(),
            "exchangegroup": "CIZ",
            "tradingflg": "Y" if day.weekday() < 5 else "N",
        }
        for day in days
    ]
    sessions = [day for day in days if day.weekday() < 5]
    returns: list[JSONObject] = []
    for permno in (1, 2, 3):
        for previous, day in pairwise(sessions):
            ret = 0.0
            if permno == 1 and day == date(2019, 1, 2):
                ret = 0.1
            elif permno == 1 and day == date(2019, 2, 1):
                ret = -0.2
            elif permno == 1 and day == date(2019, 12, 2):
                ret = 10.0
            returns.append(
                {
                    "permno": permno,
                    "dlycaldt": day.isoformat(),
                    "dlyprevdt": previous.isoformat(),
                    "dlyret": ret,
                    "dlyretmissflg": "NA",
                    "dlyretdurflg": "DU",
                    "dlydelflg": "N",
                    "dlyprc": 10.0,
                    "dlyprcflg": "TR",
                    "dlyvol": 50.0 if permno == 3 else 100.0,
                }
            )
    membership: list[JSONObject] = [
        {
            "permno": permno,
            "namedt": "2018-01-01",
            "nameendt": "2024-12-31",
            "shrcd": 10,
            "exchcd": 1,
        }
        for permno in (1, 2, 3)
    ]
    return {"membership": membership, "returns": returns, "calendar": calendar, "delists": []}


def write_sources(directory: Path, rows: dict[str, list[JSONObject]]) -> tuple[Path, JSONObject]:
    """Write real protocol, SQL and CSV bytes and checksum all nine evidence files."""
    protocol = protocol_document()
    protocol["max_universe"] = 2
    protocol["history_start"] = "2018-01-01"
    protocol["train_start"] = "2019-01-01"
    protocol["validation_start"] = "2020-01-01"
    manifest: JSONObject = {"extracted_at": "2026-10-29T00:00:00+00:00"}
    manifest["protocol"] = save_file(directory, "protocol.json", dump_json_str(protocol).encode())
    for name, records in rows.items():
        columns = [
            require_str({"value": value}, "value")
            for value in require_list(protocol, name + "_columns")
        ]
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in records:
            values: list[JSONValue] = [row[column] for column in columns]
            writer.writerow(values)
        sql = f"select {', '.join(columns)} from crsp_a_stock.{require_str(protocol, name)};"
        manifest[name] = {
            "csv": save_file(directory, name + ".csv", stream.getvalue().encode()),
            "query": save_file(directory, name + ".sql", sql.encode()),
        }
    path = directory / "manifest.json"
    path.write_text(dump_json_str(manifest), encoding="utf-8")
    return path, manifest


def save_file(directory: Path, name: str, payload: bytes) -> JSONObject:
    """Persist an exact test artifact and compute its actual staging identity."""
    (directory / name).write_bytes(payload)
    return {"name": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
