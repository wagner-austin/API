"""Synthetic four-relation files exercising the real CRSP loading path."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str, require_list, require_str

from covenant_ml.markets.crsp_rows import encode_crsp_rows

from .test_crsp_protocol import protocol_document


def source_rows() -> dict[str, list[JSONObject]]:
    """Generate explicit calendar flags and three stocks with hand-computable features."""
    start, end = date(2018, 12, 1), date(2020, 3, 31)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    calendar: list[JSONObject] = [
        {
            "caldt": day.isoformat(),
            "exchangegroup": "CIZ" if day.weekday() < 5 else "N/A",
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


def panel_protocol() -> JSONObject:
    """The committed design with dates and universe size fitted to ``source_rows``."""
    protocol = protocol_document()
    protocol["max_universe"] = 2
    protocol["history_start"] = "2018-01-01"
    protocol["train_start"] = "2019-01-01"
    protocol["validation_start"] = "2020-01-01"
    return protocol


def version_three(protocol: JSONObject) -> JSONObject:
    """The same design under the stopped study's installed version-3 rule names."""
    old = dict(protocol)
    old["version"] = 3
    old["entry_rule"] = "confirmed-terminal-entry-cash"
    old["held_return_rule"] = "exact-missing-price-spans"
    old["terminal_rule"] = "dated-value-next-rebalance"
    return old


def write_sources(
    directory: Path, rows: dict[str, list[JSONObject]], protocol: JSONObject
) -> tuple[Path, JSONObject]:
    """Write real protocol, SQL and CSV bytes and checksum all nine evidence files."""
    manifest: JSONObject = {"extracted_at": "2026-10-29T00:00:00+00:00"}
    manifest["protocol"] = save_file(directory, "protocol.json", dump_json_str(protocol).encode())
    for name, records in rows.items():
        columns = tuple(
            require_str({"value": value}, "value")
            for value in require_list(protocol, name + "_columns")
        )
        sql = f"select {', '.join(columns)} from crsp_a_stock.{require_str(protocol, name)};"
        manifest[name] = {
            "csv": save_file(directory, name + ".csv", encode_crsp_rows(records, columns)),
            "query": save_file(directory, name + ".sql", sql.encode()),
        }
    path = directory / "manifest.json"
    path.write_text(dump_json_str(manifest), encoding="utf-8")
    return path, manifest


def save_file(directory: Path, name: str, payload: bytes) -> JSONObject:
    """Persist an exact test artifact and compute its actual staging identity."""
    (directory / name).write_bytes(payload)
    return {"name": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
