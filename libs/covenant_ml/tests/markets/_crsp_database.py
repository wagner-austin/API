"""A synthetic CIZ database and the exact partition its extraction SQL returns.

``database_rows`` extends the seeded study extract with rows that straddle the
train-validation boundary. ``FakeDatabase`` stands in for ``psql`` at the
writer's process seam: it writes what the four COPY statements select for each
partition, filtering the same columns the SQL filters. The real client and
server are exercised by the native rehearsal, and the process boundary itself
by ``test_crsp_process``.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from platform_core.json_utils import JSONObject, JSONValue, require_int, require_str

from covenant_ml.markets.crsp_protocol import CrspProtocol, decode_crsp_protocol
from covenant_ml.markets.crsp_queries import CrspScope, crsp_scopes
from covenant_ml.markets.crsp_writer import PSQL

from ._crsp_study_fixtures import FINAL_END, study_protocol, study_rows

#: A security never eligible: its share code is outside the frozen universe.
INELIGIBLE = 9
#: A security first eligible after the train-validation boundary.
LATE = 10
#: A delisting before the boundary whose terminal value is dated after it.
STRADDLE = 7
#: The snapshot line the fake server prints for both readings.
SNAPSHOT = "750:750:,2026-10-31 09:00:00.5+00\n"


def _return(permno: int, previous: str, day: str) -> JSONObject:
    """One ordinary daily record."""
    return {
        "permno": permno,
        "dlycaldt": day,
        "dlyprevdt": previous,
        "dlyret": 0.01,
        "dlyretmissflg": "NA",
        "dlyretdurflg": "D1",
        "dlydelflg": "N",
        "dlyprc": 12.5,
        "dlyprcflg": "TR",
        "dlyvol": 300.0,
    }


def database_rows() -> dict[str, list[JSONObject]]:
    """The study rows plus an ineligible, a late-listed and a straddling security."""
    rows = study_rows(FINAL_END)
    rows["membership"] += [
        {
            "permno": INELIGIBLE,
            "namedt": "2015-01-01",
            "nameendt": "2024-12-31",
            "shrcd": 12,
            "exchcd": 1,
        },
        {
            "permno": LATE,
            "namedt": "2020-02-03",
            "nameendt": "2024-12-31",
            "shrcd": 11,
            "exchcd": 3,
        },
    ]
    rows["returns"] += [
        _return(INELIGIBLE, "2019-06-03", "2019-06-04"),
        _return(LATE, "2020-02-03", "2020-02-04"),
    ]
    rows["delists"].append(
        {
            "permno": STRADDLE,
            "delistingdt": "2019-12-20",
            "deldlydt": "2019-12-23",
            "delamtdt": "2020-01-10",
            "delnextdt": None,
            "delret": -0.1,
            "delretmisstype": "NA",
        }
    )
    return rows


def database_protocol() -> CrspProtocol:
    """The decoded study protocol the synthetic database was built for."""
    return decode_crsp_protocol(study_protocol())


def _valuation(row: JSONObject) -> str:
    """Latest present valuation date, or the event date when none is present."""
    days = [require_str(row, key) for key in ("deldlydt", "delamtdt", "delnextdt") if row[key]]
    return max(days, default=require_str(row, "delistingdt"))


def _key(row: JSONObject, field: str) -> tuple[int, str]:
    """Sort by security, then by one date column."""
    return require_int(row, "permno"), require_str(row, field)


def partition_rows(
    rows: dict[str, list[JSONObject]], protocol: CrspProtocol, scope: CrspScope
) -> dict[str, list[JSONObject]]:
    """Select what each COPY statement returns for one partition, in key order."""
    end, start = scope.observation_end, scope.history_start
    universe = {
        require_int(row, "permno")
        for row in rows["membership"]
        if require_int(row, "shrcd") in protocol["share_codes"]
        and require_int(row, "exchcd") in protocol["exchange_codes"]
        and require_str(row, "namedt") <= end
        and require_str(row, "nameendt") >= scope.universe_start
    }
    membership: list[JSONObject] = [
        {**row, "nameendt": min(require_str(row, "nameendt"), end)}
        for row in rows["membership"]
        if require_int(row, "permno") in universe
        and require_str(row, "namedt") <= end
        and require_str(row, "nameendt") >= start
    ]
    returns = [
        row
        for row in rows["returns"]
        if require_int(row, "permno") in universe and start <= require_str(row, "dlycaldt") <= end
    ]
    calendar = [
        row
        for row in rows["calendar"]
        if scope.calendar_start <= require_str(row, "caldt") <= scope.calendar_end
    ]
    delists = [
        row
        for row in rows["delists"]
        if require_int(row, "permno") in universe
        and start <= require_str(row, "delistingdt") <= end
        and _valuation(row) <= end
    ]
    return {
        "membership": sorted(membership, key=lambda row: _key(row, "namedt")),
        "returns": sorted(returns, key=lambda row: _key(row, "dlycaldt")),
        "calendar": calendar,
        "delists": sorted(delists, key=lambda row: _key(row, "delistingdt")),
    }


def relation_columns(protocol: CrspProtocol) -> dict[str, tuple[str, ...]]:
    """Each relation role's frozen ordered columns."""
    return {
        "membership": protocol["membership_columns"],
        "returns": protocol["returns_columns"],
        "calendar": protocol["calendar_columns"],
        "delists": protocol["delists_columns"],
    }


def copy_csv(rows: list[JSONObject], columns: tuple[str, ...]) -> bytes:
    """Render rows as ``COPY ... WITH (FORMAT CSV, HEADER)`` would, nulls empty."""
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        values: list[JSONValue] = ["" if row[column] is None else row[column] for column in columns]
        writer.writerow(values)
    return stream.getvalue().encode()


class FakeDatabase:
    """Answer the writer's one client invocation with the partitioned CSV files."""

    def __init__(self, rows: dict[str, list[JSONObject]], *, code: int = 0) -> None:
        self.rows = rows
        self.code = code
        self.snapshots = (SNAPSHOT, SNAPSHOT)
        self.calls: list[tuple[int, int]] = []
        self.driver = ""
        self.extra: dict[str, bytes] = {}

    def __call__(
        self, command: tuple[str, ...], directory: Path, *, seconds: int, max_file_bytes: int
    ) -> int:
        """Write every COPY target the driver names, then return the configured code."""
        assert command == PSQL
        self.calls.append((seconds, max_file_bytes))
        self.driver = (directory / "extract.sql").read_text("utf-8")
        protocol = database_protocol()
        columns = relation_columns(protocol)
        for scope in crsp_scopes(protocol):
            for name, records in partition_rows(self.rows, protocol, scope).items():
                target = directory / scope.partition / (name + ".csv")
                target.write_bytes(copy_csv(records, columns[name]))
        for name, payload in self.extra.items():
            (directory / name).write_bytes(payload)
        (directory / "snapshot-first.csv").write_text(self.snapshots[0], encoding="utf-8")
        (directory / "snapshot-last.csv").write_text(self.snapshots[1], encoding="utf-8")
        return self.code


__all__ = [
    "INELIGIBLE",
    "LATE",
    "SNAPSHOT",
    "STRADDLE",
    "FakeDatabase",
    "copy_csv",
    "database_protocol",
    "database_rows",
    "partition_rows",
    "relation_columns",
]
