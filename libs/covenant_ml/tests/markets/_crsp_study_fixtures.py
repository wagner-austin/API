"""A seeded four-year CRSP extract that drives the whole monthly validation study."""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise
from math import prod
from pathlib import Path

import numpy as np
from platform_core.json_utils import JSONObject, require_float, require_str

from covenant_ml.markets.crsp_extract import CrspSources, load_crsp_sources

from ._crsp_fixtures import write_sources
from .test_crsp_protocol import protocol_document

#: The security that delists on 2019-03-14 with a dated -25% terminal value.
DELISTED = 8


def study_protocol() -> JSONObject:
    """The committed design with dates, universe and bootstrap work fitted to the fixture.

    Train decisions start January 2017 and the final period starts January 2020.
    Validation retains July 2018 through October 2019 (16 months); November and
    December decisions have exits in the final period and cannot be scored.
    """
    protocol = protocol_document()
    protocol.update(
        {
            "history_start": "2016-01-01",
            "train_start": "2017-01-01",
            "validation_start": "2018-07-01",
            "holdout_start": "2020-01-01",
            "holdout_end": "2020-06-30",
            "coverage_end": "2020-08-31",
            "max_universe": 6,
            "selection_fraction": 0.5,
            "bootstrap_block_months": 3,
            "bootstrap_draws": 200,
        }
    )
    return protocol


#: The last staged day of the validation extract; validation needs no later evidence.
VALIDATION_END = date(2020, 2, 29)
#: The last staged day of the final extract, covering the June 2020 decision's exit.
FINAL_END = date(2020, 8, 31)


def study_rows(end: date = VALIDATION_END) -> dict[str, list[JSONObject]]:
    """Weekday sessions from 2016 through ``end`` and eight seeded securities.

    Liquidity rises with PERMNO, so the six most liquid form the universe until
    security 8 delists. Its daily delisting-flagged copy on 2019-03-15 carries
    +30%, which must never be compounded; the terminal value is -25%. Sessions
    after ``VALIDATION_END`` draw from a second seeded stream, so a longer
    extract repeats every validation-extract row exactly.
    """
    start = date(2016, 1, 1)
    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    calendar: list[JSONObject] = [
        {
            "caldt": day.isoformat(),
            "exchangegroup": "CIZ" if day.weekday() < 5 else "N/A",
            "tradingflg": "Y" if day.weekday() < 5 else "N",
        }
        for day in days
    ]
    sessions = [day for day in days if day.weekday() < 5]
    base = sum(1 for day in sessions if day <= VALIDATION_END)
    draws = np.random.default_rng(20261030).normal(0.0004, 0.02, size=8 * base)
    later = np.random.default_rng(20261031).normal(0.0004, 0.02, size=8 * len(sessions))

    def draw(permno: int, index: int) -> float:
        """The validation stream before the boundary, the later stream after it."""
        if index < base:
            return round(float(draws.flat[(permno - 1) * base + index]), 6)
        return round(float(later.flat[(permno - 1) * len(sessions) + index]), 6)

    returns: list[JSONObject] = []
    for permno in range(1, 9):
        for index, (previous, day) in enumerate(pairwise(sessions), start=1):
            row: JSONObject = {
                "permno": permno,
                "dlycaldt": day.isoformat(),
                "dlyprevdt": previous.isoformat(),
                "dlyret": draw(permno, index),
                "dlyretmissflg": "NA",
                "dlyretdurflg": "DU",
                "dlydelflg": "N",
                "dlyprc": 10.0,
                "dlyprcflg": "TR",
                "dlyvol": 1000.0 * permno,
            }
            if permno == DELISTED and day == date(2019, 3, 15):
                row.update({"dlyret": 0.3, "dlydelflg": "Y", "dlyretdurflg": "DD"})
            if permno != DELISTED or day <= date(2019, 3, 15):
                returns.append(row)
    membership: list[JSONObject] = [
        {
            "permno": permno,
            "namedt": "2015-01-01",
            "nameendt": "2024-12-31",
            "shrcd": 10,
            "exchcd": 1,
        }
        for permno in range(1, 9)
    ]
    delists: list[JSONObject] = [
        {
            "permno": DELISTED,
            "delistingdt": "2019-03-14",
            "deldlydt": "2019-03-15",
            "delamtdt": "2019-03-20",
            "delnextdt": None,
            "delret": -0.25,
            "delretmisstype": "NA",
        }
    ]
    return {"membership": membership, "returns": returns, "calendar": calendar, "delists": delists}


def study_sources(directory: Path) -> CrspSources:
    """Write the unmodified study evidence and load it through the verified reader."""
    path, _ = write_sources(directory, study_rows(), study_protocol())
    return load_crsp_sources(path)


def pre_entry_rows() -> dict[str, list[JSONObject]]:
    """Move the delisting to a decision date, retaining its next-day terminal copy."""
    rows = study_rows()
    rows["returns"] = [
        row
        for row in rows["returns"]
        if row["permno"] != DELISTED or require_str(row, "dlycaldt") <= "2019-03-01"
    ]
    for row in rows["returns"]:
        if row["permno"] == DELISTED and row["dlycaldt"] == "2019-03-01":
            row.update({"dlydelflg": "Y", "dlyretdurflg": "DD", "dlyprcflg": "DP"})
    rows["delists"][0].update(
        {"delistingdt": "2019-02-28", "deldlydt": "2019-03-01", "delamtdt": "2019-03-01"}
    )
    return rows


def final_manifest(directory: Path) -> Path:
    """Stage the evaluator's extract through the final period's last exit."""
    path, _ = write_sources(directory, study_rows(FINAL_END), study_protocol())
    return path


def missing_price_rows(*, price_flag: str = "MP", gap: int = 1) -> dict[str, list[JSONObject]]:
    """Replace synthetic daily observations with one exact multi-session return.

    The total held multiplier stays equal to the independent original path.
    MP/MR rows and their NA/P1-P9 successor reproduce the licensed diagnostic's
    metadata pattern, without copying a licensed value or identifier.
    """
    rows = pre_entry_rows()
    interval = [
        row
        for row in rows["returns"]
        if row["permno"] == 7 and require_str(row, "dlycaldt") >= "2019-03-11"
    ][: gap + 1]
    close = interval[-1]
    close["dlyret"] = prod(1 + require_float(row, "dlyret") for row in interval) - 1
    close.update({"dlyprevdt": "2019-03-08", "dlyretdurflg": f"P{gap}"})
    for unpriced in interval[:-1]:
        unpriced.update(
            {
                "dlyret": None,
                "dlyprevdt": None,
                "dlyretmissflg": "MP",
                "dlyretdurflg": "MR",
                "dlyprc": None,
                "dlyprcflg": price_flag,
            }
        )
    return rows


__all__ = [
    "DELISTED",
    "FINAL_END",
    "VALIDATION_END",
    "final_manifest",
    "missing_price_rows",
    "pre_entry_rows",
    "study_protocol",
    "study_rows",
    "study_sources",
]
