"""Explicit corporate-action CSV fixtures at the shared bounded-client seam."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from platform_core.json_utils import JSONObject

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.crsp_action_schema import CRSP_ACTIONS
from covenant_ml.markets.crsp_diagnostic import capture_crsp_diagnostic
from covenant_ml.markets.crsp_rows import encode_crsp_rows
from covenant_ml.markets.crsp_snapshot import PSQL

from ._crsp_database import SNAPSHOT
from ._crsp_extraction_fixtures import NOW, limits
from ._crsp_refresh_fixtures import RefreshCase
from ._encrypted_fixtures import evaluator_keys


def action_rows() -> dict[str, list[JSONObject]]:
    """A paid distribution, null-linked fields, zero cash and an event without details."""
    rows: dict[str, list[JSONObject]] = {}
    for relation in CRSP_ACTIONS:
        row: JSONObject = {name: None for name in relation.columns}
        row["permno"] = 1
        row[relation.dates[0]] = "2019-06-03"
        if relation.role.startswith("distribut"):
            row["disseqnbr"] = 1
        rows[relation.role] = [row]
    rows["delist-events"].append({"permno": 1, "delistingdt": "2019-12-31"})
    rows["delist-details"][0].update(
        deldtprc=10.0,
        deldtprcflg="BA",
        delret=0.0,
        delretmisstype="MV",
        deldivamt=0.0,
        delpermno=0,
        delpermco=0,
        deldlydt="2019-06-04",
    )
    rows["distributions"][0].update(
        disordinaryflg="N",
        distype="CD",
        dispaymenttype="CASH",
        disdivamt=0.0,
        disfacpr=-1.0,
        disfacshr=0.0,
        dispaydt="2019-06-07",
        dispermno=2,
        dispermco=3,
        disamountsourcetype="NA",
    )
    return rows


class ActionDatabase:
    """Write declared CSVs without silently filtering an intentionally invalid fixture."""

    def __init__(self, rows: dict[str, list[JSONObject]]) -> None:
        self.rows = rows
        self.calls = 0
        self.driver = ""

    def __call__(
        self, command: tuple[str, ...], directory: Path, *, seconds: int, max_file_bytes: int
    ) -> int:
        assert command == PSQL
        self.calls += 1
        self.driver = (directory / "extract.sql").read_text()
        for relation in CRSP_ACTIONS:
            (directory / "train-validation" / (relation.role + ".csv")).write_bytes(
                encode_crsp_rows(self.rows[relation.role], relation.columns)
            )
        for name in ("snapshot-first.csv", "snapshot-last.csv"):
            (directory / name).write_text(SNAPSHOT)
        return 0


@contextmanager
def installed_actions(database: ActionDatabase) -> Generator[None, None, None]:
    """Inject the declared output and deterministic clock; production never branches on tests."""
    previous = _test_hooks.bounded_runner, _test_hooks.clock
    _test_hooks.bounded_runner = database
    _test_hooks.clock = lambda: NOW
    yield
    _test_hooks.bounded_runner, _test_hooks.clock = previous


def capture_actions(case: RefreshCase, setup: Path | None = None) -> JSONObject:
    """Run the shared capture with actual authentication and encryption."""
    return capture_crsp_diagnostic(
        case.cohort.destination / "bundle.json",
        case.digest,
        case.cohort.plan,
        evaluator_keys(case.cohort.extraction.root),
        operation="actions",
        scratch=case.cohort.extraction.scratch,
        destination=case.destination,
        limits=limits(),
        setup=setup,
    )
