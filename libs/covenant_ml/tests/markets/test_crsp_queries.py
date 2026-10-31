"""Partition bounds and exact SQL derived from the committed monthly protocol."""

from __future__ import annotations

import pytest

from covenant_ml.markets.crsp_protocol import decode_crsp_protocol
from covenant_ml.markets.crsp_queries import (
    CrspScope,
    CrspSource,
    crsp_driver_sql,
    crsp_relation_sql,
    crsp_scopes,
    synthetic_source,
    vendor_source,
)

from .test_crsp_protocol import protocol_document

UNIVERSE = (
    "SELECT DISTINCT permno FROM crsp_a_stock.dsenames WHERE shrcd IN (10, 11)"
    " AND exchcd IN (1, 2, 3) AND namedt <= DATE '2021-12-31'"
    " AND nameendt >= DATE '2000-01-01'"
)


def test_committed_protocol_partitions_end_at_the_boundary_and_at_coverage() -> None:
    validation, final = crsp_scopes(decode_crsp_protocol(protocol_document()))
    assert validation == CrspScope(
        partition="train-validation",
        history_start="1998-01-01",
        universe_start="2000-01-01",
        observation_end="2021-12-31",
        calendar_start="1998-01-01",
        calendar_end="2022-02-28",
        last_decision="2021-12-31",
    )
    assert final == CrspScope(
        partition="final",
        history_start="1998-01-01",
        universe_start="2000-01-01",
        observation_end="2025-02-28",
        calendar_start="1998-01-01",
        calendar_end="2025-02-28",
        last_decision="2024-12-31",
    )


def test_calendar_starts_at_the_month_of_the_history_start() -> None:
    document = protocol_document()
    document["history_start"] = "1998-03-17"
    validation, _ = crsp_scopes(decode_crsp_protocol(document))
    assert validation.calendar_start == "1998-03-01"
    assert validation.history_start == "1998-03-17"


def test_validation_calendar_beyond_coverage_is_refused() -> None:
    document = protocol_document()
    document["holdout_end"] = "2022-01-15"
    document["coverage_end"] = "2022-02-28"
    assert crsp_scopes(decode_crsp_protocol(document))[0].calendar_end == "2022-02-28"
    document["coverage_end"] = "2022-02-27"
    with pytest.raises(ValueError, match=r"\[CRSP-010\] the validation calendar"):
        crsp_scopes(decode_crsp_protocol(document))


def test_validation_queries_clip_membership_and_date_every_relation() -> None:
    protocol = decode_crsp_protocol(protocol_document())
    source = vendor_source(protocol)
    assert source == CrspSource("crsp_a_stock", "crsp_a_stock.", "")
    sql = crsp_relation_sql(protocol, crsp_scopes(protocol)[0], source, 5)
    assert sql == {
        "membership": (
            "COPY (SELECT permno, namedt, LEAST(nameendt, DATE '2021-12-31') AS nameendt,"
            " shrcd, exchcd FROM crsp_a_stock.dsenames WHERE permno IN (" + UNIVERSE + ")"
            " AND namedt <= DATE '2021-12-31' AND nameendt >= DATE '1998-01-01'"
            " ORDER BY permno, namedt, nameendt LIMIT 6) TO STDOUT WITH (FORMAT CSV, HEADER);\n"
        ),
        "returns": (
            "COPY (SELECT permno, dlycaldt, dlyprevdt, dlyret, dlyretmissflg, dlyretdurflg,"
            " dlydelflg, dlyprc, dlyprcflg, dlyvol FROM crsp_a_stock.stkdlysecuritydata"
            " WHERE permno IN (" + UNIVERSE + ")"
            " AND dlycaldt BETWEEN DATE '1998-01-01' AND DATE '2021-12-31'"
            " ORDER BY permno, dlycaldt LIMIT 6) TO STDOUT WITH (FORMAT CSV, HEADER);\n"
        ),
        "calendar": (
            "COPY (SELECT caldt, exchangegroup, tradingflg FROM crsp_a_stock.metaexchangecalendar"
            " WHERE exchangegroup = 'CIZ'"
            " AND caldt BETWEEN DATE '1998-01-01' AND DATE '2022-02-28'"
            " ORDER BY caldt LIMIT 6) TO STDOUT WITH (FORMAT CSV, HEADER);\n"
        ),
        "delists": (
            "COPY (SELECT permno, delistingdt, deldlydt, delamtdt, delnextdt, delret,"
            " delretmisstype FROM crsp_a_stock.stkdelists WHERE permno IN (" + UNIVERSE + ")"
            " AND delistingdt BETWEEN DATE '1998-01-01' AND DATE '2021-12-31'"
            " AND COALESCE(GREATEST(deldlydt, delamtdt, delnextdt), delistingdt)"
            " <= DATE '2021-12-31'"
            " ORDER BY permno, delistingdt LIMIT 6) TO STDOUT WITH (FORMAT CSV, HEADER);\n"
        ),
    }


def test_synthetic_definitions_open_every_final_statement_with_unqualified_names() -> None:
    protocol = decode_crsp_protocol(protocol_document())
    source = synthetic_source("\n dsenames(permno) AS (SELECT 1)\n")
    assert source == CrspSource("synthetic", "", "WITH dsenames(permno) AS (SELECT 1)\n")
    sql = crsp_relation_sql(protocol, crsp_scopes(protocol)[1], source, 1)
    opening = "COPY (WITH dsenames(permno) AS (SELECT 1)\nSELECT "
    assert [statement.startswith(opening) for statement in sql.values()] == [True] * 4
    assert "FROM dsenames WHERE shrcd IN (10, 11)" in sql["membership"]
    assert "LEAST(nameendt, DATE '2025-02-28') AS nameendt" in sql["membership"]
    assert "AND caldt BETWEEN DATE '1998-01-01' AND DATE '2025-02-28'" in sql["calendar"]
    ending = "LIMIT 2) TO STDOUT WITH (FORMAT CSV, HEADER);\n"
    assert [statement.endswith(ending) for statement in sql.values()] == [True] * 4
    assert "crsp_a_stock" not in "".join(sql.values())


@pytest.mark.parametrize("text", ["", "  \n", "a AS (SELECT 1); DROP x", "a AS (SELECT 1)\n\\! id"])
def test_definitions_that_could_end_the_statement_or_reach_the_client_are_refused(
    text: str,
) -> None:
    with pytest.raises(ValueError, match=r"\[CRSP-010\] synthetic definitions must be"):
        synthetic_source(text)


def test_row_limit_must_be_positive() -> None:
    protocol = decode_crsp_protocol(protocol_document())
    with pytest.raises(ValueError, match=r"\[CRSP-010\] the row limit"):
        crsp_relation_sql(protocol, crsp_scopes(protocol)[0], vendor_source(protocol), 0)


def test_driver_runs_every_relation_inside_one_read_only_transaction() -> None:
    driver = crsp_driver_sql(120).splitlines()
    assert driver[:6] == [
        "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;",
        "SET LOCAL statement_timeout = '120s';",
        "SET LOCAL DateStyle = 'ISO, YMD';",
        "SET LOCAL TimeZone = 'UTC';",
        "\\o snapshot-first.csv",
        "COPY (SELECT pg_current_snapshot()::text, now()) TO STDOUT WITH (FORMAT CSV);",
    ]
    assert driver[6:10] == [
        "\\o train-validation/membership.csv",
        "\\i train-validation/membership.sql",
        "\\o train-validation/returns.csv",
        "\\i train-validation/returns.sql",
    ]
    assert driver[-6:] == [
        "\\o final/delists.csv",
        "\\i final/delists.sql",
        "\\o snapshot-last.csv",
        "COPY (SELECT pg_current_snapshot()::text, now()) TO STDOUT WITH (FORMAT CSV);",
        "\\o",
        "COMMIT;",
    ]
    assert len(driver) == 6 + 16 + 4


def test_statement_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match=r"\[CRSP-010\] the statement timeout"):
        crsp_driver_sql(0)
