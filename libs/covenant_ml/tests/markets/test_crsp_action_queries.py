"""Corporate-action queries retain event identity without leaking future valuations."""

from __future__ import annotations

import pytest

from covenant_ml.markets.crsp_action_schema import CRSP_ACTIONS
from covenant_ml.markets.crsp_protocol import decode_crsp_protocol
from covenant_ml.markets.crsp_queries import (
    crsp_action_sql,
    crsp_driver_sql,
    crsp_scopes,
    synthetic_source,
    vendor_source,
)

from .test_crsp_protocol import protocol_document


def test_events_and_details_use_separate_dated_projections() -> None:
    protocol = decode_crsp_protocol(protocol_document())
    scope = crsp_scopes(protocol)[0]
    queries = crsp_action_sql(protocol, scope, vendor_source(protocol), 9, (12, 34))
    assert list(queries) == [
        "delist-events",
        "delist-details",
        "distribution-events",
        "distributions",
    ]
    assert queries["delist-events"] == (
        "COPY (SELECT permno, delistingdt FROM crsp_a_stock.stkdelists WHERE permno IN ("
        "SELECT DISTINCT permno FROM crsp_a_stock.dsenames WHERE shrcd IN (10, 11)"
        " AND exchcd IN (1, 2, 3) AND namedt <= DATE '2021-12-31'"
        " AND nameendt >= DATE '2000-01-01' AND permno IN (12, 34))"
        " AND delistingdt BETWEEN DATE '1998-01-01' AND DATE '2021-12-31'"
        " AND GREATEST(delistingdt) <= DATE '2021-12-31'"
        " ORDER BY permno, delistingdt LIMIT 10) TO STDOUT WITH (FORMAT CSV, HEADER);\n"
    )
    assert "GREATEST(delistingdt, delnextdt, delamtdt, deldlydt)" in queries["delist-details"]
    assert "GREATEST(disexdt, disdeclaredt, disrecorddt, dispaydt)" in queries["distributions"]
    assert "SELECT permno, disexdt, disseqnbr FROM" in queries["distribution-events"]
    assert "dispaydt" not in queries["distribution-events"]
    for statement in queries.values():
        assert "AND permno IN (12, 34)" in statement
        assert "LIMIT 10) TO STDOUT" in statement
        assert "2022" not in statement
    definitions = "dsenames(permno) AS (SELECT 12)"
    synthetic = crsp_action_sql(protocol, scope, synthetic_source(definitions), 9, (12, 34))
    assert all(sql.startswith("COPY (WITH " + definitions + "\n") for sql in synthetic.values())
    assert all("crsp_a_stock." not in sql for sql in synthetic.values())


def test_action_queries_cannot_disguise_final_as_validation_or_expand_the_cohort() -> None:
    protocol = decode_crsp_protocol(protocol_document())
    scope, final = crsp_scopes(protocol)
    for invalid in (final, final._replace(partition="train-validation")):
        with pytest.raises(ValueError, match="diagnostic cohorts require the frozen"):
            crsp_action_sql(protocol, invalid, vendor_source(protocol), 9, (12,))
    for securities in ((), (12, 12), (34, 12), (0,)):
        with pytest.raises(ValueError, match="sorted unique positive integers"):
            crsp_action_sql(protocol, scope, vendor_source(protocol), 9, securities)


def test_one_snapshot_driver_runs_only_fixed_validation_action_roles() -> None:
    roles = tuple(item.role for item in CRSP_ACTIONS)
    driver = crsp_driver_sql(30, partitions=("train-validation",), relations=roles)
    assert driver.count("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;") == 1
    assert driver.count("pg_current_snapshot()") == 2
    assert "final/" not in driver
    assert [line for line in driver.splitlines() if line.startswith("\\i")] == [
        "\\i train-validation/delist-events.sql",
        "\\i train-validation/delist-details.sql",
        "\\i train-validation/distribution-events.sql",
        "\\i train-validation/distributions.sql",
    ]
    for invalid in ((), ("../secret",), roles[::-1], (*roles, roles[0])):
        with pytest.raises(ValueError, match="invalid relations"):
            crsp_driver_sql(30, partitions=("train-validation",), relations=invalid)
    with pytest.raises(ValueError, match="final corporate-action"):
        crsp_driver_sql(30, partitions=("train-validation", "final"), relations=roles)
