"""Captured original and revised disclosures enforce actual information timing."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from httpx import HTTPStatusError
from platform_core.json_utils import JSONObject, require_dict, require_list

from covenant_ml.markets import _test_hooks
from covenant_ml.markets.archive import ResponseArchive
from covenant_ml.markets.disclosure_history import (
    decode_company_facts,
    decode_submission_columns,
    select_disclosed_fact,
)
from covenant_ml.markets.disclosure_records import (
    Disclosure,
    FinancialFact,
    decode_disclosure,
    decode_financial_fact,
    encode_disclosure,
    encode_financial_fact,
)
from covenant_ml.markets.validation import read_document

from ._server import ApiServer, api_server

__all__ = ["api_server"]


def captured() -> JSONObject:
    """Read the real source-derived issuer, filing and revision fixture."""
    return read_document(Path(__file__).parent / "fixtures/disclosure_apple_assets_2009.json")


def records() -> tuple[tuple[FinancialFact, ...], tuple[Disclosure, ...]]:
    """Decode all captured records through production validators."""
    document = captured()
    return (
        decode_company_facts(
            require_dict(document, "companyfacts"), 320193, "us-gaap", "Assets", "USD"
        ),
        decode_submission_columns(require_dict(document, "submissions"), 320193),
    )


def test_original_and_revision_boundaries() -> None:
    facts, filings = records()
    assert len(facts) == len(filings) == 3
    original = next(fact for fact in facts if fact["accession"] == "0001193125-09-214859")
    original_filing = next(row for row in filings if row["accession"] == original["accession"])
    quarter = next(fact for fact in facts if fact["form"] == "10-Q")
    quarter_filing = next(row for row in filings if row["form"] == "10-Q")
    amended = next(fact for fact in facts if fact["form"] == "10-K/A")
    amended_filing = next(row for row in filings if row["form"] == "10-K/A")
    for cutoff in ("2009-10-27T23:59:59Z", "2009-10-28T00:18:29Z"):
        assert (
            select_disclosed_fact(facts, filings, start=None, end="2009-09-26", cutoff=cutoff)
            is None
        )
    for cutoff in ("2009-10-28T00:18:30Z", "2010-01-26T02:23:57Z"):
        assert select_disclosed_fact(
            facts, filings, start=None, end="2009-09-26", cutoff=cutoff
        ) == (original, original_filing)
    assert original["value"] == 53851000000.0
    assert select_disclosed_fact(
        facts, filings, start=None, end="2009-09-26", cutoff="2010-01-26T02:23:58Z"
    ) == (quarter, quarter_filing)
    assert quarter["value"] == amended["value"] == 47501000000.0
    assert select_disclosed_fact(
        facts, filings, start=None, end="2009-09-26", cutoff="2010-01-26T02:25:59Z"
    ) == (amended, amended_filing)
    assert (
        select_disclosed_fact(
            facts, filings, start=None, end="2009-09-25", cutoff="2010-01-27T00:00:00Z"
        )
        is None
    )
    assert (
        select_disclosed_fact(
            facts, filings, start="2009-01-01", end="2009-09-26", cutoff="2010-01-27T00:00:00Z"
        )
        is None
    )
    assert (
        select_disclosed_fact((), (), start=None, end="2009-09-26", cutoff="2010-01-27T00:00:00Z")
        is None
    )


def test_native_submission_errors_and_empty_input() -> None:
    document = require_dict(captured(), "submissions")
    short = dict(document, form=[])
    with pytest.raises(ValueError, match=r"DISCLOSURE-003.*equal"):
        decode_submission_columns(short, 320193)
    duplicate: JSONObject = {key: require_list(document, key) * 2 for key in document}
    with pytest.raises(ValueError, match=r"DISCLOSURE-003.*unique"):
        decode_submission_columns(duplicate, 320193)
    empty: JSONObject = {key: [] for key in document}
    assert decode_submission_columns(empty, 320193) == ()


def test_native_company_identity_unit_and_start_validation() -> None:
    document = require_dict(captured(), "companyfacts")
    with pytest.raises(ValueError, match="DISCLOSURE-004"):
        decode_company_facts(document, 1, "us-gaap", "Assets", "USD")
    with pytest.raises(TypeError, match="EUR"):
        decode_company_facts(document, 320193, "us-gaap", "Assets", "EUR")
    units = require_dict(
        require_dict(require_dict(require_dict(document, "facts"), "us-gaap"), "Assets"), "units"
    )
    from platform_core.json_utils import narrow_json_to_dict

    first = narrow_json_to_dict(require_list(units, "USD")[0])
    first["start"] = "2009-01-01"
    assert (
        decode_company_facts(document, 320193, "us-gaap", "Assets", "USD")[0]["start"]
        == "2009-01-01"
    )
    first["start"] = None
    with pytest.raises(TypeError, match="start"):
        decode_company_facts(document, 320193, "us-gaap", "Assets", "USD")


@pytest.mark.parametrize(
    "field,value", [("cik", 1), ("tag", "us-gaap:Liabilities"), ("unit", "EUR")]
)
def test_mixed_fact_identity(field: str, value: int | str) -> None:
    facts, filings = records()
    changed = encode_financial_fact(facts[0])
    changed[field] = value
    with pytest.raises(ValueError, match="DISCLOSURE-005"):
        select_disclosed_fact(
            (*facts, decode_financial_fact(changed)),
            filings,
            start=None,
            end="2009-09-26",
            cutoff="2010-01-27T00:00:00Z",
        )


def test_duplicate_missing_and_conflicting_metadata() -> None:
    facts, filings = records()
    with pytest.raises(ValueError, match="DISCLOSURE-005"):
        select_disclosed_fact(
            facts,
            (*filings, filings[0]),
            start=None,
            end="2009-09-26",
            cutoff="2010-01-27T00:00:00Z",
        )
    with pytest.raises(ValueError, match=r"DISCLOSURE-006.*no issuer"):
        select_disclosed_fact(
            facts, filings[1:], start=None, end="2009-09-26", cutoff="2010-01-27T00:00:00Z"
        )
    for key, value in (("form", "10-Q/A"), ("filed", "2010-01-24")):
        changed = encode_disclosure(filings[0])
        changed[key] = value
        with pytest.raises(ValueError, match=r"DISCLOSURE-006.*differs"):
            select_disclosed_fact(
                facts,
                (decode_disclosure(changed), *filings[1:]),
                start=None,
                end="2009-09-26",
                cutoff="2010-01-27T00:00:00Z",
            )
    with pytest.raises(ValueError, match=r"DISCLOSURE-007.*duplicate fact"):
        select_disclosed_fact(
            (*facts, facts[0]), filings, start=None, end="2009-09-26", cutoff="2010-01-27T00:00:00Z"
        )
    changed = encode_disclosure(filings[0])
    changed["accepted_at"] = filings[1]["accepted_at"]
    with pytest.raises(ValueError, match=r"DISCLOSURE-007.*tied"):
        select_disclosed_fact(
            facts,
            (decode_disclosure(changed), *filings[1:]),
            start=None,
            end="2009-09-26",
            cutoff="2010-01-27T00:00:00Z",
        )
    with pytest.raises(ValueError, match="DISCLOSURE-008"):
        select_disclosed_fact(
            facts, filings, start="2009-09-27", end="2009-09-26", cutoff="2010-01-27T00:00:00Z"
        )


def test_real_http_archive_and_offline_replay(api_server: ApiServer, tmp_path: Path) -> None:
    document = captured()
    api_server.documents["/submissions"] = [require_dict(document, "submissions")]
    api_server.documents["/facts"] = [require_dict(document, "companyfacts")]
    with closing(_test_hooks.client_factory(5.0, None)) as client:
        archive = ResponseArchive(
            tmp_path, api_server.url, client, _test_hooks.clock, _test_hooks.pause
        )
        filings = decode_submission_columns(archive.capture("/submissions", {}), 320193)
        facts = decode_company_facts(
            archive.capture("/facts", {}), 320193, "us-gaap", "Assets", "USD"
        )
        assert decode_submission_columns(archive.replay("/submissions", {}), 320193) == filings
        assert (
            decode_company_facts(archive.replay("/facts", {}), 320193, "us-gaap", "Assets", "USD")
            == facts
        )
        selected = select_disclosed_fact(
            facts, filings, start=None, end="2009-09-26", cutoff="2009-10-29T00:00:00Z"
        )
        expected_fact: FinancialFact = {
            "cik": 320193,
            "accession": "0001193125-09-214859",
            "form": "10-K",
            "filed": "2009-10-27",
            "tag": "us-gaap:Assets",
            "unit": "USD",
            "start": None,
            "end": "2009-09-26",
            "value": 53851000000.0,
        }
        expected_filing: Disclosure = {
            "cik": 320193,
            "accession": "0001193125-09-214859",
            "form": "10-K",
            "filed": "2009-10-27",
            "accepted_at": "2009-10-28T00:18:29.000Z",
        }
        assert selected == (expected_fact, expected_filing)
        envelope = archive.file_for(archive.request_url("/facts", {}))
        envelope.write_text("{}", encoding="utf-8")
        with pytest.raises(TypeError, match="body"):
            archive.replay("/facts", {})
        with pytest.raises(HTTPStatusError, match="404"):
            archive.capture("/unavailable", {})
        assert not archive.file_for(archive.request_url("/unavailable", {})).exists()
    assert len(api_server.requests) == 3
