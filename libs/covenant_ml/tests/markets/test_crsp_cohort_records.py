"""Strict diagnostic lineage codecs and the shared lifted CSV representation."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONValue, require_dict

from covenant_ml.markets.crsp_cohort_records import (
    CrspCohort,
    decode_crsp_cohort,
    decode_crsp_cohort_plan,
    encode_crsp_cohort,
    encode_crsp_cohort_plan,
)
from covenant_ml.markets.crsp_extract import decode_crsp_extract
from covenant_ml.markets.crsp_rows import decode_crsp_rows, encode_crsp_rows

from ._crsp_fixtures import panel_protocol, source_rows, write_sources


def _plan() -> JSONObject:
    return {
        "bundle_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "audit_sha256": "3" * 64,
        "expected_cases": 27,
        "max_source_rows": 40_000_000,
        "max_cohort_bytes": 100_000_000,
    }


def test_plan_roundtrips_without_identifiers_or_observations() -> None:
    assert encode_crsp_cohort_plan(decode_crsp_cohort_plan(_plan())) == _plan()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unknown", 1),
        ("bundle_sha256", "wrong"),
        ("protocol_sha256", "wrong"),
        ("audit_sha256", "wrong"),
        ("expected_cases", 0),
        ("expected_cases", 65),
        ("max_source_rows", 0),
        ("max_cohort_bytes", 0),
    ],
)
def test_plan_refuses_unbounded_or_unpinned_inputs(field: str, value: JSONValue) -> None:
    document = _plan()
    document[field] = value
    with pytest.raises(ValueError, match="CRSP_COHORT_PLAN|CAMPAIGN-001|MARKET-001"):
        decode_crsp_cohort_plan(document)


def _document(tmp_path: Path) -> JSONObject:
    _, manifest = write_sources(tmp_path, source_rows(), panel_protocol())
    source = decode_crsp_extract(manifest)
    plan = _plan()
    plan["protocol_sha256"] = source["protocol"]["sha256"]
    return encode_crsp_cohort(
        CrspCohort(
            kind="crsp-diagnostic-cohort-v1",
            derived_at="2026-11-02T00:00:00+00:00",
            plan=decode_crsp_cohort_plan(plan),
            source=source,
            subset=source,
        )
    )


def test_cohort_roundtrips_but_refuses_study_manifest_decoding(tmp_path: Path) -> None:
    document = _document(tmp_path)
    assert encode_crsp_cohort(decode_crsp_cohort(document)) == document
    with pytest.raises(ValueError, match="CRSP-006.*exact extract fields"):
        decode_crsp_extract(document)


@pytest.mark.parametrize(
    "mutation", ["fields", "kind", "time", "source", "subset", "vintage", "earlier"]
)
def test_inconsistent_lineage_is_refused(tmp_path: Path, mutation: str) -> None:
    document = _document(tmp_path)
    if mutation == "fields":
        document["unknown"] = 1
    elif mutation == "kind":
        document["kind"] = "study"
    elif mutation == "time":
        document["derived_at"] = "2026-11-02T00:00:00"
    elif mutation == "source":
        require_dict(require_dict(document, "source"), "protocol")["sha256"] = "f" * 64
    elif mutation == "subset":
        require_dict(require_dict(document, "subset"), "protocol")["size_bytes"] = 1
    elif mutation == "vintage":
        require_dict(document, "subset")["extracted_at"] = "2026-11-01T00:00:00+00:00"
    else:
        document["derived_at"] = "2026-10-28T00:00:00+00:00"
    with pytest.raises(ValueError, match="CRSP_COHORT_MANIFEST"):
        decode_crsp_cohort(document)


def test_csv_encoder_preserves_nulls_and_frozen_column_order() -> None:
    rows: tuple[JSONObject, ...] = (
        {
            "permno": 7,
            "delistingdt": "2020-01-02",
            "deldlydt": "2020-01-03",
            "delamtdt": None,
            "delnextdt": None,
            "delret": None,
            "delretmisstype": "DM",
        },
    )
    columns = (
        "permno",
        "delistingdt",
        "deldlydt",
        "delamtdt",
        "delnextdt",
        "delret",
        "delretmisstype",
    )
    encoded = encode_crsp_rows(rows, columns)
    assert encoded == (
        b"permno,delistingdt,deldlydt,delamtdt,delnextdt,delret,delretmisstype\n"
        b"7,2020-01-02,2020-01-03,,,,DM\n"
    )
    assert decode_crsp_rows(encoded, "delists", columns, "2026-10-31T00:00:00Z") == rows
    assert encode_crsp_rows((), columns) == encoded.splitlines(keepends=True)[0]
