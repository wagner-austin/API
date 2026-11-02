"""Strict refresh lineage codecs and refusal of diagnostic envelopes as study manifests."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, require_dict
from platform_core.stage_manifest import StagedFile

from covenant_ml.markets.crsp_cohort_records import CrspCohort, decode_crsp_cohort_plan
from covenant_ml.markets.crsp_extract import decode_crsp_extract
from covenant_ml.markets.crsp_refresh_records import (
    CrspRefresh,
    decode_crsp_refresh,
    encode_crsp_refresh,
)

from ._crsp_extraction_fixtures import limits
from ._crsp_fixtures import panel_protocol, source_rows, write_sources


def _document(root: Path) -> JSONObject:
    """Use real source identities and distinct historical extraction/derivation times."""
    _, manifest = write_sources(root, source_rows(), panel_protocol())
    source = decode_crsp_extract(manifest)
    plan = decode_crsp_cohort_plan(
        {
            "bundle_sha256": "1" * 64,
            "protocol_sha256": source["protocol"]["sha256"],
            "audit_sha256": "2" * 64,
            "expected_cases": 2,
            "max_source_rows": 20000,
            "max_cohort_bytes": 4_000_000,
        }
    )
    fresh = decode_crsp_extract({**manifest, "extracted_at": "2026-10-31T09:00:00+00:00"})
    return encode_crsp_refresh(
        CrspRefresh(
            kind="crsp-diagnostic-refresh-v1",
            cohort_bundle_sha256="3" * 64,
            cohort=CrspCohort(
                kind="crsp-diagnostic-cohort-v1",
                derived_at="2026-10-30T09:00:00+00:00",
                plan=plan,
                source=source,
                subset=source,
            ),
            fresh=fresh,
            source="crsp_a_stock",
            setup_sha256=None,
            completed_at="2026-10-31T09:01:00+00:00",
            snapshot="750:750:",
            transaction_started_at="2026-10-31 09:00:00.5+00",
            driver=StagedFile(name="extract.sql", sha256="4" * 64, size_bytes=1),
            differences=StagedFile(name="differences.json", sha256="5" * 64, size_bytes=1),
            limits=limits(),
        )
    )


def test_refresh_codec_roundtrip_and_study_refusal(tmp_path: Path) -> None:
    document = _document(tmp_path)
    assert encode_crsp_refresh(decode_crsp_refresh(document)) == document
    with pytest.raises(ValueError, match=r"CRSP-006.*exact extract fields"):
        decode_crsp_extract(document)


@pytest.mark.parametrize(
    ("field", "value"), [("unknown", "extra"), ("kind", "study"), ("source", "crsp")]
)
def test_unknown_fields_and_identities_refuse(tmp_path: Path, field: str, value: str) -> None:
    document = _document(tmp_path)
    document[field] = value
    with pytest.raises(ValueError, match="CRSP_REFRESH_RECORD"):
        decode_crsp_refresh(document)


@pytest.mark.parametrize(
    "mutation",
    [
        "synthetic",
        "setup",
        "protocol",
        "completed_zone",
        "transaction_zone",
        "before_cohort",
        "before_start",
        "snapshot",
    ],
)
def test_ambiguous_refresh_metadata_refuses(tmp_path: Path, mutation: str) -> None:
    document = _document(tmp_path)
    if mutation == "synthetic":
        document["source"] = "synthetic"
    elif mutation == "setup":
        document["setup_sha256"] = "f" * 64
    elif mutation == "protocol":
        require_dict(require_dict(document, "fresh"), "protocol")["sha256"] = "f" * 64
    elif mutation == "completed_zone":
        document["completed_at"] = "2026-10-31T09:01:00"
    elif mutation == "transaction_zone":
        document["transaction_started_at"] = "2026-10-31T09:00:00"
    elif mutation == "before_cohort":
        require_dict(document, "fresh")["extracted_at"] = "2026-10-29T09:00:00+00:00"
    elif mutation == "before_start":
        document["completed_at"] = "2026-10-31T08:00:00+00:00"
    else:
        document["snapshot"] = " "
    with pytest.raises(ValueError, match="CRSP_REFRESH_RECORD"):
        decode_crsp_refresh(document)
