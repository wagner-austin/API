"""Publication codecs preserve exact identity, chronology and decimal precision."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.bls_records import decode_cpi_release, encode_cpi_release


def release_document() -> JSONObject:
    """Return the observed August release's normalized identity."""
    return {
        "target_month": "2026-08",
        "published_at": "2026-09-11T12:30:00+00:00",
        "release_id": "USDL-26-1496",
        "annual_change": 3.4,
    }


def test_release_roundtrip_and_timezone_normalization() -> None:
    document = release_document()
    assert encode_cpi_release(decode_cpi_release(document)) == document
    shifted = dict(document, published_at="2026-09-11T08:30:00-04:00")
    assert encode_cpi_release(decode_cpi_release(shifted)) == document
    negative = dict(document, annual_change=-3.4)
    assert encode_cpi_release(decode_cpi_release(negative)) == negative


@pytest.mark.parametrize(
    "key,value,code",
    [
        ("release_id", "OTHER-26-1496", "CPI-001"),
        ("release_id", "USDL-25-1496", "CPI-001"),
        ("published_at", "2026-08-11T12:30:00Z", "CPI-001"),
        ("annual_change", float("inf"), "CPI-002"),
        ("annual_change", float("nan"), "CPI-002"),
        ("annual_change", 3.45, "CPI-002"),
    ],
)
def test_invalid_normalized_release(key: str, value: JSONValue, code: str) -> None:
    with pytest.raises(ValueError, match=code):
        decode_cpi_release(dict(release_document(), **{key: value}))


def test_required_release_types() -> None:
    with pytest.raises(TypeError, match="annual_change"):
        decode_cpi_release(dict(release_document(), annual_change="3.4"))
