"""Strict measurement codecs and threshold boundaries for causal diagnostics."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue

from covenant_ml.markets.regime_records import decode_quote_regime, encode_quote_regime
from covenant_ml.markets.regimes import regime_label


def document() -> JSONObject:
    """Provide a complete measured grid at the declared variability boundary."""
    return {"usable_quotes": 6, "hourly_change_std": 100.0, "net_change": 0.0}


@pytest.mark.parametrize("variation,band", [(99.5, "low"), (100.0, "high"), (100.5, "high")])
@pytest.mark.parametrize("change,direction", [(-0.5, "falling"), (0.0, "flat"), (0.5, "rising")])
def test_roundtrip_and_exact_classification(
    variation: float, band: str, change: float, direction: str
) -> None:
    source = dict(document(), hourly_change_std=variation, net_change=change)
    record = decode_quote_regime(source)
    assert encode_quote_regime(record) == source
    assert regime_label(record) == band + "_" + direction


def test_explicit_unavailable_measurements() -> None:
    source: JSONObject = {"usable_quotes": 5, "hourly_change_std": None, "net_change": None}
    record = decode_quote_regime(source)
    assert encode_quote_regime(record) == source
    assert regime_label(record) == "unclassified"


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("usable_quotes", 7, "MARKET-001"),
        ("usable_quotes", -1, "MARKET-001"),
        ("usable_quotes", 5, "REGIME-002"),
        ("hourly_change_std", None, "REGIME-002"),
        ("net_change", None, "REGIME-002"),
        ("hourly_change_std", -0.1, "REGIME-001"),
        ("hourly_change_std", 10000.1, "REGIME-001"),
        ("hourly_change_std", float("nan"), "REGIME-001"),
        ("net_change", float("inf"), "REGIME-001"),
        ("net_change", -10000.1, "REGIME-001"),
    ],
)
def test_invalid_measurements(field: str, value: JSONValue, code: str) -> None:
    source = document()
    source[field] = value
    with pytest.raises(ValueError, match=code):
        decode_quote_regime(source)


def test_missing_measurement_is_not_null() -> None:
    source = document()
    del source["net_change"]
    with pytest.raises(KeyError, match="net_change"):
        decode_quote_regime(source)
