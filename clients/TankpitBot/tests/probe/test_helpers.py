"""Tests for probe helper functions."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.probe import ProbeError, extract_cdp_evaluate_value


def test_extract_cdp_evaluate_value_success() -> None:
    """Test extract_cdp_evaluate_value extracts value from valid result."""
    result: JSONObject = {"result": {"value": "test_value"}}
    assert extract_cdp_evaluate_value(result) == "test_value"


def test_extract_cdp_evaluate_value_converts_to_string() -> None:
    """Test extract_cdp_evaluate_value converts non-string values to string."""
    result: JSONObject = {"result": {"value": 123}}
    assert extract_cdp_evaluate_value(result) == "123"


def test_extract_cdp_evaluate_value_raises_on_invalid_result() -> None:
    """Test extract_cdp_evaluate_value raises ProbeError when result is not dict."""
    result: JSONObject = {"error": "simulated error"}
    with pytest.raises(ProbeError, match=r"CDP Runtime\.evaluate returned invalid result"):
        extract_cdp_evaluate_value(result)


def test_extract_cdp_evaluate_value_raises_on_missing_value() -> None:
    """Test extract_cdp_evaluate_value raises ProbeError when value is missing."""
    result: JSONObject = {"result": {}}
    with pytest.raises(ProbeError, match=r"CDP Runtime\.evaluate result missing value"):
        extract_cdp_evaluate_value(result)


def test_extract_cdp_evaluate_value_raises_on_none_value() -> None:
    """Test extract_cdp_evaluate_value raises ProbeError when value is None."""
    result: JSONObject = {"result": {"value": None}}
    with pytest.raises(ProbeError, match=r"CDP Runtime\.evaluate result missing value"):
        extract_cdp_evaluate_value(result)


def test_get_current_time_ms_returns_int() -> None:
    """Test get_current_time_ms returns an integer."""
    result = get_current_time_ms()
    assert type(result) is int
    assert result > 0
