"""Tests for BrowserSession fuel prober functionality."""

from __future__ import annotations

from platform_core.json_utils import JSONObject

from tankpit_bot.browser import BrowserSession, FuelProbeResult
from tests.conftest import FakeCDPSessionSimple


def test_browser_session_init_fuel_prober() -> None:
    """Test _init_fuel_prober creates FuelProber and enables polling."""
    session = BrowserSession("https://example.com")
    # Initially no fuel prober
    assert session._fuel_prober is None

    # After initialization, can poll (which proves prober was created)
    cdp = FakeCDPSessionSimple()
    # Add responses for the 3 probes that FuelProber.probe() does
    cdp.add_response({"result": {"value": []}})  # dom_bars
    cdp.add_response({"result": {"value": []}})  # js_variables
    cdp.add_response({"result": {"value": []}})  # numeric_globals

    session._init_fuel_prober(cdp)

    # Now poll should work (proves prober was created)
    cdp.add_response({"result": {"value": []}})  # dom_bars
    cdp.add_response({"result": {"value": []}})  # js_variables
    cdp.add_response({"result": {"value": []}})  # numeric_globals
    poll_result: FuelProbeResult | None = session._poll_fuel()
    if poll_result is None:
        raise AssertionError("Expected FuelProbeResult after init")
    assert poll_result["dom_bars"] == []
    assert poll_result["js_variables"] == []


def test_browser_session_poll_fuel_no_prober() -> None:
    """Test _poll_fuel returns None when prober not initialized."""
    session = BrowserSession("https://example.com")
    result = session._poll_fuel()
    assert result is None


def test_browser_session_poll_fuel_with_results() -> None:
    """Test _poll_fuel returns results and logs findings."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSessionSimple()

    # Configure fake responses for FuelProber.probe()
    bar_data: JSONObject = {
        "tag": "DIV",
        "id": "hp-bar",
        "class_name": "health",
        "width": "80%",
        "computed_width": "200px",
        "parent_class": "",
    }
    var_data: JSONObject = {"name": "fuel", "value": 800, "path": "player.fuel"}
    result_inner: JSONObject = {"value": [bar_data]}
    cdp.add_response({"result": result_inner})
    result_inner2: JSONObject = {"value": [var_data]}
    cdp.add_response({"result": result_inner2})
    result_inner3: JSONObject = {"value": []}
    cdp.add_response({"result": result_inner3})

    session._init_fuel_prober(cdp)

    # Add more responses for the poll
    result_inner4: JSONObject = {"value": [bar_data]}
    cdp.add_response({"result": result_inner4})
    result_inner5: JSONObject = {"value": [var_data]}
    cdp.add_response({"result": result_inner5})
    result_inner6: JSONObject = {"value": []}
    cdp.add_response({"result": result_inner6})

    result: FuelProbeResult | None = session._poll_fuel()
    # Narrow type via conditional that raises
    if result is None:
        raise AssertionError("_poll_fuel returned None when prober was initialized")
    # Now mypy knows result is FuelProbeResult
    assert len(result["dom_bars"]) == 1
    assert result["dom_bars"][0]["id"] == "hp-bar"
    assert result["dom_bars"][0]["width"] == "80%"
    assert len(result["js_variables"]) == 1
    assert result["js_variables"][0]["path"] == "player.fuel"
    assert result["js_variables"][0]["value"] == 800


def test_browser_session_poll_fuel_dom_bar_with_empty_width() -> None:
    """Test _poll_fuel skips logging for DOM bars with empty width."""
    session = BrowserSession("https://example.com")
    cdp = FakeCDPSessionSimple()

    # DOM bar with empty width - should not log
    bar_no_width: JSONObject = {
        "tag": "DIV",
        "id": "empty-bar",
        "class_name": "empty",
        "width": "",  # Empty width triggers the 405->404 branch
        "computed_width": "",
        "parent_class": "",
    }
    result_inner1: JSONObject = {"value": [bar_no_width]}
    cdp.add_response({"result": result_inner1})
    result_inner2: JSONObject = {"value": []}
    cdp.add_response({"result": result_inner2})
    result_inner3: JSONObject = {"value": []}
    cdp.add_response({"result": result_inner3})

    session._init_fuel_prober(cdp)

    # Add more responses for the poll
    result_inner4: JSONObject = {"value": [bar_no_width]}
    cdp.add_response({"result": result_inner4})
    result_inner5: JSONObject = {"value": []}
    cdp.add_response({"result": result_inner5})
    result_inner6: JSONObject = {"value": []}
    cdp.add_response({"result": result_inner6})

    result: FuelProbeResult | None = session._poll_fuel()
    if result is None:
        raise AssertionError("_poll_fuel returned None when prober was initialized")
    # The bar is returned but not logged (empty width branch)
    assert len(result["dom_bars"]) == 1
    assert result["dom_bars"][0]["width"] == ""
