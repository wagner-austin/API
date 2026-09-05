"""Entering the game client's own fullscreen with a trusted click."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.browser.fullscreen import (
    FULLSCREEN_TOGGLE_SELECTOR,
    FullscreenToggleMissingError,
    enter_game_fullscreen,
)
from tests.conftest import FakeCDPSessionSimple


class TestEnterGameFullscreen:
    """Locate the client's button, click it with trusted input."""

    def test_locates_the_button_and_dispatches_a_full_click(self) -> None:
        """One evaluate for the rect, then press + release at its centre."""
        cdp = FakeCDPSessionSimple()
        cdp.add_response({"result": {"value": {"x": 320.5, "y": 431.0}}})
        cdp.add_response({})
        cdp.add_response({})

        enter_game_fullscreen(cdp)

        calls = cdp.get_calls()
        assert len(calls) == 3
        method, params = calls[0]
        assert method == "Runtime.evaluate"
        if params is None:
            raise AssertionError("the locate evaluate must carry parameters")
        assert FULLSCREEN_TOGGLE_SELECTOR in str(params["expression"])
        assert params["returnByValue"] is True
        expected_press: JSONObject = {
            "type": "mousePressed",
            "x": 320.5,
            "y": 431.0,
            "button": "left",
            "clickCount": 1,
        }
        expected_release: JSONObject = {
            "type": "mouseReleased",
            "x": 320.5,
            "y": 431.0,
            "button": "left",
            "clickCount": 1,
        }
        assert calls[1] == ("Input.dispatchMouseEvent", expected_press)
        assert calls[2] == ("Input.dispatchMouseEvent", expected_release)

    def test_a_missing_button_is_a_loud_structure_drift(self) -> None:
        """No matching element refuses the session rather than letterboxing."""
        cdp = FakeCDPSessionSimple()
        cdp.add_response({"result": {"value": None}})

        with pytest.raises(FullscreenToggleMissingError, match="settings bar has drifted"):
            enter_game_fullscreen(cdp)

        assert len(cdp.get_calls()) == 1  # no click was attempted

    def test_a_malformed_rect_fails_strict_decoding(self) -> None:
        """A rect without numeric coordinates propagates the type error."""
        cdp = FakeCDPSessionSimple()
        cdp.add_response({"result": {"value": {"x": "left", "y": 431.0}}})

        with pytest.raises(JSONTypeError):
            enter_game_fullscreen(cdp)

    def test_a_response_without_a_value_field_propagates(self) -> None:
        """The CDP surface dropping its value field surfaces as ValueError."""
        cdp = FakeCDPSessionSimple()
        cdp.add_response({"result": {}})

        with pytest.raises(ValueError, match="missing value"):
            enter_game_fullscreen(cdp)
