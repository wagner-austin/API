"""Tests for the in-page diagnostic overlay HUD."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.browser.overlay import (
    OverlayStateDict,
    decode_overlay_state,
    encode_overlay_state,
    render_overlay_lines,
    update_bot_overlay,
)


def _make_overlay() -> OverlayStateDict:
    """Return a fully populated HUD payload."""
    return OverlayStateDict(
        hfsm_state="IDLE",
        ai_mode="HUNT",
        ai_mode_state="ENGAGE",
        behavior_mode="HUNT",
        behavior_reason="shoot red-8",
        command_type="shoot",
        target_x=194,
        target_y=178,
        command_sent=True,
        in_flight_kind="none",
        fuel=633,
        self_x=88,
        self_y=112,
    )


def test_overlay_state_round_trips_through_json() -> None:
    """``OverlayStateDict`` round-trips through JSON encoding."""
    overlay = _make_overlay()

    decoded = decode_overlay_state(
        narrow_json_to_dict(load_json_str(dump_json_str(encode_overlay_state(overlay))))
    )

    assert decoded == overlay


def test_overlay_state_rejects_non_bool_command_sent() -> None:
    """A non-bool ``command_sent`` raises ``JSONTypeError`` at decode."""
    raw = encode_overlay_state(_make_overlay())
    raw["command_sent"] = "yes"

    with pytest.raises(JSONTypeError, match="command_sent"):
        decode_overlay_state(raw)


def test_render_overlay_lines_shows_decision_and_position() -> None:
    """The rendered HUD lines carry state, position, decision, and action."""
    lines = render_overlay_lines(_make_overlay())

    assert lines == [
        "BOT IDLE | HUNT/ENGAGE",
        "pos (88,112) fuel 633",
        "do  shoot -> (194,178) [sent]",
        "why HUNT: shoot red-8",
        "act none",
    ]


def test_render_overlay_lines_flags_undispatched_commands() -> None:
    """A rejected dispatch renders loudly as NOT SENT."""
    overlay = OverlayStateDict(**{**_make_overlay(), "command_sent": False})

    lines = render_overlay_lines(overlay)

    assert "do  shoot -> (194,178) [NOT SENT]" in lines


class _RecordingCDPSession:
    """Fake CDP session recording evaluate expressions."""

    def __init__(self) -> None:
        """Initialize with an empty expression log."""
        self.expressions: list[str] = []

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Record the expression and return an empty result.

        Args:
            method: CDP method name.
            params: CDP call parameters.

        Returns:
            CDP-style result object.
        """
        if method == "Runtime.evaluate" and params is not None:
            self.expressions.append(str(params.get("expression", "")))
        return {"result": {"value": True}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Unused protocol member."""

    def detach(self) -> None:
        """Unused protocol member."""


def test_update_bot_overlay_injects_hud_element_with_rendered_lines() -> None:
    """The update expression targets the HUD element and embeds the lines."""
    cdp = _RecordingCDPSession()

    update_bot_overlay(cdp, _make_overlay())

    assert len(cdp.expressions) == 1
    expression = cdp.expressions[0]
    assert "tankpit-bot-hud" in expression
    assert "why HUNT: shoot red-8" in expression
    assert "pos (88,112) fuel 633" in expression
