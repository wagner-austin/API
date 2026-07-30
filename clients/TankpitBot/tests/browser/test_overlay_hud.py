"""Tests for the in-page HUD template + CDP update call.

Covers the token substitution in the update expression (payload, CSS,
DOM body, flag binding), the fixed-geometry guarantees the card makes
(fixed width, ellipsis clipping, pointer-events discipline), and the
one-evaluate-per-tick CDP contract.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject

from tankpit_bot.browser.flag_capture import FLAG_BINDING_NAME
from tankpit_bot.browser.overlay_hud import (
    HUD_ELEMENT_ID,
    build_hud_expression,
    update_bot_overlay,
)
from tests.browser.test_overlay import make_overlay


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


class TestBuildHudExpression:
    """Token substitution + fixed-geometry contract."""

    def test_substitutes_every_template_token(self) -> None:
        """Payload, CSS, body, and binding tokens are all replaced."""
        expression = build_hud_expression(make_overlay())

        assert "__PAYLOAD__" not in expression
        assert "__CSS__" not in expression
        assert "__BODY__" not in expression
        assert "__FLAG_BINDING__" not in expression
        assert f"window.{FLAG_BINDING_NAME}(JSON.stringify" in expression

    def test_embeds_rendered_slot_values(self) -> None:
        """The payload JSON carries the rendered slot values.

        Non-ASCII glyphs (the ``·`` separator, the ``→`` arrow) are
        JSON-escaped by ``dump_json_str``, so the assertions stick to
        the ASCII-safe parts of each slot value.
        """
        expression = build_hud_expression(make_overlay())

        assert HUD_ELEMENT_ID in expression
        assert '"mode_text":"HUNT \\u00b7 ENGAGE"' in expression
        assert "633/1100" in expression
        assert "purple-4 #512" in expression

    def test_card_geometry_is_fixed(self) -> None:
        """The stylesheet pins the card width and clips variable text."""
        expression = build_hud_expression(make_overlay())

        assert "width:272px" in expression
        assert "text-overflow:ellipsis" in expression
        assert "font-variant-numeric:tabular-nums" in expression

    def test_card_never_eats_game_input_except_the_flag_button(self) -> None:
        """The card is pointer-inert; only the flag button opts back in."""
        expression = build_hud_expression(make_overlay())

        assert "pointer-events:none" in expression
        assert "pointer-events:auto" in expression

    def test_carries_the_fiesta_glass_recipe(self) -> None:
        """The stylesheet carries the fiesta frosted-glass panel recipe."""
        expression = build_hud_expression(make_overlay())

        assert "rgba(24,34,80,0.28)" in expression
        assert "backdrop-filter:blur(6px) saturate(1.1)" in expression
        assert "rgba(25,50,230,0.18)" in expression


class TestUpdateBotOverlay:
    """One evaluate per tick against the CDP session."""

    def test_sends_single_evaluate_with_hud_expression(self) -> None:
        """The update lands as one Runtime.evaluate carrying the HUD."""
        cdp = _RecordingCDPSession()

        update_bot_overlay(cdp, make_overlay())

        assert len(cdp.expressions) == 1
        assert HUD_ELEMENT_ID in cdp.expressions[0]
        assert "HUNT: shoot red-8" in cdp.expressions[0]
