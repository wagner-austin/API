"""Tests for the HUD payload model + fiesta slot renderer.

Covers the JSON round-trip, the loud decode rejections, and every
display rule the renderer owns: mode banner colors, fuel meter
percent/color bands, stock-slot colors against the rank cap, the
sent/held indicator, and the combat-target slot text.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.browser.overlay import (
    FIESTA_FG,
    FIESTA_GREEN,
    FIESTA_PINK,
    FIESTA_PURPLE,
    OverlayStateDict,
    decode_overlay_state,
    encode_overlay_state,
    render_overlay_payload,
)


def make_overlay() -> OverlayStateDict:
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
        fuel_cap=1100,
        self_x=88,
        self_y=112,
        armor=25,
        duals=25,
        missiles=19,
        homings=25,
        radars=21,
        inv_cap=25,
        kills=3,
        hits=12,
        misses=4,
        rejects=1,
        target_id=512,
        target_name="purple-4",
    )


def test_overlay_state_round_trips_through_json() -> None:
    """``OverlayStateDict`` round-trips through JSON encoding."""
    overlay = make_overlay()

    decoded = decode_overlay_state(
        narrow_json_to_dict(load_json_str(dump_json_str(encode_overlay_state(overlay))))
    )

    assert decoded == overlay


def test_overlay_state_rejects_non_bool_command_sent() -> None:
    """A non-bool ``command_sent`` raises ``JSONTypeError`` at decode."""
    raw = encode_overlay_state(make_overlay())
    raw["command_sent"] = "yes"

    with pytest.raises(JSONTypeError, match="command_sent"):
        decode_overlay_state(raw)


def test_overlay_state_rejects_missing_stock_field() -> None:
    """A payload without the radar count raises ``JSONTypeError``."""
    raw = encode_overlay_state(make_overlay())
    del raw["radars"]

    with pytest.raises(JSONTypeError, match="radars"):
        decode_overlay_state(raw)


class TestModeBanner:
    """Mode banner text + color rules."""

    def test_hunt_renders_hot_pink_with_substate(self) -> None:
        """HUNT gets the fiesta ATK pink and a mode·substate label."""
        payload = render_overlay_payload(make_overlay())

        assert payload["mode_text"] == "HUNT · ENGAGE"
        assert payload["mode_color"] == FIESTA_PINK
        assert payload["mode_band"] == "rgba(255, 20, 147, 0.20)"

    def test_collect_renders_neon_green(self) -> None:
        """COLLECT gets the fiesta highlight green."""
        overlay = OverlayStateDict(
            **{**make_overlay(), "ai_mode": "COLLECT", "ai_mode_state": "PICKUP"}
        )

        payload = render_overlay_payload(overlay)

        assert payload["mode_text"] == "COLLECT · PICKUP"
        assert payload["mode_color"] == FIESTA_GREEN

    def test_unset_renders_purple_without_substate(self) -> None:
        """UNSET has no substate and gets the neutral neon purple."""
        overlay = OverlayStateDict(**{**make_overlay(), "ai_mode": "UNSET", "ai_mode_state": ""})

        payload = render_overlay_payload(overlay)

        assert payload["mode_text"] == "UNSET"
        assert payload["mode_color"] == FIESTA_PURPLE

    def test_unknown_mode_falls_back_to_purple(self) -> None:
        """A mode outside the vocabulary still renders in purple."""
        overlay = OverlayStateDict(**{**make_overlay(), "ai_mode": "REPLAY"})

        assert render_overlay_payload(overlay)["mode_color"] == FIESTA_PURPLE


class TestFuelMeter:
    """Fuel meter percent + color bands."""

    def test_mid_band_is_purple_with_exact_percent(self) -> None:
        """633/1100 renders 57% in the neutral purple."""
        payload = render_overlay_payload(make_overlay())

        assert payload["fuel_text"] == "633/1100"
        assert payload["fuel_pct"] == 57
        assert payload["fuel_color"] == FIESTA_PURPLE

    def test_full_is_green_and_overfull_clamps_to_100(self) -> None:
        """At or above capacity the meter pins to 100% green."""
        overlay = OverlayStateDict(**{**make_overlay(), "fuel": 1250})

        payload = render_overlay_payload(overlay)

        assert payload["fuel_pct"] == 100
        assert payload["fuel_color"] == FIESTA_GREEN

    def test_near_death_quartile_is_pink(self) -> None:
        """Below 25% — the wire damage tier 0 quartile — goes pink."""
        overlay = OverlayStateDict(**{**make_overlay(), "fuel": 260})

        payload = render_overlay_payload(overlay)

        assert payload["fuel_pct"] == 23
        assert payload["fuel_color"] == FIESTA_PINK

    def test_negative_fuel_clamps_to_zero(self) -> None:
        """A negative reading never renders a negative bar."""
        overlay = OverlayStateDict(**{**make_overlay(), "fuel": -50})

        assert render_overlay_payload(overlay)["fuel_pct"] == 0

    def test_non_positive_capacity_renders_empty_meter(self) -> None:
        """A zero capacity fixture renders 0% instead of dividing."""
        overlay = OverlayStateDict(**{**make_overlay(), "fuel_cap": 0})

        assert render_overlay_payload(overlay)["fuel_pct"] == 0


class TestStockSlots:
    """Inventory slot values + hunt-gate colors."""

    def test_slots_carry_counts_in_order(self) -> None:
        """AR/DU/MI/HO/RA map to s0..s4."""
        payload = render_overlay_payload(make_overlay())

        assert (
            payload["s0"],
            payload["s1"],
            payload["s2"],
            payload["s3"],
            payload["s4"],
        ) == (25, 25, 19, 25, 21)

    def test_slot_colors_follow_the_hunt_gate_bands(self) -> None:
        """At cap green, within 5 off-white, deeper shortfall pink."""
        payload = render_overlay_payload(make_overlay())

        assert payload["s0c"] == FIESTA_GREEN  # 25/25 full
        assert payload["s2c"] == FIESTA_PINK  # 19/25, 6 short
        assert payload["s4c"] == FIESTA_FG  # 21/25, within 5


class TestDecisionSlots:
    """Decision, sent indicator, reason, target, and stats slots."""

    def test_dispatched_command_renders_green_dot(self) -> None:
        """A sent command shows the target coords and a green dot."""
        payload = render_overlay_payload(make_overlay())

        assert payload["do_text"] == "shoot → (194,178)"
        assert payload["sent_text"] == "●"
        assert payload["sent_color"] == FIESTA_GREEN

    def test_held_command_renders_pink_cross(self) -> None:
        """A rejected dispatch renders loudly as a pink cross."""
        overlay = OverlayStateDict(**{**make_overlay(), "command_sent": False})

        payload = render_overlay_payload(overlay)

        assert payload["sent_text"] == "✕"
        assert payload["sent_color"] == FIESTA_PINK

    def test_reason_and_action_slots_carry_decision_context(self) -> None:
        """The why/act slots carry the behavior reason and in-flight kind."""
        payload = render_overlay_payload(make_overlay())

        assert payload["why_text"] == "HUNT: shoot red-8"
        assert payload["act_text"] == "none"
        assert payload["state_text"] == "IDLE"
        assert payload["pos_text"] == "88,112"

    def test_target_with_name_renders_name_and_id(self) -> None:
        """A named combat target renders ``name #id``."""
        assert render_overlay_payload(make_overlay())["tgt_text"] == "purple-4 #512"

    def test_target_without_name_renders_id_only(self) -> None:
        """An unnamed target renders just its wire id."""
        overlay = OverlayStateDict(**{**make_overlay(), "target_name": ""})

        assert render_overlay_payload(overlay)["tgt_text"] == "#512"

    def test_no_target_renders_em_dash(self) -> None:
        """No combat target renders the em-dash placeholder."""
        overlay = OverlayStateDict(**{**make_overlay(), "target_id": -1})

        assert render_overlay_payload(overlay)["tgt_text"] == "—"

    def test_session_stats_pass_through(self) -> None:
        """K/H/M/RJ slots carry the session counters verbatim."""
        payload = render_overlay_payload(make_overlay())

        assert (
            payload["kills"],
            payload["hits"],
            payload["misses"],
            payload["rejects"],
        ) == (3, 12, 4, 1)
