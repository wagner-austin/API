"""Tests for the durable AI state record."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_default_ai_config,
    make_initial_ai_state,
)
from tankpit_bot.bot.ai.types_codecs import (
    decode_ai_state,
    encode_ai_config,
    encode_ai_state,
)


class TestAIStateDetail:
    """Tests for the durable AI state record."""

    def test_decode_killed_tank_ids_non_int_value_raises(self) -> None:
        """Decode rejects killed_tank_ids with non-int values."""
        config = encode_ai_config(make_default_ai_config())
        data: JSONObject = {
            "config": config,
            "mode": "UNSET",
            "mode_state": "",
            "mode_started_ms": 0,
            "last_scan_ms": 0,
            "last_shoot_ms": 0,
            "last_map_open_ms": 0,
            "combat_target_id": -1,
            "wind_down": False,
            "combat_target_x": 0,
            "combat_target_y": 0,
            "killed_tank_ids": {"50": "not_an_int"},
            "break_escape_until_fuel": 0,
            "blocked_combat_targets": {},
            "last_shot_target_id": -1,
            "last_shot_target_name": "",
        }
        with pytest.raises(ValueError, match="must be int"):
            decode_ai_state(data)

    def test_encode_decode_roundtrip_with_killed_tanks(self) -> None:
        """Encode then decode preserves killed_tank_ids."""
        from tankpit_bot.bot.ai.types import AIStateDict

        original = make_initial_ai_state()
        state = AIStateDict(
            **{
                **original,
                "killed_tank_ids": {"50": 10000, "60": 20000},
                "last_shot_target_id": 50,
                "last_shot_target_name": "Enemy",
            }
        )
        encoded = encode_ai_state(state)
        decoded = decode_ai_state(encoded)
        assert decoded == state

    def test_encode_decode_roundtrip_with_manual_mode_hunt(self) -> None:
        """Encode/decode preserves ``manual_mode = HUNT``."""
        from tankpit_bot.bot.ai.types import AIStateDict

        original = make_initial_ai_state()
        state = AIStateDict(**{**original, "manual_mode": "HUNT"})
        encoded = encode_ai_state(state)
        decoded = decode_ai_state(encoded)
        assert decoded["manual_mode"] == "HUNT"
        assert decoded == state

    def test_encode_decode_roundtrip_with_manual_mode_collect(self) -> None:
        """Encode/decode preserves ``manual_mode = COLLECT``."""
        from tankpit_bot.bot.ai.types import AIStateDict

        original = make_initial_ai_state()
        state = AIStateDict(**{**original, "manual_mode": "COLLECT"})
        encoded = encode_ai_state(state)
        decoded = decode_ai_state(encoded)
        assert decoded["manual_mode"] == "COLLECT"

    def test_encode_decode_roundtrip_with_manual_mode_unset(self) -> None:
        """Encode/decode preserves ``manual_mode = UNSET`` (idle-pin)."""
        from tankpit_bot.bot.ai.types import AIStateDict

        original = make_initial_ai_state()
        state = AIStateDict(**{**original, "manual_mode": "UNSET"})
        encoded = encode_ai_state(state)
        decoded = decode_ai_state(encoded)
        assert decoded["manual_mode"] == "UNSET"

    def test_encode_decode_roundtrip_with_live_counters(self) -> None:
        """Encode/decode preserves ``live_radars_used`` / ``live_teleports``."""
        from tankpit_bot.bot.ai.types import AIStateDict

        original = make_initial_ai_state()
        state = AIStateDict(**{**original, "live_radars_used": 17, "live_teleports": 42})
        encoded = encode_ai_state(state)
        decoded = decode_ai_state(encoded)
        assert decoded["live_radars_used"] == 17
        assert decoded["live_teleports"] == 42

    def test_encode_decode_roundtrip_with_greet_and_visit_maps(self) -> None:
        """Encode/decode preserves the per-id HELLO and visit maps."""
        from tankpit_bot.bot.ai.types import AIStateDict

        original = make_initial_ai_state()
        assert original["greeted_tank_ids"] == {}
        assert original["visited_tank_ids"] == {}
        state = AIStateDict(
            **{
                **original,
                "greeted_tank_ids": {"1229": 100000, "31": 105000},
                "visited_tank_ids": {"1229": 101000},
            }
        )
        encoded = encode_ai_state(state)
        decoded = decode_ai_state(encoded)
        assert decoded["greeted_tank_ids"] == {"1229": 100000, "31": 105000}
        assert decoded["visited_tank_ids"] == {"1229": 101000}

    def test_encode_decode_roundtrip_with_maroon_pan_latch(self) -> None:
        """Encode/decode preserves the marooned-pan movement-law latch."""
        from tankpit_bot.bot.ai.types import AIStateDict

        original = make_initial_ai_state()
        assert original["maroon_pan_x"] == -1
        assert original["maroon_pan_y"] == -1
        state = AIStateDict(**{**original, "maroon_pan_x": 113, "maroon_pan_y": 221})
        encoded = encode_ai_state(state)
        decoded = decode_ai_state(encoded)
        assert decoded["maroon_pan_x"] == 113
        assert decoded["maroon_pan_y"] == 221

    def test_decode_missing_greeted_tank_ids_raises(self) -> None:
        """Missing ``greeted_tank_ids`` raises — no back-compat default."""
        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        del encoded["greeted_tank_ids"]
        with pytest.raises(ValueError, match="greeted_tank_ids"):
            decode_ai_state(encoded)

    def test_encode_decode_roundtrip_with_mine_pin_presses(self) -> None:
        """Encode/decode preserves the per-target mine-pin press map."""
        from tankpit_bot.bot.ai.types import AIStateDict

        original = make_initial_ai_state()
        assert original["mine_pin_presses"] == {}
        state = AIStateDict(**{**original, "mine_pin_presses": {"50": "210,57", "77": "199,42"}})
        encoded = encode_ai_state(state)
        decoded = decode_ai_state(encoded)
        assert decoded["mine_pin_presses"] == {"50": "210,57", "77": "199,42"}

    def test_decode_missing_mine_pin_presses_raises(self) -> None:
        """Missing ``mine_pin_presses`` raises — no back-compat default."""
        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        del encoded["mine_pin_presses"]
        with pytest.raises(ValueError, match="mine_pin_presses"):
            decode_ai_state(encoded)

    def test_decode_non_object_mine_pin_presses_raises(self) -> None:
        """A non-object ``mine_pin_presses`` field raises."""
        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        encoded["mine_pin_presses"] = "50:210,57"
        with pytest.raises(ValueError, match="mine_pin_presses must be an object"):
            decode_ai_state(encoded)

    def test_decode_mine_pin_presses_non_str_value_raises(self) -> None:
        """Decode rejects mine_pin_presses with non-str tile values."""
        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        encoded["mine_pin_presses"] = {"50": 21057}
        with pytest.raises(ValueError, match="mine_pin_presses values must be str"):
            decode_ai_state(encoded)

    def test_decode_missing_manual_mode_raises(self) -> None:
        """Missing ``manual_mode`` raises — no back-compat default."""
        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        del encoded["manual_mode"]
        with pytest.raises(KeyError, match="manual_mode"):
            decode_ai_state(encoded)

    def test_decode_null_manual_mode_becomes_none(self) -> None:
        """A serialised null decodes as None (auto-arbitration)."""
        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        encoded["manual_mode"] = None
        decoded = decode_ai_state(encoded)
        assert decoded["manual_mode"] is None

    def test_decode_invalid_manual_mode_raises(self) -> None:
        """An unknown ``manual_mode`` string surfaces as ValueError."""
        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        encoded["manual_mode"] = "PATROL"
        with pytest.raises(ValueError, match="manual_mode must be one of"):
            decode_ai_state(encoded)

    def test_decode_non_string_manual_mode_raises(self) -> None:
        """A non-string, non-null ``manual_mode`` surfaces as JSONTypeError."""

        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        encoded["manual_mode"] = 7
        with pytest.raises(JSONTypeError):
            decode_ai_state(encoded)

    def test_decode_missing_live_radars_used_raises(self) -> None:
        """Missing ``live_radars_used`` raises — no back-compat default."""

        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        del encoded["live_radars_used"]
        with pytest.raises(JSONTypeError):
            decode_ai_state(encoded)

    def test_decode_missing_live_teleports_raises(self) -> None:
        """Missing ``live_teleports`` raises — no back-compat default."""

        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        del encoded["live_teleports"]
        with pytest.raises(JSONTypeError):
            decode_ai_state(encoded)

    def test_decode_missing_held_ticks_raises(self) -> None:
        """Missing ``resource_target_held_ticks`` raises — no default.

        The stall counter behind the 2026-09-02 progress invariant is
        part of the lock; a record without it is corruption, not an
        old version to soften for.
        """

        original = make_initial_ai_state()
        encoded = encode_ai_state(original)
        del encoded["resource_target_held_ticks"]
        with pytest.raises(JSONTypeError):
            decode_ai_state(encoded)

    def test_held_ticks_round_trip(self) -> None:
        """A non-zero stall count survives encode -> decode exactly."""
        original = make_initial_ai_state()
        counted = AIStateDict(**{**original, "resource_target_held_ticks": 5})

        encoded = encode_ai_state(counted)

        assert decode_ai_state(encoded)["resource_target_held_ticks"] == 5
