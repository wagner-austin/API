"""Tests for shared resource-search helpers."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.resource_search import _short_hop_fallback
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def test_short_hop_fallback_skips_clamped_directions_near_corner() -> None:
    """At a map corner, some short-hop directions clamp below minimum displacement."""
    world, self_state = make_world(self_x=1, self_y=1, fuel=300)
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), make_inventory(), 100000, None, "")

    result = _short_hop_fallback(ctx)

    if result is not None:
        tx, ty = result
        assert abs(tx - 1) + abs(ty - 1) >= 4


def test_short_hop_fallback_returns_none_when_fuel_too_low() -> None:
    """Near a corner with low fuel: clamped directions are too close, others unaffordable."""
    world, self_state = make_world(self_x=2, self_y=2, fuel=120)
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), make_inventory(), 100000, None, "")

    assert _short_hop_fallback(ctx) is None
