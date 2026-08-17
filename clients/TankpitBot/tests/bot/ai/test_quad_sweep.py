"""Quad-sweep recon ([[quad-sweep-doctrine]]); harvest pins live in
``test_block_harvest``."""

from __future__ import annotations

from tankpit_bot.bot.ai.quad_sweep import (
    plan_quad_sweep,
    quadrant_bounds,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.protocol.commands import (
    SCOPE_NORTHWEST,
    SCOPE_SOUTHEAST,
)
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_sweep_ctx,
)

_NOW = 100000


def _anchored_ai_state(x: int, y: int) -> AIStateDict:
    return AIStateDict(**{**make_scanned_ai_state(), "sweep_anchor_x": x, "sweep_anchor_y": y})


def _cover_window(world_tiles: dict[str, int], left: int, top: int) -> None:
    for y in range(top, top + 16):
        for x in range(left, left + 16):
            world_tiles[f"{x},{y}"] = _NOW


def test_quadrant_bounds_follow_the_anchor_law_and_clamp() -> None:
    """Quadrant origins are tank-anchored and map-clamped like the server's."""
    assert quadrant_bounds(100, 100, -15, -15) == (85, 85, 100, 100)
    assert quadrant_bounds(100, 100, 0, 0) == (100, 100, 115, 115)
    # A corner anchor clamps both quadrant origins onto the same window.
    assert quadrant_bounds(3, 3, -15, -15) == (0, 0, 15, 15)
    assert quadrant_bounds(252, 252, 0, 0) == (240, 240, 255, 255)


def test_sweep_declines_without_extras_and_at_low_fuel() -> None:
    """No extras or fuel at the low break: recon never wins the tick."""
    empty = make_inventory()
    empty["extra_radars"]["count"] = 0
    assert (
        plan_quad_sweep(make_sweep_ctx(now_ms=_NOW, inventory=empty), make_scanned_ai_state())
        is None
    )

    low_fuel = make_sweep_ctx(now_ms=_NOW, fuel=100)
    assert plan_quad_sweep(low_fuel, make_scanned_ai_state()) is None


def test_sweep_declines_on_a_covered_block() -> None:
    """A block below the start floor never begins a sweep."""
    ctx = make_sweep_ctx(now_ms=_NOW, scanned=True)
    assert plan_quad_sweep(ctx, make_scanned_ai_state()) is None


def test_virgin_block_opens_with_a_steering_shift() -> None:
    """A fresh block starts by steering toward the first quadrant.

    The centered landing window matches no quadrant's anchored
    bounds, so the sweep's first act is the NW steer. An earlier cut
    radared the CURRENT window here whenever it cleared the bare
    spend floor; that branch bought 32-tile scraps of already-covered
    windows live (2026-08-13 HUD flags 4/5) and is gone -- only
    quadrant-framed windows draw sweep radars now.
    """
    ctx = make_sweep_ctx(now_ms=_NOW, scanned=False)

    decision = plan_quad_sweep(ctx, make_scanned_ai_state())

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "quad_sweep_shift"
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["command"]["direction"] == SCOPE_NORTHWEST
    updated = decision["updated_ai_state"]
    assert updated["sweep_anchor_x"] == 100
    assert updated["sweep_anchor_y"] == 100


def test_covered_window_steers_toward_the_first_pending_quadrant() -> None:
    """With the current window spent, the sweep shifts NW first."""
    ctx = make_sweep_ctx(now_ms=_NOW, scanned=True, block_scanned=False)

    decision = plan_quad_sweep(ctx, make_scanned_ai_state())

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "quad_sweep_shift"
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["command"]["direction"] == SCOPE_NORTHWEST
    assert decision["behavior"]["reason_context"]["direction"] == SCOPE_NORTHWEST


def test_framed_quadrant_fires_its_radar() -> None:
    """A window parked exactly on a pending quadrant draws the radar."""
    ctx = make_sweep_ctx(now_ms=_NOW, scanned=True, block_scanned=False, viewport_origin=(85, 85))
    ctx.world["scanned_tiles"].clear()
    _cover_window(ctx.world["scanned_tiles"], 92, 92)

    decision = plan_quad_sweep(ctx, _anchored_ai_state(100, 100))

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "quad_sweep_radar"
    assert decision["behavior"]["reason_context"]["direction"] == SCOPE_NORTHWEST


def test_moved_tank_abandons_the_sweep_until_the_block_is_fresh() -> None:
    """A stale anchor in a mostly-covered block never resumes the sweep.

    Same coverage as the continuation test below (one quadrant still
    fresh, block under the start floor) -- but with the anchor
    latched to a tile the tank no longer stands on, the sweep is
    abandoned rather than continued.
    """
    ctx = make_sweep_ctx(now_ms=_NOW, scanned=True, block_scanned=False)
    for y in range(85, 116):
        for x in range(85, 116):
            if not (x >= 100 and y >= 100):
                ctx.world["scanned_tiles"][f"{x},{y}"] = _NOW

    decision = plan_quad_sweep(ctx, _anchored_ai_state(90, 100))

    assert decision is None


def test_anchored_sweep_continues_below_the_start_floor() -> None:
    """Standing on the anchor, per-quadrant economics alone continue it."""
    ctx = make_sweep_ctx(now_ms=_NOW, scanned=True, block_scanned=False)
    # Cover most of the block so the START floor would refuse, leaving
    # one quadrant's worth of fresh ground.
    for y in range(85, 116):
        for x in range(85, 116):
            if not (x >= 100 and y >= 100):
                ctx.world["scanned_tiles"][f"{x},{y}"] = _NOW

    decision = plan_quad_sweep(ctx, _anchored_ai_state(100, 100))

    if decision is None:
        raise AssertionError("expected a decision")
    assert decision["behavior"]["reason_kind"] == "quad_sweep_shift"
    assert decision["command"]["cmd_type"] == "scope_shift"
    assert decision["command"]["direction"] == SCOPE_SOUTHEAST


def test_last_extra_needs_the_reserve_floor() -> None:
    """At the reserve, a quadrant must clear the bigger reveal floor."""
    inventory = make_inventory()
    inventory["extra_radars"]["count"] = 1
    ctx = make_sweep_ctx(now_ms=_NOW, scanned=True, block_scanned=False, inventory=inventory)
    # Leave under 128 uncovered tiles per quadrant: cover all but a
    # 7-row strip of the block.
    for y in range(85, 116):
        for x in range(85, 116):
            if y >= 92:
                ctx.world["scanned_tiles"][f"{x},{y}"] = _NOW

    assert plan_quad_sweep(ctx, _anchored_ai_state(100, 100)) is None


def test_zero_extras_never_qualify_a_quadrant() -> None:
    """The sweep is an extras strategy: zero extras refuse any spend."""
    from tankpit_bot.bot.ai.quad_sweep import _quadrant_spend_worthwhile

    assert _quadrant_spend_worthwhile(961, 0) is False
