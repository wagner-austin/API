"""The lock continuation's progress bound: held plans cannot spin forever.

The 2026-09-02 livelock ([[flag-triage-20260902]]) held one fuel lock
for nine minutes with no dispatch and no release — every enumerated
release said "hold" and nothing counted the holds. These tests pin
the invariant: a transient hold advances a counter, a dispatch or
fresh latch resets it, and at
:data:`~tankpit_bot.bot.ai.intent.RESOURCE_LOCK_HOLD_BOUND_TICKS`
the continuation releases the plan with its own enumerated reason.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_locks import continue_or_release_lock
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.intent import RESOURCE_LOCK_HOLD_BOUND_TICKS, set_resource_target
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.bot.ai.test_collect_pocket_serving import _rock_ring
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _held_ctx(*, kind: str, held_ticks: int, is_fuel: bool) -> DecideCtx:
    """Build a context whose lock is held-but-not-executable.

    The 2026-09-01 pocket shape: the locked container sits in-viewport
    at (104,100) behind a rock ring, so the pickup dispatch has no
    walk route (hold), while the container tile itself stays a legal
    teleport landing (NOT unservable — the hold is the committed-
    intent law working).

    Args:
        kind: Lock kind, ``"fuel"`` or ``"equipment"``.
        held_ticks: Pre-existing consecutive holds on the lock.
        is_fuel: Whether the locked container is fuel.

    Returns:
        Context whose continuation will hold (or stall) this tick.
    """
    containers = {
        "104,100": make_container_state(
            x=104,
            y=100,
            is_fuel=is_fuel,
            volume=700 if is_fuel else 0,
            timestamp_ms=100000,
            failed_pickups=0,
        )
    }
    world, self_state = make_world(fuel=700, containers=containers)
    locked = set_resource_target(make_scanned_ai_state(), kind, 104, 100)
    state = AIStateDict(**{**locked, "resource_target_held_ticks": held_ticks})
    return DecideCtx(
        world,
        self_state,
        state,
        make_inventory(default_count=15),
        100000,
        InMemoryTerrainMap(_rock_ring(104, 100)),
        "",
        ws=WorldService(),
    )


def test_a_fuel_hold_advances_the_counter_and_keeps_the_plan() -> None:
    """One dispatch-less tick is a legitimate transient: hold, count it."""
    ctx = _held_ctx(kind="fuel", held_ticks=0, is_fuel=True)

    decision, state = continue_or_release_lock(ctx, ctx.base)

    assert decision is None
    assert state["resource_target_kind"] == "fuel"
    assert state["resource_target_held_ticks"] == 1


def test_an_equipment_hold_advances_the_counter_and_keeps_the_plan() -> None:
    """The equipment continuation carries the same invariant."""
    ctx = _held_ctx(kind="equipment", held_ticks=0, is_fuel=False)

    decision, state = continue_or_release_lock(ctx, ctx.base)

    assert decision is None
    assert state["resource_target_kind"] == "equipment"
    assert state["resource_target_held_ticks"] == 1


def test_the_hold_below_the_bound_still_holds() -> None:
    """The last legitimate hold: one tick short of the bound."""
    ctx = _held_ctx(
        kind="fuel",
        held_ticks=RESOURCE_LOCK_HOLD_BOUND_TICKS - 2,
        is_fuel=True,
    )

    decision, state = continue_or_release_lock(ctx, ctx.base)

    assert decision is None
    assert state["resource_target_kind"] == "fuel"
    assert state["resource_target_held_ticks"] == RESOURCE_LOCK_HOLD_BOUND_TICKS - 1


def test_the_hold_at_the_bound_releases_progress_stalled() -> None:
    """The bound converts an unknown hold-forever shape into a release.

    Nine minutes of spin in the live run; sixteen seconds and a
    ``plan_released``/``progress_stalled`` diagnostic now.
    """
    ctx = _held_ctx(
        kind="fuel",
        held_ticks=RESOURCE_LOCK_HOLD_BOUND_TICKS - 1,
        is_fuel=True,
    )

    decision, state = continue_or_release_lock(ctx, ctx.base)

    assert decision is None
    assert state["resource_target_kind"] == ""
    assert state["resource_target_held_ticks"] == 0
