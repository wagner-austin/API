"""Tests for the mine-clearance same-tile re-aim latch + cascade wiring.

Closes the coverage the latch shipped without (audit 2026-07-30): the
double-shot at (162,94) two seconds apart was fixed by
``mine_clearance_aim_key``/``mine_clearance_shot_ms``, but the
consumer branch and the cascade's clearance step had no pins.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.collect_pickups import mine_clearance_decision
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_container_state, make_mine_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _ctx_with_covered_container(ai_state: AIStateDict) -> DecideCtx:
    """World with one hostile-mine-covered container in view."""
    ws = WorldService()
    world, self_state = make_world(self_x=100, self_y=100, fuel=1050)
    world["containers"]["104,100"] = make_container_state(
        x=104,
        y=100,
        is_fuel=False,
        volume=0,
    )
    # Self is team 1 in make_world; team 2 makes the mine hostile.
    world["mines"]["104,100"] = make_mine_state(x=104, y=100, mine_type=0, tank_id=-1, team=2)
    return DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )


def _collect_state(
    *,
    mine_clearance_aim_key: str = "",
    mine_clearance_shot_ms: int = 0,
) -> AIStateDict:
    """COLLECT-mode AI state with optional latch overrides.

    Args:
        mine_clearance_aim_key: Last clearance aim tile key.
        mine_clearance_shot_ms: Last clearance shot timestamp.

    Returns:
        AI state in COLLECT/SEARCH with the latch fields applied.
    """
    return AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "mine_clearance_aim_key": mine_clearance_aim_key,
            "mine_clearance_shot_ms": mine_clearance_shot_ms,
        }
    )


def test_same_tile_reaim_inside_the_effect_window_is_refused() -> None:
    """A fresh latch on the exact aim tile suppresses the re-shot.

    The live double-shot (run bot-20260730-015x, (162,94) at
    01:59:57/:59): the 0x45 detonation had not been applied when the
    next tick re-derived the same aim.
    """
    state = _collect_state(mine_clearance_aim_key="104,100", mine_clearance_shot_ms=98000)
    ctx = _ctx_with_covered_container(state)

    assert mine_clearance_decision(ctx, ctx.base) is None


def test_stale_latch_allows_the_reclear() -> None:
    """Past the effect window the same tile may be cleared again.

    A recruit's 1-mine blast can leave covered neighbors, so a genuine
    re-clear must not be latched out forever.
    """
    state = _collect_state(mine_clearance_aim_key="104,100", mine_clearance_shot_ms=90000)
    ctx = _ctx_with_covered_container(state)

    decision = mine_clearance_decision(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected a clearance shot after the window")
    assert decision["command"]["cmd_type"] == "shoot"
    updated = decision["updated_ai_state"]
    assert updated["mine_clearance_aim_key"] == "104,100"
    assert updated["mine_clearance_shot_ms"] == 100000


def test_cascade_dispatches_the_clearance_shot() -> None:
    """decide_collect_mode routes through the clearance step.

    Pins the cascade wiring line (clearance before larder) that the
    planner-level tests bypass.
    """
    ctx = _ctx_with_covered_container(_collect_state())

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected a collect decision")
    assert decision["behavior"]["reason_kind"] == "mine_clearance_shot"
    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["command"]["target_x"] == 104
    assert decision["command"]["target_y"] == 100
