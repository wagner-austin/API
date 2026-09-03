"""The mine pin: one 3x3 press on the way into a close fight.

Operator order (2026-09-01, verbatim): "when we get or teleport
adjacent to an enemy we should be able to use mines to pin them in."
The press spends the tick a shot would have used, so the doctrine is
ONE press per engagement, latched by ``mine_pin_target_id``
([[mine-mechanics]] for the placement physics).
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import engage_target
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.mine_pin import MINE_PIN_REACH_TILES, mine_pin_decision
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._combat_fixtures import _enemy_threat
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world

_NOW = 100000


def _enemy_tank(x: int, y: int) -> TankStateDict:
    """A live, viewport-fresh enemy at the given tile (id 50)."""
    return make_tank_state(
        tank_id=50,
        x=x,
        y=y,
        team=2,
        rank=1,
        name="red-1",
        is_self=False,
        is_bot=False,
        damage_state=3,
        timestamp_ms=_NOW,
        last_wire_seen_ms=_NOW,
        last_position_update_ms=_NOW,
        last_viewport_observation_ms=_NOW,
    )


def _ctx(enemy_x: int, enemy_y: int, *, fuel: int = 800, pinned_id: int = -1) -> DecideCtx:
    """An engage-ready ctx with self at (100,100) and one enemy."""
    ws = WorldService()
    world, self_state = make_world(
        self_x=100, self_y=100, fuel=fuel, tanks={"50": _enemy_tank(enemy_x, enemy_y)}
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "HUNT",
            "mode_state": "ENGAGE",
            "mode_started_ms": 90000,
            "combat_target_id": 50,
            "combat_target_x": enemy_x,
            "combat_target_y": enemy_y,
            "mine_pin_target_id": pinned_id,
        }
    )
    return DecideCtx(world, self_state, ai_state, make_inventory(), _NOW, None, "", ws=ws)


def test_first_close_engage_tick_presses_the_pin() -> None:
    """Reach 2 with a fresh latch: the tick is the press, and it latches."""
    ctx = _ctx(102, 100)
    target = _enemy_threat(x=102, y=100, name="red-1")

    decision = engage_target(ctx, target)

    assert decision["command"]["cmd_type"] == "mine_drop"
    assert decision["behavior"]["reason_kind"] == "mine_pin"
    assert decision["updated_ai_state"]["mine_pin_target_id"] == 50
    # The lock survives the press: the next tick fights on.
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_latched_engagement_shoots_instead_of_pressing_again() -> None:
    """The same target never draws a second press."""
    ctx = _ctx(102, 100, pinned_id=50)
    target = _enemy_threat(x=102, y=100, name="red-1")

    decision = engage_target(ctx, target)

    assert decision["command"]["cmd_type"] == "shoot"


def test_beyond_pressing_reach_the_tick_stays_a_shot() -> None:
    """One tile past the reach the 3x3 misses the ring: no press."""
    ctx = _ctx(100 + MINE_PIN_REACH_TILES + 1, 100)
    target = _enemy_threat(x=100 + MINE_PIN_REACH_TILES + 1, y=100, name="red-1")

    decision = engage_target(ctx, target)

    assert decision["command"]["cmd_type"] == "shoot"


def test_survival_floor_bars_the_press() -> None:
    """At the fuel-low break plus the press cost, the tick is not spent."""
    probe = _ctx(102, 100, fuel=800)
    ctx = _ctx(102, 100, fuel=probe.fuel_low_floor + 10)
    target = _enemy_threat(x=102, y=100, name="red-1")

    assert mine_pin_decision(ctx, target) is None


def test_a_new_target_rearms_the_press() -> None:
    """The latch is per target id: a fresh enemy gets its own press."""
    ctx = _ctx(102, 100, pinned_id=999)
    target = _enemy_threat(x=102, y=100, name="red-1")

    decision = mine_pin_decision(ctx, target)

    if decision is None:
        raise AssertionError("expected the fresh target to draw the press")
    assert decision["command"]["cmd_type"] == "mine_drop"
