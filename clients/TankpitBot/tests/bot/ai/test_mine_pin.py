"""The mine pin: one 3x3 press on the way into a close fight.

Operator order (2026-09-01, verbatim): "when we get or teleport
adjacent to an enemy we should be able to use mines to pin them in."
The press spends the tick a shot would have used, so the doctrine is
ONE press per engagement, recorded in the per-target
``mine_pin_presses`` map ([[mine-mechanics]] for the placement
physics; flag-triage-20260902 row 7 for why the map replaced the
scalar latch).
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


def _ctx(
    enemy_x: int,
    enemy_y: int,
    *,
    fuel: int = 800,
    presses: dict[str, str] | None = None,
) -> DecideCtx:
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
            "mine_pin_presses": {} if presses is None else presses,
        }
    )
    return DecideCtx(world, self_state, ai_state, make_inventory(), _NOW, None, "", ws=ws)


def test_first_close_engage_tick_presses_the_pin() -> None:
    """Reach 2 with no press history: the tick is the press, recorded."""
    ctx = _ctx(102, 100)
    target = _enemy_threat(x=102, y=100, name="red-1")

    decision = engage_target(ctx, target)

    assert decision["command"]["cmd_type"] == "mine_drop"
    assert decision["behavior"]["reason_kind"] == "mine_pin"
    assert decision["updated_ai_state"]["mine_pin_presses"] == {"50": "100,100"}
    # The lock survives the press: the next tick fights on.
    assert decision["updated_ai_state"]["combat_target_id"] == 50


def test_a_pressed_target_shoots_instead_of_pressing_again() -> None:
    """The same target never draws a second press."""
    ctx = _ctx(102, 100, presses={"50": "98,98"})
    target = _enemy_threat(x=102, y=100, name="red-1")

    decision = engage_target(ctx, target)

    assert decision["command"]["cmd_type"] == "shoot"


def test_an_intervening_target_does_not_rearm_the_press() -> None:
    """The A->B->A shuttle: B's press must not forget A's.

    The scalar latch this map replaced re-armed on every lock move,
    and the 2026-09-01 shuttle bought four presses on two tiles
    (flag-triage-20260902 row 7). With B (id 77) pressed AFTER A
    (id 50), re-engaging A finds its own entry intact.
    """
    ctx = _ctx(102, 100, presses={"50": "98,98", "77": "97,97"})
    target = _enemy_threat(x=102, y=100, name="red-1")

    assert mine_pin_decision(ctx, target) is None


def test_a_press_keeps_every_earlier_entry() -> None:
    """A new target's press extends the map, never replaces it."""
    ctx = _ctx(102, 100, presses={"999": "90,90"})
    target = _enemy_threat(x=102, y=100, name="red-1")

    decision = mine_pin_decision(ctx, target)

    if decision is None:
        raise AssertionError("expected the fresh target to draw the press")
    assert decision["updated_ai_state"]["mine_pin_presses"] == {
        "999": "90,90",
        "50": "100,100",
    }


def test_already_pressed_ground_is_never_re_pressed() -> None:
    """A fresh target on ground pressed before draws no second press.

    Self stands on (100,100) and SOME earlier press was laid from
    exactly there: the identical 3x3 pattern buys no new ground, so
    the tick stays a shot even though target id 50 has no entry.
    """
    ctx = _ctx(102, 100, presses={"999": "100,100"})
    target = _enemy_threat(x=102, y=100, name="red-1")

    assert mine_pin_decision(ctx, target) is None


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
