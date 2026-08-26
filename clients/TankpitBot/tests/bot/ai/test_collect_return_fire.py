"""Return fire from COLLECT-owned ticks (operator ruling 2026-08-26).

The Yuppler receipt (run bot/arterial 03:14): ten human shots landed
while the between-kills restock out-collected the damage, and the
first return shot waited 37 s for the tank to touch cap. Ticks under
fire are DAMAGE ticks — the divert fires from where the tank stands
and the refill rides as the secondary command. The 2026-07-25
survival contract stays senior via the stock bars.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.combat_opportunity import collect_return_fire
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIConfigDict, AIStateDict
from tankpit_bot.fleetshare.types import FleetRole
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import ContainerStateDict, TankStateDict, make_tank_state
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
    seed_confirmed_incoming,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOW = 100000
_ATTACKER_ID = 60


def _attacker(x: int, y: int) -> TankStateDict:
    """The seeded shooter (id 60, the ``seed_confirmed_incoming`` id)."""
    return make_tank_state(
        tank_id=_ATTACKER_ID,
        x=x,
        y=y,
        team=2,
        rank=1,
        name="ganker",
        is_self=False,
        is_bot=False,
        damage_state=3,
        timestamp_ms=_NOW,
        last_wire_seen_ms=_NOW,
        last_position_update_ms=_NOW,
        last_viewport_observation_ms=_NOW,
    )


def _ctx(
    *,
    fuel: int = 800,
    tanks: dict[str, TankStateDict] | None = None,
    containers: dict[str, ContainerStateDict] | None = None,
    dual_count: int = 30,
    role: FleetRole | None = None,
    hits: int = 3,
) -> DecideCtx:
    """Build a COLLECT-owned ctx with the seeded attacker's fire landed."""
    ws = WorldService()
    if hits:
        seed_confirmed_incoming(ws, hits)
    world, self_state = make_world(fuel=fuel, tanks=tanks, containers=containers)
    base = make_scanned_ai_state()
    if role is not None:
        base = AIStateDict(**{**base, "config": AIConfigDict(**{**base["config"], "role": role})})
    ai_state = AIStateDict(
        **{
            **base,
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    return DecideCtx(
        world,
        self_state,
        ai_state,
        make_inventory(dual_count=dual_count),
        _NOW,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )


def test_stocked_collect_tick_returns_fire_at_the_visible_attacker() -> None:
    """One confirmed hit from a tank in view flips the tick to a shot.

    The full cascade path: the return-fire rung preempts the escape
    handler, so the between-kills restock answers the first confirmed
    hit instead of out-collecting the damage in silence.
    """
    ctx = _ctx(tanks={str(_ATTACKER_ID): _attacker(103, 100)})

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected the return-fire decision")
    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["command"]["target_id"] == _ATTACKER_ID
    assert decision["behavior"]["reason_kind"] == "opportunity_shot"
    assert decision["updated_ai_state"]["last_shot_target_id"] == _ATTACKER_ID


def test_return_fire_carries_the_refill_as_its_secondary() -> None:
    """The adjacent-container pickup rides the shot — no refill tick lost."""
    ctx = _ctx(
        tanks={str(_ATTACKER_ID): _attacker(103, 100)},
        containers={"101,100": make_container(101, 100, 400, is_fuel=True)},
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected the return-fire decision")
    assert decision["command"]["cmd_type"] == "shoot"
    secondary = decision["secondary_command"]
    if secondary is None:
        raise AssertionError("expected the refill secondary on the return-fire shot")
    assert secondary["cmd_type"] == "pickup_fuel"


def test_fuel_at_the_low_break_leaves_the_escape_doctrine_in_charge() -> None:
    """At the fuel-low bar survival stays senior: no return fire."""
    ctx = _ctx(tanks={str(_ATTACKER_ID): _attacker(103, 100)}, fuel=200)

    assert ctx.fuel <= ctx.config["fuel_low_threshold"]
    assert collect_return_fire(ctx, ctx.base) is None


def test_weapon_break_leaves_the_escape_doctrine_in_charge() -> None:
    """A weapon reserve below its break bar vetoes the return shot."""
    ctx = _ctx(tanks={str(_ATTACKER_ID): _attacker(103, 100)}, dual_count=3)

    assert collect_return_fire(ctx, ctx.base) is None


def test_gatherer_role_never_returns_fire() -> None:
    """The fleet role gate holds: a gatherer's ticks never shoot."""
    ctx = _ctx(tanks={str(_ATTACKER_ID): _attacker(103, 100)}, role="gatherer")

    assert collect_return_fire(ctx, ctx.base) is None


def test_no_recent_attacker_means_no_return_fire() -> None:
    """Without a confirmed hit in the window the rung declines."""
    ctx = _ctx(tanks={str(_ATTACKER_ID): _attacker(103, 100)}, hits=0)

    assert collect_return_fire(ctx, ctx.base) is None


def test_unseen_attacker_cannot_draw_the_shot() -> None:
    """Hits from a tank with no registry entry leave the tick to escape.

    The divert legality is inherited wholesale: no viewport
    confirmation, no shot — the escape verbs keep the tick exactly as
    before the rung existed.
    """
    ctx = _ctx(tanks=None)

    assert collect_return_fire(ctx, ctx.base) is None


def _with_lock(ctx: DecideCtx, lock_id: int) -> DecideCtx:
    """Rebuild the ctx holding a combat lock (a break-restock tick)."""
    locked = AIStateDict(**{**ctx.ai_state, "combat_target_id": lock_id})
    return DecideCtx(
        ctx.world,
        ctx.self_state,
        locked,
        ctx.inventory,
        ctx.timestamp_ms,
        ctx.terrain,
        ctx.combat_feedback,
        ws=ctx.ws,
    )


def test_break_restock_never_re_fights_the_broken_from_lock() -> None:
    """The held lock's own tank draws no return fire during its break.

    The first live hour proved the inverse (artax vs red-8, 03:50:16):
    the solvency break walked away at projected fuel 318 < floor 354
    and the rung shot the same tank six times anyway, fuel 851→686.
    The broken-from enemy belongs to the resume machinery.
    """
    ctx = _with_lock(_ctx(tanks={str(_ATTACKER_ID): _attacker(103, 100)}), _ATTACKER_ID)

    assert collect_return_fire(ctx, ctx.base) is None


def test_a_second_attacker_still_draws_fire_during_a_break_restock() -> None:
    """Only the lock is excluded: another consented attacker gets shot."""
    ctx = _with_lock(_ctx(tanks={str(_ATTACKER_ID): _attacker(103, 100)}), 999)

    decision = collect_return_fire(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected return fire at the non-lock attacker")
    assert decision["command"]["cmd_type"] == "shoot"
    assert decision["command"]["target_id"] == _ATTACKER_ID
