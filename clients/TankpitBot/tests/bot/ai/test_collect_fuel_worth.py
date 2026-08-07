"""Fuel pickup worth-the-walk economics and the walkworthy iteration."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_pickups import (
    _first_walkworthy_fuel,
    pickup_not_worth_walk,
    select_and_pickup_fuel,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import SelfStateDict, make_container_state
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
)


def test_pickup_refused_when_clamped_gain_not_worth_the_walk() -> None:
    """A distant cap-clamped sliver is refused: the 2026-07-06 waste class.

    Private (rank 1, cap 1100) at fuel 1096: headroom 4, so a
    386-volume container transfers only 4. At a 2-tile walk the
    threshold is ``3 * 2 = 6 > 4`` -- not worth it, refuse. (The
    per-tile price is the MEASURED-walking-speed derivation of
    2026-08-06, not the falsified one-tick-per-tile premise.)
    """

    base_world, base_self = make_world(fuel=1096)
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")
    container = make_container_state(
        x=102,
        y=100,
        is_fuel=True,
        volume=386,
        timestamp_ms=100000,
        failed_pickups=0,
    )

    assert pickup_not_worth_walk(ctx, container) is True


def test_adjacent_clamped_sliver_is_worth_taking() -> None:
    """The same sliver one tile away IS taken: 46 >= 25 * 1.

    Under the old binary gate this exact geometry (the 2026-07-06
    canonical shape) was refused; with code=5 handled cleanly, +46
    fuel for one 2-second walk tile is the same rate a good dot hop
    pays, so the pickup is worth it.
    """

    base_world, base_self = make_world(fuel=1054)
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")
    container = make_container_state(
        x=101,
        y=100,
        is_fuel=True,
        volume=386,
        timestamp_ms=100000,
        failed_pickups=0,
    )

    assert pickup_not_worth_walk(ctx, container) is False


def test_big_clamped_container_is_worth_a_long_walk() -> None:
    """Fuel 600 vs a 1000-volume container: 500 effective gain wins.

    The falsifying case for the old binary gate (2026-07-19): because
    ``volume >= headroom``, the old formula refused this pickup at ANY
    walk distance -- walking past half a tank of fuel one tile away.
    The rate predicate scores the actual 500-fuel transfer: worth up
    to a 20-tile walk at 25/tile, so a 12-tile walk clears easily.
    """

    base_world, base_self = make_world(fuel=600)
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")
    container = make_container_state(
        x=106,
        y=106,
        is_fuel=True,
        volume=1000,
        timestamp_ms=100000,
        failed_pickups=0,
    )

    assert pickup_not_worth_walk(ctx, container) is False


def test_unclamped_pickup_is_worth_it_at_the_exact_rate_boundary() -> None:
    """Gain exactly equal to the walk threshold is taken, not refused.

    Fuel 500 (headroom 600), 100-volume container 4 tiles away:
    effective gain 100 == ``25 * 4`` -- the predicate refuses only
    strictly-below-rate pickups, so the boundary case dispatches.
    """

    base_world, base_self = make_world(fuel=500)
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")
    container = make_container_state(
        x=102,
        y=102,
        is_fuel=True,
        volume=100,
        timestamp_ms=100000,
        failed_pickups=0,
    )

    assert pickup_not_worth_walk(ctx, container) is False


def test_critical_fuel_takes_any_reachable_sliver() -> None:
    """Below the fuel-low break the worth-the-walk rule is suspended.

    Run bot-20260728-090813 exited ``out_of_fuel`` at fuel 98 with a
    pickable 39-fuel container two tiles away, refused by the rate
    predicate as "not worth 2-tile walk". The predicate is efficiency
    logic for a healthy tank; at critical fuel the alternative to the
    walk is ending the session, so any reachable fuel dispatches.
    """

    base_world, base_self = make_world(
        fuel=98,
        scanned=True,
        containers={
            "102,100": make_container_state(
                x=102,
                y=100,
                is_fuel=True,
                volume=39,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = select_and_pickup_fuel(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected critical-fuel pickup decision")
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["target_x"] == 102
    assert decision["behavior"]["reason_context"]["volume"] == 39


def test_select_and_pickup_fuel_refuses_when_projected_pickup_overflows() -> None:
    """``select_and_pickup_fuel`` returns None on a not-worth-it sliver.

    Wire the 2026-07-06 waste class end-to-end: private at fuel 1092
    (headroom 8) with a single visible 386-volume fuel container 3
    tiles east -- effective gain 8 against a ``3 * 3 = 9``
    threshold (measured-speed pricing, 2026-08-06). The at-cap gate
    passes (fuel below cap), the fuel
    target is selected successfully, but ``pickup_not_worth_walk``
    fires and the planner returns None instead of dispatching. The
    container is left untouched -- not blacklisted -- so a later
    tick with more headroom (or from an adjacent tile) can still
    consume it.
    """

    base_world, base_self = make_world(
        fuel=1092,
        scanned=True,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=True,
                volume=386,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    self_state = SelfStateDict(**{**base_self, "rank": 1})
    world = base_world
    world["self_state"] = self_state
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory()
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = select_and_pickup_fuel(ctx, ctx.base)

    assert decision is None
    assert world["containers"]["103,100"]["failed_pickups"] == 0


def test_walkworthy_iteration_takes_the_next_candidate_after_a_veto() -> None:
    """The best-scored container failing the walk rate does not end the search.

    Flag s9-2/3: the 1183-volume container 13 tiles away was vetoed
    ("clamped gain 24 not worth 13-tile walk") and the single-candidate
    logic sent the cascade into an in-viewport larder teleport while a
    walk-worthy container sat 3 tiles away.
    """
    world, self_state = make_world(
        fuel=1076,
        containers={
            "87,100": make_container_state(
                x=87,
                y=100,
                is_fuel=True,
                volume=1183,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=True,
                volume=762,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        None,
        "",
    )

    selection = _first_walkworthy_fuel(ctx)

    if selection is None:
        raise AssertionError("the 3-tile candidate must be selected")
    container, command = selection
    assert (container["x"], container["y"]) == (103, 100)
    assert command["cmd_type"] in ("move", "pickup_fuel")


def test_low_volume_candidates_stay_out_of_the_ranked_list() -> None:
    """The minimum-volume floor filters candidates before ranking."""
    from tankpit_bot.bot.ai.equipment_search import find_fuel_candidates

    world, self_state = make_world(
        fuel=500,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=True,
                volume=40,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )

    assert find_fuel_candidates(world, self_state, None, minimum_volume=100) == []


def test_fuel_lock_steal_requires_an_executable_candidate() -> None:
    """The fuel steal applies the same executability bar (session-12)."""
    from tankpit_bot.bot.ai.collect_locks import _superior_fuel_candidate
    from tankpit_bot.sniffer.world_state import mark_move_target_failed, reset_world_state

    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=True,
            volume=90,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
        "103,100": make_container_state(
            x=103,
            y=100,
            is_fuel=True,
            volume=900,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=300, containers=containers)
    reset_world_state()
    mark_move_target_failed(103, 100, 99000)
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), make_inventory(), 100000, None, "")

    try:
        result = _superior_fuel_candidate(ctx, containers["130,100"])
    finally:
        reset_world_state()

    assert result is None
