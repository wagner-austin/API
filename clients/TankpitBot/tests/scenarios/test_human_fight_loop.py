"""End-to-end human-fight loop through real wire dispatch + production decide.

Exercises the 2026-07-31 fair-fight contracts as one continuous story
against the scenario harness (typed messages through the production
dispatcher, no mocks, no hand-built world state):

1. an unconsented human is never fired on; their chat consents them;
2. the fight holds under sustained fire while fuel is above half
   capacity (the human break band);
3. below half capacity the break fires, the lock survives, and the
   escape latch is the RESUME floor (750), never full capacity;
4. fuel back at the resume floor re-engages the same human in person;
5. pursuit fire at the departed human is capped at ONE homing per
   departure window — the second tick chases via the map;
6. the deactivation confirms the kill and clears the lock.

The bot's own tank is rank 1: capacity 1100, half-capacity band 550,
resume floor 200 + 100 + 450 = 750, inventory cap 25. Engagement
rhythm per the behavior contract: the locking tick CLOSES (walk for a
short close), the landing tick spends the combat scan, and only then
the shots fly.
"""

from __future__ import annotations

from tests.scenarios._harness import DEFAULT_SELF_TANK_ID, BotScenario
from tests.scenarios._wire_builders import (
    chat_message,
    deactivation,
    movement_response,
    self_status_sync,
    shoot_event,
)

YUPPLER_ID = 50


def _confirmed_hit(scenario: BotScenario, fuel_after: int) -> None:
    """One Yuppler dual on the wire, confirmed by the self fuel drop."""
    scenario.ingest(
        shoot_event(
            shooter_id=YUPPLER_ID,
            source_x=101,
            source_y=100,
            target_x=100,
            target_y=100,
        )
    )
    scenario.ingest(self_status_sync(fuel=fuel_after, tank_id=DEFAULT_SELF_TANK_ID))
    # He is firing, so his own wire stays fresh — keeps the fight
    # in-person (viewport-confirmed) through the damage phase.
    scenario.ingest(movement_response(YUPPLER_ID, x=101, y=100))


def _engaged_fight() -> BotScenario:
    """A consented adjacent Yuppler fight advanced to the first shot.

    With the target consented, cardinally adjacent, and inside the
    established viewport, the locking tick is the MINE PIN (operator
    order 2026-09-01: the first close engage tick salts the ring) and
    the very next tick fires — the in-view shot short-circuit (F8:
    in-view alone is the firing criterion), no teleport and therefore
    no landing scan.
    """
    scenario = BotScenario()
    scenario.place_self(x=100, y=100, fuel=1100)
    scenario.place_enemy(tank_id=YUPPLER_ID, x=101, y=100, name="Yuppler")
    scenario.ingest(chat_message(YUPPLER_ID))

    pin_tick = scenario.decide()
    assert pin_tick["command"]["cmd_type"] == "mine_drop"
    assert pin_tick["behavior"]["reason_kind"] == "mine_pin"
    assert pin_tick["updated_ai_state"]["combat_target_id"] == YUPPLER_ID
    assert pin_tick["updated_ai_state"]["mine_pin_target_id"] == YUPPLER_ID
    scenario.advance_clock()

    lock_tick = scenario.decide()
    assert lock_tick["command"]["cmd_type"] == "shoot"
    assert lock_tick["updated_ai_state"]["combat_target_id"] == YUPPLER_ID
    return scenario


def test_unconsented_human_is_not_engaged_until_they_chat() -> None:
    """The consent gate end to end: no lock before the chat, lock after."""
    scenario = BotScenario()
    scenario.place_self(x=100, y=100, fuel=1100)
    scenario.place_enemy(tank_id=YUPPLER_ID, x=101, y=100, name="Yuppler")

    before = scenario.decide()

    # Unconsented: the adjacent human is invisible to targeting; with
    # no terrain the greet visit cannot be vouched for either, so the
    # tick is the enemy-search map open — never a shot, never a lock.
    # The HELLO fires on ENCOUNTER (2026-07-30 contract): it rides
    # this very tick as the secondary command, latched once per human.
    assert before["command"]["cmd_type"] == "map_open"
    assert before["updated_ai_state"]["combat_target_id"] == -1
    greeting = before["secondary_command"]
    if greeting is None:
        raise AssertionError("expected the HELLO greeting to ride the encounter tick")
    assert greeting["cmd_type"] == "chat"
    assert before["updated_ai_state"]["greeted_tank_ids"] == {str(YUPPLER_ID): 100000}

    scenario.ingest(chat_message(YUPPLER_ID))
    scenario.advance_clock()
    after = scenario.decide()

    assert after["updated_ai_state"]["combat_target_id"] == YUPPLER_ID
    # Greeted exactly once: the locking tick carries no second HELLO.
    assert after["secondary_command"] is None


def test_full_human_fight_loop_break_partial_restock_reengage_cap_kill() -> None:
    """The whole 2026-07-31 contract stack as one continuous fight."""
    scenario = _engaged_fight()

    # --- Phase 2: sustained fire, fuel still above the 550 band ---
    # Six confirmed duals walk fuel 1100 -> 560. Every tick the band
    # holds: the bot keeps shooting instead of fleeing (the old
    # projection broke these fights near 900 fuel).
    for fuel_after in (1010, 920, 830, 740, 650, 560):
        scenario.advance_clock()
        _confirmed_hit(scenario, fuel_after)
        held = scenario.decide()
        assert held["command"]["cmd_type"] == "shoot"
        assert held["updated_ai_state"]["combat_target_id"] == YUPPLER_ID
        assert held["updated_ai_state"]["break_escape_until_fuel"] == 0

    # --- Phase 3: two more hits cross the band; the break fires ---
    # Two, not one, since the rank-scaled reserves (row 6,
    # [[flag-triage-20260902]]): the smaller floor keeps the 470-fuel
    # fight fundable below capacity, so the projection latch — not the
    # human resume branch — would own it. The hotter window pushes the
    # projection past capacity and exercises the branch this scenario
    # exists to pin.
    scenario.advance_clock()
    _confirmed_hit(scenario, 470)
    scenario.advance_clock()
    _confirmed_hit(scenario, 380)
    broke = scenario.decide()

    assert broke["behavior"]["mode"] == "COLLECT"
    assert broke["updated_ai_state"]["combat_target_id"] == YUPPLER_ID
    # The human latch is the RESUME floor, not capacity: refuel to the
    # rank-scaled floor (capacity//2 + hunt reserve = 550 + 78 = 628
    # for this private) and get back in, never a full-tank rebuild.
    assert broke["updated_ai_state"]["break_escape_until_fuel"] == 628

    # --- Phase 4: fuel recovers past the floor; re-engage in person ---
    scenario.advance_clock()
    scenario.ingest(self_status_sync(fuel=760, tank_id=DEFAULT_SELF_TANK_ID))
    scenario.ingest(movement_response(YUPPLER_ID, x=101, y=100))
    resumed = scenario.decide()

    assert resumed["command"]["cmd_type"] == "shoot"
    assert resumed["updated_ai_state"]["combat_target_id"] == YUPPLER_ID
    assert resumed["updated_ai_state"]["break_escape_until_fuel"] == 0

    # --- Phase 5: he vanishes from view; ONE pursuit homing, then chase ---
    # 6 s without wire: past the 5 s viewport-presence TTL (he is no
    # longer a visible threat) but inside both the 7 s wire gates and
    # the 12 s homing-trace wall — the exact reroute-milking window.
    scenario.advance_clock(6000)
    pursuit = scenario.decide()
    assert pursuit["command"]["cmd_type"] == "shoot"
    assert pursuit["updated_ai_state"]["pursuit_shot_target_id"] == YUPPLER_ID
    assert pursuit["updated_ai_state"]["pursuit_shot_ms"] == scenario.timestamp_ms

    scenario.advance_clock()
    capped = scenario.decide()
    assert capped["command"]["cmd_type"] == "map_open"
    assert capped["behavior"]["reason_kind"] == "find_target"
    assert capped["updated_ai_state"]["combat_target_id"] == YUPPLER_ID

    # --- Phase 6: he reappears, eats the shot, dies; confirm-kill ---
    scenario.advance_clock()
    scenario.ingest(movement_response(YUPPLER_ID, x=101, y=100))
    reengaged = scenario.decide()
    assert reengaged["command"]["cmd_type"] == "shoot"

    scenario.advance_clock()
    scenario.ingest(deactivation(victim_id=YUPPLER_ID, killer_id=DEFAULT_SELF_TANK_ID))
    confirmed = scenario.decide()

    assert confirmed["behavior"]["reason_kind"] == "confirm_kill"
    assert confirmed["updated_ai_state"]["combat_target_id"] == -1
