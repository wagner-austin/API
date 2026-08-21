"""The clearance-shot scenario: a ground-aimed shot resolves over the seam.

The deterministic cell for the 2026-08-21 echo-receipt fix: a real
``Bot`` against the sim, a wanted fuel container covered by a REVEALED
hostile mine. The collect planner has no actionable target, so it fires
the mine-clearance shot at the covered tile (``target_id == 0``) — the
exact shape whose ledger decisions all rotted into ``superseded`` in
soak bot-20260821-013519 (13 wire dispatches, 0 completions) and
tripped the liveness detector's false alarm. Post-fix, the sim's own
0x53 echo resolves the shot as ``shoot:fired`` and no clearance
decision dies superseded.
"""

from __future__ import annotations

from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.ledger.ring import outcome_counts
from tankpit_bot.sim.session import deliver_batch
from tankpit_bot.sim.world import place_mine
from tankpit_bot.state.types import make_mine_state
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim.seam import boot_seam

_COVERED_X, _COVERED_Y = 104, 100
_ENEMY_TEAM = 1


def test_clearance_shot_resolves_fired_over_the_seam() -> None:
    """The ground shot's echo becomes ``shoot:fired``, never superseded.

    The bot wants fuel (800 of 1100), the only believed container is
    covered by a believed hostile mine, so the collect flow's
    clearance shot fires at the tile. The sim answers with the bare
    0x53 echo (92.4% of the 11,051 archived shot windows); the tick
    loop's ground-shot resolver must consume it into ``fired`` — the
    completion path whose absence made shoot the one outcome-less
    action kind.
    """
    bot, server, link, table = boot_seam(
        client_fuel=800,
        containers=((_COVERED_X, _COVERED_Y, 400),),
        enemy_alive=False,
    )
    del table
    ws = bot.world
    # A static map arms the hop lanes and the blocked-landing arm of
    # the clearance trigger (terrain-less worlds walk everything).
    ws.terrain_map = InMemoryTerrainMap()
    # The soak geometry: the container tile AND its full neighborhood
    # carry hostile mines, in the sim's truth and the bot's belief
    # alike (revealed by "an earlier scan") — every walk approach and
    # every service landing is denied, so the collect flow's only move
    # is the clearance shot at the covered tile.
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            mine_x, mine_y = _COVERED_X + dx, _COVERED_Y + dy
            place_mine(server.world, mine_x, mine_y, _ENEMY_TEAM)
            ws.world_state["mines"][f"{mine_x},{mine_y}"] = make_mine_state(
                x=mine_x,
                y=mine_y,
                mine_type=1,
                tank_id=99,
                team=_ENEMY_TEAM,
                timestamp_ms=1,
            )

    try:
        for _ in range(10):
            _tick_once(bot)
            deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
    except SessionExitError as error:
        # Once the shot clears the way and the container is consumed
        # (or released), the session ends the production way — either
        # collect exhaustion or the hunt lane finding an empty room.
        # Any other exit is a scenario bug and propagates.
        if error.reason not in ("no_productive_collect", "no_viable_targets"):
            raise

    assert link.sent_commands.count("shoot") >= 1, (
        "the covered container never drew a clearance shot -- scenario setup bug"
    )
    shoot_outcomes = outcome_counts(ws.ledger, "shoot")
    assert shoot_outcomes.get("fired", 0) >= 1, (
        f"no clearance shot resolved on its echo: {shoot_outcomes}"
    )
    assert shoot_outcomes.get("superseded", 0) == 0, (
        f"a dispatched clearance shot still rotted into superseded: {shoot_outcomes}"
    )
    assert ws.pending_ground_shot_dispatch_ms == 0
