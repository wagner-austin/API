"""The ring-refusal scenario: the full loop-breaking chain over the seam.

The deterministic cell the scenario matrix was missing ([[coding-standards]]
§ Verification discipline): a real ``Bot`` against the sim, an equipment
container fully ringed by hostile mines the bot has never revealed. The
mined refusal law (137/137 archived receipts, [[teleport-mechanics]]):
the sim refuses the hop with a confirm-at-origin — no 0x52, no charge —
the bot ingests it as a landing refusal, and the identical hop can never
be re-certified. Pre-fix, this exact geometry ran 534 identical hops in
the 08-05 ancestor and 4-in-10-seconds in the 2026-08-21 marooning.
"""

from __future__ import annotations

from collections.abc import Callable

from tankpit_bot import _test_hooks
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.sim.session import deliver_batch
from tankpit_bot.sim.world import place_mine
from tankpit_bot.state.types import make_container_state
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim.seam import SEAM_CLIENT_ID, SeamClock, boot_seam

_RING_X, _RING_Y = 130, 100
_ENEMY_TEAM = 1


def test_ringed_hop_is_refused_once_and_never_re_certified() -> None:
    """One hop, one refusal, zero repeats — the loop is unrepresentable.

    The bot knows the equipment (an earlier scan's belief) but not the
    ring (never revealed). It hops; the sim's ring-blocked teleport
    answers with the measured confirm-at-origin; the dispatcher books
    the landing refusal; and from that tick the composed terrain
    refuses every service tile of the ringed container, so the ledger
    shows exactly ONE teleport decision at the ring — pre-fix it
    showed one per replan cycle, forever.
    """
    clock = SeamClock(100_000)
    original_clock: Callable[[], int] = _test_hooks.get_current_time_ms
    _test_hooks.get_current_time_ms = clock
    rounds = 0
    try:
        bot, server, link, table = boot_seam(
            client_fuel=1100,
            containers=(),
            counts=(15, 15, 15, 15, 15),
            equipment=((_RING_X, _RING_Y),),
        )
        del table
        ws = bot.world
        # The hop lane needs a static map (terrain-less worlds have no
        # teleport planning at all — the walk lanes own everything).
        ws.terrain_map = InMemoryTerrainMap()
        # The full service ring the sim's teleport tries (target, E, N, W,
        # S) is mined by the enemy team — and the bot has NEVER revealed
        # any of it: its beliefs are mine-blind, the exact marooning state.
        for dx, dy in ((0, 0), (1, 0), (0, -1), (-1, 0), (0, 1)):
            place_mine(server.world, _RING_X + dx, _RING_Y + dy, _ENEMY_TEAM)
        # The equipment belief from "an earlier scan": the hop's tracked
        # container, injected through the production state shape.
        ws.world_state["containers"][f"{_RING_X},{_RING_Y}"] = make_container_state(
            x=_RING_X,
            y=_RING_Y,
            is_fuel=False,
            volume=0,
            timestamp_ms=1,
            failed_pickups=0,
        )

        try:
            for _ in range(14):
                rounds += 1
                _tick_once(bot)
                deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
                clock.advance(1000)
        except SessionExitError as error:
            # An exhausted world ends the session the production way once
            # the ringed container is released — a legitimate end state
            # (the seam smoke's seeding rule, inverted on purpose), and
            # only that one: any other exit reason is a scenario bug and
            # propagates.
            if error.reason != "no_productive_collect":
                raise
    finally:
        _test_hooks.get_current_time_ms = original_clock

    # The frontier legitimately travels to stale blocks (2026-08-28
    # staleness forage), so the wire carries its hops too; the law
    # under test binds only the RINGED hop, pinned on the decision
    # ledger and the tank's final position below. The decision layer
    # legitimately shows two ring decisions: the original plan, then
    # its re-derivation against the opened map (the teleport/map-open
    # precondition, [[teleport-mechanics]]). Anything beyond that
    # pair is the loop.
    del rounds
    ring_teleport_decisions = [
        decision
        for decision in ws.ledger.decisions.values()
        if decision["cmd_type"] == "teleport"
        and (decision["target_x"], decision["target_y"]) == (_RING_X, _RING_Y)
    ]
    assert len(ring_teleport_decisions) <= 2
    assert f"{_RING_X},{_RING_Y}" in ws.landing_refusals
    truth = server.world["tanks"][SEAM_CLIENT_ID]
    assert (truth["x"], truth["y"]) != (_RING_X, _RING_Y)
