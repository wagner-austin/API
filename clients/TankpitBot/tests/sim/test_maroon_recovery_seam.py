"""The marooned pan-and-walk recovery: fuel beyond the window is reached.

The deterministic cell for the 2026-08-25 maroon-pan fix (run
bot-20260825-133452): Artax entered at fuel 0 with every known fuel dot
outside the stored window, autoscroll pinned OFF, and the pre-pan
walker shuttled between window-edge clamp tiles for the entire 331 s
session — 74 successful moves, zero net progress, zero pickups. This
scenario reproduces the exact shape over the seam: a real ``Bot`` at
fuel 0, the only fuel container a whole window's width away, and the
recovery must be the pan-walk gait — free ``Rb`` pans anchoring the
window toward the fuel (the measured anchor law the sim enforces),
walking legs crossing each revealed stretch, and the normal cascade
finishing with the pickup.
"""

from __future__ import annotations

from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.ledger.ring import outcome_counts
from tankpit_bot.sim.session import deliver_batch
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim.seam import SEAM_CLIENT_ID, boot_seam

_FUEL_X, _FUEL_Y = 130, 100
_ROUNDS = 40


def test_marooned_tank_pans_and_walks_to_fuel_beyond_the_window() -> None:
    """A fuel-0 tank reaches a container a full window away and refuels.

    The container at (130,100) is 30 tiles east of the spawn — outside
    the initial window and beyond every teleport at fuel 0, but inside
    the 48-tile walk cap once the map atlas names its dot. The session
    must NOT oscillate at the window edge: the gait is walk-to-edge,
    pan, walk the revealed ground, pan again, then the ordinary pickup.
    """
    bot, server, link, table = boot_seam(
        client_fuel=0,
        containers=((_FUEL_X, _FUEL_Y, 400),),
        enemy_alive=False,
    )
    del table
    ws = bot.world
    ws.terrain_map = InMemoryTerrainMap()

    try:
        for _ in range(_ROUNDS):
            _tick_once(bot)
            deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
    except SessionExitError as error:
        # After the refuel the world holds no more containers and no
        # enemies, so the session ends the production way. Any exit
        # BEFORE the refuel — out_of_fuel above all — is the marooning
        # this test exists to prevent, and propagates.
        if error.reason not in ("no_productive_collect", "no_viable_targets"):
            raise

    assert link.sent_commands.count("scope") >= 2, (
        f"the window-edge clamp never drew its pans: {link.sent_commands}"
    )
    scope_outcomes = outcome_counts(ws.ledger, "scope")
    assert scope_outcomes.get("confirmed", 0) >= 2, (
        f"pans dispatched but not confirmed over the seam: {scope_outcomes}"
    )
    collect_outcomes = outcome_counts(ws.ledger, "collect")
    assert sum(collect_outcomes.values()) >= 1, (
        f"the walk never ended in a pickup: {collect_outcomes}"
    )
    fuel = server.world["tanks"][SEAM_CLIENT_ID]["fuel"]
    assert fuel > 0, f"the tank never refueled (server fuel {fuel})"
