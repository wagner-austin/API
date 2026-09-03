"""The mine pin over the seam: press once, land the 3x3, keep fighting.

Operator question 2026-09-01: "have we tried it? is it part of the
sim btw?" The sim models the press physics (law 6,
``process_mine_press``: 3x3 self-centered, rock/water/tank skipped,
enemy mines traded 1:1) — this cell proves the BOT side end to end: a
real ``Bot`` acquires the seeded enemy, closes, and its first
in-reach engage tick sends ``CMD_MINE`` over the wire; the sim's
``0x4B`` answer lands the pattern back in the bot's own mine
registry; and the latch holds the press to exactly one for the whole
engagement.
"""

from __future__ import annotations

from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_body import _tick_once
from tankpit_bot.sim.session import deliver_batch
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim.seam import SEAM_ENEMY_ID, boot_seam

_CLIENT_TEAM = 2


def test_adjacent_engage_presses_the_pin_once_over_the_wire() -> None:
    """One engagement, one wire press, a landed 3x3, and the fight goes on."""
    bot, server, link, table = boot_seam(client_fuel=1100)
    del table
    ws = bot.world
    ws.terrain_map = InMemoryTerrainMap()

    try:
        for _ in range(14):
            _tick_once(bot)
            deliver_batch(bot._cdp_message_buffer, server.advance_tick(), link)
    except SessionExitError as error:
        # Killing the seeded enemy empties the room and the session
        # ends the production way; any other exit is a scenario bug.
        if error.reason != "no_viable_targets":
            raise

    presses = link.sent_commands.count("mine")
    assert presses == 1, f"expected exactly one pin press per engagement, got {presses}"
    # The press was billed at the flat 10 ([[mine-mechanics]]).
    press_entries = [e for e in ws.fuel_book["entries"] if e["kind"] == "mine_press"]
    assert [(e["lo"], e["hi"]) for e in press_entries] == [(-10, -10)]
    # The sim's 0x4B answer landed OUR pattern in OUR registry.
    own_mines = [m for m in ws.world_state["mines"].values() if m["team"] == _CLIENT_TEAM]
    assert own_mines, "the 0x4B placement never landed in the bot's own mine registry"
    # The pin never displaced the fight: shots still went out.
    assert link.sent_commands.count("shoot") >= 1
    assert str(SEAM_ENEMY_ID) in bot._ai_state["mine_pin_presses"]
