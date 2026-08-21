"""Shared seam boot for the sim tests.

One place builds "a real ``Bot`` wired to a fresh sim world over the
seam" so the step-(c) smoke, the step-(e) divergence soak, and the
audit cross-check all exercise the identical wiring: real command
service, real encoders, real ingestion — only ``bot._cdp`` is the sim
link.
"""

from __future__ import annotations

from tankpit_bot.bot.base import Bot
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.session import SimCDPSession, deliver_batch
from tankpit_bot.sim.world import (
    SimBlockDict,
    SimContainerDict,
    SimEquipmentDict,
    SimFerryDict,
    make_sim_tank,
    make_sim_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

SEAM_MAGIC = "seammagic5uk3et4epiexu"
"""Twenty-plus lowercase alphanumerics, the shape every archived magic
has — the production AUTH reader rejects anything under ten characters
([[session-state-deglobalisation]])."""
SEAM_CLIENT_ID = 9
SEAM_ENEMY_ID = 11

#: Default container seeding: (x, y, volume) triples near the client
#: spawn, enough that a 12-round smoke never runs the world dry
#: (an exhausted world ends the session the production way with
#: ``SessionExitError: no_productive_collect`` — a real finding from
#: the step-(c) smoke, now a seeding rule).
DEFAULT_CONTAINERS: tuple[tuple[int, int, int], ...] = (
    (103, 100, 300),
    (97, 104, 400),
    (106, 95, 400),
)

#: Rich seeding for long soaks: enough fuel on the ground that a
#: 30-40 round session stays productive even after the enemy dies.
RICH_CONTAINERS: tuple[tuple[int, int, int], ...] = (
    *DEFAULT_CONTAINERS,
    (94, 97, 400),
    (109, 104, 400),
    (100, 108, 400),
    (92, 103, 300),
    (105, 92, 300),
    # Spread satellites >= 16 tiles apart so the zero-overlap clean-
    # viewport hop rule (user ruling 2026-07-26) always has fresh
    # ground within teleport range — the real field's 620 dots span
    # the whole 256x256 map; a single 17-tile cluster is the
    # unrealistic case that starves the cascade.
    (130, 100, 400),
    (100, 130, 400),
    (70, 100, 400),
    (100, 70, 300),
    (132, 132, 300),
    (68, 132, 300),
)


class SeamClock:
    """Deterministic stepping clock for multi-round seam sessions.

    Install on ``_test_hooks.get_current_time_ms`` via save-and-restore
    (the scenarios-harness discipline) so every timestamp in the
    system — wire stamps, diagnostics, the seam's capture recording —
    advances at the pace the test dictates instead of collapsing into
    one wall-clock millisecond burst.
    """

    def __init__(self, start_ms: int) -> None:
        """Start the clock.

        Args:
            start_ms: Initial clock value in milliseconds.
        """
        self.now_ms = start_ms

    def __call__(self) -> int:
        """Return the current clock value.

        Returns:
            The clock value in milliseconds.
        """
        return self.now_ms

    def advance(self, delta_ms: int) -> None:
        """Advance the clock.

        Args:
            delta_ms: Milliseconds to add.
        """
        self.now_ms += delta_ms


def boot_seam(
    *,
    enemy_fuel: int = 1800,
    client_fuel: int = 800,
    containers: tuple[tuple[int, int, int], ...] = DEFAULT_CONTAINERS,
    counts: tuple[int, int, int, int, int] = (25, 25, 25, 25, 25),
    equipment: tuple[tuple[int, int], ...] = (),
    enemy_counts: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    ferries: tuple[tuple[int, int], ...] = (),
    blocks: tuple[tuple[int, int], ...] = (),
    enemy_alive: bool = True,
) -> tuple[Bot, SimServer, SimCDPSession, bytes]:
    """Build a real Bot wired to a fresh sim world over the seam.

    Args:
        enemy_fuel: Starting fuel for the seeded enemy (rank 8, so a
            damage tier survives long enough for real fights).
        client_fuel: Starting fuel for the client tank (rank 1,
            capacity 1100). Fighting soaks boot at 1100 so the
            hunt-only-when-full contract lets combat start at tick 0.
        containers: Fuel-container seeding as (x, y, volume) triples.
        counts: The client's starting 0x49 slot counts.
        equipment: Equipment-container seeding as (x, y) pairs.
        enemy_counts: The enemy's slot counts (arm it for fighting
            soaks driven by ``sim.opponent``).
        ferries: Ferry seeding as (x, y) pairs (water tiles).
        blocks: Resting movable-block seeding as (x, y) pairs.
        enemy_alive: False boots an enemy-free room (the handshake
            announces only living tanks) — for collect-only scenarios
            where any combat lane would hijack the behavior under
            test.

    Returns:
        The bot, the sim server, the sim CDP link, and the XOR table,
        with the join handshake already delivered to the bot's buffer.

    Raises:
        XorStaticKeyUnavailableError: If the repo's XOR static key is
            unavailable.
    """
    table = build_session_xor_table(SEAM_MAGIC)
    world = make_sim_world("field01_r.gif")
    world["tanks"][SEAM_CLIENT_ID] = make_sim_tank(SEAM_CLIENT_ID, 2, 1, 100, 100, client_fuel)
    world["tanks"][SEAM_CLIENT_ID]["counts"] = list(counts)
    world["tanks"][SEAM_ENEMY_ID] = make_sim_tank(SEAM_ENEMY_ID, 1, 8, 110, 100, enemy_fuel)
    world["tanks"][SEAM_ENEMY_ID]["counts"] = list(enemy_counts)
    world["tanks"][SEAM_ENEMY_ID]["alive"] = enemy_alive
    for x, y, volume in containers:
        world["containers"].append(SimContainerDict(x=x, y=y, volume=volume, dotted=True))
    for x, y in equipment:
        world["equipment"].append(SimEquipmentDict(x=x, y=y))
    for x, y in ferries:
        world["ferries"].append(SimFerryDict(x=x, y=y))
    for x, y in blocks:
        world["blocks"].append(SimBlockDict(x=x, y=y))
    server = SimServer(world, InMemoryTerrainMap(), client_id=SEAM_CLIENT_ID)
    bot = Bot("https://sim.tankpit.local/", headless=True)
    bot._magic = SEAM_MAGIC
    bot._on_magic_captured(SEAM_MAGIC)
    link = SimCDPSession(server, SEAM_MAGIC)
    bot._cdp = link
    deliver_batch(bot._cdp_message_buffer, server.handshake(), link)
    return bot, server, link, table


__all__ = [
    "DEFAULT_CONTAINERS",
    "RICH_CONTAINERS",
    "SEAM_CLIENT_ID",
    "SEAM_ENEMY_ID",
    "SEAM_MAGIC",
    "SeamClock",
    "boot_seam",
]
