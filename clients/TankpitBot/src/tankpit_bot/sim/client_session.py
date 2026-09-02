"""Per-connection simulator state — everything one CLIENT owns.

The other half of the field/connection split. Each of these three
already took ``client_id`` at construction, so the boundary was latent
long before it was drawn: the stored 0x5A window and its patch memory,
the rank recovery window, and the award ledger are all facts about ONE
connection, not about the field.

Nothing here is shared between connections, which is the property that
makes a second one possible. What is NOT here matters as much: the
combat clocks moved out to :class:`tankpit_bot.sim.combat_clock`
because a corpse clears once for the room and a firing cost is billed
once against the shooter, however many connections are watching.

The emission side is no longer the blocker it was. The law modules
RESOLVE and :mod:`tankpit_bot.sim.narrate` NARRATES, so a second
connection is narrated to by calling the same pure narrators with a
different ``observer_id`` — no mutation is repeated
([[recipient-policy]] records which messages each connection is even
entitled to see).
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.sim.awards import AwardLedger
from tankpit_bot.sim.progression import RankProgression
from tankpit_bot.sim.viewport_window import ViewportTracker
from tankpit_bot.sim.world import SimWorldDict


class ClientSession:
    """One connection's own state on a field.

    Attributes:
        client_id: The connected client's tank id.
        viewport: The client's stored 0x5A window, its dynamic-terrain
            patch memory, and the viewport-membership set driving 0x58
            exits and 0x3D entries.
        progression: The rank recovery window opened by a deactivation.
        awards: The decoration ledger and its grant thresholds.
    """

    def __init__(
        self,
        world: SimWorldDict,
        terrain: TerrainMapProtocol,
        client_id: int,
    ) -> None:
        """Bind one connection's state to a world and its terrain.

        Args:
            world: Simulated world (read, never owned).
            terrain: Static terrain for the world's field.
            client_id: The connected client's tank id.
        """
        self.client_id = client_id
        self.viewport = ViewportTracker(world, terrain, client_id)
        self.progression = RankProgression(client_id)
        self.awards = AwardLedger(client_id)


__all__ = ["ClientSession"]
