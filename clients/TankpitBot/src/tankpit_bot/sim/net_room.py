"""One networked room: a sim field that players join over a socket.

A room is one :class:`~tankpit_bot.sim.server.SimServer` and the room
row the lobby advertises for it. Its world is built the way a field
session builds one (:func:`~tankpit_bot.sim.field_run.make_field_world`,
then ``run_boot._seed_world``), on any shipped field
(:mod:`tankpit_bot.sim.field_choice`), with the practice roster when the
room is a practice room. The layout's client spawn is not a player and
is taken off the field: players are seated as they enter.

A seated player is a tank the room places at an open tile, on the team
it entered with, under its account's name and rank, and connects to the
server. From there the server treats it as it treats any connection: it
asks for its join burst with CMD_ENTER_GAME, its commands join the one
queue, and :meth:`NetRoom.advance` returns its batch each tick.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.parser import RoomInfo
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.field_choice import resolve_field
from tankpit_bot.sim.field_run import make_field_world
from tankpit_bot.sim.net_accounts import AdmittedAccount, SeatResult
from tankpit_bot.sim.practice_room import PracticeRoomDriver
from tankpit_bot.sim.run_boot import _queue_round_opponents, _seed_world, resolve_named_world
from tankpit_bot.sim.scenarios import SIM_CLIENT_ID, SIM_ENEMY_ID
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.spawn import find_open_tile
from tankpit_bot.sim.world import make_sim_tank
from tankpit_bot.types.constants import TROOP_COLOR_NAMES

log = get_logger(__name__)

NET_PLAYER_ID_BASE = 2000
"""First wire id a networked player gets: above the practice roster
(500-535) and below the room's churn visitors (3000 on)."""

NET_PLAYER_IDS = 1000
"""How many players a room admits over its life; ids are never reused."""

MAX_ROOM_PLAYERS = 32
"""How many players a room seats at once."""


class NetError(ValueError):
    """A request a networked room cannot honour (``SIM_NET_*`` codes)."""


def field_room_info(room_id: str, name: str, image: str, *, practice: bool) -> RoomInfo:
    """The lobby row for a room on a shipped field.

    Args:
        room_id: The room's id, as clients select it.
        name: The room's display name.
        image: The field as the lobby names it (``field05.gif``).
        practice: A practice room (mode ``p``, no game modes) rather
            than an open one (mode ``n``).

    Returns:
        The row, in the shape the archive's rows have.

    Raises:
        FieldChoiceError: If no shipped minimap answers to ``image``.
    """
    resolve_field(image)
    return RoomInfo(
        room_id=room_id,
        name=name,
        field_id=int(image.removeprefix("field").removesuffix(".gif")),
        game_modes="0,0,0,0,0,0,0" if practice else "1,1,1,0,1,0,0",
        default_troop=2,
        mode_code="p" if practice else "n",
        image=image,
        year="2026",
    )


class NetRoom:
    """One room's server, its roster and the players it has seated."""

    def __init__(
        self, info: RoomInfo, server: SimServer, driver: PracticeRoomDriver | None
    ) -> None:
        """Bind the room's lobby row to its server.

        Args:
            info: The row the lobby advertises.
            server: The room's server, owning its world.
            driver: The practice roster's driver, or None for a room
                without one.
        """
        self.info = info
        self.server = server
        self._driver = driver
        self._admitted = 0
        self._seated_at: dict[int, int] = {}

    def seat(self, player: AdmittedAccount, troop: int) -> int:
        """Place a player's tank on the field and connect it.

        The tank carries the account's rank and decoration levels, so a
        player rejoins as what it left as.

        Args:
            player: The player's account, as the book admitted it.
            troop: The team the player entered with, 0 to 3.

        Returns:
            The player's tank id.

        Raises:
            NetError: If the troop is not a team (``SIM_NET_TROOP``), the
                room seats :data:`MAX_ROOM_PLAYERS` already or the field
                has no open tile (``SIM_NET_ROOM_FULL``), or the room has
                used every id it may give (``SIM_NET_IDS_EXHAUSTED``).
        """
        if not 0 <= troop < len(TROOP_COLOR_NAMES):
            raise NetError(f"SIM_NET_TROOP: troop {troop} is not a team (0 to 3)")
        if len(self.server.sessions) >= MAX_ROOM_PLAYERS:
            raise NetError(
                f"SIM_NET_ROOM_FULL: room {self.info['room_id']} seats {MAX_ROOM_PLAYERS}"
            )
        if self._admitted >= NET_PLAYER_IDS:
            raise NetError(f"SIM_NET_IDS_EXHAUSTED: room {self.info['room_id']} gave every id")
        world = self.server.world
        landing = find_open_tile(world, self.server.terrain, world["tick"] + self._admitted)
        if landing is None:
            raise NetError(f"SIM_NET_ROOM_FULL: room {self.info['room_id']} has no open tile")
        tank_id = NET_PLAYER_ID_BASE + self._admitted
        self._admitted += 1
        name, rank = player.account["name"], player.account["rank"]
        world["tanks"][tank_id] = make_sim_tank(
            tank_id, troop, rank, landing[0], landing[1], fuel_capacity(rank), name=name
        )
        self.server.connect(tank_id).awards.levels = list(player.decorations)
        self._seated_at[tank_id] = world["tick"]
        log.info("room %s: %s seated as tank %d", self.info["room_id"], name, tank_id)
        return tank_id

    def leave(self, tank_id: int) -> SeatResult:
        """Take a player's tank off the field, closing its connection.

        Args:
            tank_id: The player's tank.

        Returns:
            What the seat came to, read before the tank leaves.

        Raises:
            SimError: If nothing is connected for the tank.
        """
        session = self.server.require_session(tank_id)
        world = self.server.world
        result = SeatResult(
            room_id=self.info["room_id"],
            field=world["field"],
            ticks=world["tick"] - self._seated_at.pop(tank_id),
            rank=world["tanks"][tank_id]["rank"],
            kills=self.server.combat.destroyed_by(tank_id),
            deaths=self.server.combat.deactivations_of(tank_id),
            decorations=tuple(session.awards.levels),
        )
        self.server.disconnect(tank_id)
        return result

    def advance(self) -> dict[int, list[BinaryMessage]]:
        """Play one tick: the roster decides, the field advances.

        Returns:
            Each connected player's batch, keyed by tank id.
        """
        _queue_round_opponents(
            self.server, self._driver, False, None, SIM_ENEMY_ID, self.server.world["tick"]
        )
        batches = self.server.advance_tick()
        if self._driver is not None:
            self._driver.note_field_batches(self.server.world, list(batches.values()))
        return batches


def open_field_room(info: RoomInfo, *, layout: str | None, population_seed: int | None) -> NetRoom:
    """Build a room's world on its field and wrap it in a server.

    Args:
        info: The room's lobby row; its image names the field, and a
            ``p`` mode code seeds the practice roster.
        layout: The practice layout's provenance; None derives it.
        population_seed: The container seed; None derives it.

    Returns:
        The room, no players seated.

    Raises:
        FieldChoiceError: If no shipped minimap answers to the row's image.
    """
    practice = info["mode_code"] == "p"
    _, run_layout, run_population_seed = resolve_named_world(
        f"room-{info['room_id']}", layout, population_seed
    )
    world = make_field_world(practice)
    world["field"] = resolve_field(info["image"])
    terrain, roster_ids, driver = _seed_world(
        world,
        practice=practice,
        layout=run_layout,
        population_seed=run_population_seed,
        atlas_path=None,
        ghost_spec=None,
        rivals=0,
    )
    del world["tanks"][SIM_CLIENT_ID]
    return NetRoom(info, SimServer(world, terrain, roster_ids=roster_ids), driver)


__all__ = [
    "MAX_ROOM_PLAYERS",
    "NET_PLAYER_IDS",
    "NET_PLAYER_ID_BASE",
    "NetError",
    "NetRoom",
    "field_room_info",
    "open_field_room",
]
