"""What the archive last said about the world, for anchoring the replay.

A replay that let the sim's state run free would compare the sim's answer
to the wrong question within a few ticks: one missed pickup and every
later fuel reading differs, and from then on the comparison measures the
drift, not the law. So before each tick the replay ANCHORS the sim to the
archive: the client tank's tile, fuel, rank, equipment and window, every
other tank's tile, and every container the wire has shown, all set to
what the real server most recently stated. The sim then answers that
tick's commands from the real tick's starting state, and what it emits is
a statement about its laws alone.

:class:`ArchiveMirror` is the record of those statements. It reads every
received message of a tick (after the tick is compared, so the anchor for
tick N is what the server had said before tick N's batch) and
:meth:`ArchiveMirror.anchor` writes it into a sim world.
"""

from __future__ import annotations

from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.world import SimContainerDict, SimEquipmentDict, SimTankDict, SimWorldDict

EQUIPMENT_VOLUME = -1
"""A radar container entry with this volume is an equipment container."""


class _OwnState:
    """The client tank's last stated fields; None until first stated."""

    def __init__(self) -> None:
        self.tile: tuple[int, int] | None = None
        self.fuel: int | None = None
        self.rank: int | None = None
        self.counts: list[int] | None = None
        self.enabled: list[bool] | None = None
        self.window: tuple[int, int] | None = None
        self.alive = True


class ArchiveMirror:
    """The archive's latest statement of each anchored fact.

    Attributes:
        self_id: The capturing client's tank id, from its first 0x21, or
            None before the server has introduced it.
    """

    def __init__(self) -> None:
        self.self_id: int | None = None
        self._own = _OwnState()
        self._others: dict[int, tuple[int, int, bool]] = {}
        self._fuel: dict[tuple[int, int], int] = {}
        self._equipment: dict[tuple[int, int], bool] = {}

    @property
    def window(self) -> tuple[int, int] | None:
        """The client's stored window origin, as the last 0x5A stated it."""
        return self._own.window

    def observe(self, message: BinaryMessage) -> None:
        """Record what one received message states.

        Args:
            message: A decoded received message.
        """
        if message["msg_type"] == 0x21:
            if self.self_id is None:
                self.self_id = message["tank_id"]
        elif message["msg_type"] == 0x3D:
            self._place(message["tank_id"], message["x"], message["y"])
        elif message["msg_type"] == 0x47:
            end = message["waypoints"][-1] if message["waypoints"] else None
            if end is not None:
                self._place(message["tank_id"], end[0], end[1])
        elif message["msg_type"] == 0x41:
            self._kill(message["victim_id"])
        else:
            self._observe_own(message)
            self._observe_containers(message)

    def _place(self, tank_id: int, x: int, y: int) -> None:
        """A tank was stated at a tile, which also states it alive."""
        if tank_id == self.self_id:
            self._own.tile = (x, y)
            self._own.alive = True
        else:
            self._others[tank_id] = (x, y, True)

    def _kill(self, tank_id: int) -> None:
        """A tank was stated destroyed."""
        if tank_id == self.self_id:
            self._own.alive = False
            return
        stated = self._others.get(tank_id)
        if stated is not None:
            self._others[tank_id] = (stated[0], stated[1], False)

    def _observe_own(self, message: BinaryMessage) -> None:
        """Record the client tank's own fuel, rank, equipment and window."""
        if message["msg_type"] == 0x44:
            self._own.fuel = message["fuel_total"]
        elif message["msg_type"] == 0x2E:
            if message["tank_id"] == self.self_id:
                self._own.rank = message["rank"]
                if message["fuel"] is not None:
                    self._own.fuel = message["fuel"]
        elif message["msg_type"] == 0x49:
            self._own.counts = list(message["counts"])
            self._own.enabled = list(message["enabled"])
        elif message["msg_type"] == 0x74:
            self._own.enabled = list(message["enabled"])
        elif message["msg_type"] == 0x5A:
            self._own.window = (message["viewport_left"], message["viewport_top"])

    def _observe_containers(self, message: BinaryMessage) -> None:
        """Record container contents from radar scans and pickup records.

        A pickup record at an equipment tile with nothing remaining is
        the container being consumed; at any other tile it is a fuel
        container's remaining volume.
        """
        if message["msg_type"] == 0x4F:
            for entry in message["containers"]:
                tile = (entry["x"], entry["y"])
                if entry["volume"] == EQUIPMENT_VOLUME:
                    self._equipment[tile] = True
                else:
                    self._fuel[tile] = entry["volume"]
        elif message["msg_type"] == "container_pickup":
            for record in message["pickups"]:
                tile = (record["x"], record["y"])
                if tile in self._equipment:
                    self._equipment[tile] = record["remaining_volume"] != 0
                else:
                    self._fuel[tile] = record["remaining_volume"]

    def anchor(self, world: SimWorldDict, client: SimTankDict) -> None:
        """Write every recorded statement into a sim world.

        A tank the sim does not hold is not created: the replay seeds the
        world's tanks from the capture once, and a tank it could not seat
        has no sim counterpart to anchor.

        Args:
            world: The sim world (mutated).
            client: The sim tank standing in for the capturing client.
        """
        own = self._own
        if own.tile is not None:
            client["x"], client["y"] = own.tile
        if own.fuel is not None:
            client["fuel"] = own.fuel
        if own.rank is not None:
            client["rank"] = own.rank
        if own.counts is not None:
            client["counts"] = list(own.counts)
        if own.enabled is not None:
            client["enabled"] = list(own.enabled)
        client["alive"] = own.alive
        for tank_id, (x, y, alive) in self._others.items():
            tank = world["tanks"].get(tank_id)
            if tank is not None and tank is not client:
                tank["x"], tank["y"], tank["alive"] = x, y, alive
        self._anchor_containers(world)

    def _anchor_containers(self, world: SimWorldDict) -> None:
        """Set every stated container's contents in the sim world."""
        fuel_index = {(c["x"], c["y"]): c for c in world["containers"]}
        for (x, y), volume in self._fuel.items():
            known = fuel_index.get((x, y))
            if known is None:
                world["containers"].append(SimContainerDict(x=x, y=y, volume=volume, dotted=False))
            else:
                known["volume"] = volume
        present = {tile for tile, stocked in self._equipment.items() if stocked}
        consumed = {tile for tile, stocked in self._equipment.items() if not stocked}
        held = {(e["x"], e["y"]) for e in world["equipment"]}
        world["equipment"] = [e for e in world["equipment"] if (e["x"], e["y"]) not in consumed]
        for x, y in sorted(present - held):
            world["equipment"].append(SimEquipmentDict(x=x, y=y))


__all__ = ["EQUIPMENT_VOLUME", "ArchiveMirror"]
