"""Per-command wire emission for the sim server's non-combat actions.

Each function processes one already-routed command through its law
module and appends the resulting decoded messages. Supervisor
rejections (0x52) are PER-CONNECTION on the real wire — the client
only ever sees its own, so every rejection here is gated on
``tank_id == client_id`` (the same per-recipient discipline as the
fuel-sync rule).
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.container.types import (
    ContainerPickupDict,
    ContainerPickupRecordDict,
    MineDetonationDict,
    MinePlacementDict,
    TeleportLandedDict,
)
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_GO,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
    SUPERVISOR_ERROR_INVENTORY_FULL,
)
from tankpit_bot.protocol.types import (
    BinaryMessage,
    BuildPickupDict,
    EquipmentGainDict,
    EquipmentToggleDict,
    RadarResultDict,
    RadarScanResultDict,
    SupervisorDict,
    TerrainUpdateDict,
)
from tankpit_bot.sim.actions import (
    MINE_PRESS_FUEL_COST,
    RADAR_FUEL_COST,
    process_mine_press,
    process_radar,
    process_teleport,
)
from tankpit_bot.sim.blocks import process_block_press
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.equipment import resolve_equipment_pickup
from tankpit_bot.sim.movement import MoveOutcomeDict, PickupRecordDict
from tankpit_bot.sim.wire_statements import movement_echo, position_statement
from tankpit_bot.sim.world import SimWorldDict

# TeleportLanded's 1-byte body observed in production captures.
_TELEPORT_LANDED_SUBTYPE = 0x0C


def _pickup_message(pickups: list[PickupRecordDict]) -> ContainerPickupDict:
    """Wrap resolved pickups in the container-pickup wire message."""
    return ContainerPickupDict(
        msg_type="container_pickup",
        pickups=tuple(
            ContainerPickupRecordDict(x=p["x"], y=p["y"], remaining_volume=p["remaining_volume"])
            for p in pickups
        ),
    )


def emit_move(
    world: SimWorldDict,
    client_id: int,
    outcome: MoveOutcomeDict,
    messages: list[BinaryMessage],
) -> None:
    """Emit the wire consequences of one processed move.

    Args:
        world: Simulated world (post-move).
        client_id: The connected client's tank id.
        outcome: The move's outcome.
        messages: This tick's outgoing batch (appended).
    """
    if outcome["kind"] == "cant_go":
        if outcome["tank_id"] == client_id:
            messages.append(
                SupervisorDict(
                    msg_type=0x52,
                    reset_action=1,
                    close_map=0,
                    error_code=SUPERVISOR_ERROR_CANT_GO,
                )
            )
        return
    messages.append(movement_echo(world, outcome))
    for x, y in outcome["mine_positions"]:
        messages.append(MineDetonationDict(msg_type=0x45, positions=[(x, y)]))
    if outcome["pickups"]:
        messages.append(_pickup_message(list(outcome["pickups"])))


def emit_teleport(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    client_id: int,
    tank_id: int,
    command: ClientCommandDict,
    messages: list[BinaryMessage],
) -> bool:
    """Process one teleport command and emit its wire consequences.

    Args:
        world: Simulated world.
        terrain: Static terrain.
        client_id: The connected client's tank id.
        tank_id: The teleporting tank.
        command: The teleport command.
        messages: This tick's outgoing batch (appended).

    Returns:
        True when the hop landed (the tank counts as a mover).
    """
    outcome = process_teleport(world, terrain, tank_id, command["x"], command["y"])
    if outcome["kind"] != "landed":
        code = (
            SUPERVISOR_ERROR_INSUFFICIENT_FUEL
            if outcome["kind"] == "insufficient_fuel"
            else SUPERVISOR_ERROR_CANT_GO
        )
        if tank_id == client_id:
            messages.append(
                SupervisorDict(msg_type=0x52, reset_action=1, close_map=1, error_code=code)
            )
        return False
    messages.append(
        TeleportLandedDict(msg_type="teleport_landed", subtype=_TELEPORT_LANDED_SUBTYPE)
    )
    messages.append(position_statement(world, tank_id))
    if outcome["pickups"]:
        messages.append(_pickup_message(list(outcome["pickups"])))
    return True


def emit_radar(
    world: SimWorldDict,
    client_id: int,
    window: tuple[int, int],
    tank_id: int,
    messages: list[BinaryMessage],
    ammo_changed: set[int],
) -> None:
    """Process one radar command and emit its wire consequences.

    Args:
        world: Simulated world.
        client_id: The connected client's tank id.
        window: The client's stored 0x5A window origin (the client's
            scan covers it; other tanks scan self-centered).
        tank_id: The scanning tank.
        messages: This tick's outgoing batch (appended).
        ammo_changed: Accumulator of tanks whose counts moved.
    """
    outcome = process_radar(world, tank_id, window if tank_id == client_id else None)
    tank = world["tanks"][tank_id]
    tank["fuel"] = max(0, tank["fuel"] - RADAR_FUEL_COST)
    if outcome["consumed_extra"]:
        ammo_changed.add(tank_id)
    messages.append(
        RadarScanResultDict(
            msg_type=0x4F,
            containers=outcome["containers"],
            mines=outcome["mines"],
            mine_clears=[],
        )
    )
    messages.append(RadarResultDict(msg_type=0x46, detection_type=0, found=outcome["enemy_found"]))


def emit_mine_press(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    tank_id: int,
    messages: list[BinaryMessage],
) -> None:
    """Process one mine press and emit its wire consequences.

    Args:
        world: Simulated world.
        terrain: Static terrain.
        tank_id: The placing tank.
        messages: This tick's outgoing batch (appended).
    """
    outcome = process_mine_press(world, terrain, tank_id)
    tank = world["tanks"][tank_id]
    tank["fuel"] = max(0, tank["fuel"] - MINE_PRESS_FUEL_COST)
    if outcome["placed"]:
        messages.append(
            MinePlacementDict(
                msg_type=0x4B,
                mine_type=outcome["mine_type"],
                tank_id=tank_id,
                positions=outcome["placed"],
            )
        )
    if outcome["detonated"]:
        messages.append(MineDetonationDict(msg_type=0x45, positions=outcome["detonated"]))


def emit_equipment_pickup(
    world: SimWorldDict,
    client_id: int,
    tank_id: int,
    kind: str,
    messages: list[BinaryMessage],
    ammo_changed: set[int],
) -> None:
    """Resolve an equipment container under an arriving tank.

    A grant emits the 0x67 gained array (the following 0x49 rides
    the ``ammo_changed`` snapshot — the archive shows every 0x67
    immediately followed by its inventory sync). A full-inventory
    attempt on an explicit ``pickup_equipment`` click answers with
    the measured 0x52 error 7 and leaves the container; incidental
    arrivals at full inventory are silent. Both 0x67 and the 0x52
    are PER-RECIPIENT: production treats any 0x67 as a SELF gain,
    so another tank's grant resolves silently server-side.

    Args:
        world: Simulated world.
        client_id: The connected client's tank id.
        tank_id: The arriving tank.
        kind: The command kind that caused the arrival.
        messages: This tick's outgoing batch (appended).
        ammo_changed: Accumulator of tanks whose counts moved.
    """
    grant = resolve_equipment_pickup(world, tank_id)
    if grant is None:
        return
    if grant["kind"] == "granted":
        if tank_id == client_id:
            messages.append(
                EquipmentGainDict(msg_type=0x67, show_message=True, gained=grant["gained"])
            )
        ammo_changed.add(tank_id)
        return
    if kind == "pickup_equipment" and tank_id == client_id:
        messages.append(
            SupervisorDict(
                msg_type=0x52,
                reset_action=1,
                close_map=0,
                error_code=SUPERVISOR_ERROR_INVENTORY_FULL,
            )
        )


def emit_block_action(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    client_id: int,
    tank_id: int,
    command: ClientCommandDict,
    messages: list[BinaryMessage],
) -> bool:
    """Resolve one 'b' press and emit its wire consequences.

    Success emits the 0x42 BuildPickup event plus the 0x4A tile
    update carrying the tile's post-action value; failures answer
    the client with the measured 0x52 code 1. Block operations
    are FREE (zero fuel delta measured across seven pick/drop
    pairs).

    Args:
        world: Simulated world.
        terrain: Static terrain.
        client_id: The connected client's tank id.
        tank_id: The pressing tank.
        command: The block command.
        messages: This tick's outgoing batch (appended).

    Returns:
        True when the action succeeded (the caller refreshes the
        client's viewport patch — the 2026-07-20 captures show 0x5A
        after block operations).
    """
    outcome = process_block_press(world, terrain, tank_id, command["x"], command["y"])
    if outcome["kind"] in ("out_of_reach", "refused"):
        if tank_id == client_id:
            messages.append(
                SupervisorDict(
                    msg_type=0x52,
                    reset_action=1,
                    close_map=0,
                    error_code=SUPERVISOR_ERROR_CANT_GO,
                )
            )
        return False
    tank = world["tanks"][tank_id]
    messages.append(
        BuildPickupDict(
            msg_type=0x42,
            tank_id=tank_id,
            source_x=tank["x"],
            source_y=tank["y"],
            drop_x=outcome["x"],
            drop_y=outcome["y"],
            direction=outcome["direction"],
            obstacle_type=outcome["tile_value"],
            flag=0,
        )
    )
    messages.append(
        TerrainUpdateDict(
            msg_type=0x4A,
            updates=[(outcome["x"], outcome["y"], outcome["tile_value"])],
        )
    )
    return True


def emit_equipment_toggle(
    world: SimWorldDict, tank_id: int, slot: int, messages: list[BinaryMessage]
) -> None:
    """Flip one equipment slot and answer with the 0x74 state.

    The toggle is free and server-authoritative: the response
    carries all five enabled flags (the wire's documented
    ``t + 5 bytes`` shape).

    Args:
        world: Simulated world.
        tank_id: The toggling tank.
        slot: Equipment slot, 1-5 (out-of-range presses are the
            client's problem and are ignored like the real UI).
        messages: This tick's outgoing batch (appended).
    """
    tank = world["tanks"][tank_id]
    if 1 <= slot <= len(tank["enabled"]):
        tank["enabled"][slot - 1] = not tank["enabled"][slot - 1]
    messages.append(EquipmentToggleDict(msg_type=0x74, enabled=list(tank["enabled"])))


__all__ = [
    "emit_block_action",
    "emit_equipment_pickup",
    "emit_equipment_toggle",
    "emit_mine_press",
    "emit_move",
    "emit_radar",
    "emit_teleport",
]
