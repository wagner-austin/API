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
from tankpit_bot.physics.costs import MINE_PRESS_COST, RADAR_COST
from tankpit_bot.physics.supervisor import fuel_pickup_close_code
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_GO,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
    SUPERVISOR_ERROR_INVENTORY_FULL,
)
from tankpit_bot.protocol.types import (
    BinaryMessage,
    BuildPickupDict,
    ChatMessageDict,
    EquipmentGainDict,
    EquipmentToggleDict,
    FuelGainDict,
    InventoryDict,
    RadarResultDict,
    RadarScanResultDict,
    SupervisorDict,
    TerrainUpdateDict,
)
from tankpit_bot.sim.actions import (
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


def emit_chat(
    tank_id: int,
    command: ClientCommandDict,
    messages: list[BinaryMessage],
) -> None:
    """Echo one chat command as the 0x4D broadcast.

    Mirrors the un-muted real server (sniff-20260729-214411): an
    accepted chat comes back as ``M + sender_id + message_id + x + y``
    to everyone, INCLUDING the sender — the echo is the client's
    delivery receipt. The sim does not model the flood mute; bot
    policy (one greeting per human lock) keeps live sends far below
    the mute threshold.

    Args:
        tank_id: The chatting tank.
        command: The decoded chat command.
        messages: This tick's outgoing batch (appended).
    """
    messages.append(
        ChatMessageDict(
            msg_type=0x4D,
            sender_id=tank_id,
            message_type=command["message_id"],
            x=command["x"],
            y=command["y"],
        )
    )


def emit_move(
    world: SimWorldDict,
    client_id: int,
    outcome: MoveOutcomeDict,
    messages: list[BinaryMessage],
    *,
    include_pickups: bool = True,
) -> None:
    """Emit the wire consequences of one processed move.

    Arrival auto-picks emit their container record TWICE — the
    measured duplicate-record law (2026-08-01 archive: 129 move and
    2,200+ teleport windows all read ``...pickup+pickup``; the real
    server always doubles the record).

    A ``cant_go`` is a partial-walk receipt, not a bare rejection
    (exact-window measure 2026-08-04, 12 live code-1s): when the
    server walked a non-empty prefix before stopping, the 0x47 echo
    for the walked tiles precedes the 0x52 close in the same batch
    (live pairs landed within ±100 ms). The zero-tile pure refusal
    (1 of the 12) emits the bare 0x52 — no echo, nothing moved.

    A surface-transition stop SHORT of the click gets the same code-1
    close even though the walk itself is lawful: the 2026-08-03 run's
    cluster-A collects (bot riding the ferry afloat on (59,28) water,
    land targets inland) each echoed the one-step disembark and then
    the 0x52 — the receipt says "your command did not finish", not
    "your walk was refused". A transition stop that IS the click
    (boarding the clicked ferry tile) closes silently, and a mine
    walk-over arrest closes silently too (18 archive detonations,
    zero paired code-1s).

    Args:
        world: Simulated world (post-move).
        client_id: The connected client's tank id.
        outcome: The move's outcome.
        messages: This tick's outgoing batch (appended).
        include_pickups: False when an explicit fuel-pickup command's
            choreography (:func:`emit_fuel_pickup_close`) owns the
            records instead.
    """
    if outcome["kind"] == "moved" or outcome["path"]:
        messages.append(movement_echo(world, outcome))
        for x, y in outcome["mine_positions"]:
            messages.append(MineDetonationDict(msg_type=0x45, positions=[(x, y)]))
        if include_pickups and outcome["pickups"]:
            messages.append(_pickup_message(list(outcome["pickups"])))
            messages.append(_pickup_message(list(outcome["pickups"])))
    unfinished_transition = outcome["stop_reason"] == "transition" and not outcome["dest_reached"]
    if (outcome["kind"] == "cant_go" or unfinished_transition) and outcome["tank_id"] == client_id:
        messages.append(
            SupervisorDict(
                msg_type=0x52,
                reset_action=1,
                close_map=0,
                error_code=SUPERVISOR_ERROR_CANT_GO,
            )
        )


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
    if outcome["kind"] == "insufficient_fuel":
        if tank_id == client_id:
            messages.append(
                SupervisorDict(
                    msg_type=0x52,
                    reset_action=1,
                    close_map=1,
                    error_code=SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
                )
            )
        return False
    if outcome["kind"] == "blocked":
        # The refusal law, mined 2026-08-21 (137/137 archived
        # receipts, 8,718 landed vs 4 rejected teleports overall): a
        # fully ring-blocked hop is NOT answered with 0x52 CANT_GO —
        # the real server confirms the position AT THE ORIGIN,
        # uncharged, and the client perceives "landed where I stood".
        # The pre-correction sim sent CANT_GO here, a wire shape the
        # live server never produces for teleports, which would have
        # steered the bot down the rejection path instead of the
        # landing-refusal evidence path ([[teleport-mechanics]] § the
        # refusal law).
        if tank_id == client_id:
            messages.append(position_statement(world, tank_id))
            messages.append(
                TeleportLandedDict(msg_type="teleport_landed", subtype=_TELEPORT_LANDED_SUBTYPE)
            )
        return False
    # Wire order law: the SelfMovement position update PRECEDES the
    # landed confirm on the real wire — the displacement receipt
    # (`_emit_teleport_displacement`) reads the self position AT
    # confirm time as the landed tile. The pre-2026-08-01 sim sent
    # the confirm first, so every exact landing compared the OLD
    # position against the request, read as a displacement, and
    # spuriously consumed ferry beliefs (`_expire_disproven_ferry_
    # belief`) the landing had just proven TRUE.
    messages.append(position_statement(world, tank_id))
    messages.append(
        TeleportLandedDict(msg_type="teleport_landed", subtype=_TELEPORT_LANDED_SUBTYPE)
    )
    if outcome["pickups"]:
        # The duplicate-record law: landing auto-picks double their
        # container record (31% of 7,176 live teleports read
        # ``...landed+pickup+pickup``).
        messages.append(_pickup_message(list(outcome["pickups"])))
        messages.append(_pickup_message(list(outcome["pickups"])))
    return True


def emit_fuel_pickup_close(
    world: SimWorldDict,
    client_id: int,
    tank_id: int,
    x: int,
    y: int,
    *,
    volume_before: int,
    walked: bool,
    messages: list[BinaryMessage],
) -> None:
    """Answer an explicit fuel-pickup command with the measured choreography.

    Byte-mined 2026-08-01 from ~1,600 archive windows ([[fuel-system]],
    [[capture-differ]]) — four branches, all closing with a 0x52 the
    production ledger types (code 5 = clamped SUCCESS, code 4 = empty):

    * **transfer + clamp** (tank filled, container keeps a remainder):
      record x2, 0x44 absolute fuel (``is_free=True, flag=0``),
      record x1, code 5 ``reset_action=0``.
    * **transfer + drain** (container empties): record x2 at
      remaining 0, code 4 — ``reset_action=1`` after a walk.
    * **no transfer, walked** (arrived to find it empty, or a
      full-tank walk-up): record x2, close by stockedness.
    * **no transfer, no walk** (own-tile/adjacent click): 0x44 in its
      no-gain form (``is_free=False, flag=43`` — the measured bytes),
      record x1, close by stockedness, ``reset_action=0``.

    Records broadcast (observers track consumption through them); the
    0x44 and the 0x52 close are per-connection.

    Args:
        world: Simulated world (post-move).
        client_id: The connected client's tank id.
        tank_id: The picking tank.
        x: The clicked container tile X.
        y: The clicked container tile Y.
        volume_before: The container's volume before the command.
        walked: Whether the command's walk covered any tiles.
        messages: This tick's outgoing batch (appended).
    """
    remaining = 0
    for container in world["containers"]:
        if (container["x"], container["y"]) == (x, y):
            remaining = container["volume"]
            break
    transfer = volume_before - remaining
    is_client = tank_id == client_id
    record = _pickup_message([PickupRecordDict(x=x, y=y, remaining_volume=remaining)])
    tank = world["tanks"][tank_id]
    close_code = fuel_pickup_close_code(remaining)
    if transfer > 0 and remaining > 0:
        messages.append(record)
        messages.append(record)
        if is_client:
            messages.append(
                FuelGainDict(msg_type=0x44, fuel_total=tank["fuel"], is_free=True, flag=0)
            )
        messages.append(record)
        if is_client:
            messages.append(
                SupervisorDict(msg_type=0x52, reset_action=0, close_map=0, error_code=close_code)
            )
        return
    if walked or transfer > 0:
        messages.append(record)
        messages.append(record)
        if is_client:
            messages.append(
                SupervisorDict(
                    msg_type=0x52,
                    reset_action=1 if walked else 0,
                    close_map=0,
                    error_code=close_code,
                )
            )
        return
    if is_client:
        messages.append(
            FuelGainDict(msg_type=0x44, fuel_total=tank["fuel"], is_free=False, flag=43)
        )
    messages.append(record)
    if is_client:
        messages.append(
            SupervisorDict(msg_type=0x52, reset_action=0, close_map=0, error_code=close_code)
        )


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
    tank["fuel"] = max(0, tank["fuel"] - RADAR_COST)
    del ammo_changed
    if outcome["consumed_extra"] and tank_id == client_id:
        # The extra-consumption snapshot LEADS the scan results —
        # live radar windows are 84% ``49+4F+46`` (response-shape
        # differ 2026-08-01); the sim's end-of-tick snapshot had it
        # trailing.
        messages.append(
            InventoryDict(
                msg_type=0x49,
                show=False,
                alternate=False,
                counts=list(tank["counts"]),
                enabled=list(tank["enabled"]),
            )
        )
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
    client_id: int,
    tank_id: int,
    messages: list[BinaryMessage],
) -> None:
    """Process one mine press and emit its wire consequences.

    The 0x4B placement is PER-RECIPIENT — it is the placer's own
    receipt, the same discipline the 0x52 rejections and the fuel sync
    follow. Every one of the 23 placements in the whole archive
    (``runs/bot`` plus ``runs/sniff``) names tank 1301, the capturing
    client; not one reports another player's press, however heavily
    they mined. Other players' mines reach the client as overlay and
    radar reveals instead ([[mine-mechanics]],
    [[session-state-deglobalisation]]).

    The 0x45 detonation is NOT per-recipient: it broadcasts, which is
    why the archive carries 296 of them against 23 placements.

    Args:
        world: Simulated world.
        terrain: Static terrain.
        client_id: The connected client's tank id.
        tank_id: The placing tank.
        messages: This tick's outgoing batch (appended).
    """
    outcome = process_mine_press(world, terrain, tank_id)
    tank = world["tanks"][tank_id]
    tank["fuel"] = max(0, tank["fuel"] - MINE_PRESS_COST)
    if outcome["placed"] and tank_id == client_id:
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
            # Live order (2,170 archive windows): the 0x67 gain, its
            # 0x49 snapshot, then the container-pickup record closing
            # the drained container (remaining 0) — the sim had no
            # pickup record at all (response-shape differ 2026-08-01).
            tank = world["tanks"][tank_id]
            messages.append(
                EquipmentGainDict(msg_type=0x67, show_message=True, gained=grant["gained"])
            )
            messages.append(
                InventoryDict(
                    msg_type=0x49,
                    show=False,
                    alternate=False,
                    counts=list(tank["counts"]),
                    enabled=list(tank["enabled"]),
                )
            )
            messages.append(
                _pickup_message([PickupRecordDict(x=tank["x"], y=tank["y"], remaining_volume=0)])
            )
        del ammo_changed
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
    "emit_chat",
    "emit_equipment_pickup",
    "emit_equipment_toggle",
    "emit_mine_press",
    "emit_move",
    "emit_radar",
    "emit_teleport",
]
