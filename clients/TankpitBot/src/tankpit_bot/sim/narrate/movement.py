"""Narration for the position-changing commands.

Pure functions from a resolved outcome to the messages ONE observer
receives. Nothing here mutates the world: the resolvers own every
effect, and these decide only what a given connection is told about it
([[recipient-policy]]).

``observer_id`` is the connection being narrated FOR, and is the only
thing that varies across a fan-out. ``outcome["tank_id"]`` is the tank
that acted. When the two are equal the observer is the actor and
receives its own private receipts; otherwise it sees only what the
room sees.
"""

from __future__ import annotations

from tankpit_bot.container.types import (
    ContainerPickupDict,
    ContainerPickupRecordDict,
    MineDetonationDict,
    TeleportLandedDict,
)
from tankpit_bot.physics.supervisor import fuel_pickup_close_code
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_GO,
    SUPERVISOR_ERROR_EMPTY_CONTAINER,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
)
from tankpit_bot.protocol.types import (
    BinaryMessage,
    FuelGainDict,
    SupervisorDict,
)
from tankpit_bot.sim.actions import TeleportOutcomeDict
from tankpit_bot.sim.fuel_pickup import FuelPickupOutcomeDict
from tankpit_bot.sim.movement import MoveOutcomeDict, PickupRecordDict
from tankpit_bot.sim.wire_statements import movement_echo, position_statement
from tankpit_bot.sim.world import SimWorldDict

# TeleportLanded's 1-byte body observed in production captures.
TELEPORT_LANDED_SUBTYPE = 0x0C


def pickup_message(pickups: list[PickupRecordDict]) -> ContainerPickupDict:
    """Wrap resolved pickups in the container-pickup wire message.

    Args:
        pickups: The resolved pickup records.

    Returns:
        The container-pickup message carrying them.
    """
    return ContainerPickupDict(
        msg_type="container_pickup",
        pickups=tuple(
            ContainerPickupRecordDict(x=p["x"], y=p["y"], remaining_volume=p["remaining_volume"])
            for p in pickups
        ),
    )


def narrate_move(
    world: SimWorldDict,
    outcome: MoveOutcomeDict,
    observer_id: int,
    *,
    include_pickups: bool = True,
) -> list[BinaryMessage]:
    """Narrate one resolved move to a single observer.

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
        world: Simulated world, post-move. Read only.
        outcome: The move's resolved outcome.
        observer_id: The connection being narrated for.
        include_pickups: False when an explicit fuel-pickup command's
            choreography (:func:`narrate_fuel_pickup`) owns the
            records instead.

    Returns:
        The messages this observer receives, in emission order.
    """
    messages: list[BinaryMessage] = []
    if outcome["path"]:
        # NO MOVEMENT, NO ECHO. A click on the tile the tank already
        # occupies resolves as a "moved" outcome with an EMPTY path,
        # and the real server answers it without a 0x47: measured
        # 2026-09-02 by tracking each capture's own 0x3D position and
        # finding every command clicked at exactly that tile — 1,042
        # of 1,044 own-tile ``pickup_equipment`` clicks drew no echo.
        # (``pickup_fuel`` reads 23 silent to 9, and plain ``move``
        # 22 to 21; the residuals are consistent with a stale tracked
        # position — a click that looked own-tile because the last
        # 0x3D had not caught up — but only the equipment ratio is
        # lopsided enough to stand on alone.) The sim echoed every
        # one, which is why `pickup_equipment 67 49 pickup` — 1,324
        # live windows — read as a missing law ([[capture-differ]]).
        messages.append(movement_echo(world, outcome))
    if outcome["kind"] == "moved" or outcome["path"]:
        for x, y in outcome["mine_positions"]:
            messages.append(MineDetonationDict(msg_type=0x45, positions=[(x, y)]))
        if include_pickups and outcome["pickups"]:
            messages.append(pickup_message(list(outcome["pickups"])))
            messages.append(pickup_message(list(outcome["pickups"])))
    unfinished_transition = outcome["stop_reason"] == "transition" and not outcome["dest_reached"]
    if (outcome["kind"] == "cant_go" or unfinished_transition) and (
        outcome["tank_id"] == observer_id
    ):
        messages.append(
            SupervisorDict(
                msg_type=0x52,
                reset_action=1,
                close_map=0,
                error_code=SUPERVISOR_ERROR_CANT_GO,
            )
        )
    return messages


def narrate_teleport(
    world: SimWorldDict,
    outcome: TeleportOutcomeDict,
    observer_id: int,
) -> list[BinaryMessage]:
    """Narrate one resolved teleport to a single observer.

    Wire order law: the SelfMovement position update PRECEDES the
    landed confirm on the real wire — the displacement receipt
    (``_emit_teleport_displacement``) reads the self position AT
    confirm time as the landed tile. The pre-2026-08-01 sim sent the
    confirm first, so every exact landing compared the OLD position
    against the request, read as a displacement, and spuriously
    consumed ferry beliefs the landing had just proven TRUE.

    The refusal law, mined 2026-08-21 (137/137 archived receipts,
    8,718 landed vs 4 rejected teleports overall): a fully
    ring-blocked hop is NOT answered with 0x52 CANT_GO — the real
    server confirms the position AT THE ORIGIN, uncharged, and the
    client perceives "landed where I stood". The pre-correction sim
    sent CANT_GO here, a wire shape the live server never produces
    for teleports ([[teleport-mechanics]] § the refusal law).

    A LANDING IS PER-RECIPIENT, measured 2026-09-02: 10,541
    TeleportLanded arrived against 10,683 own teleport commands, with
    ZERO zero-trigger arrivals across 341 sessions
    ([[recipient-policy]]) — another tank's hop is never announced to
    this connection. The position statement is withheld for the same
    reason the join burst withholds one: a foreign tank's new tile
    reaches the client from the end-of-tick membership diff, and
    stating it here as well would DOUBLE it, the identical trap
    ``SimServer.relocate_tank`` documents.

    The sim narrated every landing to every observer until the first
    one-generation baseline caught it (2026-09-02): 31 of the practice
    roster's 76 teleport windows read ``3Dself landed`` with no
    leading 0x5A — a shape the live archive does not contain once in
    10,683 teleport windows, because on the real wire a confirm only
    ever follows the connection's OWN hop. Only the practice scenario
    produced it, being the only one whose bots teleport off.

    The auto-pick records still broadcast: observers track container
    consumption through them ([[recipient-policy]]).

    Args:
        world: Simulated world, post-teleport. Read only.
        outcome: The teleport's resolved outcome.
        observer_id: The connection being narrated for.

    Returns:
        The messages this observer receives, in emission order.
    """
    tank_id = outcome["tank_id"]
    is_actor = tank_id == observer_id
    if outcome["kind"] == "insufficient_fuel":
        if not is_actor:
            return []
        return [
            SupervisorDict(
                msg_type=0x52,
                reset_action=1,
                close_map=1,
                error_code=SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
            )
        ]
    if outcome["kind"] == "blocked":
        if not is_actor:
            return []
        return [
            position_statement(world, tank_id),
            TeleportLandedDict(msg_type="teleport_landed", subtype=TELEPORT_LANDED_SUBTYPE),
        ]
    messages: list[BinaryMessage] = []
    if is_actor:
        messages.append(position_statement(world, tank_id))
        messages.append(
            TeleportLandedDict(msg_type="teleport_landed", subtype=TELEPORT_LANDED_SUBTYPE)
        )
    if outcome["pickups"]:
        # The duplicate-record law: landing auto-picks double their
        # container record (31% of 7,176 live teleports read
        # ``...landed+pickup+pickup``).
        messages.append(pickup_message(list(outcome["pickups"])))
        messages.append(pickup_message(list(outcome["pickups"])))
    return messages


def narrate_fuel_pickup(
    outcome: FuelPickupOutcomeDict,
    observer_id: int,
) -> list[BinaryMessage]:
    """Narrate one explicit fuel-pickup command to a single observer.

    Byte-mined 2026-08-01 from ~1,600 archive windows ([[fuel-system]],
    [[capture-differ]]) — four branches, all closing with a 0x52 the
    production ledger types (code 5 = clamped SUCCESS, code 4 = empty):

    * **transfer + clamp** (tank filled, container keeps a remainder):
      record x2, 0x44 absolute fuel (``is_free=True, flag=0``),
      record x1, code 5 ``reset_action=0``.
    * **transfer + drain** (container empties): record x2 at
      remaining 0, code 4 — ``reset_action=1`` after a walk, 0
      without one. Code 5 never resets, walked or not (2,537/2,537).
    * **no transfer, walked** (arrived to find it empty, or a
      full-tank walk-up): record x2, close by stockedness.
    * **no transfer, no walk** (own-tile/adjacent click): 0x44 in its
      no-gain form (``is_free=False, flag=43`` — the measured bytes),
      record x1, close by stockedness, ``reset_action=0``.

    Records broadcast (observers track consumption through them); the
    0x44 and the 0x52 close are per-connection.

    Args:
        outcome: The pickup's resolved snapshot.
        observer_id: The connection being narrated for.

    Returns:
        The messages this observer receives, in emission order.
    """
    remaining = outcome["remaining"]
    transfer = outcome["volume_before"] - remaining
    is_actor = outcome["tank_id"] == observer_id
    record = pickup_message(
        [PickupRecordDict(x=outcome["x"], y=outcome["y"], remaining_volume=remaining)]
    )
    close_code = fuel_pickup_close_code(remaining)
    messages: list[BinaryMessage] = []

    if transfer > 0 and remaining > 0:
        messages.append(record)
        messages.append(record)
        if is_actor:
            messages.append(
                FuelGainDict(msg_type=0x44, fuel_total=outcome["fuel_total"], is_free=True, flag=0)
            )
        messages.append(record)
        if is_actor:
            messages.append(
                SupervisorDict(msg_type=0x52, reset_action=0, close_map=0, error_code=close_code)
            )
        return messages

    if outcome["walked"] or transfer > 0:
        messages.append(record)
        messages.append(record)
        if is_actor:
            messages.append(
                SupervisorDict(
                    msg_type=0x52,
                    # reset_action follows the CODE, not the walk. The
                    # 2026-09-02 field sweep: code 4 splits (1, 0) x610
                    # against (0, 0) x71 — the walk distinction — while
                    # code 5 is (0, 0) in ALL 2,537 archived windows,
                    # walked or not. [[fuel-system]] already said so
                    # ("code 5, reset_action=0"; "code 4,
                    # reset_action=1 after a walk"); the sim keyed both
                    # on the walk alone and emitted code 5 with
                    # reset_action=1. Caught by the differ the first
                    # time its 0x52 token carried the fields.
                    reset_action=(
                        1
                        if outcome["walked"] and close_code == SUPERVISOR_ERROR_EMPTY_CONTAINER
                        else 0
                    ),
                    close_map=0,
                    error_code=close_code,
                )
            )
        return messages

    if is_actor:
        messages.append(
            FuelGainDict(msg_type=0x44, fuel_total=outcome["fuel_total"], is_free=False, flag=43)
        )
    messages.append(record)
    if is_actor:
        messages.append(
            SupervisorDict(msg_type=0x52, reset_action=0, close_map=0, error_code=close_code)
        )
    return messages


__all__ = [
    "TELEPORT_LANDED_SUBTYPE",
    "narrate_fuel_pickup",
    "narrate_move",
    "narrate_teleport",
    "pickup_message",
]
