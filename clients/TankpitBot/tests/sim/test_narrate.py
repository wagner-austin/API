"""Observer semantics: what each connection is told about one action.

Every test here resolves an action ONCE and narrates it twice — for
the actor and for a bystander — because that is the property the whole
resolve/narrate split exists to make true. A narrator that ignored its
``observer_id`` would pass every single-client test in the suite and
leak another player's private receipts the moment a second connection
existed ([[recipient-policy]]).

The bystander is a real second tank in the world, not a fabricated id,
so a narrator that reads world state for the observer still resolves.
"""

from __future__ import annotations

from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_INVENTORY_FULL
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.sim.actions import process_mine_press, process_radar, process_teleport
from tankpit_bot.sim.blocks import process_block_press
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.equipment import (
    EquipmentGrantDict,
    resolve_equipment_pickup,
    toggle_equipment_slot,
)
from tankpit_bot.sim.fuel_pickup import resolve_fuel_pickup
from tankpit_bot.sim.movement import process_move
from tankpit_bot.sim.narrate import (
    narrate_block_action,
    narrate_chat,
    narrate_equipment_pickup,
    narrate_equipment_toggle,
    narrate_fuel_pickup,
    narrate_mine_press,
    narrate_move,
    narrate_radar,
    narrate_teleport,
)
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimEquipmentDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
    place_mine,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

ACTOR = 9
BYSTANDER = 11


def _world() -> SimWorldDict:
    """Actor 9 at (10, 10) and bystander 11 at (30, 30), both alive."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][ACTOR] = make_sim_tank(ACTOR, 0, 1, 10, 10, 1000)
    world["tanks"][BYSTANDER] = make_sim_tank(BYSTANDER, 1, 1, 30, 30, 1000)
    return world


def _kinds(messages: list[BinaryMessage]) -> list[int | str]:
    """The msg_type of every message, in emission order."""
    return [message["msg_type"] for message in messages]


def _chat() -> ClientCommandDict:
    """A decoded chat command."""
    return ClientCommandDict(
        kind="chat",
        command=77,
        x=1,
        y=2,
        target_id=0,
        slot=0,
        message_id=3,
        direction=0,
        amount=0,
    )


def test_radar_results_reach_only_the_scanner() -> None:
    """0x4F and 0x46 are per-recipient: 0 zero-trigger arrivals in 341
    archived sessions ([[recipient-policy]])."""
    world = _world()
    outcome = process_radar(world, ACTOR, None)

    assert _kinds(narrate_radar(world, outcome, ACTOR)) == [0x4F, 0x46]
    assert narrate_radar(world, outcome, BYSTANDER) == []


def test_radar_extra_consumption_snapshot_leads_the_results() -> None:
    """The 0x49 leads: live radar windows are 84% ``49+4F+46``."""
    world = _world()
    world["tanks"][ACTOR]["counts"][4] = 3
    outcome = process_radar(world, ACTOR, None)
    assert outcome["consumed_extra"] is True
    assert _kinds(narrate_radar(world, outcome, ACTOR)) == [0x49, 0x4F, 0x46]


def test_mine_placement_is_private_and_detonation_broadcasts() -> None:
    """0x4B names the placer only; 0x45 reaches the room.

    The archive's 23 placements every one name the capturing client,
    against 296 detonations ([[mine-mechanics]]).
    """
    world = _world()
    world["mines"]["11,10"] = {"x": 11, "y": 10, "team": 3}
    outcome = process_mine_press(world, InMemoryTerrainMap(), ACTOR)
    assert outcome["placed"]
    assert outcome["detonated"] == [(11, 10)]

    assert _kinds(narrate_mine_press(outcome, ACTOR)) == [0x4B, 0x45]
    assert _kinds(narrate_mine_press(outcome, BYSTANDER)) == [0x45]


def test_equipment_toggle_reaches_only_the_toggler() -> None:
    """0x74 names no tank, so it can only describe its recipient."""
    world = _world()
    toggle_equipment_slot(world, ACTOR, 2)

    assert _kinds(narrate_equipment_toggle(world, ACTOR, ACTOR)) == [0x74]
    assert narrate_equipment_toggle(world, ACTOR, BYSTANDER) == []


def test_an_out_of_range_toggle_changes_nothing() -> None:
    """The real UI has no sixth button; the press is ignored, not refused."""
    world = _world()
    before = list(world["tanks"][ACTOR]["enabled"])
    toggle_equipment_slot(world, ACTOR, 6)
    assert world["tanks"][ACTOR]["enabled"] == before


def test_equipment_grant_is_private_to_the_arriving_tank() -> None:
    """0x67 is a SELF gain in production, so a bystander hears nothing."""
    world = _world()
    world["equipment"].append(SimEquipmentDict(x=10, y=10))
    granted = EquipmentGrantDict(kind="granted", gained=[7, 0, 0, 0, 0])
    assert resolve_equipment_pickup(world, ACTOR) == granted

    actor_view = narrate_equipment_pickup(world, granted, ACTOR, "move", ACTOR)
    assert _kinds(actor_view) == [0x67, 0x49, "container_pickup"]
    gain = actor_view[0]
    assert gain["msg_type"] == 0x67
    assert gain["gained"] == [7, 0, 0, 0, 0]
    assert narrate_equipment_pickup(world, granted, ACTOR, "move", BYSTANDER) == []


def test_a_full_inventory_refuses_only_an_explicit_click() -> None:
    """An incidental arrival at full inventory is silent; a click is not."""
    world = _world()
    world["tanks"][ACTOR]["counts"] = [25, 25, 25, 25, 25]
    world["equipment"].append(SimEquipmentDict(x=10, y=10))
    refused = EquipmentGrantDict(kind="inventory_full", gained=[0, 0, 0, 0, 0])
    assert resolve_equipment_pickup(world, ACTOR) == refused

    assert narrate_equipment_pickup(world, refused, ACTOR, "move", ACTOR) == []
    clicked = narrate_equipment_pickup(world, refused, ACTOR, "pickup_equipment", ACTOR)
    assert _kinds(clicked) == [0x52]
    close = clicked[0]
    assert close["msg_type"] == 0x52
    assert close["error_code"] == SUPERVISOR_ERROR_INVENTORY_FULL


def test_a_cant_go_close_reaches_only_the_walker() -> None:
    """The walk echo is the room's; the 0x52 receipt is the actor's."""
    world = _world()
    world["tanks"][BYSTANDER]["x"] = 11
    world["tanks"][BYSTANDER]["y"] = 10
    # Walking ONTO an occupied tile refuses; walking PAST one routes
    # around it, which is the server's own behaviour ([[walk-mechanics]]).
    outcome = process_move(world, InMemoryTerrainMap(), ACTOR, 11, 10)
    assert outcome["kind"] == "cant_go"

    assert 0x52 in _kinds(narrate_move(world, outcome, ACTOR))
    assert 0x52 not in _kinds(narrate_move(world, outcome, BYSTANDER))


def test_a_teleport_refusal_reaches_only_the_hopper() -> None:
    """An unaffordable hop answers the actor and nobody else."""
    world = _world()
    world["tanks"][ACTOR]["fuel"] = 1
    outcome = process_teleport(world, InMemoryTerrainMap(), ACTOR, 200, 200)
    assert outcome["kind"] == "insufficient_fuel"

    assert _kinds(narrate_teleport(world, outcome, ACTOR)) == [0x52]
    assert narrate_teleport(world, outcome, BYSTANDER) == []


def test_a_blocked_hop_confirms_the_origin_to_the_hopper_only() -> None:
    """The measured refusal law: position + landed, never a CANT_GO."""
    world = _world()
    # A hop is refused only when the target AND every displacement tile
    # the server would try are unavailable: the target carries a mine
    # and the ring is sealed rock ([[teleport-mechanics]]).
    place_mine(world, 60, 60, 1)
    sealed = {
        (60, 59): "#",
        (60, 61): "#",
        (59, 60): "#",
        (61, 60): "#",
    }
    outcome = process_teleport(world, InMemoryTerrainMap(terrain_data=sealed), ACTOR, 60, 60)
    assert outcome["kind"] == "blocked"

    assert _kinds(narrate_teleport(world, outcome, ACTOR)) == [0x3D, "teleport_landed"]
    assert narrate_teleport(world, outcome, BYSTANDER) == []


def test_a_click_on_your_own_tile_draws_no_movement_echo() -> None:
    """NO MOVEMENT, NO ECHO — measured 2026-09-02.

    An own-tile click resolves as a "moved" outcome with an EMPTY
    path. Tracking each live capture's own 0x3D position and finding
    every command clicked at exactly that tile: 1,042 of 1,044
    ``pickup_equipment`` own-tile clicks drew NO 0x47. The sim echoed
    every one, which is why `pickup_equipment 67 49 pickup` — 1,324
    live windows — read as a missing law ([[capture-differ]]).

    The pickup records still ride: the click did collect.
    """
    world = _world()
    world["equipment"].append(SimEquipmentDict(x=10, y=10))
    outcome = process_move(world, InMemoryTerrainMap(), ACTOR, 10, 10)
    assert outcome["kind"] == "moved"
    assert outcome["path"] == ""

    assert narrate_move(world, outcome, ACTOR) == []


def test_a_walk_of_even_one_tile_still_echoes() -> None:
    """The law is about movement, not about arriving somewhere new.

    Suppressing the echo whenever the destination was already reached
    would silence real one-step walks; the gate is the PATH.
    """
    world = _world()
    outcome = process_move(world, InMemoryTerrainMap(), ACTOR, 11, 10)
    assert outcome["path"] != ""

    assert _kinds(narrate_move(world, outcome, ACTOR)) == [0x47]


def test_an_own_tile_arrival_pickup_still_reports_its_records() -> None:
    """Silence is only the echo — consumption is still narrated.

    Observers track container drain through the records, so dropping
    them with the echo would hide the collection from the room.
    """
    world = _world()
    world["containers"].append(SimContainerDict(x=10, y=10, volume=500, dotted=True))
    world["tanks"][ACTOR]["fuel"] = 100
    outcome = process_move(world, InMemoryTerrainMap(), ACTOR, 10, 10)
    assert outcome["pickups"]

    assert _kinds(narrate_move(world, outcome, ACTOR)) == [
        "container_pickup",
        "container_pickup",
    ]


def test_a_landing_confirms_to_the_hopper_only() -> None:
    """THE HOLE THE FIRST ONE-GENERATION BASELINE FOUND.

    Refusals and blocked hops were pinned for both observers; a
    SUCCESSFUL landing was pinned only for the actor, so the narrator
    announced every tank's hop to every connection and nothing failed.
    It showed up as wire: 31 of the practice roster's 76 teleport
    windows read ``3Dself landed`` with no leading 0x5A, a shape the
    live archive does not contain once in 10,683 teleport windows
    (2026-09-02). TeleportLanded is per-recipient — 10,541 arrivals
    against 10,683 own commands, ZERO zero-trigger
    ([[recipient-policy]]) — and a foreign tank's new tile reaches the
    client from the membership diff instead.
    """
    world = _world()
    outcome = process_teleport(world, InMemoryTerrainMap(), ACTOR, 60, 60)
    assert outcome["kind"] == "landed"

    assert _kinds(narrate_teleport(world, outcome, ACTOR)) == [0x3D, "teleport_landed"]
    assert narrate_teleport(world, outcome, BYSTANDER) == []


def test_a_landings_auto_pick_records_still_broadcast() -> None:
    """Only the confirm is private — observers still see consumption.

    The records are how another connection learns the container
    drained ([[recipient-policy]]), so gating the landing on the actor
    must not take them with it.
    """
    world = _world()
    world["containers"].append(SimContainerDict(x=60, y=60, volume=500, dotted=True))
    # Fuel is left alone: the hop itself has to be affordable, and its
    # cost is what opens the headroom the landing then picks up into.
    outcome = process_teleport(world, InMemoryTerrainMap(), ACTOR, 60, 60)
    assert outcome["kind"] == "landed"
    assert outcome["pickups"]

    assert _kinds(narrate_teleport(world, outcome, ACTOR)) == [
        0x3D,
        "teleport_landed",
        "container_pickup",
        "container_pickup",
    ]
    assert _kinds(narrate_teleport(world, outcome, BYSTANDER)) == [
        "container_pickup",
        "container_pickup",
    ]


def test_fuel_pickup_records_broadcast_and_the_close_does_not() -> None:
    """Observers track consumption through the records; the 0x44 and
    the 0x52 close are per-connection ([[fuel-system]])."""
    world = _world()
    world["containers"].append(SimContainerDict(x=10, y=10, volume=500, dotted=True))
    world["tanks"][ACTOR]["fuel"] = 100
    outcome = resolve_fuel_pickup(world, ACTOR, 10, 10, volume_before=500, walked=True)

    bystander = narrate_fuel_pickup(outcome, BYSTANDER)
    assert set(_kinds(bystander)) == {"container_pickup"}
    actor = narrate_fuel_pickup(outcome, ACTOR)
    assert 0x52 in _kinds(actor)


def test_block_action_broadcasts_and_its_refusal_does_not() -> None:
    """0x42 and 0x4A are broadcast; the 0x52 answers the presser."""
    world = _world()
    refused = process_block_press(world, InMemoryTerrainMap(), ACTOR, 200, 200)
    assert refused["kind"] == "out_of_reach"

    assert _kinds(narrate_block_action(world, refused, ACTOR, ACTOR)) == [0x52]
    assert narrate_block_action(world, refused, ACTOR, BYSTANDER) == []


def test_chat_reaches_every_observer_identically() -> None:
    """The echo is the sender's delivery receipt AND the room's copy."""
    assert narrate_chat(ACTOR, _chat()) == narrate_chat(ACTOR, _chat())
    assert _kinds(narrate_chat(ACTOR, _chat())) == [0x4D]
