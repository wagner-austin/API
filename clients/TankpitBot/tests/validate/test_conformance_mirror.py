"""The archive mirror records what the real server stated and anchors a sim world to it."""

from __future__ import annotations

from tankpit_bot.container.types import ContainerPickupDict, ContainerPickupRecordDict
from tankpit_bot.protocol.types import (
    DeactivationDict,
    EquipmentToggleDict,
    FuelGainDict,
    InventoryDict,
    MovementDict,
    MovementResponseDict,
    RadarContainerDict,
    RadarResultDict,
    RadarScanResultDict,
    TankInfoDict,
    TankStatusSyncDict,
    ViewportUpdateDict,
)
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimEquipmentDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)
from tankpit_bot.validate.conformance_mirror import EQUIPMENT_VOLUME, ArchiveMirror

SELF = 1301
RIVAL = 517
CLIENT = 9


def _info(tank_id: int) -> TankInfoDict:
    return TankInfoDict(
        msg_type=0x21,
        tank_id=tank_id,
        team=2,
        decoration_state=bytes(4),
        persistent_tank_id=1,
        name="red-9",
    )


def _at(tank_id: int, x: int, y: int) -> MovementResponseDict:
    return MovementResponseDict(
        msg_type=0x3D,
        team=2,
        tank_id=tank_id,
        x=x,
        y=y,
        direction=8,
        damage_state=3,
        rank=1,
        lb_score=0,
        carrying=0,
    )


def _walk(tank_id: int, waypoints: list[tuple[int, int]]) -> MovementDict:
    return MovementDict(
        msg_type=0x47,
        tank_id=tank_id,
        start_x=1,
        start_y=1,
        direction=0,
        damage_state=3,
        lb_score=0,
        rank=1,
        flag=0,
        is_carrying=False,
        waypoints=waypoints,
        path_tiles=len(waypoints),
        path="e" * len(waypoints),
    )


def _killed(tank_id: int) -> DeactivationDict:
    return DeactivationDict(
        msg_type=0x41,
        status=0,
        victim_id=tank_id,
        promo_eligible=False,
        killer_id=0,
        is_mine_kill=False,
    )


def _sync(tank_id: int, rank: int, fuel: int | None) -> TankStatusSyncDict:
    return TankStatusSyncDict(
        msg_type=0x2E,
        subtype=2,
        tank_id=tank_id,
        damage_state=3,
        rank=rank,
        lb_score=0,
        promo_state=None if fuel is None else 0,
        promo_bar_lit=None if fuel is None else True,
        fuel=fuel,
    )


def _pickup(x: int, y: int, remaining: int) -> ContainerPickupDict:
    return ContainerPickupDict(
        msg_type="container_pickup",
        pickups=(ContainerPickupRecordDict(x=x, y=y, remaining_volume=remaining),),
    )


def _world() -> SimWorldDict:
    """A world holding the sim client, one rival and some containers."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][CLIENT] = make_sim_tank(CLIENT, 2, 1, 50, 50, 900)
    world["tanks"][RIVAL] = make_sim_tank(RIVAL, 1, 1, 60, 60, 900)
    world["containers"] = [SimContainerDict(x=10, y=10, volume=500, dotted=True)]
    world["equipment"] = [SimEquipmentDict(x=20, y=20), SimEquipmentDict(x=21, y=21)]
    return world


def test_a_fresh_mirror_anchors_nothing_but_a_living_client() -> None:
    """Before the server has stated anything, the sim keeps its own state."""
    world = _world()
    client = world["tanks"][CLIENT]
    client["alive"] = False
    mirror = ArchiveMirror()
    assert mirror.self_id is None
    assert mirror.window is None
    mirror.anchor(world, client)
    assert (client["x"], client["y"], client["fuel"], client["alive"]) == (50, 50, 900, True)
    assert world["containers"] == [SimContainerDict(x=10, y=10, volume=500, dotted=True)]


def test_the_first_introduction_names_the_client() -> None:
    """Only the first 0x21 is the capturing client; later ones are others."""
    mirror = ArchiveMirror()
    mirror.observe(_info(SELF))
    mirror.observe(_info(RIVAL))
    assert mirror.self_id == SELF


def test_own_statements_anchor_the_client_tank() -> None:
    """Tile, fuel, rank, counts, toggles and window all reach the sim client."""
    mirror = ArchiveMirror()
    mirror.observe(_info(SELF))
    mirror.observe(_at(SELF, 30, 31))
    mirror.observe(_walk(SELF, [(32, 33)]))
    mirror.observe(_walk(SELF, []))
    mirror.observe(FuelGainDict(msg_type=0x44, fuel_total=700, is_free=False, flag=0))
    mirror.observe(_sync(SELF, rank=4, fuel=650))
    mirror.observe(_sync(SELF, rank=5, fuel=None))
    mirror.observe(_sync(RIVAL, rank=9, fuel=10))
    mirror.observe(
        InventoryDict(
            msg_type=0x49,
            show=False,
            alternate=False,
            counts=[1, 2, 3, 4, 5],
            enabled=[True, False, True, False, True],
        )
    )
    mirror.observe(EquipmentToggleDict(msg_type=0x74, enabled=[False] * 5))
    mirror.observe(
        ViewportUpdateDict(msg_type=0x5A, viewport_left=24, viewport_top=25, entities=[])
    )
    mirror.observe(RadarResultDict(msg_type=0x46, detection_type=0, found=True))
    world = _world()
    client = world["tanks"][CLIENT]
    mirror.anchor(world, client)
    assert (client["x"], client["y"]) == (32, 33)
    assert (client["fuel"], client["rank"]) == (650, 5)
    assert client["counts"] == [1, 2, 3, 4, 5]
    assert client["enabled"] == [False] * 5
    assert mirror.window == (24, 25)


def test_a_killed_client_is_anchored_dead_until_placed_again() -> None:
    """A 0x41 naming the client kills it; its next placement revives it."""
    mirror = ArchiveMirror()
    mirror.observe(_info(SELF))
    mirror.observe(_killed(SELF))
    world = _world()
    client = world["tanks"][CLIENT]
    mirror.anchor(world, client)
    assert client["alive"] is False
    mirror.observe(_at(SELF, 40, 40))
    mirror.anchor(world, client)
    assert client["alive"] is True


def test_other_tanks_are_anchored_where_and_whether_stated() -> None:
    """A stated rival moves and dies; unknown and colliding ids are left alone."""
    mirror = ArchiveMirror()
    mirror.observe(_info(SELF))
    mirror.observe(_killed(4444))
    mirror.observe(_at(RIVAL, 70, 71))
    mirror.observe(_killed(RIVAL))
    mirror.observe(_at(8888, 1, 1))
    mirror.observe(_at(CLIENT, 2, 2))
    world = _world()
    client = world["tanks"][CLIENT]
    mirror.anchor(world, client)
    rival = world["tanks"][RIVAL]
    assert (rival["x"], rival["y"], rival["alive"]) == (70, 71, False)
    assert 8888 not in world["tanks"]
    assert 4444 not in world["tanks"]
    assert (client["x"], client["y"]) == (50, 50)


def test_containers_are_anchored_from_radar_and_pickup_records() -> None:
    """Radar states fuel and equipment; pickups drain fuel and consume equipment."""
    mirror = ArchiveMirror()
    mirror.observe(
        RadarScanResultDict(
            msg_type=0x4F,
            containers=[
                RadarContainerDict(x=10, y=10, volume=420),
                RadarContainerDict(x=11, y=11, volume=900),
                RadarContainerDict(x=20, y=20, volume=EQUIPMENT_VOLUME),
                RadarContainerDict(x=22, y=22, volume=EQUIPMENT_VOLUME),
                RadarContainerDict(x=23, y=23, volume=EQUIPMENT_VOLUME),
            ],
            mines=[],
            mine_clears=[],
        )
    )
    mirror.observe(_pickup(11, 11, 300))
    mirror.observe(_pickup(12, 12, 50))
    mirror.observe(_pickup(23, 23, 0))
    mirror.observe(_pickup(22, 22, 5))
    world = _world()
    mirror.anchor(world, world["tanks"][CLIENT])
    assert world["containers"] == [
        SimContainerDict(x=10, y=10, volume=420, dotted=True),
        SimContainerDict(x=11, y=11, volume=300, dotted=False),
        SimContainerDict(x=12, y=12, volume=50, dotted=False),
    ]
    assert world["equipment"] == [
        SimEquipmentDict(x=20, y=20),
        SimEquipmentDict(x=21, y=21),
        SimEquipmentDict(x=22, y=22),
    ]
