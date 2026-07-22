"""World-state constructors and codecs."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue

from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.sim.world import (
    SimBlockDict,
    SimContainerDict,
    SimEquipmentDict,
    SimFerryDict,
    SimMineDict,
    decode_sim_tank,
    decode_sim_world,
    encode_sim_tank,
    encode_sim_world,
    make_sim_tank,
    make_sim_world,
)


def test_make_sim_tank_clamps_fuel_to_rank_capacity() -> None:
    """Starting fuel above capacity clamps; below passes through."""
    over = make_sim_tank(9, 0, 1, 10, 10, 99999)
    assert over["fuel"] == fuel_capacity(1)
    under = make_sim_tank(9, 0, 1, 10, 10, 500)
    assert under["fuel"] == 500
    assert under["alive"] is True
    assert under["counts"] == [0, 0, 0, 0, 0]


def test_tank_codec_round_trip() -> None:
    """encode/decode of one tank is lossless."""
    tank = make_sim_tank(1301, 2, 3, 42, 161, 900)
    tank["counts"] = [5, 10, 0, 3, 2]
    tank["enabled"] = [True, False, True, True, True]
    assert decode_sim_tank(encode_sim_tank(tank)) == tank


def test_world_codec_round_trip() -> None:
    """encode/decode of a populated world is lossless."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 5, 5, 1000)
    world["containers"].append(SimContainerDict(x=3, y=4, volume=200))
    world["mines"].append(SimMineDict(x=7, y=8, team=2))
    world["equipment"].append(SimEquipmentDict(x=9, y=2))
    world["ferries"].append(SimFerryDict(x=11, y=6))
    world["blocks"].append(SimBlockDict(x=2, y=14))
    world["tick"] = 12
    assert decode_sim_world(encode_sim_world(world)) == world


def test_decode_tank_rejects_malformed_lists() -> None:
    """Equipment lists must be exactly five correctly-typed entries."""
    good = encode_sim_tank(make_sim_tank(9, 0, 1, 5, 5, 1000))
    cases: list[tuple[str, JSONValue]] = [
        ("counts", [1, 2, 3]),
        ("counts", "nope"),
        ("counts", [1, 2, 3, 4, True]),
        ("counts", [1, 2, 3, 4, "x"]),
        ("enabled", [True, True]),
        ("enabled", [True, True, True, True, 1]),
    ]
    for key, bad in cases:
        broken = dict(good)
        broken[key] = bad
        with pytest.raises(ValueError):
            decode_sim_tank(broken)


def test_decode_world_rejects_malformed_sections() -> None:
    """Every top-level world section is validated."""
    good = encode_sim_world(make_sim_world("field01_r.gif"))
    cases: list[tuple[str, JSONValue]] = [
        ("field", 7),
        ("tanks", "nope"),
        ("containers", 5),
        ("mines", {}),
        ("equipment", "nope"),
        ("ferries", 3),
    ]
    for key, bad in cases:
        broken = dict(good)
        broken[key] = bad
        with pytest.raises(ValueError):
            decode_sim_world(broken)
