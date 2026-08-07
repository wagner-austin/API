"""World-state constructors and codecs."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue, narrow_json_to_list

from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.sim.world import (
    SimBlockDict,
    SimContainerDict,
    SimEquipmentDict,
    SimFerryDict,
    decode_sim_tank,
    decode_sim_world,
    encode_sim_tank,
    encode_sim_world,
    make_sim_tank,
    make_sim_world,
    place_mine,
)


def test_make_sim_tank_clamps_fuel_to_rank_capacity() -> None:
    """Starting fuel above capacity clamps; below passes through."""
    over = make_sim_tank(9, 0, 1, 10, 10, 99999)
    assert over["fuel"] == fuel_capacity(1)
    under = make_sim_tank(9, 0, 1, 10, 10, 500)
    assert under["fuel"] == 500
    assert under["alive"] is True
    assert under["counts"] == [0, 0, 0, 0, 0]


def test_make_sim_tank_names_default_to_the_practice_shape() -> None:
    """An unnamed tank gets the farmable ``red-<id>`` wire name."""
    assert make_sim_tank(7, 1, 1, 5, 5, 500)["name"] == "red-7"


def test_make_sim_tank_accepts_a_human_persona() -> None:
    """A seeded human name survives construction — the consent-gate seam."""
    assert make_sim_tank(7, 1, 1, 5, 5, 500, name="guest")["name"] == "guest"


def test_tank_codec_round_trip() -> None:
    """encode/decode of one tank is lossless."""
    tank = make_sim_tank(1301, 2, 3, 42, 161, 900, name="guest")
    tank["counts"] = [5, 10, 0, 3, 2]
    tank["enabled"] = [True, False, True, True, True]
    assert decode_sim_tank(encode_sim_tank(tank)) == tank


def test_world_codec_round_trip() -> None:
    """encode/decode of a populated world is lossless."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 5, 5, 1000)
    world["containers"].append(SimContainerDict(x=3, y=4, volume=200, dotted=True))
    place_mine(world, 7, 8, 2)
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
        ("revealed_mine_keys_by_team", "nope"),
        ("revealed_mine_keys_by_team", {"0": "10,11"}),
        ("revealed_mine_keys_by_team", {"0": [7]}),
    ]
    for key, bad in cases:
        broken = dict(good)
        broken[key] = bad
        with pytest.raises(ValueError):
            decode_sim_world(broken)


def test_decode_rejects_two_mines_on_one_tile() -> None:
    """Mines are keyed by tile, so a duplicate tile is a corrupt world.

    ``mines`` is a dict keyed by ``"x,y"``, and the decoder builds it
    from a list -- silently letting the second record win would drop a
    mine the encoder had written and make the round trip lossy.
    """
    world = make_sim_world("field01_r.gif")
    place_mine(world, 10, 11, 0)
    encoded = encode_sim_world(world)
    mines = narrow_json_to_list(encoded["mines"])
    encoded["mines"] = [*mines, {"x": 10, "y": 11, "team": 2}]

    with pytest.raises(ValueError, match="two mines on tile 10,11"):
        decode_sim_world(encoded)


def test_revealed_mine_keys_round_trip_per_team() -> None:
    """Team-scoped reveal knowledge survives encode/decode intact."""
    world = make_sim_world("field01_r.gif")
    world["revealed_mine_keys_by_team"]["0"] = ["10,11", "12,13"]
    world["revealed_mine_keys_by_team"]["2"] = ["200,143"]
    decoded = decode_sim_world(encode_sim_world(world))
    assert decoded["revealed_mine_keys_by_team"] == {
        "0": ["10,11", "12,13"],
        "2": ["200,143"],
    }
