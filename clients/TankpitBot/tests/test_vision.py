"""Tests for bot vision module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.bot.vision import (
    ContainerEntryDict,
    PositionEntryDict,
    TankRegistryEntryDict,
    VisionStateDict,
    add_fuel_delta,
    add_tank_to_registry,
    decode_container_entry,
    decode_position_entry,
    decode_tank_registry_entry,
    decode_vision_state,
    encode_container_entry,
    encode_position_entry,
    encode_tank_registry_entry,
    encode_vision_state,
    get_merged_fuel,
    get_merged_fuel_containers,
    make_container_entry,
    make_empty_vision_state,
    make_position_entry,
    make_tank_registry_entry,
    pickup_container_vision,
    remove_container,
    render_vision_ascii,
    render_vision_debug,
    set_self_tank_id,
    update_container,
    update_self_fuel_vision,
    update_tank_position,
)
from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_make_tank_registry_entry(self) -> None:
        entry = make_tank_registry_entry(tank_id=500, name="Player1", team=2)
        assert entry["tank_id"] == 500
        assert entry["name"] == "Player1"
        assert entry["team"] == 2

    def test_make_position_entry(self) -> None:
        entry = make_position_entry(tank_id=501, x=100, y=150)
        assert entry["tank_id"] == 501
        assert entry["x"] == 100
        assert entry["y"] == 150

    def test_make_container_entry(self) -> None:
        entry = make_container_entry(x=50, y=75, volume=300)
        assert entry["x"] == 50
        assert entry["y"] == 75
        assert entry["volume"] == 300

    def test_make_empty_vision_state(self) -> None:
        state = make_empty_vision_state()
        assert state["tank_registry"] == {}
        assert state["position_cache"] == {}
        assert state["container_cache"] == {}
        assert state["self_fuel"] == 1000
        assert state["self_tank_id"] == -1


class TestEncodeFunctions:
    """Tests for encode functions."""

    def test_encode_tank_registry_entry(self) -> None:
        entry: TankRegistryEntryDict = {
            "tank_id": 500,
            "name": "TestPlayer",
            "team": 1,
        }
        encoded = encode_tank_registry_entry(entry)
        assert encoded["tank_id"] == 500
        assert encoded["name"] == "TestPlayer"
        assert encoded["team"] == 1

    def test_encode_position_entry(self) -> None:
        entry: PositionEntryDict = {
            "tank_id": 501,
            "x": 120,
            "y": 80,
        }
        encoded = encode_position_entry(entry)
        assert encoded["tank_id"] == 501
        assert encoded["x"] == 120
        assert encoded["y"] == 80

    def test_encode_container_entry(self) -> None:
        entry: ContainerEntryDict = {
            "x": 30,
            "y": 40,
            "volume": 250,
        }
        encoded = encode_container_entry(entry)
        assert encoded["x"] == 30
        assert encoded["y"] == 40
        assert encoded["volume"] == 250

    def test_encode_vision_state(self) -> None:
        state: VisionStateDict = {
            "tank_registry": {
                "500": {"tank_id": 500, "name": "P1", "team": 0},
            },
            "position_cache": {
                "501": {"tank_id": 501, "x": 10, "y": 20},
            },
            "container_cache": {
                "5,10": {"x": 5, "y": 10, "volume": 100},
            },
            "self_fuel": 800,
            "self_tank_id": 502,
        }
        encoded = encode_vision_state(state)
        assert encoded["self_fuel"] == 800
        assert encoded["self_tank_id"] == 502
        # Verify nested dicts are encoded with content
        tank_reg = encoded.get("tank_registry")
        assert tank_reg == {"500": {"tank_id": 500, "name": "P1", "team": 0}}
        pos_cache = encoded.get("position_cache")
        assert pos_cache == {"501": {"tank_id": 501, "x": 10, "y": 20}}
        cont_cache = encoded.get("container_cache")
        assert cont_cache == {"5,10": {"x": 5, "y": 10, "volume": 100}}


class TestDecodeFunctions:
    """Tests for decode functions."""

    def test_decode_tank_registry_entry(self) -> None:
        data: JSONObject = {"tank_id": 500, "name": "Player", "team": 2}
        entry = decode_tank_registry_entry(data)
        assert entry["tank_id"] == 500
        assert entry["name"] == "Player"
        assert entry["team"] == 2

    def test_decode_tank_registry_entry_missing_field(self) -> None:
        data: JSONObject = {"tank_id": 500, "name": "Player"}
        with pytest.raises(JSONTypeError):
            decode_tank_registry_entry(data)

    def test_decode_position_entry(self) -> None:
        data: JSONObject = {"tank_id": 501, "x": 100, "y": 150}
        entry = decode_position_entry(data)
        assert entry["tank_id"] == 501
        assert entry["x"] == 100
        assert entry["y"] == 150

    def test_decode_position_entry_missing_field(self) -> None:
        data: JSONObject = {"tank_id": 501, "x": 100}
        with pytest.raises(JSONTypeError):
            decode_position_entry(data)

    def test_decode_container_entry(self) -> None:
        data: JSONObject = {"x": 50, "y": 75, "volume": 300}
        entry = decode_container_entry(data)
        assert entry["x"] == 50
        assert entry["y"] == 75
        assert entry["volume"] == 300

    def test_decode_container_entry_missing_field(self) -> None:
        data: JSONObject = {"x": 50, "y": 75}
        with pytest.raises(JSONTypeError):
            decode_container_entry(data)

    def test_decode_vision_state(self) -> None:
        data: JSONObject = {
            "tank_registry": {
                "500": {"tank_id": 500, "name": "P1", "team": 0},
            },
            "position_cache": {
                "501": {"tank_id": 501, "x": 10, "y": 20},
            },
            "container_cache": {
                "5,10": {"x": 5, "y": 10, "volume": 100},
            },
            "self_fuel": 900,
            "self_tank_id": 502,
        }
        state = decode_vision_state(data)
        assert state["self_fuel"] == 900
        assert state["self_tank_id"] == 502
        assert "500" in state["tank_registry"]
        assert "501" in state["position_cache"]
        assert "5,10" in state["container_cache"]

    def test_decode_vision_state_empty_dicts(self) -> None:
        data: JSONObject = {
            "tank_registry": {},
            "position_cache": {},
            "container_cache": {},
            "self_fuel": 1000,
            "self_tank_id": -1,
        }
        state = decode_vision_state(data)
        assert state["tank_registry"] == {}
        assert state["position_cache"] == {}
        assert state["container_cache"] == {}

    def test_decode_vision_state_none_dicts(self) -> None:
        data: JSONObject = {
            "tank_registry": None,
            "position_cache": None,
            "container_cache": None,
            "self_fuel": 1000,
            "self_tank_id": -1,
        }
        state = decode_vision_state(data)
        assert state["tank_registry"] == {}
        assert state["position_cache"] == {}
        assert state["container_cache"] == {}

    def test_decode_vision_state_non_dict_values_in_registry(self) -> None:
        data: JSONObject = {
            "tank_registry": {"500": "invalid"},
            "position_cache": {"501": 123},
            "container_cache": {"5,10": None},
            "self_fuel": 1000,
            "self_tank_id": -1,
        }
        state = decode_vision_state(data)
        # Non-dict values should be skipped
        assert state["tank_registry"] == {}
        assert state["position_cache"] == {}
        assert state["container_cache"] == {}


class TestMutationFunctions:
    """Tests for immutable mutation functions."""

    def test_add_tank_to_registry(self) -> None:
        state = make_empty_vision_state()
        new_state = add_tank_to_registry(state, tank_id=500, name="Player1", team=1)

        # Original unchanged
        assert "500" not in state["tank_registry"]

        # New state updated
        assert "500" in new_state["tank_registry"]
        assert new_state["tank_registry"]["500"]["name"] == "Player1"
        assert new_state["tank_registry"]["500"]["team"] == 1

    def test_update_tank_position(self) -> None:
        state = make_empty_vision_state()
        new_state = update_tank_position(state, tank_id=501, x=100, y=150)

        assert "501" not in state["position_cache"]
        assert "501" in new_state["position_cache"]
        assert new_state["position_cache"]["501"]["x"] == 100
        assert new_state["position_cache"]["501"]["y"] == 150

    def test_update_container(self) -> None:
        state = make_empty_vision_state()
        new_state = update_container(state, x=50, y=75, volume=300)

        assert "50,75" not in state["container_cache"]
        assert "50,75" in new_state["container_cache"]
        assert new_state["container_cache"]["50,75"]["volume"] == 300

    def test_remove_container(self) -> None:
        state = make_empty_vision_state()
        state_with_container = update_container(state, x=50, y=75, volume=300)
        new_state = remove_container(state_with_container, x=50, y=75)

        assert "50,75" in state_with_container["container_cache"]
        assert "50,75" not in new_state["container_cache"]

    def test_remove_container_not_exists(self) -> None:
        state = make_empty_vision_state()
        new_state = remove_container(state, x=99, y=99)
        # Should not raise, just return state without that key
        assert "99,99" not in new_state["container_cache"]

    def test_update_self_fuel_vision(self) -> None:
        state = make_empty_vision_state()
        new_state = update_self_fuel_vision(state, fuel=500)

        assert state["self_fuel"] == 1000
        assert new_state["self_fuel"] == 500

    def test_add_fuel_delta(self) -> None:
        state = make_empty_vision_state()
        new_state = add_fuel_delta(state, delta=200)

        assert state["self_fuel"] == 1000
        assert new_state["self_fuel"] == 1200

    def test_add_fuel_delta_negative(self) -> None:
        state = make_empty_vision_state()
        new_state = add_fuel_delta(state, delta=-300)

        assert new_state["self_fuel"] == 700

    def test_set_self_tank_id(self) -> None:
        state = make_empty_vision_state()
        new_state = set_self_tank_id(state, tank_id=505)

        assert state["self_tank_id"] == -1
        assert new_state["self_tank_id"] == 505

    def test_pickup_container_vision_with_fuel(self) -> None:
        state = make_empty_vision_state()
        state_with_container = update_container(state, x=50, y=75, volume=300)
        new_state = pickup_container_vision(state_with_container, x=50, y=75)

        # Container should be removed
        assert "50,75" not in new_state["container_cache"]
        # Fuel should be added
        assert new_state["self_fuel"] == 1300

    def test_pickup_container_vision_equipment(self) -> None:
        state = make_empty_vision_state()
        state_with_equip = update_container(state, x=50, y=75, volume=0)
        new_state = pickup_container_vision(state_with_equip, x=50, y=75)

        # Container should be removed
        assert "50,75" not in new_state["container_cache"]
        # Fuel unchanged (volume was 0)
        assert new_state["self_fuel"] == 1000

    def test_pickup_container_vision_not_exists(self) -> None:
        state = make_empty_vision_state()
        new_state = pickup_container_vision(state, x=99, y=99)

        # Should not raise, fuel unchanged
        assert new_state["self_fuel"] == 1000


class TestMergeFunctions:
    """Tests for merge functions that combine vision and world state."""

    def setup_method(self) -> None:
        reset_world_state()

    def test_get_merged_fuel_containers_vision_only(self) -> None:
        state = make_empty_vision_state()
        state_with_container = update_container(state, x=50, y=75, volume=300)

        containers = get_merged_fuel_containers(state_with_container)
        assert len(containers) == 1
        assert containers[0]["x"] == 50
        assert containers[0]["y"] == 75
        assert containers[0]["volume"] == 300

    def test_get_merged_fuel_containers_empty(self) -> None:
        state = make_empty_vision_state()
        containers = get_merged_fuel_containers(state)
        assert containers == []

    def test_get_merged_fuel_containers_excludes_equipment(self) -> None:
        state = make_empty_vision_state()
        state_with_equip = update_container(state, x=50, y=75, volume=0)

        containers = get_merged_fuel_containers(state_with_equip)
        assert containers == []

    def test_get_merged_fuel_containers_world_state_wins(self) -> None:
        from tankpit_bot.protocol import RadarContainerDict
        from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar

        # Add container at (50,75) to world state with volume 500
        radar_containers: list[RadarContainerDict] = [
            {"x": 50, "y": 75, "volume": 500},
        ]
        update_world_state_from_radar(get_world_service(), radar_containers, [])

        # Add same location to vision cache with different volume
        state = make_empty_vision_state()
        state_with_dup = update_container(state, x=50, y=75, volume=100)

        containers = get_merged_fuel_containers(state_with_dup)
        # Should have only world state container (volume 500), not vision (100)
        assert len(containers) == 1
        assert containers[0]["volume"] == 500

    def test_get_merged_fuel_with_world_state(self) -> None:
        from tankpit_bot.sniffer.world_state import update_world_state_from_position
        from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total

        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(get_world_service(), 1400)

        state = make_empty_vision_state()
        fuel = get_merged_fuel(state)
        # World state has self_state with fuel from TankStatusSync
        assert fuel == 1400

    def test_get_merged_fuel_vision_fallback(self) -> None:
        state = make_empty_vision_state()
        state = update_self_fuel_vision(state, fuel=750)

        fuel = get_merged_fuel(state)
        # No world state self_state, falls back to vision
        assert fuel == 750


class TestRenderFunctions:
    """Tests for render functions."""

    def setup_method(self) -> None:
        reset_world_state()

    def test_render_vision_ascii(self) -> None:
        # Render ASCII viewport - may return None or string depending on terrain
        result = render_vision_ascii()
        # Either None (no terrain) or a string with viewport info
        if result is not None:
            assert "Viewport" in result
            assert "Legend" in result

    def test_render_vision_debug(self) -> None:
        state = make_empty_vision_state()
        state = add_tank_to_registry(state, tank_id=500, name="P1", team=0)
        state = update_tank_position(state, tank_id=500, x=50, y=60)
        state = update_container(state, x=10, y=20, volume=100)
        state = update_container(state, x=15, y=25, volume=0)

        debug = render_vision_debug(state)

        assert "Vision Cache Debug" in debug
        assert "Tanks registered: 1" in debug
        assert "Positions cached: 1" in debug
        assert "Containers cached: 2" in debug
        assert "Fuel containers: 1" in debug
        assert "Equipment: 1" in debug
        assert "World State Comparison" in debug


class TestEncodeDecode:
    """Round-trip encode/decode tests."""

    def test_roundtrip_tank_registry_entry(self) -> None:
        original = make_tank_registry_entry(tank_id=500, name="Player", team=2)
        encoded = encode_tank_registry_entry(original)
        decoded = decode_tank_registry_entry(encoded)

        assert decoded["tank_id"] == original["tank_id"]
        assert decoded["name"] == original["name"]
        assert decoded["team"] == original["team"]

    def test_roundtrip_position_entry(self) -> None:
        original = make_position_entry(tank_id=501, x=100, y=150)
        encoded = encode_position_entry(original)
        decoded = decode_position_entry(encoded)

        assert decoded["tank_id"] == original["tank_id"]
        assert decoded["x"] == original["x"]
        assert decoded["y"] == original["y"]

    def test_roundtrip_container_entry(self) -> None:
        original = make_container_entry(x=50, y=75, volume=300)
        encoded = encode_container_entry(original)
        decoded = decode_container_entry(encoded)

        assert decoded["x"] == original["x"]
        assert decoded["y"] == original["y"]
        assert decoded["volume"] == original["volume"]

    def test_roundtrip_vision_state(self) -> None:
        state = make_empty_vision_state()
        state = add_tank_to_registry(state, tank_id=500, name="P1", team=1)
        state = update_tank_position(state, tank_id=500, x=50, y=60)
        state = update_container(state, x=10, y=20, volume=100)
        state = set_self_tank_id(state, tank_id=500)
        state = update_self_fuel_vision(state, fuel=850)

        encoded = encode_vision_state(state)
        decoded = decode_vision_state(encoded)

        assert decoded["self_fuel"] == 850
        assert decoded["self_tank_id"] == 500
        assert "500" in decoded["tank_registry"]
        assert decoded["tank_registry"]["500"]["name"] == "P1"
