"""Tests for ``TankObservation`` and ``apply_tank_observation``.

This file is the contract for the three-timestamp freshness model.
Every test here pins a single invariant; deleting or weakening any
test is a deliberate contract change that must be matched by a
docstring + wiki update.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.state.mutations import apply_tank_observation
from tankpit_bot.state.types import (
    WorldStateDict,
    decode_tank_observation,
    encode_tank_observation,
    make_empty_world_state,
    make_tank_observation,
    make_tank_state,
)


def make_world_with_seed(
    *,
    tank_id: int = 100,
    x: int = 10,
    y: int = 20,
    team: int = 0,
    rank: int = 0,
    damage_state: int = 0,
    direction: int = 0,
    name: str = "seed",
    is_bot: bool = False,
    timestamp_ms: int = 1000,
    last_wire_seen_ms: int = 900,
    last_position_update_ms: int = 800,
) -> tuple[WorldStateDict, str]:
    """Seed a world state with one tank and return the state + its key.

    Args:
        tank_id: Tank id to seed.
        x: Seeded x coordinate.
        y: Seeded y coordinate.
        team: Seeded team.
        rank: Seeded rank.
        damage_state: Seeded damage tier.
        direction: Seeded direction byte.
        name: Seeded player name.
        is_bot: Seeded bot flag.
        timestamp_ms: Seeded any-source timestamp.
        last_wire_seen_ms: Seeded wire-presence timestamp.
        last_position_update_ms: Seeded position-freshness timestamp.

    Returns:
        ``(WorldStateDict, str)`` where the second element is the
        registry key of the seeded tank.
    """
    state = make_empty_world_state()
    tank = make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        damage_state=damage_state,
        direction=direction,
        name=name,
        is_bot=is_bot,
        is_self=False,
        source="viewport",
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=last_wire_seen_ms,
        last_position_update_ms=last_position_update_ms,
    )
    state["tanks"][str(tank_id)] = tank
    return state, str(tank_id)


# =============================================================================
# Invariant 1: timestamp_ms always advances
# =============================================================================


class TestInvariantTimestampAlwaysAdvances:
    """``timestamp_ms`` advances on every observation."""

    def test_any_observation_advances_timestamp(self) -> None:
        """Even a fully-null observation advances ``timestamp_ms``."""
        state, key = make_world_with_seed(tank_id=42, timestamp_ms=1000)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["timestamp_ms"] == 5000

    def test_map_only_observation_advances_timestamp(self) -> None:
        """A map-only observation still advances ``timestamp_ms``."""
        state, key = make_world_with_seed(tank_id=42, timestamp_ms=1000)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["timestamp_ms"] == 5000


# =============================================================================
# Invariant 2: last_wire_seen_ms advances iff is_wire_sourced
# =============================================================================


class TestInvariantWireSeenRequiresWire:
    """``last_wire_seen_ms`` advances only on wire-sourced observations."""

    def test_wire_observation_advances_wire_seen(self) -> None:
        """Any wire observation refreshes ``last_wire_seen_ms``."""
        state, key = make_world_with_seed(tank_id=42, last_wire_seen_ms=900)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_wire_seen_ms"] == 5000

    def test_map_observation_does_not_advance_wire_seen(self) -> None:
        """Map-sourced observations leave ``last_wire_seen_ms`` untouched."""
        state, key = make_world_with_seed(tank_id=42, last_wire_seen_ms=900)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_wire_seen_ms"] == 900

    def test_map_observation_starts_wire_seen_at_zero_for_new_tank(self) -> None:
        """A first-sight map observation must start wire-seen at zero."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=99,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["99"]["last_wire_seen_ms"] == 0


# =============================================================================
# Invariant 3: last_position_update_ms requires position_is_authoritative AND position
# =============================================================================


class TestInvariantPositionFreshnessRequiresAuthoritativePosition:
    """``last_position_update_ms`` advances iff ``position_is_authoritative`` AND ``position``.

    ``position_is_authoritative`` decouples the kill-shot gate from the
    wire-presence gate: MAP_DATA snapshots are not wire-sourced (a
    departed tank can linger in the snapshot for minutes) but their
    listed coordinates ARE the server's authoritative statement of
    where each tank IS at snapshot time, so they advance the position
    freshness gate without claiming wire presence. Radar EnemyDetect
    and DOM-scraped client-registry refinements do NOT (tile-coarse /
    out-of-band estimates).
    """

    def test_wire_with_position_advances_position_freshness(self) -> None:
        """0x3D-like observation refreshes ``last_position_update_ms``."""
        state, key = make_world_with_seed(tank_id=42, last_position_update_ms=800)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_position_update_ms"] == 5000

    def test_wire_without_position_preserves_position_freshness(self) -> None:
        """TankStatusSync-like observation MUST NOT lie about position freshness.

        This is the locked invariant whose violation produced the
        stale-registry combat-miss loop in 2026-06-19 runs.
        """
        state, key = make_world_with_seed(tank_id=42, last_position_update_ms=800)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            damage_state=2,
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_position_update_ms"] == 800

    def test_authoritative_map_position_advances_position_freshness(self) -> None:
        """MAP_DATA's listed coordinates ARE the server's authoritative position.

        A wire-quiet stationary target stays kill-shot-fresh after the
        bot opens the map, even though the wire-presence stamp
        deliberately does NOT advance (a departed tank can linger in
        the snapshot for minutes). Live run 20260620-191622 fix: the
        bot was blocking targets it was actively engaging because the
        gate could only advance on wire-position-bearing messages.
        """
        state, key = make_world_with_seed(tank_id=42, last_position_update_ms=800)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=True,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_position_update_ms"] == 5000

    def test_non_authoritative_position_preserves_position_freshness(self) -> None:
        """Radar / DOM-refinement positions are not authoritative; freshness stays.

        Radar EnemyDetect (0x48) returns a tile-coarse estimate that
        may not match the target's actual wire position by the next
        tick. Client-registry refinements come from DOM scrape, an
        out-of-band channel with no server proof. Neither must gate a
        kill shot.
        """
        state, key = make_world_with_seed(tank_id=42, last_position_update_ms=800)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=False,
            storage_source="radar",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["last_position_update_ms"] == 800

    def test_authoritative_map_position_starts_freshness_for_new_tank(
        self,
    ) -> None:
        """A first-sight authoritative map observation seeds position freshness."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=99,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=True,
            storage_source="world_state",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["99"]["last_position_update_ms"] == 5000

    def test_non_authoritative_position_starts_freshness_at_zero_for_new_tank(
        self,
    ) -> None:
        """A first-sight radar / DOM-refinement observation cannot seed freshness."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=99,
            timestamp_ms=5000,
            is_wire_sourced=False,
            position_is_authoritative=False,
            storage_source="radar",
            position=(50, 60),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["99"]["last_position_update_ms"] == 0

    def test_wire_status_only_starts_position_freshness_at_zero_for_new_tank(
        self,
    ) -> None:
        """A first-sight damage-only wire observation cannot bootstrap position freshness."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=99,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            damage_state=2,
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["99"]["last_position_update_ms"] == 0


# =============================================================================
# Field-merge semantics: present overwrites, None preserves
# =============================================================================


class TestFieldMergeSemantics:
    """Present aspects overwrite; ``None`` aspects preserve existing values."""

    def test_position_present_overwrites(self) -> None:
        """Position field updates ``x``/``y`` on the tank."""
        state, key = make_world_with_seed(tank_id=42, x=10, y=20)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(99, 88),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["x"] == 99
        assert result["tanks"][key]["y"] == 88

    def test_position_none_preserves(self) -> None:
        """Position=None preserves existing ``(x, y)``."""
        state, key = make_world_with_seed(tank_id=42, x=10, y=20)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            damage_state=2,
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["x"] == 10
        assert result["tanks"][key]["y"] == 20

    def test_team_rank_damage_direction_present_overwrites(self) -> None:
        """Aspect fields write through when present."""
        state, key = make_world_with_seed(tank_id=42, team=0, rank=1, damage_state=0, direction=8)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            team=3,
            rank=5,
            damage_state=2,
            direction=33,
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"][key]
        assert tank["team"] == 3
        assert tank["rank"] == 5
        assert tank["damage_state"] == 2
        assert tank["direction"] == 33

    def test_team_rank_damage_direction_none_preserves(self) -> None:
        """Aspect fields preserve when absent."""
        state, key = make_world_with_seed(tank_id=42, team=2, rank=4, damage_state=1, direction=12)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"][key]
        assert tank["team"] == 2
        assert tank["rank"] == 4
        assert tank["damage_state"] == 1
        assert tank["direction"] == 12

    def test_name_is_bot_present_overwrites(self) -> None:
        """``name`` and ``is_bot`` write through when present."""
        state, key = make_world_with_seed(tank_id=42, name="old", is_bot=False)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            name="new",
            is_bot=True,
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"][key]
        assert tank["name"] == "new"
        assert tank["is_bot"] is True

    def test_name_is_bot_none_preserves(self) -> None:
        """``name`` and ``is_bot`` preserve when absent."""
        state, key = make_world_with_seed(tank_id=42, name="keep", is_bot=True)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"][key]
        assert tank["name"] == "keep"
        assert tank["is_bot"] is True


# =============================================================================
# Tank creation: observation arrives for a tank that doesn't exist yet
# =============================================================================


class TestTankCreationOnFirstObservation:
    """Observations create tanks that did not exist in the registry."""

    def test_first_wire_position_observation_creates_tank(self) -> None:
        """First sighting via wire with position populates the registry."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=77,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(40, 50),
            team=1,
            rank=2,
            damage_state=0,
            direction=8,
            name="Newbie",
            is_bot=False,
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"]["77"]
        assert tank["x"] == 40
        assert tank["y"] == 50
        assert tank["team"] == 1
        assert tank["rank"] == 2
        assert tank["name"] == "Newbie"
        assert tank["timestamp_ms"] == 5000
        assert tank["last_wire_seen_ms"] == 5000
        assert tank["last_position_update_ms"] == 5000

    def test_first_observation_without_position_creates_tank_at_origin(self) -> None:
        """First sighting without position seeds ``(0, 0)``."""
        state = make_empty_world_state()
        obs = make_tank_observation(
            tank_id=77,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            damage_state=2,
        )
        result = apply_tank_observation(state, obs)
        tank = result["tanks"]["77"]
        assert tank["x"] == 0
        assert tank["y"] == 0
        assert tank["damage_state"] == 2

    def test_self_tank_observation_sets_is_self(self) -> None:
        """Observations for the bot's own tank id set ``is_self`` to True."""
        state = make_empty_world_state()
        # Seed self_state so the observation can identify itself.
        from tankpit_bot.state.types.self_state import make_self_state

        state["self_state"] = make_self_state(
            tank_id=42,
            x=0,
            y=0,
            team=0,
            rank=0,
            fuel=0,
            leaderboard_position=0,
        )
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(40, 50),
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"]["42"]["is_self"] is True


# =============================================================================
# Storage source is recorded on the tank
# =============================================================================


class TestStorageSourceIsRecorded:
    """``storage_source`` becomes the tank's ``source`` field."""

    def test_viewport_source_recorded(self) -> None:
        """Viewport storage source is stored on the tank."""
        state, key = make_world_with_seed(tank_id=42)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["source"] == "viewport"

    def test_radar_source_recorded(self) -> None:
        """Radar storage source is stored on the tank."""
        state, key = make_world_with_seed(tank_id=42)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="radar",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["source"] == "radar"

    def test_world_state_source_recorded(self) -> None:
        """World-state storage source is stored on the tank."""
        state, key = make_world_with_seed(tank_id=42)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
        )
        result = apply_tank_observation(state, obs)
        assert result["tanks"][key]["source"] == "world_state"


# =============================================================================
# Outer state timestamp also advances
# =============================================================================


class TestOuterTimestampAdvances:
    """``state["timestamp_ms"]`` advances alongside the per-tank timestamp."""

    def test_outer_timestamp_advances(self) -> None:
        """The outer world-state timestamp tracks the observation."""
        state, _ = make_world_with_seed(tank_id=42, timestamp_ms=1000)
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
        )
        result = apply_tank_observation(state, obs)
        assert result["timestamp_ms"] == 5000


# =============================================================================
# Codec round-trip
# =============================================================================


class TestTankObservationCodec:
    """``encode_tank_observation``/``decode_tank_observation`` round-trip."""

    def test_round_trip_full(self) -> None:
        """Every field round-trips through encode + decode."""
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=True,
            storage_source="viewport",
            position=(40, 50),
            team=2,
            rank=3,
            damage_state=1,
            direction=8,
            name="player",
            is_bot=True,
        )
        encoded = encode_tank_observation(obs)
        decoded = decode_tank_observation(encoded)
        assert decoded == obs

    def test_round_trip_all_nullable_none(self) -> None:
        """``None`` aspects survive the round-trip as ``None``."""
        obs = make_tank_observation(
            tank_id=42,
            timestamp_ms=5000,
            is_wire_sourced=False,
            storage_source="world_state",
        )
        encoded = encode_tank_observation(obs)
        decoded = decode_tank_observation(encoded)
        assert decoded == obs

    def test_decode_rejects_missing_required_field(self) -> None:
        """Missing required field raises ``JSONTypeError``."""
        data: JSONObject = {"tank_id": 42}
        with pytest.raises(JSONTypeError):
            decode_tank_observation(data)

    def test_decode_rejects_malformed_position_wrong_length(self) -> None:
        """Position list with wrong length raises ``JSONTypeError``."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": [1, 2, 3],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="2-element list"):
            decode_tank_observation(data)

    def test_decode_rejects_malformed_position_non_int(self) -> None:
        """Position with non-int component raises ``JSONTypeError``."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": ["x", 2],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match=r"position\[0\] must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_malformed_position_y_non_int(self) -> None:
        """Position with non-int y raises ``JSONTypeError``."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": [1, "y"],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match=r"position\[1\] must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_malformed_position_not_list(self) -> None:
        """Position that is not a list raises ``JSONTypeError``."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": "10,20",
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="2-element list"):
            decode_tank_observation(data)

    def test_decode_rejects_optional_int_with_wrong_type(self) -> None:
        """Optional int fields reject non-int values."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": None,
            "team": "two",
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="team must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_optional_str_with_wrong_type(self) -> None:
        """Optional string fields reject non-string values."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": None,
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": 123,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="name must be str"):
            decode_tank_observation(data)

    def test_decode_rejects_optional_bool_with_wrong_type(self) -> None:
        """Optional bool fields reject non-bool values."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": None,
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": "yes",
        }
        with pytest.raises(JSONTypeError, match="is_bot must be bool"):
            decode_tank_observation(data)

    def test_decode_rejects_unknown_storage_source(self) -> None:
        """Unknown ``storage_source`` raises via shared validator."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "storage_source": "moon",
            "position": None,
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="storage_source must be"):
            decode_tank_observation(data)

    def test_decode_rejects_bool_in_optional_int_field(self) -> None:
        """Booleans must not be accepted by optional int validators.

        Python's ``bool`` is a subclass of ``int``, so the optional-int
        validator explicitly rejects ``bool`` to avoid silent
        misinterpretation.
        """
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": None,
            "team": True,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match="team must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_bool_in_position_x_int_field(self) -> None:
        """Booleans must not pass the position-component int check.

        ``isinstance(True, int)`` is True; the validator must guard
        against that or callers could pass ``[True, False]`` as a
        position.
        """
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": [True, 5],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match=r"position\[0\] must be int"):
            decode_tank_observation(data)

    def test_decode_rejects_bool_in_position_y_int_field(self) -> None:
        """Position y must reject booleans for the same reason as x."""
        data: JSONObject = {
            "tank_id": 42,
            "timestamp_ms": 5000,
            "is_wire_sourced": True,
            "position_is_authoritative": True,
            "storage_source": "viewport",
            "position": [5, True],
            "team": None,
            "rank": None,
            "damage_state": None,
            "direction": None,
            "name": None,
            "is_bot": None,
        }
        with pytest.raises(JSONTypeError, match=r"position\[1\] must be int"):
            decode_tank_observation(data)
