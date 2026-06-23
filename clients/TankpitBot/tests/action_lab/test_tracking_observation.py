"""Tests for the pure tracking-observation row builders."""

from __future__ import annotations

from tankpit_bot.action_lab.tracking_observation import (
    build_js_belief,
    build_our_belief,
    build_tracking_observation,
    find_js_entry_by_position,
    find_js_tank_entry,
    select_js_identity_key,
)
from tankpit_bot.bot.ai.types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.state.types import (
    SelfStateDict,
    TankStateDict,
    WorldStateDict,
    make_empty_world_state,
)
from tankpit_bot.state.types.self_state import make_self_state
from tankpit_bot.state.types.tank import make_tank_state


def _self_state(team: int = 1) -> SelfStateDict:
    return make_self_state(
        tank_id=42,
        x=100,
        y=100,
        team=team,
        rank=4,
        fuel=800,
        leaderboard_position=0,
    )


def _alive_tank(
    *,
    tank_id: int = 511,
    x: int = 99,
    y: int = 100,
    last_wire_seen_ms: int = 10_000,
    last_position_update_ms: int = 9_500,
) -> TankStateDict:
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=3,
        rank=4,
        damage_state=0,
        name="orange-7",
        is_bot=False,
        is_self=False,
        timestamp_ms=last_wire_seen_ms,
        last_wire_seen_ms=last_wire_seen_ms,
        last_position_update_ms=last_position_update_ms,
    )


def _world_with(tanks: dict[str, TankStateDict]) -> WorldStateDict:
    world = make_empty_world_state()
    world["self_state"] = _self_state()
    world["tanks"] = tanks
    return world


def _threat_for(tank: TankStateDict, distance: int = 1) -> EnemyThreatDict:
    return make_enemy_threat(
        tank_id=tank["tank_id"],
        x=tank["x"],
        y=tank["y"],
        distance=distance,
        damage_state=tank["damage_state"],
        rank=tank["rank"],
        team=tank["team"],
        name=tank["name"],
        is_bot=tank["is_bot"],
        timestamp_ms=tank["timestamp_ms"],
        last_wire_seen_ms=tank["last_wire_seen_ms"],
        last_position_update_ms=tank["last_position_update_ms"],
    )


def test_find_js_tank_entry_returns_match() -> None:
    """find_js_tank_entry returns the entry whose tracked key/value matches."""
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"id": 1, "x": 0}, {"id": 511, "x": 99}],
    }
    result = find_js_tank_entry(collections, "id", "511")
    assert result == {"id": 511, "x": 99}


def test_find_js_tank_entry_returns_none_when_key_empty() -> None:
    """find_js_tank_entry returns None when tracked_js_key is the empty sentinel."""
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"id": 511}],
    }
    assert find_js_tank_entry(collections, "", "511") is None


def test_find_js_tank_entry_returns_none_when_registry_missing() -> None:
    """find_js_tank_entry returns None when P.j is absent from collections."""
    assert find_js_tank_entry({}, "id", "511") is None


def test_find_js_tank_entry_returns_none_when_no_match() -> None:
    """find_js_tank_entry returns None when no entry has the tracked value."""
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"id": 1}, {"id": 2}],
    }
    assert find_js_tank_entry(collections, "id", "511") is None


def test_find_js_tank_entry_skips_entries_missing_key() -> None:
    """Entries that lack the tracked key are skipped, not crashed on."""
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"other": 999}, {"id": 511}],
    }
    result = find_js_tank_entry(collections, "id", "511")
    assert result == {"id": 511}


def test_build_js_belief_present_when_match_found() -> None:
    """build_js_belief returns present=True with matched entry fields."""
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"id": 511, "x": 99}],
    }
    belief = build_js_belief(collections, "id", "511")
    assert belief["present"] is True
    assert belief["fields"] == {"id": 511, "x": 99}


def test_build_js_belief_absent_when_no_match() -> None:
    """build_js_belief returns present=False + empty fields when no match."""
    belief = build_js_belief({}, "id", "511")
    assert belief["present"] is False
    assert belief["fields"] == {}


def test_build_our_belief_returns_absent_when_tank_unknown() -> None:
    """build_our_belief flags absent + locked_target_source='none' when tank missing."""
    world = _world_with({})
    belief = build_our_belief(
        tank_id=511,
        world=world,
        threats=[],
        sample_timestamp_ms=15_000,
    )
    assert belief["present"] is False
    assert belief["locked_target_source"] == "none"
    assert belief["would_locked_target_return"] is False


def test_build_our_belief_returns_threats_source_when_in_threats() -> None:
    """build_our_belief returns locked_target_source='threats' when in threats list."""
    tank = _alive_tank()
    world = _world_with({str(tank["tank_id"]): tank})
    threats = [_threat_for(tank)]
    belief = build_our_belief(
        tank_id=tank["tank_id"],
        world=world,
        threats=threats,
        sample_timestamp_ms=11_000,
    )
    assert belief["present"] is True
    assert belief["is_in_threats"] is True
    assert belief["locked_target_source"] == "threats"
    assert belief["would_locked_target_return"] is True
    assert belief["wire_age_ms"] == 1_000
    assert belief["position_age_ms"] == 1_500


def test_build_our_belief_returns_none_when_dropped_from_threats() -> None:
    """build_our_belief returns ``locked_target_source='none'`` post-2026-06-21.

    Mirrors the production change that removed
    ``get_locked_target``'s world-state fallback: a tank still in
    ``world.tanks`` but missing from ``analyze_threats`` no longer
    counts as a viable lock target.
    """
    tank = _alive_tank()
    world = _world_with({str(tank["tank_id"]): tank})
    belief = build_our_belief(
        tank_id=tank["tank_id"],
        world=world,
        threats=[],
        sample_timestamp_ms=11_000,
    )
    assert belief["locked_target_source"] == "none"
    assert belief["would_locked_target_return"] is False


def test_build_our_belief_returns_none_for_zero_position_sentinel() -> None:
    """build_our_belief still routes the (0,0) sentinel through 'none'.

    Post-fallback-removal there is no difference between the
    ``(0, 0)`` sentinel case and the general "not in threats" case
    -- both return ``locked_target_source='none'``. Kept as a
    distinct test because the historical fallback gated on the
    sentinel explicitly; if we ever revisit that path this test
    documents the intent.
    """
    tank = _alive_tank(x=0, y=0)
    world = _world_with({str(tank["tank_id"]): tank})
    belief = build_our_belief(
        tank_id=tank["tank_id"],
        world=world,
        threats=[],
        sample_timestamp_ms=11_000,
    )
    assert belief["locked_target_source"] == "none"
    assert belief["would_locked_target_return"] is False


def test_build_tracking_observation_combines_our_and_js_belief() -> None:
    """build_tracking_observation returns one row pairing both beliefs."""
    tank = _alive_tank()
    world = _world_with({str(tank["tank_id"]): tank})
    threats = [_threat_for(tank)]
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"id": tank["tank_id"], "x": tank["x"]}],
    }
    observation = build_tracking_observation(
        sample_index=7,
        sample_timestamp_ms=11_000,
        tank_id=tank["tank_id"],
        tracked_label=tank["name"],
        tracked_js_key="id",
        tracked_js_value=str(tank["tank_id"]),
        world=world,
        threats=threats,
        world_collections=collections,
        bot_combat_target_id=tank["tank_id"],
        bot_mode_state="ENGAGE",
    )
    assert observation["sample_index"] == 7
    assert observation["tank_id"] == tank["tank_id"]
    assert observation["our_belief"]["locked_target_source"] == "threats"
    assert observation["js_belief"]["present"] is True
    assert observation["bot_combat_target_id"] == tank["tank_id"]
    assert observation["bot_mode_state"] == "ENGAGE"


def test_select_js_identity_key_returns_field_matching_tank_id() -> None:
    """select_js_identity_key picks the field whose value equals tank_id."""
    tank = _alive_tank(tank_id=511)
    js_entry: dict[str, int | float | bool | str | None] = {
        "x": 99,
        "id": 511,
        "team": 3,
    }
    key, value = select_js_identity_key(js_entry, tank)
    assert (key, value) == ("id", "511")


def test_select_js_identity_key_skips_bool_fields() -> None:
    """Booleans equal-to-1 don't get mistakenly returned as identity."""
    tank = _alive_tank(tank_id=1)
    js_entry: dict[str, int | float | bool | str | None] = {
        "alive": True,
        "id": 1,
    }
    key, _value = select_js_identity_key(js_entry, tank)
    assert key == "id"


def test_select_js_identity_key_returns_empty_when_no_match() -> None:
    """select_js_identity_key returns ('','') when no field matches tank_id."""
    tank = _alive_tank(tank_id=511)
    js_entry: dict[str, int | float | bool | str | None] = {"x": 99, "y": 100}
    key, value = select_js_identity_key(js_entry, tank)
    assert (key, value) == ("", "")


def test_find_js_entry_by_position_matches_x_y_pair() -> None:
    """find_js_entry_by_position matches when x AND y values both appear."""
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"x": 99, "y": 100, "id": 511}],
    }
    entry = find_js_entry_by_position(collections, 99, 100)
    assert entry == {"x": 99, "y": 100, "id": 511}


def test_find_js_entry_by_position_returns_none_when_no_registry() -> None:
    """find_js_entry_by_position returns None when P.j is absent."""
    empty: dict[str, list[dict[str, int | float | bool | str | None]]] = {}
    assert find_js_entry_by_position(empty, 99, 100) is None


def test_find_js_entry_by_position_returns_none_when_only_x_matches() -> None:
    """find_js_entry_by_position needs both x and y to match for a hit."""
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"x": 99, "other": 7}],
    }
    assert find_js_entry_by_position(collections, 99, 100) is None


def test_find_js_entry_by_position_skips_bool_fields() -> None:
    """find_js_entry_by_position does not treat True/False as numeric matches."""
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"alive": True, "ready": False}],
    }
    assert find_js_entry_by_position(collections, 1, 0) is None


def test_find_js_entry_by_position_skips_non_int_primitive_fields() -> None:
    """Strings, floats, and None values pass through without confusing the match."""
    collections: dict[str, list[dict[str, int | float | bool | str | None]]] = {
        "P.j": [{"name": "orange-7", "skin": 1.5, "info": None, "x": 99, "y": 100}],
    }
    entry = find_js_entry_by_position(collections, 99, 100)
    assert entry == {"name": "orange-7", "skin": 1.5, "info": None, "x": 99, "y": 100}
