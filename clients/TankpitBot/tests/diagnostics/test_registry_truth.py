"""End-to-end tests for page-client registry tank-truth ingestion.

Fixtures mirror live registry entries from run 20260611-035438 (ids,
names, field shapes, the (-8,-8) roster-ghost sentinel). Every test
drives the REAL pipeline: decode -> world-state tank upsert via the
sniffer globals -> JSONL diagnostic artifact via
:class:`tests.conftest.FakeFileSystem`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import FakeFileSystem

from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.registry_truth import (
    RegistryTankDict,
    decode_registry_tank,
    register_tank_truth_from_page_snapshot,
)
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    configure_bot_runtime_logging,
)
from tankpit_bot.sniffer import world_state
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_combat import mark_tank_killed
from tankpit_bot.sniffer.world_state_tanks import (
    update_world_state_from_tank_damage,
    update_world_state_from_tank_entry,
)
from tankpit_bot.state.types import (
    ViewportStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)

_RegistryItem = dict[str, int | float | bool | str | None]

# Live entry shape from run 20260611-035438: purple-3 (id 511) rendered
# at col 15, row 9 while the viewport sat at (123,118).
_PURPLE_3: _RegistryItem = {
    "id": 511,
    "name": "purple-3",
    "u": 2,
    "h": 1,
    "j": 15,
    "i": 9,
    "s": 559,
    "l": 1,
    "aa": 0,
}

_GHOST_RED_1: _RegistryItem = {
    "id": 1,
    "name": "red-1",
    "u": 0,
    "h": 0,
    "j": -8,
    "i": -8,
    "s": 100000,
}

_SELF_ARTAX: _RegistryItem = {
    "id": 1301,
    "name": "Artax",
    "u": 3,
    "h": 2,
    "j": 9,
    "i": 9,
    "s": 146,
}


def _make_snapshot(
    collections: dict[str, list[dict[str, int | float | bool | str | None]]],
) -> PageClientSnapshotDict:
    """Return a healthy live-client snapshot carrying ``collections``."""
    return PageClientSnapshotDict(
        timestamp_ms=1000,
        client_present=True,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=50,
        last_page_client_send_age_ms=100,
        last_bot_send_age_ms=100,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections=collections,
    )


def _world_with_self_and_viewport() -> WorldStateDict:
    """Return a world whose self is Artax with the live viewport bounds."""
    world = make_empty_world_state()
    world["self_state"] = make_self_state(
        tank_id=1301,
        x=131,
        y=126,
        team=2,
        rank=1,
        fuel=800,
        leaderboard_position=0,
    )
    world["viewport"] = ViewportStateDict(left=123, top=118, width=16, height=16)
    return world


def _announce_tank_via_wire(tank_id: int, x: int, y: int, name: str) -> None:
    """Establish a tank as wire-known by simulating a TankEntry message."""
    update_world_state_from_tank_entry(get_world_service(), tank_id, x, y, name)


def _ingest_records(latest_events_path: str) -> list[RuntimeEventRecordDict]:
    """Return every ``registry_truth_ingested`` record from the artifact."""
    return [
        record
        for record in load_event_records(Path(latest_events_path))
        if record["fields"].get("diagnostic_kind") == "registry_truth_ingested"
    ]


def test_decode_live_entry() -> None:
    """A live registry entry decodes to the exact typed reading."""
    assert decode_registry_tank(_PURPLE_3) == RegistryTankDict(
        tank_id=511,
        name="purple-3",
        team=1,
        damage_tier=2,
        drawn_col=15,
        drawn_row=9,
    )


def test_decode_returns_none_for_partial_entry() -> None:
    """A field-capped entry missing a required key is a defined absence."""
    partial = {k: v for k, v in _PURPLE_3.items() if k != "u"}
    assert decode_registry_tank(partial) is None


def test_decode_returns_none_for_missing_name() -> None:
    """An entry without a name cannot be ingested."""
    unnamed = {k: v for k, v in _PURPLE_3.items() if k != "name"}
    assert decode_registry_tank(unnamed) is None


def test_decode_raises_on_mistyped_int_field() -> None:
    """A present field with the wrong type means the client changed shape."""
    with pytest.raises(ValueError, match="'j' must be an int"):
        decode_registry_tank({**_PURPLE_3, "j": "15"})


def test_decode_raises_on_bool_masquerading_as_int() -> None:
    """Booleans are not accepted where the client publishes integers."""
    with pytest.raises(ValueError, match="'u' must be an int"):
        decode_registry_tank({**_PURPLE_3, "u": True})


def test_decode_raises_on_mistyped_name() -> None:
    """A non-string name means the client changed shape."""
    with pytest.raises(ValueError, match="'name' must be a str"):
        decode_registry_tank({**_PURPLE_3, "name": 511})


def test_register_anchors_rendered_enemy(fake_fs: FakeFileSystem) -> None:
    """A rendered, wire-known enemy is refined to the mapped world tile.

    Mapping verified live: viewport (123,118) + render (15,9) - 1 =
    world (137,126); 58/60 wire shot targets matched this formula.
    """
    artifacts = configure_bot_runtime_logging("20260611-120000")
    _announce_tank_via_wire(511, 135, 125, "purple-3")
    snapshot = _make_snapshot({"P.j": [_PURPLE_3, _GHOST_RED_1, _SELF_ARTAX]})

    ingested = register_tank_truth_from_page_snapshot(
        snapshot,
        _world_with_self_and_viewport(),
    )

    assert ingested == 1
    tanks = world_state.get_world_state()["tanks"]
    assert "511" in tanks
    assert tanks["511"]["x"] == 137
    assert tanks["511"]["y"] == 126
    assert tanks["511"]["team"] == 1
    assert tanks["511"]["name"] == "purple-3"
    assert tanks["511"]["damage_state"] == 2
    records = _ingest_records(artifacts["latest_events_path"])
    assert len(records) == 1
    assert records[0]["fields"] == {
        "diagnostic_kind": "registry_truth_ingested",
        "tank_count": 1,
        "corpse_count": 0,
    }


def test_register_skips_ghosts_and_self(fake_fs: FakeFileSystem) -> None:
    """Roster ghosts at (-8,-8) and the self entry are never ingested."""
    artifacts = configure_bot_runtime_logging("20260611-120000")
    snapshot = _make_snapshot({"P.j": [_GHOST_RED_1, _SELF_ARTAX]})

    ingested = register_tank_truth_from_page_snapshot(
        snapshot,
        _world_with_self_and_viewport(),
    )

    assert ingested == 0
    assert world_state.get_world_state()["tanks"] == {}
    assert _ingest_records(artifacts["latest_events_path"]) == []


def test_register_zero_tier_never_downgrades_known_tier(fake_fs: FakeFileSystem) -> None:
    """A stale registry tier of 0 cannot erase a wire-known tier.

    Run 20260611-003415: red-6's registry entry held u=0 through a
    fight the wire tracked tier-by-tier; 0 means both "full" and
    "unsynced" so it is never authoritative.
    """
    configure_bot_runtime_logging("20260611-120000")
    world = _world_with_self_and_viewport()
    _announce_tank_via_wire(511, 135, 125, "purple-3")
    register_tank_truth_from_page_snapshot(_make_snapshot({"P.j": [_PURPLE_3]}), world)
    update_world_state_from_tank_damage(get_world_service(), 511, 1)
    snapshot = _make_snapshot({"P.j": [{**_PURPLE_3, "u": 0}]})

    ingested = register_tank_truth_from_page_snapshot(snapshot, world)

    assert ingested == 1
    assert world_state.get_world_state()["tanks"]["511"]["damage_state"] == 1


def test_register_without_self_state_is_a_no_op(fake_fs: FakeFileSystem) -> None:
    """No self identity means no viewport anchor and nothing ingested."""
    configure_bot_runtime_logging("20260611-120000")
    snapshot = _make_snapshot({"P.j": [_PURPLE_3]})

    assert register_tank_truth_from_page_snapshot(snapshot, make_empty_world_state()) == 0
    assert world_state.get_world_state()["tanks"] == {}


def test_register_without_registry_collection_is_a_no_op(fake_fs: FakeFileSystem) -> None:
    """A capture without the P.j collection ingests nothing."""
    configure_bot_runtime_logging("20260611-120000")
    snapshot = _make_snapshot({"pa": []})

    assert register_tank_truth_from_page_snapshot(snapshot, _world_with_self_and_viewport()) == 0


def test_register_skips_corpse_at_death_tile(fake_fs: FakeFileSystem) -> None:
    """A tank rendered at its recorded death tile is never resurrected.

    The registry keeps rendering corpses for minutes; run
    20260611-092159 re-ingested the dead purple-1 after its kill
    cooldown expired and spent three minutes shooting it (28 shots,
    28 miss-driven map reopens).
    """
    artifacts = configure_bot_runtime_logging("20260611-120000")
    world = _world_with_self_and_viewport()
    # The tank is wire-known, ingested live at (137,126), then killed there.
    _announce_tank_via_wire(511, 135, 125, "purple-3")
    register_tank_truth_from_page_snapshot(_make_snapshot({"P.j": [_PURPLE_3]}), world)
    mark_tank_killed(get_world_service(), 511)

    ingested = register_tank_truth_from_page_snapshot(
        _make_snapshot({"P.j": [_PURPLE_3]}),
        world,
    )

    assert ingested == 0
    records = _ingest_records(artifacts["latest_events_path"])
    assert records[-1]["fields"] == {
        "diagnostic_kind": "registry_truth_ingested",
        "tank_count": 0,
        "corpse_count": 1,
    }


def test_register_respawn_at_new_tile_clears_death_anchor(fake_fs: FakeFileSystem) -> None:
    """An observation away from the death tile is respawn evidence.

    The anchor clears permanently: the respawned tank is ingested even
    if it later wanders back across its old death tile.
    """
    configure_bot_runtime_logging("20260611-120000")
    world = _world_with_self_and_viewport()
    _announce_tank_via_wire(511, 135, 125, "purple-3")
    register_tank_truth_from_page_snapshot(_make_snapshot({"P.j": [_PURPLE_3]}), world)
    mark_tank_killed(get_world_service(), 511)

    respawned = {**_PURPLE_3, "j": 5, "i": 4}
    assert register_tank_truth_from_page_snapshot(_make_snapshot({"P.j": [respawned]}), world) == 1
    # Back across the old death tile: still ingested, anchor is gone.
    assert register_tank_truth_from_page_snapshot(_make_snapshot({"P.j": [_PURPLE_3]}), world) == 1
    tanks = world_state.get_world_state()["tanks"]
    assert tanks["511"]["x"] == 137
    assert tanks["511"]["y"] == 126


def test_register_skips_tank_the_wire_never_announced(fake_fs: FakeFileSystem) -> None:
    """A registry entry without wire backing is never ingested.

    The wire vouches for presence; the registry only refines. A drawn
    entry the wire never announced is a stale afterimage -- the class
    that absorbed 52 wasted shots in run 20260611-103309 (the tank had
    died and respawned elsewhere ~10s later; only its sprite state
    lingered).
    """
    artifacts = configure_bot_runtime_logging("20260611-120000")

    ingested = register_tank_truth_from_page_snapshot(
        _make_snapshot({"P.j": [_PURPLE_3]}),
        _world_with_self_and_viewport(),
    )

    assert ingested == 0
    assert world_state.get_world_state()["tanks"] == {}
    assert _ingest_records(artifacts["latest_events_path"]) == []


def test_register_skips_partial_entries(fake_fs: FakeFileSystem) -> None:
    """Field-capped partial entries are skipped, full ones ingested."""
    configure_bot_runtime_logging("20260611-120000")
    _announce_tank_via_wire(511, 135, 125, "purple-3")
    _announce_tank_via_wire(513, 130, 120, "purple-5")
    partial = {k: v for k, v in _PURPLE_3.items() if k != "h"}
    other = {**_PURPLE_3, "id": 513, "name": "purple-5", "j": 14}
    snapshot = _make_snapshot({"P.j": [partial, other]})

    ingested = register_tank_truth_from_page_snapshot(
        snapshot,
        _world_with_self_and_viewport(),
    )

    assert ingested == 1
    tanks = world_state.get_world_state()["tanks"]
    # 513's full entry refined to the mapped tile; 511's partial entry
    # left its wire-announced position untouched.
    assert tanks["513"]["x"] == 136
    assert tanks["513"]["y"] == 126
    assert tanks["511"]["x"] == 135
    assert tanks["511"]["y"] == 125
