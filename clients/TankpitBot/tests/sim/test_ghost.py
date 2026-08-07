"""Tests for the sim ghost: state tracking.

``test_ghost.py`` was 628 lines; the emission tests are now a sibling.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
)

from tankpit_bot.protocol.types import (
    ChatMessageDict,
)
from tankpit_bot.sim.commands import SimError
from tankpit_bot.sim.ghost import (
    GhostTracker,
    compile_ghost_spec,
    ghost_events_for_tick,
)
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.world import (
    make_sim_tank,
    make_sim_world,
)
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.sim._ghost_fixtures import (
    _T0,
    _capture,
    _fight_capture,
    _rich_capture,
)


def test_compiler_builds_the_replayable_spec() -> None:
    """Identities, timelines, dots, and the client's opening state."""
    spec = compile_ghost_spec(_fight_capture())
    assert spec["client_x"] == 100
    assert spec["client_y"] == 100
    assert spec["client_team"] == 2
    assert spec["client_rank"] == 3
    assert spec["client_fuel"] == 900
    assert spec["client_counts"] == [25, 20, 25, 19, 21]
    assert spec["dot_atlas"] == [(140, 90), (150, 95)]
    assert [g["tank_id"] for g in spec["ghosts"]] == [500]
    ghost = spec["ghosts"][0]
    assert (ghost["x"], ghost["y"], ghost["name"], ghost["team"], ghost["rank"]) == (
        108,
        100,
        "Yuppler",
        1,
        2,
    )
    assert ghost_events_for_tick(spec, 1) == [
        {"tick": 1, "tank_id": 500, "kind": "place", "x": 106, "y": 100, "message_id": 0}
    ]
    shots = [e for e in spec["events"] if e["kind"] == "shoot"]
    chats = [e for e in spec["events"] if e["kind"] == "chat"]
    assert shots == [
        {"tick": 2, "tank_id": 500, "kind": "shoot", "x": 100, "y": 100, "message_id": 0}
    ]
    assert chats == [
        {"tick": 2, "tank_id": 500, "kind": "chat", "x": 106, "y": 100, "message_id": 41}
    ]
    assert spec["recorded_path"] == {0: (100, 100), 3: (102, 100)}
    assert spec["ticks"] == 4
    assert spec["unplaced_tanks"] == 0


def test_compiler_dot_atlas_seeds_unread_dots_as_drained() -> None:
    """Atlas dots without a volume read become volume-0 dotted seeds."""
    spec = compile_ghost_spec(_fight_capture())
    drained = [c for c in spec["containers"] if c["volume"] == 0]
    assert {(c["x"], c["y"]) for c in drained} == {(140, 90), (150, 95)}
    assert all(c["dotted"] for c in drained)


def test_compiler_refuses_a_selfless_capture() -> None:
    """A recording that never placed its own tank is unreplayable."""
    with pytest.raises(RuntimeError, match="never identified"):
        compile_ghost_spec(_capture([(_T0, [])]))
    with pytest.raises(RuntimeError, match="missing its magic"):
        compile_ghost_spec(dump_json_str({"messages": []}))


def test_relocate_tank_announces_in_window_placements_only() -> None:
    """A ghost placed inside the client's window rides the next batch
    as a 0x3D; an off-window placement stays silent until the
    membership diff speaks. Dead and unknown tanks raise."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 100, 100, 1000)
    world["tanks"][500] = make_sim_tank(500, 1, 1, 200, 200, 600)
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    server.relocate_tank(500, 104, 100)
    entered = server.advance_tick()
    positions = [m for m in entered if m["msg_type"] == 0x3D]
    # Entering the window: exactly ONE 0x3D (the membership diff's).
    assert [(m["tank_id"], m["x"], m["y"]) for m in positions] == [(500, 104, 100)]
    server.relocate_tank(500, 105, 100)
    moved = server.advance_tick()
    moved_positions = [m for m in moved if m["msg_type"] == 0x3D]
    # In-window movement of a visible tank: the explicit re-statement.
    assert [(m["tank_id"], m["x"], m["y"]) for m in moved_positions] == [(500, 105, 100)]
    server.relocate_tank(500, 220, 220)
    away = server.advance_tick()
    announced = [m for m in away if m["msg_type"] == 0x3D and m.get("tank_id") == 500]
    assert announced == []
    world["tanks"][500]["alive"] = False
    with pytest.raises(SimError):
        server.relocate_tank(500, 104, 100)
    with pytest.raises(SimError):
        server.relocate_tank(404, 104, 100)


def test_tracker_measures_the_divergence_point() -> None:
    """Within-threshold rounds track; the first breach is recorded."""
    tracker = GhostTracker({0: (100, 100), 2: (110, 100)})
    tracker.note_round(0, 101, 100)
    tracker.note_round(1, 103, 100)  # falls back to the last known tile
    tracker.note_round(2, 130, 100)
    tracker.note_round(3, 131, 100)
    assert tracker.compared_ticks == 4
    assert tracker.tracked_ticks == 2
    assert tracker.first_divergence_tick == 2
    assert tracker.final_drift == 21


def test_ghost_session_replays_the_recording_end_to_end(fake_fs: FakeFileSystem) -> None:
    """``--ghost`` boots the recorded world and replays its opponents.

    Asserted from the artifacts: the ghost spawns under its recorded
    name, its replayed shot and chat cross the wire to the live bot,
    and the ``ghost_summary`` diagnostic reports the tracking verdict.
    """
    from pathlib import Path

    from tankpit_bot import _test_hooks
    from tankpit_bot.sim.run import main
    from tankpit_bot.sim.scenarios import SIM_FIELD

    capture_text = _fight_capture()
    fake_fs.write_text(Path(SIM_FIELD), "fake-gif-bytes")
    fake_fs.write_text(Path("runs/ghost-input.capture_session.json"), capture_text)
    _test_hooks.load_terrain_map = lambda gif_path: InMemoryTerrainMap()
    exit_code = main(
        [
            "--ghost",
            "runs/ghost-input.capture_session.json",
            "--rounds",
            "10",
            "--stamp",
            "20260801-000004",
        ]
    )
    assert exit_code == 0
    files = fake_fs.get_written_files()
    world_path = next(path for path in files if "sim-20260801-000004.world.json" in path)
    world_doc = narrow_json_to_dict(load_json_str(files[world_path]))
    tanks = [narrow_json_to_dict(t) for t in narrow_json_to_list(world_doc["tanks"])]
    ghost_tank = next(t for t in tanks if t["tank_id"] == 500)
    assert narrow_json_to_str(ghost_tank["name"]) == "Yuppler"
    events_path = next(path for path in files if path.endswith("latest.sim.events.jsonl"))
    summaries = []
    chat_seen = False
    for line in files[events_path].splitlines():
        if not line:
            continue
        record = narrow_json_to_dict(load_json_str(line))
        if record.get("diagnostic_kind") == "ghost_summary":
            summaries.append(record)
        if record.get("diagnostic_kind") == "chat_received":
            chat_seen = True
    assert len(summaries) == 1
    assert narrow_json_to_int(summaries[0]["compared_ticks"]) >= 1
    # The replayed 0x4D reached the live bot (the ghost-replayed chat
    # is the consent signal of the human-fight contract).
    assert chat_seen


def test_compiler_consumes_every_read_flavor_and_skips_strays() -> None:
    """Radar/cache/viewport reads, 0x47 paths, and stray frames."""
    spec = compile_ghost_spec(_rich_capture())
    by_tile = {(c["x"], c["y"]): c for c in spec["containers"]}
    # 0x4F fuel read on an atlas dot stays dotted; the cache and
    # viewport reads are visible-layer; the radar equipment read
    # lands in the equipment list; volume-0 reads seed nothing.
    assert by_tile[(140, 90)] == {"x": 140, "y": 90, "volume": 520, "dotted": True}
    assert by_tile[(143, 90)]["dotted"] is True
    assert by_tile[(100, 100)] == {"x": 100, "y": 100, "volume": 77, "dotted": True}
    assert (142, 90) not in by_tile
    assert spec["equipment"] == [(141, 90)]
    assert spec["unplaced_tanks"] == 1
    # the ghost's 0x47 walk became a tick-1 place at the path's end
    assert ghost_events_for_tick(spec, 1) == [
        {"tick": 1, "tank_id": 500, "kind": "place", "x": 106, "y": 100, "message_id": 0}
    ]
    # self and unknown shooters/chatters never become ghost events
    assert [e for e in spec["events"] if e["kind"] in ("shoot", "chat")] == []
    # the FIRST 0x4C and 0x44 win; later ones are ignored
    assert spec["dot_atlas"] == [(140, 90)]
    assert spec["client_fuel"] == 900


def test_consume_chat_without_position_defaults_to_zero() -> None:
    """A 0x4D lacking its tile (decoder ``None``) records (0, 0)."""
    from tankpit_bot.sim.ghost import _consume, _Walk

    walk = _Walk()
    walk.self_id = 77
    walk.tick(_T0)
    _consume(
        walk,
        _T0 + 2000,
        ChatMessageDict(msg_type=0x4D, sender_id=500, message_type=41, x=None, y=None),
    )
    assert walk.chats == [(1, 500, 41, 0, 0)]
