"""Ghost replay — compiling a recording and replaying its opponents."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
)

from tankpit_bot.capture.xor import build_session_xor_table, xor_decode_body
from tankpit_bot.protocol.types import (
    BinaryMessage,
    ChatMessageDict,
    FuelGainDict,
    InventoryDict,
    MapDataDict,
    ShootEventDict,
)
from tankpit_bot.sim.commands import SimError
from tankpit_bot.sim.ghost import GhostTracker, compile_ghost_spec, ghost_events_for_tick
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.transport import encode_tick_payload
from tankpit_bot.sim.wire_statements import identity_statement, position_statement
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank, make_sim_world
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

_MAGIC = "ghosttestmagic"
_T0 = 1_785_000_000_000


def _world_for_statements(ghost_name: str = "Yuppler") -> SimWorldDict:
    """A statement-builder world: recorded self 77 and ghost 500."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][77] = make_sim_tank(77, 2, 3, 100, 100, 900)
    world["tanks"][500] = make_sim_tank(500, 1, 2, 108, 100, 600, name=ghost_name)
    return world


def _capture(timeline: list[tuple[int, list[BinaryMessage]]]) -> str:
    """Encode decoded message batches into a capture session text.

    Args:
        timeline: ``(timestamp_ms, batch)`` pairs, received direction.

    Returns:
        The ``capture_session.json`` text the compiler consumes.
    """
    table = build_session_xor_table(_MAGIC)
    messages = [
        {
            "timestamp_ms": t,
            "direction": "received",
            "payload": encode_tick_payload(batch, table),
        }
        for t, batch in timeline
    ]
    return dump_json_str(
        {
            "session_id": "ghost-test",
            "start_timestamp_ms": _T0,
            "end_timestamp_ms": _T0 + 60_000,
            "base_url": "wss://test/",
            "magic": _MAGIC,
            "messages": messages,
            "game_log": [],
            "tank_names": {},
        }
    )


def _fight_capture(ghost_name: str = "Yuppler") -> str:
    """A tiny recorded fight: join, ghost sighted, moves, shoots, chats."""
    world = _world_for_statements(ghost_name)
    join: list[BinaryMessage] = [
        identity_statement(world, 77),
        position_statement(world, 77),
        FuelGainDict(msg_type=0x44, fuel_total=900, is_free=False, flag=1),
        InventoryDict(
            msg_type=0x49,
            show=True,
            alternate=False,
            counts=[25, 20, 25, 19, 21],
            enabled=[True] * 5,
        ),
        MapDataDict(msg_type=0x4C, fuel_dots=[(140, 90), (150, 95)], tanks=[]),
        identity_statement(world, 500),
        position_statement(world, 500),
    ]
    world["tanks"][500]["x"] = 106
    ghost_moves: list[BinaryMessage] = [position_statement(world, 500)]
    ghost_shoots: list[BinaryMessage] = [
        ShootEventDict(
            msg_type=0x53,
            team=1,
            shooter_id=500,
            source_x=106,
            source_y=100,
            target_x=100,
            target_y=100,
            aim_x=100,
            aim_y=100,
            weapon=1,
        ),
        ChatMessageDict(msg_type=0x4D, sender_id=500, message_type=41, x=106, y=100),
    ]
    world["tanks"][77]["x"] = 102
    self_moves: list[BinaryMessage] = [position_statement(world, 77)]
    return _capture(
        [
            (_T0, join),
            (_T0 + 2000, ghost_moves),
            (_T0 + 4000, ghost_shoots),
            (_T0 + 6000, self_moves),
        ]
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


def _rich_capture() -> str:
    """The fight capture plus every world-read and stray-frame flavor."""
    import base64

    from tankpit_bot.protocol.types import (
        MovementDict,
        RadarContainerDict,
        RadarScanResultDict,
        ViewportEntityDict,
        ViewportUpdateDict,
    )

    world = _world_for_statements()
    world["tanks"][501] = make_sim_tank(501, 3, 1, 50, 50, 500, name="stranger")
    join: list[BinaryMessage] = [
        identity_statement(world, 77),
        position_statement(world, 77),
        FuelGainDict(msg_type=0x44, fuel_total=900, is_free=False, flag=1),
        InventoryDict(
            msg_type=0x49,
            show=True,
            alternate=False,
            counts=[25, 20, 25, 19, 21],
            enabled=[True] * 5,
        ),
        MapDataDict(msg_type=0x4C, fuel_dots=[(140, 90)], tanks=[]),
        identity_statement(world, 500),
        position_statement(world, 500),
        identity_statement(world, 501),  # never sighted -> unplaced
    ]
    reads: list[BinaryMessage] = [
        RadarScanResultDict(
            msg_type=0x4F,
            containers=[
                RadarContainerDict(x=140, y=90, volume=520),
                RadarContainerDict(x=141, y=90, volume=-1),
                RadarContainerDict(x=142, y=90, volume=0),
            ],
            mines=[],
            mine_clears=[],
        ),
        ViewportUpdateDict(
            msg_type=0x5A,
            viewport_left=92,
            viewport_top=92,
            entities=[
                ViewportEntityDict(col=9, row=9, cache_value=77, overlay_value=8, terrain_type=0)
            ],
        ),
        MovementDict(
            msg_type=0x47,
            tank_id=500,
            start_x=108,
            start_y=100,
            direction=0,
            damage_state=0,
            lb_score=0,
            rank=2,
            flag=0,
            is_carrying=False,
            waypoints=[(106, 100)],
            path_tiles=2,
            path="ww",
        ),
        ShootEventDict(
            msg_type=0x53,
            team=2,
            shooter_id=77,  # SELF shot: never a ghost event
            source_x=100,
            source_y=100,
            target_x=106,
            target_y=100,
            aim_x=106,
            aim_y=100,
            weapon=1,
        ),
        ShootEventDict(
            msg_type=0x53,
            team=3,
            shooter_id=502,  # unknown shooter: filtered at assembly
            source_x=50,
            source_y=50,
            target_x=100,
            target_y=100,
            aim_x=100,
            aim_y=100,
            weapon=0,
        ),
        ChatMessageDict(msg_type=0x4D, sender_id=502, message_type=3, x=50, y=50),
        ChatMessageDict(msg_type=0x4D, sender_id=77, message_type=2, x=100, y=100),  # self chat
        identity_statement(world, 77),  # duplicate SELF identity: ignored
        MapDataDict(msg_type=0x4C, fuel_dots=[(1, 1)], tanks=[]),  # later atlas: first wins
        FuelGainDict(msg_type=0x44, fuel_total=950, is_free=False, flag=1),  # later fuel read
    ]
    text = _capture([(_T0, join), (_T0 + 2000, reads)])
    session = narrow_json_load(text)
    stray_messages = list(narrow_json_to_list(session["messages"]))
    # 0x43 CacheUpdate rides the wire as a TOP-LEVEL frame — inside a
    # 0x2E envelope the 0x43 subtype byte means container_pickup, so
    # the sim's envelope encoder cannot carry it. Craft the raw frame:
    # the type byte travels in the clear; ``xor_decode_body`` at
    # ``offset=1`` strips it and XORs the remainder, so encoding the
    # payload = decoding a body with any placeholder type byte in front.
    table = build_session_xor_table(_MAGIC)
    cache_payload = bytes([143, 90, 260 & 0xFF, 260 >> 8])
    cache_body = bytes([0x43]) + xor_decode_body(bytes(1) + cache_payload, table, offset=1)
    cache_frame = bytes([len(cache_body), 0]) + cache_body
    stray_messages.append(
        {
            "timestamp_ms": _T0 + 2100,
            "direction": "received",
            "payload": base64.b64encode(cache_frame).decode(),
        }
    )
    # stray-frame flavors the compiler must skip: a sent command, an
    # empty payload, a plaintext ack, a text row, garbage bytes, and
    # a torn frame.
    stray_messages.extend(
        [
            {"timestamp_ms": _T0 + 100, "direction": "sent", "payload": "AAA="},
            {"timestamp_ms": _T0 + 200, "direction": "received", "payload": ""},
            {
                "timestamp_ms": _T0 + 300,
                "direction": "received",
                "payload": base64.b64encode(bytes([2, 0]) + b"A1").decode(),
            },
            {
                "timestamp_ms": _T0 + 400,
                "direction": "received",
                "payload": base64.b64encode(bytes([3, 0]) + b"+r|").decode(),
            },
            {
                "timestamp_ms": _T0 + 500,
                "direction": "received",
                "payload": base64.b64encode(
                    bytes([2, 0, 0x99, 0x01]) + bytes([9, 0]) + b"xx"
                ).decode(),
            },
            {
                "timestamp_ms": _T0 + 600,
                "direction": "received",
                "payload": base64.b64encode(bytes([3, 0]) + b"$ok").decode(),
            },
        ]
    )
    session["messages"] = stray_messages
    return dump_json_str(session)


def narrow_json_load(text: str) -> dict[str, JSONValue]:
    """Load capture JSON back to a mutable dict for fixture edits."""
    return dict(narrow_json_to_dict(load_json_str(text)))


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


def test_tracker_waits_for_the_first_recorded_tile() -> None:
    """Rounds before any recorded position compare against nothing."""
    tracker = GhostTracker({5: (100, 100)})
    tracker.note_round(0, 10, 10)
    assert tracker.compared_ticks == 0
    tracker.note_round(5, 100, 100)
    assert tracker.compared_ticks == 1


def test_seeding_skips_rock_but_keeps_water(fake_fs: FakeFileSystem) -> None:
    """Wire-real reads on water seed; rock reads are skipped."""
    from tankpit_bot.sim.ghost import seed_ghost_world_population

    spec = compile_ghost_spec(_rich_capture())
    terrain = InMemoryTerrainMap(terrain_data={(140, 90): "W", (143, 90): "#", (141, 90): "#"})
    from tankpit_bot.sim.world import SimContainerDict, SimEquipmentDict

    containers: list[SimContainerDict] = []
    equipment: list[SimEquipmentDict] = []
    skipped = seed_ghost_world_population(containers, equipment, spec, terrain)
    tiles = {(c["x"], c["y"]) for c in containers}
    assert (140, 90) in tiles  # water: kept
    assert (143, 90) not in tiles  # rock: skipped
    assert equipment == []  # rock equipment skipped
    assert skipped == 2
    open_containers: list[SimContainerDict] = []
    open_equipment: list[SimEquipmentDict] = []
    assert (
        seed_ghost_world_population(open_containers, open_equipment, spec, InMemoryTerrainMap())
        == 0
    )
    assert [(e["x"], e["y"]) for e in open_equipment] == [(141, 90)]


def test_ghost_atlas_composition_underlays_the_mined_room(fake_fs: FakeFileSystem) -> None:
    """``--ghost --from-atlas``: the atlas fills unobserved tiles, the
    recording's own dot atlas is the exposed set, and the capture's
    per-tile reads override the underlay."""
    from pathlib import Path

    from tankpit_bot import _test_hooks
    from tankpit_bot.sim.atlas_seed import DEFAULT_ATLAS_PATH
    from tankpit_bot.sim.run import main
    from tankpit_bot.sim.scenarios import SIM_FIELD

    entry = {
        "observations": 4,
        "sessions": 2,
        "first_ms": 1_000,
        "last_ms": 2_000,
        "max_fuel": 0,
        "equipment_seen": False,
        "visible_seen": True,
    }
    atlas_tiles = {
        # overridden by the capture's own (140, 90) read
        "140,90": {**entry, "last_v": 999, "max_fuel": 999},
        # unobserved by the capture, in its dot atlas -> dotted
        "150,95": {**entry, "last_v": 300, "max_fuel": 300},
        # unobserved, NOT in the dot atlas -> hidden
        "60,60": {**entry, "last_v": 400, "max_fuel": 400},
        # drained, not in the dot atlas -> seeds nothing in ghost mode
        "61,60": {**entry, "last_v": 0},
    }
    fake_fs.write_text(DEFAULT_ATLAS_PATH, dump_json_str({"1|field01.gif": atlas_tiles}))
    fake_fs.write_text(Path(SIM_FIELD), "fake-gif-bytes")
    # A BOT-named ghost: the reactive-policy driver constructs over it
    # (the certified roster policy under the recorded timeline).
    fake_fs.write_text(Path("runs/ghost-input.capture_session.json"), _fight_capture("orange-2"))
    _test_hooks.load_terrain_map = lambda gif_path: InMemoryTerrainMap()
    exit_code = main(
        [
            "--ghost",
            "runs/ghost-input.capture_session.json",
            "--from-atlas",
            "--rounds",
            "3",
            "--stamp",
            "20260801-000005",
        ]
    )
    assert exit_code == 0
    files = fake_fs.get_written_files()
    world_path = next(path for path in files if "sim-20260801-000005.world.json" in path)
    world_doc = narrow_json_to_dict(load_json_str(files[world_path]))
    containers = [narrow_json_to_dict(c) for c in narrow_json_to_list(world_doc["containers"])]
    by_tile = {(narrow_json_to_int(c["x"]), narrow_json_to_int(c["y"])): c for c in containers}
    # dot-atlas tile: dotted from the recording's exposed set
    assert by_tile[(150, 95)]["dotted"] is True
    # off-atlas underlay tile: hidden
    assert by_tile[(60, 60)]["dotted"] is False
    assert (61, 60) not in by_tile
    # the capture's own drained-dot read at (140, 90) beats the atlas
    assert narrow_json_to_int(by_tile[(140, 90)]["volume"]) == 0


def test_seed_ghost_world_relocates_or_skips_blocked_spawns() -> None:
    """A ghost sighted on impassable ground spawns at the nearest open
    tile; one with no open ground nearby is skipped."""
    from tankpit_bot.sim.ghost import GhostSpecDict, GhostTankDict
    from tankpit_bot.sim.run import _seed_ghost_world

    spec = GhostSpecDict(
        client_team=0,
        client_rank=1,
        client_x=10,
        client_y=10,
        client_fuel=800,
        client_counts=[25] * 5,
        ghosts=[
            GhostTankDict(tank_id=500, team=1, rank=1, name="rider", x=40, y=40),
            GhostTankDict(tank_id=501, team=1, rank=1, name="stuck", x=80, y=80),
        ],
        events=[],
        recorded_path={0: (10, 10)},
        containers=[],
        equipment=[],
        dot_atlas=[],
        ticks=1,
        unplaced_tanks=0,
    )
    rocks = {(40, 40): "#"}
    rocks.update({(80 + dx, 80 + dy): "#" for dx in range(-5, 6) for dy in range(-5, 6)})
    world = make_sim_world("field01_r.gif")
    _seed_ghost_world(world, InMemoryTerrainMap(terrain_data=rocks), spec, None)
    assert 500 in world["tanks"]
    spawned = world["tanks"][500]
    assert (spawned["x"], spawned["y"]) != (40, 40)
    assert abs(spawned["x"] - 40) <= 4 and abs(spawned["y"] - 40) <= 4
    assert 501 not in world["tanks"]


def test_dead_ghosts_skip_their_remaining_timeline() -> None:
    """Events of a corpse are dropped, not replayed."""
    from tankpit_bot.sim.ghost import GhostEventDict, GhostSpecDict
    from tankpit_bot.sim.run import _queue_ghost_round

    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][500] = make_sim_tank(500, 1, 1, 20, 20, 600)
    world["tanks"][500]["alive"] = False
    server = SimServer(world, InMemoryTerrainMap(), client_id=9)
    spec = GhostSpecDict(
        client_team=0,
        client_rank=1,
        client_x=10,
        client_y=10,
        client_fuel=800,
        client_counts=[25] * 5,
        ghosts=[],
        events=[
            GhostEventDict(tick=0, tank_id=500, kind="place", x=22, y=20, message_id=0),
            GhostEventDict(tick=0, tank_id=999, kind="shoot", x=10, y=10, message_id=0),
        ],
        recorded_path={},
        containers=[],
        equipment=[],
        dot_atlas=[],
        ticks=1,
        unplaced_tanks=0,
    )
    _queue_ghost_round(server, spec, 0)
    assert (world["tanks"][500]["x"], world["tanks"][500]["y"]) == (20, 20)
    batch = server.advance_tick()
    # nothing was queued: the tick carries only the client's sync
    assert [m["msg_type"] for m in batch] == [0x2E]
