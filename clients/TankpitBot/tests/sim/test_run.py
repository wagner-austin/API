"""``tankpit-sim-run`` — the sim session CLI and its scenario laws."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_bool,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_list,
)

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.capture.xor import XorStaticKeyUnavailableError, reset_static_key_cache
from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
from tankpit_bot.sim.run import (
    main,
    run_sim_session,
)
from tankpit_bot.sim.scenarios import (
    SIM_FIELD,
    SIM_MAGIC,
    _require_seeds_passable,
    make_default_sim_world,
    make_ferry_sim_world,
)
from tankpit_bot.sim.world_seed import (
    DOTTED_FUEL_COUNT,
    HIDDEN_EQUIPMENT_COUNT,
    HIDDEN_FUEL_COUNT,
    select_practice_layout,
)
from tankpit_bot.types import decode_capture_session
from tests.conftest import FakeEnv, FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _install_fake_terrain(fake_fs: FakeFileSystem) -> None:
    """Provide the field GIF and an all-passable terrain loader.

    Args:
        fake_fs: The installed fake file system.
    """
    fake_fs.write_text(Path(SIM_FIELD), "fake-gif-bytes")

    def load_fake_terrain(gif_path: Path) -> TerrainMapProtocol:
        """Return an open in-memory terrain for any requested field."""
        del gif_path
        return InMemoryTerrainMap()

    _test_hooks.load_terrain_map = load_fake_terrain


def test_default_scenario_seeds_are_passable_on_the_real_field() -> None:
    """The shipped arena must sit on genuinely open field01 ground.

    The first real-terrain run seeded the coastal (100, 100) region
    and drowned six containers — this pin keeps the default honest
    against the actual GIF (a git-tracked repo asset).
    """
    terrain = _test_hooks.load_terrain_map(Path(SIM_FIELD))
    _require_seeds_passable(make_default_sim_world(), terrain)


def test_seed_validation_rejects_impassable_tiles() -> None:
    """Seeds on rock — tank, container, or equipment — fail loudly."""
    world = make_default_sim_world()
    client = world["tanks"][9]
    container = world["containers"][0]
    equipment = world["equipment"][0]
    walls = {
        (client["x"], client["y"]): "#",
        (container["x"], container["y"]): "#",
        (equipment["x"], equipment["y"]): "#",
    }
    with pytest.raises(RuntimeError, match="fuel container"):
        _require_seeds_passable(world, InMemoryTerrainMap(terrain_data=walls))


def test_run_writes_capture_world_and_events(fake_fs: FakeFileSystem) -> None:
    """A short session archives all three artifacts in standard shapes."""
    _install_fake_terrain(fake_fs)
    result = run_sim_session(12, opponent=True, stamp="20260722-000004")
    assert result["stamp"] == "20260722-000004"
    assert 0 < result["rounds_played"] <= 12
    assert result["commands_sent"] > 0
    files = fake_fs.get_written_files()
    capture = decode_capture_session(
        narrow_json_to_dict(load_json_str(files[result["capture_path"]]))
    )
    assert capture["magic"] == SIM_MAGIC
    assert capture["messages"] != []
    world_snapshot = narrow_json_to_dict(load_json_str(files[result["world_path"]]))
    assert world_snapshot["field"] == SIM_FIELD
    assert files[result["events_path"]] != ""


def test_run_records_a_production_session_exit(fake_fs: FakeFileSystem) -> None:
    """A session that runs dry ends via the production exit path.

    Deterministic under the tick-paced session clock (2026-08-01:
    the decision clock advances one real 2 s tick per round, so TTL
    dynamics no longer depend on machine load — the old wall-clock
    version exited anywhere from round ~40 to never). With the
    scripted opponent passive (``opponent=False``) the combat-ready
    client kills the seeded enemy, finds nothing else alive, and the
    run must record the ``SessionExitError`` instead of crashing.
    """
    _install_fake_terrain(fake_fs)
    result = run_sim_session(200, opponent=False, stamp="20260722-000005")
    assert result["rounds_played"] < 200
    assert result["exit_reason"] in (
        "no_viable_targets",
        "out_of_fuel",
        "no_productive_collect",
        "deactivated",
    )
    assert result["exit_detail"] != ""


def test_boot_refuses_missing_terrain_and_missing_key(fake_fs: FakeFileSystem) -> None:
    """Both boot preconditions fail loudly, never best-effort."""
    with pytest.raises(RuntimeError, match="terrain GIF"):
        run_sim_session(1, opponent=False, stamp="20260722-000006")
    _install_fake_terrain(fake_fs)
    fake_fs.remove(DEFAULT_STATIC_KEY_PATH)
    # The static KEY is cached process-wide (only the session TABLE is
    # per-session), so removing the file is not enough on its own —
    # the earlier boot already read it ([[session-state-deglobalisation]]).
    reset_static_key_cache()
    with pytest.raises(XorStaticKeyUnavailableError, match="static XOR key unavailable"):
        run_sim_session(1, opponent=False, stamp="20260722-000007")


def test_main_parses_arguments_and_reports(fake_fs: FakeFileSystem) -> None:
    """The manual argument loop drives one session and returns 0."""
    _install_fake_terrain(fake_fs)
    exit_code = main(
        ["--rounds", "6", "--no-opponent", "--stamp", "20260722-000008", "--mystery", "--rounds"]
    )
    assert exit_code == 0
    files = fake_fs.get_written_files()
    assert any("sim-20260722-000008.capture_session.json" in path for path in files)


def test_human_opponent_session_runs_under_the_consent_gate(fake_fs: FakeFileSystem) -> None:
    """A ``--human-opponent`` session soaks the human-fight paths end to end.

    The full story, asserted from the events stream so a session that
    stalls at the consent gate can never pass silently (the first
    real soak, 2026-07-31, exited ``no_viable_targets`` two rounds in
    — the greeting had burned its latch on an unsynced tank and the
    original weaker asserts still passed): the bot GREET-APPROACHES
    the unconsented human (stand-off teleport), the scripted opponent
    shoots first — which consents it under the 2026-07-30 contract —
    and the production bot fights and KILLS it as a human.
    """
    _install_fake_terrain(fake_fs)
    exit_code = main(["--rounds", "60", "--stamp", "20260731-000001", "--human-opponent", "guest"])
    assert exit_code == 0
    files = fake_fs.get_written_files()
    world_path = next(path for path in files if "sim-20260731-000001.world.json" in path)
    world_snapshot = narrow_json_to_dict(load_json_str(files[world_path]))
    tanks = narrow_json_to_list(world_snapshot["tanks"])
    names = [narrow_json_to_dict(tank)["name"] for tank in tanks]
    assert "guest" in names
    capture_path = next(
        path for path in files if "sim-20260731-000001.capture_session.json" in path
    )
    capture = decode_capture_session(narrow_json_to_dict(load_json_str(files[capture_path])))
    assert capture["messages"] != []
    events_path = next(path for path in files if path.endswith("latest.sim.events.jsonl"))

    def _event_kind_seen(kind: str) -> bool:
        for line in files[events_path].splitlines():
            if not line:
                continue
            if narrow_json_to_dict(load_json_str(line)).get("diagnostic_kind") == kind:
                return True
        return False

    assert _event_kind_seen("greeting_approach") is True
    assert _event_kind_seen("chat_greeting") is True
    assert _event_kind_seen("tank_deactivated") is True


def test_practice_session_seeds_the_mined_layout_and_population(
    fake_fs: FakeFileSystem,
) -> None:
    """Practice mode seeds the stamp-selected real layout: the full
    36-bot roster, the client at its mined join spawn, and the static
    container field (dotted atlas + hidden fuel + hidden equipment).
    The session archives normally."""
    _install_fake_terrain(fake_fs)
    result = run_sim_session(14, practice=True, stamp="20260725-000001")
    assert result["rounds_played"] >= 1
    layout = select_practice_layout("20260725-000001")
    world_doc = narrow_json_to_dict(load_json_str(fake_fs.read_text(Path(result["world_path"]))))
    raw_tanks = narrow_json_to_list(world_doc["tanks"])
    tank_ids = {narrow_json_to_dict(entry)["tank_id"] for entry in raw_tanks}
    for roster_id, _team, _rank, _x, _y in layout["roster"]:
        assert roster_id in tank_ids
    raw_containers = narrow_json_to_list(world_doc["containers"])
    dotted = sum(1 for entry in raw_containers if narrow_json_to_dict(entry)["dotted"] is True)
    hidden = sum(1 for entry in raw_containers if narrow_json_to_dict(entry)["dotted"] is False)
    # Conservation: exposure moves containers hidden -> dotted, never
    # creates or destroys them (the measured-density world is sparse
    # enough that a short session may dot nothing; the exposure law
    # itself is pinned by test_radar_exposure_dots_large_hidden_fuel).
    assert dotted + hidden == DOTTED_FUEL_COUNT + HIDDEN_FUEL_COUNT
    assert dotted >= DOTTED_FUEL_COUNT
    assert len(narrow_json_to_list(world_doc["equipment"])) >= HIDDEN_EQUIPMENT_COUNT


def test_main_practice_flag_drives_a_roster_session(fake_fs: FakeFileSystem) -> None:
    """`--practice` reaches run_sim_session and archives normally."""
    _install_fake_terrain(fake_fs)
    exit_code = main(["--rounds", "4", "--practice", "--stamp", "20260725-000002"])
    assert exit_code == 0
    files = fake_fs.get_written_files()
    assert any("sim-20260725-000002.capture_session.json" in path for path in files)


def test_ferry_scenario_seeds_are_legal_on_the_real_field() -> None:
    """The shipped ferry world must fit the actual field01 lake.

    The floating containers sit on real water served by the seeded
    ferry, the ferry floats, and every land seed is genuinely open —
    all verified against the git-tracked GIF.
    """
    terrain = _test_hooks.load_terrain_map(Path(SIM_FIELD))
    _require_seeds_passable(make_ferry_sim_world(), terrain)


def test_water_containers_are_legal_without_a_ferry() -> None:
    """Floating containers pass validation on their own.

    The longitudinal atlas ([[game-economy]] 2026-08-01) proved the
    water-locked population is real live state and ferries drift, so
    a ferry-service requirement would reject the true field. Only
    rock is a typo.
    """
    world = make_ferry_sim_world()
    world["ferries"].clear()
    lake = {(x, y): "W" for x in range(111, 131) for y in range(105, 121)}
    _require_seeds_passable(world, InMemoryTerrainMap(terrain_data=lake))


def test_seed_validation_rejects_grounded_ferries() -> None:
    """A ferry seeded on dry land is a harness typo, not a scenario."""
    world = make_ferry_sim_world()
    world["ferries"][0]["x"] = 5
    world["ferries"][0]["y"] = 5
    lake = {(x, y): "W" for x in range(111, 131) for y in range(105, 121)}
    with pytest.raises(RuntimeError, match="ferry at"):
        _require_seeds_passable(world, InMemoryTerrainMap(terrain_data=lake))


def _install_lake_terrain(fake_fs: FakeFileSystem) -> None:
    """Provide the field GIF and a terrain with the scenario's lake.

    The water body mirrors the real field01 lake the ferry scenario
    is built on: the floating seeds are water-locked, the ferry
    floats clear of land, and the west shore is open ground.

    Args:
        fake_fs: The installed fake file system.
    """
    fake_fs.write_text(Path(SIM_FIELD), "fake-gif-bytes")
    lake = {(x, y): "W" for x in range(111, 131) for y in range(105, 121)}

    def load_lake_terrain(gif_path: Path) -> TerrainMapProtocol:
        """Return the lake terrain for any requested field."""
        del gif_path
        return InMemoryTerrainMap(terrain_data=lake)

    _test_hooks.load_terrain_map = load_lake_terrain


def test_ferry_session_boards_and_drains_the_water_larder(
    fake_fs: FakeFileSystem,
) -> None:
    """``--ferry`` soaks the full F5 chain through the production bot.

    Asserted from the final world so a session that never touches the
    water cannot pass: a ferry belief forms, the larder hop boards
    the ferry, the ride drains the water container, and the client
    ends richer than its 400-fuel spawn.

    Re-scoped 2026-08-13 with the recon reorder (known stock preempts
    scanning, HUD flags 8/9/14): the soak no longer requires a scope
    pan. In the sim the belief arrives from the drift law's 0x4A
    broadcast before any pan is needed, and the larder boards
    directly — the fastest correct chain. Live discovery when no
    broadcast serves the belief stays covered where it lives: the
    ferry scout's firing conditions in
    ``tests/bot/ai/test_scope_scout.py`` (it pans on the larder's
    water-locked ``no_landing`` decline), and the sweep's own pins in
    ``tests/bot/ai/test_quad_sweep.py``.
    """
    _install_lake_terrain(fake_fs)
    exit_code = main(["--ferry", "--rounds", "60", "--stamp", "20260801-000001"])
    assert exit_code == 0
    files = fake_fs.get_written_files()
    world_path = next(path for path in files if "sim-20260801-000001.world.json" in path)
    world_doc = narrow_json_to_dict(load_json_str(files[world_path]))
    containers = [narrow_json_to_dict(c) for c in narrow_json_to_list(world_doc["containers"])]
    lake_container = next(c for c in containers if c["x"] == 112 and c["y"] == 112)
    assert narrow_json_to_int(lake_container["volume"]) < 700
    tanks = [narrow_json_to_dict(t) for t in narrow_json_to_list(world_doc["tanks"])]
    client = next(t for t in tanks if t["tank_id"] == 9)
    assert narrow_json_to_int(client["fuel"]) > 400
    assert narrow_json_to_bool(client["alive"]) is True


def _install_wrong_pond_terrain(fake_fs: FakeFileSystem) -> None:
    """The live-deadlock geometry: the ferry's puddle is not the pond.

    Runs bot-20260804-234008 and bot-20260805-070006 both stalled on a
    ferry docked one land ridge away from the goal's pond (field01
    truth: the (106,11-12) pond holds 4,456 water tiles and does not
    contain (112,15)). Here each water seed floats on its own tiny
    pond and the scenario's ferry tile (118,112) sits alone in a 3x3
    puddle, so no goal ever shares a water body with a ferry. The
    ponds total 43 water tiles — below ``_FERRY_WATER_SPACING`` (300)
    — because ``seed_ferries`` floats a ferry on every 300th water
    tile map-wide, and a first cut of this world used a 304-tile lake:
    the seeder placed one SAME-pond ferry on it and the gate correctly
    boarded it, which is the happy soak again, not this test.

    Args:
        fake_fs: The installed fake file system.
    """
    fake_fs.write_text(Path(SIM_FIELD), "fake-gif-bytes")
    goal_pond = {(x, y) for x in range(110, 115) for y in range(110, 115)}
    ferry_puddle = {(x, y) for x in range(117, 120) for y in range(111, 114)}
    deep_pond = {(x, y) for x in range(119, 122) for y in range(115, 118)}
    water = dict.fromkeys(goal_pond | ferry_puddle | deep_pond, "W")

    def load_wrong_pond_terrain(gif_path: Path) -> TerrainMapProtocol:
        """Return the split-pond terrain for any requested field."""
        del gif_path
        return InMemoryTerrainMap(terrain_data=water)

    _test_hooks.load_terrain_map = load_wrong_pond_terrain


def test_ferry_session_never_boards_the_wrong_pond_ferry(
    fake_fs: FakeFileSystem,
) -> None:
    """A ferry off the goal's water body is declined, never boarded.

    The sim-side pin of the pond gate the 2026-08-05 log queued: the
    happy soak above proves a same-pond ferry IS boarded; this world
    proves the adversarial geometry that live play authored twice is
    declined instead of deadlocked on. The paired assertions kill the
    gate's mutant both ways — the water fuel must be CONSIDERED and
    refused ``no_landing`` (a session that never looks at the water
    proves nothing), and no larder hop may ever land inside the
    ferry's puddle, drifted or not.
    """
    _install_wrong_pond_terrain(fake_fs)
    exit_code = main(["--ferry", "--rounds", "60", "--stamp", "20260813-000001"])
    assert exit_code == 0
    files = fake_fs.get_written_files()
    events_path = next(path for path in files if path.endswith("latest.sim.events.jsonl"))
    declined_no_landing = 0
    for line in files[events_path].splitlines():
        if not line:
            continue
        event = narrow_json_to_dict(load_json_str(line))
        if event.get("hop_kind") != "fuel_larder":
            continue
        if event.get("diagnostic_kind") == "hop_selected":
            landing_x = narrow_json_to_int(event["landing_x"])
            landing_y = narrow_json_to_int(event["landing_y"])
            assert not (117 <= landing_x <= 119 and 111 <= landing_y <= 113)
        if event.get("diagnostic_kind") == "hop_declined":
            assert narrow_json_to_int(event["ferry_served"]) == 0
            if narrow_json_to_int(event["no_landing"]) > 0:
                declined_no_landing += 1
    assert declined_no_landing > 0
    world_path = next(path for path in files if "sim-20260813-000001.world.json" in path)
    world_doc = narrow_json_to_dict(load_json_str(files[world_path]))
    containers = [narrow_json_to_dict(c) for c in narrow_json_to_list(world_doc["containers"])]
    lake_fuel = next(c for c in containers if c["x"] == 112 and c["y"] == 112)
    deep_fuel = next(c for c in containers if c["x"] == 120 and c["y"] == 115)
    assert narrow_json_to_int(lake_fuel["volume"]) == 700
    assert narrow_json_to_int(deep_fuel["volume"]) == 500
    tanks = [narrow_json_to_dict(t) for t in narrow_json_to_list(world_doc["tanks"])]
    client = next(t for t in tanks if t["tank_id"] == 9)
    assert narrow_json_to_bool(client["alive"]) is True


def _write_small_atlas(fake_fs: FakeFileSystem, path: Path, stamp: str) -> tuple[int, int]:
    """A three-tile mined atlas beside the stamp-selected spawn.

    The stamp picks the practice layout and therefore the client
    spawn; the seeded fuel must be walk-close or a lean 400-fuel
    spawn cannot afford the hop (first cut placed tiles at a fixed
    (134,126) and the (12,56)-spawning layout starved 122 tiles
    away).

    Returns:
        The fuel container's tile.
    """
    spawn_x, spawn_y = select_practice_layout(stamp)["client_spawn"]
    entry = {
        "observations": 4,
        "sessions": 2,
        "first_ms": 1_000,
        "last_ms": 2_000,
        "max_fuel": 0,
        "equipment_seen": False,
        "visible_seen": True,
    }
    fuel_tile = (spawn_x + 3, spawn_y)
    tiles = {
        f"{fuel_tile[0]},{fuel_tile[1]}": {**entry, "last_v": 650, "max_fuel": 650},
        f"{spawn_x + 4},{spawn_y + 1}": {**entry, "last_v": -1, "equipment_seen": True},
        f"{spawn_x + 2},{spawn_y + 2}": {**entry, "last_v": 0},
    }
    fake_fs.write_text(path, dump_json_str({"1|field01.gif": tiles}))
    return fuel_tile


def test_atlas_session_forages_the_mined_room(fake_fs: FakeFileSystem) -> None:
    """``--from-atlas PATH`` runs a pure-forage session on the real field.

    The lean spawn forages the mined stock: by session end the seeded
    650-volume container is drained into the client (fuel above the
    400 spawn) and the world holds the atlas population, not the
    statistical one.
    """
    _install_fake_terrain(fake_fs)
    atlas_path = Path("runs/analysis/small_atlas.json")
    fuel_tile = _write_small_atlas(fake_fs, atlas_path, "20260801-000002")
    exit_code = main(
        ["--rounds", "40", "--stamp", "20260801-000002", "--from-atlas", str(atlas_path)]
    )
    assert exit_code == 0
    files = fake_fs.get_written_files()
    world_path = next(path for path in files if "sim-20260801-000002.world.json" in path)
    world_doc = narrow_json_to_dict(load_json_str(files[world_path]))
    containers = [narrow_json_to_dict(c) for c in narrow_json_to_list(world_doc["containers"])]
    drained = next(c for c in containers if (c["x"], c["y"]) == fuel_tile)
    assert narrow_json_to_int(drained["volume"]) == 0
    tanks = [narrow_json_to_dict(t) for t in narrow_json_to_list(world_doc["tanks"])]
    client = next(t for t in tanks if t["tank_id"] == 9)
    assert narrow_json_to_int(client["fuel"]) > 400


def test_practice_atlas_composition_replaces_the_statistical_field(
    fake_fs: FakeFileSystem,
) -> None:
    """``--practice --from-atlas`` seeds the roster ON the mined room."""
    from tankpit_bot.sim.atlas_seed import DEFAULT_ATLAS_PATH

    _install_fake_terrain(fake_fs)
    _write_small_atlas(fake_fs, DEFAULT_ATLAS_PATH, "20260801-000003")
    exit_code = main(["--rounds", "3", "--practice", "--from-atlas", "--stamp", "20260801-000003"])
    assert exit_code == 0
    files = fake_fs.get_written_files()
    world_path = next(path for path in files if "sim-20260801-000003.world.json" in path)
    world_doc = narrow_json_to_dict(load_json_str(files[world_path]))
    tanks = [narrow_json_to_dict(t) for t in narrow_json_to_list(world_doc["tanks"])]
    layout = select_practice_layout("20260801-000003")
    tank_ids = {narrow_json_to_int(t["tank_id"]) for t in tanks}
    for roster_id, _team, _rank, _x, _y in layout["roster"]:
        assert roster_id in tank_ids
    # The atlas population REPLACES the ~1,600-container statistical
    # field: only the mined tiles (minus whatever the session drained
    # or picked) are seeded.
    containers = narrow_json_to_list(world_doc["containers"])
    assert len(containers) <= 2


def test_a_session_whose_join_never_reaches_a_room_fails_loudly(
    fake_fs: FakeFileSystem,
    fake_env: FakeEnv,
) -> None:
    """A sim run that never entered a room is not a session.

    The lobby is the real production flow now, so it can genuinely
    fail — ask for a room the sim does not advertise and the join
    times out. Booting on anyway would put the bot in a world it never
    joined, which is the shape the hand-installed selected room used to
    hide ([[session-state-deglobalisation]]).
    """
    _install_fake_terrain(fake_fs)
    fake_env.set("TANKPIT_ROOM", "Atlantis")

    with pytest.raises(RuntimeError, match="did not reach a room"):
        run_sim_session(3, opponent=True, stamp="20260807-000001")
