"""Tests for the larder-gate (own-tile equipment pickup) probe."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_list
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    ClockAdvancingPage,
    ReplayClock,
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.conftest import FakeFileSystem
from tests.in_memory_terrain_map import InMemoryTerrainMap

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.larder_probe import (
    LarderAttemptDict,
    LarderAttemptStatus,
    LarderProbe,
    LarderProbeSessionDict,
    encode_larder_probe_session,
    format_larder_probe_summary,
    run_larder_probe,
)
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state
from tankpit_bot.state.types import (
    ContainerStateDict,
    make_container_state,
    make_self_state,
    make_viewport_state,
)

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


def _make_world(timestamp_ms: int, x: int, y: int, fuel: int) -> WorldStateDict:
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1,
            x=x,
            y=y,
            team=2,
            rank=1,
            fuel=fuel,
            leaderboard_position=5,
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=make_viewport_state(left=0, top=0, width=16, height=16),
        scanned_tiles=world["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def _slots(count: int) -> InventoryState:
    item = InventoryItem(count=count, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=item,
        missile_shots=item,
        homing_shots=item,
        extra_radars=item,
    )


def _equipment(x: int, y: int, *, failed_pickups: int = 0) -> ContainerStateDict:
    return make_container_state(
        x=x,
        y=y,
        is_fuel=False,
        volume=0,
        timestamp_ms=1000,
        failed_pickups=failed_pickups,
    )


def _install_noop_drain() -> None:
    def _drain(provider: BufferedMessageSourceProtocol) -> int:
        del provider
        return 0

    action_hooks.drain_buffered_messages = _drain


class _LarderHarness(LarderProbe):
    """Fake-wire harness: commands mutate local state, no browser."""

    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=True)
        self._clock = ReplayClock(1000)
        self._page = ClockAdvancingPage(self._clock)
        self.fuel = 5000
        self.position = (100, 100)
        self.visible_containers: dict[str, ContainerStateDict] = {}
        self.slot_count = 1
        self.map_calls = 0
        self.radar_calls = 0
        self.teleports: list[tuple[int, int]] = []
        self.moves: list[tuple[int, int]] = []
        self.pickups: list[tuple[int, int]] = []
        self.pays_on_tile = False
        self.pays_adjacent = False
        self.move_script: list[bool] = []
        get_world_service().inventory_state = _slots(self.slot_count)
        get_world_service().terrain_map = InMemoryTerrainMap()

    def open_map(self) -> bool:
        self.map_calls += 1
        return True

    def use_radar(self) -> bool:
        self.radar_calls += 1
        return True

    def request_inventory(self) -> bool:
        return True

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleports.append((x, y))
        self.position = (x, y)
        return True

    def move_to(self, x: int, y: int) -> bool:
        self.moves.append((x, y))
        arrives = self.move_script.pop(0) if self.move_script else True
        if arrives:
            self.position = (x, y)
        return True

    def pickup_equipment(self, x: int, y: int) -> bool:
        self.pickups.append((x, y))
        on_tile = self.position == (x, y)
        adjacent = abs(self.position[0] - x) + abs(self.position[1] - y) == 1
        if (on_tile and self.pays_on_tile) or (adjacent and self.pays_adjacent):
            self.slot_count += 1
            get_world_service().inventory_state = _slots(self.slot_count)
        return True

    def get_world_state(self) -> WorldStateDict:
        world = _make_world(1000, self.position[0], self.position[1], self.fuel)
        return WorldStateDict(
            self_state=world["self_state"],
            tanks=world["tanks"],
            containers=self.visible_containers,
            mines=world["mines"],
            terrain=world["terrain"],
            viewport=world["viewport"],
            scanned_tiles=world["scanned_tiles"],
            timestamp_ms=world["timestamp_ms"],
        )

    def get_self_state(self) -> SelfStateDict | None:
        return make_self_state(
            tank_id=1,
            x=self.position[0],
            y=self.position[1],
            team=2,
            rank=1,
            fuel=self.fuel,
            leaderboard_position=1,
        )


def _harness() -> _LarderHarness:
    probe = _LarderHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    return probe


def test_inventory_total_reads_the_wire_state() -> None:
    """The total is the live sum of all five slot counts."""
    probe = _harness()
    get_world_service().inventory_state = _slots(3)
    assert probe._inventory_total() == 15


def test_nearest_equipment_skips_fuel_failed_and_tried() -> None:
    """Fuel, blacklisted, and already-attempted containers never win."""
    probe = _harness()
    probe.visible_containers = {
        "90,100": make_container_state(
            x=90, y=100, is_fuel=True, volume=500, timestamp_ms=1000, failed_pickups=0
        ),
        "99,100": _equipment(99, 100, failed_pickups=2),
        "98,100": _equipment(98, 100),
        "140,140": _equipment(140, 140),
        "104,100": _equipment(104, 100),
        "150,150": _equipment(150, 150),
    }
    found = probe._nearest_equipment({(98, 100)})
    assert found == _equipment(104, 100)


def test_nearest_equipment_skips_water_sitting_containers() -> None:
    """The first live run's failure mode, pinned: shore containers ON
    water can never host the own-tile trial and are never candidates."""
    probe = _harness()
    get_world_service().terrain_map = InMemoryTerrainMap({(101, 100): "W"})
    probe.visible_containers = {
        "101,100": _equipment(101, 100),
        "110,100": _equipment(110, 100),
    }
    assert probe._nearest_equipment(set()) == _equipment(110, 100)


def test_nearest_equipment_none_when_no_candidates() -> None:
    probe = _harness()
    assert probe._nearest_equipment(set()) is None


def test_nearest_equipment_requires_the_terrain_map() -> None:
    probe = _harness()
    get_world_service().terrain_map = None
    get_world_service().selected_room = None
    with pytest.raises(ProbeError, match="terrain map is unavailable"):
        probe._nearest_equipment(set())


def test_search_equipment_returns_visible_without_spending() -> None:
    """An already-believed container costs zero scans and zero hops."""
    probe = _harness()
    probe.visible_containers = {"104,100": _equipment(104, 100)}
    found, scans, hops = probe._search_equipment(set(), 6)
    assert found == _equipment(104, 100)
    assert (scans, hops) == (0, 0)
    assert probe.radar_calls == 0


def test_search_equipment_hops_and_scans_until_found() -> None:
    """The nearest-first site sweep reveals equipment via one extra scan."""

    class _RevealingHarness(_LarderHarness):
        def use_radar(self) -> bool:
            self.visible_containers = {"97,96": _equipment(97, 96)}
            return super().use_radar()

    probe = _RevealingHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    found, scans, hops = probe._search_equipment(set(), 6)
    assert found == _equipment(97, 96)
    assert (scans, hops) == (1, 1)
    assert probe.teleports == [(96, 96)]


def test_search_equipment_stops_at_the_scan_budget() -> None:
    """A dry sweep never exceeds the extras budget."""
    probe = _harness()
    found, scans, hops = probe._search_equipment(set(), 2)
    assert found is None
    assert (scans, hops) == (2, 2)
    assert probe.radar_calls == 2


def test_search_equipment_skips_unlanded_sites_without_scanning() -> None:
    """A rejected site teleport preserves its extra, like the density sweep."""

    class _StuckHarness(_LarderHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            return True

    probe = _StuckHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().map_fuel_dots = ()
    found, scans, hops = probe._search_equipment(set(), 2)
    # (96, 96) is within landing tolerance of the (100, 100) start, so
    # exactly one site scans; every other rejected teleport is skipped.
    assert found is None
    assert scans == 1
    assert hops == 16


def test_step_off_returns_immediately_when_already_adjacent() -> None:
    probe = _harness()
    probe.position = (105, 100)
    assert probe._step_off(104, 100) is True
    assert probe.moves == []


def test_step_off_walks_to_the_first_cardinal_neighbor() -> None:
    probe = _harness()
    probe.position = (104, 100)
    assert probe._step_off(104, 100) is True
    assert probe.moves == [(105, 100)]


def test_step_off_fails_when_no_neighbor_is_reachable() -> None:
    probe = _harness()
    probe.position = (104, 100)
    probe.move_script = [False, False, False, False]
    assert probe._step_off(104, 100) is False
    assert len(probe.moves) == 4


def test_attempt_landing_on_tile_and_own_pickup_pays() -> None:
    """The gate's YES case: land ON the container, own-tile pickup credits."""
    probe = _harness()
    probe.pays_on_tile = True
    attempt = probe._attempt_container(_equipment(104, 100))
    assert attempt["status"] == "own_tile_pickup"
    assert attempt["landed_on_container"] is True
    assert attempt["walked_onto_container"] is False
    assert attempt["stood_on_container"] is True
    assert attempt["own_tile_sent"] is True
    assert attempt["own_tile_picked"] is True
    assert attempt["stepped_off"] is False
    assert attempt["adjacent_sent"] is False
    assert attempt["inventory_after"] == attempt["inventory_before"] + 5
    assert probe.pickups == [(104, 100)]


def test_attempt_own_tile_fails_then_adjacent_control_pays() -> None:
    """The gate's NO case: own-tile silent, the adjacent control credits."""
    probe = _harness()
    probe.pays_adjacent = True
    attempt = probe._attempt_container(_equipment(104, 100))
    assert attempt["status"] == "adjacent_pickup"
    assert attempt["own_tile_sent"] is True
    assert attempt["own_tile_picked"] is False
    assert attempt["stepped_off"] is True
    assert attempt["adjacent_sent"] is True
    assert attempt["adjacent_picked"] is True
    assert probe.pickups == [(104, 100), (104, 100)]
    assert probe.moves == [(105, 100)]


def test_attempt_displaced_landing_walks_onto_the_tile() -> None:
    """A displaced teleport walks onto the container before the own trial."""

    class _DisplacingHarness(_LarderHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            self.position = (x + 3, y)
            return True

    probe = _DisplacingHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.move_script = [True, False, False, False, False]
    attempt = probe._attempt_container(_equipment(104, 100))
    assert attempt["status"] == "no_pickup"
    assert attempt["landed_on_container"] is False
    assert attempt["walked_onto_container"] is True
    assert attempt["own_tile_sent"] is True
    assert attempt["own_tile_picked"] is False
    assert attempt["stepped_off"] is False
    assert attempt["adjacent_sent"] is False


def test_attempt_never_stood_still_runs_the_adjacent_control() -> None:
    """When the tile is unreachable the attempt still proves the container."""

    class _AdjacentHarness(_LarderHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            self.position = (x + 1, y)
            return True

    probe = _AdjacentHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.pays_adjacent = True
    probe.move_script = [False]
    attempt = probe._attempt_container(_equipment(104, 100))
    assert attempt["status"] == "adjacent_pickup"
    assert attempt["stood_on_container"] is False
    assert attempt["own_tile_sent"] is False
    assert attempt["stepped_off"] is True
    assert attempt["adjacent_picked"] is True


def test_execute_larder_probe_rejects_bad_budgets() -> None:
    probe = _LarderHarness()
    with pytest.raises(ProbeError, match="max_attempts must be positive"):
        probe.execute_larder_probe(max_attempts=0, max_extras=6, initial_sync_timeout_ms=1000)
    with pytest.raises(ProbeError, match="max_extras must be positive"):
        probe.execute_larder_probe(max_attempts=3, max_extras=0, initial_sync_timeout_ms=1000)


def _attempt(
    *,
    own_sent: bool,
    own_picked: bool,
    adjacent_picked: bool,
) -> LarderAttemptDict:
    if own_picked:
        status: LarderAttemptStatus = "own_tile_pickup"
    elif adjacent_picked:
        status = "adjacent_pickup"
    else:
        status = "no_pickup"
    return LarderAttemptDict(
        container_x=104,
        container_y=100,
        landed_x=104,
        landed_y=100,
        landed_on_container=True,
        walked_onto_container=False,
        stood_on_container=True,
        own_tile_sent=own_sent,
        own_tile_picked=own_picked,
        stepped_off=not own_picked,
        adjacent_sent=not own_picked,
        adjacent_picked=adjacent_picked,
        inventory_before=5,
        inventory_after=6,
        status=status,
    )


def _session() -> LarderProbeSessionDict:
    return LarderProbeSessionDict(
        session_id="larder-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="larder_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 100,
            "intel_ready_timestamp_ms": 150,
            "initial_sync_started_ms": 200,
            "initial_world_timestamp_ms": 400,
            "command_ready_timestamp_ms": 450,
            "first_attempt_started_ms": 500,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 250,
            "initial_world_to_command_ready_ms": 50,
            "command_ready_to_first_attempt_ms": 50,
        },
        max_attempts=3,
        max_extras=6,
        search_scans=2,
        search_hops=3,
        attempts=[
            _attempt(own_sent=True, own_picked=True, adjacent_picked=False),
            _attempt(own_sent=True, own_picked=False, adjacent_picked=True),
        ],
        own_tile_successes=1,
        own_tile_failures=1,
        adjacent_successes=1,
        extras_before=20,
        extras_enabled_at_start=False,
        toggles_sent=2,
        extras_after=18,
        fuel_before=5000,
        fuel_after=4400,
    )


def test_encode_and_summary() -> None:
    session = _session()
    encoded = encode_larder_probe_session(session)
    assert encoded["max_attempts"] == 3
    assert encoded["own_tile_successes"] == 1
    attempts = narrow_json_to_list(encoded["attempts"])
    assert len(attempts) == 2
    first = narrow_json_to_dict(attempts[0])
    assert first["status"] == "own_tile_pickup"
    assert first["container_x"] == 104
    assert format_larder_probe_summary(session) == (
        "Larder probe complete: attempts=2/3 own-tile 1/2 adjacent=1 "
        "scans=2 hops=3 extras 20->18 fuel 5000->4400"
    )


class _LarderModuleProtocol(Protocol):
    LarderProbe: type[LarderProbe]


_larder_module_import = __import__(
    "tankpit_bot.action_lab.larder_probe",
    fromlist=["larder_probe"],
)
larder_module: _LarderModuleProtocol = _larder_module_import


class _FakeLarderProbe(LarderProbe):
    def execute_larder_probe(
        self,
        *,
        max_attempts: int,
        max_extras: int,
        initial_sync_timeout_ms: int,
    ) -> LarderProbeSessionDict:
        session = _session()
        session["max_attempts"] = max_attempts
        session["max_extras"] = max_extras
        session["initial_sync_timeout_ms"] = initial_sync_timeout_ms
        session["capture_session_path"] = ""
        return session


def test_run_larder_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    original_class = larder_module.LarderProbe
    larder_module.LarderProbe = _FakeLarderProbe
    try:
        session = run_larder_probe(
            "https://tankpit.com/play",
            "larder_probe.json",
            max_attempts=2,
            max_extras=4,
        )
    finally:
        larder_module.LarderProbe = original_class

    written = fake_fs.read_text(Path("larder_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["capture_session_path"] == "larder_probe.capture_session.json"
    assert decoded["max_attempts"] == 2
    assert decoded["max_extras"] == 4
    assert session["own_tile_successes"] == 1


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, LarderProbe):
    def __init__(self, *, containers_to_serve: int) -> None:
        LarderProbe.__init__(self, "https://tankpit.com/play", headless=False, prefer_account=True)
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 5000)
        self.phases: list[str] = []
        self.containers_to_serve = containers_to_serve

    def _current_fuel(self) -> tuple[int, int, int]:
        self.phases.append("fuel")
        return 5000, 100, 100

    def _ensure_extras_enabled(self) -> tuple[int, bool, int]:
        self.phases.append("enable")
        return 20, False, 1

    def _search_equipment(
        self,
        tried: set[tuple[int, int]],
        scans_left: int,
    ) -> tuple[ContainerStateDict | None, int, int]:
        self.phases.append(f"search:{scans_left}")
        if self.containers_to_serve <= 0:
            return None, 1, 2
        self.containers_to_serve -= 1
        return _equipment(104 + len(tried), 100), 1, 1

    def _attempt_container(self, container: ContainerStateDict) -> LarderAttemptDict:
        self.phases.append(f"attempt:{container['x']}")
        return _attempt(own_sent=True, own_picked=False, adjacent_picked=True)

    def _restore_extras_state(self, was_enabled: bool) -> int:
        self.phases.append(f"restore:{was_enabled}")
        return 1

    def _read_extras(self) -> tuple[int, bool]:
        self.phases.append("read")
        return 18, False

    def _quit_to_lobby(self) -> None:
        self.phases.append("quit")


def _run_execute_harness(probe: _ExecuteHarness, *, max_attempts: int) -> LarderProbeSessionDict:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    original_sync_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = recorded.sync_playwright_factory

    def _wait_initial(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page, provider, started_ms, timeout_ms)
        return (
            1200,
            make_self_state(
                tank_id=1,
                x=100,
                y=100,
                team=2,
                rank=1,
                fuel=5000,
                leaderboard_position=1,
            ),
        )

    action_hooks.wait_for_initial_self_state = _wait_initial
    try:
        return probe.execute_larder_probe(
            max_attempts=max_attempts,
            max_extras=6,
            initial_sync_timeout_ms=10000,
        )
    finally:
        core_hooks.sync_playwright = original_sync_playwright


def test_execute_probe_fills_the_attempt_budget() -> None:
    """The loop ends by attempt count and books searches plus tallies."""
    probe = _ExecuteHarness(containers_to_serve=5)
    session = _run_execute_harness(probe, max_attempts=2)

    assert probe.phases == [
        "fuel",
        "enable",
        "search:6",
        "attempt:104",
        "search:5",
        "attempt:105",
        "restore:False",
        "read",
        "fuel",
        "quit",
    ]
    assert len(session["attempts"]) == 2
    assert session["search_scans"] == 2
    assert session["search_hops"] == 2
    assert session["own_tile_successes"] == 0
    assert session["own_tile_failures"] == 2
    assert session["adjacent_successes"] == 2
    assert session["extras_before"] == 20
    assert session["extras_after"] == 18
    assert session["toggles_sent"] == 2
    assert session["fuel_before"] == 5000
    assert session["fuel_after"] == 5000
    assert session["capture_session_path"] == ""


def test_execute_probe_stops_when_no_equipment_is_found() -> None:
    """A dry search breaks the loop and still restores the slot state."""
    probe = _ExecuteHarness(containers_to_serve=1)
    session = _run_execute_harness(probe, max_attempts=3)

    assert probe.phases == [
        "fuel",
        "enable",
        "search:6",
        "attempt:104",
        "search:5",
        "restore:False",
        "read",
        "fuel",
        "quit",
    ]
    assert len(session["attempts"]) == 1
    assert session["search_scans"] == 2
    assert session["search_hops"] == 3
    assert session["max_attempts"] == 3
    assert session["max_extras"] == 6
