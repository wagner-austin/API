"""Shared builders and probe doubles for the larder-probe tests."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
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
)
from tankpit_bot.inventory import (
    InventoryItem,
    InventoryState,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
)
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
    def _drain(provider: BufferedMessageSourceProtocol, ws: WorldService) -> int:
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
        self.world.inventory_state = _slots(self.slot_count)
        self.world.terrain_map = InMemoryTerrainMap()

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
            self.world.inventory_state = _slots(self.slot_count)
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
