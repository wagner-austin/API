"""Tests for the live fuel action probe harness."""

from __future__ import annotations

import types
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.json_utils import JSONObject, JSONValue, load_json_str, narrow_json_to_dict
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BrowserContextProtocol,
    BrowserProtocol,
    BrowserTypeProtocol,
    BufferedMessageSourceProtocol,
    CDPSessionProtocol,
    KeyboardProtocol,
    PageProtocol,
    PlaywrightProtocol,
    ResponseProtocol,
    SyncPlaywrightContextManagerProtocol,
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.fuel_probe import (
    FuelProbe,
    FuelProbeError,
    _find_visible_fuel_target,
    _get_completed_pickup_outcome,
    _wait_for_pickup_outcome,
    format_fuel_probe_summary,
    run_fuel_probe,
)
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
    decode_fuel_probe_session,
)
from tankpit_bot.action_lab.teleport import TeleportProbeError
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import (
    ContainerStateDict,
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    coord_key,
    make_container_state,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.types import CapturedMessage, decode_capture_session


class _FuelProbeModuleProtocol(Protocol):
    """Typed access to patchable fuel probe module globals."""

    _wait_for_teleport_outcome: _WaitForTeleportOutcomeProtocol
    _find_visible_fuel_target: Callable[[FuelProbe, bool], ContainerStateDict | None]
    _visible_fuel_requires_reposition: Callable[[FuelProbe, ContainerStateDict], bool]
    _find_visible_fuel_landing_tile: Callable[
        [FuelProbe, ContainerStateDict], tuple[int, int] | None
    ]
    _wait_for_pickup_outcome: _WaitForPickupOutcomeProtocol
    get_terrain_map: Callable[[], TerrainMapProtocol | None]
    FuelProbe: type[FuelProbe]


class _WaitForTeleportOutcomeProtocol(Protocol):
    """Callable protocol for teleport outcome waiting."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict: ...


class _WaitForPickupOutcomeProtocol(Protocol):
    """Callable protocol for pickup outcome waiting."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: FuelProbe,
        *,
        target_x: int,
        target_y: int,
        pickup_started_ms: int,
        fuel_before: int,
        timeout_ms: int,
    ) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]: ...


_fuel_module_import = __import__("tankpit_bot.action_lab.fuel_probe", fromlist=["fuel_probe"])
fuel_probe_module: _FuelProbeModuleProtocol = _fuel_module_import


def _make_world(timestamp_ms: int, x: int, y: int, fuel: int) -> WorldStateDict:
    """Build a world with one self tank."""
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1,
            x=x,
            y=y,
            team=2,
            rank=1,
            fuel=fuel,
            leaderboard_position=1,
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=ViewportStateDict(left=x - 8, top=y - 8, width=16, height=16),
        scanned_viewports=world["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


class _Clock:
    """Mutable millisecond clock."""

    def __init__(self, start_ms: int) -> None:
        self._now_ms = start_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class _FakeKeyboard:
    def press(self, key: str, *, delay: float | None = None) -> None:
        _ = (key, delay)

    def type(self, text: str, *, delay: float | None = None) -> None:
        _ = (text, delay)


class _FakePage:
    """Minimal page fake that records waits."""

    url = "https://tankpit.com/play"

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self.waits: list[float] = []
        self._keyboard = _FakeKeyboard()
        self.on_wait: Callable[[float], None] | None = None

    @property
    def keyboard(self) -> KeyboardProtocol:
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        _ = (url, referer, timeout, wait_until)
        return None

    def wait_for_timeout(self, timeout: float) -> None:
        self.waits.append(timeout)
        self._clock.advance(int(timeout))
        if self.on_wait is not None:
            self.on_wait(timeout)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        _ = (expression, timeout)

    def close(
        self,
        *,
        reason: str | None = None,
        run_before_unload: bool | None = None,
    ) -> None:
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        _ = expression
        return None


class _Terrain:
    """Minimal terrain fake."""

    ROCK = "#"
    GROUND = "."
    WATER = "W"

    def __init__(self, passable: set[tuple[int, int]]) -> None:
        self._passable = passable

    def get_terrain(self, x: int, y: int) -> str:
        return self.GROUND if (x, y) in self._passable else self.WATER

    def is_passable(self, x: int, y: int) -> bool:
        return (x, y) in self._passable

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        rows: list[list[str]] = []
        left = center_x - (width // 2)
        top = center_y - (height // 2)
        for y in range(top, top + height):
            row: list[str] = []
            for x in range(left, left + width):
                row.append(self.get_terrain(x, y))
            rows.append(row)
        return rows


def _terrain(passable: set[tuple[int, int]]) -> TerrainMapProtocol:
    return _Terrain(passable)


def _build_wait_results(
    status: Literal[
        "picked_up_fuel",
        "no_fuel_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ],
    map_sync_result: int | None,
    radar_sync_result: int | None,
) -> list[int | None]:
    """Build ordered world-sync results for one probe scenario."""
    wait_results = [map_sync_result, radar_sync_result]
    if status == "reposition_map_sync_timeout":
        wait_results.append(None)
    elif status == "reposition_teleport_timeout":
        wait_results.append(1800)
    return wait_results


def _make_world_sync_waiter(
    wait_results: list[int | None],
) -> Callable[
    [
        action_session.WaitPageProtocol,
        action_session.BufferedWorldStateProviderProtocol,
        int,
        int,
    ],
    int | None,
]:
    """Return a deterministic world-sync waiter callback."""

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return wait_results.pop(0)

    return _wait_for_world_sync


def _make_teleport_outcome_callback(
    teleport_status: Literal["landed_exact", "teleport_timeout", "reposition_teleport_timeout"]
    | None,
) -> _WaitForTeleportOutcomeProtocol:
    """Return a teleport outcome callback for one scenario."""

    def _teleport_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict:
        _ = (page, provider, timeout_ms)
        if teleport_status == "teleport_timeout":
            resolved_status: Literal[
                "landed_exact",
                "landed_offset",
                "map_sync_timeout",
                "teleport_timeout",
            ] = "teleport_timeout"
        elif teleport_status == "reposition_teleport_timeout":
            resolved_status = (
                "teleport_timeout"
                if target["label"].startswith("fuel_reposition_")
                else "landed_exact"
            )
        else:
            resolved_status = "landed_exact"
        return TeleportAttemptResultDict(
            target=target,
            status=resolved_status,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=640,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=resolved_status == "landed_exact",
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
        )

    return _teleport_outcome


def _make_find_target_callback(
    fuel_target: ContainerStateDict | None,
) -> Callable[[FuelProbe, bool], ContainerStateDict | None]:
    """Return a visible fuel selector callback."""

    def _find_target(
        current_probe: FuelProbe,
        allow_unreachable: bool,
    ) -> ContainerStateDict | None:
        _ = (current_probe, allow_unreachable)
        if fuel_target is not None:
            target_key = coord_key(fuel_target["x"], fuel_target["y"])
            current_probe.get_world_state()["containers"][target_key] = fuel_target
        return fuel_target

    return _find_target


def _make_requires_reposition_callback(
    status: Literal[
        "picked_up_fuel",
        "no_fuel_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ],
) -> Callable[[FuelProbe, ContainerStateDict], bool]:
    """Return whether a scenario should trigger blocked-fuel reposition."""

    def _requires_reposition(
        current_probe: FuelProbe,
        current_target: ContainerStateDict,
    ) -> bool:
        _ = (current_probe, current_target)
        return status in {"reposition_map_sync_timeout", "reposition_teleport_timeout"}

    return _requires_reposition


def _make_pickup_outcome_callback(
    pickup_status: Literal["picked_up_fuel", "pickup_timeout"] | None,
) -> _WaitForPickupOutcomeProtocol:
    """Return a pickup outcome callback for one scenario."""

    def _pickup_outcome(
        page: action_session.WaitPageProtocol,
        probe: FuelProbe,
        *,
        target_x: int,
        target_y: int,
        pickup_started_ms: int,
        fuel_before: int,
        timeout_ms: int,
    ) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
        _ = (
            page,
            probe,
            target_x,
            target_y,
            pickup_started_ms,
            fuel_before,
            timeout_ms,
        )
        if pickup_status is not None:
            return (pickup_status, 2000, 900)
        return ("pickup_timeout", 2000, 640)

    return _pickup_outcome


class _ProbeHarness(FuelProbe):
    """Fuel probe subclass with controllable world state."""

    def __init__(self, clock: _Clock) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=False)
        self._world_state = _make_world(1000, 100, 100, 700)
        self._fake_page = _FakePage(clock)
        self._messages = []
        self.map_open_result = True
        self.teleport_result = True
        self.radar_result = True
        self.move_result = True
        self.move_calls: list[tuple[int, int]] = []

    def _require_page(self) -> PageProtocol:
        return self._fake_page

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._world_state["self_state"]

    def open_map(self) -> bool:
        return self.map_open_result

    def teleport_to(self, x: int, y: int) -> bool:
        _ = (x, y)
        return self.teleport_result

    def use_radar(self) -> bool:
        return self.radar_result

    def move_to(self, x: int, y: int) -> bool:
        self.move_calls.append((x, y))
        return self.move_result


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore patched hooks after each test."""
    original_get_time = action_hooks.get_current_time_ms
    original_drain = action_hooks.drain_buffered_messages
    original_wait_sync = action_session.wait_for_world_sync
    original_get_terrain_map = fuel_probe_module.get_terrain_map
    original_wait_outcome = fuel_probe_module._wait_for_teleport_outcome
    original_find_visible = fuel_probe_module._find_visible_fuel_target
    original_requires_reposition = fuel_probe_module._visible_fuel_requires_reposition
    original_find_landing = fuel_probe_module._find_visible_fuel_landing_tile
    original_wait_pickup = fuel_probe_module._wait_for_pickup_outcome
    original_probe_class = fuel_probe_module.FuelProbe
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.drain_buffered_messages = original_drain
    action_session.wait_for_world_sync = original_wait_sync
    fuel_probe_module.get_terrain_map = original_get_terrain_map
    fuel_probe_module._wait_for_teleport_outcome = original_wait_outcome
    fuel_probe_module._find_visible_fuel_target = original_find_visible
    fuel_probe_module._visible_fuel_requires_reposition = original_requires_reposition
    fuel_probe_module._find_visible_fuel_landing_tile = original_find_landing
    fuel_probe_module._wait_for_pickup_outcome = original_wait_pickup
    fuel_probe_module.FuelProbe = original_probe_class


def test_find_visible_fuel_target_returns_best_visible_container() -> None:
    """Fuel target selection chooses the visible high-volume fuel container."""
    probe = _ProbeHarness(_Clock(1000))
    world = probe.get_world_state()
    world["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=world["timestamp_ms"],
    )
    world["containers"][coord_key(102, 100)] = make_container_state(
        102,
        100,
        True,
        500,
        timestamp_ms=world["timestamp_ms"],
    )
    fuel_probe_module.get_terrain_map = lambda: _terrain({(100, 100), (101, 100), (102, 100)})

    fuel_target = _find_visible_fuel_target(probe)

    assert fuel_target == world["containers"][coord_key(102, 100)]


def test_find_visible_fuel_target_requires_terrain_and_self_state() -> None:
    """Fuel target selection raises when required state is missing."""
    probe = _ProbeHarness(_Clock(1000))
    fuel_probe_module.get_terrain_map = lambda: None
    with pytest.raises(FuelProbeError, match="terrain map is unavailable"):
        _find_visible_fuel_target(probe)

    fuel_probe_module.get_terrain_map = lambda: _terrain({(100, 100)})
    probe._world_state["self_state"] = None
    with pytest.raises(FuelProbeError, match="self state is unavailable"):
        _find_visible_fuel_target(probe)


def test_wait_for_pickup_outcome_detects_fuel_gain_and_disappearance() -> None:
    """Pickup wait succeeds on fuel gain or container disappearance."""
    clock = _Clock(1000)
    probe = _ProbeHarness(clock)
    page = probe._fake_page
    worlds = [_make_world(1000, 100, 100, 300), _make_world(1100, 100, 100, 450)]
    for world in worlds:
        world["containers"][coord_key(101, 100)] = make_container_state(
            101,
            100,
            True,
            300,
            timestamp_ms=world["timestamp_ms"],
        )
    probe._world_state = worlds[0]

    def _advance(timeout: float) -> None:
        _ = timeout
        if len(worlds) > 1:
            worlds.pop(0)
        probe._world_state = worlds[0]

    page.on_wait = _advance
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda provider: 0

    status, completed_ms, fuel_after = _wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=300,
        timeout_ms=1000,
    )

    assert (status, completed_ms, fuel_after) == ("picked_up_fuel", 1100, 450)

    probe._world_state = _make_world(1000, 100, 100, 700)
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )

    def _remove_container(provider: BufferedMessageSourceProtocol) -> int:
        _ = provider
        probe.get_world_state()["containers"].pop(coord_key(101, 100), None)
        return 1

    action_hooks.drain_buffered_messages = _remove_container

    disappeared = _wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=700,
        timeout_ms=1000,
    )

    assert disappeared == ("picked_up_fuel", 1100, 700)


def test_wait_for_pickup_outcome_times_out_and_handles_missing_self_state() -> None:
    """Pickup wait handles timeout and missing-self-state failures."""
    clock = _Clock(1000)
    probe = _ProbeHarness(clock)
    page = probe._fake_page
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda provider: 0

    timed_out = _wait_for_pickup_outcome(
        page,
        probe,
        target_x=101,
        target_y=100,
        pickup_started_ms=1000,
        fuel_before=700,
        timeout_ms=150,
    )

    assert timed_out == ("pickup_timeout", 1200, 700)

    def _clear_self(provider: BufferedMessageSourceProtocol) -> int:
        _ = provider
        probe.get_world_state()["self_state"] = None
        return 1

    probe = _ProbeHarness(clock)
    action_hooks.drain_buffered_messages = _clear_self
    with pytest.raises(FuelProbeError, match="self state disappeared while waiting"):
        _wait_for_pickup_outcome(
            page,
            probe,
            target_x=101,
            target_y=100,
            pickup_started_ms=1000,
            fuel_before=700,
            timeout_ms=1000,
        )

    probe = _ProbeHarness(clock)
    probe._world_state["self_state"] = None
    action_hooks.drain_buffered_messages = lambda provider: 0
    with pytest.raises(FuelProbeError, match="self state disappeared after fuel pickup timeout"):
        _wait_for_pickup_outcome(
            page,
            probe,
            target_x=101,
            target_y=100,
            pickup_started_ms=1000,
            fuel_before=700,
            timeout_ms=0,
        )


def test_get_completed_pickup_outcome_detects_pickup_and_missing_self_state() -> None:
    """Immediate pickup helper detects queued pickup events and validates self state."""
    action_hooks.get_current_time_ms = _Clock(1000)
    probe = _ProbeHarness(_Clock(1000))
    probe._world_state["containers"][coord_key(101, 100)] = make_container_state(
        101,
        100,
        True,
        300,
        timestamp_ms=1000,
    )
    probe._world_state["self_state"] = make_self_state(
        tank_id=1,
        x=100,
        y=100,
        team=2,
        rank=1,
        fuel=850,
        leaderboard_position=1,
    )

    completed = _get_completed_pickup_outcome(
        probe,
        target_x=101,
        target_y=100,
        fuel_before=700,
    )

    assert completed == ("picked_up_fuel", 1000, 850)

    probe = _ProbeHarness(_Clock(1000))
    probe._world_state["self_state"] = None

    with pytest.raises(FuelProbeError, match="self state disappeared while waiting"):
        _get_completed_pickup_outcome(
            probe,
            target_x=101,
            target_y=100,
            fuel_before=700,
        )


def test_format_fuel_probe_summary_counts_statuses() -> None:
    """Fuel probe summary reports all terminal status buckets."""
    session = FuelProbeSessionDict(
        session_id="fuel-session",
        start_timestamp_ms=100,
        end_timestamp_ms=200,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        target_pickups=3,
        max_attempts=6,
        capture_session_path="fuel_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 300,
            "intel_ready_timestamp_ms": 350,
            "initial_sync_started_ms": 400,
            "initial_world_timestamp_ms": 450,
            "command_ready_timestamp_ms": 460,
            "first_attempt_started_ms": 500,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 10,
            "command_ready_to_first_attempt_ms": 40,
        },
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
        attempts=[
            FuelProbeAttemptResultDict(
                target={"label": "a", "x": 1, "y": 1},
                status="picked_up_fuel",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=1300,
                radar_sync_timestamp_ms=1400,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=1500,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=700,
                landed_signal_received=True,
                landed_x=1,
                landed_y=1,
                fuel_target_x=2,
                fuel_target_y=1,
                fuel_target_volume=200,
                message_start_index=0,
                message_end_index=1,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "b", "x": 2, "y": 2},
                status="no_fuel_visible",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=1300,
                radar_sync_timestamp_ms=1400,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=None,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=500,
                landed_signal_received=True,
                landed_x=2,
                landed_y=2,
                fuel_target_x=None,
                fuel_target_y=None,
                fuel_target_volume=None,
                message_start_index=1,
                message_end_index=2,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "c", "x": 3, "y": 3},
                status="radar_timeout",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=1300,
                radar_sync_timestamp_ms=None,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=None,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=450,
                landed_signal_received=True,
                landed_x=3,
                landed_y=3,
                fuel_target_x=None,
                fuel_target_y=None,
                fuel_target_volume=None,
                message_start_index=2,
                message_end_index=3,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "d", "x": 4, "y": 4},
                status="map_sync_timeout",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=None,
                teleport_started_ms=None,
                radar_started_ms=None,
                radar_sync_timestamp_ms=None,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=None,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=500,
                landed_signal_received=False,
                landed_x=4,
                landed_y=4,
                fuel_target_x=None,
                fuel_target_y=None,
                fuel_target_volume=None,
                message_start_index=3,
                message_end_index=4,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "e", "x": 5, "y": 5},
                status="teleport_timeout",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=None,
                radar_sync_timestamp_ms=None,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=None,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=450,
                landed_signal_received=False,
                landed_x=5,
                landed_y=5,
                fuel_target_x=None,
                fuel_target_y=None,
                fuel_target_volume=None,
                message_start_index=4,
                message_end_index=5,
            ),
            FuelProbeAttemptResultDict(
                target={"label": "f", "x": 6, "y": 6},
                status="pickup_timeout",
                map_open_started_ms=1000,
                map_sync_timestamp_ms=1100,
                teleport_started_ms=1200,
                radar_started_ms=1300,
                radar_sync_timestamp_ms=1400,
                reposition_map_open_started_ms=None,
                reposition_map_sync_timestamp_ms=None,
                reposition_teleport_started_ms=None,
                pickup_started_ms=1500,
                completion_timestamp_ms=1600,
                fuel_before=500,
                fuel_after=450,
                landed_signal_received=True,
                landed_x=6,
                landed_y=6,
                fuel_target_x=7,
                fuel_target_y=6,
                fuel_target_volume=200,
                message_start_index=5,
                message_end_index=6,
            ),
        ],
    )

    summary = format_fuel_probe_summary(session)

    assert "attempts=6" in summary
    assert "target_pickups=3" in summary
    assert "picked_up_fuel=1" in summary
    assert "no_fuel_visible=1" in summary
    assert "radar_timeout=1" in summary
    assert "map_sync_timeout=1" in summary
    assert "teleport_timeout=1" in summary
    assert "pickup_timeout=1" in summary
    assert "session_to_initial_sync_ms=300" in summary
    assert "initial_sync_to_command_ready_ms=60" in summary


def _run_probe_single_target_scenario(
    *,
    status: Literal[
        "picked_up_fuel",
        "no_fuel_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ],
    map_sync_result: int | None,
    teleport_status: Literal["landed_exact", "teleport_timeout", "reposition_teleport_timeout"]
    | None,
    radar_sync_result: int | None,
    fuel_target: ContainerStateDict | None,
    pickup_status: Literal["picked_up_fuel", "pickup_timeout"] | None,
) -> FuelProbeAttemptResultDict:
    """Run one configured single-target probe scenario."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    wait_results = _build_wait_results(status, map_sync_result, radar_sync_result)

    def _find_landing(
        current_probe: FuelProbe,
        current_target: ContainerStateDict,
    ) -> tuple[int, int] | None:
        _ = (current_probe, current_target)
        return (102, 100)

    wait_for_world_sync = _make_world_sync_waiter(wait_results)

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        return wait_for_world_sync(page, provider, started_ms, timeout_ms)

    action_session.wait_for_world_sync = _wait_for_world_sync
    fuel_probe_module._wait_for_teleport_outcome = _make_teleport_outcome_callback(teleport_status)
    fuel_probe_module._find_visible_fuel_target = _make_find_target_callback(fuel_target)
    fuel_probe_module._visible_fuel_requires_reposition = _make_requires_reposition_callback(status)
    fuel_probe_module._find_visible_fuel_landing_tile = _find_landing
    fuel_probe_module._wait_for_pickup_outcome = _make_pickup_outcome_callback(pickup_status)

    result = probe._probe_single_fuel_target(
        target=target,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=250,
    )

    assert result["status"] == status
    assert result["target"] == target
    assert result["message_start_index"] == 0
    assert result["message_end_index"] == 0
    assert probe._fake_page.waits[-1] == 250.0
    if status in {"picked_up_fuel", "pickup_timeout"}:
        assert probe.move_calls == [(101, 100)]
    else:
        assert probe.move_calls == []
    return result


@pytest.mark.parametrize(
    (
        "status",
        "map_sync_result",
        "teleport_status",
        "radar_sync_result",
        "fuel_target",
        "pickup_status",
    ),
    [
        ("map_sync_timeout", None, None, None, None, None),
        ("teleport_timeout", 1200, "teleport_timeout", None, None, None),
        ("radar_timeout", 1200, "landed_exact", None, None, None),
        (
            "reposition_map_sync_timeout",
            1200,
            "landed_exact",
            1600,
            make_container_state(101, 100, True, 300),
            None,
        ),
        (
            "reposition_teleport_timeout",
            1200,
            "reposition_teleport_timeout",
            1600,
            make_container_state(101, 100, True, 300),
            None,
        ),
        ("no_fuel_visible", 1200, "landed_exact", 1600, None, None),
        (
            "picked_up_fuel",
            1200,
            "landed_exact",
            1600,
            make_container_state(101, 100, True, 300),
            "picked_up_fuel",
        ),
        (
            "pickup_timeout",
            1200,
            "landed_exact",
            1600,
            make_container_state(101, 100, True, 300),
            "pickup_timeout",
        ),
    ],
)
def test_probe_single_target_records_terminal_statuses(
    status: Literal[
        "picked_up_fuel",
        "no_fuel_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    ],
    map_sync_result: int | None,
    teleport_status: Literal["landed_exact", "teleport_timeout", "reposition_teleport_timeout"]
    | None,
    radar_sync_result: int | None,
    fuel_target: ContainerStateDict | None,
    pickup_status: Literal["picked_up_fuel", "pickup_timeout"] | None,
) -> None:
    """Single-target probe records all terminal outcomes."""
    _run_probe_single_target_scenario(
        status=status,
        map_sync_result=map_sync_result,
        teleport_status=teleport_status,
        radar_sync_result=radar_sync_result,
        fuel_target=fuel_target,
        pickup_status=pickup_status,
    )


def test_probe_single_target_rejects_impossible_map_sync_timeout_teleport_outcome() -> None:
    """Fuel probe rejects a teleport outcome that reports map-sync timeout after sync success."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    def _teleport_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
        )
        return TeleportAttemptResultDict(
            target=target,
            status="map_sync_timeout",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=200,
            fuel_before=700,
            fuel_after=650,
            world_timestamp_before=1000,
            world_timestamp_after=1450,
            landed_signal_received=False,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
        )

    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome

    with pytest.raises(
        TeleportProbeError,
        match="teleport outcome reported impossible map_sync_timeout",
    ):
        probe._probe_single_fuel_target(
            target=target,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )


def test_probe_single_target_repositions_for_blocked_visible_fuel() -> None:
    """Single-target fuel probe can reposition to a blocked visible fuel container."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    wait_results = [1200, 1600, 1800]

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return wait_results.pop(0)

    action_session.wait_for_world_sync = _wait_for_world_sync

    def _teleport_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
        )
        if target["label"].startswith("fuel_reposition_"):
            landed_x = 102
            landed_y = 100
            fuel_after = 620
        else:
            landed_x = 124
            landed_y = 100
            fuel_after = 640
        return TeleportAttemptResultDict(
            target=target,
            status="landed_exact",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=landed_x,
            landed_y=landed_y,
            message_start_index=0,
            message_end_index=0,
        )

    def _find_target(
        current_probe: FuelProbe,
        allow_unreachable: bool,
    ) -> ContainerStateDict | None:
        _ = (current_probe, allow_unreachable)
        fuel_target = make_container_state(101, 100, True, 300)
        current_probe.get_world_state()["containers"][coord_key(101, 100)] = fuel_target
        return fuel_target

    def _requires_reposition(
        current_probe: FuelProbe,
        current_target: ContainerStateDict,
    ) -> bool:
        _ = (current_probe, current_target)
        return True

    def _find_landing(
        current_probe: FuelProbe,
        current_target: ContainerStateDict,
    ) -> tuple[int, int] | None:
        _ = (current_probe, current_target)
        return (102, 100)

    def _pickup_outcome(
        page: action_session.WaitPageProtocol,
        probe: FuelProbe,
        *,
        target_x: int,
        target_y: int,
        pickup_started_ms: int,
        fuel_before: int,
        timeout_ms: int,
    ) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
        _ = (
            page,
            probe,
            target_x,
            target_y,
            pickup_started_ms,
            fuel_before,
            timeout_ms,
        )
        return ("picked_up_fuel", 2000, 900)

    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome
    fuel_probe_module._find_visible_fuel_target = _find_target
    fuel_probe_module._visible_fuel_requires_reposition = _requires_reposition
    fuel_probe_module._find_visible_fuel_landing_tile = _find_landing
    fuel_probe_module._wait_for_pickup_outcome = _pickup_outcome

    result = probe._probe_single_fuel_target(
        target=target,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=0,
    )

    assert result["status"] == "picked_up_fuel"
    assert result["reposition_map_open_started_ms"] == 1000
    assert result["reposition_map_sync_timestamp_ms"] == 1800
    assert result["reposition_teleport_started_ms"] == 1000
    assert result["landed_x"] == 102
    assert result["landed_y"] == 100
    assert result["pickup_started_ms"] == 1000
    assert probe.move_calls == [(101, 100)]


def test_probe_single_target_skips_move_when_pickup_already_completed() -> None:
    """Single-target probe does not enqueue move after an immediate fuel pickup."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ProbeHarness(clock)
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)
    wait_results = [1200, 1600]

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return wait_results.pop(0)

    def _teleport_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
        )
        return TeleportAttemptResultDict(
            target=target,
            status="landed_exact",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=640,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
        )

    def _find_target(
        current_probe: FuelProbe,
        allow_unreachable: bool,
    ) -> ContainerStateDict | None:
        _ = (current_probe, allow_unreachable)
        fuel_target = make_container_state(101, 100, True, 300)
        current_probe.get_world_state()["containers"][coord_key(101, 100)] = fuel_target
        return fuel_target

    def _pickup_before_move(provider: BufferedMessageSourceProtocol) -> int:
        _ = provider
        probe.get_world_state()["self_state"] = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        probe.get_world_state()["containers"].pop(coord_key(101, 100), None)
        return 1

    action_session.wait_for_world_sync = _wait_for_world_sync
    fuel_probe_module._wait_for_teleport_outcome = _teleport_outcome
    fuel_probe_module._find_visible_fuel_target = _find_target
    fuel_probe_module._visible_fuel_requires_reposition = lambda probe, fuel_target: False
    action_hooks.drain_buffered_messages = _pickup_before_move

    result = probe._probe_single_fuel_target(
        target=target,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=0,
    )

    assert result["status"] == "picked_up_fuel"
    assert result["fuel_after"] == 900
    assert probe.move_calls == []


def test_finalize_attempt_delay_skips_wait_for_zero_delay() -> None:
    """Fuel probe does not wait when settle delay is disabled."""
    clock = _Clock(1000)
    probe = _ProbeHarness(clock)

    probe._finalize_attempt_delay(probe._fake_page, settle_delay_ms=0)

    assert probe._fake_page.waits == []


class _FakeCDPSession:
    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        _ = (method, params)
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        _ = (event, handler)

    def detach(self) -> None:
        return None


class _FakeContext:
    def __init__(self, page: _FakePage, cdp: _FakeCDPSession) -> None:
        self._page = page
        self._cdp = cdp

    def new_page(self) -> PageProtocol:
        return self._page

    def new_cdp_session(self, page: PageProtocol) -> CDPSessionProtocol:
        assert page is self._page
        return self._cdp

    def close(self, *, reason: str | None = None) -> None:
        _ = reason


class _FakeBrowser:
    def __init__(self, context: _FakeContext) -> None:
        self._context = context

    def new_context(self) -> BrowserContextProtocol:
        return self._context

    def close(self, *, reason: str | None = None) -> None:
        _ = reason


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser
        self.last_headless: bool | None = None

    def launch(
        self,
        *,
        headless: bool | None = None,
        slow_mo: float | None = None,
        timeout: float | None = None,
    ) -> BrowserProtocol:
        _ = (slow_mo, timeout)
        self.last_headless = headless
        return self._browser


class _FakePlaywright:
    def __init__(self, chromium: _FakeChromium) -> None:
        self.chromium: BrowserTypeProtocol = chromium

    def stop(self) -> None:
        pass


class _FakePlaywrightContextManager:
    def __init__(self, playwright: _FakePlaywright) -> None:
        self._playwright: PlaywrightProtocol = playwright

    def __enter__(self) -> PlaywrightProtocol:
        return self._playwright

    def start(self) -> PlaywrightProtocol:
        return self._playwright

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


class _FakePlaywrightFactory:
    def __init__(self, manager: SyncPlaywrightContextManagerProtocol) -> None:
        self._manager = manager

    def __call__(self) -> SyncPlaywrightContextManagerProtocol:
        return self._manager


class _ExecuteHarness(FuelProbe):
    """Fuel probe subclass that stubs browser/bootstrap internals."""

    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=False, prefer_account=True)
        self._world_state = _make_world(900, 100, 100, 700)
        self._messages = []
        self.cleanup_calls = 0
        self.results: list[FuelProbeAttemptResultDict] = []

    def _setup_console_listener(self, cdp: CDPSessionProtocol) -> None:
        _ = cdp

    def _setup_cdp_handlers(self, cdp: CDPSessionProtocol) -> None:
        _ = cdp

    def _navigate_and_login(
        self,
        page: PageProtocol,
        cdp: CDPSessionProtocol,
        *,
        tank_name_prefix: str = "TP",
        auto_join_room: bool = True,
    ) -> None:
        _ = (page, cdp, tank_name_prefix, auto_join_room)

    def _wait_for_game_ready(self, page: PageProtocol) -> None:
        _ = page

    def _gather_intel(self, page: PageProtocol, cdp: CDPSessionProtocol) -> None:
        _ = (page, cdp)
        self._magic = "fake-magic"

    def _cleanup(
        self,
        cdp: CDPSessionProtocol,
        page: PageProtocol,
        context: BrowserContextProtocol,
        browser: BrowserProtocol,
    ) -> None:
        _ = (cdp, page, context, browser)
        self.cleanup_calls += 1

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._world_state["self_state"]

    def _probe_single_fuel_target(
        self,
        *,
        target: TeleportTargetDict,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelProbeAttemptResultDict:
        _ = (
            target,
            map_sync_timeout_ms,
            teleport_timeout_ms,
            radar_timeout_ms,
            pickup_timeout_ms,
            settle_delay_ms,
        )
        result = self.results[0]
        if len(self.results) > 1:
            self.results = self.results[1:]
        return result


class _FakeFuelProbe(FuelProbe):
    def __init__(self, target_url: str, *, headless: bool, prefer_account: bool) -> None:
        super().__init__(target_url, headless=headless, prefer_account=prefer_account)

    def execute_probe(
        self,
        *,
        target_pickups: int,
        max_attempts: int,
        initial_sync_timeout_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelProbeSessionDict:
        return FuelProbeSessionDict(
            session_id="fuel-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self._target_url,
            spawn_x=100,
            spawn_y=100,
            target_pickups=target_pickups,
            max_attempts=max_attempts,
            capture_session_path="",
            initial_sync_timeout_ms=initial_sync_timeout_ms,
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
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            radar_timeout_ms=radar_timeout_ms,
            pickup_timeout_ms=pickup_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            attempts=[],
        )

    @property
    def messages(self) -> list[CapturedMessage]:
        return []

    @property
    def magic(self) -> str | None:
        return None


def test_probe_single_target_raises_when_dispatch_fails() -> None:
    """Single-target probe raises on command dispatch failures."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    target = TeleportTargetDict(label="fuel_ground_124_100", x=124, y=100)

    probe = _ProbeHarness(clock)
    probe.map_open_result = False
    with pytest.raises(FuelProbeError, match="map_open command dispatch failed"):
        probe._probe_single_fuel_target(
            target=target,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )

    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200
    probe = _ProbeHarness(clock)
    probe.teleport_result = False
    with pytest.raises(FuelProbeError, match="teleport command dispatch failed"):
        probe._probe_single_fuel_target(
            target=target,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )

    probe = _ProbeHarness(clock)
    fuel_probe_module._wait_for_teleport_outcome = (
        lambda page, provider, target, **kwargs: TeleportAttemptResultDict(
            target=target,
            status="landed_exact",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=200,
            fuel_before=700,
            fuel_after=650,
            world_timestamp_before=1000,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
        )
    )
    probe.radar_result = False
    with pytest.raises(FuelProbeError, match="radar command dispatch failed"):
        probe._probe_single_fuel_target(
            target=target,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )

    probe = _ProbeHarness(clock)

    def _find_target(
        current_probe: FuelProbe,
        allow_unreachable: bool,
    ) -> ContainerStateDict | None:
        _ = (current_probe, allow_unreachable)
        fuel_target = make_container_state(101, 100, True, 300)
        current_probe.get_world_state()["containers"][coord_key(101, 100)] = fuel_target
        return fuel_target

    fuel_probe_module._find_visible_fuel_target = _find_target
    probe.move_result = False
    with pytest.raises(
        FuelProbeError,
        match="move_to command dispatch failed during fuel collection",
    ):
        probe._probe_single_fuel_target(
            target=target,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=0,
        )


def test_execute_probe_raises_for_invalid_limits_and_missing_playwright() -> None:
    """Fuel probe execute validates pickup limits and Playwright presence."""
    probe = _ProbeHarness(_Clock(1000))
    with pytest.raises(ValueError, match="target_pickups must be positive"):
        probe.execute_probe(
            target_pickups=0,
            max_attempts=1,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )

    with pytest.raises(ValueError, match="max_attempts must be positive"):
        probe.execute_probe(
            target_pickups=1,
            max_attempts=0,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )

    with pytest.raises(ValueError, match="max_attempts must be at least target_pickups"):
        probe.execute_probe(
            target_pickups=2,
            max_attempts=1,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )

    original_sync_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            probe.execute_probe(
                target_pickups=1,
                max_attempts=1,
                initial_sync_timeout_ms=10000,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                radar_timeout_ms=3000,
                pickup_timeout_ms=3000,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_sync_playwright


def test_execute_probe_collects_attempts_and_requires_terrain() -> None:
    """Fuel probe execute collects attempts and rejects missing terrain."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    page = _FakePage(clock)
    cdp = _FakeCDPSession()
    chromium = _FakeChromium(_FakeBrowser(_FakeContext(page, cdp)))
    manager = _FakePlaywrightContextManager(_FakePlaywright(chromium))
    core_hooks.sync_playwright = _FakePlaywrightFactory(manager)
    action_session.wait_for_initial_self_state = lambda page, provider, started_ms, timeout_ms: (
        1200,
        make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
    )
    probe = _ExecuteHarness()
    probe.results = [
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_124_100", "x": 124, "y": 100},
            status="picked_up_fuel",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=1500,
            completion_timestamp_ms=1600,
            fuel_before=700,
            fuel_after=900,
            landed_signal_received=True,
            landed_x=124,
            landed_y=100,
            fuel_target_x=125,
            fuel_target_y=100,
            fuel_target_volume=300,
            message_start_index=0,
            message_end_index=1,
        )
    ]
    fuel_probe_module.get_terrain_map = lambda: _terrain(
        {
            (115, 99),
            (115, 100),
            (115, 101),
            (116, 99),
            (116, 100),
            (116, 101),
            (117, 99),
            (117, 100),
            (117, 101),
        }
    )

    session = probe.execute_probe(
        target_pickups=1,
        max_attempts=1,
        initial_sync_timeout_ms=10000,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 1
    assert session["spawn_x"] == 100
    assert session["target_pickups"] == 1
    assert session["startup_timing"]["initial_world_timestamp_ms"] == 1200
    assert probe.cleanup_calls == 1
    assert chromium.last_headless is False

    fuel_probe_module.get_terrain_map = lambda: None
    with pytest.raises(FuelProbeError, match="terrain map is unavailable"):
        probe.execute_probe(
            target_pickups=1,
            max_attempts=1,
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=3000,
            pickup_timeout_ms=3000,
            settle_delay_ms=500,
        )


def test_execute_probe_continues_after_pickup_until_target_pickups_reached() -> None:
    """Fuel probe execute keeps probing after a pickup until target pickups are met."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    page = _FakePage(clock)
    cdp = _FakeCDPSession()
    chromium = _FakeChromium(_FakeBrowser(_FakeContext(page, cdp)))
    manager = _FakePlaywrightContextManager(_FakePlaywright(chromium))
    core_hooks.sync_playwright = _FakePlaywrightFactory(manager)
    action_session.wait_for_initial_self_state = lambda page, provider, started_ms, timeout_ms: (
        1200,
        make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
    )
    probe = _ExecuteHarness()
    probe.results = [
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_116_100", "x": 116, "y": 100},
            status="picked_up_fuel",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=1500,
            completion_timestamp_ms=1600,
            fuel_before=700,
            fuel_after=850,
            landed_signal_received=True,
            landed_x=116,
            landed_y=100,
            fuel_target_x=117,
            fuel_target_y=100,
            fuel_target_volume=150,
            message_start_index=0,
            message_end_index=1,
        ),
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_117_100", "x": 117, "y": 100},
            status="picked_up_fuel",
            map_open_started_ms=1700,
            map_sync_timestamp_ms=1800,
            teleport_started_ms=1900,
            radar_started_ms=2000,
            radar_sync_timestamp_ms=2100,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=2200,
            completion_timestamp_ms=2300,
            fuel_before=850,
            fuel_after=1000,
            landed_signal_received=True,
            landed_x=117,
            landed_y=100,
            fuel_target_x=118,
            fuel_target_y=100,
            fuel_target_volume=150,
            message_start_index=2,
            message_end_index=3,
        ),
    ]
    fuel_probe_module.get_terrain_map = lambda: _terrain(
        {(x, y) for x in range(0, 201) for y in range(0, 201)}
    )

    session = probe.execute_probe(
        target_pickups=2,
        max_attempts=3,
        initial_sync_timeout_ms=10000,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 2
    assert [attempt["status"] for attempt in session["attempts"]] == [
        "picked_up_fuel",
        "picked_up_fuel",
    ]
    assert session["target_pickups"] == 2
    assert probe.cleanup_calls == 1


def test_execute_probe_continues_after_miss_until_pickup_succeeds() -> None:
    """Fuel probe execute keeps probing after a miss until a later pickup succeeds."""
    clock = _Clock(1000)
    action_hooks.get_current_time_ms = clock
    page = _FakePage(clock)
    cdp = _FakeCDPSession()
    chromium = _FakeChromium(_FakeBrowser(_FakeContext(page, cdp)))
    manager = _FakePlaywrightContextManager(_FakePlaywright(chromium))
    core_hooks.sync_playwright = _FakePlaywrightFactory(manager)
    action_session.wait_for_initial_self_state = lambda page, provider, started_ms, timeout_ms: (
        1200,
        make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
    )
    probe = _ExecuteHarness()
    probe.results = [
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_116_100", "x": 116, "y": 100},
            status="no_fuel_visible",
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1100,
            teleport_started_ms=1200,
            radar_started_ms=1300,
            radar_sync_timestamp_ms=1400,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=None,
            completion_timestamp_ms=1600,
            fuel_before=700,
            fuel_after=650,
            landed_signal_received=True,
            landed_x=116,
            landed_y=100,
            fuel_target_x=None,
            fuel_target_y=None,
            fuel_target_volume=None,
            message_start_index=0,
            message_end_index=1,
        ),
        FuelProbeAttemptResultDict(
            target={"label": "fuel_ground_117_100", "x": 117, "y": 100},
            status="picked_up_fuel",
            map_open_started_ms=1700,
            map_sync_timestamp_ms=1800,
            teleport_started_ms=1900,
            radar_started_ms=2000,
            radar_sync_timestamp_ms=2100,
            reposition_map_open_started_ms=None,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=2200,
            completion_timestamp_ms=2300,
            fuel_before=650,
            fuel_after=900,
            landed_signal_received=True,
            landed_x=117,
            landed_y=100,
            fuel_target_x=118,
            fuel_target_y=100,
            fuel_target_volume=250,
            message_start_index=2,
            message_end_index=3,
        ),
    ]
    fuel_probe_module.get_terrain_map = lambda: _terrain(
        {(x, y) for x in range(0, 201) for y in range(0, 201)}
    )

    session = probe.execute_probe(
        target_pickups=1,
        max_attempts=3,
        initial_sync_timeout_ms=10000,
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=3000,
        pickup_timeout_ms=3000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 2
    assert [attempt["status"] for attempt in session["attempts"]] == [
        "no_fuel_visible",
        "picked_up_fuel",
    ]
    assert session["target_pickups"] == 1
    assert probe.cleanup_calls == 1


def test_run_fuel_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    """Fuel probe runner writes both summary JSON and raw capture output."""
    original_probe_class = fuel_probe_module.FuelProbe
    fuel_probe_module.FuelProbe = _FakeFuelProbe
    try:
        session = run_fuel_probe(
            "https://tankpit.com/play",
            "fuel_probe.json",
            target_pickups=3,
            max_attempts=3,
        )
    finally:
        fuel_probe_module.FuelProbe = original_probe_class

    written = fake_fs.read_text(Path("fuel_probe.json"))
    decoded = decode_fuel_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("fuel_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))

    assert session == decoded
    assert session["capture_session_path"] == "fuel_probe.capture_session.json"
    assert capture_decoded["session_id"] == "fuel-session"
