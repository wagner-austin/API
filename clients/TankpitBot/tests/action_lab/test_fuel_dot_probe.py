"""Tests for the live fuel-dot verification probe."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    ClockAdvancingPage,
    ReplayClock,
    StubbedBootstrapMixin,
    StubSnapshotCDPSession,
    WorldStateOverrideMixin,
)
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import PageProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.fuel_dot_probe import (
    FuelDotProbe,
    format_fuel_dot_probe_summary,
    observe_dot_containers,
    run_fuel_dot_probe,
    select_next_dot,
)
from tankpit_bot.action_lab.fuel_dot_probe_types import (
    DotContainerObservationDict,
    FuelDotAttemptResultDict,
    FuelDotProbeSessionDict,
    decode_fuel_dot_probe_session,
)
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.teleport import TeleportProbeError
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import (
    ContainerStateDict,
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_container_state,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.types import CapturedMessage, decode_capture_session

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


class _WaitForTeleportOutcomeProtocol(Protocol):
    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict: ...


class _FuelDotModuleProtocol(Protocol):
    FuelDotProbe: type[FuelDotProbe]
    _wait_for_teleport_outcome: _WaitForTeleportOutcomeProtocol


fuel_dot_module: _FuelDotModuleProtocol = __import__(
    "tankpit_bot.action_lab.fuel_dot_probe",
    fromlist=["fuel_dot_probe"],
)


def _snapshot(timestamp_ms: int) -> PageClientSnapshotDict:
    return PageClientSnapshotDict(
        timestamp_ms=timestamp_ms,
        client_present=True,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=10,
        last_page_client_send_age_ms=20,
        last_bot_send_age_ms=30,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections={},
    )


def _container(x: int, y: int, *, is_fuel: bool, volume: int) -> ContainerStateDict:
    return make_container_state(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=volume,
        timestamp_ms=1000,
    )


def _make_world(
    timestamp_ms: int,
    x: int,
    y: int,
    fuel: int,
    *,
    map_fuel_dots: dict[str, int] | None = None,
    containers: dict[str, ContainerStateDict] | None = None,
    viewport: ViewportStateDict | None = None,
) -> WorldStateDict:
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
        containers=containers if containers is not None else world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=(
            viewport
            if viewport is not None
            else ViewportStateDict(left=0, top=0, width=16, height=16)
        ),
        scanned_viewports=world["scanned_viewports"],
        map_fuel_dots=map_fuel_dots if map_fuel_dots is not None else {},
        timestamp_ms=timestamp_ms,
    )


class _ProbeHarness(FuelDotProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=False)
        self._self_state: SelfStateDict | None = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        self._world_state = _make_world(1000, 100, 100, 900)
        self._fake_page = ClockAdvancingPage(ReplayClock(1000))
        self._cdp = StubSnapshotCDPSession()
        self.map_open_result = True
        self.teleport_result = True
        self.radar_result = True
        self.open_map_calls = 0
        self.radar_calls = 0
        self.teleport_calls: list[tuple[int, int]] = []

    def _require_page(self) -> PageProtocol:
        return self._fake_page

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._self_state

    def open_map(self) -> bool:
        self.open_map_calls += 1
        return self.map_open_result

    def use_radar(self) -> bool:
        self.radar_calls += 1
        return self.radar_result

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleport_calls.append((x, y))
        return self.teleport_result


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, FuelDotProbe):
    def __init__(self) -> None:
        FuelDotProbe.__init__(self, "https://tankpit.com/play", headless=False, prefer_account=True)
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.results: list[FuelDotAttemptResultDict | None] = []
        self.visited_seen: list[frozenset[str]] = []

    def _probe_single_dot_attempt(
        self,
        *,
        visited: frozenset[str],
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelDotAttemptResultDict | None:
        _ = (acquisition_timeout_ms, teleport_timeout_ms, radar_timeout_ms, settle_delay_ms)
        self.visited_seen.append(visited)
        return self.results[len(self.visited_seen) - 1]


class _FakeFuelDotProbe(FuelDotProbe):
    def __init__(self, target_url: str, *, headless: bool, prefer_account: bool) -> None:
        super().__init__(target_url, headless=headless, prefer_account=prefer_account)

    def execute_probe(
        self,
        *,
        max_dots: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelDotProbeSessionDict:
        return FuelDotProbeSessionDict(
            session_id="fuel-dot-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self._target_url,
            spawn_x=100,
            spawn_y=100,
            max_dots=max_dots,
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
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            radar_timeout_ms=radar_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            attempts=[],
        )

    @property
    def messages(self) -> list[CapturedMessage]:
        return []

    @property
    def magic(self) -> str | None:
        return None


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    original_get_time = action_hooks.get_current_time_ms
    original_wait_sync = action_session.wait_for_world_sync
    original_wait_radar = action_session.wait_for_radar_sync
    original_wait_initial = action_session.wait_for_initial_self_state
    original_check_radar = action_hooks.check_and_clear_radar_scan_complete
    original_wait_outcome = fuel_dot_module._wait_for_teleport_outcome
    original_probe_class = fuel_dot_module.FuelDotProbe
    original_sync_playwright = core_hooks.sync_playwright
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_session.wait_for_world_sync = original_wait_sync
    action_session.wait_for_radar_sync = original_wait_radar
    action_session.wait_for_initial_self_state = original_wait_initial
    action_hooks.check_and_clear_radar_scan_complete = original_check_radar
    fuel_dot_module._wait_for_teleport_outcome = original_wait_outcome
    fuel_dot_module.FuelDotProbe = original_probe_class
    core_hooks.sync_playwright = original_sync_playwright


# =============================================================================
# select_next_dot
# =============================================================================


def test_select_next_dot_returns_none_for_empty_atlas() -> None:
    """An empty atlas yields no verification target."""
    world = _make_world(1000, 100, 100, 900)
    assert select_next_dot(world, 100, 100, frozenset()) is None


def test_select_next_dot_picks_nearest() -> None:
    """The nearest dot beyond the degenerate radius wins."""
    world = _make_world(
        1000,
        100,
        100,
        900,
        map_fuel_dots={"140,100": 1000, "120,100": 1000, "110,100": 1000},
    )
    assert select_next_dot(world, 100, 100, frozenset()) == (110, 100)


def test_select_next_dot_skips_visited() -> None:
    """Visited dots are never re-selected."""
    world = _make_world(
        1000,
        100,
        100,
        900,
        map_fuel_dots={"110,100": 1000, "120,100": 1000},
    )
    assert select_next_dot(world, 100, 100, frozenset({"110,100"})) == (120, 100)


def test_select_next_dot_skips_degenerate_close_dots() -> None:
    """A dot on or beside the tank is not a teleport target."""
    world = _make_world(
        1000,
        100,
        100,
        900,
        map_fuel_dots={"101,100": 1000, "120,100": 1000},
    )
    assert select_next_dot(world, 100, 100, frozenset()) == (120, 100)


def test_select_next_dot_returns_none_when_only_degenerate_dots_remain() -> None:
    """All-degenerate atlases yield no target."""
    world = _make_world(1000, 100, 100, 900, map_fuel_dots={"100,101": 1000})
    assert select_next_dot(world, 100, 100, frozenset()) is None


# =============================================================================
# observe_dot_containers
# =============================================================================


def test_observe_dot_containers_reads_fuel_on_dot_and_viewport() -> None:
    """The dot-tile container and the visible fuel list are both read."""
    viewport = ViewportStateDict(left=112, top=104, width=16, height=16)
    containers = {
        "120,110": _container(120, 110, is_fuel=True, volume=750),
        "125,112": _container(125, 112, is_fuel=True, volume=300),
        "121,110": _container(121, 110, is_fuel=False, volume=0),
        "200,200": _container(200, 200, is_fuel=True, volume=999),
    }
    world = _make_world(1000, 118, 108, 900, containers=containers, viewport=viewport)

    on_dot, viewport_fuel = observe_dot_containers(world, 120, 110)

    assert on_dot == DotContainerObservationDict(x=120, y=110, is_fuel=True, volume=750)
    assert viewport_fuel == [
        DotContainerObservationDict(x=120, y=110, is_fuel=True, volume=750),
        DotContainerObservationDict(x=125, y=112, is_fuel=True, volume=300),
    ]


def test_observe_dot_containers_reports_equipment_and_empty() -> None:
    """Equipment on the dot is surfaced; an empty dot reads as None."""
    viewport = ViewportStateDict(left=112, top=104, width=16, height=16)
    containers = {"120,110": _container(120, 110, is_fuel=False, volume=0)}
    world = _make_world(1000, 118, 108, 900, containers=containers, viewport=viewport)

    on_dot, viewport_fuel = observe_dot_containers(world, 120, 110)
    assert on_dot == DotContainerObservationDict(x=120, y=110, is_fuel=False, volume=0)
    assert viewport_fuel == []

    empty_on_dot, _ = observe_dot_containers(world, 121, 111)
    assert empty_on_dot is None


# =============================================================================
# format_fuel_dot_probe_summary
# =============================================================================


def _attempt(
    status: Literal[
        "fuel_on_dot",
        "equipment_on_dot",
        "empty_dot",
        "acquisition_timeout",
        "teleport_timeout",
        "radar_timeout",
    ],
    *,
    container_on_dot: DotContainerObservationDict | None = None,
) -> FuelDotAttemptResultDict:
    return FuelDotAttemptResultDict(
        status=status,
        acquisition_started_ms=1000,
        acquisition_sync_timestamp_ms=1100,
        dots_in_atlas=650,
        dot_x=120,
        dot_y=110,
        dot_distance=30,
        teleport_started_ms=1200,
        radar_started_ms=1400,
        radar_sync_timestamp_ms=1500,
        completion_timestamp_ms=1600,
        fuel_before=900,
        fuel_after=720,
        landed_signal_received=True,
        landed_x=120,
        landed_y=110,
        container_on_dot=container_on_dot,
        viewport_fuel_containers=[],
        message_start_index=0,
        message_end_index=1,
        snapshot_before=_snapshot(1000),
        snapshot_after=_snapshot(1600),
    )


def test_format_summary_counts_every_status() -> None:
    """The summary line aggregates all attempt statuses and volumes."""
    session = _FakeFuelDotProbe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    ).execute_probe(
        max_dots=6,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=500,
    )
    session["attempts"] = [
        _attempt(
            "fuel_on_dot",
            container_on_dot=DotContainerObservationDict(
                x=120,
                y=110,
                is_fuel=True,
                volume=750,
            ),
        ),
        _attempt("equipment_on_dot"),
        _attempt("empty_dot"),
        _attempt("teleport_timeout"),
        _attempt("radar_timeout"),
    ]

    summary = format_fuel_dot_probe_summary(session)

    assert "dots_in_atlas=650" in summary
    assert "attempts=5" in summary
    assert "fuel_on_dot=1" in summary
    assert "equipment_on_dot=1" in summary
    assert "empty_dot=1" in summary
    assert "timeouts=2" in summary
    assert "fuel_volumes=750" in summary


def test_format_summary_skips_volume_for_fuel_attempt_without_observation() -> None:
    """A fuel_on_dot attempt with no observation contributes no volume."""
    session = _FakeFuelDotProbe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    ).execute_probe(
        max_dots=6,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=500,
    )
    session["attempts"] = [_attempt("fuel_on_dot", container_on_dot=None)]

    summary = format_fuel_dot_probe_summary(session)

    assert "fuel_on_dot=1" in summary
    assert "fuel_volumes=-" in summary


def test_format_summary_handles_empty_session() -> None:
    """A session with no attempts formats with zero counters."""
    session = _FakeFuelDotProbe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    ).execute_probe(
        max_dots=6,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=500,
    )

    summary = format_fuel_dot_probe_summary(session)

    assert "dots_in_atlas=0" in summary
    assert "attempts=0" in summary
    assert "fuel_volumes=-" in summary


# =============================================================================
# _probe_single_dot_attempt
# =============================================================================


def _landed_outcome(
    landed_x: int,
    landed_y: int,
) -> _WaitForTeleportOutcomeProtocol:
    def _outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (page, provider, message_start_index, timeout_ms, page_snapshots, capture_page_snapshot)
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="landed_exact",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=720,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=True,
            landed_x=landed_x,
            landed_y=landed_y,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    return _outcome


def _timeout_outcome() -> _WaitForTeleportOutcomeProtocol:
    def _outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (page, provider, message_start_index, timeout_ms, page_snapshots, capture_page_snapshot)
        return TeleportAttemptResultDict(
            target=target,
            teleport_cycle_id=teleport_cycle_id,
            status="teleport_timeout",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=1500,
            map_sync_elapsed_ms=200,
            teleport_elapsed_ms=300,
            fuel_before=fuel_before,
            fuel_after=900,
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=1450,
            landed_signal_received=False,
            landed_x=100,
            landed_y=100,
            message_start_index=0,
            message_end_index=0,
            page_snapshots=[],
        )

    return _outcome


def _dot_world(containers: dict[str, ContainerStateDict] | None = None) -> WorldStateDict:
    return _make_world(
        1000,
        100,
        100,
        900,
        map_fuel_dots={"120,110": 1000},
        containers=containers,
        viewport=ViewportStateDict(left=112, top=104, width=16, height=16),
    )


def test_attempt_returns_acquisition_timeout() -> None:
    """A map-sync timeout terminates the attempt before any teleport."""
    probe = _ProbeHarness()
    probe._world_state = _dot_world()
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: None

    result = probe._probe_single_dot_attempt(
        visited=frozenset(),
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=0,
    )

    if result is None:
        pytest.fail("expected attempt result")

    assert result["status"] == "acquisition_timeout"
    assert result["dot_x"] is None
    assert probe.teleport_calls == []


def test_attempt_returns_none_when_no_dot_remains() -> None:
    """An exhausted atlas ends the session instead of recording an attempt."""
    probe = _ProbeHarness()
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    result = probe._probe_single_dot_attempt(
        visited=frozenset(),
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=0,
    )

    assert result is None
    assert probe.teleport_calls == []


def test_attempt_raises_when_cdp_is_unavailable() -> None:
    """A missing CDP session is a hard probe error."""
    probe = _ProbeHarness()
    probe._cdp = None

    with pytest.raises(TeleportProbeError, match="cdp session is unavailable"):
        probe._probe_single_dot_attempt(
            visited=frozenset(),
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=5000,
            settle_delay_ms=0,
        )


def test_attempt_records_teleport_timeout() -> None:
    """A teleport timeout records the dot but never radars."""
    probe = _ProbeHarness()
    probe._world_state = _dot_world()
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200
    fuel_dot_module._wait_for_teleport_outcome = _timeout_outcome()

    result = probe._probe_single_dot_attempt(
        visited=frozenset(),
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=1,
    )

    if result is None:
        pytest.fail("expected attempt result")

    assert result["status"] == "teleport_timeout"
    assert result["dot_x"] == 120
    assert result["dot_y"] == 110
    assert result["dot_distance"] == 30
    assert result["radar_started_ms"] is None
    assert probe.radar_calls == 0


def test_attempt_raises_when_teleport_dispatch_fails() -> None:
    """A failed teleport dispatch raises instead of soft-failing."""
    probe = _ProbeHarness()
    probe._world_state = _dot_world()
    probe.teleport_result = False
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200

    with pytest.raises(TeleportProbeError, match="teleport command dispatch failed"):
        probe._probe_single_dot_attempt(
            visited=frozenset(),
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=5000,
            settle_delay_ms=0,
        )


def _arm_landed_radar_attempt(
    probe: _ProbeHarness,
    containers: dict[str, ContainerStateDict] | None,
    radar_sync: int | None,
) -> None:
    probe._world_state = _dot_world(containers)
    action_session.wait_for_world_sync = lambda page, provider, started_ms, timeout_ms: 1200
    action_session.wait_for_radar_sync = lambda page, provider, started_ms, timeout_ms: radar_sync
    action_hooks.check_and_clear_radar_scan_complete = lambda: False
    fuel_dot_module._wait_for_teleport_outcome = _landed_outcome(120, 110)


def test_attempt_records_fuel_on_dot() -> None:
    """A fuel container exactly on the dot tile verifies the atlas."""
    probe = _ProbeHarness()
    _arm_landed_radar_attempt(
        probe,
        {
            "120,110": _container(120, 110, is_fuel=True, volume=750),
            "125,112": _container(125, 112, is_fuel=True, volume=300),
        },
        radar_sync=2200,
    )

    result = probe._probe_single_dot_attempt(
        visited=frozenset(),
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=1,
    )

    if result is None:
        pytest.fail("expected attempt result")

    assert result["status"] == "fuel_on_dot"
    assert result["container_on_dot"] == DotContainerObservationDict(
        x=120,
        y=110,
        is_fuel=True,
        volume=750,
    )
    assert len(result["viewport_fuel_containers"]) == 2
    assert result["landed_x"] == 120
    assert result["landed_y"] == 110
    assert probe.radar_calls == 1
    assert probe.teleport_calls == [(120, 110)]


def test_attempt_records_empty_dot() -> None:
    """No container on the dot tile records a refuted dot."""
    probe = _ProbeHarness()
    _arm_landed_radar_attempt(probe, None, radar_sync=2200)

    result = probe._probe_single_dot_attempt(
        visited=frozenset(),
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=0,
    )

    if result is None:
        pytest.fail("expected attempt result")

    assert result["status"] == "empty_dot"
    assert result["container_on_dot"] is None


def test_attempt_records_equipment_on_dot() -> None:
    """An equipment container on the dot tile is its own outcome."""
    probe = _ProbeHarness()
    _arm_landed_radar_attempt(
        probe,
        {"120,110": _container(120, 110, is_fuel=False, volume=0)},
        radar_sync=2200,
    )

    result = probe._probe_single_dot_attempt(
        visited=frozenset(),
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=0,
    )

    if result is None:
        pytest.fail("expected attempt result")

    assert result["status"] == "equipment_on_dot"


def test_attempt_records_radar_timeout() -> None:
    """A radar-sync timeout is recorded as its own terminal status."""
    probe = _ProbeHarness()
    _arm_landed_radar_attempt(probe, None, radar_sync=None)

    result = probe._probe_single_dot_attempt(
        visited=frozenset(),
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=0,
    )

    if result is None:
        pytest.fail("expected attempt result")

    assert result["status"] == "radar_timeout"
    assert result["radar_sync_timestamp_ms"] is None


def test_attempt_raises_when_radar_dispatch_fails() -> None:
    """A failed radar dispatch raises instead of soft-failing."""
    probe = _ProbeHarness()
    probe.radar_result = False
    _arm_landed_radar_attempt(probe, None, radar_sync=2200)

    with pytest.raises(TeleportProbeError, match="radar command dispatch failed"):
        probe._probe_single_dot_attempt(
            visited=frozenset(),
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=5000,
            settle_delay_ms=0,
        )


# =============================================================================
# execute_probe
# =============================================================================


def test_execute_probe_raises_for_invalid_max_dots() -> None:
    probe = _ProbeHarness()

    with pytest.raises(ValueError, match="max_dots must be positive"):
        probe.execute_probe(
            max_dots=0,
            initial_sync_timeout_ms=10000,
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=5000,
            settle_delay_ms=500,
        )


def test_execute_probe_raises_when_playwright_is_missing() -> None:
    probe = _ProbeHarness()
    core_hooks.sync_playwright = None

    with pytest.raises(PlaywrightNotInstalledError):
        probe.execute_probe(
            max_dots=1,
            initial_sync_timeout_ms=10000,
            acquisition_timeout_ms=3000,
            teleport_timeout_ms=10000,
            radar_timeout_ms=5000,
            settle_delay_ms=500,
        )


def test_execute_probe_collects_attempts_and_tracks_visited() -> None:
    """Attempts accumulate, visited dots are excluded, and None ends the loop."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    first = _attempt(
        "fuel_on_dot",
        container_on_dot=DotContainerObservationDict(x=120, y=110, is_fuel=True, volume=750),
    )
    probe.results = [first, None]

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
                fuel=900,
                leaderboard_position=1,
            ),
        )

    action_session.wait_for_initial_self_state = _wait_initial

    session = probe.execute_probe(
        max_dots=3,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 1
    assert probe.visited_seen == [frozenset(), frozenset({"120,110"})]
    assert session["max_dots"] == 3
    assert session["startup_timing"]["first_attempt_started_ms"] == 1000
    assert probe.cleanup_calls == 1


def test_execute_probe_runs_to_max_dots_without_visiting_unselected_dots() -> None:
    """The loop runs to its cap and skips visited-tracking for dotless attempts."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    timeout_attempt = _attempt("acquisition_timeout")
    timeout_attempt["dot_x"] = None
    timeout_attempt["dot_y"] = None
    probe.results = [timeout_attempt, _attempt("empty_dot")]

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
                fuel=900,
                leaderboard_position=1,
            ),
        )

    action_session.wait_for_initial_self_state = _wait_initial

    session = probe.execute_probe(
        max_dots=2,
        initial_sync_timeout_ms=10000,
        acquisition_timeout_ms=3000,
        teleport_timeout_ms=10000,
        radar_timeout_ms=5000,
        settle_delay_ms=500,
    )

    assert len(session["attempts"]) == 2
    assert probe.visited_seen == [frozenset(), frozenset()]


def test_run_fuel_dot_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    fuel_dot_module.FuelDotProbe = _FakeFuelDotProbe
    session = run_fuel_dot_probe(
        "https://tankpit.com/play",
        "fuel_dot_probe.json",
        max_dots=4,
    )

    written = fake_fs.read_text(Path("fuel_dot_probe.json"))
    decoded = decode_fuel_dot_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("fuel_dot_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))

    assert session == decoded
    assert session["capture_session_path"] == "fuel_dot_probe.capture_session.json"
    assert session["max_dots"] == 4
    assert capture_decoded["session_id"] == "fuel-dot-session"
