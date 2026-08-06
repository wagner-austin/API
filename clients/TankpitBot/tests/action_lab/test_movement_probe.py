"""Tests for live movement probe helpers."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal

import pytest
from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    ClockAdvancingPage,
    ReplayClock,
    StubbedBootstrapMixin,
)
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    BufferedMessageSourceProtocol,
    CDPSessionProtocol,
    PageProtocol,
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import movement_probe as movement_probe_module
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.movement_probe import (
    MovementOutcomeProbeProtocol,
    MovementProbe,
    MovementProbeError,
    _build_probe_targets,
    _create_movement_probe,
    _find_first_sent_label_timestamp,
    _get_probe_terrain_map,
    _require_positive,
    _wait_for_move_outcome,
    format_movement_probe_summary,
    run_movement_probe,
)
from tankpit_bot.action_lab.movement_probe_types import (
    MovementProbeAttemptResultDict,
    MovementProbeSessionDict,
    decode_movement_probe_session,
)
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state, make_self_state
from tankpit_bot.types import CapturedMessage, decode_capture_session

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


class _SequencedWorld:
    def __init__(self, worlds: list[WorldStateDict]) -> None:
        self._worlds = worlds
        self._index = 0

    def current(self) -> WorldStateDict:
        return self._worlds[self._index]

    def advance(self) -> None:
        if self._index + 1 < len(self._worlds):
            self._index += 1


class _MoveWaitProbe:
    def __init__(self, worlds: _SequencedWorld) -> None:
        self._worlds = worlds
        self._cdp_message_buffer: list[str] = []

    def _update_state_from_world(self) -> None:
        return None

    def get_world_state(self) -> WorldStateDict:
        return self._worlds.current()

    def get_self_state(self) -> SelfStateDict | None:
        return self._worlds.current()["self_state"]


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
        viewport=world["viewport"],
        scanned_tiles=world["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def _make_snapshot(timestamp_ms: int) -> PageClientSnapshotDict:
    """Build a sample page-client snapshot used by movement attempt fakes."""
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


def _make_attempt(
    status: Literal["arrived_exact", "move_timeout"],
) -> MovementProbeAttemptResultDict:
    return MovementProbeAttemptResultDict(
        target=TeleportTargetDict(label=status, x=120, y=121),
        status=status,
        move_started_ms=1000,
        map_open_requested_ms=1200,
        map_open_message_timestamp_ms=1250,
        completion_timestamp_ms=1800,
        move_elapsed_ms=800,
        fuel_before=900,
        fuel_after=890,
        world_timestamp_before=990,
        world_timestamp_after=1790,
        settled_x=120,
        settled_y=121,
        message_start_index=10,
        message_end_index=18,
        snapshot_before=_make_snapshot(1000),
        snapshot_after=_make_snapshot(1800),
    )


@pytest.fixture(autouse=True)
def _restore_action_hooks() -> Generator[None, None, None]:
    original_get_time = action_hooks.get_current_time_ms
    original_drain = action_hooks.drain_buffered_messages
    original_sync_playwright = core_hooks.sync_playwright
    original_wait_for_initial_self_state = action_hooks.wait_for_initial_self_state
    original_advance_startup_state = action_hooks.advance_startup_state
    original_get_terrain_map = movement_probe_module._get_probe_terrain_map
    original_build_targets = movement_probe_module._build_probe_targets
    original_wait_for_move_outcome = movement_probe_module._wait_for_move_outcome
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.drain_buffered_messages = original_drain
    core_hooks.sync_playwright = original_sync_playwright
    action_hooks.wait_for_initial_self_state = original_wait_for_initial_self_state
    action_hooks.advance_startup_state = original_advance_startup_state
    movement_probe_module._get_probe_terrain_map = original_get_terrain_map
    movement_probe_module._build_probe_targets = original_build_targets
    movement_probe_module._wait_for_move_outcome = original_wait_for_move_outcome


def test_find_first_sent_label_timestamp_returns_first_matching_bot_send() -> None:
    messages = [
        CapturedMessage(
            timestamp_ms=10,
            direction="sent",
            payload="a",
            ws_url="wss://x",
            sent_origin="page_client",
        ),
        CapturedMessage(timestamp_ms=20, direction="received", payload="b", ws_url="wss://x"),
        CapturedMessage(
            timestamp_ms=30,
            direction="sent",
            payload="c",
            ws_url="wss://x",
            sent_origin="bot_injected",
            sent_label="map_open",
        ),
        CapturedMessage(
            timestamp_ms=40,
            direction="sent",
            payload="d",
            ws_url="wss://x",
            sent_origin="bot_injected",
            sent_label="map_open",
        ),
    ]
    assert _find_first_sent_label_timestamp(messages, start_index=0, label="map_open") == 30
    assert _find_first_sent_label_timestamp(messages, start_index=3, label="map_open") == 40
    assert _find_first_sent_label_timestamp(messages, start_index=0, label="move") is None


def test_require_positive_returns_value() -> None:
    assert _require_positive(5, "max_targets") == 5


def test_create_movement_probe_returns_concrete_probe() -> None:
    probe = _create_movement_probe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    )
    assert type(probe) is MovementProbe
    assert probe._target_url == "https://tankpit.com/play"
    assert probe._headless is True


def test_get_probe_terrain_map_defaults_to_none_without_loaded_map() -> None:
    assert _get_probe_terrain_map() is None


def test_build_probe_targets_uses_real_target_builder() -> None:
    targets = _build_probe_targets(
        100,
        104,
        _TerrainMapStub(),
        max_targets=2,
    )
    assert len(targets) == 2
    assert targets[0]["label"].startswith("fuel_ground_")


def test_wait_for_move_outcome_returns_arrived_exact() -> None:
    clock = ReplayClock(1000)
    worlds = _SequencedWorld(
        [
            _make_world(1000, 100, 100, 900),
            _make_world(1100, 101, 100, 899),
            _make_world(1200, 120, 121, 890),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=worlds.advance)
    probe = _MoveWaitProbe(worlds)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    status, completion_ms, elapsed_ms, settled_x, settled_y = _wait_for_move_outcome(
        page,
        probe,
        target_x=120,
        target_y=121,
        move_started_ms=1000,
        timeout_ms=5000,
    )
    assert status == "arrived_exact"
    assert completion_ms == 1200
    assert elapsed_ms == 200
    assert (settled_x, settled_y) == (120, 121)


def test_wait_for_move_outcome_returns_timeout() -> None:
    clock = ReplayClock(1000)
    worlds = _SequencedWorld(
        [
            _make_world(1000, 100, 100, 900),
            _make_world(1100, 101, 100, 899),
            _make_world(1200, 101, 101, 898),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=worlds.advance)
    probe = _MoveWaitProbe(worlds)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    status, completion_ms, elapsed_ms, settled_x, settled_y = _wait_for_move_outcome(
        page,
        probe,
        target_x=120,
        target_y=121,
        move_started_ms=1000,
        timeout_ms=250,
    )
    assert status == "move_timeout"
    assert completion_ms == 1300
    assert elapsed_ms == 300
    assert (settled_x, settled_y) == (101, 101)


class _MissingSelfWaitProbe:
    def __init__(self, states: list[SelfStateDict | None]) -> None:
        self._states = states
        self._index = 0
        self._cdp_message_buffer: list[str] = []
        self._world = _make_world(1000, 100, 100, 900)

    def _update_state_from_world(self) -> None:
        if self._index + 1 < len(self._states):
            self._index += 1

    def get_world_state(self) -> WorldStateDict:
        return self._world

    def get_self_state(self) -> SelfStateDict | None:
        return self._states[self._index]


def test_wait_for_move_outcome_raises_when_self_state_disappears_mid_wait() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _MissingSelfWaitProbe([make_self_state(1, 100, 100, 2, 1, 900, 5), None])
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    with pytest.raises(MovementProbeError, match="disappeared while waiting for movement"):
        _wait_for_move_outcome(
            page,
            probe,
            target_x=120,
            target_y=121,
            move_started_ms=1000,
            timeout_ms=5000,
        )


def test_wait_for_move_outcome_raises_when_self_state_missing_after_timeout() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _MissingSelfWaitProbe([None])
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    with pytest.raises(MovementProbeError, match="disappeared after movement timeout"):
        _wait_for_move_outcome(
            page,
            probe,
            target_x=120,
            target_y=121,
            move_started_ms=1000,
            timeout_ms=0,
        )


def test_format_movement_probe_summary_counts_statuses() -> None:
    session = MovementProbeSessionDict(
        session_id="movement-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        max_targets=2,
        capture_session_path="movement_probe.capture_session.json",
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
        move_timeout_ms=5000,
        settle_delay_ms=500,
        queue_map_open_during_move=True,
        map_open_delay_ms=150,
        targets=[],
        attempts=[_make_attempt("arrived_exact"), _make_attempt("move_timeout")],
    )
    summary = format_movement_probe_summary(session)
    assert "arrived_exact=1" in summary
    assert "move_timeout=1" in summary


class _FakeMovementProbe(MovementProbe):
    def __init__(self, target_url: str, *, headless: bool, prefer_account: bool) -> None:
        super().__init__(target_url, headless=headless, prefer_account=prefer_account)

    def execute_probe(
        self,
        *,
        explicit_targets: list[TeleportTargetDict] | None,
        max_targets: int,
        initial_sync_timeout_ms: int,
        move_timeout_ms: int,
        queue_map_open_during_move: bool,
        map_open_delay_ms: int,
        settle_delay_ms: int,
    ) -> MovementProbeSessionDict:
        targets = (
            explicit_targets
            if explicit_targets is not None
            else [TeleportTargetDict(label="move_1", x=120, y=121)][:max_targets]
        )
        return MovementProbeSessionDict(
            session_id="fake-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self._target_url,
            spawn_x=100,
            spawn_y=100,
            max_targets=max_targets,
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
            move_timeout_ms=move_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            queue_map_open_during_move=queue_map_open_during_move,
            map_open_delay_ms=map_open_delay_ms,
            targets=targets,
            attempts=[],
        )

    @property
    def messages(self) -> list[CapturedMessage]:
        return []

    @property
    def magic(self) -> str | None:
        return None


def test_run_movement_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    original_factory = movement_probe_module._create_movement_probe
    movement_probe_module._create_movement_probe = (
        lambda target_url, *, headless, prefer_account: _FakeMovementProbe(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
        )
    )
    try:
        session = run_movement_probe(
            "https://tankpit.com/play",
            "movement_probe.json",
            explicit_targets=[TeleportTargetDict(label="move_1", x=120, y=121)],
            queue_map_open_during_move=True,
            map_open_delay_ms=150,
        )
    finally:
        movement_probe_module._create_movement_probe = original_factory

    written = fake_fs.read_text(Path("movement_probe.json"))
    decoded = decode_movement_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("movement_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))
    assert session == decoded
    assert session["capture_session_path"] == "movement_probe.capture_session.json"
    assert session["targets"] == [TeleportTargetDict(label="move_1", x=120, y=121)]
    assert capture_decoded["session_id"] == "fake-session"


class _ExecuteHarness(StubbedBootstrapMixin, MovementProbe):
    def __init__(self) -> None:
        MovementProbe.__init__(
            self, "https://tankpit.com/play", headless=False, prefer_account=True
        )
        self._init_bootstrap_stubs()


class _SnapshotFakeCDPSession:
    """CDP fake returning a fixed valid page-client snapshot payload.

    Used by single-target movement probe tests so the new in-loop
    ``capture_page_client_snapshot`` call has a deterministic CDP
    surface to drive without spinning up a real browser.
    """

    def __init__(self) -> None:
        """Initialise the fake with no recorded calls."""
        self.send_calls = 0

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Return a stable snapshot payload from ``Runtime.evaluate``."""
        _ = params
        if method == "Runtime.evaluate":
            self.send_calls += 1
            return {
                "result": {
                    "value": {
                        "timestamp_ms": 1000 + self.send_calls,
                        "client_present": True,
                        "map_visible": False,
                        "client_state": 1,
                        "client_busy": False,
                        "pending_actions": 0,
                        "heartbeat_age_ms": 5,
                        "last_page_client_send_age_ms": 10,
                        "last_bot_send_age_ms": 15,
                        "ws_ready_state": 1,
                        "current_send_label": None,
                        "sent_frame_meta_queue_length": 0,
                        "self_fields": {},
                        "world_fields": {},
                        "map_fields": {},
                        "world_collections": {},
                    }
                }
            }
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Ignore event subscription."""
        _ = (event, handler)

    def detach(self) -> None:
        """No-op detach."""
        return


class _SingleTargetHarness(MovementProbe):
    def __init__(
        self,
        page: PageProtocol,
        *,
        cdp: CDPSessionProtocol | None = None,
    ) -> None:
        super().__init__("https://tankpit.com/play", headless=False, prefer_account=True)
        self._page = page
        self._cdp = _SnapshotFakeCDPSession() if cdp is None else cdp
        self._messages = []
        self._world = _make_world(900, 100, 100, 900)
        self._self_state = self._world["self_state"]
        self.reset_calls = 0
        self.move_calls: list[tuple[int, int]] = []
        self.open_map_calls = 0
        self.move_result = True
        self.open_map_result = True

    def _require_page(self) -> PageProtocol:
        page = self._page
        if page is None:
            raise AssertionError("page should be set for test")
        return page

    def get_world_state(self) -> WorldStateDict:
        return self._world

    def _require_self_state(self) -> SelfStateDict:
        self_state = self._self_state
        if self_state is None:
            raise AssertionError("self state should be set for test")
        return self_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._self_state

    def _reset_probe_state_to_idle(self) -> None:
        self.reset_calls += 1

    def move_to(self, x: int, y: int) -> bool:
        self.move_calls.append((x, y))
        return self.move_result

    def open_map(self) -> bool:
        self.open_map_calls += 1
        self._messages.append(
            CapturedMessage(
                timestamp_ms=action_hooks.get_current_time_ms(),
                direction="sent",
                payload="map",
                ws_url="wss://x",
                sent_origin="bot_injected",
                sent_label="map_open",
            )
        )
        return self.open_map_result


class _ExecuteSuccessHarness(_ExecuteHarness):
    def __init__(
        self,
        *,
        attempts: list[MovementProbeAttemptResultDict],
        default_targets: list[TeleportTargetDict],
    ) -> None:
        super().__init__()
        self._attempts = attempts
        self._default_targets = default_targets
        self.probed_targets: list[TeleportTargetDict] = []

    def _build_default_targets(self, *, max_targets: int) -> list[TeleportTargetDict]:
        return self._default_targets[:max_targets]

    def _probe_single_movement_target(
        self,
        target: TeleportTargetDict,
        *,
        move_timeout_ms: int,
        queue_map_open_during_move: bool,
        map_open_delay_ms: int,
        settle_delay_ms: int,
    ) -> MovementProbeAttemptResultDict:
        _ = (
            move_timeout_ms,
            queue_map_open_during_move,
            map_open_delay_ms,
            settle_delay_ms,
        )
        self.probed_targets.append(target)
        return self._attempts[len(self.probed_targets) - 1]


class _TerrainHarness(MovementProbe):
    def __init__(self, self_state: SelfStateDict) -> None:
        super().__init__("https://tankpit.com/play", headless=False, prefer_account=True)
        self._fixed_self_state = self_state

    def _require_self_state(self) -> SelfStateDict:
        return self._fixed_self_state


class _TerrainMapStub:
    ROCK = "#"
    GROUND = "."
    WATER = "W"

    def get_terrain(self, x: int, y: int) -> str:
        _ = (x, y)
        return self.GROUND

    def is_passable(self, x: int, y: int) -> bool:
        _ = (x, y)
        return True

    def is_landing_legal(self, x: int, y: int) -> bool:
        _ = (x, y)
        return True

    def render_viewport(
        self,
        center_x: int,
        center_y: int,
        width: int = 16,
        height: int = 16,
    ) -> list[list[str]]:
        _ = (center_x, center_y, width, height)
        return [[self.GROUND]]


def test_execute_probe_rejects_non_positive_max_targets() -> None:
    probe = MovementProbe("https://tankpit.com/play", headless=False, prefer_account=True)
    with pytest.raises(ValueError, match="max_targets must be positive"):
        probe.execute_probe(
            explicit_targets=None,
            max_targets=0,
            initial_sync_timeout_ms=10000,
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=500,
        )


def test_execute_probe_raises_when_playwright_is_missing() -> None:
    probe = MovementProbe("https://tankpit.com/play", headless=False, prefer_account=True)
    original_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            probe.execute_probe(
                explicit_targets=[TeleportTargetDict(label="move_1", x=120, y=121)],
                max_targets=1,
                initial_sync_timeout_ms=10000,
                move_timeout_ms=5000,
                queue_map_open_during_move=False,
                map_open_delay_ms=0,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_playwright


def test_build_default_targets_raises_when_terrain_is_missing() -> None:
    probe = _TerrainHarness(make_self_state(1, 100, 104, 2, 1, 900, 5))
    movement_probe_module._get_probe_terrain_map = lambda: None
    with pytest.raises(MovementProbeError, match="terrain map is unavailable"):
        probe._build_default_targets(max_targets=1)


def test_build_default_targets_uses_spawn_position_and_limits() -> None:
    probe = _TerrainHarness(make_self_state(1, 100, 104, 2, 1, 900, 5))
    expected = [TeleportTargetDict(label="move_1", x=104, y=108)]
    captured: dict[str, int] = {}
    movement_probe_module._get_probe_terrain_map = lambda: _TerrainMapStub()

    def _fake_build_targets(
        origin_x: int,
        origin_y: int,
        terrain: TerrainMapProtocol,
        *,
        max_targets: int,
    ) -> list[TeleportTargetDict]:
        _ = terrain
        captured["x"] = origin_x
        captured["y"] = origin_y
        captured["max_targets"] = max_targets
        return expected

    movement_probe_module._build_probe_targets = _fake_build_targets
    assert probe._build_default_targets(max_targets=2) == expected
    assert captured == {
        "x": 100,
        "y": 104,
        "max_targets": 2,
    }


def test_probe_single_movement_target_records_queue_map_open() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    after_world = _make_world(1700, 120, 121, 880)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    def _fake_wait_for_move_outcome(
        page: action_session.WaitPageProtocol,
        probe: MovementOutcomeProbeProtocol,
        *,
        target_x: int,
        target_y: int,
        move_started_ms: int,
        timeout_ms: int,
    ) -> tuple[Literal["arrived_exact", "move_timeout"], int, int, int, int]:
        _ = (page, target_x, target_y, move_started_ms, timeout_ms)
        if not isinstance(probe, _SingleTargetHarness):
            raise AssertionError("expected single-target harness")
        probe._world = after_world
        probe._self_state = after_world["self_state"]
        return ("arrived_exact", 1600, 600, 120, 121)

    movement_probe_module._wait_for_move_outcome = _fake_wait_for_move_outcome
    result = probe._probe_single_movement_target(
        TeleportTargetDict(label="move_1", x=120, y=121),
        move_timeout_ms=5000,
        queue_map_open_during_move=True,
        map_open_delay_ms=150,
        settle_delay_ms=200,
    )
    assert result["status"] == "arrived_exact"
    assert result["map_open_requested_ms"] == 1150
    assert result["map_open_message_timestamp_ms"] == 1150
    assert result["fuel_before"] == 900
    assert result["fuel_after"] == 880
    assert result["world_timestamp_after"] == 1700
    assert result["message_start_index"] == 0
    assert result["message_end_index"] == 1
    assert probe.move_calls == [(120, 121)]
    assert probe.open_map_calls == 1
    assert probe.reset_calls == 2
    assert page.waits == [150.0, 200.0]


def test_probe_single_movement_target_raises_when_cdp_session_unavailable() -> None:
    """The attempt fails fast when no CDP session is attached.

    The movement probe captures a page-client snapshot before and after
    each attempt; if CDP is unavailable there is no live source to read
    from and the probe must not silently proceed.
    """
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    probe._cdp = None
    action_hooks.get_current_time_ms = clock
    with pytest.raises(MovementProbeError, match="cdp session is unavailable"):
        probe._probe_single_movement_target(
            TeleportTargetDict(label="move_1", x=120, y=121),
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )


def test_probe_single_movement_target_records_snapshots_before_and_after() -> None:
    """The attempt result carries both bracketing page-client snapshots."""
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    after_world = _make_world(1700, 120, 121, 880)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    def _fake_wait_for_move_outcome(
        page: action_session.WaitPageProtocol,
        probe: MovementOutcomeProbeProtocol,
        *,
        target_x: int,
        target_y: int,
        move_started_ms: int,
        timeout_ms: int,
    ) -> tuple[Literal["arrived_exact", "move_timeout"], int, int, int, int]:
        _ = (page, target_x, target_y, move_started_ms, timeout_ms)
        if not isinstance(probe, _SingleTargetHarness):
            raise AssertionError("expected single-target harness")
        probe._world = after_world
        probe._self_state = after_world["self_state"]
        return ("arrived_exact", 1600, 600, 120, 121)

    movement_probe_module._wait_for_move_outcome = _fake_wait_for_move_outcome
    result = probe._probe_single_movement_target(
        TeleportTargetDict(label="move_1", x=120, y=121),
        move_timeout_ms=5000,
        queue_map_open_during_move=False,
        map_open_delay_ms=0,
        settle_delay_ms=0,
    )

    assert result["snapshot_before"]["client_present"] is True
    assert result["snapshot_after"]["client_present"] is True
    assert result["snapshot_after"]["timestamp_ms"] > result["snapshot_before"]["timestamp_ms"]


def test_probe_single_movement_target_raises_on_move_dispatch_failure() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    probe.move_result = False
    action_hooks.get_current_time_ms = clock
    with pytest.raises(MovementProbeError, match="move command dispatch failed"):
        probe._probe_single_movement_target(
            TeleportTargetDict(label="move_1", x=120, y=121),
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )


def test_probe_single_movement_target_raises_on_map_open_dispatch_failure() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    probe.open_map_result = False
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    with pytest.raises(MovementProbeError, match="map_open command dispatch failed"):
        probe._probe_single_movement_target(
            TeleportTargetDict(label="move_1", x=120, y=121),
            move_timeout_ms=5000,
            queue_map_open_during_move=True,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )


class _MapAlreadyOpenCDPSession:
    """CDP fake whose ``Runtime.evaluate`` always reports ``map_visible=True``.

    Drives the short-circuit branch of
    ``MovementProbe._probe_single_movement_target`` where a stale open
    map -- inherited from a prior attempt -- causes the probe to skip
    the redundant ``CMD_MAP_OPEN`` dispatch.
    """

    def __init__(self) -> None:
        self.send_calls = 0

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        _ = params
        if method == "Runtime.evaluate":
            self.send_calls += 1
            return {
                "result": {
                    "value": {
                        "timestamp_ms": 1000 + self.send_calls,
                        "client_present": True,
                        "map_visible": True,
                        "client_state": 1,
                        "client_busy": False,
                        "pending_actions": 0,
                        "heartbeat_age_ms": 5,
                        "last_page_client_send_age_ms": 10,
                        "last_bot_send_age_ms": 15,
                        "ws_ready_state": 1,
                        "current_send_label": None,
                        "sent_frame_meta_queue_length": 0,
                        "self_fields": {},
                        "world_fields": {},
                        "map_fields": {},
                        "world_collections": {},
                    }
                }
            }
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        _ = (event, handler)

    def detach(self) -> None:
        return


def test_probe_single_movement_target_skips_queued_map_open_when_map_already_open() -> None:
    """Mid-move queued ``map_open`` short-circuits when the JS client shows the map.

    Mirrors the ``run_tracked_acquisition_phase`` short-circuit: the
    wire ``CMD_MAP_OPEN`` is one-way, and re-sending it against an
    already-open overlay is a server-side no-op. The probe records the
    skip via ``map_open_requested_ms=None`` and refrains from calling
    ``self.open_map()``.
    """
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page, cdp=_MapAlreadyOpenCDPSession())
    action_hooks.get_current_time_ms = clock
    drain_calls = {"count": 0}

    def _count_drain(source: BufferedMessageSourceProtocol) -> int:
        _ = source
        drain_calls["count"] += 1
        return 0

    action_hooks.drain_buffered_messages = _count_drain

    def _fake_wait_for_move_outcome(
        page: action_session.WaitPageProtocol,
        probe: MovementOutcomeProbeProtocol,
        *,
        target_x: int,
        target_y: int,
        move_started_ms: int,
        timeout_ms: int,
    ) -> tuple[Literal["arrived_exact", "move_timeout"], int, int, int, int]:
        _ = (page, probe, target_x, target_y, move_started_ms, timeout_ms)
        return ("arrived_exact", 1800, 800, target_x, target_y)

    movement_probe_module._wait_for_move_outcome = _fake_wait_for_move_outcome

    result = probe._probe_single_movement_target(
        TeleportTargetDict(label="move_1", x=120, y=121),
        move_timeout_ms=5000,
        queue_map_open_during_move=True,
        map_open_delay_ms=150,
        settle_delay_ms=0,
    )

    assert probe.open_map_calls == 0
    assert result["map_open_requested_ms"] is None
    assert result["map_open_message_timestamp_ms"] is None
    assert result["snapshot_before"]["map_visible"] is True
    assert result["snapshot_after"]["map_visible"] is True
    assert drain_calls["count"] == 1
    assert result["status"] == "arrived_exact"


def test_probe_single_movement_target_raises_when_self_state_missing_after_outcome() -> None:
    clock = ReplayClock(1000)
    page = ClockAdvancingPage(
        clock,
        on_wait=_SequencedWorld([_make_world(1000, 100, 100, 900)]).advance,
    )
    probe = _SingleTargetHarness(page)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    def _fake_wait_for_missing_self(
        page: action_session.WaitPageProtocol,
        probe: MovementOutcomeProbeProtocol,
        *,
        target_x: int,
        target_y: int,
        move_started_ms: int,
        timeout_ms: int,
    ) -> tuple[Literal["arrived_exact", "move_timeout"], int, int, int, int]:
        _ = (page, target_x, target_y, move_started_ms, timeout_ms)
        if not isinstance(probe, _SingleTargetHarness):
            raise AssertionError("expected single-target harness")
        probe._self_state = None
        return ("move_timeout", 1600, 600, 118, 119)

    movement_probe_module._wait_for_move_outcome = _fake_wait_for_missing_self
    with pytest.raises(
        MovementProbeError,
        match="self state is unavailable after movement probe attempt",
    ):
        probe._probe_single_movement_target(
            TeleportTargetDict(label="move_1", x=120, y=121),
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )


class _SteppingClock:
    def __init__(self, start_ms: int, step_ms: int) -> None:
        self._current_ms = start_ms
        self._step_ms = step_ms

    def __call__(self) -> int:
        value = self._current_ms
        self._current_ms += self._step_ms
        return value


def _wait_for_initial_self_state_101_102(
    page: action_session.WaitPageProtocol,
    provider: action_session.BufferedWorldStateProviderProtocol,
    started_ms: int,
    timeout_ms: int,
) -> tuple[int, SelfStateDict]:
    _ = (page, provider, started_ms, timeout_ms)
    return (1500, make_self_state(1, 101, 102, 2, 1, 900, 5))


def _wait_for_initial_self_state_103_104(
    page: action_session.WaitPageProtocol,
    provider: action_session.BufferedWorldStateProviderProtocol,
    started_ms: int,
    timeout_ms: int,
) -> tuple[int, SelfStateDict]:
    _ = (page, provider, started_ms, timeout_ms)
    return (1400, make_self_state(1, 103, 104, 2, 1, 900, 5))


def _advance_startup_state_stub(bot: action_session.StartupStateDriverProtocol) -> None:
    _ = bot


def test_execute_probe_rejects_negative_map_open_delay() -> None:
    probe = MovementProbe("https://tankpit.com/play", headless=False, prefer_account=True)
    with pytest.raises(ValueError, match="map_open_delay_ms must be non-negative"):
        probe.execute_probe(
            explicit_targets=None,
            max_targets=1,
            initial_sync_timeout_ms=10000,
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=-1,
            settle_delay_ms=500,
        )


def test_execute_probe_rejects_negative_settle_delay() -> None:
    probe = MovementProbe("https://tankpit.com/play", headless=False, prefer_account=True)
    with pytest.raises(ValueError, match="settle_delay_ms must be non-negative"):
        probe.execute_probe(
            explicit_targets=None,
            max_targets=1,
            initial_sync_timeout_ms=10000,
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=-1,
        )


def test_execute_probe_runs_successfully_with_explicit_targets() -> None:
    attempts = [_make_attempt("arrived_exact")]
    harness = _ExecuteSuccessHarness(
        attempts=attempts,
        default_targets=[],
    )
    clock = _SteppingClock(1000, 100)
    action_hooks.get_current_time_ms = clock
    recorded = RecordedChromiumSession.from_capture_path(harness, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = _wait_for_initial_self_state_101_102
    action_hooks.advance_startup_state = _advance_startup_state_stub

    explicit_targets = [TeleportTargetDict(label="move_1", x=120, y=121)]
    session = harness.execute_probe(
        explicit_targets=explicit_targets,
        max_targets=1,
        initial_sync_timeout_ms=5000,
        move_timeout_ms=3000,
        queue_map_open_during_move=True,
        map_open_delay_ms=150,
        settle_delay_ms=250,
    )
    assert session["targets"] == explicit_targets
    assert session["attempts"] == attempts
    assert session["spawn_x"] == 101
    assert session["spawn_y"] == 102
    assert harness.probed_targets == explicit_targets
    assert harness._page is None
    assert harness._cdp is None


def test_execute_probe_uses_default_targets_when_explicit_targets_are_absent() -> None:
    default_targets = [TeleportTargetDict(label="move_1", x=120, y=121)]
    attempts = [_make_attempt("move_timeout")]
    harness = _ExecuteSuccessHarness(
        attempts=attempts,
        default_targets=default_targets,
    )
    clock = _SteppingClock(1000, 100)
    action_hooks.get_current_time_ms = clock
    recorded = RecordedChromiumSession.from_capture_path(harness, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = _wait_for_initial_self_state_103_104
    action_hooks.advance_startup_state = _advance_startup_state_stub

    session = harness.execute_probe(
        explicit_targets=None,
        max_targets=1,
        initial_sync_timeout_ms=5000,
        move_timeout_ms=3000,
        queue_map_open_during_move=False,
        map_open_delay_ms=0,
        settle_delay_ms=0,
    )
    assert session["targets"] == default_targets
    assert session["attempts"] == attempts
    assert harness.probed_targets == default_targets


def test_execute_probe_raises_when_target_builder_returns_empty_list() -> None:
    harness = _ExecuteSuccessHarness(
        attempts=[],
        default_targets=[],
    )
    clock = _SteppingClock(1000, 100)
    action_hooks.get_current_time_ms = clock
    recorded = RecordedChromiumSession.from_capture_path(harness, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = _wait_for_initial_self_state_103_104
    action_hooks.advance_startup_state = _advance_startup_state_stub

    with pytest.raises(MovementProbeError, match="requires at least one target"):
        harness.execute_probe(
            explicit_targets=None,
            max_targets=1,
            initial_sync_timeout_ms=5000,
            move_timeout_ms=3000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )
