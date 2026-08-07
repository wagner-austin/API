"""Shared builders and probe doubles for the movement-probe tests.

``test_movement_probe.py`` was 1,180 lines; it is now four modules over
this harness.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from pathlib import Path
from typing import Literal

from platform_core.json_utils import (
    JSONObject,
)
from tests.action_lab._replay_core import StubbedBootstrapMixin

from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.movement_probe import (
    MovementProbe,
)
from tankpit_bot.action_lab.movement_probe_types import (
    MovementProbeAttemptResultDict,
    MovementProbeSessionDict,
)
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.types import (
    CapturedMessage,
)

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
        self.xor_table: bytes | None = None

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


class _MissingSelfWaitProbe:
    def __init__(self, states: list[SelfStateDict | None]) -> None:
        self._states = states
        self._index = 0
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None
        self._world = _make_world(1000, 100, 100, 900)

    def _update_state_from_world(self) -> None:
        if self._index + 1 < len(self._states):
            self._index += 1

    def get_world_state(self) -> WorldStateDict:
        return self._world

    def get_self_state(self) -> SelfStateDict | None:
        return self._states[self._index]


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

    def is_landing_attainable(self, x: int, y: int) -> bool:
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
