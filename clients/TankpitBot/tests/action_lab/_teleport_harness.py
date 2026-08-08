"""Shared builders and probe doubles for the teleport test modules.

``test_teleport.py`` was 1,506 lines; it is now five modules over this
harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from tests.action_lab._replay_cdp import StubSnapshotCDPSession
from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot._test_hooks import (
    PageProtocol,
)
from tankpit_bot.action_lab.teleport import (
    TeleportProbe,
)
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    build_box_targets,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportProbeSessionDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import make_viewport_state
from tankpit_bot.types import (
    CapturedMessage,
)

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


class _SequencedProvider:
    def __init__(self, worlds: list[WorldStateDict]) -> None:
        self.world = get_world_service()
        self._worlds = worlds
        self._index = 0
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None
        self._messages: list[CapturedMessage] = []
        self._magic: str | None = None

    def get_world_state(self) -> WorldStateDict:
        return self._worlds[self._index]

    def advance(self) -> None:
        if self._index + 1 < len(self._worlds):
            self._index += 1

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages

    @property
    def magic(self) -> str | None:
        return self._magic


class _AckSequence:
    def __init__(self, values: list[bool]) -> None:
        self._values = values
        self._index = 0

    def __call__(self) -> bool:
        if self._index >= len(self._values):
            return False
        value = self._values[self._index]
        self._index += 1
        return value


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


def _make_attempt(
    status: Literal["landed_exact", "landed_offset", "map_sync_timeout", "teleport_timeout"],
) -> TeleportAttemptResultDict:
    return TeleportAttemptResultDict(
        target=TeleportTargetDict(label=status, x=150, y=171),
        teleport_cycle_id=1,
        status=status,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200 if status != "map_sync_timeout" else None,
        teleport_started_ms=1300 if status != "map_sync_timeout" else None,
        completion_timestamp_ms=1500,
        map_sync_elapsed_ms=200 if status != "map_sync_timeout" else None,
        teleport_elapsed_ms=200 if status in ("landed_exact", "landed_offset") else None,
        fuel_before=900,
        fuel_after=840,
        world_timestamp_before=950,
        world_timestamp_after=1450,
        landed_signal_received=status in ("landed_exact", "landed_offset"),
        landed_x=150,
        landed_y=171,
        message_start_index=10,
        message_end_index=14,
        page_snapshots=[],
    )


def _make_page_snapshot(
    phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
) -> TeleportPageSnapshotDict:
    """Build a sample teleport page snapshot."""
    return TeleportPageSnapshotDict(
        phase=phase,
        timestamp_ms=1000,
        client_present=True,
        map_visible=True,
        client_state=13,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=12,
        last_page_client_send_age_ms=250,
        last_bot_send_age_ms=10,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections={},
    )


class _ProbeMethodHarness(TeleportProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=False)
        self._self_state: SelfStateDict | None = make_self_state(
            tank_id=1,
            x=158,
            y=132,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        self._world_state = _make_world(1000, 158, 132, 900)
        self._fake_page = ClockAdvancingPage(
            ReplayClock(1000),
            on_wait=_SequencedProvider([self._world_state]).advance,
        )
        self._cdp = StubSnapshotCDPSession()
        self.map_open_result = True
        self.teleport_result = True
        self.teleport_calls: list[tuple[int, int]] = []

    def _require_page(self) -> PageProtocol:
        return self._fake_page

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._self_state

    def open_map(self) -> bool:
        return self.map_open_result

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleport_calls.append((x, y))
        return self.teleport_result


class _ProbeMissingPageHarness(_ProbeMethodHarness):
    def _require_page(self) -> PageProtocol:
        raise TeleportProbeError("page is unavailable")


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, TeleportProbe):
    def __init__(self) -> None:
        TeleportProbe.__init__(
            self, "https://tankpit.com/play", headless=False, prefer_account=True
        )
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 158, 132, 900)
        self.probed_targets: list[TeleportTargetDict] = []
        self.result_attempts: list[TeleportAttemptResultDict] = []

    def _probe_single_target(
        self,
        target: TeleportTargetDict,
        *,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
    ) -> TeleportAttemptResultDict:
        assert teleport_strategy == "sync_before_teleport"
        assert map_sync_timeout_ms == 3000
        assert teleport_timeout_ms == 10000
        assert settle_delay_ms == 500
        self.probed_targets.append(target)
        return self.result_attempts[len(self.probed_targets) - 1]


class _FakeTeleportProbe(TeleportProbe):
    def __init__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
    ) -> None:
        _ = (cdp_service, command_service)
        self._fake_target_url = target_url
        self._fake_headless = headless
        self._fake_prefer_account = prefer_account

    def execute(
        self,
        *,
        explicit_targets: list[TeleportTargetDict] | None,
        box_step_x: int,
        box_step_y: int,
        max_targets: int | None,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        initial_sync_timeout_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
    ) -> TeleportProbeSessionDict:
        targets = (
            explicit_targets
            if explicit_targets is not None
            else build_box_targets(100, 100, box_step_x, box_step_y)
        )
        limited_targets = targets if max_targets is None else targets[:max_targets]
        return TeleportProbeSessionDict(
            session_id="fake-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self._fake_target_url,
            spawn_x=100,
            spawn_y=100,
            teleport_strategy=teleport_strategy,
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
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            targets=limited_targets,
            attempts=[],
        )

    @property
    def messages(self) -> list[CapturedMessage]:
        return []

    @property
    def magic(self) -> str | None:
        return None

    @property
    def session_id(self) -> str:
        return "fake-session"
