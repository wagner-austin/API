"""Shared builders, probe doubles, and clock stubs for the queue-probe tests."""

from __future__ import annotations

from pathlib import Path

from tests.action_lab._replay_core import StubbedBootstrapMixin
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.queue_probe import QueueProbe
from tankpit_bot.action_lab.queue_probe_types import (
    QueueCommandTimingDict,
    QueueExperimentKind,
    QueueExperimentResultDict,
    QueueProbeSessionDict,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.types import CapturedMessage

_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


def _make_world(timestamp_ms: int, x: int, y: int, fuel: int) -> WorldStateDict:
    empty = make_empty_world_state()
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
        tanks=empty["tanks"],
        containers=empty["containers"],
        mines=empty["mines"],
        terrain=empty["terrain"],
        viewport=empty["viewport"],
        scanned_tiles=empty["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def _make_world_no_self(timestamp_ms: int) -> WorldStateDict:
    empty = make_empty_world_state()
    return WorldStateDict(
        self_state=None,
        tanks=empty["tanks"],
        containers=empty["containers"],
        mines=empty["mines"],
        terrain=empty["terrain"],
        viewport=empty["viewport"],
        scanned_tiles=empty["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


class _SequencedWorld:
    def __init__(self, states: list[WorldStateDict]) -> None:
        self._states = states
        self._index = 0

    def current(self) -> WorldStateDict:
        return self._states[self._index]

    def advance(self) -> None:
        if self._index + 1 < len(self._states):
            self._index += 1


class _FakeExperimentProbe:
    """Satisfies QueueExperimentProbeProtocol structurally."""

    def __init__(self, worlds: _SequencedWorld, clock: ReplayClock) -> None:
        ws = WorldService()
        self.world = ws
        self._worlds = worlds
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None
        self._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
        self._messages_list: list[CapturedMessage] = []
        self._commands_sent: list[str] = []

    def get_world_state(self) -> WorldStateDict:
        return self._worlds.current()

    def get_self_state(self) -> SelfStateDict | None:
        return self._worlds.current()["self_state"]

    def _update_state_from_world(self) -> None:
        pass

    def _require_page(self) -> ClockAdvancingPage:
        return self._page

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages_list

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        self._commands_sent.append(f"shoot({x},{y},id={target_id})")
        return True

    def pickup_fuel(self, x: int, y: int) -> bool:
        self._commands_sent.append(f"pickup_fuel({x},{y})")
        return True

    def move_to(self, x: int, y: int) -> bool:
        self._commands_sent.append(f"move({x},{y})")
        return True


def _setup_probe(
    worlds: _SequencedWorld,
    clock: ReplayClock,
) -> _FakeExperimentProbe:
    probe = _FakeExperimentProbe(worlds, clock)
    probe._page = ClockAdvancingPage(clock, on_wait=worlds.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source, ws: 0
    return probe


class _FailingCommandProbe(_FakeExperimentProbe):
    """Probe whose commands return False to test error branches."""

    def __init__(
        self,
        worlds: _SequencedWorld,
        clock: ReplayClock,
        *,
        fail_shoot: bool = False,
        fail_pickup: bool = False,
        fail_move: bool = False,
    ) -> None:
        super().__init__(worlds, clock)
        self._fail_shoot = fail_shoot
        self._fail_pickup = fail_pickup
        self._fail_move = fail_move

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        if self._fail_shoot:
            return False
        return super().shoot(x, y, target_id)

    def pickup_fuel(self, x: int, y: int) -> bool:
        if self._fail_pickup:
            return False
        return super().pickup_fuel(x, y)

    def move_to(self, x: int, y: int) -> bool:
        if self._fail_move:
            return False
        return super().move_to(x, y)


class _SecondShootFailsProbe(_FakeExperimentProbe):
    """Probe where only the second shoot call fails."""

    def __init__(self, worlds: _SequencedWorld, clock: ReplayClock) -> None:
        super().__init__(worlds, clock)
        self._shoot_count = 0

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        self._shoot_count += 1
        if self._shoot_count >= 2:
            return False
        return super().shoot(x, y, target_id)


class _SteppingClock:
    def __init__(self, start_ms: int, step_ms: int) -> None:
        self._current_ms = start_ms
        self._step_ms = step_ms

    def __call__(self) -> int:
        value = self._current_ms
        self._current_ms += self._step_ms
        return value


class _ExecuteHarness(StubbedBootstrapMixin, QueueProbe):
    def __init__(self) -> None:
        QueueProbe.__init__(
            self,
            "https://tankpit.com/play",
            headless=False,
            prefer_account=True,
        )
        self._init_bootstrap_stubs()


def _make_experiment_result(kind: QueueExperimentKind) -> QueueExperimentResultDict:
    return QueueExperimentResultDict(
        kind=kind,
        status="both_processed",
        primary=QueueCommandTimingDict(
            label="primary",
            sent_ms=100,
            ack_ms=200,
            elapsed_ms=100,
        ),
        secondary=QueueCommandTimingDict(
            label="secondary",
            sent_ms=105,
            ack_ms=210,
            elapsed_ms=105,
        ),
        inter_send_delay_ms=5,
        total_elapsed_ms=115,
        message_start_index=0,
        message_end_index=1,
    )


def _wait_for_initial_self_state_spawn(
    page: action_session.WaitPageProtocol,
    provider: action_session.BufferedWorldStateProviderProtocol,
    started_ms: int,
    timeout_ms: int,
) -> tuple[int, SelfStateDict]:
    _ = (page, provider, started_ms, timeout_ms)
    return (1500, make_self_state(1, 101, 102, 2, 1, 900, 5))


def _advance_startup_state_stub(
    bot: action_session.StartupStateDriverProtocol,
) -> None:
    _ = bot


class _FakeQueueProbeForRunner(QueueProbe):
    """QueueProbe that returns a canned session from execute_probe."""

    def execute_probe(
        self,
        *,
        initial_sync_timeout_ms: int,
        experiment_timeout_ms: int,
        experiment_kinds: list[QueueExperimentKind],
    ) -> QueueProbeSessionDict:
        _ = (initial_sync_timeout_ms, experiment_timeout_ms, experiment_kinds)
        return QueueProbeSessionDict(
            session_id="fake-queue-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self._target_url,
            spawn_x=100,
            spawn_y=100,
            capture_session_path="",
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            experiment_timeout_ms=experiment_timeout_ms,
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
            experiments=[],
        )
