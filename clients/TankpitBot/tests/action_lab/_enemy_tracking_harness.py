"""Shared record builders for the enemy-tracking probe tests.

The codec suite, the record suite and the live-session suite all need
the same sample beliefs, observations and session payload, so they are
built here once rather than three times.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from tests.action_lab._combat_probe_harness import (
    AcquisitionPhaseFn,
    TeleportCommandFn,
)
from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)

from tankpit_bot._test_hooks import CDPSessionProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.enemy_tracking import EnemyTrackingProbe
from tankpit_bot.action_lab.enemy_tracking_types import (
    EnemyTrackingProbeSessionDict,
    JSTankBeliefDict,
    OurTankBeliefDict,
    ShotEventDict,
    TrackedEnemyDict,
    TrackingObservationDict,
)
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.types import (
    TeleportStartupTimingDict,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import SelfStateDict, WorldStateDict

_SnapshotPhase = Literal[
    "before_map_open",
    "before_teleport",
    "after_map_data",
    "landed",
    "timeout",
]


class AnalyzeThreatsFn(Protocol):
    """The threat analysis call, spelled out so stubs are checked against it.

    Spelled as a Protocol rather than a ``Callable`` alias because the
    world service is the first parameter: a bare ``Callable`` cannot
    express the keyword-only rank bounds, so a stub that dropped ``ws``
    typechecked against it while silently reading a different world.
    """

    def __call__(
        self,
        ws: WorldService,
        world: WorldStateDict,
        self_state: SelfStateDict,
        now_ms: int,
        *,
        human_min_rank: int = ...,
        human_max_rank: int = ...,
    ) -> list[EnemyThreatDict]:
        """Return the sorted enemy threats visible in ``world``."""
        ...


class ShotFeedbackFn(Protocol):
    """The shot-feedback wait, spelled out so stubs are checked against it."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: ProbeBase,
        *,
        timeout_ms: int,
    ) -> tuple[bool, bool]:
        """Report ``(got_response, was_hit)`` for the dispatched shot."""


def _make_our_belief() -> OurTankBeliefDict:
    return OurTankBeliefDict(
        tank_id=511,
        present=True,
        x=99,
        y=100,
        liveness="alive",
        last_wire_seen_ms=10_000,
        last_position_update_ms=9_500,
        wire_age_ms=500,
        position_age_ms=1_000,
        is_in_threats=True,
        would_locked_target_return=True,
        locked_target_source="threats",
    )


def _make_js_belief(*, present: bool = True) -> JSTankBeliefDict:
    fields: dict[str, int | float | bool | str | None] = (
        {"a": 511, "b": 99, "c": 100} if present else {}
    )
    return JSTankBeliefDict(present=present, fields=fields)


def _make_observation() -> TrackingObservationDict:
    return TrackingObservationDict(
        sample_index=3,
        sample_timestamp_ms=12_345,
        tank_id=511,
        tracked_label="orange-7",
        our_belief=_make_our_belief(),
        js_belief=_make_js_belief(),
        bot_combat_target_id=511,
        bot_mode_state="ENGAGE",
    )


def _make_tracked() -> TrackedEnemyDict:
    return TrackedEnemyDict(
        tank_id=511,
        name="orange-7",
        team=3,
        rank=4,
        acquired_x=99,
        acquired_y=100,
        tracked_js_key="a",
        tracked_js_value="511",
    )


def _make_shot() -> ShotEventDict:
    return ShotEventDict(
        target_tank_id=511,
        target_x=99,
        target_y=100,
        self_x=100,
        self_y=100,
        sent_ms=10_500,
        responded_ms=10_600,
        outcome="hit",
    )


def _make_snapshot() -> PageClientSnapshotDict:
    return PageClientSnapshotDict(
        timestamp_ms=11_000,
        client_present=True,
        map_visible=True,
        client_state=2,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=80,
        last_page_client_send_age_ms=60,
        last_bot_send_age_ms=40,
        ws_ready_state=1,
        current_send_label="shoot",
        sent_frame_meta_queue_length=0,
        self_fields={"x": 100, "y": 100},
        world_fields={"timestamp": 11_000},
        map_fields={},
        world_collections={"P.j": [{"a": 511, "b": 99, "c": 100}]},
    )


def _make_startup_timing() -> TeleportStartupTimingDict:
    return TeleportStartupTimingDict(
        game_ready_timestamp_ms=1_000,
        intel_ready_timestamp_ms=1_100,
        initial_sync_started_ms=1_200,
        initial_world_timestamp_ms=1_300,
        command_ready_timestamp_ms=1_400,
        first_attempt_started_ms=1_500,
        game_ready_to_intel_ready_ms=100,
        intel_ready_to_initial_world_ms=200,
        initial_world_to_command_ready_ms=100,
        command_ready_to_first_attempt_ms=100,
    )


def _make_session(*, with_shot: bool = True) -> EnemyTrackingProbeSessionDict:
    return EnemyTrackingProbeSessionDict(
        session_id="session-abc",
        start_timestamp_ms=1_000,
        end_timestamp_ms=2_000,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="runs/track/cap.json",
        initial_sync_timeout_ms=10_000,
        startup_timing=_make_startup_timing(),
        acquisition_timeout_ms=5_000,
        teleport_timeout_ms=10_000,
        shot_feedback_timeout_ms=4_000,
        sample_interval_ms=1_000,
        sample_duration_ms=120_000,
        tracked=[_make_tracked()],
        shot=_make_shot() if with_shot else None,
        snapshot_at_acquisition=_make_snapshot(),
        observations=[_make_observation()],
    )


class _TrackingModuleProtocol(Protocol):
    """Every probe-module attribute the tracking tests swap.

    Typed here rather than reached at on the module, so each stub is
    checked against the real signature and a change to any of these
    fails type-checking instead of leaving a test driving a shape the
    production code no longer has.
    """

    EnemyTrackingProbe: type[EnemyTrackingProbe]
    run_tracked_acquisition_phase: AcquisitionPhaseFn
    run_tracked_teleport_command: TeleportCommandFn
    choose_combat_landing_tile: Callable[
        [
            WorldStateDict,
            SelfStateDict,
            EnemyThreatDict,
            TerrainMapProtocol | None,
            int,
            WorldService,
        ],
        tuple[int, int],
    ]
    get_terrain_map: Callable[[], TerrainMapProtocol | None]
    analyze_threats: AnalyzeThreatsFn
    capture_page_client_snapshot: Callable[[CDPSessionProtocol], PageClientSnapshotDict]
    _wait_for_shot_feedback: ShotFeedbackFn


_tracking_module_import = __import__(
    "tankpit_bot.action_lab.enemy_tracking",
    fromlist=["enemy_tracking"],
)


tracking_module: _TrackingModuleProtocol = _tracking_module_import


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, EnemyTrackingProbe):
    """Probe whose four session stages are scripted, for execute_probe.

    Each stage is replaced by a recorder so the session assembly --
    which stages run, in what order, and what lands in the payload --
    is what the test observes, without a browser behind any of them.
    """

    def __init__(self, *, target: EnemyThreatDict | None) -> None:
        """Install bootstrap stubs and decide whether a target is found."""
        EnemyTrackingProbe.__init__(
            self,
            "https://tankpit.com/play",
            headless=False,
            prefer_account=True,
        )
        self._init_bootstrap_stubs()
        self._target = target
        self.stages: list[str] = []

    def _acquire_enemies(
        self,
        *,
        cdp: CDPSessionProtocol,
        acquisition_timeout_ms: int,
    ) -> tuple[list[TrackedEnemyDict], PageClientSnapshotDict, list[EnemyThreatDict]]:
        """Record the acquisition stage and hand back one tracked enemy."""
        _ = (cdp, acquisition_timeout_ms)
        self.stages.append("acquire")
        return ([_make_tracked()], _make_snapshot(), [])

    def _teleport_to_closest_enemy(
        self,
        *,
        cdp: CDPSessionProtocol,
        threats: list[EnemyThreatDict],
        teleport_timeout_ms: int,
        message_start_index: int,
    ) -> EnemyThreatDict | None:
        """Record the approach stage and return the scripted target."""
        _ = (cdp, threats, teleport_timeout_ms, message_start_index)
        self.stages.append("teleport")
        return self._target

    def _fire_one_shot(
        self,
        *,
        target: EnemyThreatDict,
        shot_feedback_timeout_ms: int,
    ) -> ShotEventDict:
        """Record the shot stage and return a canned shot event."""
        _ = (target, shot_feedback_timeout_ms)
        self.stages.append("shoot")
        return _make_shot()

    def _sample_loop(
        self,
        *,
        cdp: CDPSessionProtocol,
        tracked: list[TrackedEnemyDict],
        sample_interval_ms: int,
        sample_duration_ms: int,
    ) -> list[TrackingObservationDict]:
        """Record the sampling stage and return one observation row."""
        _ = (cdp, tracked, sample_interval_ms, sample_duration_ms)
        self.stages.append("sample")
        return [_make_observation()]
