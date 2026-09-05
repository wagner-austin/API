"""Shared builders and probe doubles for the combat-probe tests.

``test_combat_probe_harness.py`` holds the unit-level paths and
``test_combat_probe_execute.py`` the live-session ones; both build the
same world and the same two probe doubles, so they live here rather
than in either file.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from tests.action_lab._replay_cdp import StubSnapshotCDPSession
from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.action_lab._replay_page import ClockAdvancingPage, ReplayClock

from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
    PageProtocol,
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.combat_probe import CombatProbe
from tankpit_bot.action_lab.combat_probe_types import (
    CombatEngagementDict,
    CombatProbeSessionDict,
    CombatShotResultDict,
)
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterProtocol,
    TeleportPhaseProbeProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.ai.world_types import EnemyThreatDict, make_enemy_threat
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import make_viewport_state

#: A real recorded session, replayed so the bootstrap path runs against
#: bytes the server actually sent rather than a hand-rolled fixture.
_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"

_SnapshotPhase = Literal[
    "before_map_open",
    "before_teleport",
    "after_map_data",
    "landed",
    "timeout",
]


class AcquisitionPhaseFn(Protocol):
    """The acquisition phase, spelled out so stubs are checked against it."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        *,
        cdp: CDPSessionProtocol | None,
        send_command: Callable[[], bool],
        command_name: str,
        capture_before_map_open: bool,
        wait_for_sync: bool,
        sync_timeout_ms: int,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
    ) -> tuple[
        int,
        int | None,
        list[TeleportPageSnapshotDict],
        Callable[[_SnapshotPhase], TeleportPageSnapshotDict],
    ]:
        """Run the phase and report when the map synced."""


class TeleportCommandFn(Protocol):
    """The tracked teleport command, spelled out for the same reason."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        probe: TeleportPhaseProbeProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle: ActionPhaseCycleDict,
        message_start_index: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[[_SnapshotPhase], TeleportPageSnapshotDict],
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = ...,
    ) -> tuple[TeleportAttemptResultDict, int]:
        """Dispatch the teleport and wait for its outcome."""


def _make_world(
    timestamp_ms: int,
    x: int,
    y: int,
    fuel: int,
) -> WorldStateDict:
    """Build a world state centred on the given self position.

    Args:
        timestamp_ms: World timestamp.
        x: Self X tile.
        y: Self Y tile.
        fuel: Self fuel.

    Returns:
        A world whose viewport is the 16x16 window around the tank.
    """
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
        viewport=make_viewport_state(left=x - 8, top=y - 8, width=16, height=16),
        scanned_tiles=world["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def _engagement(target_id: int, *, shot_ms: int | None = None) -> CombatEngagementDict:
    """Build an engagement record, optionally carrying one shot.

    Args:
        target_id: Tank id the engagement locked.
        shot_ms: Timestamp of a single recorded shot, or ``None`` for
            an engagement that never fired.

    Returns:
        An engagement record shaped as the probe emits one.
    """
    shots: list[CombatShotResultDict] = []
    if shot_ms is not None:
        shots.append(
            CombatShotResultDict(
                shot_number=1,
                self_x=100,
                self_y=100,
                target_x=101,
                target_y=100,
                distance=1,
                result="hit",
                weapon_byte=1,
                target_name=f"enemy-{target_id}",
                target_id=target_id,
                timestamp_ms=shot_ms,
            )
        )
    return CombatEngagementDict(
        target_id=target_id,
        target_name=f"enemy-{target_id}",
        initial_target_x=101,
        initial_target_y=100,
        initial_distance=1,
        landed_x=100,
        landed_y=100,
        shots=shots,
        total_hits=1 if shot_ms is not None else 0,
        total_misses=0,
        total_timeouts=0,
        kill_confirmed=False,
        target_fled=False,
        final_target_x=101,
        final_target_y=100,
        final_distance=1,
    )


class _ProbeHarness(CombatProbe):
    """Combat probe with page, CDP, world and shooting stubbed."""

    def __init__(self) -> None:
        """Seed a spawned tank at (100, 100) with a full tank of fuel."""
        ws = WorldService()
        super().__init__(
            "https://tankpit.com/play",
            headless=True,
            prefer_account=False,
            world=ws,
        )
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
        self.shoot_calls: list[tuple[int, int, int]] = []

    def _require_page(self) -> PageProtocol:
        """Return the clock-advancing fake page."""
        return self._fake_page

    def get_world_state(self) -> WorldStateDict:
        """Return the seeded world."""
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        """Return the seeded self state."""
        return self._self_state

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        """Record a shot instead of dispatching it."""
        self.shoot_calls.append((x, y, target_id))
        return True


class _ExecuteHarness(
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
    CombatProbe,
):
    """Probe whose engagement step is scripted, for session-level tests."""

    def __init__(self) -> None:
        """Install the bootstrap stubs and an empty engagement script."""
        CombatProbe.__init__(
            self,
            "https://tankpit.com/play",
            headless=False,
            prefer_account=True,
        )
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.engagement_results: list[CombatEngagementDict | None] = []
        self.excluded_ids_log: list[frozenset[int]] = []
        self._call_count = 0
        self._scripted_engagement: CombatEngagementDict | None = None

    def _acquire_adjacent_enemy(
        self,
        *,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        excluded_ids: frozenset[int],
    ) -> EnemyThreatDict | None:
        """Advance the script, logging the exclusions.

        A ``None`` script slot plays a failed acquisition; otherwise
        the slot's target id becomes the threat handed to the (also
        scripted) engagement step — the same two-seam composition the
        production loop runs.
        """
        _ = (acquisition_timeout_ms, teleport_timeout_ms)
        self.excluded_ids_log.append(excluded_ids)
        result = self.engagement_results[self._call_count]
        self._call_count += 1
        self._scripted_engagement = result
        if result is None:
            return None
        return make_enemy_threat(
            tank_id=result["target_id"],
            x=result["initial_target_x"],
            y=result["initial_target_y"],
            distance=result["initial_distance"],
            damage_state=3,
            rank=1,
            team=2,
            name=result["target_name"],
            is_bot=True,
            timestamp_ms=0,
            last_wire_seen_ms=0,
            last_position_update_ms=0,
        )

    def _engage_single_target(
        self,
        target: EnemyThreatDict,
        max_shots: int,
    ) -> CombatEngagementDict:
        """Return the scripted engagement for the acquired threat."""
        _ = (target, max_shots)
        scripted = self._scripted_engagement
        if scripted is None:
            raise AssertionError("engage reached without a scripted engagement")
        return scripted


class _FakeCombatProbe(CombatProbe):
    """Probe whose whole session is canned, for the run/save wiring."""

    def execute_probe(
        self,
        *,
        max_engagements: int,
        max_shots_per_engagement: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
    ) -> CombatProbeSessionDict:
        """Echo the bounds it was handed back in a canned session."""
        _ = (acquisition_timeout_ms, teleport_timeout_ms)
        return CombatProbeSessionDict(
            session_id="combat-session",
            start_timestamp_ms=1,
            end_timestamp_ms=2,
            base_url="https://tankpit.com/play",
            spawn_x=100,
            spawn_y=100,
            max_engagements=max_engagements,
            max_shots_per_engagement=max_shots_per_engagement,
            capture_session_path="",
            initial_sync_timeout_ms=initial_sync_timeout_ms,
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
            engagements=[_engagement(7, shot_ms=1500)],
        )


class _CombatModuleProtocol(Protocol):
    """Every probe-module attribute the combat tests swap.

    Typing the surface here rather than reaching at the module keeps
    each stub checked against the real signature, so a change to any
    of these functions fails type-checking instead of silently
    leaving a test driving a shape the production code no longer has.
    """

    CombatProbe: type[CombatProbe]
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
    _find_fresh_enemy: Callable[[ProbeBase, int, frozenset[int]], EnemyThreatDict | None]
    _current_enemy_by_id: Callable[[ProbeBase, int], EnemyThreatDict | None]


_combat_module_import = __import__(
    "tankpit_bot.action_lab.combat_probe",
    fromlist=["combat_probe"],
)


combat_module: _CombatModuleProtocol = _combat_module_import


def stub_initial_sync() -> None:
    """Resolve the bootstrap's initial world sync at a fixed spawn."""

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
                leaderboard_position=5,
            ),
        )

    from tankpit_bot.action_lab import _test_hooks as action_hooks

    action_hooks.wait_for_initial_self_state = _wait_initial


def require_engagement(result: CombatEngagementDict | None) -> CombatEngagementDict:
    """Return the engagement, failing loudly when the probe gave up.

    Lets a caller assert on the engagement's contents rather than on
    its mere existence, which proves nothing about what happened.

    Args:
        result: What the acquire-then-engage cycle returned.

    Returns:
        The engagement.

    Raises:
        AssertionError: When the probe returned no engagement.
    """
    if result is None:
        raise AssertionError("expected an engagement; the probe gave up instead")
    return result


def require_threat(threat: EnemyThreatDict | None) -> EnemyThreatDict:
    """Return the threat, failing loudly when the lookup found nothing.

    Args:
        threat: What a threat lookup returned.

    Returns:
        The threat.

    Raises:
        AssertionError: When the lookup found no threat.
    """
    if threat is None:
        raise AssertionError("expected a threat; the lookup found none")
    return threat
