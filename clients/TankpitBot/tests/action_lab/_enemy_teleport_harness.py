"""Shared handles, builders, and probe doubles for the enemy-teleport tests.

``test_enemy_teleport.py`` was 1,391 lines; it is now four modules over
this harness. The autouse hook-restore fixture lives in
``tests/action_lab/conftest.py``.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from pathlib import Path
from typing import (
    Literal,
    Protocol,
)

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
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.enemy_teleport import EnemyTeleportProbe
from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportAttemptResultDict,
    EnemyTeleportProbeSessionDict,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.ai.world_types import (
    EnemyThreatDict,
    make_enemy_threat,
)
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
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


class _EnemyTeleportModuleProtocol(Protocol):
    analyze_threats: Callable[[WorldStateDict, SelfStateDict], list[EnemyThreatDict]]
    choose_combat_landing_tile: Callable[
        [WorldStateDict, SelfStateDict, EnemyThreatDict, TerrainMapProtocol | None, int],
        tuple[int, int],
    ]
    _wait_for_teleport_outcome: _WaitForTeleportOutcomeProtocol


class _EnemyTargetingModuleProtocol(Protocol):
    """Typed access to the patchable enemy-targeting module globals.

    Enemy selection moved to
    :mod:`tankpit_bot.action_lab.enemy_teleport_targeting` when the
    647-line probe module was split; the probe reaches these through
    that module, so this is where tests swap them.
    """

    _require_fresh_enemy_threat: Callable[
        [EnemyTeleportProbe, int, frozenset[int]],
        EnemyThreatDict | None,
    ]
    _enemy_by_id: Callable[[EnemyTeleportProbe, int], EnemyThreatDict | None]


class _WaitForTeleportOutcomeProtocol(Protocol):
    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
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


class _EnemyTeleportProbeClassModuleProtocol(Protocol):
    EnemyTeleportProbe: type[EnemyTeleportProbe]


_enemy_module_import = __import__(
    "tankpit_bot.action_lab.enemy_teleport",
    fromlist=["enemy_teleport"],
)


enemy_module: _EnemyTeleportModuleProtocol = _enemy_module_import

_enemy_targeting_import = __import__(
    "tankpit_bot.action_lab.enemy_teleport_targeting",
    fromlist=["enemy_teleport_targeting"],
)
enemy_targeting_module: _EnemyTargetingModuleProtocol = _enemy_targeting_import


enemy_probe_module: _EnemyTeleportProbeClassModuleProtocol = _enemy_module_import


def _enemy(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 130,
    distance: int = 4,
    timestamp_ms: int = 1000,
) -> EnemyThreatDict:
    return make_enemy_threat(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=distance,
        damage_state=0,
        rank=1,
        team=1,
        name=f"red-{tank_id}",
        is_bot=False,
        timestamp_ms=timestamp_ms,
    )


def _target() -> TeleportTargetDict:
    return TeleportTargetDict(label="enemy_50_120_130", x=119, y=130)


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


class _SequencedProvider:
    def __init__(self, worlds: list[WorldStateDict]) -> None:
        self.world = get_world_service()
        self._worlds = worlds
        self._index = 0
        self._cdp_message_buffer: list[str] = []

    def get_world_state(self) -> WorldStateDict:
        return self._worlds[self._index]

    def advance(self) -> None:
        if self._index + 1 < len(self._worlds):
            self._index += 1


class _ProbeHarness(EnemyTeleportProbe):
    def __init__(self) -> None:
        self.world = get_world_service()
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
        self._fake_page = ClockAdvancingPage(
            ReplayClock(1000),
            on_wait=_SequencedProvider([self._world_state]).advance,
        )
        self._cdp = StubSnapshotCDPSession()
        self.map_open_result = True
        self.request_enemy_result = True
        self.teleport_result = True
        self.open_map_calls = 0
        self.request_enemy_calls = 0
        self.inventory_calls = 0
        self.move_calls: list[tuple[int, int]] = []
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

    def request_nearest_enemy(self) -> bool:
        self.request_enemy_calls += 1
        return self.request_enemy_result

    def request_inventory(self) -> bool:
        self.inventory_calls += 1
        return True

    def move_to(self, x: int, y: int) -> bool:
        self.move_calls.append((x, y))
        return True

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleport_calls.append((x, y))
        return self.teleport_result


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, EnemyTeleportProbe):
    def __init__(self) -> None:
        EnemyTeleportProbe.__init__(
            self, "https://tankpit.com/play", headless=False, prefer_account=True
        )
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.results: list[EnemyTeleportAttemptResultDict] = []
        self.acquisition_strategies: list[str] = []
        self.excluded_tank_ids: list[frozenset[int]] = []

    def _probe_single_enemy_attempt(
        self,
        *,
        acquisition_strategy: Literal["map_open", "nearest_enemy"],
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
        heartbeat_interval_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyTeleportAttemptResultDict:
        _ = (acquisition_timeout_ms, teleport_timeout_ms, settle_delay_ms, heartbeat_interval_ms)
        self.acquisition_strategies.append(acquisition_strategy)
        self.excluded_tank_ids.append(excluded_tank_ids)
        return self.results[len(self.acquisition_strategies) - 1]


class _FakeEnemyTeleportProbe(EnemyTeleportProbe):
    def __init__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
    ) -> None:
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            cdp_service=cdp_service,
            command_service=command_service,
        )

    def execute_probe(
        self,
        *,
        acquisition_strategy: Literal["map_open", "nearest_enemy"],
        max_attempts: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
        heartbeat_interval_ms: int,
    ) -> EnemyTeleportProbeSessionDict:
        return EnemyTeleportProbeSessionDict(
            session_id="enemy-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self._target_url,
            spawn_x=100,
            spawn_y=100,
            acquisition_strategy=acquisition_strategy,
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
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            heartbeat_interval_ms=0,
            attempts=[],
        )

    @property
    def messages(self) -> list[CapturedMessage]:
        return []

    @property
    def magic(self) -> str | None:
        return None
