"""Shared builders and probe doubles for the density-probe tests."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.density_probe import (
    DensityProbe,
    DensityProbeSessionDict,
)
from tankpit_bot.inventory import (
    InventoryItem,
    InventoryState,
)
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
)
from tankpit_bot.state.types import (
    ContainerStateDict,
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


def _inventory(*, radar_count: int, radar_enabled: bool) -> InventoryState:
    item = InventoryItem(count=25, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=item,
        missile_shots=item,
        homing_shots=item,
        extra_radars=InventoryItem(count=radar_count, enabled=radar_enabled),
    )


class _DensityHarness(DensityProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=True)
        self._clock = ReplayClock(1000)
        self._page = ClockAdvancingPage(self._clock)
        self.inventory_calls = 0
        self.radar_calls = 0
        self.map_calls = 0
        self.sent_toggles: list[int] = []
        self.inventory_script: list[InventoryState] = []
        self.teleports: list[tuple[int, int]] = []
        self.pickups: list[tuple[int, int]] = []
        self.fuel = 1100
        self.position = (100, 100)
        self.visible_containers: dict[str, ContainerStateDict] = {}

    def request_inventory(self) -> bool:
        if self.inventory_script:
            get_world_service().inventory_state = self.inventory_script.pop(0)
        self.inventory_calls += 1
        return True

    def use_radar(self) -> bool:
        self.radar_calls += 1
        return True

    def open_map(self) -> bool:
        self.map_calls += 1
        return True

    def toggle_equipment_slot(self, slot: int) -> bool:
        self.sent_toggles.append(slot)
        return True

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleports.append((x, y))
        self.position = (x, y)
        return True

    def pickup_fuel(self, x: int, y: int) -> bool:
        self.pickups.append((x, y))
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


def _install_noop_drain() -> None:
    def _drain(provider: BufferedMessageSourceProtocol) -> int:
        del provider
        return 0

    action_hooks.drain_buffered_messages = _drain


class _PayingDotHarness(_DensityHarness):
    """A harness whose refuel landings always pay out."""

    def teleport_to(self, x: int, y: int) -> bool:
        self.fuel = 900
        return super().teleport_to(x, y)


class _PayingPickupHarness(_DensityHarness):
    """A harness whose fuel pickups always pay out."""

    def pickup_fuel(self, x: int, y: int) -> bool:
        self.fuel += 400
        return super().pickup_fuel(x, y)


class _PayingBlindWalkHarness(_DensityHarness):
    """Blind dot-walk pickups pay out."""

    def pickup_fuel(self, x: int, y: int) -> bool:
        self.fuel += 600
        return super().pickup_fuel(x, y)

    def move_to(self, x: int, y: int) -> bool:
        self.position = (x, y)
        return True


class _ToggleRecorder(DensityProbe):
    def __init__(self) -> None:
        self._dispatched: list[tuple[bytes, str]] = []
        self._commands_xor_table = None

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        self._dispatched.append((data, cmd_name))
        return True


def _session() -> DensityProbeSessionDict:
    return DensityProbeSessionDict(
        session_id="density-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="density_probe.capture_session.json",
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
        max_extras=12,
        sites_planned=16,
        sites_scanned=12,
        sites_skipped=1,
        refuel_hops=4,
        bootstrap_pickups=2,
        extras_before=22,
        extras_enabled_at_start=False,
        toggles_sent=2,
        extras_after=10,
        fuel_before=1100,
        fuel_after=430,
    )


class _FakeDensityProbe(DensityProbe):
    def execute_probe(
        self,
        *,
        max_extras: int,
        initial_sync_timeout_ms: int,
    ) -> DensityProbeSessionDict:
        session = _session()
        session["max_extras"] = max_extras
        session["initial_sync_timeout_ms"] = initial_sync_timeout_ms
        session["capture_session_path"] = ""
        return session


class _DensityModuleProtocol(Protocol):
    DensityProbe: type[DensityProbe]


_density_module_import = __import__(
    "tankpit_bot.action_lab.density_probe",
    fromlist=["density_probe"],
)


density_module: _DensityModuleProtocol = _density_module_import


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, DensityProbe):
    def __init__(self) -> None:
        DensityProbe.__init__(self, "https://tankpit.com/play", headless=False, prefer_account=True)
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.phases: list[str] = []

    def _current_fuel(self) -> tuple[int, int, int]:
        self.phases.append("fuel")
        return 900, 100, 100

    def _ensure_extras_enabled(self) -> tuple[int, bool, int]:
        self.phases.append("enable")
        return 22, False, 1

    def _sweep_sites(self, max_extras: int) -> tuple[int, int, int, int]:
        self.phases.append(f"sweep:{max_extras}")
        return 12, 3, 2, 1

    def _restore_extras_state(self, was_enabled: bool) -> int:
        self.phases.append(f"restore:{was_enabled}")
        return 1

    def _read_extras(self) -> tuple[int, bool]:
        self.phases.append("read")
        return 10, False

    def _quit_to_lobby(self) -> None:
        self.phases.append("quit")
