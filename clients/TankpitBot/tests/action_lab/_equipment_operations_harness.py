"""Shared builders for the equipment-probe operation tests."""

from __future__ import annotations

from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseOverlapDict,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportTargetDict,
)
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import (
    ContainerStateDict,
    make_viewport_state,
)
from tankpit_bot.types import CapturedMessage


def _self_state() -> SelfStateDict:
    """Build a sample self state."""
    return make_self_state(
        tank_id=1, x=100, y=100, team=2, rank=1, fuel=700, leaderboard_position=1
    )


def _make_world() -> WorldStateDict:
    """Build a minimal world state."""
    base = make_empty_world_state()
    return WorldStateDict(
        self_state=_self_state(),
        tanks=base["tanks"],
        containers=base["containers"],
        mines=base["mines"],
        terrain=base["terrain"],
        viewport=make_viewport_state(left=92, top=92, width=16, height=16),
        scanned_tiles=base["scanned_tiles"],
        timestamp_ms=2000,
    )


def _target() -> TeleportTargetDict:
    """Build a sample teleport target."""
    return TeleportTargetDict(label="t", x=10, y=20)


def _teleport_result() -> TeleportAttemptResultDict:
    """Build a successful landed teleport result."""
    return TeleportAttemptResultDict(
        target=_target(),
        teleport_cycle_id=1,
        status="landed_exact",
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        completion_timestamp_ms=1500,
        map_sync_elapsed_ms=100,
        teleport_elapsed_ms=300,
        fuel_before=700,
        fuel_after=690,
        world_timestamp_before=1100,
        world_timestamp_after=1450,
        landed_signal_received=True,
        landed_x=10,
        landed_y=20,
        message_start_index=0,
        message_end_index=0,
        page_snapshots=[],
    )


class _Clock:
    """Mutable millisecond clock."""

    def __init__(self, start_ms: int) -> None:
        self._now_ms = start_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class _BuilderProbe:
    """Minimal probe satisfying the builder context protocol."""

    def __init__(
        self,
        *,
        messages: list[CapturedMessage] | None = None,
        self_state: SelfStateDict | None = None,
    ) -> None:
        self.world = get_world_service()
        self._messages = messages if messages is not None else []
        self._self_state = self_state if self_state is not None else _self_state()

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages

    def _require_self_state(self) -> SelfStateDict:
        return self._self_state


class _PickupProbe:
    """Minimal probe satisfying the equipment pickup context."""

    def __init__(
        self,
        *,
        clock: _Clock,
        move_result: bool,
    ) -> None:
        self.world = get_world_service()
        self._clock = clock
        self._messages: list[CapturedMessage] = []
        self._self_state = _self_state()
        self._world = _make_world()
        self._cycles: list[ActionPhaseCycleDict] = []
        self._cycle_id = 0
        self.move_result = move_result
        self.move_calls: list[tuple[int, int]] = []
        self.reset_calls = 0
        self._overlaps: list[ActionPhaseOverlapDict] = []
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages

    @property
    def magic(self) -> str | None:
        return None

    def get_world_state(self) -> WorldStateDict:
        return self._world

    def _require_self_state(self) -> SelfStateDict:
        return self._self_state

    def move_to(self, x: int, y: int) -> bool:
        self.move_calls.append((x, y))
        return self.move_result

    def _start_action_phase(
        self,
        phase: str,
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        _ = attempt_label
        self._cycle_id += 1
        if phase == "move":
            named: ActionPhaseCycleDict = ActionPhaseCycleDict(
                phase="move",
                cycle_id=self._cycle_id,
                started_ms=self._clock(),
            )
        else:
            named = ActionPhaseCycleDict(
                phase="pickup",
                cycle_id=self._cycle_id,
                started_ms=self._clock(),
            )
        self._cycles.append(named)
        return named

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        _ = cycle

    def _reset_probe_state_to_idle(self) -> None:
        self.reset_calls += 1

    def _get_attempt_phase_overlaps(self) -> list[ActionPhaseOverlapDict]:
        return list(self._overlaps)


_ContainerAlias = ContainerStateDict
