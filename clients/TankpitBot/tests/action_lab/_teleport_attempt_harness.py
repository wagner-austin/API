"""Shared probe doubles and result builders for the tracked-teleport tests."""

from __future__ import annotations

from typing import Literal

from typing_extensions import Unpack

from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterKwargs,
    TeleportOutcomeWaiterProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import (
    WorldStateDict,
    make_empty_world_state,
)
from tankpit_bot.types import CapturedMessage


class _Probe:
    """Minimal probe fake for tracked teleport-attempt tests."""

    def __init__(self) -> None:
        """Initialize the fake probe."""
        ws = WorldService()
        self.world = ws
        self._messages = [
            CapturedMessage(
                direction="sent",
                payload="a",
                timestamp_ms=1,
                ws_url="wss://example.test/ws/",
            )
        ]
        self.started_cycles: list[tuple[str, str]] = []
        self.reset_idle_calls = 0
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return the captured message buffer."""
        return self._messages

    @property
    def magic(self) -> str:
        """Return a stable fake magic key."""
        return "magic"

    def get_world_state(self) -> WorldStateDict:
        """Return an empty world state."""
        return make_empty_world_state()

    def teleport_to(self, x: int, y: int) -> bool:
        """Reject direct teleport dispatch in this test layer."""
        _ = (x, y)
        raise AssertionError("teleport_to should not be called directly by this test")

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """Reject direct phase ending in this test layer."""
        _ = cycle
        raise AssertionError("_end_action_phase should not be called directly by this test")

    def _reset_probe_state_to_idle(self) -> None:
        """Record one idle reset."""
        self.reset_idle_calls += 1

    def _start_action_phase(
        self,
        phase: Literal["teleport"],
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Return one started teleport phase."""
        self.started_cycles.append((phase, attempt_label))
        return ActionPhaseCycleDict(phase="teleport", cycle_id=7, started_ms=1200)


class _Page:
    """Minimal page fake satisfying the wait protocol."""

    def wait_for_timeout(self, timeout: float) -> None:
        """Ignore wait requests."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)


def _target() -> TeleportTargetDict:
    """Build one sample teleport target."""
    return TeleportTargetDict(label="target", x=147, y=110)


def _snapshot(
    phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
) -> TeleportPageSnapshotDict:
    """Build one sample page snapshot."""
    return TeleportPageSnapshotDict(
        phase=phase,
        timestamp_ms=1000,
        client_present=True,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=1,
        last_page_client_send_age_ms=2,
        last_bot_send_age_ms=3,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        world_collections={},
        map_fields={},
    )


def _result(target: TeleportTargetDict) -> TeleportAttemptResultDict:
    """Build one sample teleport result."""
    return TeleportAttemptResultDict(
        target=target,
        teleport_cycle_id=7,
        status="landed_exact",
        map_open_started_ms=1500,
        map_sync_timestamp_ms=1700,
        teleport_started_ms=1800,
        completion_timestamp_ms=2200,
        map_sync_elapsed_ms=200,
        teleport_elapsed_ms=400,
        fuel_before=1100,
        fuel_after=1004,
        world_timestamp_before=900,
        world_timestamp_after=2100,
        landed_signal_received=True,
        landed_x=147,
        landed_y=110,
        message_start_index=1,
        message_end_index=5,
        page_snapshots=[_snapshot("before_map_open"), _snapshot("landed")],
    )


class _WaitForOutcome(TeleportOutcomeWaiterProtocol):
    """Typed teleport-outcome waiter returning a stable sample result."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        **kwargs: Unpack[TeleportOutcomeWaiterKwargs],
    ) -> TeleportAttemptResultDict:
        """Return one stable teleport result for typed helper calls."""
        _ = (page, provider, kwargs)
        return _result(target)
