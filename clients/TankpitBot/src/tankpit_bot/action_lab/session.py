"""Shared session-synchronization helpers for live action probes."""

from __future__ import annotations

from typing import Literal, Protocol

from platform_core.json_utils import JSONValue, require_dict

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.types import (
    TeleportPageSnapshotDict,
    decode_teleport_page_snapshot,
)
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage

_POLL_INTERVAL_MS = 100.0


class ActionLabSessionError(Exception):
    """Raised when a live action probe session is not yet ready."""


class WaitPageProtocol(Protocol):
    """Minimal page protocol for action-lab wait loops."""

    def wait_for_timeout(self, timeout: float) -> None:
        """Block for the requested number of milliseconds.

        Args:
            timeout: Milliseconds to wait.
        """
        ...


class WorldStateProviderProtocol(Protocol):
    """Minimal world-state provider for action-lab wait loops."""

    def get_world_state(self) -> WorldStateDict:
        """Return the latest world-state snapshot.

        Returns:
            Current world-state snapshot.
        """
        ...


class BufferedWorldStateProviderProtocol(WorldStateProviderProtocol, Protocol):
    """Minimal world-state provider that also buffers raw protocol messages."""

    _cdp_message_buffer: list[str]

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return captured protocol messages for the current session.

        Returns:
            Captured sent and received protocol messages.
        """
        ...

    @property
    def magic(self) -> str | None:
        """Return the captured session magic key, if available.

        Returns:
            Session magic key for protocol decoding, or None when unavailable.
        """
        ...


class StartupStateDriverProtocol(Protocol):
    """Minimal startup-state driver for a live action probe bot."""

    def get_state(self) -> str:
        """Return the current state-machine state name.

        Returns:
            Current state name.
        """
        ...

    def _update_state_from_world(self) -> None:
        """Advance the state machine from the current world snapshot."""
        ...


def _extract_runtime_value(result: dict[str, JSONValue]) -> JSONValue:
    """Return the ``Runtime.evaluate`` value field.

    Args:
        result: Raw CDP result object.

    Returns:
        Evaluated JavaScript value.

    Raises:
        ValueError: If the CDP result is missing the value field.
    """
    result_obj = require_dict(result, "result")
    if "value" not in result_obj:
        raise ValueError(f"Runtime.evaluate result missing value: {result_obj}")
    return result_obj["value"]


def capture_teleport_page_snapshot(
    cdp: CDPSessionProtocol,
    phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
) -> TeleportPageSnapshotDict:
    """Capture the current page-client state relevant to teleport investigation.

    Args:
        cdp: Active CDP session.
        phase: Attempt phase associated with this snapshot.

    Returns:
        Validated page-client snapshot.
    """
    expression = """
    (() => {
        const activeGame =
            window.__tankpitActiveGame && typeof window.__tankpitActiveGame === 'object'
                ? window.__tankpitActiveGame
                : null;
        const mapObject =
            activeGame && activeGame.map && typeof activeGame.map === 'object'
                ? activeGame.map
                : null;
        const actionSource =
            activeGame &&
            activeGame.h &&
            typeof activeGame.h === 'object' &&
            activeGame.h.j &&
            typeof activeGame.h.j === 'object'
                ? activeGame.h.j
                : null;
        const actions =
            actionSource && Array.isArray(actionSource.actions)
                ? actionSource.actions
                : null;
        const heartbeatSource =
            activeGame && activeGame.va && typeof activeGame.va === 'object'
                ? activeGame.va
                : null;
        const lastHeartbeat =
            heartbeatSource && typeof heartbeatSource.j === 'number'
                ? heartbeatSource.j
                : null;
        const lastPageClientSend =
            typeof window.__lastPageClientSendPerfMs === 'number'
                ? window.__lastPageClientSendPerfMs
                : null;
        const lastBotSend =
            typeof window.__lastBotInjectedSendPerfMs === 'number'
                ? window.__lastBotInjectedSendPerfMs
                : null;
        const ws =
            window.__capturedWS instanceof WebSocket
                ? window.__capturedWS
                : null;
        const queue =
            Array.isArray(window.__sentFrameMetaQueue)
                ? window.__sentFrameMetaQueue
                : [];
        const now = performance.now();
        return {
            phase: %PHASE%,
            timestamp_ms: Date.now(),
            client_present: activeGame !== null,
            map_visible: mapObject === null ? null : !!mapObject.h,
            client_state:
                activeGame !== null && typeof activeGame.s === 'number'
                    ? activeGame.s
                    : null,
            client_busy:
                activeGame !== null && typeof activeGame.Ha === 'boolean'
                    ? activeGame.Ha
                    : null,
            pending_actions: actions === null ? null : actions.length,
            heartbeat_age_ms:
                lastHeartbeat === null ? null : Math.max(0, Math.floor(now - lastHeartbeat)),
            last_page_client_send_age_ms:
                lastPageClientSend === null
                    ? null
                    : Math.max(0, Math.floor(now - lastPageClientSend)),
            last_bot_send_age_ms:
                lastBotSend === null
                    ? null
                    : Math.max(0, Math.floor(now - lastBotSend)),
            ws_ready_state: ws === null ? null : ws.readyState,
            current_send_label:
                typeof window.__codexCurrentSendLabel === 'string'
                    ? window.__codexCurrentSendLabel
                    : null,
            sent_frame_meta_queue_length: queue.length
        };
    })()
    """
    result = cdp.send(
        "Runtime.evaluate",
        {
            "expression": expression.replace("%PHASE%", repr(phase)),
            "returnByValue": True,
        },
    )
    raw_value = _extract_runtime_value(result)
    return decode_teleport_page_snapshot(require_dict({"snapshot": raw_value}, "snapshot"))


def wait_for_world_sync(
    page: WaitPageProtocol,
    provider: BufferedWorldStateProviderProtocol,
    started_ms: int,
    timeout_ms: int,
) -> int | None:
    """Wait for the first world-state sync newer than ``started_ms``.

    Args:
        page: Page-like object used for short waits.
        provider: World-state provider to poll.
        started_ms: Lower bound timestamp for a fresh sync.
        timeout_ms: Maximum time to wait.

    Returns:
        The fresh world-state timestamp, or None on timeout.
    """
    while action_hooks.get_current_time_ms() - started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(provider)
        world = provider.get_world_state()
        if world["timestamp_ms"] > started_ms:
            return world["timestamp_ms"]
        page.wait_for_timeout(_POLL_INTERVAL_MS)
    return None


def wait_for_radar_sync(
    page: WaitPageProtocol,
    provider: BufferedWorldStateProviderProtocol,
    started_ms: int,
    timeout_ms: int,
) -> int | None:
    """Wait for radar completion after a radar command.

    Args:
        page: Page-like object used for short waits.
        provider: World-state provider to poll.
        started_ms: Lower bound timestamp for the wait window.
        timeout_ms: Maximum time to wait.

    Returns:
        Local completion timestamp when radar results have been applied, or
        None on timeout.
    """
    observed_post_start_activity = False
    radar_completion_seen = False
    while action_hooks.get_current_time_ms() - started_ms < timeout_ms:
        drained_count = action_hooks.drain_buffered_messages(provider)
        if drained_count > 0 or provider.get_world_state()["timestamp_ms"] > started_ms:
            observed_post_start_activity = True
        if action_hooks.check_and_clear_radar_scan_complete():
            radar_completion_seen = True
        if observed_post_start_activity and radar_completion_seen:
            return action_hooks.get_current_time_ms()
        page.wait_for_timeout(_POLL_INTERVAL_MS)
    return None


def wait_for_initial_self_state(
    page: WaitPageProtocol,
    provider: BufferedWorldStateProviderProtocol,
    started_ms: int,
    timeout_ms: int,
) -> tuple[int, SelfStateDict]:
    """Wait for an initial self state from a fresh world sync.

    Args:
        page: Page-like object used for short waits.
        provider: World-state provider to poll.
        started_ms: Timestamp immediately before readiness polling begins.
        timeout_ms: Maximum time to wait for an initial self state.

    Returns:
        A tuple of the fresh world timestamp and the current self state.

    Raises:
        ActionLabSessionError: If a fresh self state does not arrive in time.
    """
    while action_hooks.get_current_time_ms() - started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(provider)
        world = provider.get_world_state()
        self_state = world["self_state"]
        if world["timestamp_ms"] > started_ms and self_state is not None:
            return world["timestamp_ms"], self_state
        page.wait_for_timeout(_POLL_INTERVAL_MS)
    raise ActionLabSessionError("initial self state is unavailable after initial sync wait")


def advance_startup_state(bot: StartupStateDriverProtocol) -> None:
    """Advance a bot from startup bootstrap states to an executable state.

    The real bot reaches ``IDLE`` over successive tick-loop sync passes:
    ``INITIALIZING -> WAITING_FOR_POSITION -> IDLE``. Action-lab probes do
    not run that tick loop, so they must explicitly drive those startup
    transitions after initial world/self-state sync has completed.

    Args:
        bot: Bot-like object with startup state transitions.

    Raises:
        ActionLabSessionError: If the bot remains in a bootstrap state after
            the expected startup transitions.
    """
    while bot.get_state() in ("INITIALIZING", "WAITING_FOR_POSITION"):
        previous_state = bot.get_state()
        bot._update_state_from_world()
        if bot.get_state() == previous_state:
            raise ActionLabSessionError(f"startup state did not advance from {previous_state}")


__all__ = [
    "ActionLabSessionError",
    "BufferedWorldStateProviderProtocol",
    "StartupStateDriverProtocol",
    "WaitPageProtocol",
    "WorldStateProviderProtocol",
    "advance_startup_state",
    "capture_teleport_page_snapshot",
    "wait_for_initial_self_state",
    "wait_for_radar_sync",
    "wait_for_world_sync",
]
