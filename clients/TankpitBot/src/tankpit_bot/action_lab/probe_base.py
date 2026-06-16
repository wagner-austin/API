"""Shared probe infrastructure — commands, world state, action tracking.

Provides the base class for all live action lab probes. Each specific
probe (TeleportProbe, FuelProbe, EquipmentProbe, MovementProbe,
EnemyTeleportProbe) inherits from ProbeBase instead of from each other.

ProbeBase owns:
- CommandService for wire dispatch
- CDP message buffer for world sync
- Action phase cycle tracking
- World state and self state access
- Browser lifecycle hooks (magic capture, message buffering)
"""

from __future__ import annotations

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace import ActionCycleTracker, log_phase_overlaps
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseName,
    ActionPhaseOverlapDict,
)
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser import BrowserSession
from tankpit_bot.sniffer.world_state import get_world_state
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage


class ProbeError(Exception):
    """Raised when a probe cannot proceed."""


class ProbeBase(BrowserSession):
    """Shared infrastructure for all live action lab probes.

    Provides command dispatch via CommandService, world state access,
    CDP message buffering, and action phase tracking. Specific probes
    add their own execute() methods and result builders.
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
    ) -> None:
        """Initialize the probe base.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Whether to prefer account login.
        """
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
        )
        self._commands = CommandService(send_ws_bytes=self._send_websocket_bytes)
        self._cdp_message_buffer: list[str] = []
        self._action_cycle_tracker = ActionCycleTracker()
        self._attempt_phase_overlaps: list[ActionPhaseOverlapDict] = []

    # -----------------------------------------------------------------
    # Command dispatch
    # -----------------------------------------------------------------

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        """XOR encode and send command bytes via WebSocket.

        Subclasses (replay harnesses) override this to capture dispatched
        commands without hitting the wire.

        Args:
            data: Framed command bytes.
            cmd_name: Command name for logging.

        Returns:
            True if sent, False if CDP unavailable.
        """
        self._commands.cdp = self._cdp
        return self._commands.send_bytes(data, cmd_name)

    def open_map(self) -> bool:
        """Send map open command.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_MAP_OPEN, build_query_command

        return self._send_bytes(build_query_command(CMD_MAP_OPEN), "map_open")

    def teleport_to(self, x: int, y: int) -> bool:
        """Send teleport command. Map must already be open.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.bot.commands import encode_teleport_command
        from tankpit_bot.bot.types import make_teleport_command

        if self._cdp is None:
            return False
        cmd = make_teleport_command(x, y)
        return self._send_bytes(encode_teleport_command(cmd), f"teleport({x},{y})")

    def move_to(self, x: int, y: int) -> bool:
        """Send move command.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.bot.commands import encode_move_command
        from tankpit_bot.bot.types import make_move_command

        cmd = make_move_command(x, y)
        return self._send_bytes(encode_move_command(cmd), "move")

    def use_radar(self) -> bool:
        """Send radar scan command.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_RADAR, build_query_command

        return self._send_bytes(build_query_command(CMD_RADAR), "radar")

    def request_nearest_enemy(self) -> bool:
        """Send nearest enemy query command.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_NEAREST_ENEMY, build_query_command

        return self._send_bytes(build_query_command(CMD_NEAREST_ENEMY), "nearest_enemy")

    def shoot(self, x: int, y: int, target_id: int = 0) -> bool:
        """Send shoot command.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            target_id: Target entity ID (0 if no specific target).

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import build_shoot_command

        return self._send_bytes(
            build_shoot_command(x, y, target_id),
            f"shoot({x},{y},id={target_id})",
        )

    def pickup_fuel(self, x: int, y: int) -> bool:
        """Send fuel pickup command.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.bot.commands import encode_pickup_fuel_command
        from tankpit_bot.bot.types import make_pickup_fuel_command

        cmd = make_pickup_fuel_command(x, y)
        return self._send_bytes(encode_pickup_fuel_command(cmd), "pickup_fuel")

    def pickup_equipment(self, x: int, y: int) -> bool:
        """Send equipment pickup command.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.bot.commands import encode_pickup_equipment_command
        from tankpit_bot.bot.types import make_pickup_equipment_command

        cmd = make_pickup_equipment_command(x, y)
        return self._send_bytes(encode_pickup_equipment_command(cmd), "pickup_equipment")

    # -----------------------------------------------------------------
    # Lifecycle hooks
    # -----------------------------------------------------------------

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Buffer received messages for probe sync.

        Args:
            message: The captured message.
        """
        super()._on_message_captured(message)
        if message["direction"] == "received":
            self._cdp_message_buffer.append(message["payload"])

    def _on_magic_captured(self, magic: str) -> None:
        """Build XOR table and init trackers when magic key is captured.

        Args:
            magic: The session magic string.
        """
        from tankpit_bot.protocol.codec import (
            DEFAULT_STATIC_KEY_PATH,
            build_xor_table,
            load_static_key,
        )
        from tankpit_bot.sniffer.trackers import init_trackers_with_magic

        init_trackers_with_magic(magic)
        static_key = load_static_key(DEFAULT_STATIC_KEY_PATH)
        self._commands.xor_table = build_xor_table(static_key, magic)

    # -----------------------------------------------------------------
    # World state access
    # -----------------------------------------------------------------

    def get_world_state(self) -> WorldStateDict:
        """Get current world state.

        Returns:
            Current WorldStateDict.
        """
        return get_world_state()

    def get_state(self) -> str:
        """Get current probe state.

        Returns:
            Always "IDLE" — probes do not use the Bot HFSM.
        """
        return "IDLE"

    def _update_state_from_world(self) -> None:
        """Update state from world data. No-op for probes."""

    def get_self_state(self) -> SelfStateDict | None:
        """Get self tank state.

        Returns:
            SelfStateDict if available, None if not yet tracked.
        """
        return get_world_state()["self_state"]

    # -----------------------------------------------------------------
    # Action phase tracking
    # -----------------------------------------------------------------

    def _reset_action_cycle_tracker(self) -> None:
        """Reset action phase tracking for a new live session."""
        self._action_cycle_tracker.reset()
        self._attempt_phase_overlaps = []

    def _reset_attempt_phase_overlaps(self) -> None:
        """Reset overlap diagnostics for a new attempt."""
        self._attempt_phase_overlaps = []

    def _get_attempt_phase_overlaps(self) -> list[ActionPhaseOverlapDict]:
        """Return a copy of current overlap diagnostics for the active attempt."""
        return list(self._attempt_phase_overlaps)

    def _start_action_phase(
        self,
        phase: ActionPhaseName,
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Start a traced action phase and log invariant overlaps.

        Args:
            phase: Phase being started.
            attempt_label: Human-readable attempt label.

        Returns:
            Started phase cycle.
        """
        cycle, overlaps = self._action_cycle_tracker.begin_phase(
            phase,
            started_ms=action_hooks.get_current_time_ms(),
        )
        self._attempt_phase_overlaps.extend(overlaps)
        log_phase_overlaps(overlaps, attempt_label=attempt_label)
        return cycle

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """End a traced action phase.

        Args:
            cycle: Cycle to close.
        """
        self._action_cycle_tracker.end_phase(cycle)

    def _require_self_state(self) -> SelfStateDict:
        """Return the current self state or raise when absent.

        Returns:
            Current self tank state.

        Raises:
            ProbeError: If self state is not yet available.
        """
        self_state = self.get_self_state()
        if self_state is None:
            raise ProbeError("self state is unavailable")
        return self_state

    def _require_page(self) -> action_session.WaitPageProtocol:
        """Return the current Playwright page or raise when absent.

        Returns:
            Current page handle.

        Raises:
            ProbeError: If the page has not been initialized.
        """
        if self._page is None:
            raise ProbeError("page is unavailable")
        return self._page

    def _clear_in_flight_action(self) -> None:
        """Clear any pending action record between phases."""

    def _reset_probe_state_to_idle(self) -> None:
        """Reset the probe to idle state between attempts."""


__all__ = [
    "ProbeBase",
    "ProbeError",
]
