"""Shared probe infrastructure — commands, world state, action tracking.

Provides the base class for all live action lab probes. Each specific
probe (TeleportProbe, FuelProbe, EquipmentProbe, MovementProbe,
EnemyTeleportProbe) inherits from ProbeBase.

ProbeBase inherits CDPService + CommandService composition from
SessionBase. Adds action tracking and probe-specific command methods.
Browser lifecycle is handled by standalone functions called through
action_hooks.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import PageProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace import ActionCycleTracker, log_phase_overlaps
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseName,
    ActionPhaseOverlapDict,
)
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.browser.session_base import SessionBase
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage


class ProbeError(Exception):
    """Raised when a probe cannot proceed."""


class ProbeBase(SessionBase):
    """Shared infrastructure for all live action lab probes.

    Inherits CDPService + CommandService composition from SessionBase.
    Adds action tracking, probe-specific commands, and convenience
    properties for captured messages and magic key.
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
        world: WorldService | None = None,
    ) -> None:
        """Initialize the probe base.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Whether to prefer account login.
            cdp_service: Injected CDPService. Created internally if None.
            command_service: Injected CommandService. Created internally if None.
            world: Injected WorldService. Created internally if None.
        """
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            cdp_service=cdp_service,
            command_service=command_service,
            world=world,
        )
        self._page: PageProtocol | None = None
        self._action_cycle_tracker = ActionCycleTracker()
        self._attempt_phase_overlaps: list[ActionPhaseOverlapDict] = []

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    @property
    def messages(self) -> list[CapturedMessage]:
        """Get captured messages."""
        return self._cdp_service.messages

    @property
    def magic(self) -> str | None:
        """Get captured magic key."""
        return self._cdp_service.magic

    def open_map(self) -> bool:
        """Send map open command.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_MAP_OPEN, build_query_command

        return self._send_bytes(build_query_command(CMD_MAP_OPEN), "map_open")

    def quit_to_lobby(self) -> bool:
        """Send the graceful quit so the tank never lingers in-world.

        Standing rule from the 2026-07-25 incident: an unattended
        probe tank is a target in a PvP world (an immobilized tank
        was killed and the account lost a rank). Every probe end —
        normal or aborted — exits the room deliberately instead of
        leaving the tank standing until the socket drops.

        Returns:
            True if the command was sent.
        """
        from tankpit_bot.protocol.commands import build_quit_command

        return self._send_bytes(build_quit_command(), "quit_game")

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

    def request_inventory(self) -> bool:
        """Send the inventory request — the cheapest game action.

        Two bytes, free, no world effect: the watch-dwell heartbeat
        uses it to hold the push-on-activity stream open (wiki log
        2026-07-24).

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_INVENTORY, build_query_command

        return self._send_bytes(build_query_command(CMD_INVENTORY), "inventory")

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
    # World state access
    # -----------------------------------------------------------------

    def get_world_state(self) -> WorldStateDict:
        """Get current world state.

        Returns:
            Current WorldStateDict.
        """
        return self.world.get_world_state()

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
        return self.world.get_world_state()["self_state"]

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
