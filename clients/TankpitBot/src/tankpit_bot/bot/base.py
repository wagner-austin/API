"""Bot base class for TankPit automation.

This module provides the Bot class that extends WebSocketSniffer with:
- State machine for behavior control
- Command sending capabilities (move, shoot, radar, etc.)
- Fuel/HP tracking from world state
- Convenience methods for finding containers
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.bot.commands import (
    encode_move_command,
    encode_pickup_move_command,
    encode_teleport_command,
)
from tankpit_bot.bot.states import (
    BotStateDataDict,
    StateName,
    make_initial_state_data,
    transition_to,
    validate_transition,
)
from tankpit_bot.bot.types import (
    make_move_command,
    make_pickup_move_command,
    make_teleport_command,
)
from tankpit_bot.bot.vision import (
    VisionStateDict,
    get_merged_fuel,
    get_merged_fuel_containers,
    make_empty_vision_state,
    render_vision_ascii,
    render_vision_debug,
)
from tankpit_bot.browser import BrowserSession, get_current_time_ms
from tankpit_bot.protocol.commands import (
    CMD_MAP_OPEN,
    CMD_RADAR,
    build_query_command,
    build_shoot_command,
    build_toggle_equipment_command,
)
from tankpit_bot.sniffer.world_state import get_world_state
from tankpit_bot.state import ContainerStateDict, SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)


class BotError(Exception):
    """Base exception for bot errors.

    All bot-specific exceptions inherit from this class,
    allowing callers to catch all bot errors with a single except clause.
    """


class ProtocolNotDiscoveredError(BotError):
    """Raised when the XOR protocol keys have not been discovered.

    The bot requires the magic key and static key to be discovered
    before it can send commands. This error is raised when attempting
    to send commands without valid protocol keys.
    """


class Bot(BrowserSession):
    """Bot that can send commands and track game state with state machine.

    Extends BrowserSession with:
    - State machine for behavior control
    - Command sending via WebSocket
    - Fuel/HP tracking from decoded messages
    - Convenience methods for game actions

    Attributes:
        _cdp: CDP session for sending commands (set during run).
        _state_data: Current state machine data.
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
    ) -> None:
        """Initialize the bot.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Skip guest login and use account credentials.
        """
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
        )
        self._cdp: _test_hooks.CDPSessionProtocol | None = None
        self._page: _test_hooks.PageProtocol | None = None
        self._state_data: BotStateDataDict = make_initial_state_data()
        # Equipment state: [armor, dual, missile, homing, radar] - 0=off, 1=on
        self._equipment_enabled: list[bool] = [False, False, False, False, False]
        # Map state
        self._map_is_open: bool = False
        # Vision state for fallback tracking
        self._vision_state: VisionStateDict = make_empty_vision_state()

    # =========================================================================
    # State Machine
    # =========================================================================

    def get_state(self) -> StateName:
        """Get current bot state.

        Returns:
            Current state name.
        """
        return self._state_data["state"]

    def get_state_data(self) -> BotStateDataDict:
        """Get full state data (immutable copy).

        Returns:
            Current BotStateDataDict.
        """
        return self._state_data

    def _transition(
        self,
        new_state: StateName,
        *,
        target_x: int | None = None,
        target_y: int | None = None,
        scan_pending: bool | None = None,
    ) -> None:
        """Transition to a new state with validation.

        Args:
            new_state: State to transition to.
            target_x: Optional target X coordinate.
            target_y: Optional target Y coordinate.
            scan_pending: Optional scan pending flag.

        Raises:
            ValueError: If transition is invalid.
        """
        current_state = self._state_data["state"]
        validate_transition(current_state, new_state)

        self._state_data = transition_to(
            self._state_data,
            new_state,
            target_x=target_x,
            target_y=target_y,
            scan_pending=scan_pending,
            last_action_ms=get_current_time_ms(),
        )
        log.info("State: %s -> %s", current_state, new_state)

    # =========================================================================
    # Message Handling
    # =========================================================================

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Handle captured WebSocket message.

        Extends parent to update state machine based on game events.

        Args:
            message: The captured message.
        """
        # Call parent for decoding and state updates
        super()._on_message_captured(message)

        # Update state machine based on world state changes
        self._update_state_from_world()

    def _update_state_from_world(self) -> None:
        """Update state machine based on current world state."""
        current_state = self._state_data["state"]
        world = self.get_world_state()
        self_state = world["self_state"]

        # Handle INITIALIZING -> WAITING_FOR_POSITION when magic is received
        if current_state == "INITIALIZING" and self._magic is not None:
            self._transition("WAITING_FOR_POSITION")
            return

        # Handle WAITING_FOR_POSITION -> IDLE when position is known
        if current_state == "WAITING_FOR_POSITION" and self_state is not None:
            self._transition("IDLE")
            return

        # Check for LOW_FUEL condition
        excluded_states = ("LOW_FUEL", "INITIALIZING", "WAITING_FOR_POSITION", "DISCONNECTED")
        is_low_fuel = (
            self_state is not None
            and current_state not in excluded_states
            and self_state["fuel"] < self._state_data["fuel_threshold"]
        )
        if is_low_fuel:
            self._transition("LOW_FUEL")
            return

        # Handle SCANNING -> IDLE when scan completes
        scan_complete = (
            current_state == "SCANNING"
            and self._state_data["scan_pending"]
            and len(world["containers"]) > 0
        )
        if scan_complete:
            self._transition("IDLE", scan_pending=False)
            return

        # Handle MOVING/COLLECTING completion (reached target)
        if current_state in ("MOVING", "COLLECTING") and self_state is not None:
            tx, ty = self._state_data["target_x"], self._state_data["target_y"]
            if self_state["x"] == tx and self_state["y"] == ty:
                self._transition("IDLE")
                return

    # =========================================================================
    # State Access
    # =========================================================================

    def get_world_state(self) -> WorldStateDict:
        """Get current world state.

        Returns:
            Current WorldStateDict with all tracked entities.
        """
        return get_world_state()

    def get_self_state(self) -> SelfStateDict | None:
        """Get self tank state (position, fuel, etc.).

        Returns:
            SelfStateDict if available, None if not yet tracked.
        """
        return get_world_state()["self_state"]

    def get_fuel(self) -> int:
        """Get current fuel (HP).

        Returns:
            Current fuel amount, or 0 if self_state not yet tracked.
        """
        state = self.get_self_state()
        return state["fuel"] if state is not None else 0

    def get_position(self) -> tuple[int, int] | None:
        """Get current position.

        Returns:
            Tuple of (x, y) coordinates, or None if not yet tracked.
        """
        state = self.get_self_state()
        if state is None:
            return None
        return (state["x"], state["y"])

    def get_containers(self) -> dict[str, ContainerStateDict]:
        """Get all known containers.

        Returns:
            Dict of container key ("x,y") to ContainerStateDict.
        """
        return get_world_state()["containers"]

    def get_fuel_containers(self) -> list[ContainerStateDict]:
        """Get all known fuel containers (not equipment).

        Returns:
            List of fuel containers with volume > 0.
        """
        containers = self.get_containers()
        return [c for c in containers.values() if c["is_fuel"] and c["volume"] > 0]

    def get_nearest_fuel_container(self) -> ContainerStateDict | None:
        """Get nearest fuel container to current position.

        Returns:
            Nearest ContainerStateDict, or None if no containers or no position.
        """
        pos = self.get_position()
        if pos is None:
            return None

        fuel_containers = self.get_fuel_containers()
        if not fuel_containers:
            return None

        # Sort by Manhattan distance
        my_x, my_y = pos
        fuel_containers.sort(key=lambda c: abs(c["x"] - my_x) + abs(c["y"] - my_y))
        return fuel_containers[0]

    # =========================================================================
    # Vision (Multi-Perspective Tracking)
    # =========================================================================

    def get_vision_state(self) -> VisionStateDict:
        """Get current vision state (fallback caches).

        Returns:
            VisionStateDict with tank registry, position cache, containers.
        """
        return self._vision_state

    def get_all_fuel_containers(self) -> list[ContainerStateDict]:
        """Get fuel containers from both world state and vision cache.

        Merges containers from both sources for more complete coverage.
        Prefers world state when both have same location.

        Returns:
            List of fuel containers with volume > 0.
        """
        return get_merged_fuel_containers(self._vision_state)

    def get_all_fuel(self) -> int:
        """Get fuel from world state, falling back to vision cache.

        Uses world state when available, vision cache as fallback.

        Returns:
            Current fuel amount.
        """
        return get_merged_fuel(self._vision_state)

    def render_ascii(self) -> str | None:
        """Render current world state as ASCII viewport.

        Returns:
            Multi-line ASCII string showing visible area, or None if
            terrain map not loaded.
        """
        return render_vision_ascii()

    def render_debug(self) -> str:
        """Render vision debug info.

        Returns:
            Multi-line debug string with cache stats and comparison.
        """
        return render_vision_debug(self._vision_state)

    def get_nearest_all_fuel_container(self) -> ContainerStateDict | None:
        """Get nearest fuel container using merged sources.

        Uses both world state and vision cache for more complete coverage.

        Returns:
            Nearest ContainerStateDict, or None if none found.
        """
        pos = self.get_position()
        if pos is None:
            return None

        fuel_containers = self.get_all_fuel_containers()
        if not fuel_containers:
            return None

        my_x, my_y = pos
        fuel_containers.sort(key=lambda c: abs(c["x"] - my_x) + abs(c["y"] - my_y))
        return fuel_containers[0]

    # =========================================================================
    # Command Sending
    # =========================================================================

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        """Send raw bytes via WebSocket.

        Args:
            data: Encoded command bytes to send.
            cmd_name: Command name for logging.

        Returns:
            True if sent, False if CDP session not available.
        """
        if self._cdp is None:
            log.warning("Cannot send %s: CDP session not available", cmd_name)
            return False
        self._send_websocket_bytes(self._cdp, data)
        log.info("Sent: %s", cmd_name)
        return True

    def move_to(self, x: int, y: int) -> bool:
        """Send move command and transition to MOVING state.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        cmd = make_move_command(x, y)
        if not self._send_bytes(encode_move_command(cmd), "move"):
            return False
        self._transition("MOVING", target_x=x, target_y=y)
        return True

    def pickup_move_to(self, x: int, y: int) -> bool:
        """Send pickup move command and transition to COLLECTING state.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        cmd = make_pickup_move_command(x, y)
        if not self._send_bytes(encode_pickup_move_command(cmd), "pickup_move"):
            return False
        self._transition("COLLECTING", target_x=x, target_y=y)
        return True

    def teleport_to(self, x: int, y: int) -> bool:
        """Send teleport command and transition to MOVING state.

        Opens map if needed, sends teleport, closes map.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        # Map must be open to teleport
        if not self.open_map():
            return False

        # Wait for map to open (if we have page reference)
        if self._page is not None:
            self._page.wait_for_timeout(200)

        # Send teleport command
        cmd = make_teleport_command(x, y)
        if not self._send_bytes(encode_teleport_command(cmd), f"teleport({x},{y})"):
            return False

        # Wait for teleport to process
        if self._page is not None:
            self._page.wait_for_timeout(500)

        # Close map
        self.close_map()

        self._transition("MOVING", target_x=x, target_y=y)
        return True

    def shoot_at(self, x: int, y: int, target_id: int = 0) -> bool:
        """Send shoot command and transition to COMBAT state.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            target_id: Target entity ID (0 if no specific target).

        Returns:
            True if command was sent.
        """
        encoded = build_shoot_command(x, y, target_id)
        if not self._send_bytes(encoded, f"shoot({x},{y},id={target_id})"):
            return False
        if self.get_state() != "COMBAT":
            self._transition("COMBAT")
        return True

    def use_radar(self) -> bool:
        """Send radar scan command and transition to SCANNING state.

        Returns:
            True if command was sent.
        """
        encoded = build_query_command(CMD_RADAR)
        if not self._send_bytes(encoded, "radar"):
            return False
        self._transition("SCANNING", scan_pending=True)
        return True

    # =========================================================================
    # Equipment Management
    # =========================================================================

    def toggle_equipment(self, slot: int) -> bool:
        """Toggle equipment slot.

        Args:
            slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile, 4=homing, 5=radar.

        Returns:
            True if command was sent.
        """
        if slot < 1 or slot > 5:
            log.warning("Invalid equipment slot: %d (must be 1-5)", slot)
            return False
        encoded = build_toggle_equipment_command(slot)
        slot_names = ["armor", "dual", "missile", "homing", "radar"]
        return self._send_bytes(encoded, f"toggle_{slot_names[slot - 1]}")

    def enable_equipment(self, slot: int) -> bool:
        """Enable equipment slot if not already enabled.

        Args:
            slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile, 4=homing, 5=radar.

        Returns:
            True if equipment is now enabled (or was already enabled).
        """
        if slot < 1 or slot > 5:
            log.warning("Invalid equipment slot: %d (must be 1-5)", slot)
            return False
        slot_idx = slot - 1
        if self._equipment_enabled[slot_idx]:
            return True
        return self.toggle_equipment(slot)

    def enable_homing(self) -> bool:
        """Enable homing shot equipment (slot 4)."""
        return self.enable_equipment(4)

    def enable_dual(self) -> bool:
        """Enable dual shot equipment (slot 2)."""
        return self.enable_equipment(2)

    def enable_radar_equipment(self) -> bool:
        """Enable radar equipment (slot 5)."""
        return self.enable_equipment(5)

    def is_equipment_enabled(self, slot: int) -> bool:
        """Check if equipment slot is enabled.

        Args:
            slot: Equipment slot (1-5).

        Returns:
            True if enabled.
        """
        if slot < 1 or slot > 5:
            return False
        return self._equipment_enabled[slot - 1]

    def update_equipment_state(self, states: list[bool]) -> None:
        """Update equipment state from server response.

        Called when receiving equipment toggle response (0x74).

        Args:
            states: List of 5 booleans [armor, dual, missile, homing, radar].
        """
        if len(states) == 5:
            self._equipment_enabled = states.copy()

    # =========================================================================
    # Map Operations
    # =========================================================================

    def open_map(self) -> bool:
        """Open the map view.

        Returns:
            True if command was sent.
        """
        if self._map_is_open:
            return True
        encoded = build_query_command(CMD_MAP_OPEN)
        if self._send_bytes(encoded, "map_open"):
            self._map_is_open = True
            return True
        return False

    def close_map(self) -> bool:
        """Close the map view (toggle).

        Returns:
            True if command was sent.
        """
        if not self._map_is_open:
            return True
        encoded = build_query_command(CMD_MAP_OPEN)
        if self._send_bytes(encoded, "map_close"):
            self._map_is_open = False
            return True
        return False

    # =========================================================================
    # High-Level Actions
    # =========================================================================

    def go_to_nearest_fuel(self) -> bool:
        """Move to the nearest fuel container.

        Returns:
            True if movement command was sent, False if no fuel found.
        """
        container = self.get_nearest_fuel_container()
        if container is None:
            log.info("No fuel containers found")
            return False

        log.info(
            "Moving to fuel at (%d, %d) with %d fuel",
            container["x"],
            container["y"],
            container["volume"],
        )
        return self.pickup_move_to(container["x"], container["y"])

    def scan_and_collect_fuel(self) -> bool:
        """Scan for containers and move to nearest fuel.

        Convenience method that scans first if no containers known.

        Returns:
            True if an action was taken.
        """
        if not self.get_fuel_containers():
            log.info("No fuel containers known, scanning...")
            return self.use_radar()
        return self.go_to_nearest_fuel()

    # =========================================================================
    # Run Loop
    # =========================================================================

    def run(self) -> None:
        """Run the bot.

        Launches browser, logs in, joins game, and runs the game loop.
        The bot will scan for fuel and collect it automatically.

        Raises:
            RuntimeError: If Playwright is not installed.
        """
        from tankpit_bot.browser import PlaywrightNotInstalledError, reset_cdp_time_offset
        from tankpit_bot.sniffer.viewport import reset_viewport_tracking

        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError("Playwright is not installed.")

        self._start_timestamp_ms = get_current_time_ms()
        self._messages = []
        self._ws_urls = {}
        self._magic = None
        self._state_data = make_initial_state_data()
        self._vision_state = make_empty_vision_state()

        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self._headless)
            context = browser.new_context()
            page = context.new_page()
            cdp = context.new_cdp_session(page)

            # Store CDP session and page for command sending
            self._cdp = cdp
            self._page = page

            # Reset session state
            reset_cdp_time_offset()
            reset_viewport_tracking()

            # Set up handlers
            self._setup_console_listener(cdp)
            self._setup_cdp_handlers(cdp)

            # Navigate and login
            self._navigate_and_login(page, cdp, tank_name_prefix="Bot", auto_join_room=True)

            # Wait for game to be ready
            self._wait_for_game_ready(page)

            log.info("Bot started, entering game loop")
            log.info("State: %s", self.get_state())

            # Game loop
            try:
                self._game_loop(page)
            except KeyboardInterrupt:
                log.info("Bot interrupted by user")
            finally:
                self._cdp = None
                self._page = None
                self._cleanup(cdp, page, context, browser)

    def _game_loop(self, page: _test_hooks.PageProtocol) -> None:
        """Main game loop that takes actions based on state.

        Args:
            page: Playwright page for waiting.
        """
        loop_delay_ms = 500
        idle_action_delay_ms = 2000
        last_idle_action_ms = 0

        while True:
            state = self.get_state()
            now = get_current_time_ms()
            time_to_act = now - last_idle_action_ms > idle_action_delay_ms

            # Take action based on current state
            if state == "IDLE" and time_to_act:
                self._handle_idle_state()
                last_idle_action_ms = now
            elif state == "LOW_FUEL" and time_to_act:
                self._handle_low_fuel_state()
                last_idle_action_ms = now

            # Wait before next iteration
            page.wait_for_timeout(loop_delay_ms)

    def _handle_idle_state(self) -> None:
        """Handle IDLE state - scan or collect fuel."""
        fuel_containers = self.get_fuel_containers()
        if not fuel_containers:
            log.info("IDLE: No fuel known, scanning...")
            self.use_radar()
        else:
            log.info("IDLE: %d fuel containers known, moving to nearest", len(fuel_containers))
            self.go_to_nearest_fuel()

    def _handle_low_fuel_state(self) -> None:
        """Handle LOW_FUEL state - urgently find fuel."""
        fuel_containers = self.get_fuel_containers()
        if not fuel_containers:
            log.info("LOW_FUEL: No fuel known, scanning urgently...")
            self.use_radar()
        else:
            log.info("LOW_FUEL: Moving to nearest fuel!")
            self.go_to_nearest_fuel()


def main() -> None:
    """Entry point for tankpit-bot command."""
    from dotenv import load_dotenv
    from platform_core.logging import setup_rich_logging

    load_dotenv()
    setup_rich_logging(level="INFO")

    if _test_hooks.sync_playwright is None:
        _test_hooks.sync_playwright = _test_hooks.get_sync_playwright()

    target_url = _test_hooks.get_env("TANKPIT_URL") or "https://tankpit.com/"
    prefer_account_str = _test_hooks.get_env("TANKPIT_PREFER_ACCOUNT")
    prefer_account = prefer_account_str is not None and prefer_account_str.lower() in (
        "true",
        "1",
        "yes",
    )

    bot = Bot(target_url, headless=False, prefer_account=prefer_account)
    bot.run()


__all__ = [
    "Bot",
    "BotError",
    "ProtocolNotDiscoveredError",
    "main",
]
