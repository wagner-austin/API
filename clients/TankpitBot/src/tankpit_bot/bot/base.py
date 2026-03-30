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
from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state
from tankpit_bot.bot.commands import (
    encode_move_command,
    encode_pickup_move_command,
    encode_teleport_command,
)
from tankpit_bot.bot.states import (
    BotStateDataDict,
    InFlightActionDict,
    StateName,
    make_in_flight_action,
    make_initial_state_data,
    make_no_action,
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
    make_empty_vision_state,
)
from tankpit_bot.browser import BrowserSession, get_current_time_ms
from tankpit_bot.protocol.codec import (
    DEFAULT_STATIC_KEY_PATH,
    build_xor_table,
    load_static_key,
    xor_bytes,
)
from tankpit_bot.protocol.commands import (
    CMD_ENTER_GAME,
    CMD_MAP_OPEN,
    CMD_NEAREST_ENEMY,
    CMD_RADAR,
    COMMAND_PREFIX,
    build_query_command,
    build_shoot_command,
    build_toggle_equipment_command,
)
from tankpit_bot.sniffer.trackers import init_trackers_with_magic
from tankpit_bot.sniffer.world_state import (
    check_and_clear_radar_scan_complete,
    check_and_clear_teleport_landed,
    get_inventory_state,
    get_world_state,
)
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
        self._ai_state: AIStateDict = make_initial_ai_state()
        # XOR encoding table for outgoing commands
        self._xor_table: bytes | None = None
        # Map state
        # Legacy test/debug field only. The game does not expose a reliable
        # authoritative map-open flag, so bot behavior must not depend on it.
        self._map_is_open: bool = False
        # Vision state for fallback tracking
        self._vision_state: VisionStateDict = make_empty_vision_state()
        # CDP message buffer — received payloads for tick loop sync
        self._cdp_message_buffer: list[str] = []

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
        in_flight_action: InFlightActionDict | None = None,
    ) -> None:
        """Transition to a new state with validation.

        Args:
            new_state: State to transition to.
            in_flight_action: New action record. Pass
                make_in_flight_action() for states with an active
                command, or make_no_action() for IDLE. If None,
                inherits the current action.

        Raises:
            ValueError: If transition is invalid.
        """
        current_state = self._state_data["state"]
        validate_transition(current_state, new_state)

        self._state_data = transition_to(
            self._state_data,
            new_state,
            in_flight_action=in_flight_action,
        )
        log.info("State: %s -> %s", current_state, new_state)

    # =========================================================================
    # Message Handling
    # =========================================================================

    def _on_magic_captured(self, magic: str) -> None:
        """Initialize trackers and build XOR table when magic key is captured.

        Args:
            magic: The session magic string.
        """
        init_trackers_with_magic(magic)
        static_key = load_static_key(DEFAULT_STATIC_KEY_PATH)
        self._xor_table = build_xor_table(static_key, magic)
        log.info("Built XOR table for command encoding")

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Buffer received messages for tick loop sync phase.

        Extracts magic key (via base class) and buffers received payloads.
        The tick loop drains the buffer and decodes in batch.

        Args:
            message: The captured message.
        """
        super()._on_message_captured(message)
        if message["direction"] == "received":
            self._cdp_message_buffer.append(message["payload"])
            log.info("CDP_BUFFER: +1 (total=%d)", len(self._cdp_message_buffer))

    def _maybe_transition_from_initializing(self, self_state: SelfStateDict | None) -> bool:
        """Advance startup states when required bootstrap data is available.

        Args:
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        current_state = self._state_data["state"]
        if current_state == "INITIALIZING" and self._magic is not None:
            self._transition("WAITING_FOR_POSITION")
            return True
        if current_state == "WAITING_FOR_POSITION" and self_state is not None:
            self._transition("IDLE")
            return True
        return False

    def _maybe_transition_to_low_fuel(self, self_state: SelfStateDict | None) -> bool:
        """Enter LOW_FUEL when the tracked fuel total is below threshold.

        Args:
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        current_state = self._state_data["state"]
        excluded_states = (
            "LOW_FUEL",
            "INITIALIZING",
            "WAITING_FOR_POSITION",
            "DISCONNECTED",
            "TELEPORTING",
            "COLLECTING",
            "SCANNING",
        )
        is_low_fuel = (
            self_state is not None
            and current_state not in excluded_states
            and self_state["fuel"] < self._state_data["fuel_threshold"]
        )
        if is_low_fuel:
            self._transition("LOW_FUEL")
            return True
        return False

    def _maybe_complete_scan(self, world: WorldStateDict) -> bool:
        """Finish a pending scan when the server responds.

        Completion is keyed off the in-flight action record, not the
        state name. This means a scan completes correctly even if
        something else moved the state away from SCANNING while the
        radar was in flight.

        Args:
            world: Current world state snapshot.

        Returns:
            True if a transition was applied.
        """
        action = self._state_data["in_flight_action"]
        if action["kind"] != "scan" or action["outcome"] != "pending":
            return False
        if not check_and_clear_radar_scan_complete():
            return False
        self._transition("IDLE", in_flight_action=make_no_action())
        return True

    def _maybe_complete_walk(self, self_state: SelfStateDict | None) -> bool:
        """Finish MOVING once the tank reaches the exact walking target.

        Args:
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        if self._state_data["state"] != "MOVING" or self_state is None:
            return False
        action = self._state_data["in_flight_action"]
        tx, ty = action["target_x"], action["target_y"]
        if self_state["x"] == tx and self_state["y"] == ty:
            self._transition("IDLE", in_flight_action=make_no_action())
            return True
        return False

    def _maybe_complete_teleport(self, self_state: SelfStateDict | None) -> bool:
        """Finish TELEPORTING when the server confirms landing.

        Args:
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        if (
            self._state_data["state"] == "TELEPORTING"
            and self_state is not None
            and check_and_clear_teleport_landed()
        ):
            self._transition("IDLE", in_flight_action=make_no_action())
            return True
        return False

    def _maybe_complete_collection(
        self,
        world: WorldStateDict,
        self_state: SelfStateDict | None,
    ) -> bool:
        """Finish COLLECTING when the pickup target is reached or removed.

        Args:
            world: Current world state snapshot.
            self_state: Current self tank state, if known.

        Returns:
            True if a transition was applied.
        """
        if self._state_data["state"] != "COLLECTING" or self_state is None:
            return False
        action = self._state_data["in_flight_action"]
        tx, ty = action["target_x"], action["target_y"]
        target_key = f"{tx},{ty}"
        if (self_state["x"] == tx and self_state["y"] == ty) or target_key not in world[
            "containers"
        ]:
            self._transition("IDLE", in_flight_action=make_no_action())
            return True
        return False

    def _update_state_from_world(self) -> None:
        """Update state machine based on current world state.

        Order matters: in-flight action completions (teleport, walk,
        collection, scan) are checked BEFORE low-fuel transitions.
        Otherwise LOW_FUEL would stomp TELEPORTING/COLLECTING states
        and cause repeated command spam.
        """
        world = self.get_world_state()
        self_state = world["self_state"]

        if self._maybe_transition_from_initializing(self_state):
            return
        if self._maybe_complete_teleport(self_state):
            return
        if self._maybe_complete_walk(self_state):
            return
        if self._maybe_complete_scan(world):
            return
        if self._maybe_complete_collection(world, self_state):
            return
        self._maybe_transition_to_low_fuel(self_state)

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
    # Command Sending
    # =========================================================================

    def _xor_encode_command(self, data: bytes) -> bytes:
        """XOR encode a framed command for wire transmission.

        Commands are framed as: [len_lo, len_hi, '!', type, cmd_id, ...data]
        XOR encoding applies to everything after '!' (bytes at index 3+).

        Args:
            data: Framed command bytes (with 2-byte length header).

        Returns:
            XOR-encoded framed command bytes.
        """
        if self._xor_table is None or len(data) < 4:
            return data

        # data[0:2] = length header, data[2] = '!' prefix, data[3:] = type+cmd+payload
        header = data[:2]
        prefix = data[2:3]  # '!' byte stays as-is
        payload = data[3:]  # type + cmd_id + optional data

        encoded_payload = xor_bytes(self._xor_table, payload, offset=0)
        return header + prefix + encoded_payload

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        """XOR encode and send command bytes via WebSocket.

        Args:
            data: Framed command bytes (with 2-byte length header).
            cmd_name: Command name for logging.

        Returns:
            True if sent, False if CDP session not available.
        """
        if self._cdp is None:
            log.warning("Cannot send %s: CDP session not available", cmd_name)
            return False

        # XOR encode if it's a '!' command
        if len(data) > 2 and data[2] == COMMAND_PREFIX:
            data = self._xor_encode_command(data)

        self._send_websocket_bytes(self._cdp, data)
        log.info("Sent: %s", cmd_name)
        return True

    def enter_game(self) -> bool:
        """Send CMD_ENTER_GAME to activate the tank in the game world.

        Must be sent after joining a room and before any movement or combat
        commands. Without this, the server rejects actions with "You can't
        do this".

        Returns:
            True if command was sent.
        """
        encoded = build_query_command(CMD_ENTER_GAME)
        return self._send_bytes(encoded, "enter_game")

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
        now = get_current_time_ms()
        self._transition(
            "MOVING",
            in_flight_action=make_in_flight_action("move", x, y, now),
        )
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
        now = get_current_time_ms()
        self._transition(
            "COLLECTING",
            in_flight_action=make_in_flight_action("collect", x, y, now),
        )
        return True

    def teleport_to(self, x: int, y: int) -> bool:
        """Send teleport command only. Map must already be open.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent, False if CDP unavailable.
        """
        if self._cdp is None:
            return False

        cmd = make_teleport_command(x, y)
        self._send_bytes(encode_teleport_command(cmd), f"teleport({x},{y})")
        now = get_current_time_ms()
        self._transition(
            "TELEPORTING",
            in_flight_action=make_in_flight_action("teleport", x, y, now),
        )
        return True

    def shoot_at(self, x: int, y: int, target_id: int = 0) -> bool:
        """Send shoot command and record the action.

        Every shot records a "shoot" action regardless of whether
        the bot is already in COMBAT state. This keeps the
        in_flight_action authoritative across consecutive shots.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            target_id: Target entity ID (0 if no specific target).

        Returns:
            True if command was sent, False if CDP unavailable.
        """
        encoded = build_shoot_command(x, y, target_id)
        if not self._send_bytes(encoded, f"shoot({x},{y},id={target_id})"):
            return False
        now = get_current_time_ms()
        action = make_in_flight_action("shoot", x, y, now)
        if self.get_state() != "COMBAT":
            self._transition("COMBAT", in_flight_action=action)
        else:
            self._state_data = transition_to(
                self._state_data,
                "COMBAT",
                in_flight_action=action,
            )
        return True

    def use_radar(self) -> bool:
        """Send radar scan command and transition to SCANNING state.

        Returns:
            True if command was sent.
        """
        encoded = build_query_command(CMD_RADAR)
        if not self._send_bytes(encoded, "radar"):
            return False
        now = get_current_time_ms()
        self._transition(
            "SCANNING",
            in_flight_action=make_in_flight_action("scan", 0, 0, now),
        )
        return True

    def request_nearest_enemy(self) -> bool:
        """Send CMD_NEAREST_ENEMY ('e' key) to get nearest enemy position.

        Server responds with EnemyDetection (0x48) containing absolute x,y
        of the nearest enemy. The response is dispatched to world state
        automatically via process_received_message.

        Returns:
            True if command was sent.
        """
        encoded = build_query_command(CMD_NEAREST_ENEMY)
        return self._send_bytes(encoded, "nearest_enemy")

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

        Checks the server-tracked inventory state to avoid redundant toggles.

        Args:
            slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile, 4=homing, 5=radar.

        Returns:
            True if equipment is now enabled (or was already enabled).
        """
        if slot < 1 or slot > 5:
            log.warning("Invalid equipment slot: %d (must be 1-5)", slot)
            return False
        if self.is_equipment_enabled(slot):
            return True
        return self.toggle_equipment(slot)

    def disable_equipment(self, slot: int) -> bool:
        """Disable equipment slot if currently enabled.

        Checks the server-tracked inventory state to avoid redundant toggles.

        Args:
            slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile, 4=homing, 5=radar.

        Returns:
            True if equipment is now disabled (or was already disabled).
        """
        if slot < 1 or slot > 5:
            log.warning("Invalid equipment slot: %d (must be 1-5)", slot)
            return False
        if not self.is_equipment_enabled(slot):
            return True
        return self.toggle_equipment(slot)

    def _has_equipment_stock(self, slot: int) -> bool:
        """Check if equipment slot has remaining stock.

        Returns True if the inventory count is above zero, OR if the
        item is currently enabled (meaning the server knows we have it,
        even if we haven't received an explicit count update yet).

        Args:
            slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile, 4=homing, 5=radar.

        Returns:
            True if equipment is available to use.
        """
        inventory = get_inventory_state()
        if slot == 1:
            item = inventory["armor_shields"]
        elif slot == 2:
            item = inventory["dual_shots"]
        elif slot == 3:
            item = inventory["missile_shots"]
        elif slot == 4:
            item = inventory["homing_shots"]
        elif slot == 5:
            item = inventory["extra_radars"]
        else:
            return False
        return item["count"] > 0 or item["enabled"]

    def is_equipment_enabled(self, slot: int) -> bool:
        """Check if equipment slot is enabled using server-tracked inventory.

        Reads the canonical inventory state from the protocol decoder,
        which is updated by 0x49, 0x67, and 0x74 messages.

        Args:
            slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile, 4=homing, 5=radar.

        Returns:
            True if enabled.
        """
        inventory = get_inventory_state()
        if slot == 1:
            return inventory["armor_shields"]["enabled"]
        if slot == 2:
            return inventory["dual_shots"]["enabled"]
        if slot == 3:
            return inventory["missile_shots"]["enabled"]
        if slot == 4:
            return inventory["homing_shots"]["enabled"]
        if slot == 5:
            return inventory["extra_radars"]["enabled"]
        return False

    # =========================================================================
    # Map Operations
    # =========================================================================

    def open_map(self) -> bool:
        """Send the map-open toggle and record the action.

        The game does not expose a reliable "map is open" flag. This method
        therefore always sends the toggle when called and records a "map_open"
        action for tick-loop timing/sync purposes. Callers must not treat local
        state as authoritative UI truth.

        Returns:
            True if the command was sent.
        """
        encoded = build_query_command(CMD_MAP_OPEN)
        if self._send_bytes(encoded, "map_open"):
            now = get_current_time_ms()
            action = make_in_flight_action("map_open", 0, 0, now)
            self._state_data = transition_to(
                self._state_data,
                self._state_data["state"],
                in_flight_action=action,
            )
            return True
        return False

    def close_map(self) -> bool:
        """Send the map toggle once.

        This helper remains available for explicit/manual use, but normal AI
        flow should not depend on tracked map-open state. The game uses the
        same protocol command as a toggle, and teleports close the map
        automatically.

        Returns:
            True if the toggle command was sent, False if CDP unavailable.
        """
        encoded = build_query_command(CMD_MAP_OPEN)
        if self._send_bytes(encoded, "map_close"):
            log.info("Map: closed via protocol toggle")
            return True
        return False

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
        self._ai_state = make_initial_ai_state()
        self._vision_state = make_empty_vision_state()
        self._cdp_message_buffer = []

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

            # Gather intel (saves tpclient.js for protocol analysis)
            self._gather_intel(page, cdp)

            # Start game log scraper for server feedback visibility
            self._init_game_log_scraper(cdp)

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
        """Run the tick loop: sync, decide, execute on each server tick."""
        from tankpit_bot.bot.tick_loop import run_tick_loop

        run_tick_loop(self, page)


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
