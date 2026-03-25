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
from tankpit_bot.bot.ai.loop import ai_tick
from tankpit_bot.bot.ai.tactics import (
    compute_desired_equipment,
    find_teleport_target,
    should_proactive_radar,
    should_teleport_search,
)
from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state
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
    BotCommand,
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
    TICK_RATE_MS,
    build_query_command,
    build_shoot_command,
    build_toggle_equipment_command,
)
from tankpit_bot.sniffer.decoders import process_received_message
from tankpit_bot.sniffer.trackers import init_trackers_with_magic
from tankpit_bot.sniffer.world_state import get_inventory_state, get_terrain_map, get_world_state
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
        """Handle captured WebSocket message.

        Decodes incoming messages to build world state and updates
        the state machine. Magic extraction is handled by the base class.

        Args:
            message: The captured message.
        """
        super()._on_message_captured(message)

        # Decode received messages to update world state
        if message["direction"] == "received":
            process_received_message(message["payload"])

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
        # Single open, teleport, single close
        self.open_map()

        if self._page is not None:
            self._page.wait_for_timeout(TICK_RATE_MS)

        cmd = make_teleport_command(x, y)
        self._send_bytes(encode_teleport_command(cmd), f"teleport({x},{y})")

        if self._page is not None:
            self._page.wait_for_timeout(TICK_RATE_MS)

        # Map auto-closes when server sends TeleportLanded response
        self._map_is_open = False
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
        """Open the map view (skips if already open).

        Returns:
            True if command was sent or map already open.
        """
        if self._map_is_open:
            return True
        encoded = build_query_command(CMD_MAP_OPEN)
        if self._send_bytes(encoded, "map_open"):
            self._map_is_open = True
            return True
        return False

    def close_map(self) -> bool:
        """Close the map by simulating 'f' keypress via CDP."""
        if self._cdp is not None:
            try:
                self._cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyDown", "key": "f", "code": "KeyF", "text": "f",
                })
                self._cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyUp", "key": "f", "code": "KeyF",
                })
                log.info("Map: closed via 'f' keypress")
            except (OSError, RuntimeError) as e:
                log.warning("close_map keypress failed: %s", e)
        self._map_is_open = False
        return True

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
        self._ai_state = make_initial_ai_state()
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
        """Run the game loop (delegated to game_loop module)."""
        from tankpit_bot.bot.game_loop import run_game_loop

        run_game_loop(self, page)

    def _ai_tick_once(self) -> None:
        """Run one AI decision cycle and dispatch the resulting command.

        Gets world state and self state, runs ai_tick to decide the best
        behavior, manages equipment based on the chosen mode, dispatches
        the command, and persists updated AI state.
        """
        world = self.get_world_state()
        self_state = world["self_state"]

        # Can't act without self state (not yet positioned in game)
        if self_state is None:
            return

        # Each AI tick is a fresh decision — reset to IDLE before dispatching
        self._state_data = transition_to(self._state_data, "IDLE")

        terrain = get_terrain_map()
        now = get_current_time_ms()
        new_ai_state, command, behavior = ai_tick(
            world,
            self_state,
            self._ai_state,
            now,
            terrain,
        )

        config = self._ai_state["config"]
        fuel = self_state["fuel"]
        last_scan = self._ai_state["last_scan_ms"]

        # Proactive radar: scan when fuel is getting low and no fuel visible
        if should_proactive_radar(fuel, world, last_scan, now, config):
            log.info("AI: proactive radar (fuel=%d, no fuel visible)", fuel)
            self.enable_equipment(5)
            self.use_radar()
            self._ai_state = AIStateDict(**{**new_ai_state, "last_scan_ms": now})
            return

        # Teleport search: relocate when low on fuel with nothing nearby
        if should_teleport_search(
            behavior,
            fuel,
            world,
            last_scan,
            now,
            config,
        ):
            tx, ty = find_teleport_target(config, self_state)
            log.info("AI: teleport search to (%d,%d) (fuel=%d)", tx, ty, fuel)
            teleport_cmd = make_teleport_command(tx, ty)
            self._apply_equipment(
                behavior["mode"],
                fuel,
                0,
                is_teleport=True,
            )
            self._dispatch_command(teleport_cmd)
            self._ai_state = new_ai_state
            return

        log.info(
            "AI: %s score=%d target=(%d,%d) reason=%s",
            behavior["mode"],
            behavior["score"],
            behavior["target_x"],
            behavior["target_y"],
            behavior["reason"],
        )

        # Equipment management: toggle based on behavior, fuel, and target damage
        target_damage = 0
        if behavior["mode"] == "HUNT":
            target_damage = next(
                (
                    t["damage_state"]
                    for t in world["tanks"].values()
                    if t["x"] == behavior["target_x"] and t["y"] == behavior["target_y"]
                ),
                0,
            )
        is_teleport = command["cmd_type"] == "teleport"
        self._apply_equipment(
            behavior["mode"],
            fuel,
            target_damage,
            is_teleport=is_teleport,
        )

        self._dispatch_command(command)
        self._ai_state = new_ai_state

    def _apply_equipment(
        self,
        mode: str,
        fuel: int,
        target_damage: int,
        is_teleport: bool = False,
    ) -> None:
        """Apply computed equipment state — enable desired, disable undesired.

        Uses compute_desired_equipment from tactics to determine which slots
        should be on, then enables/disables accordingly. Only enables
        equipment if inventory stock is available.

        Args:
            mode: Current AI behavior mode name.
            fuel: Current fuel level.
            target_damage: Damage state of the hunt target (0-3), 0 if not hunting.
            is_teleport: Whether the current command is a teleport.
        """
        critical = self._ai_state["config"]["fuel_critical_threshold"]
        desired = compute_desired_equipment(
            mode,
            fuel,
            target_damage,
            critical,
            is_teleport,
        )

        # Extra radar: always enable if desired and stocked
        if 5 in desired and self._has_equipment_stock(5):
            self.enable_equipment(5)

        # Combat slots (1-4): enable if desired + stocked, disable otherwise
        for slot in (1, 2, 4):
            if slot in desired and self._has_equipment_stock(slot):
                self.enable_equipment(slot)
            else:
                self.disable_equipment(slot)

    def _dispatch_command(self, command: BotCommand) -> None:
        """Dispatch a BotCommand through the appropriate bot action method.

        Args:
            command: The bot command to execute.
        """
        if command["cmd_type"] == "move":
            self.move_to(command["target_x"], command["target_y"])
        elif command["cmd_type"] == "pickup_move":
            self.pickup_move_to(command["target_x"], command["target_y"])
        elif command["cmd_type"] == "shoot":
            self.shoot_at(command["target_x"], command["target_y"])
        elif command["cmd_type"] == "radar":
            self.use_radar()
        else:  # teleport
            self.teleport_to(command["target_x"], command["target_y"])


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
