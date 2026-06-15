"""Bot base class for TankPit automation.

This module provides the Bot class that extends WebSocketSniffer with:
- State machine for behavior control
- Command sending capabilities (move, shoot, radar, etc.)
- Fuel/HP tracking from world state
- Convenience methods for finding containers
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state
from tankpit_bot.bot.command_sender import send_command_bytes
from tankpit_bot.bot.commands import (
    encode_move_command,
    encode_pickup_equipment_command,
    encode_pickup_fuel_command,
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
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_teleport_command,
)
from tankpit_bot.bot.vision import (
    VisionStateDict,
    make_empty_vision_state,
)
from tankpit_bot.browser import BrowserSession, get_current_time_ms
from tankpit_bot.browser.dom_scraper import (
    GameLogEntry,
    GameLogScraper,
    scrape_page_text,
)
from tankpit_bot.diagnostics.account_stats import (
    emit_account_stats_sample,
    parse_account_stats,
)
from tankpit_bot.diagnostics.combat_screenshot import save_screenshot
from tankpit_bot.diagnostics.teleport_attempts import emit_teleport_attempt_outcome
from tankpit_bot.protocol.codec import (
    DEFAULT_STATIC_KEY_PATH,
    build_xor_table,
    load_static_key,
)
from tankpit_bot.protocol.commands import (
    CMD_ENTER_GAME,
    CMD_MAP_OPEN,
    CMD_NEAREST_ENEMY,
    CMD_RADAR,
    build_query_command,
    build_shoot_command,
    build_toggle_equipment_command,
)
from tankpit_bot.runtime_logging import (
    emit_diagnostic,
    emit_state,
    emit_sync,
    emit_wire_complete,
)
from tankpit_bot.sniffer.trackers import init_trackers_with_magic
from tankpit_bot.sniffer.world_state import (
    check_and_clear_radar_scan_complete,
    get_world_service,
    get_world_state,
    mark_move_target_failed,
)
from tankpit_bot.sniffer.world_state_combat import check_and_clear_teleport_landed
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.state import ContainerStateDict, SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage

log = get_logger(__name__)

# The C statistics panel paints incrementally: the "Statistics:" header
# can be in the DOM before the stat lines (a single 1500ms timed read
# landed in that gap and crashed sessions 20260611-004251/004405/012807).
# Poll the parse predicate instead of trusting one timed read.
_ACCOUNT_STATS_POLL_INTERVAL_MS = 300
_ACCOUNT_STATS_POLL_ATTEMPTS = 10
# Total wait budget for a single timed panel read (used by the simple
# capture path; equals one full poll budget).
_ACCOUNT_STATS_PANEL_RENDER_MS = _ACCOUNT_STATS_POLL_INTERVAL_MS * _ACCOUNT_STATS_POLL_ATTEMPTS
# The first-tick keypress itself can be swallowed by the client (run
# 20260611-013801: panel never opened across a full poll budget), so
# the startup capture retries on later ticks.
_ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS = 3


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
        self._game_log_scraper: GameLogScraper | None = None
        # Monotonic counter for opt-in per-shot screenshot filenames.
        self._shot_screenshot_seq: int = 0
        self._page: _test_hooks.PageProtocol | None = None
        self._state_data: BotStateDataDict = make_initial_state_data()
        self._ai_state: AIStateDict = make_initial_ai_state()
        # XOR encoding table for outgoing commands
        self._xor_table: bytes | None = None
        # Vision state for fallback tracking
        self._vision_state: VisionStateDict = make_empty_vision_state()
        # CDP message buffer — received payloads for tick loop sync
        self._cdp_message_buffer: list[str] = []
        # Gate for the C-panel account stats capture; fired from the
        # first HEALTHY tick rather than at bootstrap because the game
        # client ignores hotkeys until fully loaded (run 20260611-000x
        # captured panel_visible=False at startup). Failed attempts
        # retry on later ticks (bounded) since even a healthy-tick
        # keypress can be swallowed (run 20260611-013801).
        self._account_stats_captured = False
        self._account_stats_attempts = 0

    def _require_cdp(self) -> _test_hooks.CDPSessionProtocol:
        """Return the attached CDP session or raise.

        Used by tick-loop code that must read the live page-client state.
        The tick loop's readiness gates ensure ``_cdp`` is attached well
        before any code reaches the capture point, so a missing session
        is an invariant violation rather than a normal pre-bootstrap
        state.
        """
        if self._cdp is None:
            raise RuntimeError("Bot has no CDP session attached")
        return self._cdp

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
        emit_state("%s -> %s", current_state, new_state)

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
            log.debug("CDP_BUFFER: +1 (total=%d)", len(self._cdp_message_buffer))

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
        self._emit_completion(action_kind="scan", signal="radar_scan_complete", action=action)
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
            self._emit_completion(
                action_kind="move",
                signal="position_reached",
                action=action,
                landed_x=self_state["x"],
                landed_y=self_state["y"],
            )
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
            and check_and_clear_teleport_landed(get_world_service())
        ):
            action = self._state_data["in_flight_action"]
            tx, ty = action["target_x"], action["target_y"]
            if self_state["x"] != tx or self_state["y"] != ty:
                now = get_current_time_ms()
                mark_move_target_failed(tx, ty, now)
                emit_sync(
                    "teleport requested (%d,%d) but landed at (%d,%d); marked failed target",
                    tx,
                    ty,
                    self_state["x"],
                    self_state["y"],
                )
            self._emit_completion(
                action_kind="teleport",
                signal="teleport_landed",
                action=action,
                landed_x=self_state["x"],
                landed_y=self_state["y"],
            )
            landed_exactly = self_state["x"] == tx and self_state["y"] == ty
            emit_teleport_attempt_outcome(
                status="landed_exact" if landed_exactly else "landed_inexact",
                messages=self._messages,
            )
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
        position_reached = self_state["x"] == tx and self_state["y"] == ty
        if position_reached or target_key not in world["containers"]:
            signal = "position_reached" if position_reached else "container_consumed"
            self._emit_completion(
                action_kind="collect",
                signal=signal,
                action=action,
                landed_x=self_state["x"],
                landed_y=self_state["y"],
            )
            self._transition("IDLE", in_flight_action=make_no_action())
            return True
        return False

    def _emit_completion(
        self,
        *,
        action_kind: str,
        signal: str,
        action: InFlightActionDict,
        **extra: str | int | float | bool,
    ) -> None:
        """Emit a structured WIRE_COMPLETE event for an authoritative completion.

        Args:
            action_kind: Kind of action that completed.
            signal: Name of the authoritative completion signal.
            action: The in-flight action being cleared.
            **extra: Additional structured fields (e.g. landed coordinates).
        """
        started_ms = action["started_ms"]
        duration_ms = get_current_time_ms() - started_ms if started_ms > 0 else -1
        target_x = action["target_x"]
        target_y = action["target_y"]
        emit_wire_complete(
            action_kind=action_kind,
            duration_ms=duration_ms,
            signal=signal,
            target_x=target_x,
            target_y=target_y,
            **extra,
        )

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

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        """XOR encode and send command bytes via WebSocket.

        Args:
            data: Framed command bytes (with 2-byte length header).
            cmd_name: Command name for logging.

        Returns:
            True if sent, False if CDP session not available.
        """
        return send_command_bytes(
            self._cdp,
            self._xor_table,
            data,
            cmd_name,
            self._send_websocket_bytes,
        )

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

    def pickup_fuel_to(self, x: int, y: int) -> bool:
        """Send fuel pickup command and transition to COLLECTING state.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        cmd = make_pickup_fuel_command(x, y)
        if not self._send_bytes(encode_pickup_fuel_command(cmd), "pickup_fuel"):
            return False
        now = get_current_time_ms()
        self._transition(
            "COLLECTING",
            in_flight_action=make_in_flight_action("collect", x, y, now),
        )
        return True

    def pickup_equipment_to(self, x: int, y: int) -> bool:
        """Send equipment pickup command and transition to COLLECTING state.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        cmd = make_pickup_equipment_command(x, y)
        if not self._send_bytes(encode_pickup_equipment_command(cmd), "pickup_equipment"):
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

        # TeleportLanded is a single buffered flag from the protocol layer.
        # Clear any stale landing ack before issuing a new teleport so a late
        # ack from a previous teleport cannot complete this new action early.
        check_and_clear_teleport_landed(get_world_service())

        cmd = make_teleport_command(x, y)
        if not self._send_bytes(encode_teleport_command(cmd), f"teleport({x},{y})"):
            return False
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
        self._capture_shot_screenshot(x, y, target_id)
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

    def _capture_shot_screenshot(self, x: int, y: int, target_id: int) -> None:
        """Save a canvas PNG at a shot when screenshots are enabled.

        Opt-in via the ``TANKPIT_SHOT_SCREENSHOTS`` environment variable,
        whose value is the output directory. Each shot writes one PNG
        named by sequence and target tile so the fire-moment geometry can
        be reviewed as an image rather than inferred from telemetry. A
        no-op when the variable is unset or no CDP session is attached.

        Args:
            x: Shot target X tile.
            y: Shot target Y tile.
            target_id: Targeted tank id.
        """
        directory = _test_hooks.get_env("TANKPIT_SHOT_SCREENSHOTS")
        if directory is None or self._cdp is None:
            return
        self._shot_screenshot_seq += 1
        label = f"shot_{self._shot_screenshot_seq:04d}_x{x}_y{y}_id{target_id}"
        save_screenshot(self._cdp, Path(directory), label)

    def use_radar(self) -> bool:
        """Send radar scan command and transition to SCANNING state.

        Returns:
            True if command was sent.
        """
        inventory = get_inventory_state(get_world_service())
        uses_extra = inventory["extra_radars"]["enabled"] and inventory["extra_radars"]["count"] > 0
        get_world_service().record_radar_command(use_extra_radar=uses_extra)
        emit_diagnostic(
            diagnostic_kind="radar_dispatch",
            uses_extra=uses_extra,
            extra_radar_count=inventory["extra_radars"]["count"],
        )
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
        inventory = get_inventory_state(get_world_service())
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
        inventory = get_inventory_state(get_world_service())
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
        """Dispatch the wire ``CMD_MAP_OPEN`` command.

        The wire command is one-way: it only opens the map. Sending it again
        against an already-open map is a server-side no-op (no fresh map sync
        is returned). The authoritative live "is the map showing" signal is
        :func:`~tankpit_bot.action_lab.page_client_snapshot.capture_page_client_snapshot`'s
        ``map_visible`` field, which reads ``activeGame.map.h`` from the JS
        client.

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

    def _dispatch_keypress(self, key: str) -> None:
        """Send a synthetic keyDown+keyUp pair through CDP.

        Args:
            key: Single character key to dispatch (e.g. ``"m"``, ``"c"``).
        """
        if self._cdp is None:
            raise RuntimeError("_dispatch_keypress requires an attached CDP session")
        code = f"Key{key.upper()}"
        vk = ord(key.upper())
        for event_type in ("keyDown", "keyUp"):
            self._cdp.send(
                "Input.dispatchKeyEvent",
                {
                    "type": event_type,
                    "key": key,
                    "code": code,
                    "windowsVirtualKeyCode": vk,
                    "nativeVirtualKeyCode": vk,
                },
            )

    def close_map(self) -> bool:
        """Close the map overlay by dispatching a synthetic ``m`` keypress.

        No wire byte closes the map on the server: it tracks "user requested
        the map" (``CMD_MAP_OPEN``) and "user teleported" (``CMD_MAP_TELEPORT``
        auto-closes). Pressing ``m`` (or ``f``) in the browser closes the
        overlay purely client-side by toggling ``activeGame.map.h`` -- no
        WebSocket traffic. This method reproduces that behavior by sending a
        synthetic keyboard event through CDP. Verified live in
        ``discover_map_close.py``.

        Returns:
            True if the key event was dispatched, False if no CDP session is
            attached.
        """
        if self._cdp is None:
            return False
        self._dispatch_keypress("m")
        log.info("Map: closed via local 'm' keyboard event (no wire byte sent)")
        return True

    def _init_game_log_scraper(self, cdp: _test_hooks.CDPSessionProtocol) -> None:
        """Create the game log scraper for server feedback visibility.

        Args:
            cdp: Active CDP session for DOM access.
        """
        self._game_log_scraper = GameLogScraper(cdp)

    def _poll_game_log(self) -> list[GameLogEntry]:
        """Poll the game log for new entries since the last scrape.

        Returns:
            New log entries (kills, hits, empty containers, etc.).
        """
        scraper = self._game_log_scraper
        if scraper is None:
            return []
        return scraper.get_new_entries()

    def _capture_account_stats(self, phase: str) -> None:
        """Sample the in-game ``C`` statistics panel and emit it.

        The panel carries account-wide ground truth the wire never
        sends (lifetime play time, kills, deactivations, promotion
        points); the startup sample baselines every run so consecutive
        runs' deltas verify the game-log kill detection. The panel is
        toggled open, scraped, and toggled closed so it never obstructs
        play.

        Args:
            phase: Capture point label (e.g. ``startup``).
        """
        if self._cdp is None or self._page is None:
            return
        for event_type in ("keyDown", "keyUp"):
            self._cdp.send(
                "Input.dispatchKeyEvent",
                {
                    "type": event_type,
                    "key": "c",
                    "code": "KeyC",
                    "windowsVirtualKeyCode": ord("C"),
                    "nativeVirtualKeyCode": ord("C"),
                },
            )
        self._page.wait_for_timeout(_ACCOUNT_STATS_PANEL_RENDER_MS)
        page_text = scrape_page_text(self._cdp)
        for event_type in ("keyDown", "keyUp"):
            self._cdp.send(
                "Input.dispatchKeyEvent",
                {
                    "type": event_type,
                    "key": "c",
                    "code": "KeyC",
                    "windowsVirtualKeyCode": ord("C"),
                    "nativeVirtualKeyCode": ord("C"),
                },
            )
        emit_account_stats_sample(parse_account_stats(page_text), phase=phase)

    def maybe_capture_account_stats_once(self) -> None:
        """Capture account stats on the first healthy tick, with bounded retries.

        The C-panel hotkey can be swallowed by the game client (run
        20260611-013801), so failed attempts retry on later ticks up to
        a bounded maximum.
        """
        if self._account_stats_captured:
            return
        if self._account_stats_attempts >= _ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS:
            return
        self._account_stats_attempts += 1
        self._capture_account_stats("startup")
        self._account_stats_captured = True

    # =========================================================================
    # Run Loop
    # =========================================================================

    def run(self, *, session_seconds: int, stop_file_path: Path) -> None:
        """Run the bot.

        Launches browser, logs in, joins game, and runs the game loop.
        The bot will scan for fuel and collect it automatically. The
        run ends gracefully -- capture saved, browser closed -- when
        the tick budget elapses or the stop file appears.

        Args:
            session_seconds: Bounded session length in seconds; zero
                or negative runs until externally stopped.
            stop_file_path: Sentinel file whose existence requests a
                graceful shutdown.

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
            emit_state("%s", self.get_state())

            # Game loop
            try:
                self._game_loop(
                    page,
                    session_seconds=session_seconds,
                    stop_file_path=stop_file_path,
                )
            except KeyboardInterrupt:
                log.info("Bot interrupted by user")
            finally:
                self._save_capture_session()
                self._cdp = None
                self._page = None
                self._cleanup(cdp, page, context, browser)

    def _save_capture_session(self) -> None:
        """Save accumulated messages as a replayable capture session.

        Writes the capture session to the canonical bot artifact paths
        (latest + archive) so ``replay_bot.py`` can replay the run offline.
        """
        from platform_core.json_utils import dump_json_str

        from tankpit_bot.runtime_logging import get_bot_runtime_artifacts
        from tankpit_bot.types import CaptureSession, encode_capture_session

        artifacts = get_bot_runtime_artifacts()
        if artifacts is None:
            return

        session = CaptureSession(
            session_id=self._session_id,
            start_timestamp_ms=self._start_timestamp_ms,
            end_timestamp_ms=get_current_time_ms(),
            base_url=self._target_url,
            messages=self._messages,
            magic=self._magic,
            game_log=[],
            tank_names={},
        )
        encoded = encode_capture_session(session)
        json_str = dump_json_str(encoded, compact=False, indent=2)
        _test_hooks.write_text(
            Path(artifacts["latest_capture_path"]),
            json_str,
        )
        _test_hooks.write_text(
            Path(artifacts["archive_capture_path"]),
            json_str,
        )
        log.info(
            "Saved capture session: %d messages -> %s",
            len(self._messages),
            artifacts["latest_capture_path"],
        )

    def _game_loop(
        self,
        page: _test_hooks.PageProtocol,
        *,
        session_seconds: int,
        stop_file_path: Path,
    ) -> None:
        """Run the tick loop: sync, decide, execute on each server tick.

        Args:
            page: Playwright page for waiting between ticks.
            session_seconds: Bounded session length in seconds; zero
                or negative runs until externally stopped.
            stop_file_path: Sentinel file whose existence requests a
                graceful shutdown.
        """
        from tankpit_bot.bot.tick_loop import run_tick_loop

        run_tick_loop(
            self,
            page,
            session_seconds=session_seconds,
            stop_file_path=stop_file_path,
        )


__all__ = [
    "Bot",
    "BotError",
    "ProtocolNotDiscoveredError",
]
