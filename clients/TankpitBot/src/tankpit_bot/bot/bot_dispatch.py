"""Bot command dispatch — encode, send, and transition for each game action.

Each method encodes a protocol command, sends it via the inherited
``_send_bytes``, and transitions the HFSM. Equipment queries and map
operations are included because they share the same dispatch surface.

Split from bot/base.py for separation of concerns.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.bot.completions import CompletionsMixin
from tankpit_bot.bot.states import (
    make_in_flight_action,
    transition_to,
)
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.diagnostics.combat_screenshot import save_screenshot
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sniffer.world_state_combat import check_and_clear_teleport_landed
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state

log = get_logger(__name__)


class DispatchMixin(CompletionsMixin):
    """Command dispatch, equipment management, and map operations.

    Inherits ``_send_bytes`` from SessionBase and ``_transition`` from
    CompletionsMixin. Each public method encodes a protocol command,
    sends it, and transitions the HFSM.
    """

    _shot_screenshot_seq: int

    # =========================================================================
    # Command Sending
    # =========================================================================

    def enter_game(self) -> bool:
        """Send CMD_ENTER_GAME to activate the tank in the game world.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_ENTER_GAME, build_query_command

        return self._send_bytes(build_query_command(CMD_ENTER_GAME), "enter_game")

    def move_to(self, x: int, y: int) -> bool:
        """Send move command and transition to MOVING state.

        Args:
            x: Target X coordinate (0-255).
            y: Target Y coordinate (0-255).

        Returns:
            True if command was sent.
        """
        from tankpit_bot.bot.commands import encode_move_command
        from tankpit_bot.bot.types import make_move_command

        if not self._send_bytes(encode_move_command(make_move_command(x, y)), "move"):
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
        from tankpit_bot.bot.commands import encode_pickup_fuel_command
        from tankpit_bot.bot.types import make_pickup_fuel_command

        if not self._send_bytes(
            encode_pickup_fuel_command(make_pickup_fuel_command(x, y)), "pickup_fuel"
        ):
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
        from tankpit_bot.bot.commands import encode_pickup_equipment_command
        from tankpit_bot.bot.types import make_pickup_equipment_command

        if not self._send_bytes(
            encode_pickup_equipment_command(make_pickup_equipment_command(x, y)),
            "pickup_equipment",
        ):
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

        check_and_clear_teleport_landed(self.world)

        from tankpit_bot.bot.commands import encode_teleport_command
        from tankpit_bot.bot.types import make_teleport_command

        if not self._send_bytes(
            encode_teleport_command(make_teleport_command(x, y)), f"teleport({x},{y})"
        ):
            return False
        now = get_current_time_ms()
        self._transition(
            "TELEPORTING",
            in_flight_action=make_in_flight_action("teleport", x, y, now),
        )
        return True

    def shoot_at(self, x: int, y: int, target_id: int = 0) -> bool:
        """Send shoot command and record the action.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            target_id: Target entity ID (0 if no specific target).

        Returns:
            True if command was sent, False if CDP unavailable.
        """
        from tankpit_bot.protocol.commands import build_shoot_command

        if not self._send_bytes(
            build_shoot_command(x, y, target_id), f"shoot({x},{y},id={target_id})"
        ):
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
        """Save a canvas PNG when screenshots are enabled.

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
        inventory = get_inventory_state(self.world)
        uses_extra = inventory["extra_radars"]["enabled"] and inventory["extra_radars"]["count"] > 0
        self.world.record_radar_command(use_extra_radar=uses_extra)
        emit_diagnostic(
            diagnostic_kind="radar_dispatch",
            uses_extra=uses_extra,
            extra_radar_count=inventory["extra_radars"]["count"],
        )
        from tankpit_bot.protocol.commands import CMD_RADAR, build_query_command

        if not self._send_bytes(build_query_command(CMD_RADAR), "radar"):
            return False
        now = get_current_time_ms()
        self._transition(
            "SCANNING",
            in_flight_action=make_in_flight_action("scan", 0, 0, now),
        )
        return True

    def request_nearest_enemy(self) -> bool:
        """Send CMD_NEAREST_ENEMY to get nearest enemy position.

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_NEAREST_ENEMY, build_query_command

        return self._send_bytes(build_query_command(CMD_NEAREST_ENEMY), "nearest_enemy")

    def send_chat(self, message_id: int, x: int, y: int) -> bool:
        """Send a preset chat message. Fire-and-forget: no HFSM transition.

        The server's 0x4D echo is the delivery receipt (surfaced as the
        ``chat_received`` diagnostic with ``is_self_echo``); silence
        means the flood mute ate it, and the mute contract forbids a
        retry (wiki [[chat-messages]]).

        Args:
            message_id: Preset chat message ID (0-64).
            x: Sender's current X tile (0-255).
            y: Sender's current Y tile (0-255).

        Returns:
            True if command was sent, False if CDP unavailable.
        """
        from tankpit_bot.protocol.chat import build_chat_command, chat_message_text

        if not self._send_bytes(build_chat_command(message_id, x, y), f"chat({message_id})"):
            return False
        emit_diagnostic(
            diagnostic_kind="chat_sent",
            message_id=message_id,
            text=chat_message_text(message_id),
            x=x,
            y=y,
        )
        return True

    def scope_shift(self, direction: int) -> bool:
        """Send the scope-extend command and record the in-flight pan.

        Fuel-free, but NOT instantaneous: the server answers with a
        fresh 0x5A one server tick later (median exactly 2.0 s across
        759 archived pans) whose origin follows the anchor law, and it
        silently DROPS viewport-coupled commands dispatched before
        that confirm lands ([[viewport-shift-protocol]] scope-pending
        radar drop — half of all scan stalls ever recorded). Promoted
        from fire-and-forget 2026-08-20: the pan is a tracked
        ``scope`` action, resolved by the 0x5A (any stale
        viewport-update mark is cleared at dispatch so the flag means
        "a 0x5A arrived since THIS pan").

        Args:
            direction: Compass byte, clockwise from north (0=N..7=NW).

        Returns:
            True if command was sent, False if CDP unavailable.
        """
        from tankpit_bot.protocol.commands import build_scope_command

        self.world.check_and_clear_viewport_update_processed()
        if not self._send_bytes(build_scope_command(direction), f"scope({direction})"):
            return False
        emit_diagnostic(
            diagnostic_kind="scope_shift_sent",
            direction=direction,
        )
        now = get_current_time_ms()
        action = make_in_flight_action("scope", 0, 0, now)
        self._state_data = transition_to(
            self._state_data,
            self._state_data["state"],
            in_flight_action=action,
        )
        return True

    def request_inventory(self) -> bool:
        """Send CMD_INVENTORY to request the inventory snapshot.

        The cheapest game action on the wire (2 bytes, free, no world
        effect) — used as the watch-probe heartbeat to hold the
        push-on-activity stream open (wiki log 2026-07-24).

        Returns:
            True if command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_INVENTORY, build_query_command

        return self._send_bytes(build_query_command(CMD_INVENTORY), "inventory")

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
        from tankpit_bot.protocol.commands import build_toggle_equipment_command

        if slot < 1 or slot > 5:
            log.warning("Invalid equipment slot: %d (must be 1-5)", slot)
            return False
        slot_names = ["armor", "dual", "missile", "homing", "radar"]
        return self._send_bytes(
            build_toggle_equipment_command(slot), f"toggle_{slot_names[slot - 1]}"
        )

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
        if self.is_equipment_enabled(slot):
            return True
        return self.toggle_equipment(slot)

    def disable_equipment(self, slot: int) -> bool:
        """Disable equipment slot if currently enabled.

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

        Args:
            slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile, 4=homing, 5=radar.

        Returns:
            True if equipment is available to use.
        """
        inventory = get_inventory_state(self.world)
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

        Args:
            slot: Equipment slot (1-5): 1=armor, 2=dual, 3=missile, 4=homing, 5=radar.

        Returns:
            True if enabled.
        """
        inventory = get_inventory_state(self.world)
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

        Any stale map-data mark is cleared at dispatch so the
        completion flag means "a MAP_DATA arrived since THIS open" —
        the scope pan's 2026-08-20 discipline. Without the clear, a
        stalled open's late response leaves an orphan flag and the
        NEXT open "completes" in milliseconds with no fresh dots (run
        bot-20260825-212920: the final open closed in 12 ms on an
        orphan while the dying wire delivered nothing, feeding the
        no-viable-targets gate a phantom fresh snapshot).

        Returns:
            True if the command was sent.
        """
        from tankpit_bot.protocol.commands import CMD_MAP_OPEN, build_query_command

        self.world.check_and_clear_map_data_processed()
        if self._send_bytes(build_query_command(CMD_MAP_OPEN), "map_open"):
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

        Returns:
            True if the key event was dispatched, False if no CDP session.
        """
        if self._cdp is None:
            return False
        self._dispatch_keypress("m")
        log.info("Map: closed via local 'm' keyboard event (no wire byte sent)")
        return True


__all__ = [
    "DispatchMixin",
]
