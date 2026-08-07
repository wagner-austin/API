"""Shared fixtures for the executor test suite (split 2026-08-01)."""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.bot.base import Bot
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tankpit_bot.state import (
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
    make_tank_state,
)
from tests.conftest import FakeEnv
from tests.fakes import FakeCDPSession


def _make_snapshot(*, map_visible: bool = False) -> PageClientSnapshotDict:
    """Return a healthy live-client snapshot for executor tests.

    Defaults to ``map_visible=False`` so the ``map_open`` and ``teleport``
    dispatch branches exercise their normal "open the map first" path.
    Tests covering the short-circuit pass ``map_visible=True``.
    """
    return PageClientSnapshotDict(
        timestamp_ms=1000,
        client_present=True,
        map_visible=map_visible,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=50,
        last_page_client_send_age_ms=100,
        last_bot_send_age_ms=100,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections={},
    )


def _make_bot(fake_env: FakeEnv) -> tuple[Bot, FakeCDPSession]:
    """Create a Bot with FakeCDPSession in IDLE state."""
    reset_world_state()
    update_world_state_from_position(100, 100)
    update_world_state_from_fuel_total(get_world_service(), 800)
    bot = Bot("https://test.tankpit.com/", headless=True)
    fake_cdp = FakeCDPSession()
    bot._cdp = fake_cdp
    bot._state_data = bot._state_data.copy()
    bot._state_data["state"] = "IDLE"
    return bot, fake_cdp


def _store_tank(
    tank_id: int,
    *,
    x: int,
    y: int,
    source: Literal["viewport", "radar", "world_state"],
) -> None:
    """Store a tracked tank directly into world state for executor tests."""

    new_tanks = dict(get_world_service().world_state["tanks"])
    new_tanks[str(tank_id)] = make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=1,
        rank=0,
        damage_state=0,
        name="enemy",
        is_bot=True,
        is_self=False,
        source=source,
        timestamp_ms=1000,
    )
    get_world_service().world_state["tanks"] = new_tanks


def _make_world() -> WorldStateDict:
    """Create a strict world-state fixture for executor helper tests."""
    world = make_empty_world_state()
    return WorldStateDict(
        **{
            **world,
            "self_state": make_self_state(
                tank_id=1,
                x=100,
                y=100,
                team=0,
                rank=1,
                fuel=800,
                leaderboard_position=1,
            ),
            "timestamp_ms": 1000,
        }
    )


class _WorldOnlyBot:
    """Minimal bot double for _is_dispatchable helper coverage."""

    def __init__(self, world: WorldStateDict) -> None:
        """Store the provided world-state snapshot."""
        self._world = world
        self._cdp = None
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None

    def get_world_state(self) -> WorldStateDict:
        """Return the injected world-state snapshot."""
        return self._world

    def move_to(self, x: int, y: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y)
        return False

    def pickup_fuel_to(self, x: int, y: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y)
        return False

    def pickup_equipment_to(self, x: int, y: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y)
        return False

    def teleport_to(self, x: int, y: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y)
        return False

    def shoot_at(self, x: int, y: int, target_id: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (x, y, target_id)
        return False

    def use_radar(self) -> bool:
        """Unused BotProtocol stub."""
        return False

    def send_chat(self, message_id: int, x: int, y: int) -> bool:
        """Unused BotProtocol stub."""
        _ = (message_id, x, y)
        return False

    def scope_shift(self, direction: int) -> bool:
        """Unused BotProtocol stub."""
        _ = direction
        return False

    def open_map(self) -> bool:
        """Unused BotProtocol stub."""
        return False

    def close_map(self) -> bool:
        """Unused BotProtocol stub."""
        return False

    def captured_message_count(self) -> int:
        """Unused BotProtocol stub."""
        return 0

    def enable_equipment(self, slot: int) -> bool:
        """Unused BotProtocol stub."""
        _ = slot
        return False

    def disable_equipment(self, slot: int) -> bool:
        """Unused BotProtocol stub."""
        _ = slot
        return False

    def _has_equipment_stock(self, slot: int) -> bool:
        """Unused BotProtocol stub."""
        _ = slot
        return False
