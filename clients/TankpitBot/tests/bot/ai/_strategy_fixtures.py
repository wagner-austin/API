"""Shared world, container, and inventory builders for the strategy tests."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_initial_ai_state,
)
from tankpit_bot.inventory import (
    InventoryItem,
    InventoryState,
)
from tankpit_bot.state.types import (
    ContainerStateDict,
    SelfStateDict,
    TankStateDict,
    WorldStateDict,
    make_self_state,
    make_viewport_state,
)


def _make_world(
    self_x: int = 100,
    self_y: int = 100,
    fuel: int = 800,
    containers: dict[str, ContainerStateDict] | None = None,
    tanks: dict[str, TankStateDict] | None = None,
    scanned: bool = True,
    block_scanned: bool | None = None,
) -> tuple[WorldStateDict, SelfStateDict]:
    """Build a world state for testing.

    ``block_scanned`` defaults to follow ``scanned``; pass ``False``
    for covered-viewport-but-fresh-surroundings (search-hop landings
    inside the 31x31 ring must not read as freshly scanned).
    """
    self_state = make_self_state(
        tank_id=1,
        x=self_x,
        y=self_y,
        team=1,
        rank=0,
        fuel=fuel,
        leaderboard_position=0,
    )
    vp_left = self_x - 8
    vp_top = self_y - 8
    # ``scanned`` covers the whole 31x31 quad-sweep block, not just
    # the viewport: a fixture that says "the ground here is known"
    # must also decline the sweep's block gate, or the sweep outranks
    # the cascade rung under test ([[quad-sweep-doctrine]]).
    cover_block = scanned if block_scanned is None else block_scanned
    if cover_block:
        left, top = max(self_x - 15, 0), max(self_y - 15, 0)
        right, bottom = min(self_x + 15, 255), min(self_y + 15, 255)
    else:
        left, top, right, bottom = vp_left, vp_top, vp_left + 15, vp_top + 15
    scanned_tiles: dict[str, int] = (
        {f"{x},{y}": 100000 for y in range(top, bottom + 1) for x in range(left, right + 1)}
        if scanned
        else {}
    )
    return (
        WorldStateDict(
            self_state=self_state,
            tanks=tanks or {},
            containers=containers or {},
            mines={},
            terrain={},
            viewport=make_viewport_state(left=vp_left, top=vp_top, width=16, height=16),
            scanned_tiles=scanned_tiles,
            timestamp_ms=100000,
        ),
        self_state,
    )


def _c(x: int, y: int, volume: int, is_fuel: bool) -> ContainerStateDict:
    """Create a container state."""
    from tankpit_bot.state.types import make_container_state

    return make_container_state(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=volume,
        timestamp_ms=100000,
        failed_pickups=0,
    )


def _make_inventory(
    dual_count: int = 30,
    default_count: int = 30,
    radar_count: int = 30,
) -> InventoryState:
    """Build an inventory."""
    item = InventoryItem(count=default_count, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=InventoryItem(count=dual_count, enabled=True),
        missile_shots=item,
        homing_shots=item,
        extra_radars=InventoryItem(count=radar_count, enabled=True),
    )


def _scanned_ai_state() -> AIStateDict:
    """Build a scanned AI state.

    ``last_landing_scan_viewport`` matches the (92,92) viewport every
    ``_make_world`` builds around position (100,100), so COLLECT's
    unconditional scan-on-landing latch reads as already satisfied and
    tests exercise the downstream cascade steps.
    """
    return AIStateDict(
        **{
            **make_initial_ai_state(),
            "last_landing_scan_viewport": "92,92",
        }
    )
