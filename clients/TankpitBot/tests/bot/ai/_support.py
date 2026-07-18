"""Shared support for focused bot AI tests."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.state.types import (
    ContainerStateDict,
    SelfStateDict,
    TankStateDict,
    WorldStateDict,
    make_container_state,
    make_self_state,
    make_viewport_state,
)


def make_scanned_ai_state(
    *,
    landing_scan_viewport: str = "92,92",
) -> AIStateDict:
    """Create AI state that does not force an opening radar action.

    Args:
        landing_scan_viewport: ``"left,top"`` origin recorded in the
            one-radar-per-landing latch. Defaults to the (92,92)
            viewport that :func:`make_world` builds around the default
            (100,100) position, so COLLECT's unconditional
            scan-on-landing reads as already satisfied and tests
            exercise the downstream cascade steps.

    Returns:
        AI state with a non-zero last scan timestamp and the landing
        latch recorded.
    """
    return AIStateDict(
        **{
            **make_initial_ai_state(),
            "last_scan_ms": 1,
            "last_landing_scan_viewport": landing_scan_viewport,
        }
    )


def viewport_covered_tiles(world: WorldStateDict, now_ms: int = 100000) -> dict[str, int]:
    """Return a coverage map marking every tile in the world's viewport.

    Useful for tests that need the forager to treat the current
    viewport as fully scanned (e.g. when the test exercises the
    search-hop / edge-walk / map-intel fallback path beneath the
    forager). Mirrors what an extra-radar reveal writes.

    Args:
        world: World state whose viewport bounds drive the coverage map.
        now_ms: Timestamp to stamp every tile with.

    Returns:
        Coverage dict keyed by ``"x,y"`` with every viewport tile marked.
    """
    viewport = world["viewport"]
    left = viewport["left"]
    top = viewport["top"]
    right = left + viewport["width"] - 1
    bottom = top + viewport["height"] - 1
    return {f"{x},{y}": now_ms for y in range(top, bottom + 1) for x in range(left, right + 1)}


def make_world(
    *,
    self_x: int = 100,
    self_y: int = 100,
    fuel: int = 800,
    containers: dict[str, ContainerStateDict] | None = None,
    tanks: dict[str, TankStateDict] | None = None,
    scanned: bool = True,
) -> tuple[WorldStateDict, SelfStateDict]:
    """Create world and self state for bot AI tests.

    Args:
        self_x: Controlled tank X coordinate.
        self_y: Controlled tank Y coordinate.
        fuel: Current fuel amount.
        containers: Optional visible containers.
        tanks: Optional visible tanks.
        scanned: Whether the current viewport is fully tile-covered.

    Returns:
        Tuple of world state and self state.
    """
    self_state = make_self_state(
        tank_id=1,
        x=self_x,
        y=self_y,
        team=1,
        rank=2,
        fuel=fuel,
        leaderboard_position=5,
    )
    viewport = make_viewport_state(left=self_x - 8, top=self_y - 8, width=16, height=16)
    left = viewport["left"]
    top = viewport["top"]
    right = left + viewport["width"] - 1
    bottom = top + viewport["height"] - 1
    scanned_tiles: dict[str, int] = (
        {f"{x},{y}": 100000 for y in range(top, bottom + 1) for x in range(left, right + 1)}
        if scanned
        else {}
    )
    world = WorldStateDict(
        self_state=self_state,
        tanks=tanks or {},
        containers=containers or {},
        mines={},
        terrain={},
        viewport=viewport,
        scanned_tiles=scanned_tiles,
        timestamp_ms=0,
    )
    return world, self_state


def make_inventory(
    *,
    dual_count: int = 30,
    dual_enabled: bool = True,
    default_count: int = 30,
) -> InventoryState:
    """Create a basic inventory state for bot AI tests.

    Args:
        dual_count: Dual-shot count.
        dual_enabled: Whether dual shots are enabled.
        default_count: Count for all other items.

    Returns:
        Inventory state populated with basic enabled items.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=default_count, enabled=True),
        dual_shots=InventoryItem(count=dual_count, enabled=dual_enabled),
        missile_shots=InventoryItem(count=default_count, enabled=True),
        homing_shots=InventoryItem(count=default_count, enabled=True),
        extra_radars=InventoryItem(count=default_count, enabled=True),
    )


def make_container(
    x: int,
    y: int,
    volume: int,
    is_fuel: bool,
    timestamp_ms: int = 100000,
) -> ContainerStateDict:
    """Create a visible container for bot AI tests.

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.
        volume: Container volume reported by the protocol.
        is_fuel: Whether the container is fuel instead of equipment.
        timestamp_ms: Observation timestamp in milliseconds.

    Returns:
        Container state with the requested fields.
    """
    return make_container_state(
        x=x,
        y=y,
        volume=volume,
        is_fuel=is_fuel,
        timestamp_ms=timestamp_ms,
    )
