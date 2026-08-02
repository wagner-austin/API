"""Shared support for focused bot AI tests."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.ledger.damage_book import confirm_incoming_damage, record_incoming_shot
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state.types import (
    ContainerStateDict,
    SelfStateDict,
    TankStateDict,
    WorldStateDict,
    make_container_state,
    make_self_state,
    make_tank_state,
    make_viewport_state,
)


def consent_human(tank_id: int) -> None:
    """Mark a human as combat-consented for pursuit scenarios.

    The human-consent contract (2026-07-30) requires a chat response
    or a first strike before any human is targeted; pursuit tests use
    this to model a human who has already responded.
    """
    get_world_service().chat_seen_tank_ids.add(tank_id)


def make_enemy_tank(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 100,
    name: str = "red-40",
) -> TankStateDict:
    """Create a visible enemy tank for HUNT tests.

    The tank is wire-present at the HUNT tests' tick clock (100000):
    ``last_wire_seen_ms`` is set equal to ``timestamp_ms`` so it passes
    the kill-shot wire-presence gate, modelling an enemy genuinely in
    view rather than a map-only afterimage.

    Args:
        tank_id: Enemy tank id.
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        name: Enemy display name.

    Returns:
        Visible enemy tank state.
    """
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=1,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        last_viewport_observation_ms=100000,
    )


def make_pursuit_target(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 100,
    name: str = "red-9",
) -> TankStateDict:
    """Create an off-viewport but wire-fresh locked target.

    Models the case where a locked enemy teleported out of view:
    ``last_viewport_observation_ms`` is stale (so analyze_threats
    filters them out of the firing list) but ``timestamp_ms`` and
    ``last_wire_seen_ms`` are fresh (the global 0x2E broadcast or
    a recent map snapshot still vouches for them). HUNT must
    pursue this target via the world-registry path rather than
    enter CONFIRM_KILL.
    """
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=1,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
        # Left the viewport 8 s ago -- inside the ~12 s homing trace
        # ([[shoot-event-format]]#reroute-ttl-ms), so pursuit fire is
        # still live; the trace-expired behavior has its own pin.
        last_viewport_observation_ms=92000,
    )


def make_map_known_enemy(
    *,
    tank_id: int = 60,
    x: int = 240,
    y: int = 100,
    name: str = "red-50",
    timestamp_ms: int = 99800,
) -> TankStateDict:
    """Create a map-known enemy with no viewport confirmation.

    The tank carries a fresh map ``timestamp_ms`` (within the map-open
    cooldown) but no viewport observation, so it is invisible to
    ``analyze_threats`` and reachable only through the acquisition /
    relay paths.

    Args:
        tank_id: Enemy tank id.
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        name: Enemy display name.
        timestamp_ms: Map snapshot observation timestamp.

    Returns:
        Map-known enemy tank state.
    """
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=1,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=timestamp_ms,
    )


def seed_confirmed_incoming(count: int, weapon: int = 1, damage: int = -90) -> None:
    """Confirm ``count`` hits into the live world damage book.

    Defaults model dual fire (weapon 1, -90); pass ``weapon=0`` with
    ``damage=-45`` for the practice-room single-shot rate -- the
    confirm budget must cover the recorded weapon's cost or the book
    confirms nothing. Callers own ``reset_world_state`` bracketing.

    Args:
        count: Number of hits to record and confirm.
        weapon: Wire weapon byte for each recorded shot.
        damage: Fuel delta covering each confirmation.
    """
    book = get_world_service().damage_book
    for i in range(count):
        ts = 95000 + i * 1000
        record_incoming_shot(book, 60, "ganker", weapon, ts)
        confirm_incoming_damage(book, damage, ts + 100)


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
