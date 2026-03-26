"""AI strategy for the tick loop.

Wraps the AI evaluator system (ai_tick) and tactical overrides
(proactive radar, teleport search, map open for enemies, combat
feedback) into a single decide() function that returns a
TickDecisionDict.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.actions import _MAX_SHOOT_RANGE
from tankpit_bot.bot.ai.loop import ai_tick
from tankpit_bot.bot.ai.tactics import (
    compute_desired_equipment,
    should_map_open_for_enemies,
    should_proactive_radar,
)
from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
    BehaviorScoreDict,
    make_behavior_score,
)
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import (
    make_map_open_command,
    make_radar_command,
    make_teleport_command,
)
from tankpit_bot.inventory import InventoryState
from tankpit_bot.state.types import SelfStateDict, WorldStateDict

log = get_logger(__name__)


def decide(
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
    inventory: InventoryState,
    timestamp_ms: int,
    terrain: TerrainMapProtocol | None,
    combat_feedback: CombatFeedback = "",
) -> TickDecisionDict:
    """Run one AI decision cycle and return the tick decision.

    Handles combat feedback from the previous tick, expires kill cooldowns,
    filters killed tanks from the world, checks tactical overrides (proactive
    radar, map open for enemies, teleport search), runs core AI evaluation,
    and tracks shot targets for next-tick feedback.

    Args:
        world: Current world state with tanks, containers, mines.
        self_state: Player's own state (position, fuel, team, rank).
        ai_state: Current AI state (config, cooldowns, combat target).
        inventory: Current equipment inventory (counts and enabled flags).
        timestamp_ms: Current game timestamp in milliseconds.
        terrain: Optional terrain map for reachability checks.
        combat_feedback: Result of DOM scraping for last shot outcome.

    Returns:
        TickDecisionDict with command, behavior, updated AI state, and
        desired equipment slots.
    """
    config = ai_state["config"]
    fuel = self_state["fuel"]

    # --- 0. Expire old kill cooldowns ---
    killed = _expire_kills(
        ai_state["killed_tank_ids"],
        timestamp_ms,
        config["kill_cooldown_ms"],
    )

    # Standard equipment for non-combat actions (radar, map_open)
    standard_equip = _compute_equipment("HUNT", fuel, inventory)

    # --- 1. Handle combat feedback from previous tick ---
    # On miss: open map to refresh positions, close next tick
    if combat_feedback == "miss" and ai_state["last_shot_target_id"] != -1:
        log.info("AI: miss — opening map to refresh positions")
        new_ai = AIStateDict(
            **{
                **ai_state,
                "killed_tank_ids": killed,
                "last_shot_target_id": -1,
                "last_shot_target_name": "",
                "last_map_open_ms": timestamp_ms,
            }
        )
        return make_tick_decision(
            command=make_map_open_command(),
            behavior=make_behavior_score("HUNT", 0, 0, 0, "miss_map_open"),
            updated_ai_state=new_ai,
            desired_equipment=standard_equip,
        )

    # Clear pending shot tracking
    ai_state_clean = AIStateDict(
        **{
            **ai_state,
            "killed_tank_ids": killed,
            "last_shot_target_id": -1,
            "last_shot_target_name": "",
        }
    )

    # --- 2. Filter killed tanks from world ---
    filtered_world = _filter_killed_tanks(world, killed)

    # --- 3. Tactical overrides: radar and map open ---
    override = _check_tactical_overrides(
        fuel,
        filtered_world,
        self_state,
        ai_state_clean,
        timestamp_ms,
        config,
        standard_equip,
    )
    if override is not None:
        return override

    # --- 4. Core AI decision ---
    new_ai_state, command, behavior = ai_tick(
        filtered_world,
        self_state,
        ai_state_clean,
        timestamp_ms,
        terrain,
    )

    # --- 5. Post-AI overrides (teleport to far enemy) ---
    post_override = _check_post_ai_overrides(
        behavior,
        self_state,
        fuel,
        filtered_world,
        config,
        inventory,
        terrain,
        new_ai_state,
    )
    if post_override is not None:
        return post_override

    # --- 6. Nothing to do — open map to find enemies instead of walking to (0,0)
    if behavior["score"] == 0:
        log.info("AI: nothing to do — opening map to find enemies")
        new_ai = AIStateDict(**{**new_ai_state, "last_map_open_ms": timestamp_ms})
        return make_tick_decision(
            command=make_map_open_command(),
            behavior=behavior,
            updated_ai_state=new_ai,
            desired_equipment=standard_equip,
        )

    # --- 6. Track shot target for next-tick feedback ---
    if command["cmd_type"] == "shoot":
        target_name = _find_target_name(world, command["target_id"])
        new_ai_state = AIStateDict(
            **{
                **new_ai_state,
                "last_shot_target_id": command["target_id"],
                "last_shot_target_name": target_name,
            }
        )

    # --- 7. Compute desired equipment from behavior ---
    desired = _compute_equipment(
        behavior["mode"],
        fuel,
        inventory,
    )

    return make_tick_decision(
        command=command,
        behavior=behavior,
        updated_ai_state=new_ai_state,
        desired_equipment=desired,
    )


def _check_tactical_overrides(
    fuel: int,
    filtered_world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state_clean: AIStateDict,
    timestamp_ms: int,
    config: AIConfigDict,
    standard_equip: list[int],
) -> TickDecisionDict | None:
    """Check proactive radar and map open tactical overrides.

    Args:
        fuel: Current fuel level.
        filtered_world: World state with killed tanks removed.
        self_state: Player's own state.
        ai_state_clean: AI state with shot tracking cleared.
        timestamp_ms: Current game timestamp.
        config: AI configuration.
        standard_equip: Standard equipment slots for non-combat actions.

    Returns:
        TickDecisionDict if an override fires, or None to continue.
    """
    last_scan = ai_state_clean["last_scan_ms"]
    if should_proactive_radar(fuel, filtered_world, last_scan, timestamp_ms, config):
        log.info("AI: proactive radar (fuel=%d)", fuel)
        new_ai = AIStateDict(**{**ai_state_clean, "last_scan_ms": timestamp_ms})
        behavior = make_behavior_score("HUNT", 0, 0, 0, "proactive_radar")
        return make_tick_decision(
            command=make_radar_command(),
            behavior=behavior,
            updated_ai_state=new_ai,
            desired_equipment=standard_equip,
        )

    if should_map_open_for_enemies(
        filtered_world,
        self_state,
        ai_state_clean["last_map_open_ms"],
        timestamp_ms,
        config,
    ):
        log.info("AI: map open to discover enemies")
        new_ai = AIStateDict(**{**ai_state_clean, "last_map_open_ms": timestamp_ms})
        behavior = make_behavior_score("HUNT", 0, 0, 0, "map_open_enemies")
        return make_tick_decision(
            command=make_map_open_command(),
            behavior=behavior,
            updated_ai_state=new_ai,
            desired_equipment=standard_equip,
        )

    return None


def _check_post_ai_overrides(
    behavior: BehaviorScoreDict,
    self_state: SelfStateDict,
    fuel: int,
    filtered_world: WorldStateDict,
    config: AIConfigDict,
    inventory: InventoryState,
    terrain: TerrainMapProtocol | None,
    new_ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Check post-AI-decision overrides: teleport to far enemy,
    equipment depletion redirect.

    Args:
        behavior: Chosen behavior from AI evaluators.
        self_state: Player's own state.
        fuel: Current fuel level.
        filtered_world: World state with killed tanks removed.
        config: AI configuration.
        inventory: Current equipment inventory.
        terrain: Optional terrain map for reachability checks.
        new_ai_state: Updated AI state from ai_tick.

    Returns:
        TickDecisionDict if an override fires, or None to continue.
    """
    # Teleport to far enemy — only when AI chose HUNT and target is beyond
    # viewport range. HUNT only activates when fuel > hunt_min_fuel, so no
    # separate fuel guard needed.
    if behavior["mode"] == "HUNT" and fuel > config["hunt_min_fuel"]:
        teleport_target = _find_teleport_enemy(filtered_world, self_state, config)
        if teleport_target is not None:
            tx, ty, target_dist = teleport_target
            log.info(
                "AI: teleport to enemy at (%d,%d) dist=%d",
                tx,
                ty,
                target_dist,
            )
            return make_tick_decision(
                command=make_teleport_command(tx, ty),
                behavior=behavior,
                updated_ai_state=new_ai_state,
                desired_equipment=[2, 5],
            )

    return None


# =============================================================================
# Internal helpers
# =============================================================================


def _find_teleport_enemy(
    world: WorldStateDict,
    self_state: SelfStateDict,
    config: AIConfigDict,
) -> tuple[int, int, int] | None:
    """Find the nearest enemy beyond viewport range to teleport to.

    Only returns a target if the nearest enemy is beyond _MAX_SHOOT_RANGE
    (i.e., we can't reach them by walking/shooting within the viewport).
    This mirrors the old game_loop behavior: if no enemy in combat range,
    open map → find closest → teleport.

    Args:
        world: Current world state.
        self_state: Player's own state.
        config: AI configuration (unused, reserved for future thresholds).

    Returns:
        Tuple of (x, y, distance) for the nearest far enemy, or None
        if no enemies exist or the nearest is within viewport range.
    """
    self_team = self_state["team"]
    sx, sy = self_state["x"], self_state["y"]
    best_dist = 999
    best_x = 0
    best_y = 0
    found = False

    for tank in world["tanks"].values():
        if tank["is_self"] or tank["team"] == self_team:
            continue
        if tank["x"] == 0 and tank["y"] == 0:
            continue
        dist = manhattan_distance(sx, sy, tank["x"], tank["y"])
        if dist < best_dist:
            best_dist = dist
            best_x = tank["x"]
            best_y = tank["y"]
            found = True

    if not found or best_dist <= _MAX_SHOOT_RANGE:
        return None

    return (best_x, best_y, best_dist)


def _expire_kills(
    killed: dict[str, int],
    now: int,
    cooldown_ms: int,
) -> dict[str, int]:
    """Remove expired entries from the killed tank IDs dict.

    Args:
        killed: Current killed_tank_ids mapping.
        now: Current timestamp in milliseconds.
        cooldown_ms: How long to keep entries.

    Returns:
        New dict with only non-expired entries.
    """
    return {k: v for k, v in killed.items() if now - v < cooldown_ms}


def _filter_killed_tanks(
    world: WorldStateDict,
    killed: dict[str, int],
) -> WorldStateDict:
    """Remove killed tanks from the world state.

    Creates a new world state with tanks on the kill cooldown list
    removed, so the AI evaluators don't target corpses.

    Args:
        world: Original world state.
        killed: Killed tank IDs to filter out.

    Returns:
        New WorldStateDict with killed tanks removed.
    """
    if not killed:
        return world
    filtered_tanks = {k: v for k, v in world["tanks"].items() if k not in killed}
    return WorldStateDict(
        self_state=world["self_state"],
        tanks=filtered_tanks,
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=world["viewport"],
        timestamp_ms=world["timestamp_ms"],
    )


def _find_target_name(world: WorldStateDict, target_id: int) -> str:
    """Look up a tank's name from the world state by ID.

    Args:
        world: Current world state.
        target_id: Tank ID to look up.

    Returns:
        Tank name, or empty string if not found.
    """
    tank = world["tanks"].get(str(target_id))
    if tank is not None:
        return tank["name"]
    return ""


def _compute_equipment(
    mode: str,
    fuel: int,
    inventory: InventoryState,
) -> list[int]:
    """Compute desired equipment slots as a sorted list.

    Args:
        mode: Current AI behavior mode name.
        fuel: Current fuel level.
        inventory: Current equipment inventory with counts.

    Returns:
        Sorted list of equipment slot numbers (1-5) to enable.
    """
    desired_set = compute_desired_equipment(
        mode,
        fuel,
        dual_shots_count=inventory["dual_shots"]["count"],
    )
    return sorted(desired_set)


__all__ = [
    "decide",
]
