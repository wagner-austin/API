"""Behaviour scoring vocabulary: modes, reasons, and rendered scores.

The arbitrator's explanation surface -- the behaviour modes it can
choose between, the reason kinds it cites, and the score record it
renders into a human-readable line. Imports no other AI type module.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import TypedDict


class BehaviorMode(StrEnum):
    """The behaviour a scored decision belongs to."""

    HUNT = "HUNT"
    COLLECT = "COLLECT"


class ReasonKind(StrEnum):
    """Why the arbitrator chose a decision; each member is its wire and log word."""

    # shared
    SCAN_ON_LANDING = "scan_on_landing"
    # COLLECT
    EQUIPMENT_LOCKED = "equipment_locked"
    FUEL_LOCKED = "fuel_locked"
    EQUIPMENT_RESTOCK = "equipment_restock"
    EQUIPMENT_HOP = "equipment_hop"
    FORAGE_FRONTIER_HOP = "forage_frontier_hop"
    FUEL_HOP = "fuel_hop"
    FUEL_COLLECT = "fuel_collect"
    MINE_CLEARANCE_SHOT = "mine_clearance_shot"
    MINE_PIN = "mine_pin"
    FORAGE_RADAR = "forage_radar"
    FORAGE_SWEEP = "forage_sweep"
    FORAGE_FRONTIER_WALK = "forage_frontier_walk"
    FORAGE_FRONTIER_PAN = "forage_frontier_pan"
    QUAD_SWEEP_SHIFT = "quad_sweep_shift"
    QUAD_SWEEP_RADAR = "quad_sweep_radar"
    HARVEST_FRAME_SHIFT = "harvest_frame_shift"
    HARVEST_LEG_WALK = "harvest_leg_walk"
    DESYNC_RESCAN = "desync_rescan"
    MINE_HIT_REVEAL_SCAN = "mine_hit_reveal_scan"
    SEARCH_COLLECT_LOCAL = "search_collect_local"
    CLAIM_DENIED = "claim_denied"
    WALK_FOR_FUEL = "walk_for_fuel"
    WALK_FOR_FUEL_PAN = "walk_for_fuel_pan"
    MAP_FOR_DOTS = "map_for_dots"
    AWAIT_MAP_ANSWER = "await_map_answer"
    FERRY_SCOPE_SCOUT = "ferry_scope_scout"
    GATHERER_HOLD = "gatherer_hold"
    # HUNT
    FIND_TARGET = "find_target"
    FIND_ENEMIES = "find_enemies"
    TELEPORT_TARGET = "teleport_target"
    GREET_APPROACH = "greet_approach"
    WALK_TO_TARGET = "walk_to_target"
    SHOOT_TARGET = "shoot_target"
    COMBAT_FRAME_SHIFT = "combat_frame_shift"
    OPPORTUNITY_SHOT = "opportunity_shot"
    DOT_RELAY = "dot_relay"
    HUNT_REFUEL = "hunt_refuel"
    CONFIRM_KILL = "confirm_kill"
    # controller
    MANUAL_HOLD = "manual_hold"


class BehaviorScoreDict(TypedDict):
    """A scored candidate behavior with target coordinates.

    Attributes:
        mode: Which behavior this score represents.
        score: Priority score (0-1000). Higher wins.
        target_x: Target X coordinate for this behavior.
        target_y: Target Y coordinate for this behavior.
        target_id: Tank ID of the combat target (0 if no specific target).
        reason_kind: Typed decision reason (see :data:`ReasonKind`).
        reason_context: Reason-specific scalar payload -- e.g.
            ``target_name`` for the ``*_target`` kinds, ``volume`` for
            the fuel kinds. Empty when the kind needs no parameters.
    """

    mode: BehaviorMode
    score: int
    target_x: int
    target_y: int
    target_id: int
    reason_kind: ReasonKind
    reason_context: dict[str, str | int]


def make_behavior_score(
    mode: BehaviorMode,
    score: int,
    target_x: int,
    target_y: int,
    reason_kind: ReasonKind,
    target_id: int = 0,
    reason_context: dict[str, str | int] | None = None,
) -> BehaviorScoreDict:
    """Create a BehaviorScoreDict.

    Args:
        mode: Behavior mode.
        score: Priority score (0-1000).
        target_x: Target X coordinate.
        target_y: Target Y coordinate.
        reason_kind: Typed decision reason.
        target_id: Tank ID of combat target (0 if no specific target).
        reason_context: Reason-specific scalar payload.

    Returns:
        BehaviorScoreDict with the provided values.
    """
    return BehaviorScoreDict(
        mode=mode,
        score=score,
        target_x=target_x,
        target_y=target_y,
        target_id=target_id,
        reason_kind=reason_kind,
        reason_context={} if reason_context is None else reason_context,
    )


def render_reason(behavior: BehaviorScoreDict) -> str:
    """Render a behavior's reason as a compact human-readable label.

    The single formatting path for log lines, the HUD overlay, and
    replay narration: ``kind`` alone when the context is empty,
    ``kind(k=v, ...)`` otherwise.

    Args:
        behavior: Behavior score carrying the typed reason.

    Returns:
        Compact reason label.
    """
    context = behavior["reason_context"]
    kind = behavior["reason_kind"].value
    if not context:
        return kind
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(context.items()))
    return f"{kind}({rendered})"


__all__ = [
    "BehaviorMode",
    "BehaviorScoreDict",
    "ReasonKind",
    "make_behavior_score",
    "render_reason",
]
