"""Behaviour scoring vocabulary: modes, reasons, and rendered scores.

The arbitrator's explanation surface -- the behaviour modes it can
choose between, the reason kinds it cites, and the score record it
renders into a human-readable line. Imports no other AI type module.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

BehaviorMode = Literal[
    "HUNT",
    "COLLECT",
]


BEHAVIOR_MODES: tuple[BehaviorMode, ...] = (
    "HUNT",
    "COLLECT",
)


ReasonKind = Literal[
    # shared
    "scan_on_landing",
    # COLLECT
    "equipment_locked",
    "fuel_locked",
    "equipment_restock",
    "equipment_hop",
    "fuel_hop",
    "fuel_collect",
    "mine_clearance_shot",
    "forage_radar",
    "forage_sweep",
    "forage_frontier_walk",
    "quad_sweep_shift",
    "quad_sweep_radar",
    "harvest_frame_shift",
    "harvest_leg_walk",
    "desync_rescan",
    "mine_hit_reveal_scan",
    "search_collect_local",
    "walk_for_fuel",
    "walk_for_fuel_pan",
    "map_for_dots",
    "ferry_scope_scout",
    "gatherer_hold",
    # HUNT
    "find_target",
    "find_enemies",
    "teleport_target",
    "greet_approach",
    "walk_to_target",
    "shoot_target",
    "combat_frame_shift",
    "opportunity_shot",
    "dot_relay",
    "hunt_refuel",
    "confirm_kill",
    # controller
    "manual_hold",
]


REASON_KINDS: tuple[ReasonKind, ...] = (
    "scan_on_landing",
    "equipment_locked",
    "fuel_locked",
    "equipment_restock",
    "equipment_hop",
    "fuel_hop",
    "fuel_collect",
    "mine_clearance_shot",
    "forage_radar",
    "forage_sweep",
    "forage_frontier_walk",
    "quad_sweep_shift",
    "quad_sweep_radar",
    "harvest_frame_shift",
    "harvest_leg_walk",
    "desync_rescan",
    "mine_hit_reveal_scan",
    "search_collect_local",
    "walk_for_fuel",
    "walk_for_fuel_pan",
    "map_for_dots",
    "ferry_scope_scout",
    "gatherer_hold",
    "find_target",
    "find_enemies",
    "teleport_target",
    "greet_approach",
    "walk_to_target",
    "shoot_target",
    "combat_frame_shift",
    "opportunity_shot",
    "dot_relay",
    "confirm_kill",
    "manual_hold",
)


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
    if not context:
        return behavior["reason_kind"]
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(context.items()))
    return f"{behavior['reason_kind']}({rendered})"


__all__ = [
    "BEHAVIOR_MODES",
    "REASON_KINDS",
    "BehaviorMode",
    "BehaviorScoreDict",
    "ReasonKind",
    "make_behavior_score",
    "render_reason",
]
