"""Core TypedDicts, factory functions, and encode/decode for the AI system.

All types are immutable TypedDicts with factory functions for construction,
encode functions for JSON serialization, and decode functions with require_*
validation for deserialization.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from tankpit_bot.bot.ai.modes import (
    AIMode,
    AIModeState,
)

# =============================================================================
# Behavior Mode
# =============================================================================

BehaviorMode = Literal[
    "HUNT",
    "COLLECT",
]

BEHAVIOR_MODES: tuple[BehaviorMode, ...] = (
    "HUNT",
    "COLLECT",
)


# =============================================================================
# ReasonKind + BehaviorScoreDict
# =============================================================================

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
    "forage_radar",
    "forage_sweep",
    "search_collect_local",
    "walk_for_fuel",
    "map_for_dots",
    # HUNT
    "find_target",
    "find_enemies",
    "teleport_target",
    "shoot_target",
    "dot_relay",
    "hunt_refuel",
    "confirm_kill",
    # controller
    "manual_hold",
]
"""Typed decision-reason vocabulary (Phase 2).

Replaces the free-text ``reason`` string. Several of these are
load-bearing control flow: ``derive_hunt_mode_state`` /
``derive_collect_mode_state`` branch on them to derive the AI mode
substate, so the vocabulary is closed on purpose -- adding a planner
path means adding its reason here, not inventing a string.
"""

REASON_KINDS: tuple[ReasonKind, ...] = (
    "scan_on_landing",
    "equipment_locked",
    "fuel_locked",
    "equipment_restock",
    "equipment_hop",
    "fuel_hop",
    "fuel_collect",
    "forage_radar",
    "forage_sweep",
    "search_collect_local",
    "walk_for_fuel",
    "map_for_dots",
    "find_target",
    "find_enemies",
    "teleport_target",
    "shoot_target",
    "dot_relay",
    "confirm_kill",
    "manual_hold",
)
"""All valid reason kinds, for validation messages."""


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


# =============================================================================
# EnemyThreatDict
# =============================================================================


class EnemyThreatDict(TypedDict):
    """An analyzed enemy tank with computed distance.

    Carries the three freshness timestamps from
    :class:`tankpit_bot.state.types.tank.TankStateDict` so the
    combat-strategy layer can read them without re-querying the
    registry.

    Attributes:
        tank_id: Enemy tank id.
        x: Enemy X coordinate.
        y: Enemy Y coordinate.
        distance: Manhattan distance from self.
        damage_state: Fuel-quartile health tier (0=near death .. 3=full).
        rank: Military rank (0-7). Lower rank = weaker.
        team: Enemy team id (0-3).
        name: Enemy player name.
        is_bot: Whether this enemy is a bot.
        timestamp_ms: When this tank was last confirmed by ANY source.
            Drives acquisition freshness.
        last_wire_seen_ms: When a wire-presence source last vouched the
            tank is in view. Drives the ghost gate.
        last_position_update_ms: When a wire-sourced observation last
            carried fresh ``(x, y)``. Drives the kill-shot gate.
        last_aim_x: Wire-reported barrel-aim X from this enemy's most
            recent 0x53 ShootEvent, or ``-1`` when never seen firing.
            Combat consumers (avoid-fire, predicted-LOS) read this to
            reason about which tile the enemy may target next.
        last_aim_y: Wire-reported barrel-aim Y from the same event.
        last_aim_weapon: Weapon byte from the same event
            (0=single, 1=dual, 2=missile, 3=homing). ``-1`` when never
            seen firing.
        last_aim_ms: Wall-clock of the most recent 0x53 ShootEvent
            attributed to this enemy. Consumers should age the aim
            with their own staleness threshold.
    """

    tank_id: int
    x: int
    y: int
    distance: int
    damage_state: int
    rank: int
    team: int
    name: str
    is_bot: bool
    timestamp_ms: int
    last_wire_seen_ms: int
    last_position_update_ms: int
    last_aim_x: int
    last_aim_y: int
    last_aim_weapon: int
    last_aim_ms: int


def make_enemy_threat(
    tank_id: int,
    x: int,
    y: int,
    distance: int,
    damage_state: int,
    rank: int,
    team: int,
    name: str,
    is_bot: bool,
    timestamp_ms: int = 0,
    last_wire_seen_ms: int = 0,
    last_position_update_ms: int = 0,
    last_aim_x: int = -1,
    last_aim_y: int = -1,
    last_aim_weapon: int = -1,
    last_aim_ms: int = 0,
) -> EnemyThreatDict:
    """Create an EnemyThreatDict.

    Args:
        tank_id: Enemy tank id.
        x: Enemy X coordinate.
        y: Enemy Y coordinate.
        distance: Manhattan distance from self.
        damage_state: Fuel-quartile health tier (0=near death .. 3=full).
        rank: Military rank (0-7).
        team: Team id (0-3).
        name: Player name.
        is_bot: Whether this is a bot.
        timestamp_ms: When this tank was last confirmed by any source.
        last_wire_seen_ms: When a wire-presence source last vouched the
            tank is in view. Zero means never wire-confirmed.
        last_position_update_ms: When a wire-sourced observation last
            carried fresh ``(x, y)``. Zero means position has never
            been wire-confirmed.
        last_aim_x: Wire-reported barrel-aim X from the enemy's most
            recent 0x53 ShootEvent. Defaults to ``-1`` (never seen).
        last_aim_y: Wire-reported barrel-aim Y from the same event.
        last_aim_weapon: Weapon byte (0=single, 1=dual, 2=missile,
            3=homing) from the same event. ``-1`` when never seen.
        last_aim_ms: Wall-clock of the most recent 0x53 event for
            this enemy. ``0`` when never seen.

    Returns:
        EnemyThreatDict with the provided values.
    """
    return EnemyThreatDict(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=distance,
        damage_state=damage_state,
        rank=rank,
        team=team,
        name=name,
        is_bot=is_bot,
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=last_wire_seen_ms,
        last_position_update_ms=last_position_update_ms,
        last_aim_x=last_aim_x,
        last_aim_y=last_aim_y,
        last_aim_weapon=last_aim_weapon,
        last_aim_ms=last_aim_ms,
    )


# =============================================================================
# PathStepDict
# =============================================================================


class PathStepDict(TypedDict):
    """A single step in a computed path.

    Attributes:
        x: X coordinate of this step.
        y: Y coordinate of this step.
    """

    x: int
    y: int


def make_path_step(x: int, y: int) -> PathStepDict:
    """Create a PathStepDict.

    Args:
        x: X coordinate.
        y: Y coordinate.

    Returns:
        PathStepDict with the provided values.
    """
    return PathStepDict(x=x, y=y)


# =============================================================================
# AIConfigDict
# =============================================================================


class AIConfigDict(TypedDict):
    """Tunable AI parameters.

    Attributes:
        fuel_low_threshold: Below this the bot enters COLLECT. Also
            the reserve a combat teleport must leave behind -- engaging
            below it would flip priority to COLLECT the next tick.
            (The historical ``fuel_critical_threshold`` was collapsed
            into this single value 2026-06-22; the two-tier "polite low
            vs. emergency critical" distinction was dead because both
            thresholds had drifted to the same number.)
        hunt_min_fuel: Operating reserve for search/recovery teleport hops.
        combat_range: Maximum Manhattan distance to engage an enemy.
        scan_cooldown_ms: Minimum milliseconds between radar scans.
        shot_feedback_timeout_ms: Milliseconds to wait before treating a shot as a miss.
        action_stall_timeout_ms: Milliseconds to wait before abandoning a stuck move/pickup.
        kill_cooldown_ms: Milliseconds to ignore a killed tank (avoid targeting corpse).
        map_open_cooldown_ms: Minimum milliseconds between map open commands.
        patrol_waypoints: Circuit of waypoints for PATROL behavior.
        dual_break_threshold: Emergency restock threshold for combat
            reserves. Applies to dual shots and homing shots only;
            extra radar has its own thresholds (radars are a search
            resource whose recovery SPENDS radars, so they were split
            out after the live run 20260611-232301 death spiral).
        radar_break_threshold: Extra-radar count at or below which the
            bot enters equipment restock to rebuild radars before
            hunting. The grid-sweep forager handles the zero case.
            (Restock-EXIT levels are not configured: the 2026-07-25
            hunt-only-when-full contract derives them from the rank
            caps -- ``inventory_capacity(rank)`` for weapons, cap-5
            for radars, ``fuel_capacity(rank)`` for fuel.)
        engagement_fuel_budget: Estimated fuel a typical kill consumes
            once adjacent (shot sequence + per-tick position cost; the
            approach teleport is priced separately per candidate in
            ``find_acquisition_target``). Starting a new engagement
            requires ``fuel >= fuel_low_threshold +
            engagement_fuel_budget``, and acquiring a map-known enemy
            additionally requires the candidate's exact teleport cost
            on top -- the bot never picks a fight it cannot pay for
            (user contract 2026-07-02). Mid-engagement
            (``combat_target_id != -1``) bypasses the mode-level gate
            -- a kill in progress is finished even on a tight budget.
        priority_target_name: Account name that outranks every other
            target at acquisition (case-insensitive; ``""`` for none).
            Human-classified enemies already outrank practice bots
            unconditionally ([[bot-behavior-contract]] §3.2, user
            doctrine 2026-07-28); this names the tier above that.
            Wired from ``TANKPIT_BOT_PRIORITY_TARGET``.
        human_target_min_rank: Lowest human rank the bot may target
            (ranks are integers, 0 recruit .. 8 general). Default 1:
            recruits are never targeted (user ruling 2026-07-28).
            Wired from ``TANKPIT_BOT_HUMAN_MIN_RANK``.
        human_target_max_rank: Highest human rank the bot may target.
            Default 8 (no ceiling); a main-map bot can lower it to
            leave high ranks alone out of respect (user doctrine
            2026-07-28). Wired from ``TANKPIT_BOT_HUMAN_MAX_RANK``.
            Practice bots are farmed at any rank -- the window
            applies to human-classified enemies only.
            Recalibrated 2026-07-02 from wire data: shots cost ~45
            fuel plus ~10/tick position drain (live runs 2026-07-01)
            and practice-room kills take ~8-10 hits (recorded human
            sessions), so a realistic kill costs ~450. The earlier
            value (200) let the bot start fights it could not finish:
            run 2026-07-01 20:45 burned 505 fuel on the approach and
            hit the fuel-low interrupt 8 shots into the kill.
    """

    fuel_low_threshold: int
    hunt_min_fuel: int
    combat_range: int
    scan_cooldown_ms: int
    shot_feedback_timeout_ms: int
    action_stall_timeout_ms: int
    kill_cooldown_ms: int
    map_open_cooldown_ms: int
    patrol_waypoints: list[tuple[int, int]]
    dual_break_threshold: int
    radar_break_threshold: int
    engagement_fuel_budget: int
    priority_target_name: str
    human_target_min_rank: int
    human_target_max_rank: int


def make_default_ai_config() -> AIConfigDict:
    """Create AIConfigDict with sensible defaults.

    Returns:
        AIConfigDict with default values suitable for lieutenant rank.
    """
    return AIConfigDict(
        fuel_low_threshold=200,
        hunt_min_fuel=100,
        combat_range=20,
        scan_cooldown_ms=5000,
        shot_feedback_timeout_ms=4000,
        action_stall_timeout_ms=10000,
        kill_cooldown_ms=30000,
        map_open_cooldown_ms=5000,
        patrol_waypoints=[(64, 64), (192, 64), (192, 192), (64, 192)],
        dual_break_threshold=4,
        radar_break_threshold=5,
        engagement_fuel_budget=450,
        priority_target_name="",
        human_target_min_rank=1,
        human_target_max_rank=8,
    )


class AIStateDict(TypedDict):
    """Mutable AI tick state tracking current behavior and cooldowns.

    Attributes:
        config: Tunable AI parameters.
        mode: Durable top-level AI mode owner.
        mode_state: Durable substate within the active top-level mode.
        mode_started_ms: Timestamp when the current durable mode was entered.
        last_scan_ms: Timestamp of last radar scan (milliseconds).
        last_shoot_ms: Timestamp of last shot fired (milliseconds).
        last_map_open_ms: Timestamp of last map open command (milliseconds).
        combat_target_id: Tank ID of current combat target (-1 if none).
        break_escape_until_fuel: The engagement-break LATCH (2026-07-29).
            Zero when no escape is in progress; set to the break's
            ``escape_floor`` the tick the damage-aware break fires, and
            held until fuel recovers to that floor. While latched,
            ENGAGE/CLOSE keep delegating to the lock-held refuel
            instead of re-litigating the projection each tick -- the
            projection flickers with its sliding hit window, and the
            un-latched oscillation produced the 21:59 map-fire loop
            (break -> fuel-hop defers to map_open -> next tick the
            projection recovered -> shot fired -> the SHOT CLOSED THE
            MAP -> break again -> map_open again, bleeding 27-36
            fuel/tick; user report: "stuck in a map loop... it seems
            to be queing both the map open and fire command").
        wind_down: True in the session's final stretch — no new
            engagements; disengage, top off, and exit cleanly
            (``session_complete``) once fully stocked.
        combat_target_x: X coordinate of combat target.
        combat_target_y: Y coordinate of combat target.
        killed_tank_ids: Tank IDs on kill cooldown {str(tank_id): timestamp_ms}.
        blocked_combat_targets: Tank IDs that are temporarily unengageable
            (e.g. no passable landing tile). {str(tank_id): timestamp_ms}.
            Expired by the same TTL as killed_tank_ids.
        last_shot_target_id: Tank ID we shot at last tick (-1 if none).
        last_shot_target_name: Name of tank we shot at last tick.
        resource_target_kind: Locked resource target kind ("", "fuel", or
            "equipment"). Used to continue an in-progress pickup plan across
            teleports and viewport recentering.
        resource_target_x: X coordinate of the locked resource target.
        resource_target_y: Y coordinate of the locked resource target.
        attempted_equipment_targets: Equipment targets that have been
            teleport-approached. {``"x,y"``: timestamp_ms}. Prevents
            repeated orbits around the same container.
        last_landing_scan_viewport: ``"left,top"`` origin of the last
            viewport a landing radar was dispatched in ("" before the
            first). The viewport changes only on teleport, so this is
            a one-radar-per-landing latch: HUNT's and COLLECT's
            scan-on-landing both record it, and COLLECT fires its
            landing radar only when the current origin differs
            (user policy 2026-07-03: always radar right on landing,
            before any pickup).
        suppress_landing_scan: True while a larder hop is in flight.
            The larder ruling (user, 2026-07-27, [[larder-plan]]) is
            that knowledge-based harvest hops never spend a radar on
            the landing viewport -- the target is already verified
            and hidden tiles stay hidden. COLLECT's scan-on-landing
            consumes the flag by latching the new viewport origin
            without dispatching the radar.
        manual_mode: When not ``None``, the durable mode the SPA has
            pinned the arbitrator to. ``"UNSET"`` means the bot is
            connected but idle (no ticks dispatched, hold position).
            ``"HUNT"`` / ``"COLLECT"`` force those modes. ``None``
            restores the built-in auto-arbitration and mirrors the
            historical (pre-service) behaviour of :func:`make bot`.
            Drained from :mod:`tankpit_bot.service.mode_bridge` at the
            top of every tick.
        live_radars_used: Radar-scan commands dispatched by the executor
            this session. Incremented at the radar dispatch call-site
            in :mod:`tankpit_bot.bot.executor`. Distinct from the
            end-of-session ``scans_extra`` / ``scans_builtin`` totals
            in :class:`ScorecardAccumulatorDict`, which are rolled up
            from the wire event stream — this counter is the live
            executor-side view the SPA renders in real time.
        live_teleports: Teleport commands dispatched by the executor
            this session. Same rationale as :attr:`live_radars_used` —
            live executor-side counter, not the wire-derived scorecard
            aggregate.
        greeted_target_id: Tank ID of the last human target greeted
            with the HELLO chat (-1 before the first). The greeting
            attaches once per human lock acquisition; the latch stops
            re-greets while the same lock is re-derived tick after
            tick — the server's chat flood mute silently swallows
            spam for the rest of the session ([[chat-messages]]).
    """

    config: AIConfigDict
    mode: AIMode
    mode_state: AIModeState
    mode_started_ms: int
    last_scan_ms: int
    last_shoot_ms: int
    last_map_open_ms: int
    combat_target_id: int
    wind_down: bool
    break_escape_until_fuel: int
    combat_target_x: int
    combat_target_y: int
    killed_tank_ids: dict[str, int]
    session_kill_count: int
    session_hit_count: int
    session_miss_count: int
    session_reject_count: int
    blocked_combat_targets: dict[str, int]
    last_shot_target_id: int
    last_shot_target_name: str
    resource_target_kind: str
    resource_target_x: int
    resource_target_y: int
    attempted_equipment_targets: dict[str, int]
    last_landing_scan_viewport: str
    suppress_landing_scan: bool
    manual_mode: AIMode | None
    live_radars_used: int
    live_teleports: int
    greeted_target_id: int


def make_initial_ai_state(
    config: AIConfigDict | None = None,
) -> AIStateDict:
    """Create initial AI state.

    Args:
        config: Optional AI config. Uses defaults if None.

    Returns:
        AIStateDict with initial values.
    """
    return AIStateDict(
        config=config if config is not None else make_default_ai_config(),
        mode="UNSET",
        mode_state="",
        mode_started_ms=0,
        last_scan_ms=1,  # Non-zero so radar doesn't auto-fire on first tick
        last_shoot_ms=0,
        last_map_open_ms=0,
        combat_target_id=-1,
        wind_down=False,
        break_escape_until_fuel=0,
        combat_target_x=0,
        combat_target_y=0,
        killed_tank_ids={},
        session_kill_count=0,
        session_hit_count=0,
        session_miss_count=0,
        session_reject_count=0,
        blocked_combat_targets={},
        last_shot_target_id=-1,
        last_shot_target_name="",
        resource_target_kind="",
        resource_target_x=0,
        resource_target_y=0,
        attempted_equipment_targets={},
        last_landing_scan_viewport="",
        suppress_landing_scan=False,
        manual_mode=None,
        live_radars_used=0,
        live_teleports=0,
        greeted_target_id=-1,
    )


__all__ = [
    "BEHAVIOR_MODES",
    "AIConfigDict",
    "AIStateDict",
    "BehaviorMode",
    "BehaviorScoreDict",
    "EnemyThreatDict",
    "PathStepDict",
    "make_behavior_score",
    "make_default_ai_config",
    "make_enemy_threat",
    "make_initial_ai_state",
    "make_path_step",
]
