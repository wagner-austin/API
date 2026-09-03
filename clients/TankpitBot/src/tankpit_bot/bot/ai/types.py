"""AI session state and configuration.

The durable per-session record the tick loop carries and the config
that tunes it. The scoring vocabulary is
:mod:`tankpit_bot.bot.ai.scoring_types`; the derived world views are
:mod:`tankpit_bot.bot.ai.world_types`.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.fleetshare.types import EngagementDoctrine, FleetRole
from tankpit_bot.types.modes import (
    AIMode,
    AIModeState,
)


class AIConfigDict(TypedDict):
    """Tunable AI parameters.

    Attributes:
        fuel_low_threshold: Below this the bot enters COLLECT. Also
            the reserve a combat teleport must leave behind -- engaging
            below it would flip priority to COLLECT the next tick.
            (The historical ``fuel_critical_threshold`` was collapsed
            into this single value 2026-06-22; the two-tier "polite low
            vs. emergency critical" distinction was dead because both
            thresholds had drifted to the same number.) Like the two
            reserves below, this is the RANK-4 REFERENCE tuning: every
            decision reads it rank-scaled through
            ``DecideCtx.fuel_low_floor``
            (:func:`~tankpit_bot.physics.capacity.rank_scaled_reserve`,
            [[flag-triage-20260902]] row 6 -- a flat lieutenant-tuned
            floor broke a private off a winnable fight at full tank).
        hunt_min_fuel: Operating reserve for search/recovery teleport
            hops. Rank-4 reference; read via
            ``DecideCtx.hunt_reserve_floor``.
        combat_range: Maximum Manhattan distance to engage an enemy.
        scan_cooldown_ms: Minimum milliseconds between radar scans.
        shot_feedback_timeout_ms: Milliseconds to wait before treating a shot as a miss.
        action_stall_timeout_ms: Milliseconds to wait before abandoning a stuck move/pickup.
        kill_cooldown_ms: Milliseconds to ignore a killed tank (avoid targeting corpse).
        map_open_cooldown_ms: Minimum milliseconds between map open commands.
        map_intel_horizon_ms: How long map-sourced knowledge (tank
            positions from a MAP_DATA snapshot) stays actionable for
            acquisition, pursuit, greeting, and the no-viable-targets
            gate. Split from ``map_open_cooldown_ms`` 2026-08-26: one
            constant served as both the re-open cooldown AND the
            freshness bar, but map answers measure 2-6 s of latency
            (bot-20260825-212920, 560 answers), so a 5 s freshness
            bar declared snapshots stale almost on arrival — a sixth
            of that marathon went to 559 map opens re-asking the same
            question. 12 s = cooldown + worst common answer latency +
            one decision tick; practice bots barely move in 12 s.
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
            ``find_acquisition_target``). Rank-4 reference; read via
            ``DecideCtx.engagement_budget``. Starting a new engagement
            requires ``fuel >= fuel_low_floor +
            engagement_budget``, and acquiring a map-known enemy
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
        role: This bot's :data:`~tankpit_bot.fleetshare.types.FleetRole`.
            Wired from ``TANKPIT_ROLE``.
        doctrine: This bot's
            :data:`~tankpit_bot.fleetshare.types.EngagementDoctrine` —
            how it times human engagements (operator order
            2026-09-01). Wired from ``TANKPIT_DOCTRINE``; default
            ``"skirmish"`` is the pre-doctrine behavior.
    """

    fuel_low_threshold: int
    hunt_min_fuel: int
    combat_range: int
    scan_cooldown_ms: int
    shot_feedback_timeout_ms: int
    action_stall_timeout_ms: int
    kill_cooldown_ms: int
    map_open_cooldown_ms: int
    map_intel_horizon_ms: int
    patrol_waypoints: list[tuple[int, int]]
    dual_break_threshold: int
    radar_break_threshold: int
    engagement_fuel_budget: int
    priority_target_name: str
    human_target_min_rank: int
    human_target_max_rank: int
    role: FleetRole
    doctrine: EngagementDoctrine


def make_default_ai_config() -> AIConfigDict:
    """Create AIConfigDict with sensible defaults.

    Returns:
        AIConfigDict defaults. The three fuel reserves are the RANK-4
        (lieutenant, capacity 1400) reference tuning; every decision
        reads them scaled to the tank's true capacity
        (:func:`~tankpit_bot.physics.capacity.rank_scaled_reserve`),
        so the numbers here stay exact at lieutenant and shrink or
        grow proportionally elsewhere.
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
        map_intel_horizon_ms=12000,
        patrol_waypoints=[(64, 64), (192, 64), (192, 192), (64, 192)],
        dual_break_threshold=4,
        radar_break_threshold=5,
        engagement_fuel_budget=450,
        priority_target_name="",
        human_target_min_rank=1,
        human_target_max_rank=8,
        role="fighter",
        doctrine="skirmish",
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
        resource_target_held_ticks: Consecutive continuation ticks the
            locked target was held WITHOUT producing a dispatch. The
            lock-continuation increments it on every hold; latching or
            clearing the lock resets it to 0. At
            :data:`~tankpit_bot.bot.ai.intent.RESOURCE_LOCK_HOLD_BOUND_TICKS`
            the continuation releases the plan (reason
            ``progress_stalled``) — the last-resort progress invariant
            behind the 2026-09-02 livelock ([[flag-triage-20260902]]).
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
            Drained from :mod:`tankpit_bot.bus.mode_bridge` at the
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
        mine_clearance_aim_key: ``"x,y"`` of the last mine-clearance
            shot's aim tile, or ``""``. Paired with
            ``mine_clearance_shot_ms`` to keep the planner from
            re-aiming a tile whose detonation (0x45) has not landed
            yet -- the live double-shot at (162,94), 01:59:57/:59,
            run bot-20260730-015x.
        mine_clearance_shot_ms: Dispatch timestamp of that shot.
        mine_pin_target_id: Tank id whose engagement already spent its
            one mine-pin press (operator order 2026-09-01: pin an
            adjacent enemy with the 3x3 ``CMD_MINE`` pattern), or -1.
            Re-arms by lock movement: a NEW target id differs from
            the latch, so its first close engage tick presses again.
        greeted_tank_ids: Tank IDs already given the one-shot HELLO,
            {str(tank_id): greeting timestamp_ms}. A PER-ID map, not
            a last-id scalar: the 2026-07-31 two-human arena soak
            showed the scalar latch ping-ponging between two humans —
            12 HELLOs in one session, past the server's 8-send flood
            mute that silences chat for the rest of the session
            ([[chat-messages]]).
        pursuit_shot_target_id: Tank ID of the last pursuit-fire
            target (-1 before the first). Paired with
            ``pursuit_shot_ms`` to cap pursuit homings at ONE per
            departure window against humans (user ruling 2026-07-31:
            milking the ~12 s reroute wall for up to 7 tracked hits
            is cheating). The window needs no explicit reset: the
            budget is spent while the stamp is newer than the
            target's ``last_viewport_observation_ms``, so the target
            re-entering the viewport re-arms it naturally.
        pursuit_shot_ms: Dispatch timestamp of that pursuit shot.
        visited_tank_ids: Tank IDs already given the stand-off GREET
            VISIT, {str(tank_id): visit timestamp_ms}. Decoupled from
            ``greeted_tank_ids`` (user ruling 2026-07-31: "hello can
            run anytime... as long as the other player is on the map
            logged in. you dont have to be near them") — the HELLO
            may fire long before the visit, and sharing one latch
            made an early long-range HELLO cancel the visit entirely
            (first human-opponent sim soak). Per-id like the greeting
            map so multiple unconsenting humans each get exactly one
            courtesy trip; consent (their chat or first strike) is
            what admits them to acquisition.
        last_scope_scout_ms: Timestamp of the last ferry scope-scout
            (the free Rb viewport pan toward a water-locked goal,
            [[viewport-shift-protocol]]). Cooldown latch: a pan that
            reveals no ferry leaves no negative belief behind, so
            without it the scout would re-fire every tick the larder
            declines the same water-locked container.
        sweep_anchor_x: X of the quad sweep's anchor tile (-1 when no
            sweep is latched). The sweep is atomic ([[quad-sweep-doctrine]]):
            it continues only while the tank stands exactly on this
            tile, so ANY movement abandons the remaining quadrants and
            the block-freshness gate governs the next start.
        sweep_anchor_y: Y of the quad sweep's anchor tile (-1 when no
            sweep is latched).
        maroon_pan_x: X of the tank's position at the last marooned-walk
            viewport pan dispatch (-1 before any). The movement law: a
            pan must pay for itself in movement before the next one, so
            the walk-for-fuel rung refuses a second pan from the exact
            latched tile. Without it two stuck candidates on opposite
            sides would ping-pong the free window forever (run
            bot-20260825-133452 oscillated 331 s between clamp tiles;
            the pan gait must not inherit the loop).
        maroon_pan_y: Y of that latched position (-1 before any).
        forage_goal_x: X of the latched forage-frontier goal block
            center (-1 when none). The latch keeps the goal stable
            across the beats its travel spends (a teleport's map-open
            defer), so the next tick's replan serves the same goal
            instead of swapping targets mid-prelude (the atlas hop's
            open-for-40-throw-227 waste, run bot-20260828-192801
            19:31:00). Released by arrival (the block is tombstoned)
            or by the goal failing its own qualification.
        forage_goal_y: Y of that latched goal (-1 when none).
        forage_goal_attempts: Travel dispatches served toward the
            latched goal. The displacement law can bounce a landing
            far from an unlandable block center; without a cap the
            goal never comes within the arrival radius and the
            frontier re-throws forever (arterial 2026-08-28 20:52:
            14+ teleports at (120,104), every landing 9-25 tiles
            out). At the cap the block is tombstoned as unlandable.
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
    resource_target_held_ticks: int
    attempted_equipment_targets: dict[str, int]
    last_landing_scan_viewport: str
    suppress_landing_scan: bool
    manual_mode: AIMode | None
    live_radars_used: int
    live_teleports: int
    mine_clearance_aim_key: str
    mine_clearance_shot_ms: int
    mine_pin_target_id: int
    greeted_tank_ids: dict[str, int]
    pursuit_shot_target_id: int
    pursuit_shot_ms: int
    visited_tank_ids: dict[str, int]
    last_scope_scout_ms: int
    sweep_anchor_x: int
    sweep_anchor_y: int
    maroon_pan_x: int
    maroon_pan_y: int
    forage_goal_x: int
    forage_goal_y: int
    forage_goal_attempts: int


def make_respawn_ai_state(previous: AIStateDict) -> AIStateDict:
    """Rebuild AI state after a death, preserving session-scoped fields.

    A death resets LIFE-scoped state (locks, plans, blacklists — the
    dead tank's tactical context) but must NOT reset SESSION-scoped
    state: the scorecard counters and the social maps. The pre-lift
    respawn path hand-listed the survivors inline and dropped the
    hit/miss/reject counters — run bot-20260803-180918's summary
    printed 23 shots against 223 actual wire shoots because the one
    death zeroed them mid-session.

    Session-scoped survivors, and why:

    * ``session_kill_count`` / ``session_hit_count`` /
      ``session_miss_count`` / ``session_reject_count`` — the session
      scorecard; a death is an event IN the session, not a new one.
    * ``wind_down`` — the shutdown phase belongs to the session clock.
    * ``greeted_tank_ids`` / ``visited_tank_ids`` — the social
      contracts: dying to a human must not schedule a fresh HELLO or
      courtesy visit ([[chat-messages]] flood-mute; the stand-off
      greeting is once per human per session).

    Args:
        previous: The AI state at the moment of deactivation.

    Returns:
        A fresh initial state carrying the session-scoped fields.
    """
    fresh = make_initial_ai_state(previous["config"])
    return AIStateDict(
        **{
            **fresh,
            "session_kill_count": previous["session_kill_count"],
            "session_hit_count": previous["session_hit_count"],
            "session_miss_count": previous["session_miss_count"],
            "session_reject_count": previous["session_reject_count"],
            "wind_down": previous["wind_down"],
            "greeted_tank_ids": previous["greeted_tank_ids"],
            "visited_tank_ids": previous["visited_tank_ids"],
        }
    )


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
        resource_target_held_ticks=0,
        attempted_equipment_targets={},
        last_landing_scan_viewport="",
        suppress_landing_scan=False,
        manual_mode=None,
        live_radars_used=0,
        live_teleports=0,
        mine_clearance_aim_key="",
        mine_clearance_shot_ms=0,
        mine_pin_target_id=-1,
        greeted_tank_ids={},
        pursuit_shot_target_id=-1,
        pursuit_shot_ms=0,
        visited_tank_ids={},
        last_scope_scout_ms=0,
        sweep_anchor_x=-1,
        sweep_anchor_y=-1,
        maroon_pan_x=-1,
        maroon_pan_y=-1,
        forage_goal_x=-1,
        forage_goal_y=-1,
        forage_goal_attempts=0,
    )


__all__ = [
    "AIConfigDict",
    "AIStateDict",
    "make_default_ai_config",
    "make_initial_ai_state",
    "make_respawn_ai_state",
]
