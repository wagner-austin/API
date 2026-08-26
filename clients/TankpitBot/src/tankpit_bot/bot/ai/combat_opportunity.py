"""Opportunity fire during a held lock: finishers and return fire.

The firefight doctrine (user ruling 2026-08-14): "you have a main
target ofc, but you should also return fire to anyone else engaging
and take kill shots ... when someone is in the lowest or second
lowest damage state." The held combat lock stays the MAIN target --
these are single-shot diverts from the current tile, never a lock
change, never a movement: only enemies with an immediate legal shot
(in the window, in range, clear line) qualify.

Candidate legality is inherited wholesale from
:func:`~tankpit_bot.bot.ai.threats.analyze_threats` over the
blocked-filtered world: liveness, position confirmation, viewport
observation, the human rank window, and the human-consent contract
all apply -- an unconsenting human is never a "finisher", and an
attacker has consented by attacking.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import (
    _find_combat_pickup,
    has_cardinal_combat_shot,
    has_combat_shot,
)
from tankpit_bot.bot.ai.combat_target import has_clear_shot_line
from tankpit_bot.bot.ai.context import DecideCtx, make_decision
from tankpit_bot.bot.ai.mode_gates import weapon_reserves_below_break
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_shoot_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

RETURN_FIRE_WINDOW_MS = 6_000
"""How long a confirmed incoming hit keeps its shooter "engaging us".

Sized to the incoming cadence (one shot per ~2 s server window): three
windows of silence means the attacker broke off, and return fire
stops diverting shots from the main target."""

FINISHER_MAX_DAMAGE_STATE = 1
"""Highest damage tier that counts as a kill-shot opportunity.

Tiers are fuel quartiles (corpus-fitted 2026-07-23, 0 = near death ..
3 = healthy); the user ruling names "the lowest or second lowest
damage state". Unknown tanks default to 3 and never qualify."""


def recent_attacker_ids(ctx: DecideCtx) -> set[int]:
    """Return shooter ids with a fuel-confirmed hit on us in the window.

    Reads the damage book's confirmed-incoming log -- the same
    fuel-confirmed evidence the engagement break trusts, so a counted
    but unconfirmed shot never provokes return fire.

    Args:
        ctx: Decision context.

    Returns:
        Tank ids that struck us within :data:`RETURN_FIRE_WINDOW_MS`.
    """
    floor = ctx.timestamp_ms - RETURN_FIRE_WINDOW_MS
    return {
        hit["shooter_id"]
        for hit in ctx.ws.damage_book["confirmed_incoming"]
        if hit["timestamp_ms"] >= floor
    }


def _has_immediate_shot(ctx: DecideCtx, target: EnemyThreatDict) -> bool:
    """Return True when the target can be shot from the current tile now.

    A divert never moves, shifts, or teleports -- it is a free
    single-window opportunity, so the target must be inside the
    visible window (the visibility law, flag s11-2) and either
    cardinally adjacent or in range with a clear dual line.

    Args:
        ctx: Decision context.
        target: Candidate enemy threat.

    Returns:
        True when a legal shot exists from where the bot stands.
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    if not (left <= target["x"] <= right and top <= target["y"] <= bottom):
        return False
    return has_cardinal_combat_shot(ctx.self_state, target) or (
        has_combat_shot(ctx, target) and has_clear_shot_line(ctx, target)
    )


def _finisher_rank(target: EnemyThreatDict) -> tuple[int, int]:
    """Order finishers most-damaged first, nearest breaking ties."""
    return (target["damage_state"], target["distance"])


def select_opportunity_shot(ctx: DecideCtx, main_target_id: int) -> EnemyThreatDict | None:
    """Pick the enemy a single shot should divert to this tick, if any.

    Priority per the doctrine: a FINISHER (visible enemy at damage
    tier <= :data:`FINISHER_MAX_DAMAGE_STATE` -- about to die, the
    highest-value shot in the game) beats RETURN FIRE (a visible
    enemy with a confirmed hit on us inside the window), and both
    beat the main lock's routine shot. The main target itself is
    never a "divert" -- when it is the finisher or the attacker, the
    ordinary engage path already serves it.

    Args:
        ctx: Decision context.
        main_target_id: The main target the calling combat path is
            serving (the held lock, or the fresh acquire about to
            become it).

    Returns:
        The divert target, or ``None`` when every shot belongs to the
        main target.
    """
    lock_id = main_target_id
    shootable = [
        threat
        for threat in analyze_threats(
            ctx.ws,
            ctx.filtered,
            ctx.self_state,
            ctx.timestamp_ms,
            human_min_rank=ctx.config["human_target_min_rank"],
            human_max_rank=ctx.config["human_target_max_rank"],
        )
        if threat["tank_id"] != lock_id
        # ``ctx.filtered`` removes killed tanks only -- the blocked
        # map (shielded/afterimage cooldowns, incl. a diverted miss's
        # own block) is a separate exclusion.
        and str(threat["tank_id"]) not in ctx.blocked_targets
        and _has_immediate_shot(ctx, threat)
    ]
    if not shootable:
        return None
    finishers = sorted(
        (threat for threat in shootable if threat["damage_state"] <= FINISHER_MAX_DAMAGE_STATE),
        key=_finisher_rank,
    )
    if finishers:
        return finishers[0]
    attackers = recent_attacker_ids(ctx)
    for threat in shootable:
        if threat["tank_id"] in attackers:
            return threat
    return None


def opportunity_fire(
    ctx: DecideCtx,
    target: EnemyThreatDict,
    ai_state: AIStateDict,
) -> TickDecisionDict:
    """Dispatch one diverted shot without touching the held lock.

    The lock fields (``combat_target_*``) and the engagement stamp
    (``engaged_target_id``) stay untouched -- the main fight resumes
    the next tick. The per-shot feedback fields DO move to the divert
    (``last_shot_target_id`` keys the ammo-consumption receipt), and
    :func:`~tankpit_bot.bot.ai.combat_strategy.engage_target` scopes
    its miss/rejection consequences to shots aimed at its own target,
    so a diverted miss can never block the main lock.

    The adjacent-container pickup rides as the secondary command, the
    same free rider ``engage_target`` gives its own shots (operator
    ruling 2026-08-26, Yuppler receipt: ticks under fire are DAMAGE
    ticks — the refill costs the shot nothing).

    Args:
        ctx: Decision context.
        target: The divert target from :func:`select_opportunity_shot`.
        ai_state: AI state the decision builds on (the caller's
            threaded base, so gate-latched fields survive the divert).

    Returns:
        The diverted shoot decision.
    """
    emit_ai(
        "opportunity fire at %s (%d,%d) damage=%d - main target held",
        target["name"],
        target["x"],
        target["y"],
        target["damage_state"],
    )
    secondary = _find_combat_pickup(ctx)
    if secondary is not None:
        emit_ai("mid-combat pickup %s", secondary["cmd_type"])
    return make_decision(
        make_shoot_command(target["x"], target["y"], target["tank_id"]),
        "HUNT",
        800,
        target["x"],
        target["y"],
        "opportunity_shot",
        AIStateDict(
            **{
                **ai_state,
                "last_shoot_ms": ctx.timestamp_ms,
                "last_shot_target_id": target["tank_id"],
                "last_shot_target_name": target["name"],
            }
        ),
        ctx.equip,
        reason_context={"target_name": target["name"]},
        secondary_command=secondary,
    )


def opportunity_shot_decision(
    ctx: DecideCtx,
    main_target_id: int,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Return the divert decision when one applies, else ``None``.

    The single entry the combat paths call before serving their main
    target.

    Args:
        ctx: Decision context.
        main_target_id: The main target the caller is serving.
        ai_state: AI state the divert decision builds on.

    Returns:
        Diverted shot decision, or ``None`` when the tick belongs to
        the main target.
    """
    target = select_opportunity_shot(ctx, main_target_id)
    if target is None:
        return None
    return opportunity_fire(ctx, target, ai_state)


def collect_return_fire(ctx: DecideCtx, base_state: AIStateDict) -> TickDecisionDict | None:
    """Return fire from a COLLECT-owned tick while stock permits.

    The Yuppler receipt (run bot/arterial 2026-08-26 03:14): ten human
    shots landed while the between-kills restock out-collected the
    damage on a 1,025-volume pile — and the first return shot waited
    37 s for the tank to touch cap, because only combat paths consult
    the opportunity doctrine. Operator ruling 2026-08-26: a consented
    attacker in view makes ticks under fire DAMAGE ticks — the divert
    fires from where the tank stands and the pickup rides as the
    secondary command, so returning fire costs the refill nothing.

    The 2026-07-25 survival contract stays senior: at or below the
    fuel-low break, or with any weapon reserve below its break bar,
    this rung declines and the escape doctrine owns the tick
    unchanged. Gatherers never fire ([[fleet-coordination]] role
    gate). The main-target exclusion is the HELD lock: a COLLECT tick
    with a live ``combat_target_id`` is a break-restock, and the
    broken-from enemy belongs to the solvency law and its resume
    machinery — the first live hour of this rung proved the ``-1``
    variant re-fights the exact enemy the break just walked away from
    (artax vs red-8, 03:50:16: break at projected fuel 318 < floor
    354, then six return shots at the same tank, fuel 851→686). With
    no lock held (the Yuppler shape) the exclusion matches nobody and
    every consented attacker qualifies.

    Args:
        ctx: Decision context.
        base_state: AI state threaded from the collect gates.

    Returns:
        The return-fire decision, or ``None`` when no confirmed recent
        attacker has a legal immediate shot or the survival bars veto.
    """
    if ctx.config["role"] == "gatherer":
        return None
    if ctx.fuel <= ctx.config["fuel_low_threshold"] or weapon_reserves_below_break(ctx):
        return None
    if not recent_attacker_ids(ctx):
        return None
    return opportunity_shot_decision(ctx, base_state["combat_target_id"], base_state)


__all__ = [
    "FINISHER_MAX_DAMAGE_STATE",
    "RETURN_FIRE_WINDOW_MS",
    "collect_return_fire",
    "opportunity_shot_decision",
    "select_opportunity_shot",
]
