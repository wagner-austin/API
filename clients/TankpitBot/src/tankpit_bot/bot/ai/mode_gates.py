"""Entry and exit gates for the durable HUNT/COLLECT modes.

Every predicate the arbitrator consults to decide whether a mode may
be entered or must be left -- fuel floors, weapon reserves, radar
minimums, and the human-combat lock. The arbitrator that applies them
is :mod:`tankpit_bot.bot.ai.mode_controller`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.tactics import combat_radar_min
from tankpit_bot.bot.ai.threat_primitives import human_combat_consented
from tankpit_bot.bot.config import resolve_weapon_resume_slack
from tankpit_bot.physics.capacity import fuel_capacity, inventory_capacity
from tankpit_bot.protocol.naming import is_human_name


def should_enter_collect(ctx: DecideCtx) -> bool:
    """Return True when the unified COLLECT mode should own planning.

    Entry triggers across fuel and equipment:

    * **Fuel low** -- at or below the fuel-low threshold. Interrupts
      even an active combat target (user contract 2026-07-25: the
      2026-07-13 cardinal override let a fight outrank this break and
      the bot died trading at 84 fuel in the practice-room gang-up).
    * **Weapon emergency** -- any weapon reserve below its break
      threshold, or extra radars below the radar break threshold.
      Interrupts even an active combat target.
    * **Between kills** -- no active combat target AND anything short
      of a genuinely full tank: fuel below the rank capacity or
      inventory below the rank caps (user contract 2026-07-25: "never
      hunt if it is not full on everything except -5 max radar"; caps
      are rank-derived, replacing the fixed resume thresholds that
      under-restocked high ranks). Finishes the current kill first,
      then restocks fully before the next hunt.

    Args:
        ctx: Decision context.

    Returns:
        True when fuel or equipment reserves require collection.
    """
    if ctx.fuel <= ctx.config["fuel_low_threshold"]:
        return True
    if weapon_reserves_below_break(ctx):
        return True
    if ctx.ai_state["combat_target_id"] != -1:
        return False
    return ctx.fuel < hunt_fuel_floor(ctx) or not hunt_entry_permitted(ctx)


def weapon_reserves_below_break(ctx: DecideCtx) -> bool:
    """Return True when any weapon or radar reserve is below its break bar.

    The genuine emergency bar (dual / homing below
    ``dual_break_threshold``, extra radars below
    ``radar_break_threshold``) -- as opposed to the hunt-ENTRY bar of
    exact rank caps. The entry-vs-held distinction (F21,
    [[flag-triage-20260729]]): entry bars govern starting a fight
    fully stocked; a HELD fight tops up only at this break bar, never
    the entry cap.

    Args:
        ctx: Decision context.

    Returns:
        True when any reserve is below its break threshold.
    """
    return (
        ctx.inventory["dual_shots"]["count"] < ctx.config["dual_break_threshold"]
        or ctx.inventory["homing_shots"]["count"] < ctx.config["dual_break_threshold"]
        or ctx.inventory["extra_radars"]["count"] < ctx.config["radar_break_threshold"]
    )


def hunt_fuel_floor(ctx: DecideCtx) -> int:
    """Return the fuel level that counts as a full tank for HUNT entry.

    The rank's actual fuel capacity (user ruling 2026-07-25: "just
    determine max fuel based on the tank rank") -- 1000 at recruit
    through 1800 at general. The collect cascade's pickup ceiling is
    the same physics number, so "stop collecting fuel" and "may hunt"
    can never disagree and deadlock the owner selection.

    Args:
        ctx: Decision context.

    Returns:
        ``fuel_capacity(rank)`` for the bot's current rank.
    """
    return fuel_capacity(ctx.self_state["rank"])


def should_exit_collect(ctx: DecideCtx) -> bool:
    """Return True when COLLECT can release control.

    The mode holds until the bot is FULLY restocked: fuel at the
    rank-clamped full floor AND the inventory combat-ready
    (:func:`hunt_entry_permitted` -- duals and homings at cap, extra
    radars within 5 of cap; user contract 2026-07-25: "never hunt if
    it is not full on everything except -5 max radar"). The
    entry-at-break / exit-at-full gap gives hysteresis, so the bot
    rebuilds a full stock instead of leaving the moment it scrapes
    together one radar.

    Mid-HUMAN-fight exception (user ruling 2026-07-31: "its not
    necessary to full restock but to do partial restocks so the bot
    can keep fighting the human"): while a human combat lock is held,
    the exit bar drops to the combat-resume bar
    (:func:`human_fight_resume_permitted`) so the restock detour stays
    short and the fight resumes aggressively. Wind-down keeps the full
    bar -- the session-complete exit must leave a fully stocked tank.

    Args:
        ctx: Decision context.

    Returns:
        True when the applicable restock bar is met.
    """
    if held_human_combat_lock(ctx) and not ctx.ai_state["wind_down"]:
        return human_fight_resume_permitted(ctx)
    if ctx.fuel < hunt_fuel_floor(ctx):
        return False
    return hunt_entry_permitted(ctx)


def held_human_combat_lock(ctx: DecideCtx) -> bool:
    """Return True when the held combat lock is on a human-classified target.

    Args:
        ctx: Decision context.

    Returns:
        True when ``combat_target_id`` names a registry tank whose
        name shape classifies as human. A vanished target (left the
        game) reads False -- the full restock bar applies and the
        resume machinery releases the lock on its own.
    """
    target_id = ctx.ai_state["combat_target_id"]
    tank = ctx.filtered["tanks"].get(str(target_id))
    return tank is not None and is_human_name(tank["name"])


def human_fight_resume_fuel_floor(ctx: DecideCtx) -> int:
    """Return the fuel level at which a broken human fight resumes.

    Shared by the engagement-break latch and the mid-fight restock bar
    (user ruling 2026-07-31): enough to fund the re-entry -- usually
    one good container -- never a full-tank rebuild. Sits above the
    half-capacity break band so the hysteresis never inverts.

    Args:
        ctx: Decision context.

    Returns:
        ``min(capacity, max(fuel_low + hunt_min + engagement_budget,
        capacity // 2 + hunt_min))`` -- 750 at defaults for a private.
    """
    capacity = fuel_capacity(ctx.self_state["rank"])
    return min(
        capacity,
        max(
            ctx.config["fuel_low_threshold"]
            + ctx.config["hunt_min_fuel"]
            + ctx.config["engagement_fuel_budget"],
            capacity // 2 + ctx.config["hunt_min_fuel"],
        ),
    )


def human_fight_resume_permitted(ctx: DecideCtx) -> bool:
    """Return True when a partial restock can resume a held human fight.

    The combat-resume bar (user ruling 2026-07-31, [[bot-behavior-contract]]
    §3.1): fuel at the resume floor, duals AND homings at half the rank
    cap (break fires below ``dual_break_threshold``, so half-cap keeps
    a wide hysteresis band), and extra radars at twice the radar break
    (the bare break bar would be a zero-width band -- the design law
    every re-derived gate follows).

    Args:
        ctx: Decision context.

    Returns:
        True when fuel, weapons, and radars clear the resume bar.
    """
    rank = ctx.self_state["rank"]
    weapon_floor = inventory_capacity(rank) // 2
    radar_floor = min(combat_radar_min(rank), 2 * ctx.config["radar_break_threshold"])
    return (
        ctx.fuel >= human_fight_resume_fuel_floor(ctx)
        and ctx.inventory["dual_shots"]["count"] >= weapon_floor
        and ctx.inventory["homing_shots"]["count"] >= weapon_floor
        and ctx.inventory["extra_radars"]["count"] >= radar_floor
    )


def should_enter_hunt(ctx: DecideCtx) -> bool:
    """Return True when HUNT is the valid top-level owner.

    HUNT is a privilege of a full tank (user contract 2026-07-25):
    fuel at the rank-clamped full floor, duals and homings at cap,
    extra radars within 5 of cap, and no COLLECT trigger pending.
    Starting a fight below full stock leads to abandoned kills when
    the break threshold pulls the bot away mid-fight.

    Mid-HUMAN-fight exception (user ruling 2026-07-31): a held human
    lock re-enters HUNT at the combat-resume bar
    (:func:`human_fight_resume_permitted`) -- without this override
    the partial COLLECT exit and the full entry bar deadlock the
    arbitration and the fight never resumes. The full bar still
    governs STARTING fights (no held lock).

    Args:
        ctx: Decision context.

    Returns:
        True when stocked to the applicable bar and COLLECT has no
        entry condition.
    """
    if held_human_combat_lock(ctx) and human_fight_resume_permitted(ctx):
        return True
    return ctx.fuel >= hunt_fuel_floor(ctx) and hunt_entry_permitted(ctx)


def human_war_is_live(ctx: DecideCtx) -> bool:
    """Return True while a consented enemy human is alive on the map.

    The wartime signal (operator ruling 2026-09-01, the yuppler/TESLA
    case: "he shouldve engaged the person but he kept just farming
    radars"): consent is fleet-shared and session-persistent, so the
    war is LIVE only while some consented human still stands in the
    registry as an alive enemy — a consented human who left the game
    is 0x58-removed or flips liveness, and peacetime restocking
    resumes.

    Args:
        ctx: Decision context.

    Returns:
        True when any alive enemy tank is human-classified and has
        consented (chat, shot us, or a sibling's shared consent).
    """
    for tank in ctx.filtered["tanks"].values():
        if tank["is_self"] or tank["team"] == ctx.self_state["team"]:
            continue
        if tank["liveness"] != "alive":
            continue
        if not is_human_name(tank["name"]):
            continue
        if human_combat_consented(ctx.ws, tank["tank_id"]):
            return True
    return False


def hunt_entry_permitted(ctx: DecideCtx) -> bool:
    """Return True when the bot's inventory permits entering HUNT.

    User contract (2026-07-06, Bug 0.4): the bot must never enter a
    combat engagement below full duals + full homings + at-least
    ``combat_radar_min`` extra radars. The 22:37 live run hit HUNT
    with duals 12/25 + homings 3/25, engaged orange-8 under-armed,
    exhausted its ammo mid-fight, hit the stationary-miss classifier
    (Bug 0.6), and blocked a live target. Enforce the readiness gate
    at every yield-to-hunt gesture: COLLECT never releases the tick
    unless the bot could take the fight to completion.

    The gate is inventory-only; fuel readiness is enforced alongside
    it in :func:`should_enter_hunt` and :func:`should_exit_collect`.
    Nothing bypasses this predicate: the 2026-07-13 cardinal-shot
    override that did was deleted 2026-07-25 (user contract: the bot
    never hunts below full stock, no exceptions).

    Args:
        ctx: Decision context.

    ``TANKPIT_BOT_WEAPON_RESUME_SLACK`` (default 0 = the verbatim
    contract) relaxes the weapons bar to ``cap - slack``, mirroring
    the radar rule's cap-5 shape -- equipment has no map atlas, so an
    exact-cap bar forces a hop-scan discovery loop after every kill
    (nine HUD flags on that loop, 2026-07-29 session).

    Returns:
        True when duals and homings are at ``inventory_capacity(rank)``
        minus the configured slack and extra radars are at least
        ``combat_radar_min(rank)``.
    """
    if ctx.config["role"] == "gatherer":
        # The doctrinal backstop of the router's role gate
        # ([[fleet-coordination]]): a gatherer's ticks NEVER permit
        # hunting, whatever its inventory says — every yield-to-hunt
        # gesture funnels through this predicate.
        return False
    rank = ctx.self_state["rank"]
    cap = inventory_capacity(rank)
    if human_war_is_live(ctx):
        # The wartime floor (operator ruling 2026-09-01, verbatim:
        # "like 80% equipment and 50% radar?"): while a consented
        # human fight is live anywhere on the map, a bot at 80% of
        # its weapon caps and half its radar cap joining NOW is worth
        # more than a full one joining after the human has killed
        # someone or left. The full bar below stays the peacetime
        # law for routine bot-farming, and the fuel bar in
        # should_enter_hunt is untouched — fuel is health and tops
        # up in a pickup or two.
        war_weapon_floor = (cap * 4) // 5
        war_radar_floor = cap // 2
        return (
            ctx.inventory["dual_shots"]["count"] >= war_weapon_floor
            and ctx.inventory["homing_shots"]["count"] >= war_weapon_floor
            and ctx.inventory["extra_radars"]["count"] >= war_radar_floor
        )
    weapon_floor = max(0, cap - resolve_weapon_resume_slack())
    radar_floor = combat_radar_min(rank)
    return (
        ctx.inventory["dual_shots"]["count"] >= weapon_floor
        and ctx.inventory["homing_shots"]["count"] >= weapon_floor
        and ctx.inventory["extra_radars"]["count"] >= radar_floor
    )


def should_exit_hunt(ctx: DecideCtx) -> bool:
    """Return True when HUNT should release control.

    A held HUNT releases only when a COLLECT trigger fires -- fuel at
    the low break, a weapon or radar break, or between-kills resume
    shortfalls. Deliberately NOT ``not should_enter_hunt``: entry
    requires a full stock, and the first shot of a fight spends a
    dual, so re-checking the entry bar every tick would thrash
    ownership one shot into every engagement.

    Args:
        ctx: Decision context.

    Returns:
        True when COLLECT now has an entry condition.
    """
    return should_enter_collect(ctx)


__all__ = [
    "held_human_combat_lock",
    "human_fight_resume_fuel_floor",
    "human_fight_resume_permitted",
    "human_war_is_live",
    "hunt_entry_permitted",
    "hunt_fuel_floor",
    "should_enter_collect",
    "should_enter_hunt",
    "should_exit_collect",
    "should_exit_hunt",
    "weapon_reserves_below_break",
]
