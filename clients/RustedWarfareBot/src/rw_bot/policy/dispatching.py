"""The loop's sending arm: every order leaves through here.

One of exactly two modules allowed to touch the channel -- the campaign
perceives, arbitrates and orders the tick; this module is where a decision
becomes bytes on the wire, verb gate by verb gate. The split is physical
rather than conceptual: the pair IS the loop, and the architecture guard
names them both so a third sender can never grow unnoticed
([[policy-loop]]).

Two kinds of function live here and nothing else does. The plain senders
(:func:`send_moves`, :func:`send_builds`, ...) forward validated orders and
count them for the report. The verb gates (:func:`send_tech`,
:func:`advance_creep`, ...) judge one doctrine switch each and call a pure
policy module when it is on -- judged here rather than in the loop so the
loop stays under its complexity bound, and an arm without the verb costs
one call and no claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.budget import Budget
from rw_bot.policy.creep import Creeper
from rw_bot.policy.decoy import Decoys
from rw_bot.policy.dispatch import WaveController
from rw_bot.policy.hunt import Hunter
from rw_bot.policy.intel import Intel
from rw_bot.policy.lurk import Lurker
from rw_bot.policy.nuker import NukeOrders
from rw_bot.policy.raid import Raider
from rw_bot.policy.rush import Rusher
from rw_bot.policy.scouting import ScoutRunner
from rw_bot.policy.situation import Momentum, strike_window
from rw_bot.policy.spending import PlanStep, unlock_tech
from rw_bot.policy.workforce import Workforce
from rw_bot.wire.command import (
    AttackOrder,
    BuildOrder,
    MoveOrder,
    ProduceOrder,
)
from rw_bot.wire.posture import posture_order
from rw_bot.wire.state import Entity, Sample


def send_plan_step(channel: AgentChannel, step: PlanStep) -> None:
    """Send whatever the opening plan decided on, if anything.

    Args:
        channel: An open connection to the agent.
        step: What the plan decided this observation.

    Raises:
        OSError: When the connection fails.
    """
    if step["produce"] is not None:
        channel.send_produce(step["produce"])
    if step["build"] is not None:
        channel.send_build(step["build"])


def _send_moves(channel: AgentChannel, moves: Sequence[MoveOrder]) -> int:
    """Send every move order and report how many.

    Args:
        channel: An open connection to the agent.
        moves: The orders to send.

    Returns:
        How many were sent.

    Raises:
        OSError: When the connection fails.
    """
    for move in moves:
        channel.send_move(move)
    return len(moves)


def _send_attacks(channel: AgentChannel, attacks: Sequence[AttackOrder]) -> int:
    """Send every attack order and report how many.

    Args:
        channel: An open connection to the agent.
        attacks: The orders to send.

    Returns:
        How many were sent.

    Raises:
        OSError: When the connection fails.
    """
    for attack in attacks:
        channel.send_attack(attack)
    return len(attacks)


def send_produces(channel: AgentChannel, orders: Sequence[ProduceOrder]) -> int:
    """Send every produce order and report how many.

    Args:
        channel: An open connection to the agent.
        orders: The orders to send.

    Returns:
        How many were sent.

    Raises:
        OSError: When the connection fails.
    """
    for order in orders:
        channel.send_produce(order)
    return len(orders)


def send_builds(channel: AgentChannel, orders: Sequence[BuildOrder]) -> None:
    """Send every build order.

    Args:
        channel: An open connection to the agent.
        orders: The orders to send.

    Raises:
        OSError: When the connection fails.
    """
    for order in orders:
        channel.send_build(order)


def _draft_raid(
    channel: AgentChannel,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    intel: Intel,
    army: tuple[Entity, ...],
    waves: WaveController,
    raiders: Raider,
) -> tuple[Entity, ...]:
    """Advance the raid and return the units the waves may still command.

    Whether the army can SPARE a party is decided here, against the wave
    controller's own figure, because nothing asked it in v1 and that cost
    every seat in the batch: the party came out of the wave gate and the
    guard, and the raid was refuted 0/12 for it (log: 2026-07-29). The gate
    arbitrates drafting only -- a party already out is managed to its end
    regardless. A unit cannot serve two commanders: whatever the raid drafted
    is withheld from the waves, and returns the moment the raid has nothing
    left to assault.
    """
    spare = len(army) >= waves.need() + raiders.size
    for order in raiders.strike(sample, intel, army, catalogue, spare):
        channel.send_attack_move(order)
    drafted = raiders.party()
    return tuple(u for u in army if u["unit_id"] not in drafted)


def _draft_hunt(
    channel: AgentChannel,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    intel: Intel,
    army: tuple[Entity, ...],
    targets: tuple[Entity, ...],
    waves: WaveController,
    hunters: Hunter,
    held_down: bool,
) -> tuple[Entity, ...]:
    """Advance the hunt and return the units the waves may still command.

    The same arbitration as the raid's, for the same reason: whether the
    army can SPARE a party is the wave gate's call, and a unit cannot
    serve two commanders. What is the hunt's own here is the recall --
    when the head holds the party down, it fights its way home and this
    tick sends those orders instead ([[impossible-step-three-design]]).
    """
    if held_down:
        for order in hunters.stand_down(army, catalogue, sample):
            channel.send_attack_move(order)
        return army
    spare = len(army) >= waves.need() + hunters.size
    for order in hunters.press(sample, intel, army, targets, catalogue, spare):
        channel.send_attack_move(order)
    drafted = hunters.party()
    return tuple(u for u in army if u["unit_id"] not in drafted)


def send_recon(
    channel: AgentChannel,
    scout: bool,
    lurk: int,
    decoys: int,
    scouts: ScoutRunner,
    lurkers: Lurker,
    scatter: Decoys,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
) -> None:
    """Walk the eyes, the leash and the scatter, whichever the doctrine plays.

    Scouts are allotted in a fixed order -- patrol first, lurk line next,
    scatter last -- so the three verbs never fight over one unit.
    """
    if scout:
        _send_moves(channel, scouts.patrol(sample, catalogue))
    if lurk:
        _send_moves(channel, lurkers.orders(sample, catalogue, lurk, skip=int(scout)))
    if decoys:
        _send_moves(channel, scatter.orders(sample, catalogue, decoys, skip=int(scout) + lurk))


def send_postures(
    channel: AgentChannel,
    kite: bool,
    hp_floor: int,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
) -> None:
    """Send the reflex layer its table, once, before the first observation.

    One row per profiled type: the reach rides the wire because the planner
    owns the catalogue and the agent never guesses a stat. The reflexes
    themselves apply only to our armed mobile types -- the reach rows for
    everything else are what let the agent read a THREAT's reach when one
    closes in ([[community-play-strategies]]).
    """
    if not kite and hp_floor == 0:
        return
    for type_name, profile in profiles.items():
        stats = catalogue.get(type_name)
        mobile_armed = stats is not None and stats["speed"] > 0.0 and profile["attack_range"] > 0.0
        channel.send_posture(
            posture_order(
                type_name=type_name,
                reach=profile["attack_range"],
                speed=stats["speed"] if stats is not None else 0.0,
                kite=kite and mobile_armed,
                hp_floor=hp_floor if mobile_armed else 0,
            )
        )


def send_nukes(channel: AgentChannel, orders: NukeOrders) -> None:
    """Send the finisher's tick: place, arm, or fire, whichever it decided.

    Args:
        channel: An open connection to the agent.
        orders: What the nuker decided this observation.

    Raises:
        OSError: When the connection fails.
    """
    if orders["build"] is not None:
        channel.send_build(orders["build"])
    if orders["arm"] is not None:
        channel.send_ability(orders["arm"])
    if orders["launch"] is not None:
        channel.send_targeted_ability(orders["launch"])


def send_tech(
    channel: AgentChannel,
    tech: int,
    sample: Sample,
    budget: Budget,
    teched: set[int],
) -> None:
    """Fire the factories' tier unlocks, up to the doctrine's count
    ([[mechanics-build-actions]])."""
    if len(teched) >= tech:
        return
    for unlock in unlock_tech(sample, budget, teched, limit=tech):
        channel.send_ability(unlock)


def advance_creep(
    channel: AgentChannel,
    creep: int,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    budget: Budget,
    free: Sequence[Entity],
    workforce: Workforce,
    creeper: Creeper,
) -> None:
    """Walk the wall toward its hold point, when the doctrine plays it.

    Claimed before the army on purpose: for a creep style the turret line IS
    the army, and a claim placed after production would starve on the same
    every-tick drain the tier-three conversion did ([[policy-creep]]).
    """
    if not creep:
        return
    send_builds(
        channel,
        creeper.advance(sample, catalogue, profiles, budget, free, workforce, hold=creep),
    )


def _march_rush(
    channel: AgentChannel,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    waves: WaveController,
    rusher: Rusher,
    fighting: tuple[Entity, ...],
    targets: Sequence[Entity],
    closing: bool = False,
) -> None:
    """March this tick's released units at the enemy start until contact.

    After the waves, so the released set is this tick's. While nothing is
    visible the released units march at the mirror of our base; on contact
    the engagement policy re-tasks them, the engine running the newest
    waypoint ([[policy-combat]]) -- unless the all-in has committed or the
    closer's dominance window is open, in which case the march is forced
    through contact: the all-in's dump exists to cross the map, and the
    closer exists because dominance decays -- eleven of nineteen dominant
    Very Hard positions LOST when the game ran long ([[policy-situation]],
    log 2026-08-01).
    """
    cleared = waves.released()
    marching = tuple(u for u in fighting if u["unit_id"] in cleared)
    orders = rusher.march(
        sample, catalogue, marching, bool(targets), force=waves.committed() or closing
    )
    for order in orders:
        channel.send_attack_move(order)


def fight(
    channel: AgentChannel,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    intel: Intel,
    army: tuple[Entity, ...],
    targets: tuple[Entity, ...],
    waves: WaveController,
    raiders: Raider,
    hunters: Hunter,
    rusher: Rusher,
    momentum: Momentum,
    *,
    raid: int,
    hunt: int,
    rush: bool,
    allin: int,
    strike: int,
    committed_close: bool,
    hunt_held: bool,
    pending_events: set[str],
) -> None:
    """Run one tick's combat dispatch, noting each decision as it happens.

    The loop's fighting tail, extracted whole: fill then commit, gather then
    hold one target, march past the trigger. Attacking costs nothing and is
    not arbitrated, which is why nothing here touches the budget
    ([[policy-combat]]). Decision codes land in ``pending_events`` for the
    next trace row (log 2026-08-09).

    Args:
        channel: An open connection to the agent.
        sample: One observation of the world.
        catalogue: Unit stats by type name.
        profiles: Combat profiles by type name.
        intel: The fog memory, already fed this observation.
        army: Units able to fight.
        targets: The hostile entities visible.
        waves: The wave controller, carrying its own commitments.
        raiders: The raid controller.
        hunters: The hunt controller.
        rusher: The forced-march controller.
        momentum: The rival army-value window the strike release reads.
        raid: The raid party's size, zero for no raiding.
        hunt: The hunt party's size, zero for no hunting.
        rush: Whether released waves march at the estimated enemy start.
        allin: The all-in release observation, zero for never.
        strike: The momentum release window's size, zero for off.
        committed_close: Whether the closer holds its latched commitment.
        hunt_held: Whether the head holds the hunt party home this tick.
        pending_events: Decision codes accumulating toward the next row.
    """
    fighting = army
    if waves.committed():
        # The strike force is withheld from the engagement like the
        # raid party, or first contact re-tasks the whole dump onto
        # the replaceable army and the income is never reached.
        struck = rusher.ordered()
        fighting = tuple(u for u in fighting if u["unit_id"] not in struck)
    if raid:
        raids_before = raiders.raids
        fighting = _draft_raid(channel, sample, catalogue, intel, army, waves, raiders)
        if raiders.raids > raids_before:
            pending_events.add("R")
    if hunt:
        hunts_before = hunters.hunts
        fighting = _draft_hunt(
            channel, sample, catalogue, intel, fighting, targets, waves, hunters, hunt_held
        )
        if hunters.hunts > hunts_before:
            pending_events.add("H")
    window_open = strike_window(momentum, strike)
    if window_open:
        pending_events.add("S")
    if committed_close:
        pending_events.add("C")
    moves, attacks = waves.command(
        sample,
        catalogue,
        profiles,
        fighting,
        strike=window_open or committed_close,
    )
    _send_moves(channel, moves)
    _send_attacks(channel, attacks)
    # Gated on the LATCHED commitment, not the knob: before the first
    # open window a close doctrine holds exactly as it would without
    # one -- gating on the knob made every ladder release march at
    # the mirror, which is the rush verb wearing the closer's name.
    if rush or allin or committed_close:
        marches_before = raiders.marches + rusher.marches
        _march_rush(channel, sample, catalogue, waves, rusher, fighting, targets, committed_close)
        if raiders.marches + rusher.marches > marches_before:
            pending_events.add("M")


__all__ = [
    "advance_creep",
    "fight",
    "send_builds",
    "send_nukes",
    "send_plan_step",
    "send_postures",
    "send_produces",
    "send_recon",
    "send_tech",
]
