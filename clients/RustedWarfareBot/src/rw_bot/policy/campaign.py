"""Play a whole match: build the plan, then fight with what it built.

Two phases, kept apart because they stop for different reasons. The build phase
ends when the plan is satisfied or cannot proceed; the fight phase ends when
there is nothing left to attack, nothing left to attack with, or the sample
budget runs out.

This module is orchestration only. What to build is
:mod:`rw_bot.policy.build_order`, what to attack is
:mod:`rw_bot.policy.combat`, and both are pure -- the channel is touched here
and in :mod:`rw_bot.policy.runner` and nowhere else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.build_order import BUILDER_TYPE, find_anchor, find_builder
from rw_bot.policy.combat import (
    Engagement,
    engagements,
    find_army,
    find_targets,
    muster,
    rally,
)
from rw_bot.policy.economy import count_extractors, expand_economy, expand_production
from rw_bot.policy.observation import has_moved, position_of
from rw_bot.policy.production import sustain
from rw_bot.wire.command import attack_order, build_order, move_order, produce_order
from rw_bot.wire.state import Entity, Sample

#: Samples a stationary builder may sit on an unstarted expansion before the
#: order is presumed lost and sent again.
#:
#: The same reasoning as the build loop's stall window, used for the opposite
#: purpose: there it ends the run, here it retries. A builder that has neither
#: moved nor started building for this many samples is not on its way anywhere,
#: and the cost of being wrong is one duplicate order the engine collapses onto
#: the same waypoint ([[policy-loop]]).
EXPAND_RETRY_SAMPLES = 45


class Battle(TypedDict):
    """What the fight phase achieved.

    Attributes:
        orders_sent: Attack orders issued.
        army_start: Units available to fight when the phase opened.
        army_end: Units still available when it closed.
        targets_seen: Hostile entities visible when the phase opened.
        targets_end: Hostile entities visible when it closed.
        killed: Targets the bot ordered an attack on that are no longer
            visible. Not a kill count -- a target that retreated into fog reads
            the same way, which is why the field is named for what was observed
            rather than for what was concluded.
        produced: Reinforcements ordered while the fight ran.
        expanded: Extractors ordered onto resource pools while the fight ran.
        extractors_start: Finished extractors held when the phase opened.
        extractors_end: Finished extractors held when it closed.
        expand_reason: The economy's own words for its last decision, which is
            what tells a match that could not expand from one that chose not
            to.
        samples_seen: World samples read during the phase.
        outcome: Why it stopped: ``"cleared"``, ``"no_army"``, or
            ``"sample_limit"``.
    """

    produced: int
    expanded: int
    extractors_start: int
    extractors_end: int
    expand_reason: str
    orders_sent: int
    rallied: int
    army_start: int
    army_end: int
    targets_seen: int
    targets_end: int
    killed: int
    samples_seen: int
    outcome: str


class _Expander:
    """Turns economy decisions into orders, and remembers what it already asked.

    The decision is :func:`rw_bot.policy.economy.expand_economy` and stays pure.
    What lives here is the part that cannot: whether an order already sent is
    worth sending again. Those are different questions, and keeping them apart
    is what lets the choice of pool be tested without a clock.

    Re-sending matters because the economy is asked on every sample. A builder
    that has been told to claim a pool, and has not yet started walking, is
    still offered that same pool -- so without a memory the order would go out
    at sample rate. That is the churn that produced 743 attack orders against 24
    targets before the combat side learned to commit ([[policy-combat]]).

    The memory is not a ban. An order can be refused or lost, and a pool that is
    never retried is an economy that stops for the rest of the match, so a
    repeat is allowed once the builder has sat still through
    :data:`EXPAND_RETRY_SAMPLES` observations doing nothing at all.

    A caller that passes no reach table gets one of these anyway, switched off.
    Returning None instead would push a "then are we expanding?" test into every
    place that touches it, and the loop it lives in is already at the limit of
    the branching it can carry.

    Attributes:
        enabled: Whether expansion is being played at all.
        count: Expansion orders sent.
        reason: The economy's own words for its most recent decision.
    """

    def __init__(
        self,
        catalogue: Mapping[str, UnitStats],
        reaches: Mapping[str, float] | None,
        reserve: int,
    ) -> None:
        self.enabled = reaches is not None
        self.count = 0
        self.reason = "no sample seen yet" if self.enabled else "expansion disabled"
        self._catalogue = catalogue
        self._reaches = reaches
        self._reserve = reserve
        self._last_site: tuple[float, float] | None = None
        self._quiet = 0

    def step(self, channel: AgentChannel, sample: Sample, builder_moved: bool) -> None:
        """Ask the economy what to do about this sample, and do it.

        Does nothing when expansion is switched off, so the caller does not have
        to ask.

        Args:
            channel: An open connection to the agent.
            sample: One observation of the world.
            builder_moved: Whether the builder moved since the previous sample.

        Raises:
            OSError: When the connection fails.
        """
        if self._reaches is None:
            return
        # Throughput before income. Another extractor earns credits the player
        # already cannot spend, and the run that measured this banked 7,013 of
        # them behind a single factory ([[policy-production]]). A producer is
        # therefore proposed first, and only when the queue is genuinely the
        # constraint -- otherwise this falls straight through to the pool.
        growth = expand_production(sample, self._catalogue, reserve=self._reserve)
        if not growth["build"]:
            growth = expand_economy(
                sample,
                self._catalogue,
                self._reaches,
                reserve=self._reserve,
                builder_moved=builder_moved,
            )
        self.reason = growth["reason"]
        if not growth["build"]:
            # Walking, building, or saving up. The retry clock measures a
            # builder doing none of those, so anything else resets it.
            self._quiet = 0
            return
        site = (growth["x"], growth["y"])
        if site == self._last_site and self._quiet < EXPAND_RETRY_SAMPLES:
            self._quiet += 1
            return
        channel.send_build(
            build_order(
                unit_id=growth["unit_id"],
                type_name=growth["type_name"],
                x=growth["x"],
                y=growth["y"],
            )
        )
        self.count += 1
        self._last_site = site
        self._quiet = 0


def _gather_reserve(
    channel: AgentChannel,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    reserve: Sequence[Entity],
    rallying: set[int],
) -> int:
    """Send the units still gathering to the base, once each.

    Once each, not once per sample: the engine runs a waypoint until it is
    replaced, so re-issuing at the sampling rate resets the walk and nothing
    arrives. A unit knocked off course is therefore not re-ordered, which is a
    real gap and a cheaper one than never arriving at all.

    Args:
        channel: An open connection to the agent.
        sample: One observation of the world.
        catalogue: Unit stats by type name, for finding the anchor.
        reserve: Units not cleared to attack.
        rallying: Ids already sent. Extended in place.

    Returns:
        How many move orders were sent.
    """
    anchor = find_anchor(sample, catalogue)
    if anchor is None:
        # Nothing immobile left to gather at. A player who has lost every
        # structure has worse problems than formation.
        return 0
    sent = 0
    for move in rally(reserve, (anchor["x"], anchor["y"])):
        if move["unit_id"] in rallying:
            continue
        rallying.add(move["unit_id"])
        channel.send_move(move_order(unit_id=move["unit_id"], x=move["x"], y=move["y"]))
        sent += 1
    return sent


def _wanted(
    reinforce: Sequence[str],
    builder: Entity | None,
    expanding: bool,
) -> tuple[str, ...]:
    """Return what idle producers should make, in preference order.

    A lost builder ends the economy permanently, so a replacement is asked for
    when none is alive. It goes **last**, which is what keeps the factories on
    tanks: a producer takes the first type it can make, and only the command
    centre -- which cannot make a tank -- falls through to the builder.

    Nothing is added when expansion is switched off. A builder with no pools to
    claim is a unit that walks nowhere and costs 200 credits.

    Args:
        reinforce: Type names the plan keeps wanting.
        builder: The owned builder, or None when there is none.
        expanding: Whether the economy is being played at all.

    Returns:
        Type names to keep making, in preference order.
    """
    if expanding and builder is None:
        return (*reinforce, BUILDER_TYPE)
    return tuple(reinforce)


def _replace_losses(
    channel: AgentChannel,
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    wanted: Sequence[str],
) -> int:
    """Order every idle producer to make what the plan keeps wanting.

    Args:
        channel: An open connection to the agent.
        sample: One observation of the world.
        catalogue: Unit stats by type name, for prices.
        wanted: Type names to keep making, in preference order.

    Returns:
        How many produce orders were sent.

    Raises:
        OSError: When the connection fails.
    """
    sent = 0
    for order in sustain(sample, catalogue, wanted):
        channel.send_produce(produce_order(unit_id=order["unit_id"], type_name=order["type_name"]))
        sent += 1
    return sent


def _dispatch_attacks(
    channel: AgentChannel,
    current: Sequence[Engagement],
    ordered: dict[int, int],
    attacked: set[int],
) -> int:
    """Send each engagement whose attacker is not already on that target.

    The engine keeps executing a waypoint until it is replaced, so re-issuing
    an identical attack every sample would replace an in-progress order with a
    copy of itself and the unit would never close the distance.

    Args:
        channel: An open connection to the agent.
        current: The engagements the combat policy chose.
        ordered: Target each attacker was last sent at, updated in place.
        attacked: Every target ordered against, updated in place.

    Returns:
        How many attack orders were sent.

    Raises:
        OSError: When the connection fails.
    """
    sent = 0
    for engagement in current:
        attacker = engagement["attacker_id"]
        target = engagement["target_id"]
        if ordered.get(attacker) == target:
            continue
        ordered[attacker] = target
        attacked.add(target)
        channel.send_attack(attack_order(unit_id=attacker, target_id=target))
        sent += 1
    return sent


def fight(
    channel: AgentChannel,
    catalogue: Mapping[str, UnitStats],
    max_samples: int,
    reinforce: Sequence[str] = (),
    *,
    reaches: Mapping[str, float] | None = None,
    reserve: int = 0,
) -> Battle:
    """Send the army at the enemy, replacing losses and claiming ground as it goes.

    An order is re-sent only when a unit's target changes. The engine keeps
    executing a waypoint until it is replaced, so re-issuing the same attack
    every sample would replace an in-progress order with an identical one at
    ~300 Hz and the unit would never close the distance.

    Reinforcement is what makes this a fight rather than a sortie. Without it
    the bot commits a fixed force and is finished when that force is, which is
    exactly how it lost 4 tanks to nothing: the opponents replace losses
    continuously and we did not ([[ai-opponent-strategy]]).

    Expansion is what makes reinforcement affordable. Replacing losses out of a
    fixed income is a race the bot loses on arithmetic: three extractors funded
    45 replacements over a run that ended with two units alive, while the
    opponents grew from 47 visible units to 142 by taking more of the map's 46
    resource pools ([[policy-economy]]). So the same loop that spends credits
    also earns them, and a builder that would otherwise stand idle for the whole
    match claims another pool whenever one is safe and affordable.

    A lost builder ends the economy permanently, so when nothing owned can place
    an extractor the command centre is asked for another. It is put last in the
    preference order, which is what keeps the factories on tanks: a producer
    takes the first type it can make, and only the command centre -- which
    cannot make a tank -- falls through to the builder.

    Args:
        channel: An open connection to the agent.
        catalogue: Unit stats by type name, for reading who is armed.
        max_samples: Stop after this many samples regardless.
        reinforce: Type names idle producers should keep making. Empty means
            fight with what exists and make nothing.
        reaches: Attack range by type name, for judging which resource pools can
            be reached without walking through fire. None disables expansion
            entirely, which is what a caller that only wants to fight passes.
        reserve: Credits to leave unspent for the army before claiming a pool.

    Returns:
        The battle report.

    Raises:
        ChannelError: When the agent closes the connection mid-phase.
        OSError: When the connection fails.
    """
    holding: int | None = None
    released: frozenset[int] = frozenset()
    waves = 0
    rallying: set[int] = set()
    rallied = 0
    ordered: dict[int, int] = {}
    attacked: set[int] = set()
    visible_now: set[int] = set()
    orders_sent = 0
    produced = 0
    samples_seen = 0
    army_start = 0
    targets_seen = 0
    army_end = 0
    targets_end = 0
    outcome = "sample_limit"
    extractors_start = 0
    extractors_end = 0
    builder_was: tuple[float, float] | None = None
    expander = _Expander(catalogue, reaches, reserve)

    while samples_seen < max_samples:
        sample = channel.next_sample()
        samples_seen += 1

        # Acknowledged on every exit, including the ones that break out. In
        # lockstep the agent holds the simulation until this arrives
        # ([[policy-determinism]]).
        try:
            army = find_army(sample, catalogue)
            targets = find_targets(sample)
            army_end = len(army)
            targets_end = len(targets)
            visible_now = {entity["unit_id"] for entity in targets}
            if samples_seen == 1:
                army_start = army_end
                targets_seen = targets_end

            # Read unconditionally, on every sample. The builder's travel is
            # what tells expansion its order is still being carried out, so it
            # has to be sampled even on observations that never reach a
            # decision -- the same rule the build loop's stall clock follows.
            builder = find_builder(sample)
            builder_now = position_of(builder)
            builder_moved = has_moved(builder_was, builder_now)
            builder_was = builder_now

            extractors_end = count_extractors(sample)
            if samples_seen == 1:
                extractors_start = extractors_end

            # Production runs before the army check, so a wave that has just been
            # wiped still queues its replacements on the sample that notices.
            produced += _replace_losses(
                channel, sample, catalogue, _wanted(reinforce, builder, expander.enabled)
            )
            expander.step(channel, sample, builder_moved)

            if not army:
                # Nothing left to fight with. Distinct from having cleared the
                # field, and the run log has to be able to tell those apart.
                outcome = "no_army"
                break
            if not targets:
                outcome = "cleared"
                break

            # Fill, then commit. Reinforcements pool in the reserve until they
            # are a wave rather than walking into the fight one at a time, which
            # is what the shipped AI does and what a plain "have we started"
            # flag got wrong ([[engine-ai-triggers]]).
            wave = muster(army, released, waves)
            released = wave["released"]
            waves = wave["waves"]

            # The reserve gathers at the base rather than sitting wherever it
            # rolled out of the factory. Without it a released wave arrives
            # piecemeal even after the gate opens, which is the trickle again
            # one step earlier -- and units waiting at the base are the only
            # thing standing between an attacker and it ([[policy-combat]]).
            rallied += _gather_reserve(
                channel,
                sample,
                catalogue,
                tuple(u for u in army if u["unit_id"] not in released),
                rallying,
            )

            if not released:
                continue
            fighting = tuple(u for u in army if u["unit_id"] in released)

            # The target the army is already on is carried in, so the choice
            # persists across samples instead of being remade every observation.
            # Without it the whole army is re-tasked whenever its centre shifts.
            current = engagements(sample, catalogue, holding, fighting)
            holding = current[0]["target_id"] if current else None
            orders_sent += _dispatch_attacks(channel, current, ordered, attacked)
        finally:
            channel.send_ack()

    return Battle(
        produced=produced,
        expanded=expander.count,
        extractors_start=extractors_start,
        extractors_end=extractors_end,
        expand_reason=expander.reason,
        orders_sent=orders_sent,
        rallied=rallied,
        army_start=army_start,
        army_end=army_end,
        targets_seen=targets_seen,
        targets_end=targets_end,
        killed=len(attacked - visible_now),
        samples_seen=samples_seen,
        outcome=outcome,
    )


def format_battle(battle: Battle) -> tuple[str, ...]:
    """Render a battle report as lines.

    Args:
        battle: The report.

    Returns:
        One line per figure.
    """
    return (
        f"fight outcome  {battle['outcome']}",
        f"attack orders  {battle['orders_sent']}",
        f"reinforced     {battle['produced']}",
        f"army           {battle['army_start']} -> {battle['army_end']}",
        f"enemies seen   {battle['targets_seen']} -> {battle['targets_end']}",
        f"engaged gone   {battle['killed']}",
        f"extractors     {battle['extractors_start']} -> {battle['extractors_end']}",
        f"expansions     {battle['expanded']}",
        f"expand note    {battle['expand_reason']}",
        f"samples seen   {battle['samples_seen']}",
    )


__all__ = ["Battle", "fight", "format_battle"]
