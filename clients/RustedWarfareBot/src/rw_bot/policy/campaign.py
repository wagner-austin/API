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

from collections.abc import Mapping
from typing import TypedDict

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.combat import engagements, find_army, find_targets
from rw_bot.wire.command import attack_order


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
        samples_seen: World samples read during the phase.
        outcome: Why it stopped: ``"cleared"``, ``"no_army"``, or
            ``"sample_limit"``.
    """

    orders_sent: int
    army_start: int
    army_end: int
    targets_seen: int
    targets_end: int
    killed: int
    samples_seen: int
    outcome: str


def fight(
    channel: AgentChannel,
    catalogue: Mapping[str, UnitStats],
    max_samples: int,
) -> Battle:
    """Send the army at the enemy until one side runs out.

    An order is re-sent only when a unit's target changes. The engine keeps
    executing a waypoint until it is replaced, so re-issuing the same attack
    every sample would replace an in-progress order with an identical one at
    ~300 Hz and the unit would never close the distance.

    Args:
        channel: An open connection to the agent.
        catalogue: Unit stats by type name, for reading who is armed.
        max_samples: Stop after this many samples regardless.

    Returns:
        The battle report.

    Raises:
        ChannelError: When the agent closes the connection mid-phase.
        OSError: When the connection fails.
    """
    ordered: dict[int, int] = {}
    attacked: set[int] = set()
    visible_now: set[int] = set()
    orders_sent = 0
    samples_seen = 0
    army_start = 0
    targets_seen = 0
    army_end = 0
    targets_end = 0
    outcome = "sample_limit"

    while samples_seen < max_samples:
        sample = channel.next_sample()
        samples_seen += 1

        army = find_army(sample, catalogue)
        targets = find_targets(sample)
        army_end = len(army)
        targets_end = len(targets)
        visible_now = {entity["unit_id"] for entity in targets}
        if samples_seen == 1:
            army_start = army_end
            targets_seen = targets_end

        if not army:
            # Nothing left to fight with. Distinct from having cleared the
            # field, and the run log has to be able to tell those apart.
            outcome = "no_army"
            break
        if not targets:
            outcome = "cleared"
            break

        for engagement in engagements(sample, catalogue):
            attacker = engagement["attacker_id"]
            target = engagement["target_id"]
            if ordered.get(attacker) == target:
                continue
            ordered[attacker] = target
            attacked.add(target)
            channel.send_attack(attack_order(unit_id=attacker, target_id=target))
            orders_sent += 1

    return Battle(
        orders_sent=orders_sent,
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
        f"army           {battle['army_start']} -> {battle['army_end']}",
        f"enemies seen   {battle['targets_seen']} -> {battle['targets_end']}",
        f"engaged gone   {battle['killed']}",
        f"samples seen   {battle['samples_seen']}",
    )


__all__ = ["Battle", "fight", "format_battle"]
