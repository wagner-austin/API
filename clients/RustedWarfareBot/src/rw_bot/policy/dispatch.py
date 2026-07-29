"""Sending the army where it has been decided to go.

The thin layer between a decision and the wire. :mod:`rw_bot.policy.combat`
chooses what to attack and where the reserve gathers; this sends those choices
and counts what was sent.

Both functions exist because of the same engine behaviour: a waypoint keeps
being executed until it is replaced, so re-issuing an identical order every
sample replaces an in-progress order with a copy of itself and the unit never
arrives ([[issuing-orders]]). Each therefore tracks what it has already sent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile
from rw_bot.policy.combat import WAVE_SIZES, Engagement, engagements, muster, rally
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import AttackOrder, MoveOrder, attack_order, move_order
from rw_bot.wire.state import Entity, Sample


def gather_reserve(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    reserve: Sequence[Entity],
    rallying: set[int],
) -> tuple[MoveOrder, ...]:
    """Send the units still gathering to the base, once each.

    Once each, not once per sample: the engine runs a waypoint until it is
    replaced, so re-issuing at the sampling rate resets the walk and nothing
    arrives. A unit knocked off course is therefore not re-ordered, which is a
    real gap and a cheaper one than never arriving at all.

    **Once per stint in the reserve, not once per match.** The caller clears a
    unit's mark when it is released into a wave, so a wave that disbands gets a
    fresh order home. Without that the set is a permanent record of everyone
    ever rallied, and a survivor handed back to the reserve would be told
    nothing: not cleared to attack, and already marked as rallied. It would
    stand where its wave died until enough reinforcements arrived to release it
    again ([[policy-combat]]).

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for finding the anchor.
        reserve: Units not cleared to attack.
        rallying: Ids already sent, for units currently in the reserve.
            Extended in place.

    Returns:
        The move orders to send, in roster order.
    """
    anchor = find_anchor(sample, catalogue)
    if anchor is None:
        # Nothing immobile left to gather at. A player who has lost every
        # structure has worse problems than formation.
        return ()
    orders: list[MoveOrder] = []
    for move in rally(reserve, (anchor["x"], anchor["y"])):
        if move["unit_id"] in rallying:
            continue
        rallying.add(move["unit_id"])
        orders.append(move_order(unit_id=move["unit_id"], x=move["x"], y=move["y"]))
    return tuple(orders)


def dispatch_attacks(
    current: Sequence[Engagement],
    ordered: dict[int, int],
    attacked: set[int],
) -> tuple[AttackOrder, ...]:
    """Send each engagement whose attacker is not already on that target.

    The engine keeps executing a waypoint until it is replaced, so re-issuing
    an identical attack every sample would replace an in-progress order with a
    copy of itself and the unit would never close the distance.

    Args:
        current: The engagements the combat policy chose.
        ordered: Target each attacker was last sent at, updated in place.
        attacked: Every target ordered against, updated in place.

    Returns:
        The attack orders to send, in engagement order.
    """
    orders: list[AttackOrder] = []
    for engagement in current:
        attacker = engagement["attacker_id"]
        target = engagement["target_id"]
        if ordered.get(attacker) == target:
            continue
        ordered[attacker] = target
        attacked.add(target)
        orders.append(attack_order(unit_id=attacker, target_id=target))
    return tuple(orders)


class WaveController:
    """Holds the army's wave state across observations, and turns it to orders.

    The state it keeps -- who is released, who is rallying, who was last sent at
    what -- used to live as six loop locals in the campaign, which meant the
    whole of wave discipline could only be exercised by playing a match through
    the loop that owned them ([[policy-combat]]). It is one object now, the
    same shape :class:`~rw_bot.policy.runner.OrderTracker` and
    :class:`~rw_bot.policy.workforce.Workforce` already have: the decisions
    stay in :mod:`rw_bot.policy.combat` and stay pure, and what lives here is
    only the memory between observations.

    Attributes:
        attack_orders: Attack orders decided so far.
        rallied: Move orders decided so far to gather the reserve.
    """

    def __init__(self, ladder: Sequence[int] = WAVE_SIZES) -> None:
        """Open a controller.

        Args:
            ladder: How many units each successive wave waits for. Defaults to
                the shipped AI's ([[engine-ai-triggers]]).
        """
        self._ladder = tuple(ladder)
        self.attack_orders = 0
        self.rallied = 0
        self._holding: int | None = None
        self._released: frozenset[int] = frozenset()
        self._waves = 0
        self._rallying: set[int] = set()
        self._ordered: dict[int, int] = {}
        self._attacked: set[int] = set()

    def command(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        profiles: Mapping[str, CombatProfile],
        army: Sequence[Entity],
    ) -> tuple[tuple[MoveOrder, ...], tuple[AttackOrder, ...]]:
        """Decide this observation's moves and attacks.

        Fill, then commit: reinforcements pool in the reserve until they are a
        wave, the reserve gathers at the base rather than where it rolled out
        of the factory, and the released wave holds one target across samples
        instead of being re-tasked whenever its centre shifts. Each of those
        rules is documented where it is decided ([[policy-combat]],
        [[engine-ai-triggers]]); what this method owns is their order and their
        memory.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name, for mobility and the anchor.
            profiles: Combat profiles by type name, for reachability.
            army: Units available to fight, as the caller found them.

        Returns:
            The move orders and the attack orders to send, in that order.
        """
        wave = muster(army, self._released, self._waves, self._ladder)
        self._released = wave["released"]
        self._waves = wave["waves"]
        # A unit cleared to attack is no longer gathering, so it forgets it was
        # ever sent home. That is what lets a disbanded wave be sent home again
        # rather than standing where it died.
        self._rallying -= self._released

        moves = gather_reserve(
            sample,
            catalogue,
            tuple(unit for unit in army if unit["unit_id"] not in self._released),
            self._rallying,
        )
        self.rallied += len(moves)

        if not self._released:
            return moves, ()
        fighting = tuple(unit for unit in army if unit["unit_id"] in self._released)
        # The target the army is already on is carried in, so the choice
        # persists across samples instead of being remade every observation.
        current = engagements(sample, catalogue, profiles, self._holding, fighting)
        self._holding = current[0]["target_id"] if current else None
        attacks = dispatch_attacks(current, self._ordered, self._attacked)
        self.attack_orders += len(attacks)
        return moves, attacks

    def killed(self, visible_now: Set[int]) -> int:
        """Return how many targets ordered against are no longer visible.

        Not a kill count -- a target that retreated into fog reads the same
        way, which is why the report names the figure for what was observed.

        Args:
            visible_now: Engine ids of the hostiles visible on the last
                observation.

        Returns:
            Targets attacked that are now gone from sight.
        """
        return len(self._attacked - visible_now)


__all__ = ["WaveController", "dispatch_attacks", "gather_reserve"]
