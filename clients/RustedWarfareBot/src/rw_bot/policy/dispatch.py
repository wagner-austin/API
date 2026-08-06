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
from rw_bot.mechanics.combat_profile import CombatProfile, can_engage
from rw_bot.mechanics.upgrades import satisfies
from rw_bot.policy.combat import (
    WAVE_SIZES,
    Engagement,
    engagements,
    find_targets,
    muster,
    rally,
    wave_size,
)
from rw_bot.policy.guard import deepest_intruder
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import AttackOrder, MoveOrder, attack_order, move_order
from rw_bot.wire.state import Entity, Sample


def rally_post(sample: Sample, catalogue: Mapping[str, UnitStats], forward: bool) -> Entity | None:
    """Return the structure the reserve gathers at.

    The anchor by default -- the behaviour every measurement so far was taken
    under. Forward, it is the owned extractor **farthest from the anchor**:
    the frontier one. The motivation is the one invariant six consecutive
    batches have not moved: matches are decided by extractor drops, the
    per-loss table puts each death 688-1,766 world units from the army's own
    fighting cloud, and the army has spent every one of those matches
    gathered at the base on the other side of that distance
    ([[policy-holding-ground]]). The community corpus ranks the same idea
    second of everything it teaches: military *forward*, between the enemy
    and the extractors ([[community-play-strategies]]).

    An extractor of any tier qualifies -- :func:`~rw_bot.mechanics.upgrades.
    satisfies` is the same any-tier test the plan's own progress count
    trusts. Farthest, with the id as tie-break, so two runs of one seed post
    the reserve identically.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for the anchor.
        forward: Whether the reserve posts at the frontier.

    Returns:
        The structure to gather at, or None when nothing immobile stands.
    """
    anchor = find_anchor(sample, catalogue)
    if anchor is None or not forward:
        return anchor

    def frontier(entity: Entity) -> tuple[float, int]:
        dx = entity["x"] - anchor["x"]
        dy = entity["y"] - anchor["y"]
        # Negated id so max() breaks distance ties toward the LOWEST id, the
        # ordering every other draft in this codebase uses.
        return (dx * dx + dy * dy, -entity["unit_id"])

    extractors = [
        entity
        for entity in sample["entities"]
        if entity["mine"] and entity["complete"] and satisfies(entity["type_name"], "extractorT1")
    ]
    if not extractors:
        return anchor
    return max(extractors, key=frontier)


def gather_reserve(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    reserve: Sequence[Entity],
    rallying: set[int],
    forward: bool = False,
) -> tuple[MoveOrder, ...]:
    """Send the units still gathering to the rally post, once each.

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
        catalogue: Unit stats by type name, for finding the rally post.
        reserve: Units not cleared to attack.
        rallying: Ids already sent, for units currently in the reserve.
            Extended in place.
        forward: Whether the reserve posts at the frontier extractor instead
            of the base (:func:`rally_post`).

    Returns:
        The move orders to send, in roster order.
    """
    post = rally_post(sample, catalogue, forward)
    if post is None:
        # Nothing immobile left to gather at. A player who has lost every
        # structure has worse problems than formation.
        return ()
    orders: list[MoveOrder] = []
    for move in rally(reserve, (post["x"], post["y"])):
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
        intercepts: Guard engagements decided so far, for the report.
    """

    def __init__(
        self,
        ladder: Sequence[int] = WAVE_SIZES,
        *,
        intercept: bool = False,
        guard_cap: int = 0,
        forward: bool = False,
        riposte: bool = False,
        allin_at: int = 0,
    ) -> None:
        """Open a controller.

        Args:
            ladder: How many units each successive wave waits for. Defaults to
                the shipped AI's ([[engine-ai-triggers]]).
            intercept: Whether the reserve turns on a raider standing inside
                the outpost radius of one of our structures, or keeps
                gathering regardless ([[policy-holding-ground]]). False is
                the behaviour every measurement so far was taken under.
            guard_cap: The most reserve units an interception commits, or zero
                for all of them -- the behaviour both guard A/Bs were measured
                under. The cost case that makes it a question: one match
                logged 870 intercepts and never massed an attack, so
                answer-with-everything may be buying defence with the offence
                ([[policy-holding-ground]]).
            forward: Whether the reserve posts at the frontier extractor
                instead of the base (:func:`rally_post`). False is the
                behaviour every measurement so far was taken under.
            riposte: Whether the whole reserve releases the moment an
                intrusion ends -- the counter-punch a human plays: let the
                attack burn itself on the defences, then push into the
                window before the opponent's next group finishes staging
                ([[ai-opponent-strategy]]). False is the behaviour every
                measurement so far was taken under: waves release on size
                alone, at a moment that means nothing.
            allin_at: The observation from which the whole reserve releases
                every tick, or zero never. The all-in verb: forty-seven
                Impossible matches released on size and met an army that had
                compounded past answering, so this releases on TIME -- hold
                everything to the chosen moment, dump it, and stream every
                later unit straight in. The anti-trickle floor still holds:
                a release below the first wave's size is a trickle whatever
                the clock says ([[policy-combat]]).
        """
        self._ladder = tuple(ladder)
        self._intercept = intercept
        self._guard_cap = guard_cap
        self._forward = forward
        self._riposte = riposte
        self._allin_at = allin_at
        # Observations seen, counted here rather than passed in: the trigger
        # is about this controller's own timeline, and every caller already
        # calls command() exactly once per observation.
        self._observed = 0
        self._committed = False
        # The riposte's memory: whether an intruder stood on our ground last
        # observation, and whether its departure has armed a counter-punch
        # that muster has not yet consumed.
        self._intruding = False
        self._avenging = False
        self.attack_orders = 0
        self.rallied = 0
        self.intercepts = 0
        # Target held per attacker, so a group persists while its target
        # lives and only freed units are dealt into groups afresh
        # ([[policy-combat]]).
        self._held: dict[int, int] = {}
        self._released: frozenset[int] = frozenset()
        self._waves = 0
        self._rallying: set[int] = set()
        self._guarding: set[int] = set()
        self._ordered: dict[int, int] = {}
        self._attacked: set[int] = set()

    def committed(self) -> bool:
        """Return whether the all-in moment has arrived.

        The rusher's override: past the trigger the march ignores contact,
        because the all-in IS the march -- the first probe released on time
        and then stood at home fighting whatever was visible, and the dump
        never crossed the map (``marches 0``, log 2026-07-31).

        Returns:
            True from the all-in observation onward.
        """
        return self._committed

    def released(self) -> frozenset[int]:
        """Return the ids currently cleared to attack.

        The rush's draft pool: released units with nothing visible to fight
        are the ones marched at the estimated enemy start, and exposing the
        set keeps that arbitration in the campaign beside the raid's
        ([[policy-combat]]).
        """
        return self._released

    def need(self) -> int:
        """Return how many units the next wave waits for.

        The same figure :func:`~rw_bot.policy.combat.muster` will use, read
        through the same function, so a caller arbitrating against the gate
        cannot drift from the gate. The raid is that caller: its party is
        drafted only from units *beyond* this figure, because v1 drafted from
        the gate itself and was refuted 0/12 for it ([[policy-raid]]).

        Returns:
            The current rung of the ladder.
        """
        return wave_size(self._waves, self._ladder)

    def command(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        profiles: Mapping[str, CombatProfile],
        army: Sequence[Entity],
        strike: bool = False,
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
        self._observed += 1
        self._committed = self._allin_at > 0 and self._observed >= self._allin_at
        # Three ways the whole reserve releases, in the order they were
        # built: the riposte's intrusion edge, the all-in's clock, and the
        # strike window's army-value ratio -- the signal the first two were
        # approximating ([[policy-situation]]).
        wave = muster(
            army,
            self._released,
            self._waves,
            self._ladder,
            force=self._avenging or self._committed or strike,
        )
        # Consumed whether or not it released: a riposte with too few units
        # to punch is a riposte missed, not one banked for an arbitrary
        # later moment that has lost the window.
        self._avenging = False
        self._released = wave["released"]
        self._waves = wave["waves"]
        # A unit cleared to attack is no longer gathering, so it forgets it was
        # ever sent home. That is what lets a disbanded wave be sent home again
        # rather than standing where it died.
        self._rallying -= self._released

        reserve = tuple(unit for unit in army if unit["unit_id"] not in self._released)
        moves = gather_reserve(sample, catalogue, reserve, self._rallying, self._forward)
        self.rallied += len(moves)

        guard_attacks = self._guard(sample, catalogue, profiles, reserve)

        if not self._released:
            return moves, guard_attacks
        fighting = tuple(unit for unit in army if unit["unit_id"] in self._released)
        # Each attacker's current target is carried in, so the groups persist
        # across samples instead of being re-dealt every observation.
        current = engagements(sample, catalogue, profiles, self._held, fighting)
        self._held = {e["attacker_id"]: e["target_id"] for e in current}
        attacks = dispatch_attacks(current, self._ordered, self._attacked)
        self.attack_orders += len(attacks)
        return moves, (*guard_attacks, *attacks)

    def _guard(
        self,
        sample: Sample,
        catalogue: Mapping[str, UnitStats],
        profiles: Mapping[str, CombatProfile],
        reserve: Sequence[Entity],
    ) -> tuple[AttackOrder, ...]:
        """Turn the reserve on a raider inside our ground, if there is one.

        The wave gate exists to stop units trickling into *defended* ground;
        it was never an argument for standing at the rally point while a
        raider kills the extractor next to it -- inside our own outpost radius
        there are no enemy turrets and the reserve has local numbers
        ([[policy-holding-ground]]). So intrusion bypasses the gate, and only
        intrusion does.

        Guards forget they were ever sent home once the raid ends, so the
        gather pass re-rallies them instead of leaving them standing where the
        fight finished -- the same rule a disbanded wave follows.

        Args:
            sample: One observation of the world.
            catalogue: Unit stats by type name.
            profiles: Combat profiles by type name.
            reserve: Units not released to a wave.

        Returns:
            The attack orders to send, empty when interception is off or
            nothing intrudes.
        """
        if not self._intercept and not self._riposte:
            return ()
        intruder = deepest_intruder(sample, catalogue, profiles, reserve, find_targets(sample))
        # The riposte arms on the edge, not the state: an intrusion that just
        # ENDED is the enemy's attack burned out on our ground, and the window
        # before its next group finishes staging is when a stockpile converts
        # ([[ai-opponent-strategy]]). Armed here, consumed by the next
        # muster.
        if self._riposte and self._intruding and intruder is None:
            self._avenging = True
        self._intruding = intruder is not None
        if intruder is None or not self._intercept:
            if self._guarding:
                self._rallying -= self._guarding
                self._guarding.clear()
            return ()
        engageable = [unit for unit in reserve if can_engage(profiles, unit, intruder)]
        if 0 < self._guard_cap < len(engageable):
            # The nearest, because an interception is a race with the damage
            # the intruder is doing: the detachment that arrives first is the
            # one that was closest when the alarm went. Squared distance --
            # ranking needs no root -- with the id as the tie-break every
            # ordering in this codebase uses.

            def closeness(unit: Entity) -> tuple[float, int]:
                dx = unit["x"] - intruder["x"]
                dy = unit["y"] - intruder["y"]
                return (dx * dx + dy * dy, unit["unit_id"])

            engageable = sorted(engageable, key=closeness)[: self._guard_cap]
        current = tuple(
            Engagement(
                attacker_id=unit["unit_id"],
                target_id=intruder["unit_id"],
                reason=f"{unit['type_name']} intercepts {intruder['type_name']}",
            )
            for unit in engageable
        )
        self._guarding |= {engagement["attacker_id"] for engagement in current}
        attacks = dispatch_attacks(current, self._ordered, self._attacked)
        self.intercepts += len(attacks)
        return attacks

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
