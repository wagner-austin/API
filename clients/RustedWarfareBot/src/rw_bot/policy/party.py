"""The party discipline every detached force holds.

The raid learned these rules the expensive way -- v1 topped its party up
one recruit at a time and each replacement attack-moved across the map
alone, forever, refuted 0/12 (log 2026-07-29) -- and every later holder
inherits them rather than re-learning them: **drafted whole, from the
gathered** (a fresh party is only taken from units standing within the
rally radius of the anchor, lowest id first, so two runs of one seed draft
identically), and **a party or nothing** (survivors below strength
disband and attack-move home, fighting their way back rather than
standing where the party broke).

Extracted from :class:`~rw_bot.policy.raid.Raider` when the hunt became
the second holder of the same rules ([[policy-raid]]); the bookkeeping
between observations -- who is in the party, which objective it is on,
what has already been ordered -- became :class:`Detachment` when the dive
became the third, because two copies had already drifted on nothing and
a third would have drifted on exactly the questions v1 settled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.combat import FIRST_WAVE, RALLY_RADIUS
from rw_bot.wire.command import AttackMoveOrder, attack_move_order
from rw_bot.wire.state import Entity


def _gathered(army: Sequence[Entity], anchor: Entity) -> list[Entity]:
    """Return the units standing within the rally radius of the anchor.

    Args:
        army: Units available to fight.
        anchor: The structure the reserve gathers at.

    Returns:
        The gathered units, in roster order.
    """
    limit = RALLY_RADIUS**2
    return [
        unit
        for unit in army
        if (unit["x"] - anchor["x"]) ** 2 + (unit["y"] - anchor["y"]) ** 2 <= limit
    ]


def draft_gathered(army: Sequence[Entity], anchor: Entity, size: int) -> list[int]:
    """Pick a whole party from the units gathered at the anchor, or none.

    Args:
        army: Units available to fight.
        anchor: The structure the reserve gathers at.
        size: The party size a draft must fill.

    Returns:
        The new party in id order, empty when the gathering ground holds
        fewer than a party.
    """
    gathered = sorted(unit["unit_id"] for unit in _gathered(army, anchor))
    if len(gathered) < size:
        return []
    return gathered[:size]


def draft_fastest(
    army: Sequence[Entity],
    anchor: Entity,
    size: int,
    catalogue: Mapping[str, UnitStats],
) -> list[int]:
    """Pick a whole party of the FASTEST units gathered at the anchor, or none.

    The dive's draft: a party sent to close on a gun that outranges it is
    a race across the gun's reach, and the mix carries a hover slot for
    exactly that leg. Speed is the catalogue's figure; a type the
    catalogue cannot price is the slowest thing on the ground, so it is
    drafted last rather than never. Ties break on the lowest id, the
    ordering every draft in this codebase uses, so two runs of one seed
    draft identically.

    Args:
        army: Units available to fight.
        anchor: The structure the reserve gathers at.
        size: The party size a draft must fill.
        catalogue: Unit stats by type name, for the speed.

    Returns:
        The new party in id order, empty when the gathering ground holds
        fewer than a party.
    """

    def swiftness(unit: Entity) -> tuple[float, int]:
        stats = catalogue.get(unit["type_name"])
        return (-(stats["speed"] if stats is not None else 0.0), unit["unit_id"])

    gathered = sorted(_gathered(army, anchor), key=swiftness)
    if len(gathered) < size:
        return []
    return sorted(unit["unit_id"] for unit in gathered[:size])


def homeward(survivors: Sequence[int], anchor: Entity) -> tuple[AttackMoveOrder, ...]:
    """Send an under-strength party home fighting.

    Attack-move rather than move, because the road home crosses the same
    ground the road out did. Once home the survivors are the wave
    controller's again -- the campaign stops withholding whatever is no
    longer in a party.

    Args:
        survivors: The remaining members, in id order.
        anchor: The structure the reserve gathers at.

    Returns:
        The homeward orders.
    """
    return tuple(
        attack_move_order(unit_id=member, x=anchor["x"], y=anchor["y"]) for member in survivors
    )


class Detachment:
    """The memory one detached party keeps between observations.

    Decisions stay in the holders' own pure reads -- which objective, which
    draft -- and what lives here is the bookkeeping they all share: who is
    in the party, which objective it is on, and what has already been
    ordered, so an order is never re-sent while its objective holds
    ([[issuing-orders]]).

    Attributes:
        size: Party size, public because the campaign arbitrates the draft
            against it -- surplus is the wave gate's need plus this.
        objectives: Objectives taken so far, for the report.
        marches: Member-orders sent so far, for the report. The figure that
            would have convicted v1 on its first scorecard: its objective
            count read 2-6 while dozens of lone replacements marched,
            because re-drafts against the same objective counted nothing.
    """

    def __init__(self, size: int = FIRST_WAVE) -> None:
        """Open a detachment.

        Args:
            size: Party size. Defaults to the engine's own first-group size:
                below it the engine's AI calls a force a trickle, and so does
                ours ([[engine-ai-triggers]]).
        """
        self.size = size
        self.objectives = 0
        self.marches = 0
        self._party: frozenset[int] = frozenset()
        self._objective = 0
        self._ordered: dict[int, int] = {}

    def party(self) -> frozenset[int]:
        """Return the engine ids currently drafted.

        The campaign withholds these from the wave controller: a unit cannot
        serve two commanders, and assignment is the arbitration
        ([[engine-ai-zones]]).

        Returns:
            The party, empty when nothing is drafted.
        """
        return self._party

    def survivors(self, army: Sequence[Entity]) -> list[int]:
        """Return the drafted members still alive, in id order.

        A party that died whole leaves ids behind; an empty answer here is
        what lets the holder drop them, so the campaign stops withholding
        ghosts from the waves.

        Args:
            army: Units available to fight.

        Returns:
            The living members.
        """
        alive = {unit["unit_id"] for unit in army}
        return sorted(self._party & alive)

    def dissolve(self) -> None:
        """Forget the party, its objective and its orders."""
        self._party = frozenset()
        self._objective = 0
        self._ordered = {}

    def disband(self, survivors: Sequence[int], anchor: Entity) -> tuple[AttackMoveOrder, ...]:
        """Send the under-strength party home fighting, and dissolve it.

        Args:
            survivors: The remaining members, in id order.
            anchor: The structure the reserve gathers at.

        Returns:
            The homeward orders.
        """
        self.dissolve()
        return homeward(survivors, anchor)

    def muster(self, members: Sequence[int]) -> bool:
        """Hold the given members as the party.

        Args:
            members: The party in id order, possibly empty.

        Returns:
            Whether a party now stands.
        """
        self._party = frozenset(members)
        return bool(self._party)

    def advance(self, objective_id: int, x: float, y: float) -> tuple[AttackMoveOrder, ...]:
        """Send every member not already on the objective at it.

        A new objective counts once and clears the sent record, so every
        member is re-sent; a standing objective sends nothing to a member
        already ordered at it, because the engine runs a waypoint until it
        is replaced and a copy of itself restarts the walk.

        Args:
            objective_id: Engine identity of the objective.
            x: World x of the objective.
            y: World y of the objective.

        Returns:
            The attack-move orders to send, in id order.
        """
        if objective_id != self._objective:
            self._objective = objective_id
            self._ordered = {}
            self.objectives += 1
        orders = tuple(
            attack_move_order(unit_id=member, x=x, y=y)
            for member in sorted(self._party)
            if self._ordered.get(member) != objective_id
        )
        for order in orders:
            self._ordered[order["unit_id"]] = objective_id
        self.marches += len(orders)
        return orders

    def forget_objective(self) -> None:
        """Drop the standing objective so the next one counts afresh."""
        self._objective = 0


__all__ = ["Detachment", "draft_fastest", "draft_gathered", "homeward"]
