"""Forward pressure on the enemy's forming groups.

The shipped AI attacks by fill-then-commit: a group recruits toward a
fixed target size and only a full group attacks -- except that **damage
to any member inside the last 1,000 frames cancels staging and commits
the group at whatever size it has** ([[engine-ai-triggers]], watched
live at Impossible in the m32 capture). Every measured arm to date meets
waves after they commit whole. The hunt is the first that goes and
touches them first: a small party pushing at the nearest visible enemy
mover, so groups are bled while they form and committed before they
fill.

The party holds the raid's discipline, extracted for exactly this
second holder ([[policy-raid]], :mod:`rw_bot.policy.party`): drafted
whole from the gathered, disbanded home under strength, and arbitrated
by the campaign against the wave gate's own need. What is the hunt's
own is the objective class -- **visible hostile movers first, remembered
structures when nothing moves in sight** -- so with no target visible
the party pushes toward the enemy's base and finds one.

Pure in the usual sense: samples and memory in, orders out, and the
campaign sends them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.combat import FIRST_WAVE
from rw_bot.policy.intel import Intel, Sighting
from rw_bot.policy.party import draft_gathered, homeward
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import AttackMoveOrder, attack_move_order
from rw_bot.wire.state import Entity, Sample


def _quarry(
    targets: Sequence[Entity],
    intel: Intel,
    catalogue: Mapping[str, UnitStats],
    centre_x: float,
    centre_y: float,
) -> tuple[int, float, float] | None:
    """Choose the objective: nearest visible mover, else nearest memory.

    Args:
        targets: The hostile entities visible this observation.
        intel: The fog memory, read for the fallback objective.
        catalogue: Unit stats by type name; zero speed is how the
            catalogue reports immobility, and a type it cannot price
            cannot prove it moves.
        centre_x: The party centre's x -- pursuit measures from the party.
        centre_y: The party centre's y.

    Returns:
        The objective's unit id and position, or None with nothing seen
        and nothing remembered. Ties break on unit id, so two runs of one
        seed hunt identically.
    """
    movers = []
    for entity in targets:
        stats = catalogue.get(entity["type_name"])
        if stats is not None and stats["speed"] > 0:
            movers.append(entity)
    if movers:

        def nearness(entity: Entity) -> tuple[float, int]:
            dx = entity["x"] - centre_x
            dy = entity["y"] - centre_y
            return (dx * dx + dy * dy, entity["unit_id"])

        quarry = min(movers, key=nearness)
        return (quarry["unit_id"], quarry["x"], quarry["y"])
    remembered = intel.remembered()
    if not remembered:
        return None

    def sighting_nearness(sighting: Sighting) -> tuple[float, int]:
        dx = sighting["x"] - centre_x
        dy = sighting["y"] - centre_y
        return (dx * dx + dy * dy, sighting["unit_id"])

    memory = min(remembered, key=sighting_nearness)
    return (memory["unit_id"], memory["x"], memory["y"])


def _centroid(members: Sequence[Entity]) -> tuple[float, float]:
    """Return the party's own centre, the point pursuit measures from.

    Args:
        members: The party's live entities, at least one.

    Returns:
        The mean position.
    """
    n = len(members)
    return (
        sum(unit["x"] for unit in members) / n,
        sum(unit["y"] for unit in members) / n,
    )


class Hunter:
    """Keeps one small party pressing the nearest visible enemy mover.

    The same shape as the other stateful controllers: decisions stay in
    pure reads, and what lives here is the memory between observations --
    who is in the party, which mover it presses, and what has already
    been ordered ([[issuing-orders]]).

    Attributes:
        size: Party size, public because the campaign arbitrates the
            draft against it -- surplus is the wave gate's need plus this.
        hunts: Objective changes so far, for the report.
        marches: Member-orders sent so far, for the report.
    """

    def __init__(self, size: int = FIRST_WAVE) -> None:
        """Open a hunter.

        Args:
            size: Party size. Defaults to the engine's own first-group size
                for the raid's reason: below it the engine's AI calls a
                force a trickle, and so does ours ([[engine-ai-triggers]]).
                The campaign gates on the knob, so an unused hunter is
                constructed and never consulted.
        """
        self.size = size
        self.hunts = 0
        self.marches = 0
        self._party: frozenset[int] = frozenset()
        self._objective = 0
        self._ordered: dict[int, int] = {}

    def party(self) -> frozenset[int]:
        """Return the engine ids currently drafted.

        The campaign withholds these from the wave controller: a unit
        cannot serve two commanders, and assignment is the arbitration
        ([[engine-ai-zones]]).

        Returns:
            The party, empty when nothing is drafted.
        """
        return self._party

    def stand_down(
        self, army: Sequence[Entity], catalogue: Mapping[str, UnitStats], sample: Sample
    ) -> tuple[AttackMoveOrder, ...]:
        """Recall the party and dissolve it.

        The gated arm's response: when the head predicts the razing, the
        party fights its way home and rejoins the reserve. Idempotent --
        with no party out there is nothing to recall and nothing is sent.

        Args:
            army: Units available to fight.
            catalogue: Unit stats by type name, for the anchor.
            sample: One observation of the world.

        Returns:
            The homeward orders, empty with no party or no anchor.
        """
        alive = {unit["unit_id"] for unit in army}
        survivors = sorted(self._party & alive)
        self._party = frozenset()
        self._objective = 0
        self._ordered = {}
        if not survivors:
            return ()
        anchor = find_anchor(sample, catalogue)
        if anchor is None:
            return ()
        return homeward(survivors, anchor)

    def press(
        self,
        sample: Sample,
        intel: Intel,
        army: Sequence[Entity],
        targets: Sequence[Entity],
        catalogue: Mapping[str, UnitStats],
        may_draft: bool,
    ) -> tuple[AttackMoveOrder, ...]:
        """Advance the hunt by at most one objective's worth of orders.

        The objective is the visible hostile MOVER nearest the party's
        own centre -- pursuit measures from where the party stands, not
        from home -- falling back to the remembered enemy structure
        nearest that same centre when nothing moves in sight, so an
        empty horizon walks the party toward the enemy's base until one
        does. Ties break on unit id, so two runs of one seed hunt
        identically.

        Args:
            sample: One observation of the world.
            intel: The fog memory, read for the fallback objective.
            army: Units available to fight, scouts already excluded.
            targets: The hostile entities visible this observation.
            catalogue: Unit stats by type name, for the anchor and for
                telling movers from buildings (zero speed is how the
                catalogue reports immobility).
            may_draft: Whether the campaign judges the army able to spare
                a fresh party. A party already out is managed regardless.

        Returns:
            The attack-move orders to send, empty while the party is
            already pressing its objective or there is nothing to hunt.
        """
        anchor = find_anchor(sample, catalogue)
        if anchor is None:
            return ()
        alive = {unit["unit_id"] for unit in army}
        survivors = sorted(self._party & alive)
        if survivors and len(survivors) < self.size:
            self._party = frozenset()
            self._objective = 0
            self._ordered = {}
            return homeward(survivors, anchor)
        party = survivors
        if not party and may_draft:
            party = draft_gathered(army, anchor, self.size)
        self._party = frozenset(party)
        if not self._party:
            return ()

        members = [unit for unit in army if unit["unit_id"] in self._party]
        centre_x, centre_y = _centroid(members)
        quarry = _quarry(targets, intel, catalogue, centre_x, centre_y)
        if quarry is None:
            return ()
        objective_id, x, y = quarry

        if objective_id != self._objective:
            self._objective = objective_id
            self._ordered = {}
            self.hunts += 1
        orders = tuple(
            attack_move_order(unit_id=member, x=x, y=y)
            for member in sorted(self._party)
            if self._ordered.get(member) != objective_id
        )
        for order in orders:
            self._ordered[order["unit_id"]] = objective_id
        self.marches += len(orders)
        return orders


__all__ = ["Hunter"]
