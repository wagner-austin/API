"""One row per sample, so a run can be read back instead of guessed at.

A match produced 1,500 observations and the scorecard kept about a dozen
numbers. That is enough to say *what* happened and never *why*: ``army 4 -> 12``
is identical whether thirty-seven units died in one bad fight or bled away two
at a time for six minutes, and those call for opposite fixes. Two changes landed
together, the result got worse, and nothing in the run could attribute it
([[policy-combat]]).

So the loop writes what it sees. Two tables, because they answer different
questions and folding them into one would fudge both:

* **per sample** -- how many units, credits, enemies and extractors, and how
  many were lost since the previous observation. This answers *when*.
* **per loss** -- the unit, its type, and where it was standing when last seen.
  This answers *where*, which is what separates "dying on the walk home" from
  "dying at the enemy front".

A loss is inferred rather than reported: the engine sends no death event, so a
unit that was ours last sample and is absent now is counted as lost. That is not
quite the same claim -- a unit can leave the roster by other means -- and the
distinction is recorded here rather than hidden behind the word.

Pure, like the rest of the policy layer. :mod:`rw_bot.policy.campaign` is what
writes the file.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.wire.state import Entity, Sample


class Loss(TypedDict):
    """One unit that was ours and is not any more.

    Attributes:
        frame: Engine frame of the sample it went missing on.
        unit_id: Engine identity of the unit.
        type_name: What it was.
        x: World x where it was last seen.
        y: World y where it was last seen.
    """

    frame: int
    unit_id: int
    type_name: str
    x: float
    y: float


class Tick(TypedDict):
    """What one observation looked like.

    Attributes:
        frame: Engine frame counter.
        army: Units able to fight.
        credits: Credits held.
        enemies: Hostile entities visible.
        extractors: Finished extractors owned.
        lost: Units that went missing since the previous sample.
        producers: Owned units the engine says can make something wanted. Zero
            here while credits climb is a capability failure -- the unit cap or
            tech gating -- rather than a spending one, because availability is
            the engine's own predicate ([[mechanics-build-actions]]).
        idle: How many of those held nothing in their queue. Persistently zero
            means the queues are saturated and build time is the throttle;
            persistently non-zero beside a rising balance means something
            declined to fill them.
        orders: Produce orders actually issued this observation.
        refused: Credit claims the budget turned down this observation.
        worth: Everything the player holds, mobile and standing alike.
        rival: The strongest hostile player's total. Carried per observation
            because the endpoints cannot show a dip, and a dip is the only
            evidence in the whole report that the army ever cost an opponent
            anything ([[policy-verdict]]).
        world: A deterministic digest of every visible entity's identity,
            position and health -- CRC32 over a canonical rendering, never
            Python's randomised ``hash``. The divergence detector: two
            replicas of one seed agree on this column up to the exact sample
            the simulation forks, which turns "runs do not reproduce" into a
            sample number and a first divergent unit
            ([[policy-determinism]]).
    """

    frame: int
    army: int
    credits: int
    enemies: int
    extractors: int
    lost: int
    producers: int
    idle: int
    orders: int
    refused: int
    worth: int
    rival: int
    world: int


def owned_by_id(sample: Sample) -> Mapping[int, Entity]:
    """Index the player's own entities by engine identity.

    Args:
        sample: One observation of the world.

    Returns:
        Owned entities by id.
    """
    return {e["unit_id"]: e for e in sample["entities"] if e["mine"]}


def losses_between(
    previous: Mapping[int, Entity], current: Mapping[int, Entity], frame: int
) -> tuple[Loss, ...]:
    """Report the units present a sample ago and absent now.

    Inferred, not observed. The stream carries no death event, so this is
    "left the roster" rather than "was killed" — a unit can also leave by
    finishing a transformation or by the roster being read mid-change. Naming
    the inference is the point; a column called ``killed`` would be a claim the
    data does not support.

    Positions come from the *previous* sample, because the current one no
    longer has the unit to ask.

    Args:
        previous: Owned entities from the previous sample, by id.
        current: Owned entities from this sample, by id.
        frame: Engine frame of this sample.

    Returns:
        One loss per id that has gone, in the previous sample's order.
    """
    return tuple(
        Loss(
            frame=frame,
            unit_id=unit_id,
            type_name=entity["type_name"],
            x=entity["x"],
            y=entity["y"],
        )
        for unit_id, entity in previous.items()
        if unit_id not in current
    )


def format_trace(ticks: Sequence[Tick], losses: Sequence[Loss]) -> tuple[str, ...]:
    """Render both tables as aligned text.

    Aligned columns rather than CSV because the first reader is a person
    scanning for the sample where something turned, and a comma-separated wall
    hides exactly that. It parses on whitespace for anything that wants to plot
    it.

    Args:
        ticks: One entry per sample, in order.
        losses: Every inferred loss, in order.

    Returns:
        The lines, without newline terminators.
    """
    lines = [
        f"{'frame':>8}{'army':>6}{'credits':>9}{'enemies':>9}{'extractors':>12}"
        f"{'lost':>6}{'producers':>11}{'idle':>6}{'orders':>8}{'refused':>9}"
        f"{'worth':>9}{'rival':>9}{'world':>12}"
    ]
    lines.extend(
        f"{t['frame']:>8}{t['army']:>6}{t['credits']:>9}"
        f"{t['enemies']:>9}{t['extractors']:>12}{t['lost']:>6}"
        f"{t['producers']:>11}{t['idle']:>6}{t['orders']:>8}{t['refused']:>9}"
        f"{t['worth']:>9}{t['rival']:>9}{t['world']:>12}"
        for t in ticks
    )
    lines.append("")
    lines.append(f"{'frame':>8}{'unit':>8}  {'type':<18}{'x':>9}{'y':>9}")
    lines.extend(
        f"{loss['frame']:>8}{loss['unit_id']:>8}  {loss['type_name']:<18}"
        f"{loss['x']:>9.0f}{loss['y']:>9.0f}"
        for loss in losses
    )
    return tuple(lines)


def world_digest(sample: Sample) -> int:
    """Digest every visible entity into one deterministic number.

    CRC32 over a canonical rendering of (id, type, position, health), in id
    order. Never Python's ``hash``: that is salted per process, and the whole
    point is comparing two processes. Positions at a tenth of a world unit --
    coarser would hide slow drift, finer would flag float noise below what
    the simulation acts on.

    Args:
        sample: One observation of the world.

    Returns:
        The digest, stable across processes and platforms.
    """
    parts = [
        (
            f"{e['unit_id']}:{e['type_name']}:{e['x']:.1f}:{e['y']:.1f}"
            f":{e['hp']:.1f}:{int(e['complete'])}"
        )
        for e in sorted(sample["entities"], key=_by_unit_id)
    ]
    return zlib.crc32("|".join(parts).encode("utf-8"))


def _by_unit_id(entity: Entity) -> int:
    """Order entities by engine id, the one cross-run-stable ordering."""
    return entity["unit_id"]


__all__ = ["Loss", "Tick", "format_trace", "losses_between", "owned_by_id", "world_digest"]
