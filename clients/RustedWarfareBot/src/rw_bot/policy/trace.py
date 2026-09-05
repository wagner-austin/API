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
        killer: Type name of the unit that last damaged it before it went,
            empty when nothing had -- a unit that vanished untouched left
            the roster some other way (a conversion completing, a roster
            read mid-change), and the blank is that distinction on the
            record. Read off the PREVIOUS sample exactly as the position
            is: the engine keeps ``lastDamagedBy`` current as damage lands
            ([[policy-trace]]).
    """

    frame: int
    unit_id: int
    type_name: str
    x: float
    y: float
    killer: str


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
        income: Credits per second, the engine's own figure for the local
            player. The number every economy verdict regresses toward, carried
            per sample because the endpoints hide when it moved: ``income 0/s``
            at the end is the shared shape of every wipe, and the sample it
            went to zero is the finding ([[policy-economy]]).
        rival_income: The same figure for the player :func:`~rw_bot.policy.\
scoreboard.best_rival` reads its worth from, so the pair describes one
            opponent. Worth is the accumulated past; income is the compounding
            rate the match is actually decided by, and the asymmetry between
            this column and ours is the race law in one number per sample
            ([[policy-economy]]).
        plan: The opening plan's outcome this observation -- ``building``,
            ``done``, ``blocked`` or ``stalled``. Appended for the exact-timing
            collapse (log 2026-08-06): every ledger counted totals, and the
            question that decided the diagnosis -- WHEN did the plan die, and
            did it ever come back -- had no record anywhere.
        workers: Builders owned, as the workforce counts them. The other
            column that diagnosis kept needing and inferring from death
            ledgers: every economy failure this project has recorded runs
            through the worker count, and "when did the workforce die" should
            be a column read, not an inference ([[policy-economy]]).
        navy_seen: Hostile WATER-movers visible this sample, APPENDED after
            ``workers`` so every positional reader's index survives
            (autopsy at 4-13, the exporter's first fifteen). The enemy-shape
            column the fleet-doom question demanded: the first ML pass on the
            certified corpus read chance (AUC ~0.55) because the trace
            recorded our economy's shape and nothing about the enemy's
            (log 2026-08-09).
        air_seen: Hostile fliers visible this sample, the air half of the
            same record.
        navy_blood: Cumulative kills on us by fleet types seen so far -- the
            death ledger's answer for the WATER-movers, per sample, so the
            moment the fleet first draws blood is a column read.
        events: Decision codes issued since the previous row was written,
            sorted, ``-`` for none: ``T`` the counter tilt changed the mix,
            ``R`` a raid drafted, ``M`` a forced march moved, ``S`` the
            strike window stood open, ``C`` the closer held its commitment,
            ``B`` the brace armed -- razing predicted, reserve zeroed,
            expansion stood down ([[impossible-step-three-design]]),
            ``H`` the hunt took a new objective ([[engine-ai-triggers]]).
            Every tilt postmortem to date INFERRED firing from composition
            side-effects; a decision is now a column read, cut at recording
            boundaries so the loop's order never bends for the record
            (log 2026-08-09).
        eco_covered: Finished extractors of ours standing inside at least
            one visible hostile gun's reach this sample -- the spatial
            layer's first columns (log 2026-08-15), APPENDED after
            ``events`` so every positional reader's index survives, the
            same appendix rule the enemy-shape columns followed.
        own_covered: Owned complete entities standing inside hostile
            reach, structures and army alike.
        foe_covered: Visible hostiles standing inside our own guns'
            reach; the pair with ``own_covered`` is the engagement
            balance no single count carries.
        world: A deterministic digest of every visible entity's identity,
            position and health -- CRC32 over a canonical rendering, never
            Python's randomised ``hash``. The divergence detector: two
            replicas of one seed agree on this column up to the exact sample
            the simulation forks, which turns "runs do not reproduce" into a
            sample number and a first divergent unit
            ([[policy-determinism]]).
        rival_army: The strongest surviving hostile's ARMY value -- the
            exact figure :class:`~rw_bot.policy.situation.Momentum` windows,
            as distinct from ``rival`` which is worth (army plus buildings)
            and UNDERSTATES army drops whenever buildings grow through a
            wave's death. Appended after ``foe_covered`` per the appendix
            rule so every positional reader's index survives. Zero when the
            sample carries no scoreboard; a drop reader must skip zeros
            exactly as Momentum skips recording them, because a zero read
            as a value fakes a peak-sized fall. The column exists because
            every drop-gated knob (strike, rebuild) reads this signal and
            no record of its actual range at any rung existed -- the 15,000
            thresholds were calibrated against worth dips that top out at
            14,150 across 144 Impossible scorecards (imprb48, log
            2026-09-04).
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
    income: int
    navy_seen: int
    air_seen: int
    navy_blood: int
    events: str
    rival_income: int
    world: int
    plan: str
    workers: int
    eco_covered: int
    own_covered: int
    foe_covered: int
    rival_army: int


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
            killer=entity["damaged_by"],
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
    # The income pair sits between rival and world rather than at the end, so
    # every column an existing reader indexes (extractors 4, lost 5, worth 10,
    # rival 11) keeps its position and only the world digest moves. A reader of
    # the digest is comparing two traces of one build, which agree on shape; a
    # reader of the figures is often crossing the change, and those are the
    # indices that must not shift ([[policy-trace]]).
    lines = [
        f"{'frame':>8}{'army':>6}{'credits':>9}{'enemies':>9}{'extractors':>12}"
        f"{'lost':>6}{'producers':>11}{'idle':>6}{'orders':>8}{'refused':>9}"
        f"{'worth':>9}{'rival':>9}{'income':>8}{'rival_income':>14}{'world':>12}"
        f"{'plan':>10}{'workers':>9}"
        f"{'navy_seen':>11}{'air_seen':>10}{'navy_blood':>12}{'events':>8}"
        f"{'eco_covered':>13}{'own_covered':>13}{'foe_covered':>13}{'rival_army':>12}"
    ]
    lines.extend(
        f"{t['frame']:>8}{t['army']:>6}{t['credits']:>9}"
        f"{t['enemies']:>9}{t['extractors']:>12}{t['lost']:>6}"
        f"{t['producers']:>11}{t['idle']:>6}{t['orders']:>8}{t['refused']:>9}"
        f"{t['worth']:>9}{t['rival']:>9}{t['income']:>8}{t['rival_income']:>14}"
        f"{t['world']:>12}{t['plan']:>10}{t['workers']:>9}"
        f"{t['navy_seen']:>11}{t['air_seen']:>10}{t['navy_blood']:>12}{t['events']:>8}"
        f"{t['eco_covered']:>13}{t['own_covered']:>13}{t['foe_covered']:>13}"
        f"{t['rival_army']:>12}"
        for t in ticks
    )
    lines.append("")
    lines.append(f"{'frame':>8}{'unit':>8}  {'type':<18}{'x':>9}{'y':>9}  {'killer':<18}")
    lines.extend(
        f"{loss['frame']:>8}{loss['unit_id']:>8}  {loss['type_name']:<18}"
        f"{loss['x']:>9.0f}{loss['y']:>9.0f}  {loss['killer'] or '-':<18}"
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
