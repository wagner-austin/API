"""Prove the attack-move verb against a live game.

The claim under test is the one the decompile makes: a move command with the
engine's own flag set makes the unit engage what it meets instead of walking
past it ([[community-play-strategies]]). The probe produces a scout -- the
Command Center makes one directly, so no factory chain is needed -- waits for
a hostile to show itself, orders the scout to attack-move *past* it, and then
prints the observable series: the scout's position against its destination,
and the nearest hostile's health. Engagement shows as the hostile's health
falling while the scout is alive and short of its destination; a plain move
would carry it past without a shot.

Run against a game started with ``-javaagent:...=channelPort=<port>``. Every
sample is acknowledged, so the probe is lockstep-safe.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from math import sqrt

from rw_bot.control.channel import open_channel
from rw_bot.wire.command import attack_move_order, produce_order
from rw_bot.wire.state import Entity, Sample

#: The producer that makes the probe's unit without a factory chain.
_PRODUCER = "commandCenter"

#: The unit ordered to attack-move. Armed, mobile, and one produce away.
_SCOUT = "scout"

#: Samples to wait for the scout and then for a hostile before giving up.
#: Generous because the hostile arrives on the opponent's schedule, not ours.
_WAIT_SAMPLES = 1200

#: Samples observed after the order, printing the series the ruling reads.
_OBSERVE_SAMPLES = 500

#: World units past the hostile the destination is placed, so arrival and
#: engagement cannot be confused: the scout has no reason to stop at the
#: hostile unless the flag makes it.
_OVERSHOOT = 300.0

EXIT_OK = 0
EXIT_NO_PRODUCER = 1
EXIT_TIMEOUT = 2
EXIT_BAD_USAGE = 3


def find_owned(sample: Sample, type_name: str) -> Entity | None:
    """Pick the first owned, finished entity of a type.

    Args:
        sample: One observation of the world.
        type_name: The type wanted.

    Returns:
        The entity, or None when the player owns none.
    """
    for entity in sample["entities"]:
        if entity["mine"] and entity["complete"] and entity["type_name"] == type_name:
            return entity
    return None


def first_hostile(sample: Sample) -> Entity | None:
    """Pick the first visible hostile.

    Args:
        sample: One observation of the world.

    Returns:
        The hostile, or None when nothing hostile is in sight.
    """
    for entity in sample["entities"]:
        if entity["hostile"]:
            return entity
    return None


def past(scout: Entity, hostile: Entity) -> tuple[float, float]:
    """Return a destination beyond the hostile, on the scout's line to it.

    Args:
        scout: The unit that will attack-move.
        hostile: The enemy on the way.

    Returns:
        World x and y, :data:`_OVERSHOOT` units past the hostile.
    """
    dx = hostile["x"] - scout["x"]
    dy = hostile["y"] - scout["y"]
    length = max(sqrt(dx * dx + dy * dy), 1.0)
    return (
        hostile["x"] + dx / length * _OVERSHOOT,
        hostile["y"] + dy / length * _OVERSHOOT,
    )


def series_line(sample: Sample) -> str:
    """Render one observation of the ruling's series.

    The nearest hostile, not the one first sighted: the ruling is about what
    the scout does to whatever it MEETS, and what it meets is whichever enemy
    stands on the way. Engagement reads as that distance stalling around the
    scout's reach while that hostile's health falls; a plain move reads as the
    distance shrinking straight through. Empty states are named rather than
    blank -- a missing line reads as a probe failure, not a death.

    Args:
        sample: One observation of the world.

    Returns:
        One printed line, with its newline.
    """
    mine = find_owned(sample, _SCOUT)
    if mine is None:
        return f"frame {sample['frame']}: scout gone\n"
    nearest: Entity | None = None
    nearest_d2 = 0.0
    for entity in sample["entities"]:
        if not entity["hostile"]:
            continue
        d2 = (entity["x"] - mine["x"]) ** 2 + (entity["y"] - mine["y"]) ** 2
        if nearest is None or d2 < nearest_d2:
            nearest = entity
            nearest_d2 = d2
    foe_part = (
        f"nearest {nearest['unit_id']} ({nearest['type_name']}) "
        f"at {sqrt(nearest_d2):.0f} hp {nearest['hp']:.0f}"
        if nearest is not None
        else "no hostile in sight"
    )
    return (
        f"frame {sample['frame']}: scout ({mine['x']:.0f}, {mine['y']:.0f}) "
        f"hp {mine['hp']:.0f}; {foe_part}\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Connect, produce a scout, attack-move it past a hostile, and report.

    Args:
        argv: ``<port> [catalogue-path] [type-flags-path]``. The two paths are
            accepted and unused, because the play recipe hands every planner
            module the same argument tail; the probe decides everything from
            live samples. ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when the order was issued and the series printed,
        ``EXIT_NO_PRODUCER`` when the roster never offered a Command Center,
        ``EXIT_TIMEOUT`` when no scout or no hostile appeared in the window,
        ``EXIT_BAD_USAGE`` on a bad argument count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (1, 3):
        sys.stdout.write("usage: attackmove_probe <port> [catalogue] [type-flags]\n")
        return EXIT_BAD_USAGE

    channel = open_channel(int(args[0]))
    sample = channel.next_sample()
    channel.send_ack()

    producer = find_owned(sample, _PRODUCER)
    if producer is None:
        sys.stdout.write("no commandCenter in the opening roster\n")
        channel.close()
        return EXIT_NO_PRODUCER
    channel.send_produce(produce_order(unit_id=producer["unit_id"], type_name=_SCOUT))
    sys.stdout.write(f"scout ordered at {producer['unit_id']}\n")

    scout: Entity | None = None
    hostile: Entity | None = None
    for _ in range(_WAIT_SAMPLES):
        sample = channel.next_sample()
        channel.send_ack()
        scout = find_owned(sample, _SCOUT)
        hostile = first_hostile(sample)
        if scout is not None and hostile is not None:
            break
    if scout is None or hostile is None:
        sys.stdout.write(
            f"gave up after {_WAIT_SAMPLES} samples: "
            f"scout={'yes' if scout is not None else 'no'} "
            f"hostile={'yes' if hostile is not None else 'no'}\n"
        )
        channel.close()
        return EXIT_TIMEOUT

    dest_x, dest_y = past(scout, hostile)
    sys.stdout.write(
        f"attack-move scout {scout['unit_id']} past hostile {hostile['unit_id']} "
        f"({hostile['type_name']}) to ({dest_x:.0f}, {dest_y:.0f})\n"
    )
    channel.send_attack_move(attack_move_order(unit_id=scout["unit_id"], x=dest_x, y=dest_y))

    for _ in range(_OBSERVE_SAMPLES):
        sample = channel.next_sample()
        channel.send_ack()
        sys.stdout.write(series_line(sample))
    channel.close()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
