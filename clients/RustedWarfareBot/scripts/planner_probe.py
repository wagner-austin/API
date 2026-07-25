"""Drive a live game from Python over the agent channel.

The end-to-end proof for the command channel: connect, read real world state,
choose a unit from it, and issue an order the game executes. Every decision
here is made from the sample rather than from a constant, which is the whole
point of the exercise.

Run against a game started with ``-javaagent:...=channelPort=27200``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from rw_bot.control.channel import open_channel
from rw_bot.wire.command import build_order
from rw_bot.wire.state import Entity, Sample

_BUILDER = "builder"
_STRUCTURE = "landFactory"
_OFFSET_X = 200.0
_OFFSET_Y = 120.0
_SAMPLES_AFTER = 8

EXIT_OK = 0
EXIT_NO_BUILDER = 1
EXIT_BAD_USAGE = 2


def find_builder(sample: Sample) -> Entity | None:
    """Pick the first builder out of a sample.

    Selection by type name rather than roster position: position renumbers
    whenever anything is built or dies, and the agent deliberately holds no
    opinion about which unit can do what.

    Args:
        sample: One observation of the world.

    Returns:
        The first builder, or None when the player owns none.
    """
    for entity in sample["entities"]:
        if entity["type_name"] == _BUILDER:
            return entity
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Connect, order a factory, and report what changed.

    Args:
        argv: Argument list excluding the program name. ``None`` reads
            ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when a builder was found and ordered, ``EXIT_NO_BUILDER``
        when the roster had none, ``EXIT_BAD_USAGE`` on a bad argument count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 1:
        sys.stdout.write("usage: planner_probe <port>\n")
        return EXIT_BAD_USAGE

    channel = open_channel(int(args[0]))
    sample = channel.next_sample()
    sys.stdout.write(
        f"frame {sample['frame']} clock {sample['clock_ms']}ms: {len(sample['entities'])} owned\n"
    )
    for entity in sample["entities"]:
        sys.stdout.write(
            f"  id={entity['unit_id']} {entity['type_name']} at ({entity['x']}, {entity['y']})\n"
        )

    builder = find_builder(sample)
    if builder is None:
        sys.stdout.write("no builder in the roster\n")
        channel.close()
        return EXIT_NO_BUILDER

    target_x = builder["x"] + _OFFSET_X
    target_y = builder["y"] + _OFFSET_Y
    sys.stdout.write(
        f"ordering id={builder['unit_id']} to build {_STRUCTURE} at ({target_x}, {target_y})\n"
    )
    channel.send_build(
        build_order(
            unit_id=builder["unit_id"],
            type_name=_STRUCTURE,
            x=target_x,
            y=target_y,
        )
    )

    for _ in range(_SAMPLES_AFTER):
        later = channel.next_sample()
        names = sorted(e["type_name"] for e in later["entities"])
        sys.stdout.write(f"frame {later['frame']}: {names}\n")
    channel.close()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
