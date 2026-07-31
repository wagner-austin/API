"""Print every visible entity for the first N samples, then exit.

The determinism campaign's microscope: two runs of this against one seed,
diffed, name the exact entity and coordinate where the simulation forks.
The world digest localised the fork to the first lockstep window
(identical at frame 0, divergent by frame 75); this shows *what* moved
([[policy-determinism]]).

Run as the play recipe runs any planner:
``PLAY_MODULE=scripts.detprobe PLAY_ARGS=<samples>``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from rw_bot.control.channel import open_channel
from rw_bot.wire.command import move_order
from rw_bot.wire.state import Entity

EXIT_OK = 0
EXIT_BAD_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Connect, dump entities per sample, exit.

    Args:
        argv: ``<port> <catalogue> <type-dump> [samples]`` -- the two dump
            paths are accepted and ignored so the standard play recipe can
            drive this module unchanged. ``None`` reads the process
            arguments.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` on a bad argument count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4, 5):
        sys.stdout.write("usage: detprobe <port> <catalogue> <type-dump> [samples] [order]\n")
        return EXIT_BAD_USAGE
    budget = int(args[3]) if len(args) >= 4 else 2
    # The one-command variant: the observer-only run was bit-identical
    # across replicas while the playing run diverged by frame 75, so the
    # fork enters with the command path. One fixed move order at sample
    # zero is the smallest possible dose of it.
    order = len(args) == 5 and args[4] == "order"
    channel = open_channel(int(args[0]))
    for index in range(budget):
        sample = channel.next_sample()
        for e in sorted(sample["entities"], key=_by_id):
            sys.stdout.write(
                f"S{index} f{sample['frame']} c{sample['clock_ms']}"
                f" #{e['unit_id']} {e['type_name']}"
                f" ({e['x']:.3f},{e['y']:.3f}) hp={e['hp']:.3f}"
                f" mine={int(e['mine'])} complete={int(e['complete'])}\n"
            )
        if order and index == 0:
            mover = min(
                (e for e in sample["entities"] if e["mine"] and e["type_name"] == "builder"),
                key=_by_id,
            )
            channel.send_move(
                move_order(unit_id=mover["unit_id"], x=mover["x"] + 100.0, y=mover["y"])
            )
            sys.stdout.write(f"ordered #{mover['unit_id']} +100x\n")
        channel.send_ack()
    channel.close()
    return EXIT_OK


def _by_id(entity: Entity) -> int:
    """Order entities by engine id, the one cross-run-stable ordering."""
    return entity["unit_id"]


if __name__ == "__main__":
    raise SystemExit(main(None))
