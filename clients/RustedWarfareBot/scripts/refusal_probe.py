"""Force the engine's silent placement refusal, and read the report back.

The one probe that fires the detection chain deterministically: a landFactory
ordered onto the command centre's own footprint always fails the construction
attempt's blocked-pair test -- the centre occupies the site -- and the engine
drops the waypoint with no log line. The proof this probe exists to print is
the ``refused`` record arriving on the wire: engine truth read back, not a
prediction ([[engine-silent-refusal]]).

Run as the play harness's module: ``make play PLAY_MODULE=scripts.refusal_probe``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from rw_bot.control.channel import open_channel
from rw_bot.wire.command import build_order
from rw_bot.wire.state import Entity, Sample

_BUILDER = "builder"
_CENTRE = "commandCenter"
_STRUCTURE = "landFactory"

#: How many samples to wait for the report. The builder has to walk into
#: build range before the construction attempt runs, so the answer is not
#: immediate; the bound exists so a broken chain is an exit code rather than
#: a probe that never returns.
_SAMPLES_BOUND = 400

EXIT_OK = 0
EXIT_NO_UNITS = 1
EXIT_NO_REFUSAL = 2
EXIT_BAD_USAGE = 3


def find_unit(sample: Sample, type_name: str) -> Entity | None:
    """Pick the first OWNED entity of a type out of a sample.

    Ownership is not optional: fog is disabled on some maps, so the first
    builder in the roster can be the opponent's -- and the dispatch refuses
    an order for a unit the player does not own, which reads as the probe
    silently doing nothing.

    Args:
        sample: One observation of the world.
        type_name: The type to look for.

    Returns:
        The first owned match, or None when the player has none.
    """
    for entity in sample["entities"]:
        if entity["mine"] and entity["type_name"] == type_name:
            return entity
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Order a build the engine must refuse, and wait for it to say so.

    Args:
        argv: Argument list excluding the program name: the port, then the
            catalogue and type-dump paths the launcher hands every planner
            module -- unused here, the probe needs no prices, but part of the
            contract. ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` once a refusal record arrives, ``EXIT_NO_UNITS`` when the
        roster lacks a builder or a command centre, ``EXIT_NO_REFUSAL`` when
        the bound runs out first, ``EXIT_BAD_USAGE`` on a bad argument count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 3:
        sys.stdout.write("usage: refusal_probe <port> <catalogue-path> <type-dump-path>\n")
        return EXIT_BAD_USAGE

    channel = open_channel(int(args[0]))
    sample = channel.next_sample()
    channel.send_ack()
    builder = find_unit(sample, _BUILDER)
    centre = find_unit(sample, _CENTRE)
    if builder is None or centre is None:
        sys.stdout.write("roster lacks a builder or a command centre\n")
        channel.close()
        return EXIT_NO_UNITS

    # The centre's own coordinates: the one placement the blocked-pair test
    # can never pass, whatever the map or the seed.
    sys.stdout.write(
        f"ordering id={builder['unit_id']} to build {_STRUCTURE} at "
        f"({centre['x']}, {centre['y']}) -- the centre's own footprint\n"
    )
    channel.send_build(
        build_order(
            unit_id=builder["unit_id"],
            type_name=_STRUCTURE,
            x=centre["x"],
            y=centre["y"],
        )
    )

    for _ in range(_SAMPLES_BOUND):
        later = channel.next_sample()
        channel.send_ack()
        if later["refusals"]:
            for refusal in later["refusals"]:
                sys.stdout.write(
                    f"frame {later['frame']}: engine refused {refusal['type_name']} "
                    f"at ({refusal['x']}, {refusal['y']}) for unit {refusal['unit_id']}\n"
                )
            channel.close()
            return EXIT_OK

    sys.stdout.write(f"no refusal within {_SAMPLES_BOUND} samples\n")
    channel.close()
    return EXIT_NO_REFUSAL


if __name__ == "__main__":
    raise SystemExit(main(None))
