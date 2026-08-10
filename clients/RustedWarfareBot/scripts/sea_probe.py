"""Ask a live map where a sea factory may stand, by asking the engine.

The naval theater's gate ([[policy-exact-timing]], the naval wall; log
2026-08-10): the enemy's fleet core declares ``No anti-air or anti-sub``
while the $800 attack submarine outranges it submerged and untouchable --
but everything routes through a ``seaFactory``, which "can only be built
on water", and the planner has no terrain map. This probe prototypes the
answer the siting layer will ship: **terrain discovery by attempt**. Walk
candidate points along the anchor-to-mirror line -- the lake lies between
the starts on every symmetric water map -- issue a build order at each,
and let the engine's accept or ignore be the terrain sensor.

What it reports, in order: which line fraction the engine first accepts a
sea factory at; whether the factory completes; which units its options
rows offer (does the attack submarine need the T2 upgrade?); and whether a
produced submarine actually enters the world.

Run through the harness, which owns bringing the game up and tearing it
down::

    make play PLAY_MODULE=scripts.sea_probe PLAY_ARGS="600"
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.control.channel import open_channel
from rw_bot.policy.rush import mirror_point
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import build_order, produce_order
from rw_bot.wire.state import Entity, Sample
from scripts.play import load_catalogue

#: Fractions of the anchor-to-mirror line to offer the engine, nearest
#: first: the shore closest to our base is the buildable edge a builder
#: can reach, and the lake's middle is the fallback.
FRACTIONS: tuple[float, ...] = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)

#: Samples to wait on each candidate before calling it refused. Builders
#: walk before they build, so patience is part of the measurement.
PATIENCE = 40

#: The structure under test and the unit that justifies it.
FACTORY = "seaFactory"
SUBMARINE = "attackSubmarine"

DEFAULT_SAMPLES = 600

EXIT_OK = 0
EXIT_NO_WATER = 1
EXIT_BAD_USAGE = 2


def _factory(sample: Sample) -> Entity | None:
    """Return our sea factory's entity row, if one exists."""
    for entity in sample["entities"]:
        if entity["mine"] and entity["type_name"] == FACTORY:
            return entity
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Walk the line, place the factory, name what it offers.

    Args:
        argv: ``<port> <catalogue-path> <placement-path> [samples]``.
            ``None`` reads ``sys.argv[1:]``. The placement path rides the
            harness's standard argument order and is deliberately unused:
            terrain-by-attempt is the whole point.

    Returns:
        ``EXIT_OK`` when a factory stood, ``EXIT_NO_WATER`` when every
        candidate was refused, ``EXIT_BAD_USAGE`` on a bad argument count.

    Raises:
        ChannelError: When the agent closes the connection.
        OSError: When the connection fails or a dump cannot be read.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4):
        sys.stdout.write("usage: sea_probe <port> <catalogue-path> <placement-path> [samples]\n")
        return EXIT_BAD_USAGE
    samples = int(args[3]) if len(args) == 4 else DEFAULT_SAMPLES
    catalogue = load_catalogue(Path(args[1]))

    channel = open_channel(int(args[0]))
    try:
        sample = channel.next_sample()
        anchor = find_anchor(sample, catalogue)
        goal = mirror_point(sample, catalogue)
        builders = [e for e in sample["entities"] if e["mine"] and e["type_name"] == "builder"]
        if anchor is None or goal is None or not builders:
            channel.send_ack()
            sys.stdout.write("[sea] no anchor, mirror or builder to probe with\n")
            return EXIT_NO_WATER
        builder_id = builders[0]["unit_id"]
        sys.stdout.write(
            f"[sea] anchor ({anchor['x']:.0f},{anchor['y']:.0f})"
            f" mirror ({goal[0]:.0f},{goal[1]:.0f}) builder {builder_id}\n"
        )

        candidate = 0
        waited = 0
        placed_at = -1.0
        produced = False
        for _ in range(samples):
            factory = _factory(sample)
            if factory is None and candidate < len(FRACTIONS):
                share = FRACTIONS[candidate]
                x = anchor["x"] + (goal[0] - anchor["x"]) * share
                y = anchor["y"] + (goal[1] - anchor["y"]) * share
                channel.send_build(build_order(unit_id=builder_id, type_name=FACTORY, x=x, y=y))
                waited += 1
                if waited >= PATIENCE:
                    sys.stdout.write(f"[sea] fraction {share:.2f}: refused after {PATIENCE}\n")
                    candidate += 1
                    waited = 0
            elif factory is not None:
                if placed_at < 0:
                    placed_at = FRACTIONS[candidate] if candidate < len(FRACTIONS) else -1.0
                    sys.stdout.write(
                        f"[sea] STANDS at fraction {placed_at:.2f}:"
                        f" ({factory['x']:.0f},{factory['y']:.0f})"
                        f" complete={factory['complete']}\n"
                    )
                if factory["complete"] and not produced:
                    offers = [
                        f"{o['produces'] or '(none)'}[avail={o['available']} price={o['price']}]"
                        for o in sample["options"]
                        if o["unit_id"] == factory["unit_id"]
                    ]
                    sys.stdout.write(f"[sea] options: {', '.join(offers)}\n")
                    channel.send_produce(
                        produce_order(unit_id=factory["unit_id"], type_name=SUBMARINE)
                    )
                    produced = True
            subs = [e for e in sample["entities"] if e["mine"] and e["type_name"] == SUBMARINE]
            if subs:
                sys.stdout.write(
                    f"[sea] SUBMARINE afloat: unit {subs[0]['unit_id']}"
                    f" at ({subs[0]['x']:.0f},{subs[0]['y']:.0f})\n"
                )
                channel.send_ack()
                return EXIT_OK
            channel.send_ack()
            sample = channel.next_sample()
        sys.stdout.write(f"[sea] done: placed_at={placed_at:.2f} produced={produced}\n")
        return EXIT_OK if placed_at >= 0 else EXIT_NO_WATER
    finally:
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main(None))
