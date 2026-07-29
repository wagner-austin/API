"""Ask a live game what an owned structure offers to become.

Plays the opening until extractors are standing, then prints every option any
owned structure carries. That settles, from the engine rather than from the
build tree, whether the extractor upgrade path is reachable -- the largest
economic lever available and the only one needing no builder, no travel and no
contested ground ([[policy-holding-ground]]).

Run through the harness, which owns bringing the game up and tearing it down::

    make play PLAY_MODULE=scripts.upgrade_probe PLAY_ARGS="400"
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.control.channel import open_channel
from rw_bot.mechanics.upgrades import format_offers, upgrade_offers
from rw_bot.policy.campaign import play
from rw_bot.policy.economy import count_extractors
from rw_bot.policy.expand import expand
from scripts.play import (
    load_build_tree,
    load_catalogue,
    load_combat_profiles,
    load_placements,
)

#: What to build before asking. Extractors, because they are the structure whose
#: upgrade path is in question.
PROBE_GOALS: tuple[str, ...] = ("extractorT1", "extractorT1")

DEFAULT_SAMPLES = 400

EXIT_OK = 0
EXIT_NO_STRUCTURE = 1
EXIT_BAD_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Play an opening, then report what the standing structures offer.

    Args:
        argv: ``<port> <catalogue-path> <placement-path> [samples]``. ``None``
            reads ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when a structure was standing to ask, ``EXIT_NO_STRUCTURE``
        when the opening never completed one, ``EXIT_BAD_USAGE`` on a bad
        argument count.

    Raises:
        ChannelError: When the agent closes the connection.
        OSError: When the connection fails or a dump cannot be read.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4):
        sys.stdout.write(
            "usage: upgrade_probe <port> <catalogue-path> <placement-path> [samples]\n"
        )
        return EXIT_BAD_USAGE
    samples = int(args[3]) if len(args) == 4 else DEFAULT_SAMPLES

    catalogue = load_catalogue(Path(args[1]))
    placements = load_placements(Path(args[2]))
    profiles = load_combat_profiles(Path(args[2]))
    tree = load_build_tree(Path(args[2]))

    channel = open_channel(int(args[0]))
    opening = channel.next_sample()
    channel.send_ack()
    owned = [e["type_name"] for e in opening["entities"] if e["mine"] and e["complete"]]
    plan = expand(PROBE_GOALS, tree, owned, catalogue)

    # The opening is played by the real loop rather than a scripted one, so what
    # the probe observes is a world the bot actually produces.
    play(channel, plan, catalogue, placements, profiles, samples, expand=True)

    final = channel.next_sample()
    channel.send_ack()
    offers = upgrade_offers(final, catalogue)
    sys.stdout.write(f"extractors standing: {count_extractors(final)}\n")
    for line in format_offers(offers):
        sys.stdout.write(f"{line}\n")
    channel.close()
    return EXIT_OK if offers else EXIT_NO_STRUCTURE


if __name__ == "__main__":
    raise SystemExit(main(None))
