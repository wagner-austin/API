"""Ask a live game what a standing ground turret offers to become.

The zone screens' convert ledgers were byte-identical across six seeds --
``convert:c_turret_t2_gun asked 75 got 0`` -- meaning exactly one early
turret ever offered its upgrades and the eight-plus cover turrets built
after it never published a claimable offer again (`runs/sweeps/vh-zone`,
log 2026-08-04). Two suspects survive the code reading: the engine's own
``available`` flag on upgrade actions under some mid-game condition, or the
conversion channel's idle filter reading something unexpected. This asks
the engine directly: build turrets by plan, stand them, and print every
option row they carry plus the channel's own verdict on the same sample.

Run through the harness, which owns bringing the game up and tearing it
down::

    make play PLAY_MODULE=scripts.gun_probe PLAY_ARGS="500"
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.control.channel import open_channel
from rw_bot.policy.budget import Budget
from rw_bot.policy.campaign import play
from rw_bot.policy.convert import TurretLadder
from rw_bot.policy.expand import expand
from scripts.play import (
    load_build_tree,
    load_catalogue,
    load_combat_profiles,
    load_placements,
)

#: What to build before asking: an extractor to fund, then two ground
#: turrets -- the structure whose upgrade offers are in question.
PROBE_GOALS: tuple[str, ...] = ("extractorT1", "c_turret_t1", "c_turret_t1")

DEFAULT_SAMPLES = 500

EXIT_OK = 0
EXIT_NO_TURRET = 1
EXIT_BAD_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Play an opening with turrets in it, then report what they offer.

    Args:
        argv: ``<port> <catalogue-path> <placement-path> [samples]``. ``None``
            reads ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when a turret was standing to ask, ``EXIT_NO_TURRET``
        when the opening never completed one, ``EXIT_BAD_USAGE`` on a bad
        argument count.

    Raises:
        ChannelError: When the agent closes the connection.
        OSError: When the connection fails or a dump cannot be read.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4):
        sys.stdout.write("usage: gun_probe <port> <catalogue-path> <placement-path> [samples]\n")
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

    # The opening is played by the real loop rather than a scripted one, so
    # what the probe observes is a world the bot actually produces.
    play(channel, plan, catalogue, placements, profiles, samples, expand=True)

    final = channel.next_sample()
    channel.send_ack()
    turrets = {
        e["unit_id"]: e for e in final["entities"] if e["mine"] and e["type_name"] == "c_turret_t1"
    }
    sys.stdout.write(f"turrets standing: {len(turrets)}\n")
    for unit_id, turret in turrets.items():
        sys.stdout.write(
            f"turret {unit_id}: complete={turret['complete']} queued={turret['queued']}\n"
        )
    for option in final["options"]:
        if option["unit_id"] in turrets:
            sys.stdout.write(
                f"  option {option['produces'] or '(no type)'} key={option['key']}"
                f" available={option['available']} placed={option['placed']}"
                f" makes={option['makes_something']} price={option['price']}\n"
            )
    # The channel's own verdict on the identical sample: whatever it decides,
    # the rows above say why.
    ladder = TurretLadder()
    orders = ladder.convert(final, Budget(final["credits"], 0), 2)
    sys.stdout.write(f"ladder verdict: {[dict(o) for o in orders]!r}\n")
    channel.close()
    return EXIT_OK if turrets else EXIT_NO_TURRET


if __name__ == "__main__":
    raise SystemExit(main(None))
