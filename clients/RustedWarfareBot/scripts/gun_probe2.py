"""Watch turret-upgrade offers through a real contested match.

The peaceful probe exonerated the machinery: standing turrets publish all
four upgrade options and the ladder orders on sight (`runs/gun-probe.out`,
log 2026-08-04). The zone screens' silence is therefore situational, and
only a per-sample record of what the options stream carries DURING a real
match can name the condition. A teeing channel logs, for every sample, the
turret roster and each turret-upgrade option row, while the genuine
campaign -- cover, flame, guns, the works -- plays over it.

Run through the harness against a real opponent::

    make play PLAY_MODULE=scripts.gun_probe2 PLAY_OPPONENTS=1 \\
        PLAY_DIFFICULTY=2 PLAY_ARGS="1500"

The timeline lands beside the run log as ``runs/gun-probe2-timeline.txt``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.control import _test_hooks
from rw_bot.control.channel import DEFAULT_HOST, DEFAULT_TIMEOUT_S, AgentChannel
from rw_bot.policy.budget import Budget
from rw_bot.policy.campaign import play
from rw_bot.policy.convert import TurretLadder
from rw_bot.policy.expand import expand
from rw_bot.policy.match_report import format_report
from rw_bot.wire.state import Sample
from scripts.play import (
    load_build_tree,
    load_catalogue,
    load_combat_profiles,
    load_placements,
)

#: The zone's opening with the turrets forced by plan rather than left to
#: cover's builder walks -- the empty third probe showed five cover orders
#: producing zero standing turrets, so the observation window never opened.
PROBE_GOALS: tuple[str, ...] = (
    "extractorT1",
    "extractorT1",
    "c_turret_t1",
    "c_turret_t1",
    "c_tank",
    "c_tank",
    "c_tank",
)

#: The turret whose upgrade offers are in question, and its fork's targets.
BASE_TURRET = "c_turret_t1"

TIMELINE_PATH = Path("runs/gun-probe2-timeline.txt")

DEFAULT_SAMPLES = 1500

EXIT_OK = 0
EXIT_BAD_USAGE = 2


class TimelineChannel(AgentChannel):
    """An agent channel that records the turret-option timeline as it reads.

    Every sample the campaign consumes is inspected on the way through: one
    line per sample naming each standing base turret (with ``complete`` and
    ``queued``) and one line per upgrade option row any of them carries.
    Quiet samples -- no base turret standing -- write nothing, so the file
    reads as the mystery's own chronology.
    """

    def __init__(self, connection: _test_hooks.Connection, out: Path) -> None:
        """Open the channel over a connection, teeing to a timeline file.

        Args:
            connection: The connected socket line-reader, exactly what
                :class:`AgentChannel` itself is built over.
            out: Where the timeline is written.
        """
        super().__init__(connection)
        self._out = out.open("w", encoding="utf-8")
        self._sample_index = 0

    def next_sample(self) -> Sample:
        """Read the next sample, logging the turret-option rows in passing.

        Returns:
            The sample, unmodified.
        """
        sample = super().next_sample()
        index = self._sample_index
        self._sample_index += 1
        turrets = {
            entity["unit_id"]: entity
            for entity in sample["entities"]
            if entity["mine"] and entity["type_name"] == BASE_TURRET
        }
        if turrets:
            roster = " ".join(
                f"{unit_id}(c={int(turret['complete'])},q={turret['queued']})"
                for unit_id, turret in sorted(turrets.items())
            )
            self._out.write(f"s{index} turrets: {roster}\n")
            for option in sample["options"]:
                if option["unit_id"] in turrets:
                    self._out.write(
                        f"s{index}   {option['unit_id']} -> {option['produces']}"
                        f" avail={int(option['available'])} price={option['price']}\n"
                    )
            # The decisive column: a FRESH ladder, rich and stateless, asked
            # against the very sample the campaign is about to consume. If
            # this orders while the campaign's persistent ladder stays
            # silent, the fault is state; if neither moves, the fault is in
            # the sample after all.
            fresh = TurretLadder().convert(sample, Budget(99_999, 0), 2)
            self._out.write(f"s{index} fresh-ladder: {[dict(o) for o in fresh]!r}\n")
        return sample

    def close_timeline(self) -> None:
        """Flush and close the timeline file."""
        self._out.close()


def main(argv: Sequence[str] | None = None, timeline: Path = TIMELINE_PATH) -> int:
    """Play the zone's campaign over a teeing channel and keep the timeline.

    Args:
        argv: ``<port> <catalogue-path> <placement-path> [samples]``. ``None``
            reads ``sys.argv[1:]``.
        timeline: Where the timeline is written, a parameter so a test can
            point it at a scratch file.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` on a bad argument count.

    Raises:
        ChannelError: When the agent closes the connection.
        OSError: When the connection fails or a dump cannot be read.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4):
        sys.stdout.write("usage: gun_probe2 <port> <catalogue-path> <placement-path> [samples]\n")
        return EXIT_BAD_USAGE
    samples = int(args[3]) if len(args) == 4 else DEFAULT_SAMPLES

    catalogue = load_catalogue(Path(args[1]))
    placements = load_placements(Path(args[2]))
    profiles = load_combat_profiles(Path(args[2]))
    tree = load_build_tree(Path(args[2]))

    channel = TimelineChannel(
        _test_hooks.connect(DEFAULT_HOST, int(args[0]), DEFAULT_TIMEOUT_S), timeline
    )
    opening = channel.next_sample()
    channel.send_ack()
    owned = [e["type_name"] for e in opening["entities"] if e["mine"] and e["complete"]]
    plan = expand(PROBE_GOALS, tree, owned, catalogue)

    report = play(
        channel,
        plan,
        catalogue,
        placements,
        profiles,
        samples,
        reinforce=("c_tank", "c_tank", "hoverTank"),
        reserve=900,
        expand=True,
        counter=True,
        cover=True,
        intercept=True,
        aa_cover=True,
        tech=1,
        flame=2,
        guns=2,
    )
    channel.close_timeline()
    channel.close()
    for line in format_report(report):
        sys.stdout.write(f"{line}\n")
    sys.stdout.write(f"timeline written to {timeline}\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
