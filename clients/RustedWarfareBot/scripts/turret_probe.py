"""Ask a live match whether an artillery turret can answer the enemy fleet.

The naval hole's cheapest untested response (log 2026-08-11: the theater
closed at -8, the tilt closed at -5, and "do nothing" stands unbeaten by
default). The stat sheet proposes a standoff battery: the ground turret's
artillery fork reaches 350 against the battleship's 240, and ships cannot
answer what outranges them ([[mechanics-combat-profile]]). Before any
channel or panel exists, law eleven wants the mechanism proven live: a
builder cannot place ``c_turret_t1_artillery`` directly -- the buildable
chain is basic turret first, then the four-way tier fork the flame
conversion already walks ([[policy-doctrine]], the ``flame`` knob).

What it reports, in order: which line fraction the engine first accepts a
ground turret at (terrain discovery by attempt, the sea probe's own
sensor); whether the conversion to the artillery fork is accepted and
completes; every hostile that enters the battery's reach; and -- the
smoking gun -- any hostile whose ``damaged_by`` names the battery, which
is the engine's own attribution of a hit ([[wire-contract-ndjson]]).

Run through the harness, which owns bringing the game up and tearing it
down::

    make play PLAY_MODULE=scripts.turret_probe PLAY_ARGS="2500" \\
        PLAY_MAP="maps/skirmish/[p2]duel_lake.tmx" PLAY_DIFFICULTY=3
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from rw_bot.control.channel import AgentChannel, open_channel
from rw_bot.policy.rush import mirror_point
from rw_bot.policy.siting import find_anchor
from rw_bot.wire.command import build_order, produce_order
from rw_bot.wire.state import Entity, Sample
from scripts.play import load_catalogue

#: Fractions of the anchor-to-mirror line to offer the engine, nearest the
#: water first: the sea probe measured the lake starting at 0.25 on this
#: line (log 2026-08-10), so the shore-most land the builder can hold lies
#: just below it, and every step back concedes reach over the water.
FRACTIONS: Final[tuple[float, ...]] = (0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10)

#: Samples to wait on each candidate before calling it refused. Builders
#: walk before they build, so patience is part of the measurement.
PATIENCE: Final = 40

#: The structure the builder can place and the fork under test.
TURRET: Final = "c_turret_t1"
BATTERY: Final = "c_turret_t1_artillery"

#: The battery's reach, from the type registry; hostiles inside it are
#: reported because each one is a shot the engine may take.
REACH: Final = 350.0

DEFAULT_SAMPLES: Final = 2500

EXIT_OK: Final = 0
EXIT_NO_GROUND: Final = 1
EXIT_BAD_USAGE: Final = 2


def _mine_of_type(sample: Sample, type_name: str) -> Entity | None:
    """Return our first entity of the named type, if one exists."""
    for entity in sample["entities"]:
        if entity["mine"] and entity["type_name"] == type_name:
            return entity
    return None


class Probe:
    """The probe's memory across samples: siting, conversion, observation.

    Attributes:
        placed_at: Line fraction the turret stood at, negative until then.
        battery_stood: Whether the artillery fork ever existed.
        bled_types: Hostile types the battery has damaged, in first-blood
            order -- the mechanism proof a pilot's card would carry.
    """

    def __init__(self, builder_id: int, anchor: Entity, goal: tuple[float, float]) -> None:
        """Remember the line to walk and the builder walking it.

        Args:
            builder_id: The one builder, kept by id until it dies -- the
                shape every navy panel that "improved" on it died without
                (log 2026-08-10).
            anchor: Our base structure the line starts from.
            goal: The mirrored enemy start the line runs to.
        """
        self.placed_at = -1.0
        self.battery_stood = False
        self.bled_types: list[str] = []
        self._builder_id = builder_id
        self._anchor_x = anchor["x"]
        self._anchor_y = anchor["y"]
        self._goal_x = goal[0]
        self._goal_y = goal[1]
        self._candidate = 0
        self._waited = 0
        self._offers_shown = False
        self._battery_x = 0.0
        self._battery_y = 0.0
        self._in_reach: set[int] = set()
        self._blooded: set[int] = set()

    def site(self, channel: AgentChannel) -> None:
        """Offer the current fraction a turret; step back when refused."""
        if self._candidate >= len(FRACTIONS):
            return
        share = FRACTIONS[self._candidate]
        x = self._anchor_x + (self._goal_x - self._anchor_x) * share
        y = self._anchor_y + (self._goal_y - self._anchor_y) * share
        channel.send_build(build_order(unit_id=self._builder_id, type_name=TURRET, x=x, y=y))
        self._waited += 1
        if self._waited >= PATIENCE:
            sys.stdout.write(f"[battery] fraction {share:.2f}: refused after {PATIENCE}\n")
            self._candidate += 1
            self._waited = 0

    def convert(self, channel: AgentChannel, sample: Sample, turret: Entity) -> None:
        """Report the standing turret once, then re-send the fork order.

        Conversion never fills a queue, so the order re-sends until the
        fork's product exists -- the conversion channel's own duplicate
        rule, inverted for a probe that wants exactly one.
        """
        if self.placed_at < 0:
            self.placed_at = (
                FRACTIONS[self._candidate] if self._candidate < len(FRACTIONS) else -1.0
            )
            sys.stdout.write(
                f"[battery] turret STANDS at fraction {self.placed_at:.2f}:"
                f" ({turret['x']:.0f},{turret['y']:.0f}) complete={turret['complete']}\n"
            )
        if not turret["complete"]:
            return
        if not self._offers_shown:
            self._offers_shown = True
            offers = [
                f"{o['produces'] or '(none)'}[avail={o['available']} price={o['price']}]"
                for o in sample["options"]
                if o["unit_id"] == turret["unit_id"]
            ]
            sys.stdout.write(f"[battery] turret offers: {', '.join(offers)}\n")
        channel.send_produce(produce_order(unit_id=turret["unit_id"], type_name=BATTERY))

    def observe(self, sample: Sample, battery: Entity) -> None:
        """Report the standing battery, reach entries, and blood drawn."""
        if not self.battery_stood:
            self.battery_stood = True
            self._battery_x = battery["x"]
            self._battery_y = battery["y"]
            sys.stdout.write(
                f"[battery] BATTERY STANDS: unit {battery['unit_id']}"
                f" at ({self._battery_x:.0f},{self._battery_y:.0f})"
                f" complete={battery['complete']} hp={battery['hp']:.0f}\n"
            )
        for entity in sample["entities"]:
            if not entity["hostile"]:
                continue
            span = math.hypot(entity["x"] - self._battery_x, entity["y"] - self._battery_y)
            if span <= REACH and entity["unit_id"] not in self._in_reach:
                self._in_reach.add(entity["unit_id"])
                sys.stdout.write(
                    f"[battery] in reach: {entity['type_name']} ({entity['movement']})"
                    f" at {span:.0f} hp={entity['hp']:.0f}/{entity['max_hp']:.0f}\n"
                )
            if entity["damaged_by"] == BATTERY and entity["unit_id"] not in self._blooded:
                self._blooded.add(entity["unit_id"])
                self.bled_types.append(entity["type_name"])
                sys.stdout.write(
                    f"[battery] DREW BLOOD: {entity['type_name']} ({entity['movement']})"
                    f" at {span:.0f} hp={entity['hp']:.0f}/{entity['max_hp']:.0f}\n"
                )


def main(argv: Sequence[str] | None = None) -> int:
    """Place the turret, convert it, and watch who bleeds.

    Args:
        argv: ``<port> <catalogue-path> <placement-path> [samples]``.
            ``None`` reads ``sys.argv[1:]``. The placement path rides the
            harness's standard argument order and is deliberately unused:
            terrain-by-attempt is the whole point.

    Returns:
        ``EXIT_OK`` when the battery stood, ``EXIT_NO_GROUND`` when every
        candidate was refused or the conversion never completed,
        ``EXIT_BAD_USAGE`` on a bad argument count.

    Raises:
        ChannelError: When the agent closes the connection.
        OSError: When the connection fails or a dump cannot be read.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4):
        sys.stdout.write("usage: turret_probe <port> <catalogue-path> <placement-path> [samples]\n")
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
            sys.stdout.write("[battery] no anchor, mirror or builder to probe with\n")
            return EXIT_NO_GROUND
        probe = Probe(builders[0]["unit_id"], anchor, goal)
        sys.stdout.write(
            f"[battery] anchor ({anchor['x']:.0f},{anchor['y']:.0f})"
            f" mirror ({goal[0]:.0f},{goal[1]:.0f}) builder {builders[0]['unit_id']}\n"
        )

        for _ in range(samples):
            turret = _mine_of_type(sample, TURRET)
            battery = _mine_of_type(sample, BATTERY)
            if battery is not None:
                probe.observe(sample, battery)
            elif turret is not None:
                probe.convert(channel, sample, turret)
            else:
                probe.site(channel)
            channel.send_ack()
            sample = channel.next_sample()

        fate = "stands" if _mine_of_type(sample, BATTERY) is not None else "gone"
        sys.stdout.write(
            f"[battery] done: placed_at={probe.placed_at:.2f}"
            f" battery={probe.battery_stood} fate={fate}"
            f" bled={','.join(probe.bled_types) or '(nothing)'}\n"
        )
        return EXIT_OK if probe.battery_stood else EXIT_NO_GROUND
    finally:
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main(None))
