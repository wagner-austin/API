"""Fire the first targeted ability live: build a nuke and land it on a point.

The wire grew its eighth verb -- ``ability_at``, the same key dispatch as the
tech verb with the ground point chosen by the planner -- because the nuke
launch is declared ``fireTurretXAtGround: siloTop``: the engine aims the silo
at the point the command carries, so the point is the whole decision
([[mechanics-build-actions]]). This asks the engine whether the verb works
end to end, with the answer written in hit points.

The chain under test, exactly as the finisher doctrine would play it: the
plan places a ``nukeLauncherC`` (45,000, placed by the ordinary builder, no
tech gate), the ``buildNuke`` action stockpiles a warhead (11,000, ammo), and
``launchNuke`` fires at a chosen ground point. The point is the probe's own
extractor farthest from the base -- its coordinates are known to the sample,
so "the nuke landed where it was sent" is a fact about a named unit dying,
not a judgement call. The nuke's 5,400 area damage over radius 250 kills
anything so aimed at (`nuke_launcher.ini`).

Run through the harness, which owns bringing the game up::

    make play PLAY_MODULE=scripts.nuke_probe PLAY_OPPONENTS=1 \\
        PLAY_DIFFICULTY=-2 PLAY_ARGS="2500 2500"
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.control.channel import AgentChannel, open_channel
from rw_bot.mechanics.upgrades import TIER_CHAINS, satisfies
from rw_bot.policy.campaign import play
from rw_bot.policy.expand import expand
from rw_bot.policy.match_report import format_report
from rw_bot.wire.command import ability_order, targeted_ability_order
from rw_bot.wire.state import BuildOption, Entity, Sample
from scripts.play import (
    load_build_tree,
    load_catalogue,
    load_combat_profiles,
    load_placements,
)

#: What the plan funds and places before the probe takes over: income enough
#: to carry the launcher, then the launcher itself. Plan claims are protected,
#: so the 45,000 accumulates instead of leaking into the economy's own wants.
PROBE_GOALS: tuple[str, ...] = (
    "extractorT1",
    "extractorT1",
    "extractorT1",
    "nukeLauncherC",
)

#: The launcher and its target, by registry name.
LAUNCHER_TYPE = "nukeLauncherC"
EXTRACTOR_TYPE = "extractorT1"

#: Every tier of the target family. By the time the launcher is funded the
#: opening has upgraded every extractor, and the third run proved the cost of
#: matching the base name alone: armed at s160, a live launch row, and no
#: launch ever fired because "extractorT1" matched nothing on the field --
#: the same figure-that-quietly-means-something-else that once reported
#: ``extractors 0 -> 0`` on a 54/s economy ([[policy-holding-ground]],
#: `runs/nuke-probe3.out`).
TARGET_FAMILY: frozenset[str] = frozenset(
    name for chain in TIER_CHAINS for name in chain if satisfies(name, EXTRACTOR_TYPE)
)

#: Samples between launch attempts. The launch action reports available at
#: zero ammo -- the ammo price is not in the flag (`runs/nuke-probe3.out`) --
#: so a launch fired before the warhead finishes may be dropped without a
#: word, and the probe refires until the blast circle answers.
LAUNCH_RETRY_SAMPLES = 300

#: The warhead's blast radius, from the asset: anything inside dies, and the
#: probe reports the roster inside it before and after (`nuke_launcher.ini`,
#: areaDamage 5400 over areaRadius 250).
BLAST_RADIUS = 250.0

DEFAULT_OPENING_SAMPLES = 2500
DEFAULT_WATCH_SAMPLES = 2500

EXIT_OK = 0
EXIT_NO_LAUNCH = 1
EXIT_BAD_USAGE = 2


def _mine_of(sample: Sample, type_name: str) -> tuple[Entity, ...]:
    """Return the owned, finished entities of one type, in roster order."""
    return tuple(
        entity
        for entity in sample["entities"]
        if entity["mine"] and entity["complete"] and entity["type_name"] == type_name
    )


def _launcher_options(sample: Sample, launcher: Entity) -> tuple[BuildOption, ...]:
    """Return the launcher's own option rows, exactly as published."""
    return tuple(option for option in sample["options"] if option["unit_id"] == launcher["unit_id"])


def _pick_target(sample: Sample) -> Entity | None:
    """Choose the owned extractor farthest from the command centre.

    Farthest, so the blast is as far from the base as the roster allows --
    the observable is the target dying, not the base going with it. Any
    tier counts (:data:`TARGET_FAMILY`): the opening upgrades every
    extractor long before the launcher is funded.
    """
    anchors = _mine_of(sample, "commandCenter")
    extractors = tuple(
        entity
        for entity in sample["entities"]
        if entity["mine"] and entity["complete"] and entity["type_name"] in TARGET_FAMILY
    )
    if not anchors or not extractors:
        return None
    anchor = anchors[0]

    def spread(extractor: Entity) -> float:
        return (extractor["x"] - anchor["x"]) ** 2 + (extractor["y"] - anchor["y"]) ** 2

    return max(extractors, key=spread)


def _inside_blast(sample: Sample, x: float, y: float) -> tuple[int, ...]:
    """Return the owned unit ids standing inside the blast circle at (x, y)."""
    return tuple(
        entity["unit_id"]
        for entity in sample["entities"]
        if entity["mine"] and (entity["x"] - x) ** 2 + (entity["y"] - y) ** 2 <= BLAST_RADIUS**2
    )


def _refire(
    channel: AgentChannel,
    index: int,
    fired_at: int,
    options: tuple[BuildOption, ...],
    launcher: Entity,
    target_x: float,
    target_y: float,
) -> int:
    """Fire the launch again once the retry window passes unanswered.

    A launch fired before the warhead finished is dropped without a word --
    the option's flag does not carry the ammo gate, reading available at
    zero ammo (`runs/nuke-probe3.out`) -- so the point is fired at again
    until the world answers.

    Args:
        channel: The open channel.
        index: The current watch sample.
        fired_at: The sample the last launch went out on.
        options: The launcher's option rows this sample.
        launcher: The launcher itself.
        target_x: Target world x, fixed at the first launch.
        target_y: Target world y, fixed at the first launch.

    Returns:
        The sample the most recent launch went out on.
    """
    if index - fired_at < LAUNCH_RETRY_SAMPLES:
        return fired_at
    free = tuple(o for o in options if o["price"] == 0 and o["available"])
    if not free:
        return fired_at
    channel.send_targeted_ability(
        targeted_ability_order(
            unit_id=launcher["unit_id"],
            key=free[0]["key"],
            x=target_x,
            y=target_y,
        )
    )
    sys.stdout.write(
        f"s{index} relaunched: '{free[0]['key']}' at ({target_x:.0f}, {target_y:.0f})\n"
    )
    return index


def _watch(channel: AgentChannel, samples: int) -> int:
    """Drive the ability chain by hand and report what the engine did.

    One pass per sample: stockpile a warhead the first time the launcher
    offers it affordably, launch at the chosen extractor the first time the
    launch action reports available, and after the launch report the blast
    circle's roster until it empties or the watch runs out.

    Args:
        channel: The open channel, already past the opening plan.
        samples: How many samples to watch before giving up.

    Returns:
        ``EXIT_OK`` once everything inside the blast circle died,
        ``EXIT_NO_LAUNCH`` when the chain never completed.
    """
    armed = False
    launched = False
    fired_at = 0
    target_x = 0.0
    target_y = 0.0
    seen_rows: tuple[tuple[str, int, int], ...] = ()
    for index in range(samples):
        sample = channel.next_sample()
        launchers = _mine_of(sample, LAUNCHER_TYPE)
        if not launchers:
            channel.send_ack()
            continue
        launcher = launchers[0]
        options = _launcher_options(sample, launcher)
        # The launcher's state and rows, printed whenever they change: the
        # second run armed and then watched 2,300 samples of silence, and
        # nothing in its output could say whether the ammo never built or
        # the launch row publishes in a shape the filter does not match
        # (`runs/nuke-probe2.out`, log 2026-08-05). ``queued`` is the
        # disambiguator the engine's silent drops demand: 1 means the
        # warhead is genuinely building, 0 after the arm means the dispatch
        # was dropped without a word.
        rows = (
            ("queued", launcher["queued"], int(launcher["complete"])),
            *((o["key"], o["price"], int(o["available"])) for o in options),
        )
        if rows != seen_rows:
            sys.stdout.write(f"s{index} launcher: {list(rows)!r}\n")
            seen_rows = rows
        if not armed:
            # The stockpile action: the priced row. The launch is priced in
            # ammo, which the wire reports as zero credits.
            priced = tuple(o for o in options if o["price"] > 0 and o["available"])
            if priced and sample["credits"] >= priced[0]["price"]:
                channel.send_ability(
                    ability_order(unit_id=launcher["unit_id"], key=priced[0]["key"])
                )
                armed = True
                sys.stdout.write(
                    f"s{index} armed: '{priced[0]['key']}' price {priced[0]['price']}\n"
                )
        elif not launched:
            free = tuple(o for o in options if o["price"] == 0 and o["available"])
            target = _pick_target(sample)
            if free and target is not None:
                target_x, target_y = target["x"], target["y"]
                channel.send_targeted_ability(
                    targeted_ability_order(
                        unit_id=launcher["unit_id"],
                        key=free[0]["key"],
                        x=target_x,
                        y=target_y,
                    )
                )
                launched = True
                fired_at = index
                sys.stdout.write(
                    f"s{index} launched: '{free[0]['key']}' at "
                    f"({target_x:.0f}, {target_y:.0f}) -- extractor {target['unit_id']}\n"
                )
        else:
            standing = _inside_blast(sample, target_x, target_y)
            if not standing:
                sys.stdout.write(f"s{index} inside blast: []\n")
                sys.stdout.write("verdict: the targeted point was cleared\n")
                channel.send_ack()
                return EXIT_OK
            fired_at = _refire(channel, index, fired_at, options, launcher, target_x, target_y)
        channel.send_ack()
    state = "launched, blast never cleared" if launched else "armed" if armed else "unarmed"
    sys.stdout.write(f"verdict: no confirmed strike ({state})\n")
    return EXIT_NO_LAUNCH


def main(argv: Sequence[str] | None = None) -> int:
    """Play the funding opening, then fire the chain by hand.

    Args:
        argv: ``<port> <catalogue-path> <placement-path> [opening [watch]]``.
            ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` on a confirmed strike, ``EXIT_NO_LAUNCH`` when the chain
        never completed, ``EXIT_BAD_USAGE`` on a bad argument count.

    Raises:
        ChannelError: When the agent closes the connection.
        OSError: When the connection fails or a dump cannot be read.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4, 5):
        sys.stdout.write(
            "usage: nuke_probe <port> <catalogue-path> <placement-path> [opening [watch]]\n"
        )
        return EXIT_BAD_USAGE
    opening_samples = int(args[3]) if len(args) >= 4 else DEFAULT_OPENING_SAMPLES
    watch_samples = int(args[4]) if len(args) == 5 else DEFAULT_WATCH_SAMPLES

    catalogue = load_catalogue(Path(args[1]))
    placements = load_placements(Path(args[2]))
    profiles = load_combat_profiles(Path(args[2]))
    tree = load_build_tree(Path(args[2]))

    channel = open_channel(int(args[0]))
    first = channel.next_sample()
    channel.send_ack()
    owned = [e["type_name"] for e in first["entities"] if e["mine"] and e["complete"]]
    plan = expand(PROBE_GOALS, tree, owned, catalogue)

    # The opening is played by the real loop with cover OFF, and the first
    # run is why: cover buys a 500-credit turret for every bare structure,
    # and the plan's 45,000 wait does not withhold -- so the save's
    # shortfall never shrank, the wait was ruled blocked, and the launcher
    # was never ordered (`runs/nuke-probe.out`, log 2026-08-05). Income
    # only: extractors, their upgrades, and then a clean climb to the
    # launcher's price.
    report = play(
        channel, plan, catalogue, placements, profiles, opening_samples, expand=True, cover=False
    )
    for line in format_report(report):
        sys.stdout.write(f"{line}\n")

    verdict = _watch(channel, watch_samples)
    channel.close()
    return verdict


if __name__ == "__main__":
    raise SystemExit(main(None))
