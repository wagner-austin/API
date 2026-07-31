"""Play a build order against a live game and print a scorecard.

The bot's entry point. Connects to a running agent, executes a plan by
observing the world and deciding from it, and reports what it achieved so one
run can be compared with another.

Run against a game started with ``-javaagent:...=channelPort=27200``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from rw_bot.control.channel import open_channel
from rw_bot.mechanics.build_tree import decode_build_tree
from rw_bot.mechanics.catalogue import UnitStats, decode_catalogue
from rw_bot.mechanics.combat_profile import CombatProfile, decode_combat_profiles
from rw_bot.mechanics.placement import TypePlacement, decode_placements
from rw_bot.policy.campaign import play
from rw_bot.policy.combat import ladder_to
from rw_bot.policy.doctrine import (
    DEFAULT_DOCTRINE,
    DERIVE_RESERVE,
    Doctrine,
    DoctrineError,
    parse_doctrine_lines,
)
from rw_bot.policy.expand import expand
from rw_bot.policy.match_report import format_report

#: What the default doctrine asks for -- goals, not a build order.
#:
#: The distinction is the point. No factory appears here, and two of these
#: entries need one: a Land Factory is what makes a tank, and the plan the bot
#: actually executes has one inserted before them by
#: :func:`rw_bot.policy.expand.expand`. Writing the prerequisite out by hand is
#: what the build tree exists to stop.
#:
#: Extractors first, because they are the only entry that pays for the rest:
#: each generates credits, and anything built before them is bought with the
#: starting balance and nothing after it. Where they go is not a choice the plan
#: makes -- the engine allows them on resource pools and nowhere else
#: ([[mechanics-resource-pools]]).
DEFAULT_GOALS: tuple[str, ...] = DEFAULT_DOCTRINE["goals"]

DEFAULT_MAX_SAMPLES = 120

#: Samples the opening may stay unit-less before the world is called broken.
#:
#: The opening roster is what plan expansion reads, so the planner acks
#: samples until something owned and finished appears. Bounded because an
#: empty world that never populates is a failed match start, not a slow one:
#: forty samples at lockstep 75 is 3,000 frames -- ten seconds of game time --
#: and the starting units spawn with the map, so a world still empty after
#: that is not going to fill.
OPENING_SETTLE_SAMPLES = 40

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_BAD_USAGE = 2

_UNKNOWN_HEAVY = "RW-DOCTRINE-011"


def load_catalogue(path: Path) -> dict[str, UnitStats]:
    """Read the unit catalogue produced by ``-printunits``.

    Args:
        path: Archived catalogue dump.

    Returns:
        Unit stats by type name.

    Raises:
        OSError: When the file cannot be read.
        CatalogueError: When the dump cannot be decoded.
    """
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    return {unit["type_name"]: unit for unit in decode_catalogue(lines)}


def load_build_tree(path: Path) -> dict[str, frozenset[str]]:
    """Read the build tree produced by ``make type-flags``.

    The same file the placement rules come from, because both are one pass over
    one registry and two files could drift against different game builds.

    Args:
        path: Archived type dump.

    Returns:
        Product type names by producer type name.

    Raises:
        OSError: When the file cannot be read.
        BuildTreeError: When the dump cannot be decoded.
    """
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    return decode_build_tree(lines)


def load_combat_profiles(path: Path) -> Mapping[str, CombatProfile]:
    """Read every registered type's reach and reachable layers.

    The same file the placement flags come from, because both are one pass over
    one registry and two files could be regenerated against different builds and
    disagree.

    Args:
        path: Path to the registry dump.

    Returns:
        Combat profiles by type name.

    Raises:
        OSError: When the file cannot be read.
        CombatProfileError: When the dump carries a type twice.
    """
    return decode_combat_profiles(path.read_text(encoding="utf-8", errors="strict").splitlines())


def load_placements(path: Path) -> dict[str, TypePlacement]:
    """Read the placement rules produced by ``make type-flags``.

    Args:
        path: Archived placement dump.

    Returns:
        Placement rules by type name.

    Raises:
        OSError: When the file cannot be read.
        PlacementError: When the dump cannot be decoded.
    """
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    return {place["type_name"]: place for place in decode_placements(lines)}


def reinforcements(
    goals: Sequence[str],
    catalogue: Mapping[str, UnitStats],
) -> tuple[str, ...]:
    """Return the army composition worth holding once the plan is finished.

    The plan ends; wanting the units it asked for does not. So reinforcement
    repeats the goals rather than ranking units by some invented notion of
    combat worth -- a number attached to a guess is still a guess.

    **Structures are dropped, and immobility is what identifies one.** This used
    to drop only the types needing a resource pool, which was the same test by
    accident: every structure the goals had ever named was an extractor. A
    turret needs no pool and is still placed, so it passed the filter into a
    list of things producers should keep making -- where no producer can make it,
    because a queue cannot express a position ([[mechanics-resource-pools]]).
    Speed is the catalogue's own answer and it identifies every structure, not
    just the ones that stand on pools.

    **Duplicates are kept, and they are the ratio.** They used to be collapsed,
    on the reading that four tanks meant one preference stated four times. That
    reading cost the bot its army: the composition was a strict priority list,
    every producer took the head of it, and three matches ended with 33
    identical ``c_tank`` against opponents fielding aircraft no ``c_tank`` can
    shoot at ([[mechanics-combat-profile]]). Repeats now say what they look
    like they say -- three tanks per anti-air unit is a mix, not a queue
    ([[policy-production]]).

    Args:
        goals: What the plan was asked for, in order.
        catalogue: Unit stats by type name, for the speed that tells a structure
            from a unit.

    Returns:
        Unit type names to keep making, repeats meaningful as a ratio.
    """
    return tuple(name for name in goals if catalogue[name]["speed"] > 0.0)


def heavy_reinforcements(
    heavies: Sequence[str],
    catalogue: Mapping[str, UnitStats],
) -> tuple[str, ...]:
    """Validate the doctrine's extra composition entries.

    The channel the unlocked roster joins the mix through. Unlike the goals,
    these never pass through plan expansion -- the build tree would insert
    the experimental factory as a prerequisite rather than wait for the
    unlock -- so nothing else ever reads them, and the checks the plan would
    have made are made here: the type must be priced, and it must be a unit
    a queue can produce rather than a structure needing a site
    ([[mechanics-build-actions]]).

    Inert until the tier opens, by construction: production orders only what
    the engine's option stream offers as available, so an entry whose
    factory is still tier one is never chosen ([[policy-production]]).

    Args:
        heavies: Type names from the doctrine, repeats a ratio.
        catalogue: Unit stats by type name.

    Returns:
        The same names, verified.

    Raises:
        DoctrineError: ``RW-DOCTRINE-011`` when an entry is unknown to the
            catalogue or names a structure.
    """
    for name in heavies:
        stats = catalogue.get(name)
        if stats is None:
            raise DoctrineError(
                _UNKNOWN_HEAVY,
                f"heavies entry {name!r} is not in the catalogue",
            )
        if stats["speed"] <= 0.0:
            raise DoctrineError(
                _UNKNOWN_HEAVY,
                f"heavies entry {name!r} is a structure; no producer's queue can make it",
            )
    return tuple(heavies)


def expansion_reserve(
    reinforce: Sequence[str],
    catalogue: Mapping[str, UnitStats],
) -> int:
    """Return the credits to hold back for the army before claiming a pool.

    The most expensive thing the bot keeps making, so expansion never leaves it
    unable to replace a single loss. Deliberately shallow: an extractor pays
    back over the rest of the match and a bank does not, and the run that banked
    21,164 credits while its army was ground down is what the shallow end of
    this trade-off guards against ([[policy-economy]]).

    **The maximum, and it was replaced by the mean and put back.** The objection
    to the maximum is real: one expensive type raises the barrier the whole
    economy must clear, invisibly, because nothing about a unit list looks like
    a reserve. Adding a 1,400-credit ``mechArtillery`` took it from 450 to 1,400
    and expansion was then refused 232 times of 237. So it was changed to the
    mean over the composition, which moved the standard mix 450 -> 375.

    Twelve seeds at Very Hard say that was a **regression: 7 wins became 3**,
    with the same two losses and routs falling 3 to 1 -- and unlike most arms
    this session that gap sits outside the noise floor
    ([[policy-holding-ground]]). A shallower reserve starves the replacement it
    exists to fund, and at 1.8x AI income the army cannot absorb that. The
    barrier the objection complained about was doing real work.

    **The confound the objection identified is real and is fixed elsewhere.**
    Deriving the reserve from the composition means a composition A/B is also a
    reserve A/B, so ``reserve`` is now overridable per run -- see :func:`main`.
    That separates the two questions without lowering the figure that measured
    better.

    Args:
        reinforce: Type names the bot keeps making.
        catalogue: Unit stats by type name, for prices.

    Returns:
        Credits to leave unspent. Zero when there is nothing to reinforce,
        because then there is nothing to protect and every spare credit belongs
        to the economy.
    """
    return max((catalogue[name]["price"] for name in reinforce), default=0)


def load_doctrine(path: Path) -> Doctrine:
    """Read a gameplay style from a doctrine file.

    Args:
        path: The preset, e.g. ``doctrines/default.doctrine``.

    Returns:
        The doctrine it describes.

    Raises:
        OSError: When the file cannot be read.
        DoctrineError: When a line is malformed.
        DecodeError: When a field is absent or out of range.
    """
    return parse_doctrine_lines(path.read_text(encoding="utf-8", errors="strict").splitlines())


def main(argv: Sequence[str] | None = None) -> int:
    """Connect, play the doctrine, and report.

    Args:
        argv: ``<port> <catalogue-path> <placement-path> [max-samples]
            [doctrine-path] [trace-path]``. The doctrine is the whole of the
            gameplay style -- goals, worker ceiling, wave mass, reserve, the
            expansion switch and the counter switch -- so one arm of an
            experiment differs from another by a file rather than by an edit,
            and the positional tail this entry point used to grow one slot per
            question stops growing ([[policy-loop]]). ``-`` or absent plays the
            default doctrine; ``-`` or absent for the trace keeps none.
            ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``EXIT_OK`` when the plan completed, ``EXIT_INCOMPLETE`` when it did
        not, ``EXIT_BAD_USAGE`` on a bad argument count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4, 5, 6):
        sys.stdout.write(
            "usage: play <port> <catalogue-path> <placement-path> "
            "[max-samples] [doctrine-path] [trace-path]\n"
        )
        return EXIT_BAD_USAGE
    max_samples = int(args[3]) if len(args) >= 4 else DEFAULT_MAX_SAMPLES
    doctrine = (
        load_doctrine(Path(args[4])) if len(args) >= 5 and args[4] != "-" else DEFAULT_DOCTRINE
    )
    # Where to write the per-sample record. Absent means keep none, because a
    # run that is not being compared against another has nothing to read it
    # for ([[policy-trace]]).
    trace = Path(args[5]) if len(args) >= 6 and args[5] != "-" else None

    catalogue = load_catalogue(Path(args[1]))
    placements = load_placements(Path(args[2]))
    profiles = load_combat_profiles(Path(args[2]))
    tree = load_build_tree(Path(args[2]))

    # Expansion needs to know what the player already has, so it runs against a
    # real observation rather than an assumed opening roster.
    #
    # **Settled by content, not by clock.** The world used to settle on 22
    # seconds of free-running wall time before the planner attached, and runs
    # began from worlds that already differed ([[policy-determinism]]). A match
    # world is now held at its first frame, so the planner may arrive before
    # the starting units have spawned -- the roster is the thing being waited
    # for, so the roster is the condition, and every acked sample advances the
    # simulation by the same locked interval on every run. Each sample is
    # acknowledged like any other: in lockstep the agent holds the simulation
    # until the ack arrives.
    channel = open_channel(int(args[0]))
    opening = channel.next_sample()
    channel.send_ack()
    settled = 0
    while (
        not any(e["mine"] and e["complete"] for e in opening["entities"])
        and settled < OPENING_SETTLE_SAMPLES
    ):
        opening = channel.next_sample()
        channel.send_ack()
        settled += 1
    owned = [e["type_name"] for e in opening["entities"] if e["mine"] and e["complete"]]
    goals = doctrine["goals"]
    plan = expand(goals, tree, owned, catalogue)

    sys.stdout.write(f"doctrine: {doctrine['name']}\n")
    sys.stdout.write(f"goals: {' -> '.join(goals)}\n")
    sys.stdout.write(f"plan:  {' -> '.join(plan)}\n")
    for name in plan:
        site = "on a resource pool" if placements[name]["needs_pool"] else "on the ring"
        sys.stdout.write(f"  {name} costs {catalogue[name]['price']}, goes {site}\n")
    # The whole bill against the opening balance, so a plan priced beyond the
    # start is line three of every log instead of a forensic discovery -- the
    # amphib arm's 11,000-credit prerequisite sat invisible in per-entry costs
    # for twelve matches (log: 2026-07-29). Income closes the gap over time;
    # the savings clock is what judges whether it actually is
    # (:mod:`rw_bot.policy.runner`).
    total = sum(catalogue[name]["price"] for name in plan)
    sys.stdout.write(f"plan total: {total} credits, holding {opening['credits']}\n")

    # Heavies join the composition after the goals: the ratio counts them
    # from the start, and production leaves them alone until the unlock
    # makes the engine offer them.
    reinforce = (
        *reinforcements(goals, catalogue),
        *heavy_reinforcements(doctrine["heavies"], catalogue),
    )
    # The derive-or-fix choice the reserve override existed for, now carried by
    # the doctrine: a fixed figure keeps a composition A/B from silently also
    # being a reserve A/B ([[policy-economy]]).
    reserve = (
        expansion_reserve(reinforce, catalogue)
        if doctrine["reserve"] == DERIVE_RESERVE
        else doctrine["reserve"]
    )
    report = play(
        channel,
        plan,
        catalogue,
        placements,
        profiles,
        max_samples,
        reinforce=reinforce,
        reserve=reserve,
        expand=doctrine["expand"],
        max_workers=doctrine["max_workers"],
        counter=doctrine["counter"],
        cover=doctrine["cover"],
        intercept=doctrine["intercept"],
        guard_cap=doctrine["guard_cap"],
        aa_cover=doctrine["aa_cover"],
        forward=doctrine["forward"],
        scout=doctrine["scout"],
        raid=doctrine["raid"],
        rush=doctrine["rush"],
        creep=doctrine["creep"],
        riposte=doctrine["riposte"],
        tech=doctrine["tech"],
        ladder=ladder_to(doctrine["mass"]),
        trace=trace,
    )
    for line in format_report(report):
        sys.stdout.write(f"{line}\n")
    channel.close()

    return EXIT_OK if report["build_outcome"] == "done" else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main(None))
