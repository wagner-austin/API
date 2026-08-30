"""What a sweep's numbers were produced under, and the record that carries them.

WHY THIS PROJECT NEEDED IT LAST AND NEEDS IT MOST. Every verdict here is a
comparison -- this arm's win rate against that one's, the champion's rate this
batch against five batches ago -- and until now none of those numbers carried
anything saying whether they were produced under the same conditions. The
project already knew that mattered: the wiki pins ``game_version`` on every
page "because the jar is obfuscated and class names change silently between
releases". It just had no way to say it about a RESULT.

THE GAME IS THE SUBJECT, SO THE GAME IS THE AXIS. A boosting benchmark records
lightgbm's version because a bump in lightgbm moves the comparison; this
project's equivalent is the game it played, which moves everything and is not
a Python distribution at all. The packages axis is exactly what that is for --
the things whose behaviour decides the numbers.

THREE OF THEM, NOT ONE, and the two that were missing were missing for the
same reason: a jar is one file and the others are trees, so recording them
needed :mod:`rw_bot.tree_identity` before it could be done at all.

* :data:`GAME_DISTRIBUTION` -- the engine's code, ``game-lib.jar``.
* :data:`JVM_DISTRIBUTION` -- the runtime that executes it. The two platforms
  ship DIFFERENT MAJOR VERSIONS, Java 8 on Linux and Java 13 on Windows, so
  this is not a formality. Today the host axis separates those two by
  accident, because the operating systems differ; two Linux runs either side
  of a depot that bumped its bundled JRE would fingerprint identically and be
  silently incomparable.
* :data:`ASSETS_DISTRIBUTION` -- the maps, mods and unit definitions the
  simulation reads. A jar digest does not cover them, and this project has
  already lost a batch family to exactly that gap: a map missing from a clone
  sent the engine to its boot sandbox, and every scorecard was void with the
  jar digest matching throughout.

Each recorded as a DIGEST rather than a version string, for the reason the jar
was first: the string is a label somebody maintains and the digest is read off
the bytes that ran. The runtime carries both -- its vendor-shipped
``JAVA_VERSION`` in front of its digest -- because unlike the wiki's game
version that label is not one this project can get wrong, and a reader asking
why two batches differ is better served by ``1.8.0_131`` than by two hex
strings that merely differ.

WHAT IT DOES NOT CLAIM. There is no GPU and no image, so both axes are the
stated-absent :data:`~platform_core.comparability.NO_VALUE` rather than
omitted. The determinism axis records the harness's own posture; a match is
seeded, and the seed belongs to the label rather than here because two seeds
of one arm are two runs of the same configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

from platform_core.comparability import NO_VALUE, RunFingerprint, cpu_run_fingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.environment_record import (
    HostProbe,
    PackageVersion,
    capture_host_record,
    package_versions,
)
from platform_core.run_record import Observation, RunRecord, run_record

from rw_bot.harness.jvm import JVM_RELEASE_FILE, jvm_dir, release_version
from rw_bot.harness.scorecards import MatchRow
from rw_bot.tree_identity import digest_tree

#: Names the three parts of the game are recorded under in the packages axis.
#: Distinct rather than one compound value, because the axis compares
#: package-by-package: a batch played on new maps and an unchanged engine
#: should report which of the two moved, not merely that something did.
GAME_DISTRIBUTION = "rusted-warfare"
JVM_DISTRIBUTION = "rusted-warfare-jvm"
ASSETS_DISTRIBUTION = "rusted-warfare-assets"

#: What the game's own code is called inside a game directory. Named apart
#: from any one path so a caller holding a different game directory -- a
#: worker's clone, a staged tree -- composes the jar without writing the
#: filename a second time.
GAME_JAR_NAME = "game-lib.jar"

#: Where the maps, mods, unit definitions and translations live inside a game
#: directory. The simulation's INPUT, as against the jar's code.
GAME_ASSETS_DIR = "assets"

#: How many hex characters of a digest are recorded. Full length would
#: dominate every rendering of the axis; sixteen is far past the point where
#: two real builds collide.
DIGEST_LENGTH = 16

#: What separates a runtime's stated version from its digest, e.g.
#: ``1.8.0_131+9f2c...``. The spelling PEP 440 gives a local version, which is
#: how ``torch==2.6.0+cu124`` reads in this monorepo's own pinned images.
VERSION_DIGEST_SEPARATOR = "+"


def game_build(jar: Path) -> PackageVersion:
    """Identify the game binary a run played against.

    Args:
        jar: The game jar.

    Returns:
        The build, named :data:`GAME_DISTRIBUTION` and versioned by the first
        :data:`DIGEST_LENGTH` hex characters of the jar's SHA-256.

    Raises:
        FileNotFoundError: When the jar is absent. Propagated rather than
            recorded as unknown, for the reason the whole module exists: a
            record that says "some build" about an obfuscated binary says
            nothing, and this project's claims are only valid for one build.
    """
    return PackageVersion(
        name=GAME_DISTRIBUTION,
        version=sha256(jar.read_bytes()).hexdigest()[:DIGEST_LENGTH],
    )


def bundled_runtime(game_dir: Path, platform: str) -> PackageVersion:
    """Identify the JVM a run's simulation actually executed on.

    Args:
        game_dir: The game directory the run played from.
        platform: A ``sys.platform`` value, which decides the runtime's
            directory name -- ``jvm64`` on Windows and ``jvm-linux``
            elsewhere. Passed rather than read from the running interpreter so
            a Windows workstation can state what a Linux node's record would
            say.

    Returns:
        The runtime, named :data:`JVM_DISTRIBUTION` and versioned by its own
        ``JAVA_VERSION`` followed by :data:`DIGEST_LENGTH` characters of the
        digest of its whole tree. The version makes the record legible and the
        digest makes it true: a runtime repackaged under an unchanged version
        string is a different runtime and differs here.

    Raises:
        JvmReleaseError: ``RW-JVM-002`` when the runtime states no version.
        TreeIdentityError: ``RW-TREE-001`` when the runtime's directory is
            absent, ``RW-TREE-003`` when it holds a symbolic link.
        OSError: When the release file or a file under the runtime cannot be
            read.
    """
    root = game_dir / jvm_dir(platform)
    version = release_version((root / JVM_RELEASE_FILE).read_text(encoding="utf-8").splitlines())
    digest = digest_tree(root)[:DIGEST_LENGTH]
    return PackageVersion(
        name=JVM_DISTRIBUTION,
        version=f"{version}{VERSION_DIGEST_SEPARATOR}{digest}",
    )


def asset_tree(game_dir: Path) -> PackageVersion:
    """Identify the data the simulation read.

    Args:
        game_dir: The game directory the run played from.

    Returns:
        The asset tree, named :data:`ASSETS_DISTRIBUTION` and versioned by
        :data:`DIGEST_LENGTH` characters of its tree digest. There is no
        version string to put in front of it, because nothing in ``assets/``
        states one -- which is the point: a mod folder edited between two
        batches announces itself nowhere else.

    Raises:
        TreeIdentityError: ``RW-TREE-001`` when the directory is absent,
            ``RW-TREE-002`` when it is empty, ``RW-TREE-003`` on a symlink.
        OSError: When a file under the tree cannot be read.
    """
    return PackageVersion(
        name=ASSETS_DISTRIBUTION,
        version=digest_tree(game_dir / GAME_ASSETS_DIR)[:DIGEST_LENGTH],
    )


def game_packages(game_dir: Path, platform: str) -> tuple[PackageVersion, ...]:
    """Name everything about the game whose behaviour decides the numbers.

    Args:
        game_dir: The game directory the run played from.
        platform: A ``sys.platform`` value, for the runtime's directory name.

    Returns:
        The code, the runtime and the data, in the shared axis's canonical
        order. Ordered by :func:`~platform_core.environment_record.package_versions`
        rather than here, because that is what the record's own decoder will
        put them in: a tuple assembled in any other order does not equal its
        own round trip, which is how this was found. They are READ engine
        outwards, so the first failure a caller meets is the jar's.

    Raises:
        FileNotFoundError: When the jar or the release file is absent.
        JvmReleaseError: ``RW-JVM-002`` when the runtime states no version.
        TreeIdentityError: When a tree is absent, empty or holds a symlink.
        OSError: When a file under either tree cannot be read.
        ValueError: When a digest or a stated version came back empty.
    """
    code = game_build(game_dir / GAME_JAR_NAME)
    runtime = bundled_runtime(game_dir, platform)
    assets = asset_tree(game_dir)
    return package_versions(
        {
            code["name"]: code["version"],
            runtime["name"]: runtime["version"],
            assets["name"]: assets["version"],
        }
    )


def sweep_fingerprint(
    determinism: DeterminismRecord,
    get_env: Callable[[str], str | None],
    probe: HostProbe,
    game_dir: Path,
    platform: str,
) -> RunFingerprint:
    """Describe the configuration a sweep's numbers were produced under.

    Args:
        determinism: What the harness pinned.
        get_env: Reader for a process environment variable, for the image
            digest a launcher would export. This project runs on a
            workstation, so it is normally absent and recorded as such.
        probe: Reader for the machine's own facts.
        game_dir: The game directory the sweep played from. The DIRECTORY, not
            the jar it holds: two of the three package axes are trees beside
            that jar, and a signature taking only the jar could not have
            reached them.
        platform: A ``sys.platform`` value, for the runtime's directory name.

    Returns:
        The fingerprint. The GPU axes are stated absent rather than omitted:
        the game renders headless here and no arm touches a card, and empty
        differs from every real value instead of matching all of them.

    Raises:
        FileNotFoundError: When the jar or the release file is absent.
        JvmReleaseError: ``RW-JVM-002`` when the runtime states no version.
        TreeIdentityError: When a tree is absent, empty or holds a symlink.
        OSError: When a file under either tree cannot be read.
    """
    return cpu_run_fingerprint(
        determinism,
        get_env,
        capture_host_record(probe),
        game_packages(game_dir, platform),
    )


class ArmSummary(TypedDict):
    """One arm's aggregate across the matches it played.

    Attributes:
        arm: Which arm.
        matches: How many of its matches have results.
        wins: Matches whose verdict was ``won``.
        losses: Matches whose verdict was ``defeated`` or ``wiped``. Not
            ``matches - wins``: a match can end in neither, and folding those
            into losses would report a rate this project has never measured.
        drops: Extractors lost between peak and end, summed. The figure every
            verdict here turns on.
        median_worth: Median end-of-match total worth.
        unengageable: Targets seen that could not be engaged, summed.
        intercepts: Interceptions, summed.
    """

    arm: str
    matches: int
    wins: int
    losses: int
    drops: int
    median_worth: int
    unengageable: int
    intercepts: int


def summarize_arm(rows: Sequence[MatchRow], arm: str) -> ArmSummary:
    """Aggregate one arm's matches.

    Lives here rather than in the analyser that prints it, so the numbers a
    verdict rests on can be RECORDED as well as displayed. A figure that
    exists only inside a format string cannot be put in a run record.

    Args:
        rows: Every match row in the batch.
        arm: The arm to aggregate.

    Returns:
        Its aggregate.

    Raises:
        ValueError: When the arm has no matches. The caller derives the arm
            names from the rows, so an empty one means the two disagree, and
            a median over nothing has no answer to return.
    """
    sub = [row for row in rows if row["arm"] == arm]
    if not sub:
        raise ValueError(f"arm {arm!r} has no matches to summarise")
    return ArmSummary(
        arm=arm,
        matches=len(sub),
        wins=sum(1 for row in sub if row["verdict"] == "won"),
        losses=sum(1 for row in sub if row["verdict"] in ("defeated", "wiped")),
        drops=sum(row["dropped"] for row in sub),
        median_worth=sorted(row["worth_end"] for row in sub)[len(sub) // 2],
        unengageable=sum(row["targets_end"] - row["engageable"] for row in sub),
        intercepts=sum(row["intercepted"] for row in sub),
    )


SWEEP_EXPERIMENT = "rusted-warfare-doctrine-sweep"
"""What this line of work IS, for pairing two of its records.

One name across every batch, not one per sweep. The question this project
asks is longitudinal -- "how has the champion's win rate moved across five
batches" -- and two records naming different experiments are not comparable
at all, which would make exactly that question unaskable.
"""


def arm_label(batch: str, arm: str) -> str:
    """Name which run within the experiment an arm's results are.

    Args:
        batch: The sweep the arm belongs to, e.g. ``aggression``.
        arm: The arm within it.

    Returns:
        ``<batch>/<arm>``. The batch is here rather than in the experiment
        name because two batches ARE meant to be compared; the seed is in
        neither, because two seeds of one arm are two samples of one
        configuration rather than two configurations.
    """
    return f"{batch}/{arm}"


def arm_observations(summary: ArmSummary) -> tuple[Observation, ...]:
    """Name the numbers an arm's verdict rests on.

    Args:
        summary: The arm's aggregate.

    Returns:
        One observation per figure, plus the win rate the figures are read
        for. The rate is computed here rather than left to the reader
        because a rate recomputed at each reading is a rate two readers can
        disagree about -- and it is the number the stated goal is written in
        ("100% win rate against the built-in AI").

        Counts are carried alongside it and not replaced by it: three wins
        from three matches and thirty from thirty are both 1.0, and only one
        of them is evidence.
    """
    matches = summary["matches"]
    return (
        Observation(name="matches", value=float(matches)),
        Observation(name="wins", value=float(summary["wins"])),
        Observation(name="losses", value=float(summary["losses"])),
        Observation(name="win_rate", value=summary["wins"] / matches),
        Observation(name="extractor_drops", value=float(summary["drops"])),
        Observation(name="median_worth", value=float(summary["median_worth"])),
        Observation(name="unengageable", value=float(summary["unengageable"])),
        Observation(name="intercepts", value=float(summary["intercepts"])),
    )


def arm_run_record(
    batch: str,
    summary: ArmSummary,
    fingerprint: RunFingerprint,
    payload_digest: str,
) -> RunRecord:
    """Turn one arm's results into the shape every experiment emits.

    Args:
        batch: The sweep the arm belongs to.
        summary: Its aggregate.
        fingerprint: What it played under, from :func:`sweep_fingerprint`.
        payload_digest: Digest of the arm's per-match rows, or
            :data:`~platform_core.run_record.NO_PAYLOAD`.

    Returns:
        The record, observations in canonical order.

    Raises:
        ValueError: When the batch or arm is empty, which would leave the
            record unpairable.
    """
    return run_record(
        experiment=SWEEP_EXPERIMENT,
        label=arm_label(batch, summary["arm"]),
        fingerprint=fingerprint,
        observations=arm_observations(summary),
        payload_digest=payload_digest,
    )


__all__ = [
    "ASSETS_DISTRIBUTION",
    "DIGEST_LENGTH",
    "GAME_ASSETS_DIR",
    "GAME_DISTRIBUTION",
    "GAME_JAR_NAME",
    "JVM_DISTRIBUTION",
    "NO_VALUE",
    "SWEEP_EXPERIMENT",
    "VERSION_DIGEST_SEPARATOR",
    "ArmSummary",
    "arm_label",
    "arm_observations",
    "arm_run_record",
    "asset_tree",
    "bundled_runtime",
    "game_build",
    "game_packages",
    "summarize_arm",
    "sweep_fingerprint",
]
