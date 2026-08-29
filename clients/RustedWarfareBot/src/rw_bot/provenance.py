"""What a sweep's numbers were produced under, and the record that carries them.

WHY THIS PROJECT NEEDED IT LAST AND NEEDS IT MOST. Every verdict here is a
comparison -- this arm's win rate against that one's, the champion's rate this
batch against five batches ago -- and until now none of those numbers carried
anything saying whether they were produced under the same conditions. The
project already knew that mattered: the wiki pins ``game_version`` on every
page "because the jar is obfuscated and class names change silently between
releases". It just had no way to say it about a RESULT.

THE JAR IS THE SUBJECT, SO THE JAR IS THE AXIS. A boosting benchmark records
lightgbm's version because a bump in lightgbm moves the comparison; this
project's equivalent is the game build, which moves everything and is not a
Python distribution at all. :data:`GAME_DISTRIBUTION` puts it in the packages
axis, which is exactly what that axis is for -- the things whose behaviour
decides the numbers.

Recorded as a DIGEST OF THE JAR rather than the wiki's version string. The
string is maintained by hand and verified by re-reading; the digest is read
off the bytes that ran. Two builds that a human labelled identically are
distinguishable here, and that is the failure mode obfuscation creates: the
class names change silently, so the label is the last thing to notice.

WHAT IT DOES NOT CLAIM. There is no GPU and no image, so both axes are the
stated-absent :data:`~platform_core.comparability.NO_VALUE` rather than
omitted. The determinism axis records the harness's own posture; a match is
seeded, and the seed belongs to the label rather than here because two seeds
of one arm are two runs of the same configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

from platform_core.comparability import NO_VALUE, RunFingerprint, cpu_run_fingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.environment_record import (
    HostProbe,
    PackageVersion,
    capture_host_record,
)
from platform_core.run_record import Observation, RunRecord, run_record

#: Name the game build is recorded under in the packages axis.
GAME_DISTRIBUTION = "rusted-warfare"

#: The jar every claim about engine internals is pinned to. Relative, because
#: the harness runs from the client root and ``.game/`` is a working copy that
#: is deliberately not in the repository.
GAME_JAR = Path(".game") / "game-lib.jar"

#: How many hex characters of the jar digest are recorded. Full length would
#: dominate every rendering of the axis; sixteen is far past the point where
#: two real builds collide.
DIGEST_LENGTH = 16


def game_build(jar: Path = GAME_JAR) -> PackageVersion:
    """Identify the game binary a run played against.

    Args:
        jar: The game jar. Defaults to :data:`GAME_JAR`.

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


def sweep_fingerprint(
    determinism: DeterminismRecord,
    get_env: Callable[[str], str | None],
    probe: HostProbe,
    jar: Path = GAME_JAR,
) -> RunFingerprint:
    """Describe the configuration a sweep's numbers were produced under.

    Args:
        determinism: What the harness pinned.
        get_env: Reader for a process environment variable, for the image
            digest a launcher would export. This project runs on a
            workstation, so it is normally absent and recorded as such.
        probe: Reader for the machine's own facts.
        jar: The game jar, for the build axis.

    Returns:
        The fingerprint. The GPU axes are stated absent rather than omitted:
        the game renders headless here and no arm touches a card, and empty
        differs from every real value instead of matching all of them.

    Raises:
        FileNotFoundError: When the jar is absent.
    """
    return cpu_run_fingerprint(
        determinism,
        get_env,
        capture_host_record(probe),
        (game_build(jar),),
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


def summarize_arm(rows: list[dict[str, str | int]], arm: str) -> ArmSummary:
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
        drops=sum(int(str(row["dropped"])) for row in sub),
        median_worth=sorted(int(str(row["worth_end"])) for row in sub)[len(sub) // 2],
        unengageable=sum(int(str(row["targets_end"])) - int(str(row["engageable"])) for row in sub),
        intercepts=sum(int(str(row["intercepted"])) for row in sub),
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
    "DIGEST_LENGTH",
    "GAME_DISTRIBUTION",
    "GAME_JAR",
    "NO_VALUE",
    "SWEEP_EXPERIMENT",
    "ArmSummary",
    "arm_label",
    "arm_observations",
    "arm_run_record",
    "game_build",
    "summarize_arm",
    "sweep_fingerprint",
]
