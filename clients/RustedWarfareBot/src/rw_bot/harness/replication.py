"""Whether two runs of one configuration were the same simulation.

The verdict a replication panel produces, as the pure comparison it is. Two
members of a pair differ in nothing but their label, so their per-sample world
digests must agree frame for frame. Where they stop agreeing is the whole
finding: divergence is not drift, it is a rare consequential draw landing one
unit over, and the frame it happens at is what names the leak.

WHY THE WORLD COLUMN AND NOT THE SCORECARD. A scorecard is an endpoint, and
two runs can reach the same verdict by different routes -- the determinism
campaign's forked pairs frequently ended alike. The trace's world digest is
computed per sample from every visible entity's identity and position, so two
traces agreeing on it agreed about the whole board at every step.

WHAT A FORK MEANS HERE, AND WHAT IT DOES NOT. Bit-exact replication across
separate invocations was certified on 2026-08-07, but every run of that
certification was on a Windows workstation under the depot's Java 13. The
cluster ships Java 8. A fork found by this panel therefore says the regime
does not hold under that runtime; it does NOT say the fix was wrong, and it
does not invalidate a workstation batch. What it does is stop a cluster sweep
from being written up as though the regime had been checked.
"""

from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import TypedDict

from rw_bot import RwBotError

#: Column of a trace line holding the world digest, and how many columns a
#: line must have before that column means anything.
#:
#: Read off the real header, which is ``frame army credits enemies extractors
#: lost producers idle orders refused worth rival income rival_income world
#: plan workers``. The income pair was inserted before the digest rather than
#: appended, so this index moved once and will move again -- which is why the
#: header is CHECKED rather than assumed.
WORLD_COLUMN = 14
TRACE_COLUMNS = 17

#: What the header calls the columns this reads, so a trace whose shape has
#: moved is refused instead of silently compared on the wrong figure.
FRAME_HEADING = "frame"
WORLD_HEADING = "world"

_BAD_HEADER = "RW-REPLICATE-001"
_NO_SAMPLES = "RW-REPLICATE-002"


class ReplicationError(RwBotError):
    """A trace could not be read as one.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what was wrong.
    """


class PairVerdict(TypedDict):
    """What one seed's two runs did relative to each other.

    Attributes:
        seed: The seed both members played.
        samples: How many samples the two have in common. A pair that agrees
            over three samples has said almost nothing, so the count is
            carried beside the verdict rather than folded into it.
        identical: Whether every compared sample agreed.
        forked_at: The frame of the first disagreement, or -1 when there was
            none. The frame rather than the sample index, because that is what
            the determinism campaign's findings are written in.
        left_samples: How many samples the first member recorded.
        right_samples: How many the second did. Carried separately because two
            runs of different lengths are a finding in themselves -- one was
            cut short -- and comparing only the overlap would hide it.
    """

    seed: int
    samples: int
    identical: bool
    forked_at: int
    left_samples: int
    right_samples: int


#: What :attr:`PairVerdict.forked_at` holds when the pair never diverged.
NO_FORK = -1


def world_digests(lines: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Read a trace's per-sample world digests.

    Args:
        lines: The trace's lines, header included.

    Returns:
        One ``(frame, digest)`` pair per sample line, in file order.

    Raises:
        ReplicationError: ``RW-REPLICATE-001`` when the header does not name
            the frame and world columns where this expects them. The income
            pair was once inserted BEFORE the digest rather than appended, so
            the index has moved before; comparing the wrong column would
            report two runs as identical because they agreed about something
            else.
        ReplicationError: ``RW-REPLICATE-002`` when there are no sample lines.
            An interrupted match leaves a header and nothing else, and two of
            those would otherwise compare equal.
    """
    if not lines:
        raise ReplicationError(_BAD_HEADER, "a trace with no lines carries no header to check")
    headings = lines[0].split()
    if len(headings) < TRACE_COLUMNS:
        raise ReplicationError(
            _BAD_HEADER,
            f"a trace header carries {len(headings)} columns, expected at least "
            f"{TRACE_COLUMNS}: {lines[0]!r}",
        )
    if headings[0] != FRAME_HEADING or headings[WORLD_COLUMN] != WORLD_HEADING:
        raise ReplicationError(
            _BAD_HEADER,
            f"a trace header must name {FRAME_HEADING!r} first and {WORLD_HEADING!r} at "
            f"column {WORLD_COLUMN}; got {headings[0]!r} and {headings[WORLD_COLUMN]!r}. "
            "The column order has moved before, and comparing the wrong one reports two "
            "runs as identical because they agreed about something else.",
        )

    samples: list[tuple[int, int]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= TRACE_COLUMNS and parts[0].isdigit():
            samples.append((int(parts[0]), int(parts[WORLD_COLUMN])))
    if samples == []:
        raise ReplicationError(
            _NO_SAMPLES,
            "a trace with no sample lines records nothing; two interrupted matches "
            "would otherwise compare equal to each other",
        )
    return tuple(samples)


def compare_pair(
    seed: int, left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]
) -> PairVerdict:
    """Say whether one seed's two runs were the same simulation.

    Args:
        seed: The seed both played.
        left: The first member's ``(frame, digest)`` samples.
        right: The second member's.

    Returns:
        The verdict, naming the frame of the first disagreement when there is
        one. A frame present in one and not the other counts as a
        disagreement at that frame: two runs that sampled different frames did
        not run the same simulation, whatever their digests say.
    """
    for index, (here, there) in enumerate(zip(left, right, strict=False)):
        if here != there:
            return PairVerdict(
                seed=seed,
                samples=index,
                identical=False,
                forked_at=here[0],
                left_samples=len(left),
                right_samples=len(right),
            )
    shared = min(len(left), len(right))
    return PairVerdict(
        seed=seed,
        samples=shared,
        identical=len(left) == len(right),
        forked_at=NO_FORK if len(left) == len(right) else left[shared - 1][0],
        left_samples=len(left),
        right_samples=len(right),
    )


def render_verdict(verdict: PairVerdict) -> str:
    """Render one pair's verdict as a line for a person.

    Args:
        verdict: The pair's verdict.

    Returns:
        One line, the seed first so a column of them sorts by seed.
    """
    if verdict["identical"]:
        return f"seed {verdict['seed']:>9}  identical over {verdict['samples']} sample(s)"
    if verdict["left_samples"] != verdict["right_samples"]:
        return (
            f"seed {verdict['seed']:>9}  DIFFERENT LENGTHS "
            f"{verdict['left_samples']} vs {verdict['right_samples']} sample(s), "
            f"agreeing to frame {verdict['forked_at']}"
        )
    return (
        f"seed {verdict['seed']:>9}  FORKED at frame {verdict['forked_at']} "
        f"after {verdict['samples']} sample(s)"
    )


def panel_holds(verdicts: Sequence[PairVerdict]) -> bool:
    """Report whether every pair in a panel replicated.

    Args:
        verdicts: One verdict per seed.

    Returns:
        True when every pair was identical. An EMPTY panel is False, not True:
        a run that compared nothing has not certified anything, and reporting
        it as a pass is how a regime goes unchecked. Tested on the LENGTH
        rather than against ``[]``, because a tuple of no verdicts is not
        equal to an empty list and would have passed.
    """
    return len(verdicts) > 0 and all(verdict["identical"] for verdict in verdicts)


__all__ = [
    "FRAME_HEADING",
    "NO_FORK",
    "TRACE_COLUMNS",
    "WORLD_COLUMN",
    "WORLD_HEADING",
    "PairVerdict",
    "ReplicationError",
    "compare_pair",
    "panel_holds",
    "render_verdict",
    "world_digests",
]
