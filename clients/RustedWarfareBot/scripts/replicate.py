"""Read a replication panel's verdict: did the same seed replay?

The panel is ``sweeps/replicate.txt`` -- twelve seeds, each played twice under
two labels. This reads the two traces of every pair and compares them sample
by sample on the world digest.

WHAT A PASS CERTIFIES. That the determinism regime holds under whatever
runtime produced these traces. That matters because the regime was certified
on 2026-08-07 entirely on a Windows workstation under the depot's Java 13, and
the cluster ships the Linux depot's Java 8. Until this passes there, a sweep's
numbers are from a regime nobody checked.

WHAT A FAIL SAYS, AND WHAT IT DOES NOT. It says the regime does not hold under
that runtime. It does not say the fix was wrong, and it does not touch any
workstation batch. The frame it names is where to look: divergence is a rare
consequential draw landing one unit over, and the campaign's instrument for
that is the draw tap (``PLAY_RNGTAP=1``).

Exits non-zero when any pair forked, so it can gate what runs after it.

Run as ``python -m scripts.replicate <batch> [traces-root]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rw_bot.harness import _test_hooks
from rw_bot.harness.replication import (
    PairVerdict,
    compare_pair,
    panel_holds,
    render_verdict,
    world_digests,
)
from rw_bot.harness.results_layout import TRACE_ROOT, TRACE_SUFFIX

#: The two labels a pair is filed under. A batch files by label, so one
#: configuration played twice needs two of them.
LEFT_LABEL = "repA"
RIGHT_LABEL = "repB"

#: Every pair replicated.
EXIT_OK = 0

#: At least one pair forked, or a trace was missing.
EXIT_FORKED = 1

EXIT_BAD_USAGE = 2

_USAGE = "usage: replicate <batch> [traces-root]"


def seeds_in(names: Sequence[str]) -> tuple[int, ...]:
    """Return the seeds a batch has both members of, in order.

    Takes the filenames alone. It was given the root and the batch too at
    first, and used neither -- a parameter a function does not read is a claim
    about what it depends on that is not true.

    Args:
        names: Trace filenames found in the batch's trace directory.

    Returns:
        Every seed with a trace under BOTH labels, ascending. A seed with only
        one is deliberately absent rather than compared against nothing --
        it is reported by the caller as missing, because a member that never
        ran is a gap in the panel and not a pass.
    """
    left = {_seed_of(name, LEFT_LABEL) for name in names}
    right = {_seed_of(name, RIGHT_LABEL) for name in names}
    return tuple(sorted(seed for seed in left & right if seed is not None))


def _seed_of(name: str, label: str) -> int | None:
    """Return the seed a trace filename carries, for one label.

    Args:
        name: A trace filename, e.g. ``repA-s12345.ndjson``.
        label: The label to match.

    Returns:
        The seed, or None when the name is not that label's trace.
    """
    stem = f"{label}-s"
    if not name.startswith(stem) or not name.endswith(TRACE_SUFFIX):
        return None
    digits = name[len(stem) : -len(TRACE_SUFFIX)]
    if not digits.isdigit():
        return None
    return int(digits)


def verdicts_for(traces: Path, batch: str, seeds: Sequence[int]) -> list[PairVerdict]:
    """Compare both members of every seed in a panel.

    Args:
        traces: The trace root.
        batch: The batch's name.
        seeds: The seeds to compare.

    Returns:
        One verdict per seed, in the order given.

    Raises:
        ReplicationError: ``RW-REPLICATE-001`` when a trace's header does not
            name the columns this reads, ``RW-REPLICATE-002`` when it records
            no samples.
        OSError: When a trace cannot be read.
    """
    return [
        compare_pair(
            seed,
            world_digests(_read(traces, batch, LEFT_LABEL, seed)),
            world_digests(_read(traces, batch, RIGHT_LABEL, seed)),
        )
        for seed in seeds
    ]


def _read(traces: Path, batch: str, label: str, seed: int) -> tuple[str, ...]:
    """Read one member's trace.

    Args:
        traces: The trace root.
        batch: The batch's name.
        label: Which member.
        seed: Which seed.

    Returns:
        The trace's lines.

    Raises:
        OSError: When the trace cannot be read.
    """
    return tuple(_test_hooks.read_text_lines(traces / batch / f"{label}-s{seed}{TRACE_SUFFIX}"))


def main(argv: Sequence[str] | None = None) -> int:
    """Report whether every pair of a replication panel replicated.

    Args:
        argv: ``<batch> [traces-root]``. ``None`` reads the process arguments.

    Returns:
        :data:`EXIT_OK` when every pair was identical, :data:`EXIT_FORKED`
        when any forked or the panel was empty, :data:`EXIT_BAD_USAGE` on a
        bad argument count.

    Raises:
        ReplicationError: When a trace cannot be read as one.
        OSError: When the trace directory cannot be listed.
    """
    args = list(argv) if argv is not None else _test_hooks.read_argv()
    if len(args) not in (1, 2):
        _test_hooks.write_line(_USAGE)
        return EXIT_BAD_USAGE

    batch = args[0]
    traces = Path(args[1] if len(args) == 2 else TRACE_ROOT)
    seeds = seeds_in(_test_hooks.list_names(traces / batch))
    verdicts = verdicts_for(traces, batch, seeds)

    for verdict in verdicts:
        _test_hooks.write_line(f"[replicate] {render_verdict(verdict)}")

    forked = [verdict for verdict in verdicts if not verdict["identical"]]
    _test_hooks.write_line(
        f"[replicate] {len(verdicts) - len(forked)}/{len(verdicts)} pair(s) replicated"
    )
    if not panel_holds(verdicts):
        # Named rather than left to the reader: an empty panel and a forked
        # one both exit non-zero, and only one of them is a determinism
        # finding. Reporting them alike is how "we ran it and it failed"
        # becomes "the regime does not hold".
        _test_hooks.write_line(
            "[replicate] the regime is NOT certified under this runtime"
            if forked
            else "[replicate] no pair was compared: the panel certified nothing"
        )
        return EXIT_FORKED

    _test_hooks.write_line("[replicate] the regime holds: same seed, same match")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))


__all__ = [
    "EXIT_BAD_USAGE",
    "EXIT_FORKED",
    "EXIT_OK",
    "LEFT_LABEL",
    "RIGHT_LABEL",
    "main",
    "seeds_in",
    "verdicts_for",
]
