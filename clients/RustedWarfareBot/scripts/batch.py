"""One committed sweep file, played by the cluster, scorecards pulled.

Usage:
    python -m scripts.batch <hpc3:workspace.json> <batch> <sweep-file> <difficulty>

The bespoke-batch chain -- freeze, stage, extract, document, converge
until full, pull -- was five manual commands repeated for every
factorial and transfer panel (six times on 2026-09-02/03 alone), each
repetition an opportunity to skip the idempotency pass or fat-finger a
payload name. :class:`~rw_bot.harness.cluster_round.ClusterRound`
already IS that chain; this module only hands it a committed sweep
file's lines instead of generated ones. Judgement stays separate on
purpose: a factorial's questions (`scripts.pairs` per arm-pair,
`scripts.margin`) are the experiment's own, not the transport's.

Kill-resumable like every driver: the sweep file is the input, not a
generated artifact, so a relaunch replays identically and the campaign
converge fast-forwards whatever already ran.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from rw_bot.harness.cluster_round import ClusterRound
from scripts.search import (
    CLUSTER_HOST,
    CLUSTER_PREFIX,
    CLUSTER_ROOT,
    CLUSTER_SCRATCH,
    FAST_FORWARD,
    MAP_PATH,
    POLL_SECONDS,
    SWEEP_ROOT,
)

EXIT_OK = 0
EXIT_BAD_USAGE = 2


def main(argv: Sequence[str] | None = None, sweeps_root: Path = SWEEP_ROOT) -> int:
    """Play one committed sweep file to completion on the cluster.

    Args:
        argv: ``<hpc3:workspace.json> <batch> <sweep-file> <difficulty>``.
            ``None`` reads the process arguments.
        sweeps_root: Where scorecards land, injectable for tests.

    Returns:
        ``EXIT_OK``, or ``EXIT_BAD_USAGE`` on a bad argument count, a
        non-cluster destination, or a sweep file that does not exist --
        the file IS the experiment's record, and a typo'd path must not
        become an empty batch.

    Raises:
        ClusterRoundError: Through the runner, on a round that cannot
            deliver its scorecards.
        SweepError: Through the runner's job-file write, on lines the
            job parser refuses.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 4 or not args[0].startswith(CLUSTER_PREFIX):
        sys.stdout.write("usage: batch <hpc3:workspace.json> <batch> <sweep-file> <difficulty>\n")
        return EXIT_BAD_USAGE
    sweep_file = Path(args[2])
    if not sweep_file.is_file():
        sys.stdout.write(f"usage: sweep file does not exist: {sweep_file.as_posix()}\n")
        return EXIT_BAD_USAGE
    if sweep_file.stem != args[1]:
        # The runner writes <jobs_dir>/<batch>.txt; a mismatched batch
        # name would clone the sweep under a second filename and every
        # seed in it would count twice in the disjointness scans.
        sys.stdout.write(
            f"usage: batch name {args[1]!r} must match the sweep file's stem {sweep_file.stem!r}\n"
        )
        return EXIT_BAD_USAGE
    runner = ClusterRound(
        config=args[0][len(CLUSTER_PREFIX) :],
        host=CLUSTER_HOST,
        cluster_root=CLUSTER_ROOT,
        map_path=MAP_PATH,
        difficulty=int(args[3]),
        fast_forward=FAST_FORWARD,
        scratch=CLUSTER_SCRATCH,
        sweeps_root=sweeps_root,
        jobs_dir=sweep_file.parent,
        poll_seconds=POLL_SECONDS,
    )
    runner.run(args[1], tuple(sweep_file.read_text(encoding="utf-8").splitlines()))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
