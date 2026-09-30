"""The real HPC3 scheduler admits a committed run, and nothing is queued.

Board task 465689f5. Every other preflight case scripts the cluster; this one
asks it. ``hpc3-preflight`` is the submission path with the submit left off:
it opens the ssh session through the jump host, probes the run's environment
inside the project's image, uploads the REAL rendered script and runs
``sbatch --test-only`` on it by path, then parses the scheduler's own verdict.
No job id is created and no allocation is made, so the case can run on every
execution pass without spending anything on a shared cluster.

The run is ``runs/execution-preflight.json``: ``python -c pass`` in the
``tankpit`` project, whose workspace ``runs/hpc3-tankpit.json`` declares the
free CPU partition and an image. It is a committed document, so
``test_committed_runs`` also resolves it on every ordinary run, where this
case itself is skipped.
"""

from __future__ import annotations

import pathlib
import re
from typing import Final

import pytest

from hpc3.cli.preflight import main

#: The committed documents the case preflights.
RUNS: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent.parent / "runs"

#: The verdict line, as the CLI renders the scheduler's answer.
VERDICT: Final[re.Pattern[str]] = re.compile(
    r"^OK tankpit\.execution-preflight: would start \S+ on \S+ "
    r"\(\d+ cpu, free\)$"
)


@pytest.mark.host_hpc3
def test_the_real_scheduler_admits_the_committed_run(emitted: list[str]) -> None:
    code = main(
        [
            "--config",
            str(RUNS / "hpc3-tankpit.json"),
            "--run",
            str(RUNS / "execution-preflight.json"),
        ]
    )

    assert code == 0
    # The scheduler's verdict, then the three lines the CLI always closes a
    # preflight with: the admitted count, the GPU projection against the
    # workspace's cap (0.00 for this CPU-only run), and the caveat that a
    # start estimate is a queue snapshot. The first real run on 2026-09-30
    # printed all four; the case had been written expecting two.
    assert [bool(VERDICT.fullmatch(line)) for line in emitted] == [True, False, False, False]
    assert emitted[1:] == [
        "1 spec(s) would be admitted; nothing was queued",
        "projected 0.00 GPU-hours against a declared cap of 0.00; "
        "spend is measured after the fact, not projected",
        "start estimates are a snapshot of the queue, not a reservation",
    ]
