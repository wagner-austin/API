"""One lock per run, so a node runner settles several runs at once.

WHY PER RUN (MCPs board task 8993c306). A serving runner's watch settled
every ended run under one process-wide lock, so two runs ending together
closed one after the other: on 2026-10-05 lavender-wsl's libs/platform_core
run ended at 08:20:00Z and its settle held the lock to 08:20:13Z, and
tools/fleet-execution-linux's row ebb9009c, whose check ended at 08:20:01Z,
was read only at 08:20:15Z and closed at 08:20:23Z, 22.1 s after its check.
What the lock guards is one run's directory and one run's queue job: a read
must not send its script into a directory a retire is removing (job 848fa8f2
was stranded that way at 07:00Z the same day), and a run must not be settled
twice, by the watch and by the collect pass. Neither is a reason to hold a
second run back, so each run id has a lock of its own, and the record files
the settles share guard themselves (:data:`fleet.core.leases.REWRITING`, one
line per append to the ledger and the feed, and
:data:`fleet.core.venv_sweep.SWEEPING`).
"""

from __future__ import annotations

import threading
from typing import Final


class RunLocks:
    """A reentrant lock per run id, made the first time a run is named.

    A run's lock is kept for the life of the process, a serve of at most
    ``node_serve_seconds``, so two holders of one run always meet one lock.
    """

    def __init__(self) -> None:
        """Start with no run named."""
        self._guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}

    def holding(self, run_id: str) -> threading.RLock:
        """One run's lock, to hold with ``with`` for a block.

        Args:
            run_id: The run.

        Returns:
            Its lock, the same object for every caller naming the run.
        """
        with self._guard:
            return self._locks.setdefault(run_id, threading.RLock())


#: The runs of this process: the serve's watch holds a run's lock across each
#: read of its result and the settle it starts, and the collect pass holds it
#: for each job it settles and each run it stops.
SETTLING: Final = RunLocks()


__all__ = ["SETTLING", "RunLocks"]
