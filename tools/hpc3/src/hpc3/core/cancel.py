"""Cancelling jobs, including the ones that were already gone.

``scancel`` is idempotent and quiet: cancelling a job that has already
finished is not an error and produces no output. That is convenient and it is
also how a caller ends up believing it stopped something it did not, so this
module reads the state back afterwards and reports what each job actually
became.

Cancellation is never implicit. Nothing else in this package cancels anything
-- not a sweep whose member failed, not a budget overrun -- because stopping
running work destroys it, and that decision belongs to the operator rather
than to an error path.
"""

from __future__ import annotations

from collections.abc import Sequence

from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.status import JobState
from hpc3.core import remote
from hpc3.core.status import parse_sacct_output, sacct_command


class CancelOutcome:
    """What one job became after being asked to stop.

    Attributes:
        job_id: The job asked about.
        state: The state accounting reports for it now. A job cancelled while
            running reports ``CANCELLED``; one that had already finished
            reports whatever it finished as, which is not a failure of the
            cancel.
        was_running: Whether the cancel actually stopped work, as opposed to
            arriving after the job was already over.
    """

    __slots__ = ("job_id", "state", "was_running")

    def __init__(self, job_id: str, state: JobState, *, was_running: bool) -> None:
        """Record one outcome.

        Args:
            job_id: The job asked about.
            state: State accounting reports after the cancel.
            was_running: Whether the cancel stopped live work.
        """
        self.job_id = job_id
        self.state = state
        self.was_running = was_running


_LIVE_BEFORE_CANCEL: frozenset[JobState] = frozenset(
    {"PENDING", "RUNNING", "SUSPENDED", "COMPLETING", "REQUEUED"}
)


def cancel(host: str, job_ids: Sequence[str], cluster: ClusterFacts) -> list[CancelOutcome]:
    """Cancel jobs and report what each one became.

    The state is read BEFORE the cancel as well as after, because ``scancel``
    cannot distinguish "stopped your running job" from "did nothing to a job
    that finished an hour ago" and the difference is the entire question a
    caller is asking.

    Args:
        host: SSH destination.
        job_ids: Slurm job ids to cancel. Never empty.

    Returns:
        One outcome per job that accounting knows, in the order accounting
        reports them.

    Raises:
        ValueError: If no job id is given. A bare ``scancel`` with no id
            cancels every job the user has, which is never what an empty
            list meant.
        AppError: If a remote command fails, or accounting output is
            malformed.
    """
    if len(job_ids) == 0:
        raise ValueError("cancel requires at least one job id; a bare scancel takes everything")

    before = {
        status["job_id"]: status["state"]
        for status in parse_sacct_output(remote.run_remote(host, sacct_command(job_ids)), cluster)
    }
    remote.run_remote(host, f"scancel {' '.join(job_ids)}")
    after = parse_sacct_output(remote.run_remote(host, sacct_command(job_ids)), cluster)

    return [
        CancelOutcome(
            status["job_id"],
            status["state"],
            was_running=before.get(status["job_id"], "COMPLETED") in _LIVE_BEFORE_CANCEL,
        )
        for status in after
    ]


def summarise(outcomes: Sequence[CancelOutcome]) -> tuple[int, int]:
    """Count how many cancels actually stopped work.

    Args:
        outcomes: Outcomes to count.

    Returns:
        The number that stopped live work, and the number that arrived after
        the job was already over.
    """
    stopped = sum(1 for outcome in outcomes if outcome.was_running)
    return stopped, len(outcomes) - stopped


__all__ = ["CancelOutcome", "cancel", "summarise"]
