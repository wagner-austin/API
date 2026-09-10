"""What GitHub says about a sha, asked through ``gh`` and parsed at the edge.

TWO QUESTIONS, ASKED SEPARATELY AND FOR DIFFERENT REASONS. The first --
"what runs exist for this sha" -- is asked about every enrolled push on
every cycle, so it must be one cheap call with no pagination. The second --
"how many jobs did this run have, and which failed" -- is asked only about
runs that are about to be announced, because it is a second call per run and
nothing is waiting on it until there is a verdict to report.

THE JOB CALL IS NOT OPTIONAL DETAIL. It is what makes the announcement a
verdict rather than a word. The ``mcps-codebase`` wiki page
``reading-ci-run-outcomes`` measures the two ways a bare conclusion lies,
and the job count is the only thing that separates either pair:

* ``cancelled`` with jobs ran something before it stopped; ``cancelled`` with
  NO jobs never created one. The page records fourteen hours in one
  repository where every run was cancelled and three of those carried
  ``jobs=0``: no failures reported, no verdict either, and nothing executed.
  The count is the only thing that separates those two situations -- and it
  is the only thing this package says about them, because the API does not
  expose WHY a run was cancelled. See :mod:`ci_wake.announce` on the three
  successive corrections that established that.
* Both repositories narrow their matrix to the changed paths, so a green run
  that executed one workspace and a green run that executed forty-three
  render identically in a run list. Only the count tells them apart.

TERMINALITY AND FAILURE ARE BOTH DECIDED IN THE DRIFT-SAFE DIRECTION, which
is not the same direction for the two of them:

* A run is terminal when its status is exactly ``completed``. Listing the
  non-terminal statuses instead would mean that a status GitHub adds later
  reads as terminal, and this bridge would announce a verdict for a run
  still executing.
* A job counts as FAILED when its conclusion is neither ``success`` nor
  ``skipped``. Listing the failing conclusions instead would mean that a
  conclusion GitHub adds later is silently omitted from the failed list, and
  a post would name fewer broken jobs than there were.

Each rule is the one whose drift is loud rather than quiet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from platform_core.error_codes_tooling import CiWakeErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from ci_wake import _test_hooks

#: How long one ``gh`` invocation may take before it is abandoned.
GH_TIMEOUT_SECONDS: Final = 30

#: The only run status GitHub uses to mean "this run is over".
COMPLETED: Final = "completed"

#: The conclusion that means "stopped", at either the run or the job level.
#:
#: THE SAME WORD MEANS DIFFERENT THINGS AT THE TWO LEVELS, which is the whole
#: reason it is a named constant used in both places. A RUN reading
#: ``cancelled`` says only that SOMETHING in it was stopped: measured in
#: ``wagner-austin/API`` on 2026-09-09, run 34418498808 read ``cancelled`` at
#: the top while exactly 3 of its 42 jobs were cancelled. Reading the
#: run-level word as a fact about every job in it is an aggregate label
#: answering a question it was never asked.
CANCELLED: Final = "cancelled"

#: Job conclusions that are not a defect. Everything else is named in the
#: announcement's failed list -- see this module's docstring on why the rule
#: is written this way round.
_JOB_OK: Final = frozenset({"success", "skipped"})

#: Most jobs one API page returns; also this package's page size.
_PAGE: Final = 100


class WorkflowRun(TypedDict):
    """One GitHub Actions run over one commit.

    Attributes:
        run_id: The run's numeric id -- ``databaseId`` in ``gh run list``'s
            spelling and ``id`` in the REST payload's. The address the job
            query is made under.
        workflow: The workflow's display name, e.g. ``Check``. Carried into
            every announcement because a multi-workflow repository can be
            green on one and red on another for one sha, and calling that
            commit green off either alone is a category error.
        status: ``queued``, ``in_progress``, ``completed`` and whatever else
            GitHub adds. Compared only against :data:`COMPLETED`.
        conclusion: ``success``, ``failure``, ``cancelled``, ``skipped`` and
            others; the empty string while the run has not completed, which
            is how the REST payload's ``null`` arrives here.
        html_url: Where a person opens the run. Included in the post because
            the next thing anybody does with a failure is look at it.
    """

    run_id: int
    workflow: str
    status: str
    conclusion: str
    html_url: str


class JobOutcome(TypedDict):
    """One job that did not succeed, and the step that explains it.

    A JOB-LEVEL CONCLUSION IS ITSELF AN AGGREGATE OVER STEPS, which is the
    defect this type exists to remove. Measured in ``wagner-austin/MCPs`` run
    34459514888 on 2026-09-10: the bridge announced "5 failed" naming five
    packages, and only ONE had a step that concluded failure. Three died
    inside ``setup-workspace`` or ``setup-testdb`` with their check step left
    PENDING, and a fourth was killed mid-run -- none of them executed a test.
    Four sessions read that notice as five broken packages.

    Attributes:
        name: The job's name, which in this workspace is the workspace it
            checks.
        failing_step: The name of the first step that concluded failure, or
            the empty string when NO step did. The empty case is the whole
            point: it distinguishes a package whose tests failed from a job
            that never got far enough to run them. It states only what the
            payload carries and does not name a CAUSE -- "died in setup" is
            an inference, and this bridge has already been corrected three
            times for narrating causes the API does not expose.
    """

    name: str
    failing_step: str


class JobTally(TypedDict):
    """What one run's jobs came to.

    Attributes:
        total: How many jobs the run had, from the payload's own
            ``total_count`` rather than from the length of the returned
            array -- the two differ past one page, and the count is the
            field the eviction test depends on.
        listed: How many jobs this page actually carried. Equal to
            ``total`` in every ordinary run; when it is smaller the
            announcement says so rather than presenting a partial failed
            list as a complete one.
        failed: The jobs that ran and did not succeed or skip, each carrying
            the step that concluded failure -- or the empty string when NO
            step did. See :class:`JobOutcome`.
        cancelled: The names of the jobs that were CANCELLED, kept apart from
            :attr:`failed` because they are a different event and lumping them
            together is the defect this split exists to remove. A cancelled
            job did not fail; it was stopped, usually by a superseding push,
            and its package's changes are the ones that may go unchecked.
    """

    total: int
    listed: int
    failed: tuple[JobOutcome, ...]
    cancelled: tuple[str, ...]


def runs_argv(repo: str, sha: str) -> tuple[str, ...]:
    """Build the command that lists a sha's runs.

    Args:
        repo: ``owner/name``.
        sha: The full commit sha.

    Returns:
        The argument vector. Reproducible by hand: pasting it into a
        terminal answers the same question the bridge asked, which is the
        whole reason this package speaks ``gh`` rather than raw HTTPS.
    """
    return ("gh", "api", f"repos/{repo}/actions/runs?head_sha={sha}&per_page={_PAGE}")


def jobs_argv(repo: str, run_id: int) -> tuple[str, ...]:
    """Build the command that lists one run's jobs.

    Args:
        repo: ``owner/name``.
        run_id: The run's numeric id.

    Returns:
        The argument vector.
    """
    return ("gh", "api", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page={_PAGE}")


def gh_json(argv: Sequence[str]) -> JSONValue:
    """Run one ``gh`` command and parse its stdout as JSON.

    Args:
        argv: The argument vector, ``gh`` first.

    Returns:
        The decoded payload.

    Raises:
        AppError: ``GH_COMMAND_FAILED`` when the process exits non-zero,
            which covers ``gh`` being absent from PATH, being logged out,
            and the API refusing. The three are not split into three codes:
            the CLI reports all of them on stderr in words, and the
            operator's first step is ``gh auth status`` for each.
        InvalidJsonError: When the exit was clean but stdout was not JSON.
            Not folded into the above: a zero exit with unparseable output
            is a different fault from a refusal, and reporting it as one
            would send the reader to the authentication that was working.
        subprocess.TimeoutExpired: When the call outlasts
            :data:`GH_TIMEOUT_SECONDS`. Propagated so the scheduler records
            a failed cycle; a caught timeout would be a cycle that announced
            nothing and said it succeeded.
    """
    completed = _test_hooks.run_process(
        list(argv), capture_output=True, text=True, timeout=GH_TIMEOUT_SECONDS
    )
    if completed.returncode != 0:
        raise AppError(
            code=CiWakeErrorCode.GH_COMMAND_FAILED,
            message=(f"{' '.join(argv)} exited {completed.returncode}: {completed.stderr.strip()}"),
        )
    return load_json_str(completed.stdout)


def decode_runs(payload: JSONValue) -> tuple[WorkflowRun, ...]:
    """Read the runs out of a ``actions/runs`` payload.

    Args:
        payload: The decoded response.

    Returns:
        Every run it listed, in the order GitHub returned them.

    Raises:
        JSONTypeError: If the payload is not the documented shape, or a run
            is missing a field this package reads.
    """
    listing = narrow_json_to_dict(payload)
    return tuple(
        _decode_run(narrow_json_to_dict(entry)) for entry in require_list(listing, "workflow_runs")
    )


def _decode_run(entry: dict[str, JSONValue]) -> WorkflowRun:
    """Decode one run object.

    Args:
        entry: One element of ``workflow_runs``.

    Returns:
        The run.

    Raises:
        JSONTypeError: If a field this package reads is missing or mistyped.
    """
    conclusion = entry.get("conclusion")
    return WorkflowRun(
        run_id=require_int(entry, "id"),
        workflow=require_str(entry, "name"),
        status=require_str(entry, "status"),
        # ``null`` until the run completes, and the empty string is this
        # package's spelling of that. Narrowed here rather than carried as
        # ``str | None`` so nothing downstream has to hold the difference
        # between "no conclusion yet" and "a conclusion that is missing".
        conclusion="" if conclusion is None else require_str(entry, "conclusion"),
        html_url=require_str(entry, "html_url"),
    )


def decode_jobs(payload: JSONValue) -> JobTally:
    """Read one run's job tally out of a ``runs/{id}/jobs`` payload.

    Args:
        payload: The decoded response.

    Returns:
        The tally.

    Raises:
        JSONTypeError: If the payload is not the documented shape, or a job
            is missing a field this package reads.
    """
    listing = narrow_json_to_dict(payload)
    jobs = [narrow_json_to_dict(entry) for entry in require_list(listing, "jobs")]
    failed: list[JobOutcome] = []
    cancelled: list[str] = []
    for job in jobs:
        conclusion = _job_conclusion(job)
        if conclusion in _JOB_OK:
            continue
        name = require_str(job, "name")
        if conclusion == CANCELLED:
            cancelled.append(name)
            continue
        failed.append(JobOutcome(name=name, failing_step=_first_failing_step(job)))
    return JobTally(
        total=require_int(listing, "total_count"),
        listed=len(jobs),
        failed=tuple(failed),
        cancelled=tuple(cancelled),
    )


def _first_failing_step(job: dict[str, JSONValue]) -> str:
    """Name the first step of a job that concluded failure.

    ``steps`` is REQUIRED rather than treated as optional. It is documented as
    always present on a jobs-listing entry, and the loud direction is the
    right one here: an absent array read as "no failing step" would relabel a
    genuine test failure as a job that never ran, which is precisely the
    confusion this whole type exists to end. A missing field fails the cycle
    instead, where somebody sees it.

    Args:
        job: One element of the ``jobs`` array.

    Returns:
        The first failing step's name, or the empty string when no step
        concluded failure -- including when the job carries no steps at all,
        which is a job that stopped before running one.

    Raises:
        JSONTypeError: If ``steps`` is absent or is not a list, or a step is
            missing a field this package reads.
    """
    for entry in require_list(job, "steps"):
        step = narrow_json_to_dict(entry)
        if _job_conclusion(step) == "failure":
            return require_str(step, "name")
    return ""


def _job_conclusion(job: dict[str, JSONValue]) -> str:
    """Read a job's conclusion, spelling ``null`` as the empty string.

    A job in a run that has completed should always carry one. The empty
    string is not in :data:`_JOB_OK`, so a job that somehow carries none is
    NAMED in the failed list rather than passed over -- which is the loud
    direction, and the one this module's docstring argues for.

    Args:
        job: One element of ``jobs``.

    Returns:
        The conclusion, or the empty string.
    """
    conclusion = job.get("conclusion")
    return "" if conclusion is None else require_str(job, "conclusion")


def is_terminal(run: WorkflowRun) -> bool:
    """Report whether a run is over.

    Args:
        run: The run.

    Returns:
        True when its status is exactly :data:`COMPLETED`. See this module's
        docstring on why the test is written this way round.
    """
    return run["status"] == COMPLETED


__all__ = [
    "COMPLETED",
    "GH_TIMEOUT_SECONDS",
    "JobTally",
    "WorkflowRun",
    "decode_jobs",
    "decode_runs",
    "gh_json",
    "is_terminal",
    "jobs_argv",
    "runs_argv",
]
