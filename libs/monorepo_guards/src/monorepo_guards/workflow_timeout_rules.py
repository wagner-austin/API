"""Guard rule: every CI job is bounded in time.

THE INCIDENT. A GitHub Actions job with no ``timeout-minutes`` inherits a
360-minute default. On 2026-09-09 a WSL VM on the shared runner host
ballooned until the host starved it, the VM stopped being scheduled, and
every timer inside it froze. CI held for about three hours while each job
still reported ``in_progress``: nothing failed, nothing alerted, and the
queue behind it did not move.

AN IN-VM WATCHDOG CANNOT SUBSTITUTE, which is the whole reason this lives
here. A frozen VM freezes its own watchdog. The bound has to be enforced by
GitHub, outside the machine that is stuck -- so the only artifact that can
carry it is the workflow file, and the only thing that can check it is a
rule that reads the workflow file.

WHY A GUARD RULE AND NOT A TEST IN ONE PACKAGE. Each standalone workflow
lists its OWN path in its trigger, so editing ``tankpitbot.yml`` runs
TankpitBot's suite and nothing else. A test placed in any single package
would therefore be silent for the five workflows it does not belong to.
Every package runs this rule set through ``scripts/guard.py``, so a rule
here is the one artifact all of them execute.

WHY LINE-WISE AND NOT A YAML PARSER. ``monorepo_guards`` declares no runtime
dependency but Python itself, deliberately: it is a dependency of only four
of the forty-one packages and must stay installable everywhere. Adding PyYAML
to check indentation would be a poor trade, and the grammar this needs is two
regexes over a file whose shape CI itself already constrains.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, NamedTuple

from monorepo_guards import Violation
from monorepo_guards.config import GuardConfig
from monorepo_guards.util import read_lines

#: Where GitHub requires workflows to live.
WORKFLOWS_DIR: Final[tuple[str, str]] = (".github", "workflows")

#: Largest bound that still counts as a bound.
#:
#: The value only helps if it is far below the 360-minute default it
#: replaces. The slowest job measured in this repository is the packages
#: matrix at 26 minutes, so 60 leaves better than 2x headroom while staying
#: six times tighter than the default. A bound drifting past this is
#: returning to the thing it was introduced to prevent.
MAX_BOUND_MINUTES: Final[int] = 60

#: ``jobs:`` at column zero opens the job map.
JOBS_HEADER: Final[re.Pattern[str]] = re.compile(r"^jobs:\s*$")

#: A two-space-indented key inside that map is one job id.
JOB_HEADER: Final[re.Pattern[str]] = re.compile(r"^ {2}([A-Za-z0-9_-]+):\s*$")

#: A four-space-indented bound belongs to the job most recently opened.
JOB_TIMEOUT: Final[re.Pattern[str]] = re.compile(r"^ {4}timeout-minutes:\s*(\d+)\s*$")

#: Any other column-zero key closes the job map.
TOP_LEVEL_KEY: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z\"']")


class WorkflowJob(NamedTuple):
    """One job found in one workflow file.

    Attributes:
        workflow: Path of the workflow declaring it.
        job_id: The job's key under ``jobs:``.
        line_no: 1-based line of the job's header, so a violation points at
            the job rather than at the top of the file.
        bound: Its ``timeout-minutes``, or None when it declares none.
    """

    workflow: Path
    job_id: str
    line_no: int
    bound: int | None


def parse_workflow_jobs(path: Path) -> list[WorkflowJob]:
    """Read every job declared by one workflow file.

    Tracks whether the cursor is inside the ``jobs:`` map so that a
    two-space key under ``on:`` or ``permissions:`` is not mistaken for a
    job. Without that, ``on:`` with a two-space ``push:`` under it parses as
    a job named ``push`` that can never carry a bound, and the rule reports
    a violation nobody can fix.

    Args:
        path: The workflow file to read.

    Returns:
        The jobs, in the order the file declares them.
    """
    jobs: list[WorkflowJob] = []
    in_jobs = False
    for index, line in enumerate(read_lines(path), start=1):
        if JOBS_HEADER.match(line):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if TOP_LEVEL_KEY.match(line):
            in_jobs = False
            continue
        header = JOB_HEADER.match(line)
        if header is not None:
            jobs.append(
                WorkflowJob(workflow=path, job_id=header.group(1), line_no=index, bound=None)
            )
            continue
        bound = JOB_TIMEOUT.match(line)
        if bound is not None and jobs:
            jobs[-1] = jobs[-1]._replace(bound=int(bound.group(1)))
    return jobs


def workflow_files(monorepo_root: Path) -> list[Path]:
    """Find every workflow file in the repository.

    Args:
        monorepo_root: The repository root.

    Returns:
        The ``.yml`` and ``.yaml`` files under ``.github/workflows``, sorted.
        Empty when the directory does not exist -- a tree with no workflows
        has no unbounded jobs, and the rule must not invent a finding about
        one. That the REAL repository has workflows is pinned by this rule's
        own test rather than assumed here.
    """
    directory = monorepo_root.joinpath(*WORKFLOWS_DIR)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix in {".yml", ".yaml"})


class WorkflowTimeoutRule:
    """Rule requiring every CI job to declare a sane ``timeout-minutes``."""

    name = "workflow-job-timeout"

    def __init__(self, config: GuardConfig) -> None:
        """Bind the rule to the repository whose workflows it checks.

        Args:
            config: The guard run's configuration. Only ``monorepo_root``
                is read: workflows live at the repository root, not in the
                package under check, so this rule is the same for all
                forty-one callers.
        """
        self._monorepo_root = config.monorepo_root

    def run(self, files: list[Path]) -> list[Violation]:
        """Report every job with no bound, or a bound that is not a bound.

        Args:
            files: The package's Python files. Unused: this rule's subject
                is the repository's workflow files, which are never among
                them.

        Returns:
            One violation per offending job, in file then declaration order.
        """
        violations: list[Violation] = []
        for workflow in workflow_files(self._monorepo_root):
            for job in parse_workflow_jobs(workflow):
                if job.bound is None:
                    violations.append(
                        Violation(
                            file=workflow,
                            line_no=job.line_no,
                            kind="missing-timeout-minutes",
                            line=(
                                f"job '{job.job_id}' declares no timeout-minutes, so a frozen "
                                "runner holds it for GitHub's 360-minute default while it "
                                "still reports in_progress"
                            ),
                        )
                    )
                    continue
                if job.bound < 1 or job.bound > MAX_BOUND_MINUTES:
                    violations.append(
                        Violation(
                            file=workflow,
                            line_no=job.line_no,
                            kind="implausible-timeout-minutes",
                            line=(
                                f"job '{job.job_id}' has timeout-minutes {job.bound}; expected "
                                f"1 to {MAX_BOUND_MINUTES}, because a bound near the "
                                "360-minute default is the default it replaced"
                            ),
                        )
                    )
        return violations


__all__ = [
    "JOBS_HEADER",
    "JOB_HEADER",
    "JOB_TIMEOUT",
    "MAX_BOUND_MINUTES",
    "TOP_LEVEL_KEY",
    "WORKFLOWS_DIR",
    "WorkflowJob",
    "WorkflowTimeoutRule",
    "parse_workflow_jobs",
    "workflow_files",
]
