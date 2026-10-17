"""The audit row that names a WSL runner still holding a job GitHub has dropped.

MCPs board task 715b9ec5, A1. On 2026-10-02 and 2026-10-16 every lavender-wsl
listener lost GitHub, GitHub dropped the jobs and read all the runners
offline with busy=false, and each Runner.Worker ran on for up to 1 h 45 min
holding the CI slice, while nothing in the audit could say so: the reaper's
rule waits out the host's ``job_timeout_minutes`` (six hours on lavender),
because the host alone cannot tell a long job from a dropped one. GitHub can.

So the row reads both halves. The hub asks GitHub for each repository's
runners (``gh api repos/<repo>/actions/runners``, the same local auth that
mints registration tokens, :mod:`fleet.core.runner_github`),
and asks the host for the age of each WSL unit's oldest Runner.Worker, in its
cgroup or its job scopes, through the job scope helper's ``--workers`` mode
(:mod:`fleet.core.runner_job_scope_render`). A runner holds a dropped job
when GitHub lists it offline, or online with busy=false, or not at all, while
its unit holds a Worker older than :data:`DROPPED_GRACE_SECONDS`. The row is
one per host and says how many runners it examined and names each dropped
one; it reaps nothing, which stays the reaper's.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import remote
from fleet.core.runner_audit_types import AuditFinding, AuditOutcome
from fleet.core.runner_github import GitHubRunner, read_github_runners
from fleet.core.runner_job_scope_render import JOB_SCOPE_PATH
from fleet.core.script_values import scriptable

#: Seconds a Worker may run while GitHub reads its runner idle or offline
#: before the row names it. A Worker ends seconds after its job, and GitHub's
#: runner list lags a job's start by a few; five minutes is past both.
DROPPED_GRACE_SECONDS = 300

#: Why the row matters, printed beside a drift line.
DROPPED_REASON = (
    "a Runner.Worker whose job GitHub dropped holds the CI slice until the job timeout; on "
    "2026-10-16 six of them held lavender-wsl's for 1 h 45 min, and every dispatch to it "
    "was refused NODE_OWNER_RESERVED"
)


def dropped_check_id(spec: HostRunnerSpec) -> str:
    """The row's id for one host.

    Args:
        spec: The host.

    Returns:
        E.g. ``dropped-jobs:lavender``.
    """
    return f"dropped-jobs:{spec['name']}"


def _workers_argv(spec: HostRunnerSpec, installs: list[RunnerInstall]) -> tuple[str, ...]:
    """The host command that prints each unit's oldest Worker age.

    Args:
        spec: The host.
        installs: Its WSL installs.

    Returns:
        ``wsl`` running the helper's ``--workers`` mode as root in the distro.

    Raises:
        ValueError: When a value cannot be embedded verbatim.
    """
    distro = scriptable(spec["wsl_distro"], label="wsl_distro")
    units = tuple(scriptable(install["service"], label="service") for install in installs)
    return ("wsl", "-d", distro, "-u", "root", "--", JOB_SCOPE_PATH, "--workers", *units)


def _parse_ages(
    spec: HostRunnerSpec, installs: list[RunnerInstall], output: str
) -> dict[str, int | None]:
    """Each unit's oldest Worker age, from the helper's ``--workers`` lines.

    Args:
        spec: The host, named in a refusal.
        installs: The WSL installs whose units were asked about.
        output: The lines, ``<unit> <seconds>`` or ``<unit> -``.

    Returns:
        Each unit's age, None for a unit with no Worker.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` for a line of any other shape,
            or when a unit asked about has no line: an answer that stops
            early is a read that died midway, not a unit with no Worker.
    """
    ages: dict[str, int | None] = {}
    for line in output.splitlines():
        words = line.split()
        if len(words) != 2 or not (words[1] == "-" or words[1].isdigit()):
            raise AppError(
                FleetErrorCode.RUNNER_AUDIT_UNPARSABLE,
                f"{spec['name']}'s job scope helper printed {line!r}, not '<unit> <seconds>' "
                "or '<unit> -'",
            )
        ages[words[0]] = None if words[1] == "-" else int(words[1])
    missing = [install["service"] for install in installs if install["service"] not in ages]
    if missing:
        raise AppError(
            FleetErrorCode.RUNNER_AUDIT_UNPARSABLE,
            f"{spec['name']}'s job scope helper named no Worker age for {', '.join(missing)}",
        )
    return ages


def _github_said(reading: GitHubRunner | None) -> str | None:
    """How GitHub reads a runner, when that reading means it holds no job.

    Args:
        reading: GitHub's reading, None when it does not list the runner.

    Returns:
        The reading in words, or None when GitHub reads it online and busy.
    """
    if reading is None:
        return "unlisted"
    if reading["status"] == "online" and reading["busy"]:
        return None
    return f"{reading['status']} busy={'true' if reading['busy'] else 'false'}"


def attempt_dropped_check(spec: HostRunnerSpec) -> AuditOutcome:
    """The host's dropped-job row, reporting a host that stopped answering as a value.

    Args:
        spec: The host, whose audit script just answered.

    Returns:
        One finding, ok when no examined runner holds a Worker past the
        grace while GitHub reads it idle, offline or unlisted, and drifted
        with the helper's words when it ran and failed; none for a host with
        no WSL install; or no findings and the reason, when ssh could not
        reach the host for the read.

    Raises:
        AppError: ``RUNNER_GITHUB_UNREADABLE`` as :func:`read_github_runners`
            says; ``RUNNER_AUDIT_UNPARSABLE`` when the host's answer cannot
            be read.
    """
    installs = [install for install in spec["installs"] if install["side"] == "wsl"]
    if not installs:
        return AuditOutcome(findings=[], reason="")
    github = {repo: read_github_runners(repo) for repo in sorted({i["repo"] for i in installs})}
    outcome = remote.attempt_ssh(spec["host"], _workers_argv(spec, installs))
    failure = outcome["failure"]
    if failure is not None and failure["code"] is FleetErrorCode.NODE_UNREACHABLE:
        return AuditOutcome(findings=None, reason=failure["message"])
    if failure is not None:
        # The host answered and the helper did not: absent, or refusing,
        # which is this row's drift, never the host's unreachability.
        finding = AuditFinding(
            check_id=dropped_check_id(spec),
            ok=False,
            detail=f"the job scope helper did not read the Worker ages: {failure['message']}",
            reason=DROPPED_REASON,
        )
        return AuditOutcome(findings=[finding], reason="")
    ages = _parse_ages(spec, installs, outcome["output"])
    dropped: list[str] = []
    for install in installs:
        age = ages[install["service"]]
        said = _github_said(github[install["repo"]].get(install["runner_name"]))
        if age is not None and age > DROPPED_GRACE_SECONDS and said is not None:
            dropped.append(f"{install['repo']} {install['runner_name']} ({said}, Worker {age} s)")
    examined = f"examined {len(installs)} WSL runner(s) against GitHub"
    if dropped:
        detail = f"{examined}; {len(dropped)} hold a job GitHub dropped: " + "; ".join(dropped)
    else:
        detail = (
            f"{examined}; none holds a Runner.Worker over {DROPPED_GRACE_SECONDS} s that "
            "GitHub reads idle, offline or unlisted"
        )
    finding = AuditFinding(
        check_id=dropped_check_id(spec), ok=not dropped, detail=detail, reason=DROPPED_REASON
    )
    return AuditOutcome(findings=[finding], reason="")


__all__ = [
    "DROPPED_GRACE_SECONDS",
    "DROPPED_REASON",
    "attempt_dropped_check",
    "dropped_check_id",
]
