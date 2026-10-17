"""Which WSL runners still hold a job GitHub has dropped, and the audit row that names them.

MCPs board task 715b9ec5. On 2026-10-02 and 2026-10-16 every lavender-wsl
listener lost GitHub, GitHub dropped the jobs and read all the runners
offline with busy=false, and each Runner.Worker ran on for up to 1 h 45 min
holding the CI slice, while nothing in the audit could say so: the reaper's
rule waits out the host's ``job_timeout_minutes`` (six hours on lavender),
because the host alone cannot tell a long job from a dropped one. GitHub can.

So the reading takes both halves. The hub asks GitHub for each repository's
runners (``gh api repos/<repo>/actions/runners``, the same local auth that
mints registration tokens, :mod:`fleet.core.runner_github`), and asks the
host for every Runner.Worker of each WSL unit, in its cgroup or its job
scopes, by pid, start and age, through the job scope helper's ``--workers``
mode (:mod:`fleet.core.runner_job_scope_render`). A Worker older than
:data:`DROPPED_GRACE_SECONDS` holds a dropped job when it is its unit's
newest and GitHub reads its runner offline, or online with busy=false, or
not at all; or when its unit holds a newer Worker, since a runner runs one
job at a time and its listener took the next job once the older one was lost.

:func:`read_dropped` is that reading, which two callers share: the audit row
(:func:`attempt_dropped_check`), one per host, which says how many runners it
examined and names each dropped Worker and reaps nothing, and the hub's reap
pass (:mod:`fleet.core.runner_reap_dropped`), which ends each Worker two
readings name.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import remote
from fleet.core.remote import RemoteFailure
from fleet.core.runner_audit_types import AuditFinding, AuditOutcome
from fleet.core.runner_github import GitHubRunner, read_github_runners
from fleet.core.runner_job_scope_render import JOB_SCOPE_PATH
from fleet.core.script_values import scriptable

#: Seconds a Worker may run while GitHub reads its runner idle or offline
#: before the reading names it. A Worker ends seconds after its job, and
#: GitHub's runner list lags a job's start by a few; five minutes is past both.
DROPPED_GRACE_SECONDS = 300

#: Why the row matters, printed beside a drift line.
DROPPED_REASON = (
    "a Runner.Worker whose job GitHub dropped holds the CI slice until the job timeout; on "
    "2026-10-16 six of them held lavender-wsl's for 1 h 45 min, and every dispatch to it "
    "was refused NODE_OWNER_RESERVED"
)


class HostWorker(TypedDict):
    """One Runner.Worker as the host reads it.

    Attributes:
        pid: Its process id.
        start: Its start, in clock ticks since boot, which with the pid names
            this process and no later one given the same id.
        age: Its age in seconds.
    """

    pid: int
    start: int
    age: int


class DroppedWorker(TypedDict):
    """A Worker holding a job GitHub has dropped.

    Attributes:
        install: The runner whose unit holds it.
        worker: The Worker.
        said: Why it is dropped: GitHub's reading of the runner, or the newer
            Worker that superseded it.
    """

    install: RunnerInstall
    worker: HostWorker
    said: str


class DroppedReading(TypedDict):
    """One host's reading.

    Attributes:
        examined: How many WSL runners it read.
        dropped: Each Worker holding a dropped job, in roster order.
        failure: None when the host answered; otherwise why the helper's
            read gave nothing.
    """

    examined: int
    dropped: list[DroppedWorker]
    failure: RemoteFailure | None


def dropped_check_id(spec: HostRunnerSpec) -> str:
    """The row's id for one host.

    Args:
        spec: The host.

    Returns:
        E.g. ``dropped-jobs:lavender``.
    """
    return f"dropped-jobs:{spec['name']}"


def _workers_argv(spec: HostRunnerSpec, installs: list[RunnerInstall]) -> tuple[str, ...]:
    """The host command that prints each unit's Workers.

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


def _unparsable(spec: HostRunnerSpec, detail: str) -> AppError[FleetErrorCode]:
    """The refusal for an answer of the helper's that cannot be read.

    Args:
        spec: The host.
        detail: What was wrong.

    Returns:
        ``RUNNER_AUDIT_UNPARSABLE`` naming the host.
    """
    return AppError(
        FleetErrorCode.RUNNER_AUDIT_UNPARSABLE, f"{spec['name']}'s job scope helper {detail}"
    )


def _parse_workers(
    spec: HostRunnerSpec, installs: list[RunnerInstall], output: str
) -> dict[str, list[HostWorker]]:
    """Each unit's Workers, oldest first, from the helper's ``--workers`` lines.

    Args:
        spec: The host, named in a refusal.
        installs: The WSL installs whose units were asked about.
        output: The lines, ``<unit> <pid> <start> <age>`` or ``<unit> -``.

    Returns:
        Each unit's Workers, empty for a unit with none.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` for a line of any other shape,
            or when a unit asked about has no line: an answer that stops
            early is a read that died midway, not a unit with no Worker.
    """
    workers: dict[str, list[HostWorker]] = {}
    for line in output.splitlines():
        words = line.split()
        if len(words) == 2 and words[1] == "-":
            workers.setdefault(words[0], [])
            continue
        if len(words) != 4 or not all(word.isdigit() for word in words[1:]):
            raise _unparsable(
                spec, f"printed {line!r}, not '<unit> <pid> <start> <age>' or '<unit> -'"
            )
        worker = HostWorker(pid=int(words[1]), start=int(words[2]), age=int(words[3]))
        workers.setdefault(words[0], []).append(worker)
    missing = [install["service"] for install in installs if install["service"] not in workers]
    if missing:
        raise _unparsable(spec, f"named no Worker for {', '.join(missing)}")
    return {unit: sorted(held, key=_started) for unit, held in workers.items()}


def _started(worker: HostWorker) -> int:
    """A Worker's start, which orders a unit's Workers oldest first.

    Args:
        worker: The Worker.

    Returns:
        Its start, in clock ticks since boot.
    """
    return worker["start"]


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


def _dropped_of(
    install: RunnerInstall, workers: list[HostWorker], reading: GitHubRunner | None
) -> list[DroppedWorker]:
    """The Workers of one unit that hold a dropped job.

    Args:
        install: The runner.
        workers: Its unit's Workers, oldest first.
        reading: GitHub's reading of the runner.

    Returns:
        Each Worker past the grace that a newer one superseded, then its
        newest when past the grace while GitHub reads the runner jobless.
    """
    dropped: list[DroppedWorker] = []
    for older in workers[:-1]:
        if older["age"] > DROPPED_GRACE_SECONDS:
            said = f"superseded by Runner.Worker {workers[-1]['pid']}"
            dropped.append(DroppedWorker(install=install, worker=older, said=said))
    if workers and workers[-1]["age"] > DROPPED_GRACE_SECONDS:
        said_by_github = _github_said(reading)
        if said_by_github is not None:
            dropped.append(DroppedWorker(install=install, worker=workers[-1], said=said_by_github))
    return dropped


def read_dropped(spec: HostRunnerSpec) -> DroppedReading:
    """Read which of the host's WSL runners hold a job GitHub dropped.

    Args:
        spec: The host.

    Returns:
        The reading: nothing examined for a host with no WSL install, the
        helper's failure when it did not answer, else each dropped Worker.

    Raises:
        AppError: ``RUNNER_GITHUB_UNREADABLE`` as :func:`read_github_runners`
            says; ``RUNNER_AUDIT_UNPARSABLE`` when the host's answer cannot
            be read.
    """
    installs = [install for install in spec["installs"] if install["side"] == "wsl"]
    if not installs:
        return DroppedReading(examined=0, dropped=[], failure=None)
    github = {repo: read_github_runners(repo) for repo in sorted({i["repo"] for i in installs})}
    outcome = remote.attempt_ssh(spec["host"], _workers_argv(spec, installs))
    if outcome["failure"] is not None:
        return DroppedReading(examined=len(installs), dropped=[], failure=outcome["failure"])
    workers = _parse_workers(spec, installs, outcome["output"])
    dropped = [
        held
        for install in installs
        for held in _dropped_of(
            install,
            workers[install["service"]],
            github[install["repo"]].get(install["runner_name"]),
        )
    ]
    return DroppedReading(examined=len(installs), dropped=dropped, failure=None)


def describe_dropped(held: DroppedWorker) -> str:
    """One dropped Worker in words.

    Args:
        held: The Worker.

    Returns:
        E.g. ``wagner-austin/API lavender-wsl Runner.Worker 4312 (offline
        busy=false, 6300 s)``.
    """
    install, worker = held["install"], held["worker"]
    return (
        f"{install['repo']} {install['runner_name']} Runner.Worker {worker['pid']} "
        f"({held['said']}, {worker['age']} s)"
    )


def attempt_dropped_check(spec: HostRunnerSpec) -> AuditOutcome:
    """The host's dropped-job row, reporting a host that stopped answering as a value.

    Args:
        spec: The host, whose audit script just answered.

    Returns:
        One finding, ok when no examined runner holds a dropped job, and
        drifted with the helper's words when it ran and failed; none for a
        host with no WSL install; or no findings and the reason, when ssh
        could not reach the host for the read.

    Raises:
        AppError: As :func:`read_dropped` says.
    """
    reading = read_dropped(spec)
    if reading["examined"] == 0:
        return AuditOutcome(findings=[], reason="")
    failure = reading["failure"]
    if failure is not None and failure["code"] is FleetErrorCode.NODE_UNREACHABLE:
        return AuditOutcome(findings=None, reason=failure["message"])
    if failure is not None:
        # The host answered and the helper did not: absent, or refusing,
        # which is this row's drift, never the host's unreachability.
        detail = f"the job scope helper did not read the Workers: {failure['message']}"
    elif reading["dropped"]:
        detail = (
            f"examined {reading['examined']} WSL runner(s) against GitHub; "
            f"{len(reading['dropped'])} Runner.Worker(s) hold a job GitHub dropped: "
            + "; ".join(describe_dropped(held) for held in reading["dropped"])
        )
    else:
        detail = (
            f"examined {reading['examined']} WSL runner(s) against GitHub; none holds a "
            f"Runner.Worker over {DROPPED_GRACE_SECONDS} s that GitHub reads idle, offline or "
            "unlisted, or that a newer Worker superseded"
        )
    finding = AuditFinding(
        check_id=dropped_check_id(spec),
        ok=failure is None and not reading["dropped"],
        detail=detail,
        reason=DROPPED_REASON,
    )
    return AuditOutcome(findings=[finding], reason="")


__all__ = [
    "DROPPED_GRACE_SECONDS",
    "DROPPED_REASON",
    "DroppedReading",
    "DroppedWorker",
    "HostWorker",
    "attempt_dropped_check",
    "describe_dropped",
    "dropped_check_id",
    "read_dropped",
]
