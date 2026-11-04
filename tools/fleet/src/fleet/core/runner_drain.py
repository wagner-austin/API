"""Taking a runner host's runners out of GitHub's pool and putting them back.

MCPs board task f19d66a5. A runner host is switched to a new NixOS generation with its runner units
stopped, because the switch restarts a unit whose definition changed and a
restart ends the job it holds (the units' ``KillMode`` is ``control-group``).
So before a switch every runner must stop, and none may be holding a job
when it does.

READING A RUNNER IDLE AND THEN STOPPING IT DOES NOT DO THAT. On 2026-11-04
lavender was switched by a hand loop that read each unit free of a
``Runner.Worker`` and then stopped it. MCPs run 41764089450 had jobs queued
for the only two MCPs runners, and each runner's listener took its next job
between the reading and the stop: packages/wiki-cli started on ``lavender``
at 04:08:49Z, the second its unit was stopped, and taskboard-mcp on
``lavender-2`` at 04:23:36Z, one second before. Both jobs ended with no step
run and GitHub's "The self-hosted runner lost communication with the
server", and the run failed. No reading of the host can close that gap,
because the assignment is GitHub's, made while the reading is in hand.

SO THE ASSIGNMENT STOPS FIRST. Every workflow the fleet's runners serve
targets a runner by its custom label (``runs-on: [self-hosted, lavender]``,
the roster's ``labels``), never by a read-only label alone, so a runner whose
custom labels are removed matches no job and is assigned nothing new.
:func:`drain` removes them, then waits for GitHub to read each runner not
busy at every reading across :data:`DRAIN_SETTLE_SECONDS`, all taken after
the removal, which also outlasts the moment between a job's assignment and
GitHub reading the runner busy; only then is the unit stopped, and it is read
back ``inactive``. A job a runner already held runs to its end first, up to
the host's ``job_timeout_minutes``. :func:`resume` is the other half, after
the switch: it starts each unit that is not active, puts the roster's labels
back, and reads each runner online.

WHAT IS CHECKED BEFORE ANYTHING CHANGES. Each runner GitHub lists under an
install's name must answer to exactly the roster's custom labels (to none as
well, for :func:`resume`); otherwise the act refuses with
``RUNNER_LABELS_DRIFTED`` and changes nothing, since removing and restoring
the roster's labels would rewrite a pool the roster does not describe. An
install GitHub lists under no such name is one the roster gained since the
host was last registered (an onboard adds it, and the switch lays its unit):
GitHub can assign it no job, so the drain stops it without waiting and the
resume starts it and names ``--register``, which registers it.

A DRAIN THAT TIMES OUT PUTS THE HOST BACK. When a runner still holds a job
at the deadline, every unit the drain stopped is started again and every
runner's labels are restored before ``RUNNER_DRAIN_TIMEOUT`` names the
runners still busy, so a drain that could not finish leaves the host serving
as it was. A ``gh`` refusal partway through raises with gh's own words and
leaves the steps already done; its message names ``--resume yes``, which puts
the roster's labels back on every runner of the host.
"""

from __future__ import annotations

from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks, remote
from fleet.core.runner_github import (
    GitHubRunner,
    clear_custom_labels,
    read_custom_labels,
    read_github_runners,
    set_custom_labels,
)
from fleet.core.script_values import scriptable

#: How long GitHub must read a runner not busy, at every reading, after its
#: labels went, before its unit is stopped: far longer than the moment
#: between a job's assignment and the runner reading busy, so a job assigned
#: just before the labels went is seen.
DRAIN_SETTLE_SECONDS: Final = 30

#: How long to wait between readings of which runners are still busy.
DRAIN_POLL_SECONDS: Final = 15

#: How long past the host's ``job_timeout_minutes`` the drain waits: a job
#: that started just before the labels went may run that long, and GitHub
#: cancels it at its timeout, after which its runner reads idle.
DRAIN_MARGIN_SECONDS: Final = 300

#: How long the resumed runners may take to read online, in seconds: each
#: start runs a listener that connects to GitHub once.
RESUME_DEADLINE_SECONDS: Final = 300

#: How long to wait between readings of which resumed runners are offline.
RESUME_POLL_SECONDS: Final = 5

#: What ``systemctl show -p ActiveState`` reads for a stopped and a running unit.
INACTIVE: Final = "inactive"
ACTIVE: Final = "active"


class DrainTarget(TypedDict):
    """One install and GitHub's id for its registration.

    Attributes:
        install: The roster's install.
        runner_id: GitHub's id for it, which every label call names.
    """

    install: RunnerInstall
    runner_id: int


class HostReading(TypedDict):
    """GitHub's reading of a host's installs, checked before anything changes.

    Attributes:
        targets: Each install GitHub lists, with its id, in roster order.
        unregistered: Each install GitHub lists under no such name, in roster
            order: one the roster gained since the host's last registration,
            which GitHub cannot assign a job to and ``--register`` registers.
    """

    targets: list[DrainTarget]
    unregistered: list[RunnerInstall]


class DrainReport(TypedDict):
    """What a drain or a resume did, step by step.

    Attributes:
        steps: One line per step, in order, saying what it did.
    """

    steps: list[str]


def _list_runners(installs: list[RunnerInstall]) -> dict[str, GitHubRunner | None]:
    """GitHub's reading of each install's runner, by its unit, None where it lists none.

    Args:
        installs: The installs to read, one ``gh`` list per repository.

    Returns:
        Each install's runner keyed by its ``service``, or None when GitHub
        lists no runner of its name in its repository.

    Raises:
        AppError: ``RUNNER_GITHUB_UNREADABLE`` from the list.
    """
    listed = {repo: read_github_runners(repo) for repo in sorted({i["repo"] for i in installs})}
    return {i["service"]: listed[i["repo"]].get(i["runner_name"]) for i in installs}


def _read_runners(spec: HostRunnerSpec, installs: list[RunnerInstall]) -> dict[str, GitHubRunner]:
    """GitHub's reading of each install's runner, every one of which it registered.

    Args:
        spec: The host, for the refusal.
        installs: Installs whose runners GitHub listed when the act began.

    Returns:
        Each install's runner, keyed by its ``service``.

    Raises:
        AppError: ``RUNNER_LABELS_DRIFTED`` when GitHub no longer lists one:
            its registration was removed while the act ran, and
            ``RUNNER_GITHUB_UNREADABLE`` from the list.
    """
    readings: dict[str, GitHubRunner] = {}
    for service, runner in _list_runners(installs).items():
        if runner is None:
            raise AppError(
                FleetErrorCode.RUNNER_LABELS_DRIFTED,
                f"{spec['name']}: GitHub no longer lists the runner of {service}, which it "
                f"listed when this act began; fleet-runners --resume yes --host {spec['name']} "
                "puts the host back once it is registered again",
            )
        readings[service] = runner
    return readings


def _read_host(spec: HostRunnerSpec, *, cleared_allowed: bool) -> HostReading:
    """Every install of the host, checked against GitHub before anything changes.

    Args:
        spec: The host.
        cleared_allowed: Whether a runner answering to no custom label passes
            too, as one a drain left does.

    Returns:
        The installs GitHub lists, with their ids, and those it does not.

    Raises:
        AppError: ``RUNNER_LABELS_DRIFTED`` when GitHub lists an install's
            runner with custom labels that are not the roster's;
            ``RUNNER_GITHUB_UNREADABLE`` from a read.
    """
    listed = _list_runners(spec["installs"])
    targets: list[DrainTarget] = []
    unregistered: list[RunnerInstall] = []
    for install in spec["installs"]:
        runner = listed[install["service"]]
        if runner is None:
            unregistered.append(install)
            continue
        runner_id = runner["id"]
        labels = read_custom_labels(install["repo"], runner_id)
        roster = sorted(install["labels"])
        if sorted(labels) != roster and not (cleared_allowed and not labels):
            raise AppError(
                FleetErrorCode.RUNNER_LABELS_DRIFTED,
                f"{spec['name']}: runner {install['runner_name']} of {install['repo']} answers "
                f"to custom labels {labels}, and the roster declares {roster}; nothing was "
                "changed, since removing and restoring the roster's would rewrite a pool the "
                "roster does not describe",
            )
        targets.append(DrainTarget(install=install, runner_id=runner_id))
    return HostReading(targets=targets, unregistered=unregistered)


def _active_state(spec: HostRunnerSpec, service: str) -> str:
    """What systemd on the host reads a unit's state as.

    Args:
        spec: The host.
        service: The unit.

    Returns:
        Its ``ActiveState``, e.g. ``active`` or ``inactive``.

    Raises:
        AppError: The remote layer's codes.
        ValueError: When the unit's name cannot be carried verbatim.
    """
    unit = scriptable(service, label="service")
    argv = ("systemctl", "show", "-p", "ActiveState", "--value", unit)
    return remote.run_ssh(spec["host"], argv).strip()


def _change_unit(spec: HostRunnerSpec, service: str, verb: str, expected: str) -> None:
    """Start or stop a unit as root, and read that it reached the state asked for.

    Args:
        spec: The host.
        service: The unit.
        verb: ``start`` or ``stop``; ``systemctl`` waits for the change.
        expected: The ``ActiveState`` the change leaves.

    Raises:
        AppError: ``DISPATCH_FAILED`` when the unit reads another state
            afterwards, naming it, and the remote layer's codes.
        ValueError: When the unit's name cannot be carried verbatim.
    """
    argv = (*remote.ROOT_PREFIX, "systemctl", verb, scriptable(service, label="service"))
    remote.run_ssh(spec["host"], argv)
    state = _active_state(spec, service)
    if state != expected:
        raise AppError(
            FleetErrorCode.DISPATCH_FAILED,
            f"{spec['name']}: {service} reads {state} after systemctl {verb}, not {expected}; "
            "journalctl -u <unit> on the host says why",
        )


def _restore_pool(spec: HostRunnerSpec, stopped: list[str], targets: list[DrainTarget]) -> None:
    """Put a drained host back as it served: units started, labels restored.

    Args:
        spec: The host.
        stopped: The units the drain stopped.
        targets: Every registered install, whose labels the drain removed.

    Raises:
        AppError: As :func:`_change_unit` and
            :func:`fleet.core.runner_github.set_custom_labels` describe.
        ValueError: As :func:`_change_unit` describes.
    """
    for service in stopped:
        _change_unit(spec, service, "start", ACTIVE)
    for target in targets:
        install = target["install"]
        set_custom_labels(install["repo"], target["runner_id"], install["labels"])


def _stop_when_idle(
    spec: HostRunnerSpec, targets: list[DrainTarget], stopped: list[str], steps: list[str]
) -> None:
    """Stop each unlabelled runner once GitHub has read it idle across the settle.

    Args:
        spec: The host.
        targets: The registered installs, every one's labels already removed.
        stopped: The units stopped so far, appended to as each stops.
        steps: The report's lines, appended to as each stops.

    Raises:
        AppError: ``RUNNER_DRAIN_TIMEOUT`` naming each runner still busy
            :data:`DRAIN_MARGIN_SECONDS` past the host's job timeout, after
            every stopped unit was started again and every label restored;
            ``DISPATCH_FAILED`` from a unit that does not stop; GitHub's and
            the remote layer's codes.
        ValueError: As :func:`_change_unit` describes.
    """
    cleared = _test_hooks.now()
    deadline = cleared + spec["job_timeout_minutes"] * 60 + DRAIN_MARGIN_SECONDS
    idle_since: dict[str, float] = {}
    pending = list(targets)
    while pending:
        instant = _test_hooks.now()
        readings = _read_runners(spec, [target["install"] for target in pending])
        still: list[DrainTarget] = []
        for target in pending:
            service = target["install"]["service"]
            if readings[service]["busy"]:
                idle_since.pop(service, None)
                still.append(target)
            elif instant - idle_since.setdefault(service, instant) < DRAIN_SETTLE_SECONDS:
                still.append(target)
            else:
                _change_unit(spec, service, "stop", INACTIVE)
                stopped.append(service)
                steps.append(f"stopped: {service}, idle for {DRAIN_SETTLE_SECONDS}s with no label")
        pending = still
        if pending and instant >= deadline:
            _restore_pool(spec, stopped, targets)
            busy = ", ".join(target["install"]["service"] for target in pending)
            raise AppError(
                FleetErrorCode.RUNNER_DRAIN_TIMEOUT,
                f"{spec['name']}: {busy} still held a job {int(instant - cleared)}s after the "
                f"labels went, past the host's {spec['job_timeout_minutes']} minute job "
                f"timeout; the {len(stopped)} stopped unit(s) were started again and every "
                "runner's labels restored, so the host serves as it did",
            )
        if pending:
            _test_hooks.sleep(DRAIN_POLL_SECONDS)


def drain(spec: HostRunnerSpec) -> DrainReport:
    """Take every runner of one host out of GitHub's pool and stop it once it holds no job.

    An install GitHub has not registered can be assigned no job, so its unit
    is stopped at once, if it runs.

    Args:
        spec: The host's roster entry.

    Returns:
        What each step did.

    Raises:
        AppError: ``RUNNER_LABELS_DRIFTED`` before anything changes, as
            :func:`_read_host` describes; ``RUNNER_DRAIN_TIMEOUT`` as
            :func:`_stop_when_idle` describes; ``RUNNER_LABELS_REFUSED`` and
            ``RUNNER_GITHUB_UNREADABLE`` from GitHub; ``DISPATCH_FAILED`` from
            a unit that does not stop; and the remote layer's codes.
        ValueError: When a unit's name cannot be carried verbatim.
    """
    reading = _read_host(spec, cleared_allowed=False)
    targets = reading["targets"]
    for target in targets:
        clear_custom_labels(target["install"]["repo"], target["runner_id"])
    steps = [f"labels: removed from {len(targets)} runner(s); GitHub assigns them nothing new"]
    stopped: list[str] = []
    for install in reading["unregistered"]:
        service = install["service"]
        if _active_state(spec, service) != INACTIVE:
            _change_unit(spec, service, "stop", INACTIVE)
            stopped.append(service)
        steps.append(f"stopped: {service}, which GitHub has not registered")
    _stop_when_idle(spec, targets, stopped, steps)
    return DrainReport(steps=steps)


def resume(spec: HostRunnerSpec) -> DrainReport:
    """Start every runner of one host, put the roster's labels back, and read each online.

    Args:
        spec: The host's roster entry.

    Returns:
        What each step did.

    An install GitHub has not registered is started too, and named:
    ``--register yes`` registers it, with the roster's labels.

    Raises:
        AppError: ``RUNNER_LABELS_DRIFTED`` before anything changes, as
            :func:`_read_host` describes; ``DISPATCH_FAILED`` from a unit that
            does not start, or naming each runner GitHub still reads offline
            :data:`RESUME_DEADLINE_SECONDS` after its labels were restored;
            ``RUNNER_LABELS_REFUSED`` and ``RUNNER_GITHUB_UNREADABLE`` from
            GitHub; and the remote layer's codes.
        ValueError: When a unit's name cannot be carried verbatim.
    """
    reading = _read_host(spec, cleared_allowed=True)
    targets = reading["targets"]
    started: list[str] = []
    for install in spec["installs"]:
        service = install["service"]
        if _active_state(spec, service) != ACTIVE:
            _change_unit(spec, service, "start", ACTIVE)
            started.append(service)
    steps = [
        f"started: {', '.join(started)}" if started else "started: none, every unit was active",
    ]
    if reading["unregistered"]:
        names = ", ".join(install["service"] for install in reading["unregistered"])
        steps.append(
            f"unregistered: {names}; fleet-runners --register yes --host {spec['name']} "
            "registers them with the roster's labels"
        )
    for target in targets:
        install = target["install"]
        set_custom_labels(install["repo"], target["runner_id"], install["labels"])
    steps.append(f"labels: the roster's restored on {len(targets)} runner(s)")
    deadline = _test_hooks.now() + RESUME_DEADLINE_SECONDS
    installs = [target["install"] for target in targets]
    while True:
        readings = _read_runners(spec, installs)
        offline = [i["service"] for i in installs if readings[i["service"]]["status"] != "online"]
        if not offline:
            steps.append(f"online: all {len(installs)} runner(s), as GitHub reads them")
            return DrainReport(steps=steps)
        if _test_hooks.now() >= deadline:
            raise AppError(
                FleetErrorCode.DISPATCH_FAILED,
                f"{spec['name']}: GitHub reads {', '.join(offline)} offline "
                f"{RESUME_DEADLINE_SECONDS}s after their labels were restored; journalctl -u "
                "<unit> on the host says why",
            )
        _test_hooks.sleep(RESUME_POLL_SECONDS)


__all__ = [
    "ACTIVE",
    "DRAIN_MARGIN_SECONDS",
    "DRAIN_POLL_SECONDS",
    "DRAIN_SETTLE_SECONDS",
    "INACTIVE",
    "RESUME_DEADLINE_SECONDS",
    "RESUME_POLL_SECONDS",
    "DrainReport",
    "DrainTarget",
    "drain",
    "resume",
]
