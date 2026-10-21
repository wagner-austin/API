"""The run a fleet-agent tick is, named so a later tick can prove it gone (MCPs task 05dd6588).

A TICK'S RUN IS ITS UNIT. The MCPs hub runner starts every tick as a
transient unit of the user's systemd manager, ``session-audit-job-<hex>.
service``, bound to the runner's own unit, so a restart of the runner ends
every run with it (MCPs ``session_audit.process_job.unit``, board tasks
c39ee056 and 37e44f43). Measured on diphtheria on 2026-10-21: a fleet tick
and the agent it starts read ``0::/user.slice/user-1000.slice/user@1000.
service/app.slice/session-audit-job-df261b2fda99482cb3d130e721072c91.
service`` from ``/proc/self/cgroup``. A tick runs the job it claims to its
close, so the unit is the whole of the job's run: every process the job
starts is in its control group, whatever becomes of its parent. The job is
reported started under that unit and the machine whose manager holds it,
``<unit> on <machine>``, the run id :func:`encode_run_id` writes.

A RUN IS GONE WHEN ITS UNIT READS STOPPED WITH NO CONTROL GROUP, the same
two readings MCPs' process job confirms its own stop by. Measured on
diphtheria on 2026-10-21 (systemd 255): ``systemctl --user show`` of a
running tick's unit read ``ActiveState=active`` and its control group's
path, and of a unit the manager no longer knows, which is what a stopped
run becomes since its unit is started ``--collect``, it read
``ActiveState=inactive``, ``ControlGroup=`` empty and ``LoadState=
not-found``, exiting 0. Any other reading is a run that may still be going,
so nothing is said to be gone on it.

A TICK OUTSIDE A UNIT CLAIMS NOTHING. A run no unit holds cannot be proven
gone afterwards, and one ended by a restart would strand its job again, so
a tick whose process is in no unit of a user manager (an ssh login's scope,
or a machine with no ``/proc``) is refused by name before it claims, as the
hub runner refuses to start a run with nothing to bind it to; start one by
hand inside a unit, as ``systemd-run --user --pty`` does.
"""

from __future__ import annotations

import pathlib
import re
from typing import Final, TypedDict

from platform_core.errors import AppError, ErrorCode

from fleet.core import _test_hooks
from fleet.core.commit_tree import step_refusal

#: Where a process reads the control group it runs in.
PROC_SELF_CGROUP: Final = pathlib.Path("/proc/self/cgroup")

#: The systemctl the distribution installs, asking the user's own manager;
#: named, never searched for.
SYSTEMCTL: Final[tuple[str, ...]] = ("/usr/bin/systemctl", "--user")

#: The two properties a run's liveness is read from.
SHOWN: Final[tuple[str, ...]] = ("show", "--property=ActiveState", "--property=ControlGroup")

#: The longest a ``systemctl show`` may take; it answers from the manager's
#: memory, in milliseconds when the manager is well.
SHOW_TIMEOUT_SECONDS: Final = 30

#: The states of a unit with no process left to run.
STOPPED_STATES: Final[frozenset[str]] = frozenset({"inactive", "failed"})

#: A unified-hierarchy line naming a unit of a user manager, the unit last.
USER_MANAGER_UNIT: Final = re.compile(
    r"^0::/(?:[^/]+/)*user@\d+\.service/(?:[^/]+/)*([^/]+\.(?:service|scope))$"
)

#: A run id :func:`encode_run_id` wrote: the unit, then the machine.
RUN_ID: Final = re.compile(
    r"^(?P<unit>[A-Za-z0-9:_.@\\-]+\.(?:service|scope)) on (?P<machine>[a-z0-9][a-z0-9.-]*)$"
)

#: The refusal of a tick that runs in no unit.
UNBOUND_CODE: Final = "FLEET_AGENT_RUN_UNBOUND"

#: The refusal when the manager cannot be asked about a run.
UNREADABLE_CODE: Final = "FLEET_AGENT_RUN_UNREADABLE"


class TickRun(TypedDict):
    """The unit a tick runs in, and the machine whose manager holds it.

    Attributes:
        unit: The unit's name, e.g. ``session-audit-job-<hex>.service``.
        machine: The machine's hostname, lowercased.
    """

    unit: str
    machine: str


class RunState(TypedDict):
    """What the manager says of a run's unit.

    Attributes:
        live: False only for a stopped state with no control group.
        shown: The two readings, ``ActiveState=<s> ControlGroup=<path>``,
            for the line that names the run.
    """

    live: bool
    shown: str


def own_run(cgroup_text: str, machine: str) -> TickRun:
    """Name the run this process is part of.

    Args:
        cgroup_text: ``/proc/self/cgroup``'s contents.
        machine: This machine's hostname.

    Returns:
        The unit this process runs in, on ``machine``.

    Raises:
        AppError: ``CONFIG_ERROR`` carrying :data:`UNBOUND_CODE` and the
            control group read, when it is no unit of a user manager.
    """
    lines = cgroup_text.splitlines()
    found = [match for line in lines if (match := USER_MANAGER_UNIT.match(line)) is not None]
    if len(found) == 0:
        raise AppError(
            code=ErrorCode.CONFIG_ERROR,
            message=(
                f"{UNBOUND_CODE}: this tick runs in {', '.join(lines) or 'no control group'}, "
                "no unit of a user manager, so a run it claimed could not be proven gone "
                "once ended; start it inside one, as the hub runner and "
                "systemd-run --user --pty do"
            ),
        )
    return TickRun(unit=found[0].group(1), machine=machine)


def read_own_run() -> TickRun:
    """Name the run this process is part of, on this machine.

    Returns:
        The unit read from :data:`PROC_SELF_CGROUP`, on this hostname.

    Raises:
        AppError: As :func:`own_run` raises, and ``CONFIG_ERROR`` carrying
            :data:`UNBOUND_CODE` when the file does not exist, as on a
            machine with no ``/proc``.
    """
    if not _test_hooks.file_exists(PROC_SELF_CGROUP):
        return own_run("", _test_hooks.hostname())
    return own_run(_test_hooks.read_text(PROC_SELF_CGROUP), _test_hooks.hostname())


def encode_run_id(run: TickRun) -> str:
    """The run id a job is reported started under.

    Args:
        run: The tick's run.

    Returns:
        ``<unit> on <machine>``.
    """
    return f"{run['unit']} on {run['machine']}"


def decode_run_id(run_id: str) -> TickRun | None:
    """Read a run id back as the run it names.

    Args:
        run_id: A job's run id.

    Returns:
        The run, or None when the id is none :func:`encode_run_id` wrote:
        empty, as a job claimed and not started has, or one written before
        runs were named by their unit.
    """
    matched = RUN_ID.fullmatch(run_id)
    if matched is None:
        return None
    return TickRun(unit=matched.group("unit"), machine=matched.group("machine"))


def read_run_state(unit: str) -> RunState:
    """Ask this machine's user manager whether a run's unit is still going.

    Args:
        unit: The unit, as :func:`decode_run_id` read it.

    Returns:
        Whether it is live, and the two readings.

    Raises:
        AppError: ``CONFIG_ERROR`` carrying :data:`UNREADABLE_CODE`, the
            command, its exit and its stderr, when systemctl exits non-zero
            or answers without both properties.
    """
    argv = (*SYSTEMCTL, *SHOWN, unit)
    result = _test_hooks.run(argv, timeout_seconds=SHOW_TIMEOUT_SECONDS)
    shown: dict[str, str] = {}
    for line in result["stdout"].splitlines():
        name, _, value = line.partition("=")
        shown[name] = value
    state = shown.get("ActiveState")
    group = shown.get("ControlGroup")
    if result["returncode"] != 0 or state is None or group is None:
        raise AppError(
            code=ErrorCode.CONFIG_ERROR,
            message=step_refusal(UNREADABLE_CODE, " ".join(argv), result)
            + f"; it answered {result['stdout'].strip()!r}",
        )
    return RunState(
        live=state not in STOPPED_STATES or group != "",
        shown=f"ActiveState={state} ControlGroup={group}",
    )


__all__ = [
    "PROC_SELF_CGROUP",
    "RUN_ID",
    "SHOWN",
    "SHOW_TIMEOUT_SECONDS",
    "STOPPED_STATES",
    "SYSTEMCTL",
    "UNBOUND_CODE",
    "UNREADABLE_CODE",
    "USER_MANAGER_UNIT",
    "RunState",
    "TickRun",
    "decode_run_id",
    "encode_run_id",
    "own_run",
    "read_own_run",
    "read_run_state",
]
