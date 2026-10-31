"""A tick's run against the hub's own user manager (MCPs board task 05dd6588).

The real twin of ``test_tick_run.py`` and ``test_agent_orphans.py``: nothing
the manager answers is scripted. The hub is where every fleet-agent tick
runs, each in a transient unit of this user's manager, so these cases start
real units there, kill one the way a restart of the hub runner ends its
runs (``KillSignal=SIGKILL``), and read both back through
:func:`fleet.core.tick_run.read_run_state` and the claim pass: the killed
run is reported lost, the live one is left alone, and this run's own unit,
the one a tick names its jobs' runs by, reads live.

``tools/fleet-execution-hub`` runs them, pinned to the hub; GitHub's Linux
runners have no user manager to ask, which is why they are not ``host_linux``.
"""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Generator
from typing import Final

import pytest
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from fleet.cli import orphans
from fleet.core import _test_hooks, queue, tick_run
from tests._host import READING_PROPERTY
from tests._queue_fakes import QUEUE_CREDENTIALS, FakeQueue, queue_job

#: How long a killed unit may take to read stopped.
STOP_WAIT_SECONDS: Final = 15

#: The runner label the held jobs name.
AGENT: Final = "fleet-netboot-diphtheria"


def start_unit() -> str:
    """Start a real transient unit of this user's manager, a run that waits.

    Returns:
        The unit's name.
    """
    unit = f"fleet-tick-run-test-{uuid.uuid4().hex}.service"
    subprocess.run(
        [
            "/usr/bin/systemd-run",
            "--user",
            "--collect",
            "--quiet",
            f"--unit={unit}",
            "--property=KillSignal=SIGKILL",
            "/bin/sleep",
            "300",
        ],
        check=True,
        timeout=60,
    )
    return unit


def kill_unit(unit: str) -> None:
    """End a unit's every process with SIGKILL, as a restart of its caller does.

    Args:
        unit: The unit.
    """
    subprocess.run([*tick_run.SYSTEMCTL, "kill", "--signal=SIGKILL", unit], check=True, timeout=60)


def stop_unit(unit: str) -> None:
    """Stop a unit if it is still loaded, so no case leaves one behind.

    Args:
        unit: The unit.
    """
    subprocess.run([*tick_run.SYSTEMCTL, "stop", unit], check=False, timeout=60)


def wait_gone(unit: str) -> tick_run.RunState:
    """Read a killed unit until it reads gone, within :data:`STOP_WAIT_SECONDS`.

    Args:
        unit: The unit.

    Returns:
        Its first reading that is not live.
    """
    deadline = time.monotonic() + STOP_WAIT_SECONDS
    state = tick_run.read_run_state(unit)
    while state["live"] and time.monotonic() < deadline:
        time.sleep(0.2)
        state = tick_run.read_run_state(unit)
    return state


@pytest.fixture(name="units")
def _units() -> Generator[list[str], None, None]:
    """The units a case started, each stopped when it ends.

    Yields:
        The list a case appends each started unit to.
    """
    started: list[str] = []
    yield started
    for unit in started:
        stop_unit(unit)


@pytest.mark.host_fleet_hub
def test_a_running_unit_reads_live_and_reads_gone_once_killed(
    units: list[str], record_property: Callable[[str, str], None]
) -> None:
    unit = start_unit()
    units.append(unit)

    running = tick_run.read_run_state(unit)
    kill_unit(unit)
    gone = wait_gone(unit)

    assert running["live"] is True
    assert running["shown"].startswith("ActiveState=active ControlGroup=/")
    assert gone["live"] is False
    assert gone["shown"] in (
        "ActiveState=inactive ControlGroup=",
        "ActiveState=failed ControlGroup=",
    )
    record_property(READING_PROPERTY, f"{unit}: {running['shown']}, then {gone['shown']}")


@pytest.mark.host_fleet_hub
def test_a_unit_the_manager_never_ran_reads_gone() -> None:
    state = tick_run.read_run_state(f"fleet-tick-run-test-{uuid.uuid4().hex}.service")

    assert state == tick_run.RunState(live=False, shown="ActiveState=inactive ControlGroup=")


@pytest.mark.host_fleet_hub
def test_a_user_service_names_its_own_run_and_reads_live(
    units: list[str],
    record_property: Callable[[str, str], None],
) -> None:
    """Inspect from a real tick-shaped user service, not the check's system unit."""
    unit = f"fleet-tick-run-test-{uuid.uuid4().hex}.service"
    units.append(unit)
    program = (
        "from fleet.core import tick_run; "
        "from platform_core.json_utils import dump_json_str; "
        "own = tick_run.read_own_run(); "
        "state = tick_run.read_run_state(own['unit']); "
        "print(dump_json_str({'own': own, 'state': state}))"
    )
    result = subprocess.run(
        [
            "/usr/bin/systemd-run",
            "--user",
            "--collect",
            "--quiet",
            "--wait",
            "--pipe",
            f"--unit={unit}",
            "--property=RuntimeMaxSec=15",
            sys.executable,
            "-c",
            program,
        ],
        check=True,
        timeout=30,
        text=True,
        capture_output=True,
    )
    measured = narrow_json_to_dict(load_json_str(result.stdout))
    assert measured["own"] == {"unit": unit, "machine": _test_hooks.hostname()}
    state = narrow_json_to_dict(measured["state"])
    assert state["live"] is True
    assert str(state["shown"]).startswith("ActiveState=active ControlGroup=/user.slice/")
    assert wait_gone(unit)["live"] is False
    record_property(READING_PROPERTY, f"{unit}: {result.stdout.strip()}, then gone")


@pytest.mark.host_fleet_hub
@pytest.mark.parametrize("left", ["queued", "failed"])
def test_the_claim_pass_reports_the_killed_run_lost_and_leaves_the_live_one(
    units: list[str], left: str
) -> None:
    """Both of the queue's answers to a lost run: ``queued`` the first time,
    ``failed`` for a job it had requeued before (MCPs reportDispatchRunLost),
    each returned as the job the pass reported."""
    machine = _test_hooks.hostname()
    killed = start_unit()
    live = start_unit()
    units.extend([killed, live])
    kill_unit(killed)
    assert wait_gone(killed)["live"] is False
    rows = [
        queue_job(
            id=job_id,
            status="running",
            command="netboot-arm",
            project="MCPs",
            sha=None,
            claimedBy=AGENT,
            node=machine,
            runId=f"{unit} on {machine}",
        )
        for job_id, unit in (
            ("aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa", killed),
            ("bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb", live),
        )
    ]
    endpoint = FakeQueue(
        [
            dump_json_str({"jobs": rows}),
            dump_json_str({"job": queue_job(status=left, command="netboot-arm", sha=None)}),
        ]
    )
    _test_hooks.http_post = endpoint
    identity = queue.identity_arguments(AGENT, "55555555-eeee-4eee-8eee-555555555555", "/tmp")

    reported = orphans.orphan_pass(
        QUEUE_CREDENTIALS,
        identity,
        agent=AGENT,
        own=tick_run.TickRun(unit="session-audit-job-own.service", machine=machine),
    )

    assert endpoint.tools == ["dispatch_list", "dispatch_report"]
    assert endpoint.arguments[1]["jobId"] == "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
    assert str(endpoint.arguments[1]["detail"]).startswith(
        f"run {killed} on {machine} is gone: systemctl --user show read ActiveState="
    )
    assert endpoint.arguments[1]["action"] == "lost"
    assert [job["status"].value for job in reported] == [left]
