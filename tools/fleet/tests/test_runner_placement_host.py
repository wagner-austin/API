"""Where a runner host's units and jobs run, read on it, and what a full slice does beside them.

MCPs board task 715b9ec5, A3. The listeners were moved out of the CI slice
their jobs fill (:data:`~fleet.contracts.runner_slice.LISTENER_SLICE_NAME`),
and the unit suites prove only what the renderers and the job scope helper
ask systemd for, against stand-ins. Two reviews of the task's closure
rejected a note of mine transcribing the host's readings as proof that the
units run there. These cases are the proof the machine writes: the fleet
project ``tools/fleet-execution-runners``, pinned to the runner host's node,
runs them as the fleet's own user, and each passed case records what it read
as a ``host_reading`` the run's summary prints.

1. Every unit ``runners.json`` declares for the host is the host's whole
   set of runner units, each active with its cgroup in the listener slice;
   the listener slice carries no memory bound and the CI slice the one the
   roster declares; and every live ``Runner.Worker`` is in its own job scope
   of the CI slice.
2. On the same kernel, a slice over its ``memory.high`` with its swap denied
   stalls a probe allocating inside it, and not a probe allocating in an
   unbounded sibling, which is where the listeners sit. The pair runs under
   the fleet user's own systemd manager, whose ``user@.service`` has the
   memory controller delegated, so it touches no runner unit and needs no
   root.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from collections.abc import Callable
from typing import Final

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_int

from fleet.contracts.runner_slice import CI_SLICE_NAME, LISTENER_SLICE_NAME
from fleet.contracts.runners import HostRunnerSpec
from fleet.core.runner_proc_bash import JOB_SCOPE_PREFIX
from tests._host import READING_PROPERTY, command_output, unit_properties

#: The cgroup filesystem and the process table.
CGROUP_ROOT: Final[pathlib.Path] = pathlib.Path("/sys/fs/cgroup")
PROC_ROOT: Final[pathlib.Path] = pathlib.Path("/proc")

#: How a Worker's argv0 ends, under any runner's install directory.
WORKER_SUFFIX: Final[str] = "/bin/Runner.Worker"

#: systemd's ``G``: the roster's ``memory_high_gb`` is rendered ``<n>G``.
GIB: Final[int] = 1 << 30

#: The bound the full-slice case puts on its own slice, and what its hog holds.
PROBE_HIGH: Final[str] = "256M"
HOG_MIB: Final[int] = 384

#: The property that keeps the full slice from swapping its way under its high.
SWAP_DENIED: Final[str] = "MemorySwapMax=0"

#: Seconds each probe allocates for.
PROBE_SECONDS: Final[int] = 10

#: The deadline for a probe's unit, which outlasts its allocations' throttling.
PROBE_DEADLINE_SECONDS: Final[int] = 120

#: The least a probe in the full slice must stall, in microseconds of its
#: allocating seconds: one second in ten says the bound bit.
INSIDE_STALL_FLOOR_US: Final[int] = 1_000_000

#: How many times less the sibling may stall than the probe inside.
SIBLING_RATIO: Final[int] = 100

#: The hog: holds :data:`HOG_MIB` MiB, which its slice may not swap, then idles.
HOG: Final[str] = (
    "import time\n"
    f"held = [bytearray(1 << 20) for _ in range({HOG_MIB})]\n"
    f"time.sleep({PROBE_DEADLINE_SECONDS})\n"
)

#: A probe: allocates and touches 1 MiB at a time for :data:`PROBE_SECONDS`
#: seconds, and prints how many it made and how long its own cgroup stalled,
#: every task in it at once, from ``memory.pressure``'s ``full`` total.
PROBE: Final[str] = (
    "import json, mmap, pathlib, time\n"
    "cgroup = pathlib.Path('/sys/fs/cgroup' + pathlib.Path('/proc/self/cgroup')"
    ".read_text().strip().split('::', 1)[1])\n"
    "def stalled():\n"
    "    full = (cgroup / 'memory.pressure').read_text().splitlines()[1]\n"
    "    return int(full.rsplit('total=', 1)[1])\n"
    "before = stalled()\n"
    f"end = time.monotonic() + {PROBE_SECONDS}\n"
    "count = 0\n"
    "while time.monotonic() < end:\n"
    "    region = mmap.mmap(-1, 1 << 20)\n"
    "    for offset in range(0, 1 << 20, 4096):\n"
    "        region[offset] = 1\n"
    "    region.close()\n"
    "    count += 1\n"
    "print(json.dumps({'allocations': count, 'stall_us': stalled() - before}))\n"
)


def _runner_units() -> list[str]:
    """Every runner unit systemd has loaded, whatever its state.

    Returns:
        Their names, sorted.
    """
    listing = ("list-units", "actions.runner.*", "--all", "--no-legend", "--plain")
    text = command_output(["systemctl", *listing])
    return sorted(line.split()[0] for line in text.splitlines() if line)


def _workers() -> dict[int, str]:
    """Every live Runner.Worker and the cgroup it runs in.

    Returns:
        Each Worker's cgroup path, by pid.
    """
    found: dict[int, str] = {}
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv0 = (entry / "cmdline").read_bytes().split(b"\0", 1)[0].decode()
            cgroup = (entry / "cgroup").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            # The process ended between the listing and the read: it runs nowhere.
            continue
        if argv0.endswith(WORKER_SUFFIX):
            found[int(entry.name)] = cgroup.strip().split("::", 1)[1]
    return found


@pytest.mark.host_runner_units
def test_every_runner_unit_runs_in_the_listener_slice_and_every_job_in_its_scope(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    declared = sorted(each["service"] for each in runner_host["installs"])
    assert declared, f"runners.json declares no runner on {runner_host['name']}"
    assert _runner_units() == declared

    placed = {
        unit: unit_properties(unit, ("ActiveState", "Slice", "ControlGroup")) for unit in declared
    }
    assert placed == {
        unit: {
            "ActiveState": "active",
            "Slice": LISTENER_SLICE_NAME,
            "ControlGroup": f"/{LISTENER_SLICE_NAME}/{unit}",
        }
        for unit in declared
    }
    listener_high = (CGROUP_ROOT / LISTENER_SLICE_NAME / "memory.high").read_text().strip()
    ci_high = (CGROUP_ROOT / CI_SLICE_NAME / "memory.high").read_text().strip()
    budget = runner_host["ci_slice"]["memory_high_gb"] * GIB
    assert (listener_high, ci_high) == ("max", str(budget))

    workers = _workers()
    prefix = f"/{CI_SLICE_NAME}/{JOB_SCOPE_PREFIX}"
    scopes = [f"{prefix}{unit.removesuffix('.service')}" for unit in declared]
    misplaced = {
        pid: cgroup
        for pid, cgroup in workers.items()
        if cgroup not in {f"{scope}-{pid}.scope" for scope in scopes}
    }
    assert misplaced == {}
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: {len(declared)} of {len(declared)} declared runner unit(s) "
        f"active in /{LISTENER_SLICE_NAME} (memory.high {listener_high}); {len(workers)} "
        f"Runner.Worker(s), each in its own job scope of /{CI_SLICE_NAME} (memory.high {ci_high})",
    )


def _user_cgroup(environment: dict[str, str], unit: str) -> pathlib.Path:
    """Where a unit of the user's own manager keeps its cgroup, as that manager says.

    A slice's name is its place in the tree, each dash a level, so the path
    is read rather than built.

    Args:
        environment: The environment that reaches the user's manager.
        unit: The unit.

    Returns:
        The cgroup directory.
    """
    shown = subprocess.run(
        ["systemctl", "--user", "show", "-p", "ControlGroup", "--value", unit],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()
    assert shown.startswith("/"), f"{unit} runs in no cgroup: {shown!r}"
    return CGROUP_ROOT / shown.removeprefix("/")


def _probe(environment: dict[str, str], slice_name: str, unit: str) -> subprocess.Popen[str]:
    """Start one probe in a slice of the user's manager, its output piped back.

    Args:
        environment: The environment that reaches the user's manager.
        slice_name: The slice it allocates in.
        unit: Its unit's name.

    Returns:
        The running ``systemd-run``.
    """
    return subprocess.Popen(
        [
            "systemd-run",
            "--user",
            "--quiet",
            "--pipe",
            "--wait",
            "--collect",
            f"--slice={slice_name}",
            f"--unit={unit}",
            "python3",
            "-c",
            PROBE,
        ],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
    )


def _stall(run: subprocess.Popen[str]) -> tuple[int, int]:
    """A finished probe's allocations and the microseconds its cgroup stalled.

    Args:
        run: The probe's ``systemd-run``.

    Returns:
        ``(allocations, stall_us)``.
    """
    printed, _ = run.communicate(timeout=PROBE_DEADLINE_SECONDS)
    exited = run.wait()
    assert exited == 0, f"a probe exited {exited}: {printed!r}"
    reading = narrow_json_to_dict(load_json_str(printed))
    return require_int(reading, "allocations"), require_int(reading, "stall_us")


@pytest.mark.host_runner_units
def test_a_slice_over_its_high_stalls_its_own_tasks_and_not_an_unbounded_sibling(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    uid = command_output(["id", "-u"]).strip()
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "XDG_RUNTIME_DIR": f"/run/user/{uid}"}
    tag = f"{runner_host['name']}-{os.getpid()}"
    full = f"fleet-placement-full-{tag}.slice"
    sibling = f"fleet-placement-free-{tag}.slice"
    hog = f"fleet-placement-hog-{tag}.service"
    user = ["systemctl", "--user"]
    subprocess.run(
        [
            "systemd-run",
            "--user",
            "--quiet",
            "--collect",
            f"--slice={full}",
            f"--unit={hog}",
            "python3",
            "-c",
            HOG,
        ],
        env=environment,
        check=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    try:
        subprocess.run(
            [*user, "set-property", "--runtime", full, f"MemoryHigh={PROBE_HIGH}", SWAP_DENIED],
            env=environment,
            check=True,
            timeout=60,
        )
        inside = _probe(environment, full, f"fleet-placement-inside-{tag}.service")
        outside = _probe(environment, sibling, f"fleet-placement-outside-{tag}.service")
        inside_count, inside_stall = _stall(inside)
        outside_count, outside_stall = _stall(outside)
        cgroup = _user_cgroup(environment, full)
        current = int((cgroup / "memory.current").read_text())
        high = int((cgroup / "memory.high").read_text())
    finally:
        subprocess.run([*user, "stop", hog], env=environment, check=True, timeout=60)
        subprocess.run([*user, "revert", full], env=environment, check=True, timeout=60)

    assert current > high, f"{full} held {current} bytes, under its high of {high}: it never filled"
    assert inside_stall >= INSIDE_STALL_FLOOR_US
    assert outside_stall * SIBLING_RATIO <= inside_stall
    record_property(
        READING_PROPERTY,
        f"{runner_host['name']}: {full} at {current} bytes over its high {high}, swap denied: "
        f"a probe in it made {inside_count} 1 MiB allocations in {PROBE_SECONDS} s and stalled "
        f"{inside_stall} us; one in the unbounded sibling {sibling} made {outside_count} and "
        f"stalled {outside_stall} us",
    )
