"""Cases that execute this package's rendered scripts on a real host.

Board task 465689f5, rule R3 of b6eb30c8. A handful of cases here run the
PowerShell or sh this package renders on the machine running the suite,
which is the only place a defect between the code and the machine can
show. Each is bound to one platform, and until this module each carried a
``skipif`` on ``sys.platform``: API's CI runs on Linux, so every Windows
case was skipped there, and the hub is Windows, so every Linux case was
skipped at home. No run recorded both halves green at one commit.

So the binding is a marker, ``host_windows`` or ``host_linux``, and the two
ways to run the suite are platform_core's :mod:`~platform_core.host_execution`
(lifted from this module so tools/hpc3 binds its cluster case the same way):

- The ORDINARY run (``make check``) behaves as the ``skipif`` did: a host
  case bound to the other platform is skipped, naming the node that runs it.
- The EXECUTION run (``--host-execution``, ``make execution``) selects only
  the host cases bound to this machine's platform and FAILS when any of them
  skips or when none ran. It is what the fleet projects
  ``tools/fleet-execution`` (a Windows node) and
  ``tools/fleet-execution-linux`` (a Linux node) run, and a deploy refuses a
  commit without a passed run of each (MCPs' ``publish-executed``).

A THIRD marker, ``host_linux_docker``, binds a case to Linux like
``host_linux`` but runs it ONLY in the execution run: it asserts, from inside
the run, that the suite itself is running as the node's execdocker user
against that user's rootless daemon and cannot reach the stack's (MCPs board
task a8ee9b21). That is true only because ``tools/fleet-execution-linux``
requires the ``docker`` tag, which makes the fleet build it through the
isolated script, on a node that has execdocker and never on a CI runner. The
ordinary run skips it on every platform, naming the project that runs it,
and the Linux execution run requires it like any host case.

A FOURTH, ``host_runner_units``, binds a case to a host that runs the fleet's
WSL runner units, and reads where those units and their jobs run on it (MCPs
board task 715b9ec5, A3). No platform has that capability: the Linux
execution run deselects such a case, the ordinary run skips it, and only a
run given ``--runner-host=<the runners.json host>`` has it, which is what
``tools/fleet-execution-runners`` passes from the one node it is pinned to.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Final

import pytest
from platform_core import host_execution

from fleet.cli.runners import load_runner_spec
from fleet.contracts.runners import HostRunnerSpec

#: This machine's platform, as the markers and the fleet's tags name it.
HOST_PLATFORM: Final[str] = "windows" if sys.platform == "win32" else "linux"

#: The marker binding a case to each platform.
MARKERS: Final[dict[str, str]] = {"windows": "host_windows", "linux": "host_linux"}

#: The marker for a Linux host case that needs the node's rootless daemon,
#: run only by the execution run.
EXECUTION_ONLY_MARKER: Final[str] = "host_linux_docker"

#: The capability of a host that runs the fleet's WSL runner units, whose cases
#: read where those units and their jobs run (MCPs board task 715b9ec5).
RUNNER_HOST: Final[str] = "runner-host"

#: The marker for a case that reads a runner host's live units, run only by
#: that host's execution run.
RUNNER_HOST_MARKER: Final[str] = "host_runner_units"

#: The option naming the runner host this run executes on, as ``runners.json``
#: names it. Given, the run's capability is :data:`RUNNER_HOST`; it is what the
#: fleet project of :data:`RUNNER_HOST` passes, from the one node it is pinned to.
RUNNER_HOST_OPTION: Final[str] = "--runner-host"

#: The fleet project that runs each capability's host cases.
PROJECTS: Final[dict[str, str]] = {
    "windows": "tools/fleet-execution",
    "linux": "tools/fleet-execution-linux",
    RUNNER_HOST: "tools/fleet-execution-runners",
}

#: The user property a host case records what it read on the host under,
#: which the run's summary prints so a passed row carries the reading.
READING_PROPERTY: Final[str] = "host_reading"


def command_output(argv: list[str], *, timeout: int = 60) -> str:
    """Run a command on the host and return its standard output.

    Args:
        argv: The command.
        timeout: Its deadline in seconds.

    Returns:
        What it printed.
    """
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout).stdout


def unit_properties(unit: str, names: tuple[str, ...]) -> dict[str, str]:
    """Some of a system unit's properties, as systemd reads them.

    Args:
        unit: The unit.
        names: The properties.

    Returns:
        Each property's value, by name.
    """
    shown = [flag for name in names for flag in ("-p", name)]
    text = command_output(["systemctl", "show", unit, *shown])
    return dict(line.split("=", 1) for line in text.splitlines() if line)


#: The roster this checkout carries.
ROSTER: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent.parent / "runners.json"


@pytest.fixture(name="runner_host")
def _runner_host(request: pytest.FixtureRequest) -> HostRunnerSpec:
    """The roster's entry for the runner host this run executes on.

    Args:
        request: The case's request, whose run names the host.

    Returns:
        The host's spec.
    """
    name: str | None = request.config.getoption(RUNNER_HOST_OPTION)
    hosts = [host for host in load_runner_spec(str(ROSTER))["hosts"] if host["name"] == name]
    assert len(hosts) == 1, f"runners.json declares {len(hosts)} host(s) named {name!r}"
    return hosts[0]


class HostReadings:
    """Prints each passed case's host readings, on the process that reports the run.

    Attributes:
        lines: Every reading recorded so far, in report order.
    """

    def __init__(self) -> None:
        """Start from no reading."""
        self.lines: list[str] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Keep the readings a passed case recorded.

        Args:
            report: One phase's report.
        """
        if report.passed and report.when == "call":
            self.lines.extend(
                f"host reading: {value}"
                for name, value in report.user_properties
                if name == READING_PROPERTY
            )

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        """Write the readings.

        Args:
            terminalreporter: The session's terminal reporter.
        """
        for line in self.lines:
            terminalreporter.write_line(line)


#: Where each run keeps its plan, built once its options are read.
PLAN_KEY: Final[pytest.StashKey[host_execution.HostExecutionPlan]] = pytest.StashKey()


def plan_for(here: str) -> host_execution.HostExecutionPlan:
    """This package's host cases, as platform_core's host-execution run reads them.

    Args:
        here: The capability this run has.

    Returns:
        The plan.
    """
    return host_execution.HostExecutionPlan(
        markers={marker: platform for platform, marker in MARKERS.items()},
        execution_only={EXECUTION_ONLY_MARKER: "linux", RUNNER_HOST_MARKER: RUNNER_HOST},
        needs={
            EXECUTION_ONLY_MARKER: "needs a node's execdocker user and its rootless daemon",
            RUNNER_HOST_MARKER: "needs a host that runs the fleet's WSL runner units",
        },
        projects=PROJECTS,
        here=here,
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the execution option and the runner host's.

    Args:
        parser: pytest's option parser.
    """
    host_execution.add_option(parser)
    parser.addoption(
        RUNNER_HOST_OPTION,
        default=None,
        help="the runners.json host this run executes on, making it that host's execution run",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Build the run's plan, declare its markers, and count an execution run.

    Args:
        config: The run's configuration.
    """
    host: str | None = config.getoption(RUNNER_HOST_OPTION)
    plan = plan_for(HOST_PLATFORM if host is None else RUNNER_HOST)
    config.stash[PLAN_KEY] = plan
    host_execution.configure(config, plan)
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(HostReadings(), "host-readings")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Select the execution run's cases, or skip what this run cannot run.

    Args:
        config: The run's configuration.
        items: The collected cases, narrowed in place for an execution run.
    """
    host_execution.select(config, items, config.stash[PLAN_KEY])
