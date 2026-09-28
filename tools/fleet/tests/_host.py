"""Cases that execute this package's rendered scripts on a real host.

Board task 465689f5, rule R3 of b6eb30c8. A handful of cases here run the
PowerShell or sh this package renders on the machine running the suite,
which is the only place a defect between the code and the machine can
show. Each is bound to one platform, and until this module each carried a
``skipif`` on ``sys.platform``: API's CI runs on Linux, so every Windows
case was skipped there, and the hub is Windows, so every Linux case was
skipped at home. No run recorded both halves green at one commit.

So the binding is a marker, ``host_windows`` or ``host_linux``, and there
are two ways to run the suite:

- The ORDINARY run (``make check``) behaves as the ``skipif`` did: a host
  case bound to the other platform is skipped, naming the node that runs it.
- The EXECUTION run (``--host-execution``, ``make execution``) selects only
  the host cases bound to this machine's platform and FAILS when any of them
  skips or when none ran. It is what the fleet projects
  ``tools/fleet-execution`` (a Windows node) and
  ``tools/fleet-execution-linux`` (a Linux node) run, and a deploy refuses a
  commit without a passed run of each (MCPs' ``publish-executed``).

Two markers rather than one with an argument, because a marker's arguments
are untyped and this package's mypy settings refuse an expression of type
Any in tests as in src.

A THIRD, ``host_linux_docker``, binds a case to Linux like ``host_linux`` but
runs it ONLY in the execution run: it executes a docker project's build as
the node's execdocker user against that user's rootless daemon (MCPs board
task a8ee9b21), which exists on a node carrying the ``docker`` tag and on no
CI runner. The ordinary run skips it on every platform, naming the project
that runs it, and the Linux execution run requires it like any host case,
which is why ``tools/fleet-execution-linux`` requires the ``docker`` tag.
"""

from __future__ import annotations

import sys
from typing import Final

import pytest

#: This machine's platform, as the markers and the fleet's tags name it.
HOST_PLATFORM: Final[str] = "windows" if sys.platform == "win32" else "linux"

#: The marker binding a case to each platform.
MARKERS: Final[dict[str, str]] = {"windows": "host_windows", "linux": "host_linux"}

#: The marker for a Linux host case that needs the node's rootless daemon,
#: run only by the execution run.
EXECUTION_ONLY_MARKER: Final[str] = "host_linux_docker"

#: The fleet project that runs each platform's host cases.
PROJECTS: Final[dict[str, str]] = {
    "windows": "tools/fleet-execution",
    "linux": "tools/fleet-execution-linux",
}

#: The command-line option that makes a run the execution run.
EXECUTION_OPTION: Final[str] = "--host-execution"


def bound_platform(item: pytest.Item) -> str | None:
    """The platform a case is bound to, if any.

    Args:
        item: A collected case.

    Returns:
        ``"windows"`` or ``"linux"`` for a host case, else None.
    """
    if item.get_closest_marker(EXECUTION_ONLY_MARKER) is not None:
        return "linux"
    for platform, marker in MARKERS.items():
        if item.get_closest_marker(marker) is not None:
            return platform
    return None


class ExecutionTally:
    """Counts what an execution run did, on the process that reports it.

    Attributes:
        passed: Host cases whose call passed.
        skipped: Host cases that skipped in any phase.
    """

    def __init__(self) -> None:
        """Start from nothing run."""
        self.passed = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Count one phase's outcome.

        Args:
            report: The phase's report.
        """
        if report.skipped:
            self.skipped += 1
        elif report.passed and report.when == "call":
            self.passed += 1

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Fail the run unless every selected host case ran and at least one did.

        Args:
            session: The finished session.
        """
        print(
            f"\nhost execution on {HOST_PLATFORM}: {self.passed} case(s) ran, "
            f"{self.skipped} skipped"
        )
        if self.skipped or not self.passed:
            print(
                "HOST_EXECUTION_INCOMPLETE: an execution run must run every host case "
                f"bound to {HOST_PLATFORM} and at least one; a skip is not a run"
            )
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the execution option.

    Args:
        parser: pytest's option parser.
    """
    parser.addoption(
        EXECUTION_OPTION,
        action="store_true",
        help="run only the host cases bound to this platform, failing on a skip or on none",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Declare the markers, and count an execution run on the reporting process.

    Args:
        config: The run's configuration.
    """
    for platform, marker in MARKERS.items():
        config.addinivalue_line(
            "markers", f"{marker}: executes this package's scripts on a {platform} host"
        )
    config.addinivalue_line(
        "markers",
        f"{EXECUTION_ONLY_MARKER}: executes a docker project's build as execdocker on a "
        "linux node carrying the docker tag; only the execution run runs it",
    )
    execution: bool = config.getoption(EXECUTION_OPTION)
    if execution and not hasattr(config, "workerinput"):
        config.pluginmanager.register(ExecutionTally(), "host-execution-tally")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Select the execution run's cases, or skip the other platform's.

    Args:
        config: The run's configuration.
        items: The collected cases, reordered or narrowed in place.
    """
    execution: bool = config.getoption(EXECUTION_OPTION)
    if execution:
        kept = [item for item in items if bound_platform(item) == HOST_PLATFORM]
        dropped = [item for item in items if bound_platform(item) != HOST_PLATFORM]
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept
        return
    for item in items:
        if item.get_closest_marker(EXECUTION_ONLY_MARKER) is not None:
            item.add_marker(
                pytest.mark.skip(
                    reason="needs a node's execdocker user and its rootless daemon; "
                    f"{PROJECTS['linux']} runs it there"
                )
            )
            continue
        platform = bound_platform(item)
        if platform is not None and platform != HOST_PLATFORM:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"executes on a {platform} host; {PROJECTS[platform]} runs it there"
                )
            )
