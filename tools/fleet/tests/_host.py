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
"""

from __future__ import annotations

import sys
from typing import Final

import pytest
from platform_core import host_execution

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

#: This package's host cases, as platform_core's host-execution run reads them.
PLAN: Final[host_execution.HostExecutionPlan] = host_execution.HostExecutionPlan(
    markers={marker: platform for platform, marker in MARKERS.items()},
    execution_only={EXECUTION_ONLY_MARKER: "linux"},
    needs={EXECUTION_ONLY_MARKER: "needs a node's execdocker user and its rootless daemon"},
    projects=PROJECTS,
    here=HOST_PLATFORM,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the execution option.

    Args:
        parser: pytest's option parser.
    """
    host_execution.add_option(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Declare the markers, and count an execution run on the reporting process.

    Args:
        config: The run's configuration.
    """
    host_execution.configure(config, PLAN)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Select the execution run's cases, or skip what this run cannot run.

    Args:
        config: The run's configuration.
        items: The collected cases, narrowed in place for an execution run.
    """
    host_execution.select(config, items, PLAN)
