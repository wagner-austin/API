"""The cases that deploy a service for real, and the run that proves they did.

Board task 465689f5, rule R3 of b6eb30c8. Every other case in this suite
drives :func:`maketools.workspace.compose_up` against recorded fakes, so the
suite proves the package builds the commands it means to and never that a
service's image builds, its entry starts, or its healthcheck passes.
``host_linux_docker`` binds a case to the one thing that can say so: a Linux
node's execdocker user and its rootless daemon, which cannot reach the
stack's.

It is EXECUTION-ONLY (platform_core's :mod:`~platform_core.host_execution`):
the ordinary ``make check`` skips it on every machine, because that run is
also API's CI and the hub, whose daemon runs the phone stack. ``make
execution`` runs only these cases and fails on a skip or on none, and that
is the check of the fleet project ``tools/maketools-execution``, which
requires the ``linux`` and ``docker`` tags, so the fleet builds it through
the isolated script as execdocker and never on a CI runner.
"""

from __future__ import annotations

import sys
from typing import Final

import pytest
from platform_core import host_execution

#: The marker binding a case to a Linux node's rootless execution daemon.
DOCKER_MARKER: Final[str] = "host_linux_docker"

#: The fleet project whose check is this package's execution run.
PROJECT: Final[str] = "tools/maketools-execution"

#: This package's host cases, as platform_core's host-execution run reads them.
PLAN: Final[host_execution.HostExecutionPlan] = host_execution.HostExecutionPlan(
    markers={},
    execution_only={DOCKER_MARKER: "linux"},
    needs={DOCKER_MARKER: "needs a node's execdocker user and its rootless daemon"},
    projects={"linux": PROJECT},
    here="windows" if sys.platform == "win32" else "linux",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the execution option.

    Args:
        parser: pytest's option parser.
    """
    host_execution.add_option(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Declare the marker, and count an execution run on the reporting process.

    Args:
        config: The run's configuration.
    """
    host_execution.configure(config, PLAN)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Select the execution run's cases, or skip them in the ordinary run.

    Args:
        config: The run's configuration.
        items: The collected cases, narrowed in place for an execution run.
    """
    host_execution.select(config, items, PLAN)
