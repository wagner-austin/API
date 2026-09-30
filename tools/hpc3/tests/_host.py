"""The case that asks the real HPC3 scheduler, and the run that proves it did.

Board task 465689f5, rule R3 of b6eb30c8. Every other case in this suite
drives the submission path against :class:`tests.conftest.FakeRun`, so the
suite proves the package builds the commands it means to and never that the
cluster accepts them. ``host_hpc3`` binds a case to the one thing that can
say so: an ssh session to HPC3 through its jump host, from a node that holds
that route.

It is EXECUTION-ONLY (platform_core's :mod:`~platform_core.host_execution`):
the ordinary ``make check`` skips it on every machine, because that run is
also API's CI, which has no route to the cluster. ``make execution`` runs
only it and fails on a skip or on none, and that is the check of the fleet
project ``tools/hpc3-execution``, which requires the ``hpc3`` capability tag
a node carries only when it holds the route.
"""

from __future__ import annotations

from typing import Final

import pytest
from platform_core import host_execution

#: The marker binding a case to the ssh route to HPC3.
HPC3_MARKER: Final[str] = "host_hpc3"

#: The fleet project whose check is this package's execution run.
PROJECT: Final[str] = "tools/hpc3-execution"

#: This package's host case, as platform_core's host-execution run reads it.
PLAN: Final[host_execution.HostExecutionPlan] = host_execution.HostExecutionPlan(
    markers={},
    execution_only={HPC3_MARKER: "hpc3"},
    needs={HPC3_MARKER: "needs a node with the ssh route to HPC3 through its jump host"},
    projects={"hpc3": PROJECT},
    here="hpc3",
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
    """Select the execution run's case, or skip it in the ordinary run.

    Args:
        config: The run's configuration.
        items: The collected cases, narrowed in place for an execution run.
    """
    host_execution.select(config, items, PLAN)
