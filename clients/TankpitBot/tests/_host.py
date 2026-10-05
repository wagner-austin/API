"""The case that watches the public demo play, and the run that proves it did.

Board task 46934cd6, whose review asked for checks that "measure the live
service rather than file contents". Every other case in this suite drives
the demo's routes, its captions and its HLS capture against fakes and a
local server, so the suite proves the code builds the stream it means to
and never that austinwagner.org/tankpit, with a real bot behind it, is
serving sound and captions now. ``host_live_demo`` binds a case to the one
thing that can say so: the live demo service, reached over the internet,
and ``ffprobe`` to read a segment it served.

It is EXECUTION-ONLY (platform_core's :mod:`~platform_core.host_execution`):
the ordinary ``make check`` skips it on every machine, because that run is
also API's CI, where pressing the public spawn button on every push would
start a stranger-visible bot for each one. ``make execution`` runs only this
case and fails on a skip or on none, and that is the check of the fleet
project ``clients/TankpitBot-execution``, which requires the ``windows`` and
``ffmpeg`` tags.
"""

from __future__ import annotations

import sys
from typing import Final

import pytest
from platform_core import host_execution

#: The marker binding a case to the live demo service and a node's ffprobe.
LIVE_DEMO_MARKER: Final[str] = "host_live_demo"

#: The fleet project whose check is this package's execution run.
PROJECT: Final[str] = "clients/TankpitBot-execution"

#: This package's host cases, as platform_core's host-execution run reads them.
PLAN: Final[host_execution.HostExecutionPlan] = host_execution.HostExecutionPlan(
    markers={},
    execution_only={LIVE_DEMO_MARKER: "windows"},
    needs={LIVE_DEMO_MARKER: "needs the live demo service and a node's ffprobe"},
    projects={"windows": PROJECT},
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
    """Select the execution run's case, or skip it in the ordinary run.

    Args:
        config: The run's configuration.
        items: The collected cases, narrowed in place for an execution run.
    """
    host_execution.select(config, items, PLAN)
