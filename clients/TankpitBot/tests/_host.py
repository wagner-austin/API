"""The cases that watch the public demo play, and the run that proves they did.

Board task 46934cd6, whose review asked for checks that "measure the live
service rather than file contents". Every other case in this suite drives
the demo's routes, its captions and its HLS capture against fakes and a
local server, so the suite proves the code builds the stream it means to
and never that austinwagner.org/tankpit, with a real bot behind it, is
serving sound and captions now. ``host_live_demo`` binds a case to what can
say so: the live demo service and the published page, reached over the
internet, ``ffmpeg`` and ``ffprobe`` to read the segments it served, and the
installed Microsoft Edge to play the page as a visitor does
(``tests/live/_visitor.py`` says why Playwright's own Chromium cannot).

They are EXECUTION-ONLY (platform_core's :mod:`~platform_core.host_execution`):
the ordinary ``make check`` skips them on every machine, because that run is
also API's CI, where pressing the public spawn button on every push would
start a stranger-visible bot for each one. ``make execution`` runs only these
cases and fails on a skip or on none, and that is the check of the fleet
project ``clients/TankpitBot-execution``, which requires the ``windows`` and
``ffmpeg`` tags; every Windows node ships Edge.
"""

from __future__ import annotations

import sys
from typing import Final

import pytest
from platform_core import host_execution

#: The marker binding a case to the live demo, a node's ffmpeg and its Edge.
LIVE_DEMO_MARKER: Final[str] = "host_live_demo"

#: The fleet project whose check is this package's execution run.
PROJECT: Final[str] = "clients/TankpitBot-execution"

#: This package's host cases, as platform_core's host-execution run reads them.
PLAN: Final[host_execution.HostExecutionPlan] = host_execution.HostExecutionPlan(
    markers={},
    execution_only={LIVE_DEMO_MARKER: "windows"},
    needs={LIVE_DEMO_MARKER: "needs the live demo service, a node's ffmpeg and its Edge"},
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
