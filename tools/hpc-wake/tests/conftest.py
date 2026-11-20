"""The pump's one shared fixture: the recording process runner.

Everything here is a FAKE implementing the production seam, never a mock.
The one seam rebound is ``scripts._test_hooks.run_process``, through
:func:`tests.run_cycle_tree.make_runner`, which puts the real runner back
after each test.
"""

from __future__ import annotations

import pytest

from tests import run_cycle_tree

runner = pytest.fixture(run_cycle_tree.make_runner)
