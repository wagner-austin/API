"""Shared fixtures for the ensemble tests.

Several tests in this package rebind ``covenant_ml.ensemble._hooks.minimize``
to :func:`covenant_ml.ensemble.testing.fake_minimize` so the optimiser can be
driven without scipy. They did not restore it, and the leak was not local:
``_hooks`` is module state shared by every test in the same worker process, so
once one test replaced the seam, every later test in that worker saw the fake.

That made ``TestMinimizeBinding::test_the_seam_is_bound_to_scipy`` -- whose
whole job is to assert the seam reaches the real solver with nothing wired --
pass or fail depending on which modules ``pytest-xdist`` happened to place on
its worker. It is an order-dependent failure, so it stayed hidden until the
distribution changed.

Restoring after every test makes the binding assertion deterministic and holds
the package to the rule its own hook modules state: rebind, run, restore.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from covenant_ml.ensemble import _hooks


@pytest.fixture(autouse=True)
def _restore_minimize_seam() -> Generator[None, None, None]:
    """Restore the solver seam to its real implementation after each test.

    Yields:
        Control to the test, then rebinds the seam.
    """
    original = _hooks.minimize
    yield
    _hooks.minimize = original
