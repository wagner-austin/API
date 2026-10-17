"""Market tests read no host's evidence relocations unless a test declares its own."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from covenant_ml.markets import _test_hooks


@pytest.fixture(autouse=True)
def no_host_relocations() -> Generator[None, None, None]:
    """Keep each test independent of ``~/.config/covenant_ml/evidence-relocations.json``."""
    original = _test_hooks.evidence_relocations
    _test_hooks.evidence_relocations = lambda: ()
    yield
    _test_hooks.evidence_relocations = original
