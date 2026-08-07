"""Diagnostic-module isolation fixtures.

Every diagnostic emitter holds module-level gate/counter state that
persists across tests in the same process. Without explicit resets a
test can see leftover state from the previous test and fail or pass
depending on ordering.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _isolate_diagnostic_emitters() -> Generator[None, None, None]:
    """Reset the remaining diagnostic module globals before each test.

    The two alignment emitters are gone from this list: their gates are
    instance state on ``SelfAlignmentEmitter`` / ``EntityAlignmentEmitter``
    as of step 3, so a test that wants a clear gate constructs one. The
    teleport dispatch tracker left for the same reason in step 6 -- it
    is now ``WorldService.ledger.pending_teleport``, reset with the
    world ([[session-state-deglobalisation]]).
    """
    from tankpit_bot.diagnostics.registry_truth import reset_registry_truth

    reset_registry_truth()

    yield

    reset_registry_truth()
