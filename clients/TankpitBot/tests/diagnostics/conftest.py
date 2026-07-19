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
    """Reset all diagnostic emitter module globals before each test."""
    from tankpit_bot.diagnostics.entity_alignment import reset_entity_alignment_emitter
    from tankpit_bot.diagnostics.registry_truth import reset_registry_truth
    from tankpit_bot.diagnostics.self_alignment import reset_self_alignment_emitter
    from tankpit_bot.ledger.outcome.teleport import reset_teleport_dispatch_tracking

    reset_entity_alignment_emitter()
    reset_self_alignment_emitter()
    reset_teleport_dispatch_tracking()
    reset_registry_truth()

    yield

    reset_entity_alignment_emitter()
    reset_self_alignment_emitter()
    reset_teleport_dispatch_tracking()
    reset_registry_truth()
