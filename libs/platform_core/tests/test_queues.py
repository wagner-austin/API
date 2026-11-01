from __future__ import annotations

from platform_core.queues import (
    COVENANT_QUEUE,
    TRAINER_QUEUE,
    TRANSCRIPT_QUEUE,
)


def test_queue_constants() -> None:
    assert COVENANT_QUEUE == "covenant"
    assert TRAINER_QUEUE == "trainer"
    assert TRANSCRIPT_QUEUE == "transcript"
