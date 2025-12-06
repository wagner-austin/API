from __future__ import annotations

from platform_core.queues import (
    DIGITS_QUEUE,
    MUSIC_WRAPPED_QUEUE,
    TRAINER_QUEUE,
    TRANSCRIPT_QUEUE,
    TURKIC_QUEUE,
)


def test_queue_constants() -> None:
    assert DIGITS_QUEUE == "digits"
    assert MUSIC_WRAPPED_QUEUE == "music_wrapped"
    assert TRAINER_QUEUE == "trainer"
    assert TRANSCRIPT_QUEUE == "transcript"
    assert TURKIC_QUEUE == "turkic"
