from __future__ import annotations

from platform_core.queues import (
    COVENANT_QUEUE,
    DATA_BANK_QUEUE,
    DIGITS_QUEUE,
    MUSIC_WRAPPED_QUEUE,
    QR_QUEUE,
    TRAINER_QUEUE,
    TRANSCRIPT_QUEUE,
    TURKIC_QUEUE,
)


def test_queue_constants() -> None:
    assert COVENANT_QUEUE == "covenant"
    assert DATA_BANK_QUEUE == "data_bank"
    assert DIGITS_QUEUE == "digits"
    assert MUSIC_WRAPPED_QUEUE == "music_wrapped"
    assert QR_QUEUE == "qr"
    assert TRAINER_QUEUE == "trainer"
    assert TRANSCRIPT_QUEUE == "transcript"
    assert TURKIC_QUEUE == "turkic"
