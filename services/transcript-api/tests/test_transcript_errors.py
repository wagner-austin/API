from __future__ import annotations

import pytest
from platform_core.errors import AppError, TranscriptErrorCode
from platform_core.logging import get_logger


def test_app_error_is_exception() -> None:
    with pytest.raises(AppError):
        raise AppError(TranscriptErrorCode.TRANSCRIPT_UNAVAILABLE, "invalid input", 400)


def test_logger_configured() -> None:
    logger = get_logger("transcript-api-test")
    # Verify logger has expected methods by accessing them
    assert callable(logger.info)
    assert callable(logger.error)
