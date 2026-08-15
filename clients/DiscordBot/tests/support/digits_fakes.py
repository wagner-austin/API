"""Fakes for the digits service used by the digits cog's tests.

FakeDigitService answers with a fixed prediction; RejectingDigitService raises
the oversized-image error the cog is expected to surface to the user.
"""

from __future__ import annotations

from platform_core.errors import AppError, ErrorCode

from clubbot.services.digits.app import DigitService
from clubbot.services.handai.client import PredictResult


class FakeDigitService(DigitService):
    """DigitService fake returning a predictable prediction."""

    def __init__(self, max_mb: int = 2) -> None:
        self._max_image_mb = max_mb

    @property
    def max_image_bytes(self) -> int:
        return self._max_image_mb * 1024 * 1024

    async def read_image(
        self, *, data: bytes, filename: str, content_type: str, request_id: str
    ) -> PredictResult:
        _ = (data, filename, content_type, request_id)
        return PredictResult(
            digit=3,
            confidence=0.9,
            probs=(0.9, 0.05, 0.05),
            model_id="m",
            uncertain=False,
            latency_ms=10,
        )


class RejectingDigitService(FakeDigitService):
    """DigitService fake that raises a specific exception."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    async def read_image(
        self, *, data: bytes, filename: str, content_type: str, request_id: str
    ) -> PredictResult:
        _ = (data, filename, content_type, request_id)
        raise self._error


class TooLargeError(AppError[ErrorCode]):
    """The oversized-image rejection RejectingDigitService is handed to raise."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.INVALID_INPUT, "Image is too large", http_status=400)
