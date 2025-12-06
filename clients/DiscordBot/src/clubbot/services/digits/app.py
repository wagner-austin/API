from __future__ import annotations

from typing import TypedDict

from ...config import DiscordbotSettings
from ..handai.client import HandwritingClient, HandwritingReader, PredictResult


class DigitServiceConfig(TypedDict):
    base_url: str
    api_key: str | None
    timeout_seconds: int
    max_retries: int


class DigitService:
    def __init__(self, cfg: DiscordbotSettings, client: HandwritingReader | None = None) -> None:
        if not cfg["handwriting"]["api_url"]:
            raise RuntimeError("HANDWRITING_API_URL is not configured")
        self._client: HandwritingReader = client or HandwritingClient(
            base_url=cfg["handwriting"]["api_url"],
            api_key=cfg["handwriting"]["api_key"],
            timeout_seconds=cfg["handwriting"]["api_timeout_seconds"],
            max_retries=cfg["handwriting"]["api_max_retries"],
        )
        self._max_image_mb: int = cfg["digits"]["max_image_mb"]

    @property
    def max_image_bytes(self) -> int:
        return self._max_image_mb * 1024 * 1024

    async def read_image(
        self, *, data: bytes, filename: str, content_type: str, request_id: str
    ) -> PredictResult:
        return await self._client.read_digit(
            data=data,
            filename=filename,
            content_type=content_type,
            request_id=request_id,
            center=True,
            visualize=False,
        )
