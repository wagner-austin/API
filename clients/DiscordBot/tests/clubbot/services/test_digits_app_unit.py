from __future__ import annotations

import logging

import pytest
from tests.support.settings import build_settings

from clubbot.config import DiscordbotSettings
from clubbot.services.digits.app import DigitService
from clubbot.services.handai.client import PredictResult


def _cfg(url: str | None, mb: int = 2) -> DiscordbotSettings:
    return build_settings(
        handwriting_api_url=url,
        handwriting_api_timeout_seconds=5,
        handwriting_api_max_retries=1,
        digits_max_image_mb=mb,
    )


def test_digit_service_requires_base_url() -> None:
    # Empty string disables default and results in missing config
    with pytest.raises(RuntimeError):
        DigitService(_cfg(""))


@pytest.mark.asyncio
async def test_digit_service_max_bytes_and_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        async def read_digit(
            self,
            *,
            data: bytes,
            filename: str,
            content_type: str,
            request_id: str,
            center: bool,
            visualize: bool,
        ) -> PredictResult:
            # Echo a minimal PredictResult-like structure for assertions
            return PredictResult(
                digit=7,
                confidence=0.5,
                probs=(),
                model_id="m",
                uncertain=False,
                latency_ms=1,
            )

    svc = DigitService(_cfg("http://localhost", mb=3), client=_Client())
    # Property uses MB → bytes conversion
    assert svc.max_image_bytes == 3 * 1024 * 1024
    out = await svc.read_image(
        data=b"img", filename="x.png", content_type="image/png", request_id="r"
    )
    assert out.digit == 7


logger = logging.getLogger(__name__)
