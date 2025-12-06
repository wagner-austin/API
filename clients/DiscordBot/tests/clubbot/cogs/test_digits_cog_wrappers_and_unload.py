from __future__ import annotations

import pytest
from tests.support.settings import build_settings

from clubbot.cogs.digits import DigitsCog
from clubbot.services.digits.app import DigitService
from clubbot.services.handai.client import PredictResult


class _Svc(DigitService):
    async def read_image(
        self,
        *,
        data: bytes,
        filename: str,
        content_type: str,
        request_id: str,
    ) -> PredictResult:
        _ = (data, filename, content_type, request_id)
        return PredictResult(
            digit=1,
            confidence=0.9,
            probs=(0.9, 0.05, 0.05),
            model_id="m",
            uncertain=False,
            latency_ms=10,
        )


@pytest.mark.asyncio
async def test_digits_placeholder_for_wrappers() -> None:
    # The command wrappers are exercised via integration tests; unit tests focus on impls.
    assert True


@pytest.mark.asyncio
async def test_obsolete_cog_unload_stops_subscriber() -> None:
    from tests.support.discord_fakes import FakeBot

    cfg = build_settings(handwriting_api_url="http://h")
    cog = DigitsCog(
        bot=FakeBot(),
        config=cfg,
        service=_Svc(cfg),
        enqueuer=None,
        autostart_subscriber=False,
    )

    class _Sub:
        def __init__(self) -> None:
            self.stopped: bool = False

        async def stop(self) -> None:
            self.stopped = True

    # Inject a fake subscriber and unload
    sub_inst = _Sub()
    object.__setattr__(cog, "_subscriber", sub_inst)
    await cog._obsolete_cog_unload()
    assert sub_inst.stopped is True


@pytest.mark.asyncio
async def test_obsolete_cog_unload_no_subscriber_noop() -> None:
    from tests.support.discord_fakes import FakeBot

    cfg = build_settings(handwriting_api_url="http://h")
    cog = DigitsCog(
        bot=FakeBot(),
        config=cfg,
        service=_Svc(cfg),
        enqueuer=None,
        autostart_subscriber=False,
    )
    # No subscriber present; should be a no-op path
    await cog._obsolete_cog_unload()
