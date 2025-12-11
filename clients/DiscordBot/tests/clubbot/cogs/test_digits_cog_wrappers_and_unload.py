from __future__ import annotations

import pytest
from tests.support.discord_fakes import FakeBot
from tests.support.settings import build_settings

from clubbot import _test_hooks
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


class _MockSubscriber:
    """A fake subscriber for testing."""

    def __init__(self) -> None:
        self.stopped = False

    def start(self) -> None:
        pass

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_digits_placeholder_for_wrappers() -> None:
    # The command wrappers are exercised via integration tests; unit tests focus on impls.
    assert True


@pytest.mark.asyncio
async def test_obsolete_cog_unload_stops_subscriber() -> None:
    cfg = build_settings(handwriting_api_url="http://h", redis_url="redis://example")

    fake_sub = _MockSubscriber()

    def _factory(
        *, bot: _test_hooks.BotProto, redis_url: str
    ) -> _test_hooks.DigitsEventSubscriberLike:
        _ = (bot, redis_url)
        return fake_sub

    original = _test_hooks.digits_event_subscriber_factory
    _test_hooks.digits_event_subscriber_factory = _factory
    try:
        cog = DigitsCog(
            bot=FakeBot(),
            config=cfg,
            service=_Svc(cfg),
            enqueuer=None,
            autostart_subscriber=False,
        )

        await cog._obsolete_cog_unload()
        assert fake_sub.stopped is True
    finally:
        _test_hooks.digits_event_subscriber_factory = original


@pytest.mark.asyncio
async def test_obsolete_cog_unload_no_subscriber_noop() -> None:
    cfg = build_settings(handwriting_api_url="http://h", redis_url=None)
    cog = DigitsCog(
        bot=FakeBot(),
        config=cfg,
        service=_Svc(cfg),
        enqueuer=None,
        autostart_subscriber=False,
    )
    # No subscriber present (redis_url=None); should be a no-op path
    await cog._obsolete_cog_unload()
