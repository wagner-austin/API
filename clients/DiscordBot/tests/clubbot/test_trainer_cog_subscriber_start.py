from __future__ import annotations

from tests.support.discord_fakes import FakeBot
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.cogs.trainer import TrainerCog


class _MockSubscriber:
    """A fake subscriber for testing."""

    def __init__(self) -> None:
        self.n = 0

    def start(self) -> None:
        self.n += 1


def test_trainer_ensure_subscriber_started_invokes_start() -> None:
    """Test that ensure_subscriber_started() starts a pre-created subscriber.

    With autostart_subscriber=True, the factory creates and starts the subscriber.
    Then ensure_subscriber_started() can be called to start it again (idempotent).
    """
    cfg = build_settings(
        model_trainer_api_url="https://example",
        model_trainer_api_key=None,
        model_trainer_api_timeout_seconds=10,
        redis_url="redis://example",
    )

    fake_sub = _MockSubscriber()

    def _factory(
        *, bot: _test_hooks.BotProto, redis_url: str, events_channel: str
    ) -> _test_hooks.TrainerEventSubscriberLike:
        _ = (bot, redis_url, events_channel)
        result: _test_hooks.TrainerEventSubscriberLike = fake_sub
        return result

    original = _test_hooks.trainer_event_subscriber_factory
    _test_hooks.trainer_event_subscriber_factory = _factory
    try:
        # Build WITH autostart so the factory is called and subscriber is created
        cog = TrainerCog(bot=FakeBot(), config=cfg, autostart_subscriber=True)
        # Factory was called, subscriber was started in __init__
        assert fake_sub.n == 1
        # Calling ensure_subscriber_started again increments again (idempotent call)
        cog.ensure_subscriber_started()
        assert fake_sub.n == 2
    finally:
        _test_hooks.trainer_event_subscriber_factory = original


def test_trainer_autostart_with_none_bot_noop() -> None:
    cfg = build_settings(
        model_trainer_api_url="https://example",
        model_trainer_api_key=None,
        model_trainer_api_timeout_seconds=10,
        redis_url="redis://example",
    )
    _ = TrainerCog(bot=None, config=cfg)


def test_trainer_ensure_subscriber_started_noop_on_none() -> None:
    cfg = build_settings(
        model_trainer_api_url="https://example",
        model_trainer_api_key=None,
        model_trainer_api_timeout_seconds=10,
        redis_url=None,
    )
    cog = TrainerCog(bot=FakeBot(), config=cfg, autostart_subscriber=False)
    cog.ensure_subscriber_started()
