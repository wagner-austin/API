from __future__ import annotations

import asyncio

import pytest
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, MessageProto, UserProto
from platform_discord.trainer.handler import TrainerEventV1
from platform_discord.trainer.runtime import RequestAction, TrainerRuntime
from tests.support.discord_fakes import FakeEmbed

from clubbot.services.jobs import trainer_notifier
from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


def _make_config_event() -> TrainerEventV1:
    """Create a valid TrainerConfigV1 event for testing."""
    from platform_core.trainer_metrics_events import TrainerConfigV1

    event: TrainerConfigV1 = {
        "type": "trainer.metrics.config.v1",
        "job_id": "test-config-job",
        "user_id": 1,
        "model_family": "test",
        "model_size": "small",
        "total_epochs": 1,
        "queue": "default",
    }
    return event


class _BadBot:
    async def fetch_user(self, user_id: int, /) -> UserProto:
        _ = user_id

        class _BadUser:
            @property
            def id(self) -> int:
                return 1

            async def send(
                self,
                content: str | None = None,
                *,
                embed: EmbedProto | None = None,
                file: FileProto | None = None,
            ) -> MessageProto:
                _ = (content, embed, file)
                raise RuntimeError("send failed")

        u: UserProto = _BadUser()
        return u


def test_trainer_notifier_handle_event_with_config() -> None:
    """Test _handle_event processes config events (they trigger notifications)."""
    sub = TrainerEventSubscriber(bot=_BadBot(), redis_url="redis://example")
    ev: TrainerEventV1 = _make_config_event()

    async def run() -> None:
        # Config events trigger notify(), which will raise because _BadBot.send raises
        with pytest.raises(RuntimeError, match="send failed"):
            await sub._handle_event(ev)

    asyncio.run(run())


@pytest.mark.asyncio
async def test_trainer_notifier_notify_send_failure_raises() -> None:
    """Test that send() failures propagate from notify()."""
    sub = TrainerEventSubscriber(bot=_BadBot(), redis_url="redis://example")
    with pytest.raises(RuntimeError, match="send failed"):
        await sub.notify(1, "r", FakeEmbed(title="t"))


class _GoodBot:
    """Bot that doesn't raise on send (not actually called in this test)."""

    async def fetch_user(self, user_id: int, /) -> UserProto:
        _ = user_id

        class _GoodUser:
            @property
            def id(self) -> int:
                return 1

            async def send(
                self,
                content: str | None = None,
                *,
                embed: EmbedProto | None = None,
                file: FileProto | None = None,
            ) -> MessageProto:
                _ = (content, embed, file)
                raise AssertionError("send should not be called when act is None")

        u: UserProto = _GoodUser()
        return u


@pytest.mark.asyncio
async def test_trainer_notifier_handle_event_with_none_action() -> None:
    """Test _handle_event skips notify when handle_trainer_event returns None.

    This covers the 'if act is not None' branch (line 78->exit) when act is None.
    """
    original_hook = trainer_notifier.handle_trainer_event

    def _return_none(runtime: TrainerRuntime, event: TrainerEventV1) -> RequestAction | None:
        _ = (runtime, event)
        return None

    trainer_notifier.handle_trainer_event = _return_none
    try:
        sub = TrainerEventSubscriber(bot=_GoodBot(), redis_url="redis://example")
        ev: TrainerEventV1 = _make_config_event()
        # Should complete without calling notify (no exception)
        await sub._handle_event(ev)
    finally:
        trainer_notifier.handle_trainer_event = original_hook
