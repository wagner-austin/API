from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from platform_core.trainer_metrics_events import make_config_event
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, MessageProto, UserProto
from platform_discord.trainer.handler import TrainerEventV1
from tests.support.discord_fakes import FakeEmbed

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


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


def test_trainer_notifier_handles_unknown_event_branch() -> None:
    """Test that unknown events are handled gracefully (handle_trainer_event returns None)."""
    sub = TrainerEventSubscriber(bot=_BadBot(), redis_url="redis://example")
    # Create a valid config event but patch handle_trainer_event to return None
    # to simulate an unknown/unhandled event type
    ev: TrainerEventV1 = make_config_event(
        job_id="x",
        user_id=1,
        model_family="gpt2",
        model_size="small",
        total_epochs=1,
        queue="q",
    )

    async def run() -> None:
        # Patch handle_trainer_event to return None (simulating unknown event)
        with patch(
            "clubbot.services.jobs.trainer_notifier.handle_trainer_event",
            return_value=None,
        ):
            await sub._handle_event(ev)

    asyncio.run(run())


@pytest.mark.asyncio
async def test_trainer_notifier_notify_send_failure_raises() -> None:
    """Test that send() failures propagate from notify()."""
    sub = TrainerEventSubscriber(bot=_BadBot(), redis_url="redis://example")
    with pytest.raises(RuntimeError, match="send failed"):
        await sub.notify(1, "r", FakeEmbed(title="t"))
