from __future__ import annotations

import pytest
from platform_discord.protocols import (
    InteractionProto,
)
from tests.support.discord_fakes import FakeBot, FakeUser, RecordingInteraction
from tests.support.settings import build_settings

from clubbot.cogs.base import _Logger
from clubbot.cogs.trainer import TrainerCog
from clubbot.config import DiscordbotSettings


def _cfg_with_trainer() -> DiscordbotSettings:
    return build_settings(model_trainer_api_url="http://trainer.local", model_trainer_api_key="k")


def test_mk_client_success() -> None:
    cfg = _cfg_with_trainer()
    cog = TrainerCog(bot=FakeBot(), config=cfg)
    client = cog._mk_client()
    # Minimal assertion: returned object exposes aclose attribute after creation
    _ = client.aclose


class _DeferFalseCog(TrainerCog):
    """Cog subclass where _safe_defer returns False."""

    async def _safe_defer(self, interaction: InteractionProto, *, ephemeral: bool) -> bool:
        _ = (interaction, ephemeral)
        return False


class _ErrorCapturingCog(TrainerCog):
    """Cog subclass that captures error messages."""

    def __init__(
        self,
        bot: FakeBot,
        config: DiscordbotSettings,
        errors: list[str],
    ) -> None:
        super().__init__(bot=bot, config=config)
        self._errors = errors

    async def handle_user_error(
        self, interaction: InteractionProto, log: _Logger, message: str
    ) -> None:
        _ = (interaction, log)
        self._errors.append(message)


@pytest.mark.asyncio
async def test_train_model_impl_ack_false_short_circuits() -> None:
    cfg = _cfg_with_trainer()
    cog = _DeferFalseCog(bot=FakeBot(), config=cfg)

    inter = RecordingInteraction()
    await cog._train_model_impl(
        inter,
        model_family="gpt2",
        model_size="small",
        max_seq_len=32,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        corpus_path="/data",
        tokenizer_id="tok",
    )
    # No followup should be sent because we returned early
    assert inter.sent == []


@pytest.mark.asyncio
async def test_train_model_impl_user_id_missing_calls_error() -> None:
    cfg = _cfg_with_trainer()
    errors: list[str] = []
    cog = _ErrorCapturingCog(FakeBot(), cfg, errors)

    # Use a user with id=-1 to trigger the "user_id <= 0" error path
    inter = RecordingInteraction(user=FakeUser(user_id=-1))

    await cog._train_model_impl(
        inter,
        model_family="gpt2",
        model_size="small",
        max_seq_len=32,
        num_epochs=1,
        batch_size=1,
        learning_rate=0.1,
        corpus_path="/data",
        tokenizer_id="tok",
    )
    assert errors and "user id" in errors[-1]
