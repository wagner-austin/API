from __future__ import annotations

import pytest
from platform_discord.protocols import (
    InteractionProto,
    UserProto,
)
from tests.support.discord_fakes import FakeBot, RecordingInteraction
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


@pytest.mark.asyncio
async def test_train_model_impl_ack_false_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg_with_trainer()
    cog = TrainerCog(bot=FakeBot(), config=cfg)

    async def _false_ack(_i: InteractionProto, *, ephemeral: bool) -> bool:
        _ = ephemeral
        return False

    monkeypatch.setattr(cog, "_safe_defer", _false_ack, raising=True)
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
async def test_train_model_impl_user_id_missing_calls_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg_with_trainer()
    cog = TrainerCog(bot=FakeBot(), config=cfg)
    inter = RecordingInteraction()

    # Force ack to true to proceed into user id decode path
    async def _true_ack(_i: InteractionProto, *, ephemeral: bool) -> bool:
        _ = ephemeral
        return True

    def _none_decode(_o: UserProto | None, _n: str) -> int | None:
        return None

    monkeypatch.setattr(cog, "_safe_defer", _true_ack, raising=True)
    monkeypatch.setattr(TrainerCog, "decode_int_attr", staticmethod(_none_decode), raising=True)

    errors: list[str] = []

    async def capture_error(_i: InteractionProto, _l: _Logger, msg: str) -> None:
        errors.append(msg)

    monkeypatch.setattr(cog, "handle_user_error", capture_error, raising=True)

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
