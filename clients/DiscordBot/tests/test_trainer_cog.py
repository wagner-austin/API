from __future__ import annotations

import asyncio
import logging
from types import TracebackType

import pytest
from platform_core.model_trainer_client import TrainResponse
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, MessageProto, ResponseProto, UserProto
from tests.support.discord_fakes import FakeBot, FakeMessage, FakeUser
from tests.support.settings import build_settings

from clubbot.cogs.trainer import TrainerCog


class _Resp:
    def __init__(self) -> None:
        self._done = False

    def is_done(self) -> bool:
        return self._done

    async def defer(self, *, ephemeral: bool = False) -> None:
        self._done = True

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> None:
        pass


class _Follow:
    def __init__(self) -> None:
        self.last_embed: EmbedProto | None = None

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> MessageProto:
        self.last_embed = embed

        return FakeMessage()


class _User:
    @property
    def id(self) -> int:
        return 123

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        _ = (content, embed, file)
        return FakeMessage()


class _Interaction:
    def __init__(self) -> None:
        self.response: ResponseProto = _Resp()
        self.followup = _Follow()
        self.user: UserProto = FakeUser(user_id=123)


class _ClientStub:
    class _Ctx:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> bool:
            return False

    def __init__(self) -> None:
        self._client = self._Ctx()

    async def aclose(self) -> None:
        return None

    async def train(
        self,
        *,
        user_id: int,
        model_family: str,
        model_size: str,
        max_seq_len: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        corpus_path: str,
        tokenizer_id: str,
        request_id: str,
    ) -> TrainResponse:
        return TrainResponse(run_id="r", job_id="j")


def test_trainer_cog_train_model_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_settings(
        model_trainer_api_url="https://example",
        model_trainer_api_key=None,
        model_trainer_api_timeout_seconds=10,
    )
    cog = TrainerCog(bot=FakeBot(), config=cfg)
    inter = _Interaction()

    async def _run() -> None:
        monkeypatch.setattr(cog, "_mk_client", lambda: _ClientStub())
        # Call the internal implementation directly for testing
        await cog._train_model_impl(
            inter,
            model_family="gpt2",
            model_size="small",
            max_seq_len=16,
            num_epochs=1,
            batch_size=1,
            learning_rate=5e-4,
            corpus_path="/data/corpus",
            tokenizer_id="tok",
        )

    asyncio.run(_run())
    # Check the fake followup directly
    followup = inter.followup
    assert type(followup) is _Follow
    if followup.last_embed is None:
        raise AssertionError("expected last_embed")
    # Verify it's a discord.Embed by checking required attributes
    embed = followup.last_embed
    _ = embed.title
    _ = embed.description


logger = logging.getLogger(__name__)
