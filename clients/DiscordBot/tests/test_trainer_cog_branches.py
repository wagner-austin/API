from __future__ import annotations

from typing import NoReturn

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.model_trainer_client import ModelTrainerAPIError, TrainResponse
from platform_discord.protocols import BotProto, InteractionProto
from tests.support.discord_fakes import FakeBot, FakeUser, RecordingInteraction
from tests.support.settings import build_settings

from clubbot.cogs.trainer import TrainerCog
from clubbot.config import DiscordbotSettings


def _base_cfg(
    *, trainer_url: str | None = "https://trainer.local", redis_url: str | None = None
) -> DiscordbotSettings:
    return build_settings(
        model_trainer_api_url=trainer_url if trainer_url else "",
        redis_url=redis_url,
    )


def _interaction_with_user(user_id: int) -> RecordingInteraction:
    return RecordingInteraction(user=FakeUser(user_id=user_id))


def _cog(
    *, trainer_url: str | None = "https://trainer.local", redis_url: str | None = None
) -> TrainerCog:
    cfg = _base_cfg(trainer_url=trainer_url, redis_url=redis_url)
    return TrainerCog(bot=FakeBot(), config=cfg)


def test_mk_client_requires_base_url() -> None:
    cfg = _base_cfg(trainer_url="")
    cog = TrainerCog(bot=FakeBot(), config=cfg)
    with pytest.raises(AppError):
        _ = cog._mk_client()


def test_subscriber_wiring_starts_with_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, int] = {"n": 0}

    class _Sub:
        def __init__(self, *, bot: BotProto, redis_url: str, events_channel: str) -> None:
            _ = bot
            self.redis_url = redis_url
            self.events_channel = events_channel

        def start(self) -> None:
            started["n"] += 1

    monkeypatch.setattr("clubbot.cogs.trainer.TrainerEventSubscriber", _Sub, raising=True)
    _ = _cog(redis_url="redis://example")
    assert started["n"] == 1


def test_subscriber_wiring_skips_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    def _should_not_construct(*, bot: BotProto, redis_url: str, events_channel: str) -> NoReturn:
        _ = (bot, redis_url, events_channel)
        raise RuntimeError("should_not_construct")

    monkeypatch.setattr(
        "clubbot.cogs.trainer.TrainerEventSubscriber",
        _should_not_construct,
        raising=True,
    )
    cog = _cog(redis_url=None)
    assert cog._subscriber is None


@pytest.mark.asyncio
async def test_safe_defer_defers(monkeypatch: pytest.MonkeyPatch) -> None:
    cog = _cog()
    inter = _interaction_with_user(123)

    # Avoid network by forcing rate-limit block after defer
    def deny(_uid: int, _cmd: str) -> tuple[bool, float]:
        return (False, 1.0)

    monkeypatch.setattr(cog.rate_limiter, "allow", deny, raising=True)
    await cog._train_model_impl(
        inter,
        model_family="gpt2",
        model_size="small",
        max_seq_len=1,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok",
    )
    # Defer recorded as first send
    assert inter.sent and inter.sent[0]["where"] in {"response", "followup"}


def test_decode_int_attr() -> None:
    cog = _cog()
    assert cog.decode_int_attr(None, "id") is None


@pytest.mark.asyncio
async def test_train_model_user_id_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    cog = _cog()

    messages: list[str] = []

    async def capture_error(_i: InteractionProto, _log: str, msg: str) -> None:
        messages.append(msg)

    monkeypatch.setattr(cog, "handle_user_error", capture_error, raising=True)
    # user_id <= 0 triggers user id error path
    inter = _interaction_with_user(-1)
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
    assert messages and "user id" in messages[-1]


@pytest.mark.asyncio
async def test_train_model_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    cog = _cog()

    def allow_fn(user_id: int, command: str) -> tuple[bool, float]:
        _ = (user_id, command)
        return (False, 3.0)

    monkeypatch.setattr(cog.rate_limiter, "allow", allow_fn, raising=True)
    inter = _interaction_with_user(42)
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
    assert inter.sent and any(
        s["where"] == "followup" and isinstance(s["content"], str) and "Please wait" in s["content"]
        for s in inter.sent
    )


@pytest.mark.asyncio
async def test_train_model_mk_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cog = _cog()
    messages: list[str] = []

    async def handle_user_error(_i: InteractionProto, _log: str, msg: str) -> None:
        messages.append(msg)

    def raise_app_error() -> None:
        raise AppError(ErrorCode.INVALID_INPUT, "cfg", http_status=400)

    monkeypatch.setattr(cog, "_mk_client", raise_app_error, raising=True)
    monkeypatch.setattr(cog, "handle_user_error", handle_user_error, raising=True)
    inter = _interaction_with_user(99)
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
    assert messages and "cfg" in messages[-1]


@pytest.mark.asyncio
async def test_train_model_api_and_runtime_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    cog = _cog()
    messages: list[str] = []

    class _Client:
        def __init__(self, *, raise_kind: str) -> None:
            self.raise_kind = raise_kind

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
            if self.raise_kind == "api":
                raise ModelTrainerAPIError(400, "bad")
            raise RuntimeError("boom")

        async def aclose(self) -> None:
            return None

    async def handle_user_error(_i: InteractionProto, _log: str, msg: str) -> None:
        messages.append(msg)

    def mk_api_client() -> _Client:
        return _Client(raise_kind="api")

    def mk_runtime_client() -> _Client:
        return _Client(raise_kind="runtime")

    monkeypatch.setattr(cog, "handle_user_error", handle_user_error, raising=True)

    inter1 = _interaction_with_user(1)
    monkeypatch.setattr(cog, "_mk_client", mk_api_client, raising=True)
    await cog._train_model_impl(
        inter1,
        model_family="gpt2",
        model_size="small",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok",
    )
    assert messages and "API error" in messages[-1]

    inter2 = _interaction_with_user(2)
    messages.clear()
    monkeypatch.setattr(cog, "_mk_client", mk_runtime_client, raising=True)
    await cog._train_model_impl(
        inter2,
        model_family="gpt2",
        model_size="small",
        max_seq_len=16,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok",
    )
    assert messages and "An error occurred" in messages[-1]


@pytest.mark.asyncio
async def test_train_model_success(monkeypatch: pytest.MonkeyPatch) -> None:
    cog = _cog()

    class _Client:
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
            _ = (
                user_id,
                model_family,
                model_size,
                max_seq_len,
                num_epochs,
                batch_size,
                learning_rate,
                corpus_path,
                tokenizer_id,
                request_id,
            )
            return TrainResponse("run-1", "job-1")

        async def aclose(self) -> None:
            return None

    def mk_success_client() -> _Client:
        return _Client()

    monkeypatch.setattr(cog, "_mk_client", mk_success_client, raising=True)
    inter = _interaction_with_user(7)
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
    assert inter.sent and any(
        s["where"] == "followup" and s["embed"] is not None for s in inter.sent
    )
