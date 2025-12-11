from __future__ import annotations

from typing import NoReturn

import pytest
from platform_core.errors import AppError
from platform_core.model_trainer_client import ModelTrainerAPIError
from platform_discord.rate_limiter import RateLimiter

from clubbot import _test_hooks
from clubbot._test_hooks import (
    BotProto,
    TrainerApiClientLike,
    TrainerEventSubscriberLike,
)
from clubbot.cogs.trainer import TrainerCog
from clubbot.config import DiscordbotSettings
from tests.support.discord_fakes import FakeBot, FakeUser, RecordingInteraction
from tests.support.settings import build_settings


class _StartRecorder:
    """Fake subscriber that records start calls."""

    __slots__ = ("_started_count", "events_channel", "redis_url")

    def __init__(self, *, bot: BotProto, redis_url: str, events_channel: str) -> None:
        self.redis_url = redis_url
        self.events_channel = events_channel
        self._started_count = 0

    def start(self) -> None:
        self._started_count += 1


def _make_denying_rate_limiter() -> RateLimiter:
    """Create a rate limiter that's already exhausted (always denies).

    Uses per_window=1 and pre-exhausts it to avoid edge case with per_window=0.
    """
    rl = RateLimiter(per_window=1, window_seconds=60)
    # Pre-exhaust the rate limiter for user 42 and user 123
    rl.allow(42, "train_model")
    rl.allow(123, "train_model")
    return rl


class _FakeTrainerClient:
    """Fake trainer API client for testing."""

    __slots__ = ("_closed", "_train_raises", "_train_response")

    def __init__(
        self,
        *,
        train_response: dict[str, str] | None = None,
        train_raises: Exception | None = None,
    ) -> None:
        self._closed = False
        # Response needs both run_id and job_id for success path
        self._train_response = train_response or {
            "status": "ok",
            "run_id": "run-1",
            "job_id": "job-1",
        }
        self._train_raises = train_raises

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
    ) -> dict[str, str]:
        if self._train_raises:
            raise self._train_raises
        return self._train_response

    async def aclose(self) -> None:
        self._closed = True


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


def _make_subscriber_factory_recorder(
    instances: list[TrainerEventSubscriberLike],
) -> _test_hooks.TrainerEventSubscriberFactoryProtocol:
    """Create a factory that records subscriber creation."""

    def _factory(
        *, bot: BotProto, redis_url: str, events_channel: str
    ) -> TrainerEventSubscriberLike:
        sub: TrainerEventSubscriberLike = _StartRecorder(
            bot=bot, redis_url=redis_url, events_channel=events_channel
        )
        instances.append(sub)
        return sub

    return _factory


def _make_blocking_subscriber_factory() -> _test_hooks.TrainerEventSubscriberFactoryProtocol:
    """Create a factory that raises if called (to test skip path)."""

    def _factory(*, bot: BotProto, redis_url: str, events_channel: str) -> NoReturn:
        raise RuntimeError("should_not_construct")

    return _factory


def _make_client_factory(
    client: _FakeTrainerClient,
) -> _test_hooks.TrainerApiClientFactoryProtocol:
    """Create a factory that returns a fake trainer client."""

    def _factory(
        *, base_url: str, api_key: str | None, timeout_seconds: int
    ) -> TrainerApiClientLike:
        return client

    return _factory


def test_mk_client_requires_base_url() -> None:
    cfg = _base_cfg(trainer_url="")
    cog = TrainerCog(bot=FakeBot(), config=cfg)
    with pytest.raises(AppError):
        _ = cog._mk_client()


def test_subscriber_wiring_starts_with_redis() -> None:
    instances: list[TrainerEventSubscriberLike] = []
    _test_hooks.trainer_event_subscriber_factory = _make_subscriber_factory_recorder(instances)

    _ = _cog(redis_url="redis://example")

    assert len(instances) == 1
    # Verify the start method was called (via concrete type)
    sub = instances[0]
    if not isinstance(sub, _StartRecorder):
        raise AssertionError("Expected _StartRecorder instance")
    assert sub._started_count == 1


def test_subscriber_wiring_skips_without_redis() -> None:
    _test_hooks.trainer_event_subscriber_factory = _make_blocking_subscriber_factory()

    cog = _cog(redis_url=None)
    assert cog._subscriber is None


@pytest.mark.asyncio
async def test_safe_defer_defers() -> None:
    # Create a cog with rate limiter that will deny (prevents network call)
    cog = _cog()
    cog.rate_limiter = _make_denying_rate_limiter()

    inter = _interaction_with_user(123)

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
async def test_train_model_user_id_missing() -> None:
    cog = _cog()

    # user_id <= 0 triggers user id error path
    inter = _interaction_with_user(-1)
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
    # Verify messages were sent via interaction
    assert inter.sent, "Expected at least one send"
    # Find the message with content (skip defer which has content=None)
    content_msgs = [s for s in inter.sent if s.get("content") is not None]
    if not content_msgs:
        raise AssertionError("Expected at least one message with content")
    content = content_msgs[0].get("content")
    if content is None:
        raise AssertionError("Expected content in message")
    assert "user" in content.lower()


@pytest.mark.asyncio
async def test_train_sends_request() -> None:
    client = _FakeTrainerClient(
        train_response={"status": "ok", "run_id": "the-run-id", "job_id": "the-job-id"}
    )
    _test_hooks.trainer_api_client_factory = _make_client_factory(client)

    cog = _cog()
    inter = _interaction_with_user(42)

    await cog._train_model_impl(
        inter,
        model_family="gpt2",
        model_size="small",
        max_seq_len=256,
        num_epochs=1,
        batch_size=4,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok",
    )
    assert inter.sent
    assert client._closed


@pytest.mark.asyncio
async def test_train_handles_api_error() -> None:
    # ModelTrainerAPIError takes (status_code, message) - check its signature
    client = _FakeTrainerClient(train_raises=ModelTrainerAPIError(422, "Invalid input"))
    _test_hooks.trainer_api_client_factory = _make_client_factory(client)

    cog = _cog()
    inter = _interaction_with_user(42)

    await cog._train_model_impl(
        inter,
        model_family="gpt2",
        model_size="small",
        max_seq_len=256,
        num_epochs=1,
        batch_size=4,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok",
    )
    # Error path sends response (client.aclose() not called on error)
    assert inter.sent


@pytest.mark.asyncio
async def test_train_handles_rate_limit() -> None:
    cog = _cog()
    cog.rate_limiter = _make_denying_rate_limiter()

    inter = _interaction_with_user(42)

    await cog._train_model_impl(
        inter,
        model_family="gpt2",
        model_size="small",
        max_seq_len=256,
        num_epochs=1,
        batch_size=4,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok",
    )
    # Rate limited response sent
    assert inter.sent


@pytest.mark.asyncio
async def test_train_handles_runtime_error() -> None:
    client = _FakeTrainerClient(train_raises=RuntimeError("oops"))
    _test_hooks.trainer_api_client_factory = _make_client_factory(client)

    cog = _cog()
    inter = _interaction_with_user(42)

    await cog._train_model_impl(
        inter,
        model_family="gpt2",
        model_size="small",
        max_seq_len=256,
        num_epochs=1,
        batch_size=4,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok",
    )
    # Error path handles gracefully for users (client.aclose() not called on error)
    assert inter.sent


@pytest.mark.asyncio
async def test_train_handles_connection_error() -> None:
    # OSError (parent of ConnectionError) is caught by the handler
    client = _FakeTrainerClient(train_raises=OSError("network"))
    _test_hooks.trainer_api_client_factory = _make_client_factory(client)

    cog = _cog()
    inter = _interaction_with_user(42)

    await cog._train_model_impl(
        inter,
        model_family="gpt2",
        model_size="small",
        max_seq_len=256,
        num_epochs=1,
        batch_size=4,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok",
    )
    # Error path handles gracefully for users
    assert inter.sent


@pytest.mark.asyncio
async def test_train_handles_app_error() -> None:
    """Test that AppError from train() is caught and shown to user."""
    from platform_core.errors import ErrorCode

    app_err = AppError(ErrorCode.INVALID_INPUT, "Invalid corpus format")
    client = _FakeTrainerClient(train_raises=app_err)
    _test_hooks.trainer_api_client_factory = _make_client_factory(client)

    cog = _cog()
    inter = _interaction_with_user(42)

    await cog._train_model_impl(
        inter,
        model_family="gpt2",
        model_size="small",
        max_seq_len=256,
        num_epochs=1,
        batch_size=4,
        learning_rate=5e-4,
        corpus_path="/data/corpus",
        tokenizer_id="tok",
    )
    # AppError should be caught and shown to user
    assert inter.sent
    # Find message with content
    content_msgs = [s for s in inter.sent if s.get("content") is not None]
    if not content_msgs:
        raise AssertionError("Expected message with content")
    content = content_msgs[0].get("content")
    if content is None:
        raise AssertionError("Expected content in message")
    assert "Invalid corpus format" in content
