"""Tests to achieve coverage for _default_* production implementations in _test_hooks.

These functions are normally replaced by fakes in tests, so they need explicit
coverage tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from discord.abc import Snowflake as DiscordSnowflake
from discord.app_commands import AppCommand
from tests.support.discord_fakes import FakeBot
from tests.support.settings import build_settings

from clubbot import _test_hooks

logger = logging.getLogger(__name__)


class _FakeTree:
    """Fake tree for tree_sync test."""

    async def sync(self, *, guild: DiscordSnowflake | None = None) -> list[AppCommand]:
        _ = guild
        return []


def test_default_qr_service_factory_creates_service() -> None:
    """Test _default_qr_service_factory creates QRService instance."""
    cfg = build_settings(qr_api_url="http://test:8080")
    result = _test_hooks._default_qr_service_factory(cfg)
    # Should return a QRServiceLike which has generate_qr method
    assert callable(result.generate_qr)


def test_default_validate_youtube_url_for_client() -> None:
    """Test _default_validate_youtube_url_for_client with valid URL."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = _test_hooks._default_validate_youtube_url_for_client(url)
    assert result == url


def test_default_extract_video_id() -> None:
    """Test _default_extract_video_id with valid URL."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = _test_hooks._default_extract_video_id(url)
    assert result == "dQw4w9WgXcQ"


def test_default_load_httpx_module() -> None:
    """Test _default_load_httpx_module returns httpx module."""
    result = _test_hooks._default_load_httpx_module()
    # Module should have a Client class that is callable
    assert callable(result.Client)


def test_default_urlsplit() -> None:
    """Test _default_urlsplit uses stdlib urlsplit."""
    result = _test_hooks._default_urlsplit("https://example.com/path?q=1")
    assert result.scheme == "https"
    assert result.netloc == "example.com"
    assert result.path == "/path"


@pytest.mark.asyncio
async def test_default_tree_sync_calls_tree_method() -> None:
    """Test _default_tree_sync calls the tree's sync method."""
    tree = _FakeTree()
    result = await _test_hooks._default_tree_sync(tree)
    assert result == []


def test_default_build_digits_enqueuer_with_empty_url() -> None:
    """Test _default_build_digits_enqueuer returns None for empty URL."""
    result = _test_hooks._default_build_digits_enqueuer(redis_url="")
    assert result is None

    result2 = _test_hooks._default_build_digits_enqueuer(redis_url="   ")
    assert result2 is None


def test_default_setup_logging_calls_platform_core() -> None:
    """Test _default_setup_logging calls platform_core setup_logging."""
    # Call the function - it should not raise
    _test_hooks._default_setup_logging(
        level="INFO",
        service_name="test-service",
        format_mode="text",
    )


def test_default_create_bot_orchestrator_raises_on_wrong_type() -> None:
    """Test _default_create_bot_orchestrator raises TypeError on wrong type."""

    class _FakeContainer:
        cfg = build_settings()

    container = _FakeContainer()
    with pytest.raises(TypeError, match="Expected ServiceContainer"):
        _test_hooks._default_create_bot_orchestrator(container)


def test_default_trainer_event_subscriber_factory_creates_subscriber() -> None:
    """Test _default_trainer_event_subscriber_factory creates subscriber."""
    bot = FakeBot()
    result = _test_hooks._default_trainer_event_subscriber_factory(
        bot=bot,
        redis_url="redis://localhost:6379",
        events_channel="trainer:events",
    )
    # Result should have start method (TrainerEventSubscriberLike)
    assert callable(result.start)


def test_default_digits_event_subscriber_factory_creates_subscriber() -> None:
    """Test _default_digits_event_subscriber_factory creates subscriber."""
    bot = FakeBot()
    result = _test_hooks._default_digits_event_subscriber_factory(
        bot=bot,
        redis_url="redis://localhost:6379",
    )
    # Result should have start and stop methods (DigitsEventSubscriberLike)
    assert callable(result.start)
    assert callable(result.stop)


def test_default_trainer_api_client_factory_creates_client() -> None:
    """Test _default_trainer_api_client_factory creates client."""
    result = _test_hooks._default_trainer_api_client_factory(
        base_url="http://localhost:8080",
        api_key=None,
        timeout_seconds=30,
    )
    # Result should have train method (TrainerApiClientLike)
    assert callable(result.train)


def test_default_guard_find_monorepo_root_from_current() -> None:
    """Test _default_guard_find_monorepo_root finds root from current directory."""
    # Start from this file's directory and find the monorepo root
    start = Path(__file__).parent
    result = _test_hooks._default_guard_find_monorepo_root(start)
    assert (result / "libs").is_dir()


def test_default_rq_retry_creates_retry_object() -> None:
    """Test _default_rq_retry creates an rq Retry object."""
    # Just call the function to verify it doesn't raise
    _test_hooks._default_rq_retry(max_retries=3, intervals=[60, 120])


def test_default_build_digits_enqueuer_with_valid_url() -> None:
    """Test _default_build_digits_enqueuer creates enqueuer with valid URL."""
    result = _test_hooks._default_build_digits_enqueuer(redis_url="redis://localhost:6379")
    # Should return an enqueuer with enqueue_train method
    if result is None:
        raise AssertionError("Expected enqueuer, got None")
    assert callable(result.enqueue_train)


def test_default_create_bot_orchestrator_success() -> None:
    """Test _default_create_bot_orchestrator creates orchestrator."""
    from clubbot.container import ServiceContainer
    from clubbot.services.qr.client import QRService

    cfg = build_settings(qr_api_url="http://test:8080")
    container = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    result = _test_hooks._default_create_bot_orchestrator(container)
    # Should return an orchestrator with run method
    assert callable(result.run)


def test_default_captions_calls_api_client() -> None:
    """Test _default_captions calls the real captions function with proper setup."""
    from platform_core.json_utils import JSONValue
    from platform_core.testing import FakeHttpxClient, FakeHttpxResponse

    # Set up fake HTTP client that returns valid captions response
    json_body: JSONValue = {
        "url": "https://youtube.com/watch?v=abc",
        "video_id": "abc",
        "text": "Hello world",
    }
    response = FakeHttpxResponse(200, json_body=json_body)
    fake_client = FakeHttpxClient(response)

    def _fake_build_client(timeout: float) -> FakeHttpxClient:
        return fake_client

    _test_hooks.build_client = _fake_build_client

    # Call _default_captions with a dict representing the client config
    client_dict: dict[str, float | str] = {
        "base_url": "http://transcript-api:8080",
        "timeout_seconds": 30.0,
    }
    result = _test_hooks._default_captions(
        client_dict,
        url="https://youtube.com/watch?v=abc",
        preferred_langs=["en"],
    )
    assert result["url"] == "https://youtube.com/watch?v=abc"
    assert result["video_id"] == "abc"
    assert result["text"] == "Hello world"


def test_default_load_settings_calls_real_loader() -> None:
    """Test _default_load_settings calls platform_core config loader."""
    from platform_core.config import _test_hooks as config_hooks
    from platform_core.testing import FakeEnv

    # Set up environment with required settings
    env = FakeEnv(
        {
            "DISCORD_TOKEN": "test.token.value",
            "QR_API_URL": "http://qr:8080",
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "text",
            "SERVICE_NAME": "test-discord-bot",
        }
    )
    config_hooks.get_env = env

    result = _test_hooks._default_load_settings()
    assert result["discord"]["token"] == "test.token.value"


def test_default_rq_queue_creates_queue() -> None:
    """Test _default_rq_queue creates RQ queue with connection."""
    from platform_workers.testing import FakeRedisBytesClient

    fake_conn = FakeRedisBytesClient()
    result = _test_hooks._default_rq_queue("test-queue", connection=fake_conn)
    # Result should be an RQClientQueue with enqueue method
    assert callable(result.enqueue)


def test_default_create_service_container_from_env() -> None:
    """Test _default_create_service_container creates container from env."""
    from platform_core.config import _test_hooks as config_hooks
    from platform_core.testing import FakeEnv

    # Set up environment with required settings
    env = FakeEnv(
        {
            "DISCORD_TOKEN": "test.token.value",
            "QR_API_URL": "http://qr:8080",
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": "text",
            "SERVICE_NAME": "test-discord-bot",
        }
    )
    config_hooks.get_env = env

    # Reset load_settings hook to use production implementation
    # (conftest fixture overrides it with test settings)
    _test_hooks.load_settings = _test_hooks._default_load_settings

    result = _test_hooks._default_create_service_container()
    # Should return a ServiceContainerProtocol with cfg attribute
    assert result.cfg["discord"]["token"] == "test.token.value"


def test_default_wrap_interaction_wraps_discord_interaction() -> None:
    """Test _default_wrap_interaction wraps a discord.Interaction."""
    from tests.support.discord_fakes import RecordingInteraction

    fake_inter = RecordingInteraction()
    result = _test_hooks._default_wrap_interaction(fake_inter)
    # Result should be an InteractionProtoLike with response attribute
    if result.response is None:
        raise AssertionError("Expected response to be non-None")


class _FakeOrchestrator:
    """Fake orchestrator that records build_bot calls."""

    def __init__(self) -> None:
        self.build_bot_called = False

    def build_bot(self) -> _test_hooks.BotRunnerProtocol:
        self.build_bot_called = True

        class _FakeBotRunner:
            def run(self, token: str) -> None:
                _ = token

        return _FakeBotRunner()


def test_default_orchestrator_build_bot_calls_build_bot() -> None:
    """Test _default_orchestrator_build_bot calls orchestrator.build_bot()."""
    fake_orch = _FakeOrchestrator()
    result = _test_hooks._default_orchestrator_build_bot(fake_orch)
    assert fake_orch.build_bot_called
    # Result should have run method (BotRunnerProtocol)
    assert callable(result.run)
