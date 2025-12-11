"""Pytest configuration and fixtures for DiscordBot tests."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Generator

import pytest

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.services.jobs import trainer_notifier
from tests.support.settings import (
    SettingsFactory,
    build_settings,
    make_settings_factory,
)


def reset_hooks() -> None:
    """Reset all test hooks to their default implementations."""
    _test_hooks.load_settings = _test_hooks._default_load_settings
    _test_hooks.build_client = _test_hooks._default_build_client
    _test_hooks.build_async_client = _test_hooks._default_build_async_client
    _test_hooks.redis_raw_for_rq = _test_hooks._default_redis_raw_for_rq
    _test_hooks.rq_queue = _test_hooks._default_rq_queue
    _test_hooks.rq_retry = _test_hooks._default_rq_retry
    _test_hooks.guard_find_monorepo_root = _test_hooks._default_guard_find_monorepo_root
    _test_hooks.guard_load_orchestrator = _test_hooks._default_guard_load_orchestrator
    _test_hooks.load_httpx_module = _test_hooks._default_load_httpx_module
    _test_hooks.build_digits_enqueuer = _test_hooks._default_build_digits_enqueuer
    _test_hooks.setup_logging = _test_hooks._default_setup_logging
    _test_hooks.create_service_container = _test_hooks._default_create_service_container
    _test_hooks.create_bot_orchestrator = _test_hooks._default_create_bot_orchestrator
    _test_hooks.urlsplit = _test_hooks._default_urlsplit
    _test_hooks.trainer_event_subscriber_factory = (
        _test_hooks._default_trainer_event_subscriber_factory
    )
    _test_hooks.trainer_api_client_factory = _test_hooks._default_trainer_api_client_factory
    _test_hooks.digits_event_subscriber_factory = (
        _test_hooks._default_digits_event_subscriber_factory
    )
    _test_hooks.wrap_interaction = _test_hooks._default_wrap_interaction
    _test_hooks.tree_sync = _test_hooks._default_tree_sync
    _test_hooks.discord_exception_types = _test_hooks._default_discord_exception_types
    _test_hooks.orchestrator_sync_global_override = None
    _test_hooks.orchestrator_build_bot_override = None
    _test_hooks.orchestrator_build_bot = _test_hooks._default_orchestrator_build_bot
    _test_hooks.validate_youtube_url = _test_hooks._default_validate_youtube_url
    _test_hooks.asyncio_to_thread = _test_hooks._default_asyncio_to_thread
    _test_hooks.bot_fetch_user = _test_hooks._default_bot_fetch_user
    _test_hooks.app_command_error_handler = _test_hooks._default_app_command_error_handler
    # Reset trainer_notifier module-level hook
    trainer_notifier.handle_trainer_event = trainer_notifier._default_handle_trainer_event


@pytest.fixture(autouse=True)
def _reset_hooks_fixture() -> Generator[None, None, None]:
    """Autouse fixture that resets hooks after each test."""
    # Set load_settings to return test defaults for all tests
    test_settings = build_settings()

    def _test_load_settings() -> DiscordbotSettings:
        return test_settings

    _test_hooks.load_settings = _test_load_settings
    yield
    reset_hooks()


def _settings_factory() -> SettingsFactory:
    return make_settings_factory()


settings_factory = pytest.fixture(name="settings_factory")(_settings_factory)


def _settings() -> DiscordbotSettings:
    return build_settings()


settings = pytest.fixture(name="settings")(_settings)

# Convenience alias expected by some tests for direct settings construction.
_build_settings = build_settings


def pytest_sessionstart(session: pytest.Session) -> None:
    """Ensure get_event_loop() works reliably on Windows with strict asyncio mode."""
    if sys.platform.startswith("win"):
        if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
