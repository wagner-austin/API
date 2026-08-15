"""Test configuration and fixtures for Art-Trainer tests.

This module provides pytest fixtures and test setup for all tests.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from collections.abc import Generator
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.config import config_test_hooks
from platform_workers.testing import (
    FakeQueue,
    FakeRedis,
    fake_kv_store_factory,
    fake_rq_connection_factory,
    fake_rq_queue_factory,
    fake_rq_retry_factory,
)

from art_trainer.core import _test_hooks
from art_trainer.core.config.settings import Settings, load_settings
from art_trainer.core.services.captioning import (
    _test_hooks as captioning_test_hooks,
)
from art_trainer.core.services.deployment import (
    _test_hooks as deployment_test_hooks,
)
from art_trainer.core.services.training.backends.kohya import (
    _test_hooks as kohya_test_hooks,
)


class SettingsFactory(Protocol):
    """Protocol for settings factory fixture."""

    def __call__(
        self: SettingsFactory,
        *,
        data_root: str | None = None,
        output_root: str | None = None,
        logs_root: str | None = None,
        data_bank_api_url: str | None = None,
        data_bank_api_key: str | None = None,
        kohya_ss_path: str | None = None,
        comfyui_lora_path: str | None = None,
        blip_model_name: str | None = None,
        caption_trigger_word: str | None = None,
        redis_url: str | None = None,
        app_env: Literal["dev", "prod"] | None = None,
        security_api_key: str | None = None,
    ) -> Settings:
        """Create settings with optional overrides.

        Args:
            data_root: Override for data_root.
            output_root: Override for output_root.
            logs_root: Override for logs_root.
            data_bank_api_url: Override for data_bank_api_url.
            data_bank_api_key: Override for data_bank_api_key.
            kohya_ss_path: Override for kohya_ss_path.
            comfyui_lora_path: Override for comfyui_lora_path.
            blip_model_name: Override for blip_model_name.
            caption_trigger_word: Override for caption_trigger_word.
            redis_url: Override for redis_url.
            app_env: Override for app_env.
            security_api_key: Override for security_api_key.

        Returns:
            Configured Settings.
        """
        ...


def _make_fake_redis() -> FakeRedis:
    """Create FakeRedis instance."""
    return FakeRedis()


def _make_fake_queue() -> FakeQueue:
    """Create FakeQueue instance."""
    return FakeQueue()


def _reset_test_hooks_impl(
    tmp_path: Path, settings_factory: SettingsFactory
) -> Generator[None, None, None]:
    """Reset test hooks after each test to production defaults.

    Args:
        tmp_path: Pytest temporary directory.
        settings_factory: Settings factory fixture.

    Yields:
        None after setup, restores hooks after test.
    """
    # Save original hooks
    orig_kv = _test_hooks.kv_store_factory
    orig_rq_conn = _test_hooks.rq_connection_factory
    orig_queue = _test_hooks.rq_queue_factory
    orig_retry = _test_hooks.rq_retry_factory
    orig_load_settings = _test_hooks.load_settings
    orig_lora_output_dir = _test_hooks.lora_output_dir
    orig_shutil_which = _test_hooks.shutil_which
    orig_get_env = config_test_hooks.get_env

    # Set up fake factories
    _test_hooks.kv_store_factory = fake_kv_store_factory
    _test_hooks.rq_connection_factory = fake_rq_connection_factory
    _test_hooks.rq_queue_factory = fake_rq_queue_factory
    _test_hooks.rq_retry_factory = fake_rq_retry_factory

    # Set up test settings via hook
    test_settings = settings_factory(
        data_root=str(tmp_path / "data"),
        output_root=str(tmp_path / "output"),
        logs_root=str(tmp_path / "logs"),
        redis_url="redis://localhost:6379/0",
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="test-key",
        kohya_ss_path=str(tmp_path / "kohya_ss"),
    )

    def _test_load_settings() -> Settings:
        return test_settings

    _test_hooks.load_settings = _test_load_settings

    # Reset all service hooks
    kohya_test_hooks.reset_hooks()
    captioning_test_hooks.reset_hooks()
    deployment_test_hooks.reset_hooks()

    yield

    # Restore original hooks
    _test_hooks.kv_store_factory = orig_kv
    _test_hooks.rq_connection_factory = orig_rq_conn
    _test_hooks.rq_queue_factory = orig_queue
    _test_hooks.rq_retry_factory = orig_retry
    _test_hooks.load_settings = orig_load_settings
    _test_hooks.lora_output_dir = orig_lora_output_dir
    _test_hooks.shutil_which = orig_shutil_which
    config_test_hooks.get_env = orig_get_env

    # Reset all service hooks
    kohya_test_hooks.reset_hooks()
    captioning_test_hooks.reset_hooks()
    deployment_test_hooks.reset_hooks()


fake_redis = pytest.fixture(_make_fake_redis)
fake_queue = pytest.fixture(_make_fake_queue)
_reset_test_hooks = pytest.fixture(autouse=True)(_reset_test_hooks_impl)


def _apply_app_overrides(
    base: Settings,
    *,
    data_root: str | None,
    output_root: str | None,
    logs_root: str | None,
    data_bank_api_url: str | None,
    data_bank_api_key: str | None,
    kohya_ss_path: str | None,
    comfyui_lora_path: str | None,
    blip_model_name: str | None,
    caption_trigger_word: str | None,
) -> None:
    """Apply app config overrides to settings.

    Args:
        base: Base settings to modify.
        data_root: Override for data_root.
        output_root: Override for output_root.
        logs_root: Override for logs_root.
        data_bank_api_url: Override for data_bank_api_url.
        data_bank_api_key: Override for data_bank_api_key.
        kohya_ss_path: Override for kohya_ss_path.
        comfyui_lora_path: Override for comfyui_lora_path.
        blip_model_name: Override for blip_model_name.
        caption_trigger_word: Override for caption_trigger_word.
    """
    if data_root is not None:
        base["app"]["data_root"] = data_root
    if output_root is not None:
        base["app"]["output_root"] = output_root
    if logs_root is not None:
        base["app"]["logs_root"] = logs_root
    if data_bank_api_url is not None:
        base["app"]["data_bank_api_url"] = data_bank_api_url
    if data_bank_api_key is not None:
        base["app"]["data_bank_api_key"] = data_bank_api_key
    if kohya_ss_path is not None:
        base["app"]["kohya_ss_path"] = kohya_ss_path
    if comfyui_lora_path is not None:
        base["app"]["comfyui_lora_path"] = comfyui_lora_path
    if blip_model_name is not None:
        base["app"]["blip_model_name"] = blip_model_name
    if caption_trigger_word is not None:
        base["app"]["caption_trigger_word"] = caption_trigger_word


def _build_settings(
    *,
    data_root: str | None = None,
    output_root: str | None = None,
    logs_root: str | None = None,
    data_bank_api_url: str | None = None,
    data_bank_api_key: str | None = None,
    kohya_ss_path: str | None = None,
    comfyui_lora_path: str | None = None,
    blip_model_name: str | None = None,
    caption_trigger_word: str | None = None,
    redis_url: str | None = None,
    app_env: Literal["dev", "prod"] | None = None,
    security_api_key: str | None = None,
) -> Settings:
    """Build settings with overrides.

    Args:
        data_root: Override for data_root.
        output_root: Override for output_root.
        logs_root: Override for logs_root.
        data_bank_api_url: Override for data_bank_api_url.
        data_bank_api_key: Override for data_bank_api_key.
        kohya_ss_path: Override for kohya_ss_path.
        comfyui_lora_path: Override for comfyui_lora_path.
        blip_model_name: Override for blip_model_name.
        caption_trigger_word: Override for caption_trigger_word.
        redis_url: Override for redis_url.
        app_env: Override for app_env.
        security_api_key: Override for security_api_key.

    Returns:
        Configured Settings.
    """
    base = load_settings()
    _apply_app_overrides(
        base,
        data_root=data_root,
        output_root=output_root,
        logs_root=logs_root,
        data_bank_api_url=data_bank_api_url,
        data_bank_api_key=data_bank_api_key,
        kohya_ss_path=kohya_ss_path,
        comfyui_lora_path=comfyui_lora_path,
        blip_model_name=blip_model_name,
        caption_trigger_word=caption_trigger_word,
    )
    if redis_url is not None:
        base["redis"]["url"] = redis_url
    if app_env is not None:
        base["app_env"] = app_env
    if security_api_key is not None:
        base["security"]["api_key"] = security_api_key
    return base


def _make_settings_factory() -> SettingsFactory:
    """Create settings factory.

    Returns:
        Settings factory function.
    """
    return _build_settings


settings_factory = pytest.fixture(_make_settings_factory)


def _make_settings_with_paths(tmp_path: Path, settings_factory: SettingsFactory) -> Settings:
    """Create settings with temporary paths.

    Args:
        tmp_path: Pytest temporary directory.
        settings_factory: Settings factory fixture.

    Returns:
        Configured Settings.
    """
    return settings_factory(
        data_root=str(tmp_path / "data"),
        output_root=str(tmp_path / "output"),
        logs_root=str(tmp_path / "logs"),
        kohya_ss_path=str(tmp_path / "kohya_ss"),
    )


settings_with_paths = pytest.fixture(_make_settings_with_paths)
