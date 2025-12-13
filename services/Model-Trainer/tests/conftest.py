from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.config import config_test_hooks
from platform_ml import sentencepiece as _spm_init
from platform_ml import torch_types as platform_ml_torch_types
from platform_workers.testing import (
    FakeQueue,
    FakeRedis,
    fake_kv_store_factory,
    fake_rq_connection_factory,
    fake_rq_queue_factory,
    fake_rq_retry_factory,
)

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings, load_settings

# Use the import to cache sentencepiece in sys.modules with SWIG warnings suppressed
_ = _spm_init


class SettingsFactory(Protocol):
    def __call__(
        self: SettingsFactory,
        *,
        artifacts_root: str | None = None,
        runs_root: str | None = None,
        logs_root: str | None = None,
        data_root: str | None = None,
        data_bank_api_url: str | None = None,
        data_bank_api_key: str | None = None,
        threads: int | None = None,
        redis_url: str | None = None,
        app_env: Literal["dev", "prod"] | None = None,
        security_api_key: str | None = None,
    ) -> Settings: ...


def _make_fake_redis() -> FakeRedis:
    return FakeRedis()


def _make_fake_queue() -> FakeQueue:
    return FakeQueue()


def _reset_test_hooks_impl(
    tmp_path: Path, settings_factory: SettingsFactory
) -> Generator[None, None, None]:
    """Reset test hooks after each test to production defaults."""
    # Save original hooks
    orig_kv = _test_hooks.kv_store_factory
    orig_rq_conn = _test_hooks.rq_connection_factory
    orig_queue = _test_hooks.rq_queue_factory
    orig_retry = _test_hooks.rq_retry_factory
    orig_load_settings = _test_hooks.load_settings
    orig_artifact_store = _test_hooks.artifact_store_factory
    orig_service_container = _test_hooks.service_container_from_settings
    orig_corpus_fetcher = _test_hooks.corpus_fetcher_factory
    orig_load_tokenizer = _test_hooks.load_tokenizer_for_training
    orig_httpx_client = _test_hooks.httpx_client_factory
    # Training infrastructure hooks
    orig_cuda_is_available = _test_hooks.cuda_is_available
    orig_pkg_version = _test_hooks.pkg_version
    orig_model_dir = _test_hooks.model_dir
    orig_split_corpus_files = _test_hooks.split_corpus_files
    orig_freeze_embeddings = _test_hooks.freeze_embeddings
    orig_shutil_which = _test_hooks.shutil_which
    # SentencePiece backend hooks
    orig_spm_require_cli = _test_hooks.spm_require_cli
    orig_spm_train = _test_hooks.spm_train
    orig_spm_encode_ids = _test_hooks.spm_encode_ids
    # Additional hooks for edge-case testing
    orig_random_factory = _test_hooks.random_factory
    orig_shutil_rmtree = _test_hooks.shutil_rmtree
    orig_load_wandb_module = _test_hooks.load_wandb_module
    orig_load_gpt2_model = _test_hooks.load_gpt2_model
    orig_sample_token = _test_hooks.sample_token
    orig_spm_decode_ids = _test_hooks.spm_decode_ids
    # Standard library hooks
    orig_os_scandir = _test_hooks.os_scandir
    orig_shutil_disk_usage = _test_hooks.shutil_disk_usage
    orig_path_unlink = _test_hooks.path_unlink
    orig_time_sleep = _test_hooks.time_sleep
    orig_path_iterdir = _test_hooks.path_iterdir
    # Cleanup service hooks
    orig_corpus_cache_cleanup_service = _test_hooks.corpus_cache_cleanup_service_factory
    orig_tokenizer_cleanup_service = _test_hooks.tokenizer_cleanup_service_factory
    # JSON hooks
    orig_dump_json_str = _test_hooks.dump_json_str
    # Orchestrator hooks
    orig_tokenizer_enqueue_hook = _test_hooks.tokenizer_enqueue_hook
    # GPT2 backend hooks
    orig_load_prepared_gpt2_from_handle = _test_hooks.load_prepared_gpt2_from_handle
    # Guard hooks
    orig_guard_find_monorepo_root = _test_hooks.guard_find_monorepo_root
    orig_guard_load_orchestrator = _test_hooks.guard_load_orchestrator
    # Platform core config hooks
    orig_get_env = config_test_hooks.get_env
    # Platform ML torch hooks (for device resolution)
    orig_platform_ml_import_torch = platform_ml_torch_types._import_torch

    # Set up fake factories
    _test_hooks.kv_store_factory = fake_kv_store_factory
    _test_hooks.rq_connection_factory = fake_rq_connection_factory
    _test_hooks.rq_queue_factory = fake_rq_queue_factory
    _test_hooks.rq_retry_factory = fake_rq_retry_factory

    # Set up test settings via hook
    test_settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        redis_url="redis://localhost:6379/0",
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="test-key",
    )

    def _test_load_settings() -> Settings:
        return test_settings

    _test_hooks.load_settings = _test_load_settings

    yield

    # Restore original hooks
    _test_hooks.kv_store_factory = orig_kv
    _test_hooks.rq_connection_factory = orig_rq_conn
    _test_hooks.rq_queue_factory = orig_queue
    _test_hooks.rq_retry_factory = orig_retry
    _test_hooks.load_settings = orig_load_settings
    _test_hooks.artifact_store_factory = orig_artifact_store
    _test_hooks.service_container_from_settings = orig_service_container
    _test_hooks.corpus_fetcher_factory = orig_corpus_fetcher
    _test_hooks.load_tokenizer_for_training = orig_load_tokenizer
    _test_hooks.httpx_client_factory = orig_httpx_client
    # Training infrastructure hooks
    _test_hooks.cuda_is_available = orig_cuda_is_available
    _test_hooks.pkg_version = orig_pkg_version
    _test_hooks.model_dir = orig_model_dir
    _test_hooks.split_corpus_files = orig_split_corpus_files
    _test_hooks.freeze_embeddings = orig_freeze_embeddings
    _test_hooks.shutil_which = orig_shutil_which
    # SentencePiece backend hooks
    _test_hooks.spm_require_cli = orig_spm_require_cli
    _test_hooks.spm_train = orig_spm_train
    _test_hooks.spm_encode_ids = orig_spm_encode_ids
    # Additional hooks for edge-case testing
    _test_hooks.random_factory = orig_random_factory
    _test_hooks.shutil_rmtree = orig_shutil_rmtree
    _test_hooks.load_wandb_module = orig_load_wandb_module
    _test_hooks.load_gpt2_model = orig_load_gpt2_model
    _test_hooks.sample_token = orig_sample_token
    _test_hooks.spm_decode_ids = orig_spm_decode_ids
    # Standard library hooks
    _test_hooks.os_scandir = orig_os_scandir
    _test_hooks.shutil_disk_usage = orig_shutil_disk_usage
    _test_hooks.path_unlink = orig_path_unlink
    _test_hooks.time_sleep = orig_time_sleep
    _test_hooks.path_iterdir = orig_path_iterdir
    # Cleanup service hooks
    _test_hooks.corpus_cache_cleanup_service_factory = orig_corpus_cache_cleanup_service
    _test_hooks.tokenizer_cleanup_service_factory = orig_tokenizer_cleanup_service
    # JSON hooks
    _test_hooks.dump_json_str = orig_dump_json_str
    # Orchestrator hooks
    _test_hooks.tokenizer_enqueue_hook = orig_tokenizer_enqueue_hook
    # GPT2 backend hooks
    _test_hooks.load_prepared_gpt2_from_handle = orig_load_prepared_gpt2_from_handle
    # Guard hooks
    _test_hooks.guard_find_monorepo_root = orig_guard_find_monorepo_root
    _test_hooks.guard_load_orchestrator = orig_guard_load_orchestrator
    # Platform core config hooks
    config_test_hooks.get_env = orig_get_env
    # Platform ML torch hooks
    platform_ml_torch_types._import_torch = orig_platform_ml_import_torch


fake_redis = pytest.fixture(_make_fake_redis)
fake_queue = pytest.fixture(_make_fake_queue)
_reset_test_hooks = pytest.fixture(autouse=True)(_reset_test_hooks_impl)


def _build_settings(
    *,
    artifacts_root: str | None = None,
    runs_root: str | None = None,
    logs_root: str | None = None,
    data_root: str | None = None,
    data_bank_api_url: str | None = None,
    data_bank_api_key: str | None = None,
    threads: int | None = None,
    redis_url: str | None = None,
    app_env: Literal["dev", "prod"] | None = None,
    security_api_key: str | None = None,
) -> Settings:
    base = load_settings()
    _apply_app_overrides(
        base,
        artifacts_root=artifacts_root,
        runs_root=runs_root,
        logs_root=logs_root,
        data_root=data_root,
        data_bank_api_url=data_bank_api_url,
        data_bank_api_key=data_bank_api_key,
        threads=threads,
    )
    if redis_url is not None:
        base["redis"]["url"] = redis_url
    if app_env is not None:
        base["app_env"] = app_env
    if security_api_key is not None:
        base["security"]["api_key"] = security_api_key
    return base


def _apply_app_overrides(
    base: Settings,
    *,
    artifacts_root: str | None,
    runs_root: str | None,
    logs_root: str | None,
    data_root: str | None,
    data_bank_api_url: str | None,
    data_bank_api_key: str | None,
    threads: int | None,
) -> None:
    if artifacts_root is not None:
        base["app"]["artifacts_root"] = artifacts_root
    if runs_root is not None:
        base["app"]["runs_root"] = runs_root
    if logs_root is not None:
        base["app"]["logs_root"] = logs_root
    if data_root is not None:
        base["app"]["data_root"] = data_root
    if data_bank_api_url is not None:
        base["app"]["data_bank_api_url"] = data_bank_api_url
    if data_bank_api_key is not None:
        base["app"]["data_bank_api_key"] = data_bank_api_key
    if threads is not None:
        base["app"]["threads"] = threads


def _make_settings_factory() -> SettingsFactory:
    return _build_settings


settings_factory = pytest.fixture(_make_settings_factory)


def _make_settings_with_paths(tmp_path: Path, settings_factory: SettingsFactory) -> Settings:
    return settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )


settings_with_paths = pytest.fixture(_make_settings_with_paths)
