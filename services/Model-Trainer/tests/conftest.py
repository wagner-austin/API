from __future__ import annotations

import faulthandler
from collections.abc import Generator
from pathlib import Path
from typing import Final, Literal, Protocol

import pytest
from platform_core.config import config_test_hooks
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
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
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks
from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks as FtHooks
from model_trainer.core.services.model.backends.hf_lm._test_hooks import Hooks as HfLmHooks

# Use the import to cache sentencepiece in sys.modules with SWIG warnings suppressed
_ = _spm_init

UNPINNED = determinism_record(UNPINNED_STACK, {})
"""The posture a test process actually ran under, which is none.

`determinism` has no default anywhere in the training chain, so every caller
has to state what was in force. For a test that is not "unknown" -- the test
pinned nothing, and `UNPINNED_STACK` says exactly that. Shared here so nine
test modules assert the same fact rather than nine spellings of it.
"""


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
    orig_split_corpus = _test_hooks.split_corpus
    orig_reload_shipped = _test_hooks.reload_shipped_weights
    orig_time_wall_clock = _test_hooks.time_wall_clock
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
    orig_tokenizer_enqueue = _test_hooks.tokenizer_enqueue
    # GPT2 backend hooks
    orig_load_prepared_gpt2_from_handle = _test_hooks.load_prepared_gpt2_from_handle
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
    _test_hooks.split_corpus = orig_split_corpus
    _test_hooks.reload_shipped_weights = orig_reload_shipped
    _test_hooks.time_wall_clock = orig_time_wall_clock
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
    _test_hooks.tokenizer_enqueue = orig_tokenizer_enqueue
    # GPT2 backend hooks
    _test_hooks.load_prepared_gpt2_from_handle = orig_load_prepared_gpt2_from_handle
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


#: How long one test may run before its worker starts dumping stacks.
#:
#: MEASURED AGAINST A REAL HANG. On 2026-09-10 a CI worker sat inside one
#: test for 10322s -- 2h52m, eleven times the 900s ``pytest-timeout`` bound,
#: which recorded ZERO firings. The job was eventually cancelled, and the
#: cancel destroyed the only thing that would have identified the cause: a
#: stack for the blocked thread. Four sessions spent twelve hours inferring
#: from metadata that a single stack dump would have settled.
#:
#: 600 rather than 300: this package's slowest test alone is 99.18s and the
#: parallel case is several times that, and ``pyproject.toml`` records 300s
#: firing on slow-but-working tests rather than on hangs. 600 is six times
#: the slowest measured test and two thirds of the ``pytest-timeout`` bound,
#: so a dump lands BEFORE anything tries to kill the worker.
_HANG_DUMP_SECONDS: Final[int] = 600


@pytest.fixture(autouse=True)
def _dump_stacks_when_a_test_stops_making_progress() -> Generator[None, None, None]:
    """Print every thread's stack if one test runs impossibly long.

    IT DOES NOT KILL ANYTHING, and that is the point. ``pytest-timeout``'s
    thread method ends a hung worker with ``os._exit(1)``, which leaves the
    xdist controller on a dead channel -- and on the occurrence above it did
    not fire at all, so the suite hung with no diagnosis and no bound.
    Dumping is strictly additive: a healthy run never reaches the deadline, a
    hung one leaves a stack in the captured output, and nothing about the
    existing timeout changes.

    ``repeat=True`` because one dump shows where a thread is, and several
    spaced dumps show whether it is stuck there or merely slow -- which is
    the distinction the whole 2026-09-10 investigation turned on.

    The dump reaches the job log because a CANCELLED job's runner shuts down
    cleanly and uploads its output; it is the force-closed-over-a-dead-runner
    case that yields nothing, and that case yields nothing either way.

    Yields:
        None, once the deadline is armed for the test about to run.
    """
    faulthandler.dump_traceback_later(_HANG_DUMP_SECONDS, repeat=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def _reset_hook_containers() -> Generator[None, None, None]:
    """Restore the class-level hook containers around every test.

    Named on the containers rather than as bare reset_hooks() calls so the
    isolation is attributable: tests below this conftest may assign
    `Hooks.<attr>` knowing each attribute is restored per test.
    """
    Hooks.reset()
    FtHooks.reset()
    HfLmHooks.reset()
    yield
    Hooks.reset()
    FtHooks.reset()
    HfLmHooks.reset()
