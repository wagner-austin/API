"""Tests to cover remaining coverage gaps for 100% statement and branch coverage."""

from __future__ import annotations

from collections.abc import Mapping as _Mapping
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue
from platform_ml.testing import (
    WandbConfigProtocol,
    WandbModuleProtocol,
    WandbRunProtocol,
    WandbTableCtorProtocol,
    WandbTableProtocol,
)
from platform_workers.testing import FakeRedis, FakeRedisNonRedisError

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.infra.paths import tokenizer_logs_path
from model_trainer.core.infra.redis_utils import get_with_retry, set_with_retry
from model_trainer.core.logging.utils import narrow_log_level
from model_trainer.core.services.data.corpus import list_text_files

# --- paths.py coverage: line 22 (tokenizer_logs_path) ---


def test_tokenizer_logs_path_returns_expected_path() -> None:
    """Cover tokenizer_logs_path function (paths.py line 22)."""
    settings = load_settings()
    tok_id = "test-tok-123"
    result = tokenizer_logs_path(settings, tok_id)
    assert result.name == "logs.jsonl"
    assert tok_id in str(result)


# --- logging/utils.py coverage: lines 16, 19-25 (narrow_log_level branches) ---


def test_narrow_log_level_debug() -> None:
    """Cover DEBUG branch."""
    assert narrow_log_level("DEBUG") == "DEBUG"


def test_narrow_log_level_info() -> None:
    """Cover INFO branch."""
    assert narrow_log_level("INFO") == "INFO"


def test_narrow_log_level_warning() -> None:
    """Cover WARNING branch (line 19)."""
    assert narrow_log_level("WARNING") == "WARNING"


def test_narrow_log_level_error() -> None:
    """Cover ERROR branch (line 21)."""
    assert narrow_log_level("ERROR") == "ERROR"


def test_narrow_log_level_critical() -> None:
    """Cover CRITICAL branch (line 23)."""
    assert narrow_log_level("CRITICAL") == "CRITICAL"


def test_narrow_log_level_unknown_defaults_to_info() -> None:
    """Cover default branch (line 25)."""
    assert narrow_log_level("UNKNOWN") == "INFO"
    assert narrow_log_level("") == "INFO"
    assert narrow_log_level("debug") == "INFO"  # Case sensitive


# --- redis_utils.py coverage: lines 18, 35 (non-Redis error re-raises) ---


class _FakeRedisGetNonRedisError(FakeRedis):
    """FakeRedis that raises a non-Redis error on get."""

    def get(self, key: str) -> str | None:
        self._record("get")
        raise ValueError("not a redis error")


def test_redis_utils_get_non_redis_error_reraises() -> None:
    """Cover redis_utils.py line 18 (non-Redis error re-raises in get_with_retry)."""
    client = _FakeRedisGetNonRedisError()

    with pytest.raises(ValueError, match="not a redis error"):
        get_with_retry(client, "key", attempts=3)
    client.assert_only_called({"get"})


class _FakeRedisSetNonRedisError(FakeRedis):
    """FakeRedis that raises a non-Redis error on set."""

    def set(self, key: str, value: str) -> bool:
        self._record("set")
        raise ValueError("not a redis error")


def test_redis_utils_set_non_redis_error_reraises() -> None:
    """Cover redis_utils.py line 35 (non-Redis error re-raises in set_with_retry)."""
    client = _FakeRedisSetNonRedisError()

    with pytest.raises(ValueError, match="not a redis error"):
        set_with_retry(client, "key", "value", attempts=3)
    client.assert_only_called({"set"})


# --- corpus.py coverage: branch 15->14 (non-.txt files skipped) ---


def test_list_text_files_skips_non_txt_files(tmp_path: Path) -> None:
    """Cover corpus.py branch where non-.txt files are skipped (line 15->14)."""
    # Create a directory with mixed file types
    (tmp_path / "file1.txt").write_text("text file", encoding="utf-8")
    (tmp_path / "file2.json").write_text("{}", encoding="utf-8")
    (tmp_path / "file3.py").write_text("# python", encoding="utf-8")
    (tmp_path / "file4.text").write_text("another text", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")
    (tmp_path / "subdir" / "nested.md").write_text("# markdown", encoding="utf-8")

    result = list_text_files(str(tmp_path))

    # Should only include .txt and .text files
    assert len(result) == 3
    filenames = [Path(p).name for p in result]
    assert "file1.txt" in filenames
    assert "file4.text" in filenames
    assert "nested.txt" in filenames
    # Should NOT include non-txt files
    assert "file2.json" not in filenames
    assert "file3.py" not in filenames
    assert "nested.md" not in filenames


# --- container.py coverage: line 88 (wrong queue name raises) ---


def test_container_wrong_queue_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover container.py line 88 (ValueError when queue_name != TRAINER_QUEUE)."""

    from model_trainer.core.services.container import _create_enqueuer

    settings = load_settings()
    # Modify settings to have wrong queue name
    wrong_settings: Settings = {
        **settings,
        "rq": {
            **settings["rq"],
            "queue_name": "wrong-queue-name",
        },
    }
    with pytest.raises(ValueError, match="RQ queue must be trainer per platform alignment"):
        _create_enqueuer(wrong_settings)


# --- health.py coverage: line 57 (non-Redis error re-raises in readyz) ---


def test_health_readyz_non_redis_error_reraises() -> None:
    """Cover health.py line 57 (non-Redis error re-raises in readyz).

    Uses TestClient to properly invoke the readyz endpoint through FastAPI.
    """
    from fastapi.testclient import TestClient

    from model_trainer.api.main import create_app
    from model_trainer.core import _test_hooks

    # Use shared stub that raises non-Redis error on ping
    fr = FakeRedisNonRedisError()

    def _fake_redis_for_kv(url: str) -> FakeRedisNonRedisError:
        return fr

    orig_kv = _test_hooks.kv_store_factory
    _test_hooks.kv_store_factory = _fake_redis_for_kv

    try:
        app = create_app()
        client = TestClient(app, raise_server_exceptions=True)

        with pytest.raises(RuntimeError, match="simulated non-Redis failure"):
            client.get("/readyz")

        fr.assert_only_called({"ping"})
    finally:
        _test_hooks.kv_store_factory = orig_kv


# --- tokenizer_worker.py coverage: line 24 (empty Redis URL raises) ---


def test_tokenizer_worker_empty_redis_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover tokenizer_worker.py line 24 (RuntimeError when Redis URL is empty)."""
    from model_trainer.worker.tokenizer_worker import _redis_client

    settings = load_settings()
    # Modify settings to have empty Redis URL
    empty_redis_settings: Settings = {
        **settings,
        "redis": {
            **settings["redis"],
            "url": "   ",  # Empty/whitespace URL
        },
    }
    with pytest.raises(RuntimeError, match="Redis URL must not be empty"):
        _redis_client(empty_redis_settings)


# --- train_job.py coverage: (non-Redis error when recording error) ---
# This is covered via integration tests in test_training_worker_error_handling.py
# The pattern requires triggering a non-Redis error during _handle_train_error after
# a training failure, which is tested through the full worker flow.


# --- train_job.py coverage: lines 58-65 (_create_wandb_publisher branches) ---


# Helper classes for wandb testing - extracted to reduce test complexity


class _FakeWandbRunForCoverageTest(WandbRunProtocol):
    """Fake wandb run for coverage testing."""

    _id: str

    def __init__(self, run_id: str) -> None:
        self._id = run_id

    @property
    def id(self) -> str:
        return self._id


class _FakeWandbConfigForCoverageTest(WandbConfigProtocol):
    """Fake wandb config for coverage testing."""

    def update(self, d: _Mapping[str, JSONValue]) -> None:
        pass


class _FakeWandbTableForCoverageTest(WandbTableProtocol):
    """Fake wandb table for coverage testing."""

    @property
    def columns(self) -> list[str]:
        return []

    @property
    def data(self) -> list[list[float | int | str | bool]]:
        return []


class _FakeWandbTableCtorForCoverageTest(WandbTableCtorProtocol):
    """Fake wandb table constructor for coverage testing."""

    def __call__(
        self,
        columns: list[str],
        data: list[list[float | int | str | bool]],
    ) -> WandbTableProtocol:
        return _FakeWandbTableForCoverageTest()


class _FakeWandbModuleForCoverageTest(WandbModuleProtocol):
    """Fake wandb module for coverage testing."""

    _run: WandbRunProtocol | None
    _config: WandbConfigProtocol
    _table_ctor: WandbTableCtorProtocol

    def __init__(self) -> None:
        self._run = None
        self._config = _FakeWandbConfigForCoverageTest()
        self._table_ctor = _FakeWandbTableCtorForCoverageTest()

    @property
    def run(self) -> WandbRunProtocol | None:
        return self._run

    @property
    def config(self) -> WandbConfigProtocol:
        return self._config

    @property
    def table_ctor(self) -> WandbTableCtorProtocol:
        return self._table_ctor

    def init(self, *, project: str, name: str) -> WandbRunProtocol:
        self._run = _FakeWandbRunForCoverageTest("fake-run-id")
        return self._run

    def log(
        self,
        data: _Mapping[str, float | int | str | bool | WandbTableProtocol],
    ) -> None:
        pass

    def finish(self) -> None:
        pass


def _create_fake_wandb_module_for_coverage() -> WandbModuleProtocol:
    """Create a fake wandb module for coverage testing."""
    return _FakeWandbModuleForCoverageTest()


def test_create_wandb_publisher_enabled_returns_publisher() -> None:
    """Cover train_job.py lines 58-62 (wandb enabled branch)."""
    from platform_ml.testing import hooks as wandb_hooks

    from model_trainer.worker.train_job import _create_wandb_publisher

    settings = load_settings()
    # Enable wandb in settings
    enabled_settings: Settings = {
        **settings,
        "wandb": {
            "enabled": True,
            "project": "test-project",
        },
    }

    fake_wandb = _create_fake_wandb_module_for_coverage()

    # Use platform_ml hook to inject fake wandb module
    def _fake_load_wandb_module() -> WandbModuleProtocol:
        return fake_wandb

    wandb_hooks.load_wandb_module = _fake_load_wandb_module

    result = _create_wandb_publisher(enabled_settings, "run-123", "gpt2")

    # Strong assertion: check the publisher was created and is enabled
    # First narrow the type - result must not be None when enabled
    if result is None:
        raise AssertionError("Expected WandbPublisher but got None")
    init_result = result.get_init_result()
    assert init_result["status"] == "enabled"


def test_create_wandb_publisher_unavailable_returns_none() -> None:
    """Cover train_job.py lines 63-65 (WandbUnavailableError branch)."""
    from platform_ml.testing import hooks as wandb_hooks
    from platform_ml.wandb_publisher import WandbUnavailableError

    from model_trainer.worker.train_job import _create_wandb_publisher

    settings = load_settings()
    # Enable wandb in settings
    enabled_settings: Settings = {
        **settings,
        "wandb": {
            "enabled": True,
            "project": "test-project",
        },
    }

    # Use platform_ml hook to raise WandbUnavailableError
    def _raise_unavailable() -> WandbModuleProtocol:
        raise WandbUnavailableError("wandb not installed")

    wandb_hooks.load_wandb_module = _raise_unavailable

    result = _create_wandb_publisher(enabled_settings, "run-456", "char_lstm")

    # Result should be None when wandb is unavailable
    assert result is None
