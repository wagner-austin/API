from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.worker.tokenizer_worker import process_tokenizer_train_job


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        runs_root: str | None = ...,
        logs_root: str | None = ...,
        data_root: str | None = ...,
        data_bank_api_url: str | None = ...,
        data_bank_api_key: str | None = ...,
        threads: int | None = ...,
        redis_url: str | None = ...,
        app_env: Literal["dev", "prod"] | None = ...,
        security_api_key: str | None = ...,
    ) -> Settings: ...


class _FakeCorpusFetcher:
    """Fake CorpusFetcher for tests."""

    def __init__(self: _FakeCorpusFetcher, api_url: str, api_key: str, cache_dir: Path) -> None:
        pass

    def fetch(self: _FakeCorpusFetcher, fid: str) -> Path:
        return Path("/data")


def test_tokenizer_worker_sentencepiece_unavailable_sets_failed(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Test that SPM tokenizer sets failed status when CLI is unavailable."""
    # Set up hooks for settings
    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        data_root=str(tmp_path / "data"),
    )

    def _load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _load_settings

    # Ensure CLI not found via shutil_which hook
    def _which(cmd: str) -> str | None:
        return None

    _test_hooks.shutil_which = _which

    # Set up fake redis via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    # Stub fetcher via hook
    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(api_url, api_key, cache_dir)

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-spm",
        "method": "sentencepiece",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 42,
    }
    process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:tok-spm:status") == "failed"
    fake.assert_only_called({"set", "get", "publish"})
