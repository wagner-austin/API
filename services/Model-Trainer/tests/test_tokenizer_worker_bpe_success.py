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

    def __init__(self: _FakeCorpusFetcher, corpus_path: Path) -> None:
        self._corpus_path = corpus_path

    def fetch(self: _FakeCorpusFetcher, fid: str) -> Path:
        return self._corpus_path


def test_tokenizer_worker_bpe_success(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    """Test BPE tokenizer training completes successfully."""
    artifacts = tmp_path / "artifacts"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nhello ai\n", encoding="utf-8")

    # Fake redis via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    # Settings via hook
    settings = settings_factory(artifacts_root=str(artifacts), data_root=str(tmp_path / "data"))

    def _load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _load_settings

    # Stub corpus fetcher via hook
    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(corpus)

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    # Execute BPE path to full completion (hits final logging extras)
    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-bpe",
        "method": "bpe",
        "vocab_size": 64,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }

    process_tokenizer_train_job(payload)

    # Assert status and stats were stored (exercising end-of-function lines)
    assert fake.get("tokenizer:tok-bpe:status") == "completed"
    stats_json = fake.get("tokenizer:tok-bpe:stats")
    assert isinstance(stats_json, str) and "coverage" in stats_json
    fake.assert_only_called({"set", "get", "publish"})
