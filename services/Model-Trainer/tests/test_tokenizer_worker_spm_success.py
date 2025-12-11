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


def test_tokenizer_worker_spm_success(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    """Test SPM tokenizer training completes successfully."""
    # Fake redis via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    # Force CLI available via shutil_which hook
    def _which(cmd: str) -> str | None:
        return "spm"

    _test_hooks.shutil_which = _which

    # Stub spm_require_cli to not raise
    def _noop_require_cli() -> None:
        pass

    _test_hooks.spm_require_cli = _noop_require_cli

    # Stub spm_train to create mock files
    def _fake_spm_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
        model_path = Path(model_prefix + ".model")
        vocab_path = Path(model_prefix + ".vocab")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"\x00\x01mock")
        vocab_path.write_text("[UNK]\t0\nA\t0\nB\t0\n", encoding="utf-8")

    _test_hooks.spm_train = _fake_spm_train

    # Stub spm_encode_ids to return token IDs
    def _fake_spm_encode_ids(model_path: str, text: str) -> list[int]:
        return [1, 2, 3]

    _test_hooks.spm_encode_ids = _fake_spm_encode_ids

    # Settings via hook
    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        data_root=str(tmp_path / "data"),
    )

    def _load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _load_settings

    # Minimal corpus
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "a.txt").write_text("hi\n", encoding="utf-8")

    # Stub corpus fetcher via hook
    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(tmp_path / "c")

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-spm-succ",
        "method": "sentencepiece",
        "vocab_size": 64,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }

    process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:tok-spm-succ:status") == "completed"
    stats_json = fake.get("tokenizer:tok-spm-succ:stats")
    assert isinstance(stats_json, str) and "oov_rate" in stats_json
    fake.assert_only_called({"set", "get", "publish"})
