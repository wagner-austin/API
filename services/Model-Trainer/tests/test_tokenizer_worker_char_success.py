from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from _pytest.monkeypatch import MonkeyPatch
from platform_workers.testing import FakeRedis

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


def test_tokenizer_worker_char_success(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    fake = FakeRedis()

    def _redis_for_kv(url: str) -> FakeRedis:
        return fake

    monkeypatch.setattr(
        "model_trainer.worker.tokenizer_worker.redis_for_kv",
        _redis_for_kv,
    )

    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        data_root=str(tmp_path / "data"),
    )

    def _load_settings() -> Settings:
        return settings

    monkeypatch.setattr("model_trainer.worker.tokenizer_worker.load_settings", _load_settings)

    # Minimal corpus with a couple of characters
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("aba\nbaa\n", encoding="utf-8")

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-char",
        "method": "char",
        "vocab_size": 0,  # ignored by char backend
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }

    # Stub CorpusFetcher to return the local corpus directory
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return corpus

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)

    process_tokenizer_train_job(payload)

    assert fake.get("tokenizer:tok-char:status") == "completed"
    stats_json = fake.get("tokenizer:tok-char:stats")
    # Verify stats is valid JSON string with expected fields
    assert (
        isinstance(stats_json, str)
        and "token_count" in stats_json
        and "char_coverage" in stats_json
    )
