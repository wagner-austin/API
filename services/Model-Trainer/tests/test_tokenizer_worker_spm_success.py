from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from _pytest.monkeypatch import MonkeyPatch
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.core.contracts.tokenizer import TokenizerTrainStats
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


def test_tokenizer_worker_spm_success(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    fake = FakeRedis()

    def _redis_for_kv(url: str) -> FakeRedis:
        return fake

    monkeypatch.setattr(
        "model_trainer.worker.tokenizer_worker.redis_for_kv",
        _redis_for_kv,
    )

    # Force CLI available
    def _which(_name: str) -> str:
        return "spm"

    monkeypatch.setattr(
        "model_trainer.worker.tokenizer_worker.shutil.which",
        _which,
    )

    # Stub the backend class imported inside the worker branch
    class _SPMBackend:
        def train(
            self: _SPMBackend, cfg: dict[str, str | int | float | bool]
        ) -> TokenizerTrainStats:
            return TokenizerTrainStats(
                coverage=0.9,
                oov_rate=0.1,
                token_count=10,
                char_coverage=0.8,
            )

    monkeypatch.setattr(
        "model_trainer.core.services.tokenizer.spm_backend.SentencePieceBackend",
        _SPMBackend,
        raising=True,
    )

    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        data_root=str(tmp_path / "data"),
    )

    def _load_settings() -> Settings:
        return settings

    monkeypatch.setattr("model_trainer.worker.tokenizer_worker.load_settings", _load_settings)
    # Minimal corpus
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "a.txt").write_text("hi\n", encoding="utf-8")

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-spm-succ",
        "method": "sentencepiece",
        "vocab_size": 64,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    # Stub fetcher to return local corpus dir
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            return tmp_path / "c"

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:tok-spm-succ:status") == "completed"
    stats_json = fake.get("tokenizer:tok-spm-succ:stats")
    assert isinstance(stats_json, str) and "oov_rate" in stats_json
    fake.assert_only_called({"set", "get", "publish"})
