from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Protocol

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient
from platform_workers.testing import FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TokenizerTrainPayload
from model_trainer.core.services.container import ServiceContainer
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


class _Proc:
    """Mock subprocess result."""

    def __init__(self: _Proc, stdout: str) -> None:
        self.stdout: str = stdout


class _FakeSpmRunProto(Protocol):
    """Protocol for fake subprocess.run callable."""

    def __call__(
        self,
        args: list[str] | tuple[str, ...],
        *,
        check: bool = ...,
        capture_output: bool = ...,
        text: bool = ...,
        input: str | None = ...,
        cwd: str | None = ...,
        env: dict[str, str] | None = ...,
        timeout: float | None = ...,
    ) -> _Proc: ...


class _FakeCorpusFetcher:
    """Fake corpus fetcher that returns a local path."""

    def __init__(
        self: _FakeCorpusFetcher,
        api_url: str,
        api_key: str,
        cache_dir: Path,
        *,
        corpus_path: Path,
    ) -> None:
        self._corpus_path = corpus_path

    def fetch(self: _FakeCorpusFetcher, fid: str) -> Path:
        return self._corpus_path


def _create_fake_spm_run(
    model_out_path_prefix: str | None = None,
) -> _FakeSpmRunProto:
    """Create a fake subprocess.run that simulates SPM CLI behavior."""

    def _fake_run(
        args: list[str] | tuple[str, ...],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        input: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _Proc:
        cmd = str(args[0])
        if "spm_train" in cmd:
            prefix = next(a.split("=", 1)[1] for a in args if str(a).startswith("--model_prefix="))
            model_path = Path(prefix + ".model")
            vocab_path = Path(prefix + ".vocab")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"\x00\x01mock")
            vocab_path.write_text("[UNK]\t0\nA\t0\nB\t0\n", encoding="utf-8")
            return _Proc("")
        if "spm_encode" in cmd:
            return _Proc("1 2 3\n")
        if "spm_decode" in cmd:
            return _Proc(input or "")
        return _Proc("")

    return _fake_run


def test_sentencepiece_orchestrator_fails_without_cli(monkeypatch: MonkeyPatch) -> None:
    # Force CLI to be unavailable regardless of host
    def _which(name: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _which, raising=True)
    app = create_app()
    container: ServiceContainer = app.state.container
    fake = FakeRedis()
    container.redis = fake
    client = TestClient(app)
    body = {
        "method": "sentencepiece",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    r = client.post("/tokenizers/train", json=body)
    assert r.status_code == 400
    fake.assert_only_called(set())


def test_sentencepiece_worker_trains_and_writes_artifacts_with_mocked_cli(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    # Pretend CLI exists
    def _which(name: str) -> str:
        return "spm_mock"

    monkeypatch.setattr(shutil, "which", _which, raising=True)
    monkeypatch.setattr(
        "model_trainer.core.services.tokenizer.spm_backend.subprocess.run",
        _create_fake_spm_run(),
        raising=True,
    )
    fake = FakeRedis()

    def _redis_for_kv(url: str) -> FakeRedis:
        return fake

    monkeypatch.setattr(
        "model_trainer.worker.tokenizer_worker.redis_for_kv",
        _redis_for_kv,
    )

    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        data_root=str(tmp_path / "data"),
    )

    def _load_settings() -> Settings:
        return settings

    monkeypatch.setattr("model_trainer.worker.tokenizer_worker.load_settings", _load_settings)

    # Minimal corpus
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is spm\n", encoding="utf-8")

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-spm",
        "method": "sentencepiece",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    # Stub fetcher to return the local corpus directory
    from functools import partial

    from model_trainer.core.services.data import corpus_fetcher as cf

    monkeypatch.setattr(cf, "CorpusFetcher", partial(_FakeCorpusFetcher, corpus_path=corpus))
    process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:tok-spm:status") == "completed"
    out_dir = artifacts / "tokenizers" / "tok-spm"
    assert (out_dir / "tokenizer.model").exists()
    assert (out_dir / "manifest.json").exists()
    fake.assert_only_called({"set", "get", "publish"})
