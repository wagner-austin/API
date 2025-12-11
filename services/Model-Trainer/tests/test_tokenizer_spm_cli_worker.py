from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from fastapi.testclient import TestClient
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from model_trainer.api.main import create_app
from model_trainer.core import _test_hooks
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


def _create_fake_spm_train(model_prefix_box: list[str]) -> None:
    """Create fake spm_train that writes model/vocab files."""

    def _fake_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
        model_path = Path(model_prefix + ".model")
        vocab_path = Path(model_prefix + ".vocab")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"\x00\x01mock")
        vocab_path.write_text("[UNK]\t0\nA\t0\nB\t0\n", encoding="utf-8")
        model_prefix_box.append(model_prefix)

    _test_hooks.spm_train = _fake_train


def _create_fake_spm_encode_ids() -> None:
    """Create fake spm_encode_ids that returns token IDs."""

    def _fake_encode(model_path: str, text: str) -> list[int]:
        return [1, 2, 3]

    _test_hooks.spm_encode_ids = _fake_encode


def test_sentencepiece_orchestrator_fails_without_cli() -> None:
    # Force CLI to be unavailable via shutil_which hook
    def _which_none(cmd: str) -> None:
        return None

    _test_hooks.shutil_which = _which_none

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
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    # Pretend CLI exists via shutil_which hook
    def _which_path(cmd: str) -> str:
        return "/usr/bin/spm_mock"

    _test_hooks.shutil_which = _which_path

    # Set up fake spm_require_cli to do nothing (CLI "exists")
    def _noop_require_cli() -> None:
        pass

    _test_hooks.spm_require_cli = _noop_require_cli

    # Set up fake spm_train via hook
    model_prefix_box: list[str] = []
    _create_fake_spm_train(model_prefix_box)

    # Set up fake spm_encode_ids via hook
    _create_fake_spm_encode_ids()

    # Set up fake redis via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    # Set up settings via hook
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        data_root=str(tmp_path / "data"),
    )

    def _load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _load_settings

    # Set up corpus fetcher via hook
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is spm\n", encoding="utf-8")

    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(api_url, api_key, cache_dir, corpus_path=corpus)

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    payload: TokenizerTrainPayload = {
        "tokenizer_id": "tok-spm",
        "method": "sentencepiece",
        "vocab_size": 128,
        "min_frequency": 1,
        "corpus_file_id": "deadbeef",
        "holdout_fraction": 0.1,
        "seed": 1,
    }
    process_tokenizer_train_job(payload)
    assert fake.get("tokenizer:tok-spm:status") == "completed"
    out_dir = artifacts / "tokenizers" / "tok-spm"
    assert (out_dir / "tokenizer.model").exists()
    assert (out_dir / "manifest.json").exists()
    fake.assert_only_called({"set", "get", "publish"})
