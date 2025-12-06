"""Tests for process_score_job and process_generate_job worker functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.trainer_keys import artifact_file_id_key, generate_key, score_key
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import (
    BackendCapabilities,
    EvalOutcome,
    GenerateConfig,
    GenerateOutcome,
    ModelArtifact,
    ModelBackend,
    ModelTrainConfig,
    PreparedLMModel,
    ScoreConfig,
    ScoreOutcome,
    TrainOutcome,
)
from model_trainer.core.contracts.queue import GenerateJobPayload, ScoreJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import ListEncoded
from model_trainer.core.services.model.unavailable_backend import UNAVAILABLE_CAPABILITIES
from model_trainer.core.types import ForwardOutProto, NamedParameter, ParameterLike
from model_trainer.worker import generate_job, score_job


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


def _make_manifest(model_family: str, tokenizer_id: str) -> str:
    manifest = {
        "model_family": model_family,
        "tokenizer_id": tokenizer_id,
        "run_id": "run123",
        "model_size": "small",
        "epochs": 1,
        "batch_size": 4,
        "max_seq_len": 64,
        "steps": 10,
        "loss": 0.5,
        "learning_rate": 0.001,
        "corpus_path": "/data/corpus",
        "holdout_fraction": 0.05,
        "optimizer": "adam",
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "device": "cpu",
        "precision": "fp32",
        "early_stopping_patience": 5,
        "test_split_ratio": 0.15,
        "finetune_lr_cap": 5e-5,
        "early_stopped": False,
        "seed": 42,
        "pretrained_run_id": None,
        "git_commit": "abc123",
        "versions": {
            "torch": "2.0.0",
            "transformers": "4.30.0",
            "tokenizers": "0.13.0",
            "datasets": "2.10.0",
        },
        "system": {
            "cpu_count": 4,
            "platform": "Linux",
            "platform_release": "5.15.0",
            "machine": "x86_64",
        },
    }
    return dump_json_str(manifest)


class _FakeTokenizerHandle(TokenizerHandle):
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def token_to_id(self, token: str) -> int | None:
        if len(token) == 1:
            return ord(token)
        return None

    def get_vocab_size(self) -> int:
        return 256


class _FakeConfigLike:
    """Fake config for LMModelProto."""

    n_positions: int = 64


class _FakeForwardOut:
    """Fake forward output that satisfies ForwardOutProto."""

    @property
    def loss(self) -> torch.Tensor:
        return torch.tensor(0.0)


class _FakeLMModel:
    """Fake language model that satisfies LMModelProto."""

    def __init__(self) -> None:
        self.config = _FakeConfigLike()

    @classmethod
    def from_pretrained(cls, path: str) -> _FakeLMModel:
        return cls()

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        return _FakeForwardOut()

    def parameters(self) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self, device: str) -> _FakeLMModel:
        return self

    def save_pretrained(self, out_dir: str) -> None:
        pass


class _FakeEncoder:
    """Fake encoder that satisfies Encoder protocol."""

    def __init__(self, handle: _FakeTokenizerHandle) -> None:
        self._h = handle

    def encode(self, text: str) -> ListEncoded:
        return ListEncoded(self._h.encode(text))

    def decode(self, ids: list[int]) -> str:
        return self._h.decode(ids)

    def token_to_id(self, token: str) -> int | None:
        return self._h.token_to_id(token)

    def get_vocab_size(self) -> int:
        return self._h.get_vocab_size()


def _make_fake_prepared() -> PreparedLMModel:
    """Create a fake PreparedLMModel for testing."""
    handle = _FakeTokenizerHandle()
    return PreparedLMModel(
        model=_FakeLMModel(),
        tokenizer_id="fake_tok",
        eos_id=0,
        pad_id=1,
        max_seq_len=64,
        tok_for_dataset=_FakeEncoder(handle),
    )


class _FakeBackendWithTopk:
    """Fake backend that returns score results with topk."""

    def name(self) -> str:
        return "fake_with_topk"

    def capabilities(self) -> BackendCapabilities:
        return UNAVAILABLE_CAPABILITIES

    def prepare(
        self,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        return _make_fake_prepared()

    def save(self, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        return ModelArtifact(out_dir=out_dir)

    def load(
        self,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        return _make_fake_prepared()

    def train(
        self,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: PreparedLMModel,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
    ) -> TrainOutcome:
        return TrainOutcome(
            loss=0.5,
            perplexity=1.5,
            steps=10,
            out_dir="",
            cancelled=False,
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def evaluate(
        self,
        *,
        run_id: str,
        cfg: ModelTrainConfig,
        settings: Settings,
    ) -> EvalOutcome:
        return EvalOutcome(loss=0.5, perplexity=1.5)

    def score(
        self,
        *,
        prepared: PreparedLMModel,
        cfg: ScoreConfig,
        settings: Settings,
    ) -> ScoreOutcome:
        topk: list[list[tuple[str, float]]] = [[("a", 0.5), ("b", 0.3)]]
        return ScoreOutcome(
            loss=1.5,
            perplexity=4.5,
            surprisal=[0.5, 0.7],
            topk=topk,
            tokens=["h", "e", "l", "l", "o"],
        )

    def generate(
        self,
        *,
        prepared: PreparedLMModel,
        cfg: GenerateConfig,
        settings: Settings,
    ) -> GenerateOutcome:
        return GenerateOutcome(
            outputs=["generated text here"],
            steps=10,
            eos_terminated=[True],
        )


class _FakeBackendNoTopk:
    """Fake backend that returns score results without topk."""

    def name(self) -> str:
        return "fake_no_topk"

    def capabilities(self) -> BackendCapabilities:
        return UNAVAILABLE_CAPABILITIES

    def prepare(
        self,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        return _make_fake_prepared()

    def save(self, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        return ModelArtifact(out_dir=out_dir)

    def load(
        self,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        return _make_fake_prepared()

    def train(
        self,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: PreparedLMModel,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
    ) -> TrainOutcome:
        return TrainOutcome(
            loss=0.5,
            perplexity=1.5,
            steps=10,
            out_dir="",
            cancelled=False,
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def evaluate(
        self,
        *,
        run_id: str,
        cfg: ModelTrainConfig,
        settings: Settings,
    ) -> EvalOutcome:
        return EvalOutcome(loss=0.5, perplexity=1.5)

    def score(
        self,
        *,
        prepared: PreparedLMModel,
        cfg: ScoreConfig,
        settings: Settings,
    ) -> ScoreOutcome:
        return ScoreOutcome(
            loss=1.5,
            perplexity=4.5,
            surprisal=None,
            topk=None,
            tokens=None,
        )

    def generate(
        self,
        *,
        prepared: PreparedLMModel,
        cfg: GenerateConfig,
        settings: Settings,
    ) -> GenerateOutcome:
        return GenerateOutcome(
            outputs=["generated text here"],
            steps=10,
            eos_terminated=[True],
        )


class _FakeModelRegistry:
    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    def get(self, family: Literal["gpt2", "llama", "qwen", "char_lstm"]) -> ModelBackend:
        return self._backend


class _ContainerProto(Protocol):
    """Protocol for container with model_registry."""

    model_registry: _FakeModelRegistry


class _ContainerFactoryProto(Protocol):
    """Protocol for container factory with from_settings."""

    def from_settings(self, settings: Settings) -> _ContainerProto: ...


class _FakeContainer:
    def __init__(self, backend: ModelBackend) -> None:
        self.model_registry = _FakeModelRegistry(backend)


class _FakeContainerFactory:
    """Factory that creates _FakeContainer instances."""

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    def from_settings(self, settings: Settings) -> _FakeContainer:
        return _FakeContainer(self._backend)


def _make_container_factory(backend: ModelBackend) -> _FakeContainerFactory:
    """Create a container factory for testing."""
    return _FakeContainerFactory(backend)


class TestProcessScoreJob:
    def test_score_job_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test successful score job execution."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)
        run_dir = models_dir / "run123"
        run_dir.mkdir(parents=True)

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(_make_manifest("char_lstm", "tok123"), encoding="utf-8")

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run123"), "file123")

        monkeypatch.setattr(score_job, "load_settings", lambda: settings)

        def _fake_redis_client(s: Settings) -> FakeRedis:
            return fake_redis

        monkeypatch.setattr(score_job, "redis_client", _fake_redis_client)

        def _fake_load_tokenizer(s: Settings, tid: str) -> _FakeTokenizerHandle:
            return _FakeTokenizerHandle()

        monkeypatch.setattr(score_job, "load_tokenizer_for_training", _fake_load_tokenizer)

        backend = _FakeBackendWithTopk()
        container_factory = _make_container_factory(backend)
        monkeypatch.setattr(score_job, "ServiceContainer", container_factory)

        payload: ScoreJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "text": "hello",
            "path": None,
            "detail_level": "per_char",
            "top_k": 5,
            "seed": 42,
        }

        score_job.process_score_job(payload)

        cached = fake_redis.get(score_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "completed"
        assert obj.get("loss") == 1.5
        assert obj.get("perplexity") == 4.5

    def test_score_job_no_artifact_pointer_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test score job fails when artifact pointer is missing."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()

        monkeypatch.setattr(score_job, "load_settings", lambda: settings)

        def _fake_redis_client(s: Settings) -> FakeRedis:
            return fake_redis

        monkeypatch.setattr(score_job, "redis_client", _fake_redis_client)

        payload: ScoreJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "text": "hello",
            "path": None,
            "detail_level": "summary",
            "top_k": None,
            "seed": None,
        }

        with pytest.raises(AppError, match="artifact pointer not found"):
            score_job.process_score_job(payload)

        cached = fake_redis.get(score_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "failed"

    def test_score_job_no_topk_or_surprisal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test score job with no topk or surprisal in result."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)
        run_dir = models_dir / "run123"
        run_dir.mkdir(parents=True)

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(_make_manifest("char_lstm", "tok123"), encoding="utf-8")

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run123"), "file123")

        monkeypatch.setattr(score_job, "load_settings", lambda: settings)

        def _fake_redis_client(s: Settings) -> FakeRedis:
            return fake_redis

        monkeypatch.setattr(score_job, "redis_client", _fake_redis_client)

        def _fake_load_tokenizer(s: Settings, tid: str) -> _FakeTokenizerHandle:
            return _FakeTokenizerHandle()

        monkeypatch.setattr(score_job, "load_tokenizer_for_training", _fake_load_tokenizer)

        backend = _FakeBackendNoTopk()
        container_factory = _make_container_factory(backend)
        monkeypatch.setattr(score_job, "ServiceContainer", container_factory)

        payload: ScoreJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "text": "hello",
            "path": None,
            "detail_level": "summary",
            "top_k": None,
            "seed": None,
        }

        score_job.process_score_job(payload)

        cached = fake_redis.get(score_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "completed"
        assert obj.get("surprisal") is None
        assert obj.get("topk") is None
        assert obj.get("tokens") is None


class TestProcessGenerateJob:
    def test_generate_job_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test successful generate job execution."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)
        run_dir = models_dir / "run123"
        run_dir.mkdir(parents=True)

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(_make_manifest("char_lstm", "tok123"), encoding="utf-8")

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run123"), "file123")

        monkeypatch.setattr(generate_job, "load_settings", lambda: settings)

        def _fake_redis_client(s: Settings) -> FakeRedis:
            return fake_redis

        monkeypatch.setattr(generate_job, "redis_client", _fake_redis_client)

        def _fake_load_tokenizer(s: Settings, tid: str) -> _FakeTokenizerHandle:
            return _FakeTokenizerHandle()

        monkeypatch.setattr(generate_job, "load_tokenizer_for_training", _fake_load_tokenizer)

        backend = _FakeBackendWithTopk()
        container_factory = _make_container_factory(backend)
        monkeypatch.setattr(generate_job, "ServiceContainer", container_factory)

        payload: GenerateJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "prompt_text": "Hello",
            "prompt_path": None,
            "max_new_tokens": 10,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "stop_on_eos": True,
            "stop_sequences": [],
            "seed": 42,
            "num_return_sequences": 1,
        }

        generate_job.process_generate_job(payload)

        cached = fake_redis.get(generate_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "completed"
        assert obj.get("outputs") == ["generated text here"]
        assert obj.get("steps") == 10
        assert obj.get("eos_terminated") == [True]

    def test_generate_job_no_artifact_pointer_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test generate job fails when artifact pointer is missing."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()

        monkeypatch.setattr(generate_job, "load_settings", lambda: settings)

        def _fake_redis_client(s: Settings) -> FakeRedis:
            return fake_redis

        monkeypatch.setattr(generate_job, "redis_client", _fake_redis_client)

        payload: GenerateJobPayload = {
            "run_id": "run123",
            "request_id": "req123",
            "prompt_text": "Hello",
            "prompt_path": None,
            "max_new_tokens": 10,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "stop_on_eos": True,
            "stop_sequences": [],
            "seed": None,
            "num_return_sequences": 1,
        }

        with pytest.raises(AppError, match="artifact pointer not found"):
            generate_job.process_generate_job(payload)

        cached = fake_redis.get(generate_key("run123", "req123"))
        assert isinstance(cached, str) and len(cached) > 0
        obj = load_json_str(cached)
        assert isinstance(obj, dict) and obj.get("status") == "failed"


class _FakeArtifactStore:
    """Fake ArtifactStore for testing artifact download paths."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 600.0,
        include_manifest: bool = True,
        manifest_content: str | None = None,
    ) -> None:
        self._include_manifest = include_manifest
        self._manifest_content = manifest_content

    def download_artifact(
        self,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        out = dest_dir / expected_root
        out.mkdir(parents=True, exist_ok=True)
        if self._include_manifest and self._manifest_content is not None:
            manifest_path = out / "manifest.json"
            manifest_path.write_text(self._manifest_content, encoding="utf-8")
        return out


class TestArtifactDownloadPaths:
    """Tests for artifact download scenarios covering lines 846-852, 857, 958-964, 969."""

    def test_score_job_with_download(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test score job when artifact needs to be downloaded (covers lines 846-852)."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)
        # NOTE: We do NOT create run_dir - it will be "downloaded"

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run_download"), "file_download_123")

        monkeypatch.setattr(score_job, "load_settings", lambda: settings)

        def _fake_redis_client(s: Settings) -> FakeRedis:
            return fake_redis

        monkeypatch.setattr(score_job, "redis_client", _fake_redis_client)

        def _fake_load_tokenizer(s: Settings, tid: str) -> _FakeTokenizerHandle:
            return _FakeTokenizerHandle()

        monkeypatch.setattr(score_job, "load_tokenizer_for_training", _fake_load_tokenizer)

        backend = _FakeBackendWithTopk()
        container_factory = _make_container_factory(backend)
        monkeypatch.setattr(score_job, "ServiceContainer", container_factory)

        # Create factory function that captures manifest content
        manifest_content = _make_manifest("char_lstm", "tok123")

        def _make_fake_store(
            base_url: str, api_key: str, *, timeout_seconds: float = 600.0
        ) -> _FakeArtifactStore:
            return _FakeArtifactStore(
                base_url,
                api_key,
                timeout_seconds=timeout_seconds,
                include_manifest=True,
                manifest_content=manifest_content,
            )

        monkeypatch.setattr("platform_ml.ArtifactStore", _make_fake_store)

        payload: ScoreJobPayload = {
            "run_id": "run_download",
            "request_id": "req_download",
            "text": "hello",
            "path": None,
            "detail_level": "per_char",
            "top_k": 5,
            "seed": 42,
        }

        score_job.process_score_job(payload)

        cached = fake_redis.get(score_key("run_download", "req_download"))
        assert isinstance(cached, str) and len(cached) > 0
        result = load_json_str(cached)
        assert isinstance(result, dict) and result.get("status") == "completed"

    def test_score_job_missing_manifest_after_download(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test score job when manifest is missing after download (covers line 857)."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run_no_manifest"), "file_no_manifest_123")

        monkeypatch.setattr(score_job, "load_settings", lambda: settings)

        def _fake_redis_client(s: Settings) -> FakeRedis:
            return fake_redis

        monkeypatch.setattr(score_job, "redis_client", _fake_redis_client)

        # Create factory without manifest
        def _make_fake_store_no_manifest(
            base_url: str, api_key: str, *, timeout_seconds: float = 600.0
        ) -> _FakeArtifactStore:
            return _FakeArtifactStore(
                base_url,
                api_key,
                timeout_seconds=timeout_seconds,
                include_manifest=False,
                manifest_content=None,
            )

        monkeypatch.setattr("platform_ml.ArtifactStore", _make_fake_store_no_manifest)

        payload: ScoreJobPayload = {
            "run_id": "run_no_manifest",
            "request_id": "req_no_manifest",
            "text": "hello",
            "path": None,
            "detail_level": "summary",
            "top_k": None,
            "seed": None,
        }

        with pytest.raises(AppError, match="manifest missing"):
            score_job.process_score_job(payload)

    def test_generate_job_with_download(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test generate job when artifact needs to be downloaded (covers lines 958-964)."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run_gen_download"), "file_gen_download_123")

        monkeypatch.setattr(generate_job, "load_settings", lambda: settings)

        def _fake_redis_client(s: Settings) -> FakeRedis:
            return fake_redis

        monkeypatch.setattr(generate_job, "redis_client", _fake_redis_client)

        def _fake_load_tokenizer(s: Settings, tid: str) -> _FakeTokenizerHandle:
            return _FakeTokenizerHandle()

        monkeypatch.setattr(generate_job, "load_tokenizer_for_training", _fake_load_tokenizer)

        backend = _FakeBackendWithTopk()
        container_factory = _make_container_factory(backend)
        monkeypatch.setattr(generate_job, "ServiceContainer", container_factory)

        manifest_content = _make_manifest("char_lstm", "tok123")

        def _make_fake_store(
            base_url: str, api_key: str, *, timeout_seconds: float = 600.0
        ) -> _FakeArtifactStore:
            return _FakeArtifactStore(
                base_url,
                api_key,
                timeout_seconds=timeout_seconds,
                include_manifest=True,
                manifest_content=manifest_content,
            )

        monkeypatch.setattr("platform_ml.ArtifactStore", _make_fake_store)

        payload: GenerateJobPayload = {
            "run_id": "run_gen_download",
            "request_id": "req_gen_download",
            "prompt_text": "Hello",
            "prompt_path": None,
            "max_new_tokens": 10,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "stop_on_eos": True,
            "stop_sequences": [],
            "seed": None,
            "num_return_sequences": 1,
        }

        generate_job.process_generate_job(payload)

        cached = fake_redis.get(generate_key("run_gen_download", "req_gen_download"))
        assert isinstance(cached, str) and len(cached) > 0
        result = load_json_str(cached)
        assert isinstance(result, dict) and result.get("status") == "completed"

    def test_generate_job_missing_manifest_after_download(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: _SettingsFactory,
    ) -> None:
        """Test generate job when manifest is missing after download (covers line 969)."""
        artifacts_root = tmp_path / "artifacts"
        artifacts_root.mkdir(parents=True)
        models_dir = artifacts_root / "models"
        models_dir.mkdir(parents=True)

        settings = settings_factory(
            artifacts_root=str(artifacts_root),
            data_bank_api_url="http://test.api",
            data_bank_api_key="test-key",
        )

        fake_redis = FakeRedis()
        fake_redis.set(artifact_file_id_key("run_gen_no_manifest"), "file_gen_no_manifest_123")

        monkeypatch.setattr(generate_job, "load_settings", lambda: settings)

        def _fake_redis_client(s: Settings) -> FakeRedis:
            return fake_redis

        monkeypatch.setattr(generate_job, "redis_client", _fake_redis_client)

        def _make_fake_store_no_manifest(
            base_url: str, api_key: str, *, timeout_seconds: float = 600.0
        ) -> _FakeArtifactStore:
            return _FakeArtifactStore(
                base_url,
                api_key,
                timeout_seconds=timeout_seconds,
                include_manifest=False,
                manifest_content=None,
            )

        monkeypatch.setattr("platform_ml.ArtifactStore", _make_fake_store_no_manifest)

        payload: GenerateJobPayload = {
            "run_id": "run_gen_no_manifest",
            "request_id": "req_gen_no_manifest",
            "prompt_text": "Hello",
            "prompt_path": None,
            "max_new_tokens": 10,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "stop_on_eos": True,
            "stop_sequences": [],
            "seed": None,
            "num_return_sequences": 1,
        }

        with pytest.raises(AppError, match="manifest missing"):
            generate_job.process_generate_job(payload)
