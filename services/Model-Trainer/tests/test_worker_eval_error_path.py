from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import dump_json_str
from platform_core.trainer_keys import EVAL_KEY_PREFIX, artifact_file_id_key
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core._test_hooks import ArtifactStoreProto, ServiceContainerProto
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import DatasetBuilder
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
from model_trainer.core.contracts.queue import EvalJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.unavailable_backend import UNAVAILABLE_CAPABILITIES
from model_trainer.core.services.registries import BackendRegistration, ModelRegistry
from model_trainer.worker.eval_job import process_eval_job


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


class _Backend(ModelBackend):
    """Backend that raises RuntimeError on evaluate."""

    def name(self: _Backend) -> str:
        return "gpt2"

    def capabilities(self: _Backend) -> BackendCapabilities:
        return UNAVAILABLE_CAPABILITIES

    def prepare(
        self: _Backend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        raise NotImplementedError

    def load(
        self: _Backend,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        raise NotImplementedError

    def train(
        self: _Backend,
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
        wandb_publisher: WandbPublisher | None = None,
    ) -> TrainOutcome:
        raise NotImplementedError

    def save(self: _Backend, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        raise NotImplementedError

    def evaluate(
        self: _Backend,
        *,
        run_id: str,
        cfg: ModelTrainConfig,
        settings: Settings,
    ) -> EvalOutcome:
        raise RuntimeError("boom")

    def score(
        self: _Backend, *, prepared: PreparedLMModel, cfg: ScoreConfig, settings: Settings
    ) -> ScoreOutcome:
        raise NotImplementedError

    def generate(
        self: _Backend, *, prepared: PreparedLMModel, cfg: GenerateConfig, settings: Settings
    ) -> GenerateOutcome:
        raise NotImplementedError


def _backend_factory(dataset_builder: DatasetBuilder) -> ModelBackend:
    """Factory that returns _Backend instance."""
    return _Backend()


class _Container:
    """Fake ServiceContainer that provides a backend that raises on evaluate."""

    def __init__(
        self: _Container,
        settings: Settings,
        redis: RedisStrProto,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._model_registry = ModelRegistry(
            registrations={
                "gpt2": BackendRegistration(
                    factory=_backend_factory, capabilities=UNAVAILABLE_CAPABILITIES
                ),
            },
            dataset_builder=LocalTextDatasetBuilder(),
        )

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def redis(self) -> RedisStrProto:
        return self._redis

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry


class _FakeStore:
    def __init__(
        self: _FakeStore, base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> None:
        pass

    def upload_artifact(
        self: _FakeStore,
        dir_path: Path,
        *,
        artifact_name: str,
        request_id: str,
    ) -> FileUploadResponse:
        return FileUploadResponse(
            file_id="fake-upload-id",
            size=1,
            sha256="x",
            content_type="application/gzip",
            created_at=None,
        )

    def download_artifact(
        self: _FakeStore,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        # Simulate download: create at temp location (dest_dir / expected_root)
        out = dest_dir / expected_root
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": "run-err",
            "model_family": "gpt2",
            "model_size": "s",
            "epochs": 1,
            "batch_size": 1,
            "max_seq_len": 8,
            "steps": 0,
            "loss": 0.0,
            "learning_rate": 1e-3,
            "tokenizer_id": "tok",
            "corpus_path": str(dest_dir),
            "optimizer": "adamw",
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "device": "cpu",
            "precision": "fp32",
            "early_stopping_patience": 5,
            "test_split_ratio": 0.15,
            "finetune_lr_cap": 5e-5,
            "early_stopped": False,
            "holdout_fraction": 0.1,
            "pretrained_run_id": None,
            "seed": 42,
            "versions": {"torch": "0", "transformers": "0", "tokenizers": "0", "datasets": "0"},
            "system": {"cpu_count": 1, "platform": "X", "platform_release": "Y", "machine": "Z"},
            "git_commit": "g",
        }
        (out / "manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")
        return out


def test_worker_eval_backend_raises(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    """Test that eval job propagates backend exceptions."""
    # Fake redis via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    # Settings via hook
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )

    def _load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _load_settings

    # Set artifact pointer to bypass download flow
    fake.set(artifact_file_id_key("run-err"), "fid-test")

    # Mock ArtifactStore via hook
    def _fake_artifact_store_factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return _FakeStore(base_url, api_key, timeout_seconds=timeout_seconds)

    _test_hooks.artifact_store_factory = _fake_artifact_store_factory

    # Patch container factory via hook to return backend that raises
    def _from_settings(settings: Settings) -> ServiceContainerProto:
        return _Container(settings, fake)

    _test_hooks.service_container_from_settings = _from_settings

    # Now run eval and assert failure is recorded and exception propagated
    payload: EvalJobPayload = {"run_id": "run-err", "split": "validation", "path_override": None}
    with pytest.raises(RuntimeError):
        process_eval_job(payload)
    raw = fake.get(f"{EVAL_KEY_PREFIX}run-err")
    assert isinstance(raw, str) and len(raw) > 0
    fake.assert_only_called({"set", "get"})
