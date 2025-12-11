from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

import torch
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.trainer_keys import artifact_file_id_key
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
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import ListEncoded
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.unavailable_backend import UNAVAILABLE_CAPABILITIES
from model_trainer.core.services.registries import BackendRegistration, ModelRegistry
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    NamedParameter,
    ParameterLike,
)
from model_trainer.worker import train_job
from model_trainer.worker.trainer_job_store import TrainerJobStore


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


# ============================================================================
# Fake model components for PreparedLMModel
# ============================================================================


class _FakeConfig(ConfigLike):
    """Fake config for LMModelProto."""

    n_positions: int = 64


class _FakeFwd(ForwardOutProto):
    """Fake forward output."""

    @property
    def loss(self: _FakeFwd) -> torch.Tensor:
        return torch.tensor(0.1)


class _FakeLMModel(LMModelProto):
    """Fake LM model for testing."""

    def __init__(self: _FakeLMModel) -> None:
        self._config = _FakeConfig()

    @classmethod
    def from_pretrained(cls: type[_FakeLMModel], path: str) -> LMModelProto:
        return cls()

    def train(self: _FakeLMModel) -> None:
        pass

    def eval(self: _FakeLMModel) -> None:
        pass

    def forward(
        self: _FakeLMModel, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        return _FakeFwd()

    def parameters(self: _FakeLMModel) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self: _FakeLMModel) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: _FakeLMModel, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: _FakeLMModel, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "weights.bin").write_bytes(b"\x00fake")

    @property
    def config(self: _FakeLMModel) -> ConfigLike:
        return self._config


class _FakeEncoder:
    """Fake encoder for PreparedLMModel.tok_for_dataset."""

    def encode(self: _FakeEncoder, text: str) -> ListEncoded:
        return ListEncoded([ord(c) for c in text])

    def decode(self: _FakeEncoder, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if i < 128)

    def token_to_id(self: _FakeEncoder, token: str) -> int | None:
        if len(token) == 1:
            return ord(token)
        return None

    def get_vocab_size(self: _FakeEncoder) -> int:
        return 256


def _make_fake_prepared(tokenizer_id: str) -> PreparedLMModel:
    """Create a fake PreparedLMModel for testing."""
    return PreparedLMModel(
        model=_FakeLMModel(),
        tokenizer_id=tokenizer_id,
        eos_id=0,
        pad_id=1,
        max_seq_len=64,
        tok_for_dataset=_FakeEncoder(),
    )


class _Backend(ModelBackend):
    """Stub backend implementing full ModelBackend protocol."""

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
        return _make_fake_prepared("fake-tok")

    def save(self: _Backend, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "weights.bin").write_bytes(b"\x00mock")
        return ModelArtifact(out_dir=out_dir)

    def load(
        self: _Backend,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        return _make_fake_prepared("loaded-tok")

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
        # Exercise the worker's progress callback wrapper so that the
        # training_worker._progress closure is covered.
        if progress is not None:
            progress(1, 0, 0.5, 1.65, 0.1, 100.0, None, None)
        return TrainOutcome(
            cancelled=False,
            loss=0.9,
            perplexity=1.5,
            steps=10,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def evaluate(
        self: _Backend, *, run_id: str, cfg: ModelTrainConfig, settings: Settings
    ) -> EvalOutcome:
        raise NotImplementedError

    def score(
        self: _Backend, *, prepared: PreparedLMModel, cfg: ScoreConfig, settings: Settings
    ) -> ScoreOutcome:
        raise NotImplementedError

    def generate(
        self: _Backend, *, prepared: PreparedLMModel, cfg: GenerateConfig, settings: Settings
    ) -> GenerateOutcome:
        raise NotImplementedError


def _backend_factory(dataset_builder: DatasetBuilder) -> _Backend:
    """Factory that returns _Backend instance."""
    return _Backend()


class _FakeServiceContainer:
    def __init__(self: _FakeServiceContainer, settings: Settings, redis: RedisStrProto) -> None:
        self._settings = settings
        self._redis = redis
        self._model_registry = ModelRegistry(
            registrations={
                "gpt2": BackendRegistration(
                    factory=_backend_factory, capabilities=UNAVAILABLE_CAPABILITIES
                )
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


class _FakeCorpusFetcher:
    """Fake CorpusFetcher for tests."""

    def __init__(self: _FakeCorpusFetcher, corpus_path: Path) -> None:
        self._corpus_path = corpus_path

    def fetch(self: _FakeCorpusFetcher, fid: str) -> Path:
        return self._corpus_path


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
            file_id="deadbeef",
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
        return dest_dir / expected_root


def test_training_worker_spm_artifact_and_completed(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Test training worker with SPM tokenizer completes and uploads artifact."""
    # Environment roots
    artifacts = tmp_path / "artifacts"
    runs = tmp_path / "runs"
    logs = tmp_path / "logs"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(runs),
        logs_root=str(logs),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )

    # Settings via hook
    def _load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _load_settings

    # Provide sentencepiece tokenizer artifact (spm) so worker loads tok_spm path
    tok_id = "tok-spm"
    tok_dir = artifacts / "tokenizers" / tok_id
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "tokenizer.model").write_bytes(b"\x00\x01mock")
    (tok_dir / "tokenizer.vocab").write_text("[UNK]\t0\nA\t1\n", encoding="utf-8")

    # Minimal corpus
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is a test\n", encoding="utf-8")

    # Fake redis client via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    # Stub ServiceContainer via hook
    def _from_settings(settings: Settings) -> ServiceContainerProto:
        return _FakeServiceContainer(settings, fake)

    _test_hooks.service_container_from_settings = _from_settings

    # Stub corpus fetcher via hook
    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(corpus)

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    # Stub artifact store via hook
    def _fake_artifact_store_factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return _FakeStore(base_url, api_key, timeout_seconds=timeout_seconds)

    _test_hooks.artifact_store_factory = _fake_artifact_store_factory

    payload: TrainJobPayload = {
        "run_id": "run-complete",
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "tokenizer_id": tok_id,
            "corpus_file_id": "deadbeef",
            "holdout_fraction": 0.01,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "data_num_workers": None,
            "data_pin_memory": None,
            "early_stopping_patience": 5,
            "test_split_ratio": 0.15,
            "finetune_lr_cap": 5e-5,
            "precision": "auto",
        },
    }

    train_job.process_train_job(payload)

    status = TrainerJobStore(fake).load("run-complete")
    assert status is not None and status["status"] == "completed"
    # Pointer persisted for inference service
    artifact_id = fake.get(artifact_file_id_key("run-complete"))
    out_dir = artifacts / "models" / "run-complete"

    assert status["message"] == "Training completed"
    assert artifact_id == "deadbeef"
    # Cleanup enabled by default: local artifact directory should be removed
    assert not out_dir.exists()
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish"})
