"""Tests for pretrained model loading in train_job.py (lines 223-230)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

import torch
from platform_core.data_bank_protocol import FileUploadResponse
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
from model_trainer.core.contracts.tokenizer import TokenizerHandle, TokenizerTrainConfig
from model_trainer.core.encoding import Encoded
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.unavailable_backend import UNAVAILABLE_CAPABILITIES
from model_trainer.core.services.registries import BackendRegistration, ModelRegistry
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    NamedParameter,
    ParameterLike,
)
from model_trainer.worker import train_job
from model_trainer.worker.trainer_job_store import TrainerJobStore

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


class _FakeEncoded(Encoded):
    """Fake encoded result implementing Encoded protocol."""

    def __init__(self: _FakeEncoded, id_list: list[int]) -> None:
        self._ids = id_list

    @property
    def ids(self: _FakeEncoded) -> list[int]:
        return self._ids


class _FakeEncoder:
    """Fake encoder for PreparedLMModel.tok_for_dataset."""

    def encode(self: _FakeEncoder, text: str) -> Encoded:
        return _FakeEncoded([ord(c) for c in text])

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


# ============================================================================
# Test helpers and settings protocol
# ============================================================================


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


class _BackendWithLoad(ModelBackend):
    """Backend that supports loading pretrained models, implementing full ModelBackend protocol."""

    def __init__(self: _BackendWithLoad, train_losses: list[float]) -> None:
        self.load_called = False
        self.prepare_called = False
        self.loaded_from: str | None = None
        self._train_losses = train_losses

    def name(self: _BackendWithLoad) -> str:
        return "gpt2"

    def capabilities(self: _BackendWithLoad) -> BackendCapabilities:
        return UNAVAILABLE_CAPABILITIES

    def prepare(
        self: _BackendWithLoad,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        self.prepare_called = True
        raise NotImplementedError

    def load(
        self: _BackendWithLoad,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle,
    ) -> PreparedLMModel:
        self.load_called = True
        self.loaded_from = artifact_path
        # Return a fake PreparedLMModel since we're testing the load path
        return _make_fake_prepared("loaded-tok")

    def train(
        self: _BackendWithLoad,
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
        # Simulate training progress with decreasing loss
        # Args: step, epoch, loss, train_ppl, grad_norm, samples_per_sec, val_loss, val_ppl
        losses = [2.5, 1.8, 1.2, 0.9, 0.5]
        for step, loss_val in enumerate(losses):
            self._train_losses.append(loss_val)
            if progress:
                progress(step, 0, loss_val, 12.2 / (step + 1), 0.5 / (step + 1), 10.0, None, None)
        return TrainOutcome(
            cancelled=False,
            loss=0.5,
            perplexity=1.2,
            steps=5,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def save(self: _BackendWithLoad, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "weights.bin").write_bytes(b"\x00mock")
        return ModelArtifact(out_dir=out_dir)

    def evaluate(
        self: _BackendWithLoad, *, run_id: str, cfg: ModelTrainConfig, settings: Settings
    ) -> EvalOutcome:
        raise NotImplementedError

    def score(
        self: _BackendWithLoad, *, prepared: PreparedLMModel, cfg: ScoreConfig, settings: Settings
    ) -> ScoreOutcome:
        raise NotImplementedError

    def generate(
        self: _BackendWithLoad,
        *,
        prepared: PreparedLMModel,
        cfg: GenerateConfig,
        settings: Settings,
    ) -> GenerateOutcome:
        raise NotImplementedError


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
            file_id="finetuned-file-id",
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


class _BackendRegistry:
    """Simple registry wrapper for backend instance tracking."""

    def __init__(
        self, backend_instance_holder: list[_BackendWithLoad | None], train_losses: list[float]
    ) -> None:
        self._holder = backend_instance_holder
        self._train_losses = train_losses

    def get(self, name: str) -> _BackendWithLoad:
        if self._holder[0] is None:
            self._holder[0] = _BackendWithLoad(self._train_losses)
        backend = self._holder[0]
        if backend is None:
            raise AssertionError("backend should not be None after assignment")
        return backend


class _FakeServiceContainer:
    """Fake ServiceContainer for pretrained model tests."""

    def __init__(self, s: Settings, r: RedisStrProto, reg: ModelRegistry) -> None:
        self._settings = s
        self._redis = r
        self._model_registry = reg

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def redis(self) -> RedisStrProto:
        return self._redis

    @property
    def model_registry(self) -> ModelRegistry:
        return self._model_registry


def _create_service_container_factory(
    fake_redis: FakeRedis,
    backend_instance_holder: list[_BackendWithLoad | None],
    train_losses: list[float],
) -> _test_hooks.ServiceContainerFactoryProto:
    """Create a service container factory for testing."""

    def _from_settings(settings: Settings) -> ServiceContainerProto:
        backend_reg = _BackendRegistry(backend_instance_holder, train_losses)

        def _backend_factory(dataset_builder: DatasetBuilder) -> ModelBackend:
            return backend_reg.get("gpt2")

        model_registry = ModelRegistry(
            registrations={
                "gpt2": BackendRegistration(
                    factory=_backend_factory, capabilities=UNAVAILABLE_CAPABILITIES
                )
            },
            dataset_builder=LocalTextDatasetBuilder(),
        )
        return _FakeServiceContainer(settings, fake_redis, model_registry)

    return _from_settings


def _create_corpus_fetcher_factory(
    corpus_path: Path,
) -> _test_hooks.CorpusFetcherFactoryProto:
    """Create a corpus fetcher factory for testing."""

    def _factory(api_url: str, api_key: str, cache_dir: Path) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(corpus_path)

    return _factory


def _create_artifact_store_factory() -> _test_hooks.ArtifactStoreFactoryProto:
    """Create an artifact store factory for testing."""

    def _factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return _FakeStore(base_url, api_key, timeout_seconds=timeout_seconds)

    return _factory


def test_training_worker_loads_pretrained_model(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Cover train_job.py lines 223-230 - pretrained model loading branch."""
    # Track backend instance and losses for assertions
    backend_instance_holder: list[_BackendWithLoad | None] = [None]
    train_losses: list[float] = []

    # Environment roots
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )

    # Settings via hook
    _test_hooks.load_settings = lambda: settings

    # Minimal corpus for tokenizer training
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nfinetuning data\n", encoding="utf-8")

    # Train a real tokenizer using BPEBackend
    tok_id = "tok-pretrained-test"
    tok_dir = artifacts / "tokenizers" / tok_id
    tok_cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    BPEBackend().train(tok_cfg)

    # Create a pretrained model directory (simulating a previous training run)
    pretrained_run_id = "run-pretrained-base"
    pretrained_model_dir = artifacts / "models" / pretrained_run_id
    pretrained_model_dir.mkdir(parents=True, exist_ok=True)
    (pretrained_model_dir / "weights.bin").write_bytes(b"\x00pretrained")
    (pretrained_model_dir / "manifest.json").write_text(
        '{"model_family": "gpt2", "model_size": "small"}', encoding="utf-8"
    )

    # Fake redis via hook
    fake = FakeRedis()
    _test_hooks.kv_store_factory = lambda url: fake

    # Set up hooks using extracted factory functions
    _test_hooks.service_container_from_settings = _create_service_container_factory(
        fake, backend_instance_holder, train_losses
    )
    _test_hooks.corpus_fetcher_factory = _create_corpus_fetcher_factory(corpus)
    _test_hooks.artifact_store_factory = _create_artifact_store_factory()

    # Build payload with pretrained_run_id set
    payload: TrainJobPayload = {
        "run_id": "run-finetune",
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
            "pretrained_run_id": pretrained_run_id,
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

    # Verify backend.load() was called instead of backend.prepare()
    backend_instance = backend_instance_holder[0]
    assert backend_instance is not None and backend_instance.load_called is True
    assert backend_instance.prepare_called is False
    assert backend_instance.loaded_from == str(pretrained_model_dir)

    # Verify loss decreases during training (ml-train-no-loss-check)
    assert len(train_losses) >= 2, "Should have at least 2 loss values"
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before, f"Loss should decrease: {loss_before} -> {loss_after}"

    # Verify status is completed
    status = TrainerJobStore(fake).load("run-finetune")
    assert status is not None and status["status"] == "completed"
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish"})
