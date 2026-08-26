"""Shared fakes for the pretrained training worker tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

import torch
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.determinism_record import DeterminismRecord
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from model_trainer.core._hook_protocols import (
    ArtifactStoreFactoryProto,
    ArtifactStoreProto,
    ServiceContainerFactoryProto,
    ServiceContainerProto,
)
from model_trainer.core._hook_protocols_ml import CorpusFetcherFactoryProto
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
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoded
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.unavailable_backend import UNAVAILABLE_CAPABILITIES
from model_trainer.core.services.registries import BackendRegistration, ModelRegistry
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    LoadStateDictResultProto,
    NamedParameter,
    ParameterLike,
)


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

    def gradient_checkpointing_enable(self: _FakeLMModel) -> None:
        return None

    @property
    def config(self: _FakeLMModel) -> ConfigLike:
        return self._config

    def state_dict(self: _FakeLMModel) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(
        self: _FakeLMModel, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        _ = state_dict
        return self


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


def _make_fake_prepared(tokenizer_id: str | None) -> PreparedLMModel:
    """Create a fake PreparedLMModel for testing."""
    return PreparedLMModel(
        model=_FakeLMModel(),
        tokenizer_id=tokenizer_id,
        eos_id=0,
        pad_id=1,
        max_seq_len=64,
        tok_for_dataset=_FakeEncoder(),
    )


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
        self.resume_seen: bool | None = None
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
        tokenizer: TokenizerHandle | None,
    ) -> PreparedLMModel:
        self.prepare_called = True
        raise NotImplementedError

    def load(
        self: _BackendWithLoad,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle | None,
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
        resume: bool,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
        wandb_publisher: WandbPublisher | None = None,
        determinism: DeterminismRecord | None = None,
    ) -> TrainOutcome:
        self.resume_seen = resume
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
) -> ServiceContainerFactoryProto:
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
) -> CorpusFetcherFactoryProto:
    """Create a corpus fetcher factory for testing."""

    def _factory(api_url: str, api_key: str, cache_dir: Path) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(corpus_path)

    return _factory


def _create_artifact_store_factory() -> ArtifactStoreFactoryProto:
    """Create an artifact store factory for testing."""

    def _factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return _FakeStore(base_url, api_key, timeout_seconds=timeout_seconds)

    return _factory


class _HfLmBackend(ModelBackend):
    """Fake hf_lm backend for testing tokenizer_id=None path."""

    def __init__(self: _HfLmBackend, train_losses: list[float]) -> None:
        self._train_losses = train_losses
        self.prepare_called = False
        self.prepare_tokenizer_was_none = False

    def name(self: _HfLmBackend) -> str:
        return "hf_lm"

    def capabilities(self: _HfLmBackend) -> BackendCapabilities:
        return UNAVAILABLE_CAPABILITIES

    def prepare(
        self: _HfLmBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle | None,
    ) -> PreparedLMModel:
        self.prepare_called = True
        self.prepare_tokenizer_was_none = tokenizer is None
        return _make_fake_prepared(None)  # tokenizer_id is None for hf_lm

    def load(
        self: _HfLmBackend,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle | None,
    ) -> PreparedLMModel:
        raise NotImplementedError

    def save(self: _HfLmBackend, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "weights.bin").write_bytes(b"\x00hflm")
        return ModelArtifact(out_dir=out_dir)

    def train(
        self: _HfLmBackend,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        heartbeat: Callable[[float], None],
        cancelled: Callable[[], bool],
        prepared: PreparedLMModel,
        resume: bool,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
        wandb_publisher: WandbPublisher | None = None,
        determinism: DeterminismRecord | None = None,
    ) -> TrainOutcome:
        losses = [2.0, 1.5, 1.0, 0.5]
        for step, loss_val in enumerate(losses):
            self._train_losses.append(loss_val)
            if progress:
                progress(step, 0, loss_val, 7.4, 0.3, 10.0, None, None)
        return TrainOutcome(
            cancelled=False,
            loss=0.5,
            perplexity=1.2,
            steps=4,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def evaluate(
        self: _HfLmBackend, *, run_id: str, cfg: ModelTrainConfig, settings: Settings
    ) -> EvalOutcome:
        raise NotImplementedError

    def score(
        self: _HfLmBackend, *, prepared: PreparedLMModel, cfg: ScoreConfig, settings: Settings
    ) -> ScoreOutcome:
        raise NotImplementedError

    def generate(
        self: _HfLmBackend,
        *,
        prepared: PreparedLMModel,
        cfg: GenerateConfig,
        settings: Settings,
    ) -> GenerateOutcome:
        raise NotImplementedError


def _create_hf_lm_service_container_factory(
    fake_redis: FakeRedis,
    backend_instance_holder: list[_HfLmBackend | None],
    train_losses: list[float],
) -> ServiceContainerFactoryProto:
    """Create a service container factory for hf_lm testing."""

    def _from_settings(settings: Settings) -> ServiceContainerProto:
        backend = _HfLmBackend(train_losses)
        backend_instance_holder.append(backend)

        model_registry = ModelRegistry(
            registrations={
                "hf_lm": BackendRegistration(
                    factory=lambda _: backend, capabilities=UNAVAILABLE_CAPABILITIES
                )
            },
            dataset_builder=LocalTextDatasetBuilder(),
        )
        return _FakeServiceContainer(settings, fake_redis, model_registry)

    return _from_settings


class _RecordingStore:
    """Artifact store that actually materializes the downloaded directory.

    ``_FakeStore`` returns a path without creating it, which is enough for a
    test that pre-creates the directory itself. This one writes the files a
    real download would produce, so the caller's rename-into-place runs for
    real, and records the request so the test can assert it happened.

    Attributes:
        downloads: One (file_id, expected_root) per download_artifact call.
    """

    def __init__(
        self: _RecordingStore,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.downloads: list[tuple[str, str]] = []

    def upload_artifact(
        self: _RecordingStore, dir_path: Path, *, artifact_name: str, request_id: str
    ) -> FileUploadResponse:
        """Return a fixed upload response.

        Args:
            dir_path: Directory being uploaded.
            artifact_name: Name for the uploaded artifact.
            request_id: Correlation id.

        Returns:
            A response carrying a fixed file id.
        """
        return FileUploadResponse(
            file_id="finetuned-file-id",
            size=1,
            sha256="x",
            content_type="application/gzip",
            created_at=None,
        )

    def download_artifact(
        self: _RecordingStore,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        """Create the directory a real download would produce.

        Args:
            file_id: Artifact to fetch.
            dest_dir: Directory to unpack into.
            request_id: Correlation id.
            expected_root: Name of the unpacked root directory.

        Returns:
            Path to the unpacked root.
        """
        self.downloads.append((file_id, expected_root))
        out = dest_dir / expected_root
        out.mkdir(parents=True, exist_ok=True)
        (out / "weights.bin").write_bytes(b"\x00pretrained")
        (out / "manifest.json").write_text(
            '{"model_family": "gpt2", "model_size": "small"}', encoding="utf-8"
        )
        return out
