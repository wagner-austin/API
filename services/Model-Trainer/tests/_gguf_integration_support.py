"""Shared fakes for the GGUF export integration tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.determinism_record import DeterminismRecord
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.redis import RedisStrProto

from model_trainer.core._hook_protocols import (
    ArtifactStoreFactoryProto,
    ArtifactStoreProto,
    ServiceContainerFactoryProto,
    ServiceContainerProto,
)
from model_trainer.core._hook_protocols_ml import CorpusFetcherFactoryProto, CorpusFetcherProto
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
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoded
from model_trainer.core.services.model.unavailable_backend import UNAVAILABLE_CAPABILITIES
from model_trainer.core.services.registries import ModelRegistry
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
    def loss(self) -> torch.Tensor:
        return torch.tensor(0.1)


class _FakeLMModel(LMModelProto):
    """Fake LM model for testing."""

    def __init__(self) -> None:
        self._config = _FakeConfig()

    @classmethod
    def from_pretrained(cls, path: str) -> LMModelProto:
        return cls()

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        return _FakeFwd()

    def parameters(self) -> Sequence[ParameterLike]:
        return []

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self, device: str) -> LMModelProto:
        return self

    def save_pretrained(self, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "adapter_model.safetensors").write_bytes(b"\x00fake")

    def gradient_checkpointing_enable(self) -> None:
        return None

    @property
    def config(self) -> ConfigLike:
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

    def __init__(self, id_list: list[int]) -> None:
        self._ids = id_list

    @property
    def ids(self) -> list[int]:
        return self._ids


class _FakeEncoder:
    """Fake encoder for PreparedLMModel.tok_for_dataset."""

    def encode(self, text: str) -> Encoded:
        return _FakeEncoded([ord(c) for c in text])

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if i < 128)

    def token_to_id(self, token: str) -> int | None:
        if len(token) == 1:
            return ord(token)
        return None

    def get_vocab_size(self) -> int:
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


class _HfLmBackend(ModelBackend):
    """Fake hf_lm backend for GGUF export testing."""

    def name(self) -> str:
        return "hf_lm"

    def capabilities(self) -> BackendCapabilities:
        return UNAVAILABLE_CAPABILITIES

    def prepare(
        self,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle | None,
    ) -> PreparedLMModel:
        return _make_fake_prepared(None)

    def load(
        self,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle | None,
    ) -> PreparedLMModel:
        raise NotImplementedError

    def save(self, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "adapter_model.safetensors").write_bytes(b"\x00hflm")
        return ModelArtifact(out_dir=out_dir)

    def train(
        self,
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
        for step in range(3):
            if progress:
                progress(step, 0, 0.5 - step * 0.1, 1.5, 0.3, 10.0, None, None)
        return TrainOutcome(
            cancelled=False,
            loss=0.3,
            perplexity=1.2,
            steps=3,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def evaluate(self, *, run_id: str, cfg: ModelTrainConfig, settings: Settings) -> EvalOutcome:
        raise NotImplementedError

    def score(
        self, *, prepared: PreparedLMModel, cfg: ScoreConfig, settings: Settings
    ) -> ScoreOutcome:
        raise NotImplementedError

    def generate(
        self,
        *,
        prepared: PreparedLMModel,
        cfg: GenerateConfig,
        settings: Settings,
    ) -> GenerateOutcome:
        raise NotImplementedError


class _FakeArtifactStore:
    """Fake artifact store for testing."""

    def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
        pass

    def upload_artifact(
        self,
        dir_path: Path,
        *,
        artifact_name: str,
        request_id: str,
    ) -> FileUploadResponse:
        return FileUploadResponse(
            file_id="uploaded-file-id",
            size=1,
            sha256="x",
            content_type="application/gzip",
            created_at=None,
        )

    def download_artifact(
        self,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        return dest_dir / expected_root


class _FakeCorpusFetcher:
    """Fake corpus fetcher for testing."""

    def __init__(self, corpus_path: Path) -> None:
        self._corpus_path = corpus_path

    def fetch(self, fid: str) -> Path:
        return self._corpus_path


class _FakeServiceContainer:
    """Fake service container for testing."""

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
    fake_redis: RedisStrProto,
    backend: _HfLmBackend,
) -> ServiceContainerFactoryProto:
    """Create a service container factory for GGUF export testing."""
    from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
    from model_trainer.core.services.registries import BackendRegistration

    def _from_settings(settings: Settings) -> ServiceContainerProto:
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


def _create_corpus_fetcher_factory(
    corpus_path: Path,
) -> CorpusFetcherFactoryProto:
    """Create a corpus fetcher factory for testing."""

    def _factory(api_url: str, api_key: str, cache_dir: Path) -> CorpusFetcherProto:
        return _FakeCorpusFetcher(corpus_path)

    return _factory


def _create_artifact_store_factory() -> ArtifactStoreFactoryProto:
    """Create an artifact store factory for testing."""

    def _factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return _FakeArtifactStore(base_url, api_key, timeout_seconds=timeout_seconds)

    return _factory
