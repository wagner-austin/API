"""Inference worker: request handling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

import torch
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import dump_json_str
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.redis import RedisStrProto

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
from model_trainer.core.encoding import ListEncoded
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.unavailable_backend import UNAVAILABLE_CAPABILITIES
from model_trainer.core.services.registries import BackendRegistration, ModelRegistry
from model_trainer.core.types import (
    ForwardOutProto,
    LoadStateDictResultProto,
    NamedParameter,
    ParameterLike,
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
        "loss_mask_prefix_separator": None,
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
        "timing": {
            "training_duration_sec": 10.5,
            "started_at": "2024-01-15T10:00:00",
            "completed_at": "2024-01-15T10:00:10",
        },
        "performance": {
            "peak_gpu_memory_mb": None,
            "avg_samples_per_sec": 100.0,
            "total_tokens_processed": 1024,
        },
        "model_info": {
            "param_count": 1000,
            "model_size_mb": 5.0,
            "vocab_size": 256,
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

    def gradient_checkpointing_enable(self) -> None:
        return None

    def state_dict(self: _FakeLMModel) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(
        self: _FakeLMModel, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        _ = state_dict
        return self


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
        tokenizer: TokenizerHandle | None,
    ) -> PreparedLMModel:
        return _make_fake_prepared()

    def save(self, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        return ModelArtifact(out_dir=out_dir)

    def load(
        self,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle | None,
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
        resume: bool,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
        wandb_publisher: WandbPublisher | None = None,
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
        tokenizer: TokenizerHandle | None,
    ) -> PreparedLMModel:
        return _make_fake_prepared()

    def save(self, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        return ModelArtifact(out_dir=out_dir)

    def load(
        self,
        artifact_path: str,
        settings: Settings,
        *,
        tokenizer: TokenizerHandle | None,
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
        resume: bool,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
        wandb_publisher: WandbPublisher | None = None,
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


class _FakeServiceContainer:
    """Fake ServiceContainer for testing."""

    def __init__(
        self: _FakeServiceContainer,
        settings: Settings,
        redis: RedisStrProto,
        backend: ModelBackend,
    ) -> None:
        self._settings = settings
        self._redis = redis

        def _backend_factory(dataset_builder: DatasetBuilder) -> ModelBackend:
            return backend

        self._model_registry = ModelRegistry(
            registrations={
                "gpt2": BackendRegistration(
                    factory=_backend_factory, capabilities=UNAVAILABLE_CAPABILITIES
                ),
                "char_lstm": BackendRegistration(
                    factory=_backend_factory, capabilities=UNAVAILABLE_CAPABILITIES
                ),
                "llama": BackendRegistration(
                    factory=_backend_factory, capabilities=UNAVAILABLE_CAPABILITIES
                ),
                "qwen": BackendRegistration(
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


class _FakeArtifactStore:
    """Fake ArtifactStore for testing artifact download paths."""

    def __init__(
        self: _FakeArtifactStore,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 600.0,
        include_manifest: bool = True,
        manifest_content: str | None = None,
    ) -> None:
        self._include_manifest = include_manifest
        self._manifest_content = manifest_content

    def upload_artifact(
        self: _FakeArtifactStore,
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
        self: _FakeArtifactStore,
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
