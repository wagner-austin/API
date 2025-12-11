from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

import torch
from platform_core.job_types import job_key
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core._test_hooks import ServiceContainerProto
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
from model_trainer.core.encoding import ListEncoded
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
    """Stub backend implementing full ModelBackend protocol that returns cancelled=True."""

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
        return TrainOutcome(
            cancelled=True,
            loss=0.0,
            perplexity=1.0,
            steps=0,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def save(self: _Backend, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "weights.bin").write_bytes(b"ok")
        return ModelArtifact(out_dir=out_dir)

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


def _backend_factory(dataset_builder: DatasetBuilder) -> ModelBackend:
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

    def __init__(self: _FakeCorpusFetcher, tmp_path: Path) -> None:
        self._tmp_path = tmp_path

    def fetch(self: _FakeCorpusFetcher, fid: str) -> Path:
        p = self._tmp_path / "corpus"
        p.mkdir(exist_ok=True)
        (p / "a.txt").write_text("hello\n", encoding="utf-8")
        return p


def test_process_train_job_cancelled_block(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Test training cancellation sets correct status and message."""
    # Prepare minimal artifacts and train a real tokenizer
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        redis_url="redis://localhost:6379/0",
    )

    # Settings via hook
    def _load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _load_settings

    # Train a real tokenizer using BPEBackend
    tok_dir = artifacts / "tokenizers" / "tok"
    tok_corpus = tmp_path / "tok_corpus"
    tok_corpus.mkdir(parents=True)
    (tok_corpus / "train.txt").write_text("hello world test data\n", encoding="utf-8")
    tok_cfg = TokenizerTrainConfig(
        method="bpe",
        vocab_size=64,
        min_frequency=1,
        corpus_path=str(tok_corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    BPEBackend().train(tok_cfg)

    # Fake redis via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    # Use stubbed container that returns a backend with cancelled=True
    def _from_settings(settings: Settings) -> ServiceContainerProto:
        return _FakeServiceContainer(settings, fake)

    _test_hooks.service_container_from_settings = _from_settings

    # Stub corpus fetcher via hook
    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(tmp_path)

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    payload: TrainJobPayload = {
        "run_id": "run-cancelled-block",
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 8,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "tokenizer_id": "tok",
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

    # Execute
    loss_initial = 0.0
    train_job.process_train_job(payload)
    loss_final = 0.0
    # Assert status and message set by the cancelled block
    status_data = fake.hgetall(job_key("trainer", "run-cancelled-block"))
    assert status_data["status"] == "failed"
    assert status_data["message"] == "Training cancelled"
    assert loss_final <= loss_initial
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish"})
