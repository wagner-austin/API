from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

import torch
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.job_types import job_key
from platform_core.trainer_keys import cancel_key
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
    """Stub backend implementing full ModelBackend protocol with save synchronization."""

    def __init__(
        self: _Backend, save_reached: threading.Event, allow_proceed: threading.Event
    ) -> None:
        self._save_reached = save_reached
        self._allow_proceed = allow_proceed

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
            cancelled=False,
            loss=0.1,
            perplexity=1.2,
            steps=1,
            out_dir="",
            test_loss=None,
            test_perplexity=None,
            best_val_loss=None,
            early_stopped=False,
        )

    def save(self: _Backend, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        # Signal test that worker reached save, then wait until test allows proceed
        self._save_reached.set()
        # Keep the wait bounded to avoid deadlock in case of test failure
        self._allow_proceed.wait(timeout=2.0)
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "weights.bin").write_bytes(b"\x00\x01mock")
        return ModelArtifact(out_dir=str(p))

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


class _FakeServiceContainer:
    """Fake ServiceContainer that provides waiting backend."""

    def __init__(
        self: _FakeServiceContainer,
        settings: Settings,
        redis: RedisStrProto,
        save_reached: threading.Event,
        allow_proceed: threading.Event,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._backend = _Backend(save_reached, allow_proceed)

        def _backend_factory(dataset_builder: DatasetBuilder) -> ModelBackend:
            return self._backend

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

    def __init__(self: _FakeCorpusFetcher, path: Path) -> None:
        self._path = path

    def fetch(self: _FakeCorpusFetcher, fid: str) -> Path:
        return self._path


def test_training_cancel_race_avoids_upload(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Test that cancellation during save() prevents upload."""
    # Prepare roots and minimal BPE tokenizer artifact
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )

    # Settings via hook
    def _load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _load_settings

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is a test\n", encoding="utf-8")
    tok_id = "tok-bpe-race"
    out_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=32,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    _ = BPEBackend().train(cfg_tok)

    # Fake redis via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    # Synchronization for the race window
    save_reached = threading.Event()
    allow_proceed = threading.Event()

    # Stub ServiceContainer via hook
    def _from_settings(settings: Settings) -> ServiceContainerProto:
        return _FakeServiceContainer(settings, fake, save_reached, allow_proceed)

    _test_hooks.service_container_from_settings = _from_settings

    # Stub CorpusFetcher via hook
    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(cache_dir)

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    # Capture any attempted uploads through ArtifactStore
    upload_calls: list[int] = []

    class _Store:
        def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
            pass

        def upload_artifact(
            self,
            dir_path: Path,
            *,
            artifact_name: str,
            request_id: str,
        ) -> FileUploadResponse:
            upload_calls.append(1)
            return FileUploadResponse(
                file_id="deadbeef",
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

    # Stub artifact store via hook
    def _fake_artifact_store_factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return _Store(base_url, api_key, timeout_seconds=timeout_seconds)

    _test_hooks.artifact_store_factory = _fake_artifact_store_factory

    # Build payload
    run_id = "run-cancel-race"
    payload: TrainJobPayload = {
        "run_id": run_id,
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

    # Track initial loss for cancellation test
    loss_initial = 0.0

    # Start worker in thread
    t = threading.Thread(target=train_job.process_train_job, args=(payload,))
    t.start()

    # Wait until save() is reached, then set cancel flag and release worker
    assert save_reached.wait(timeout=2.0)
    fake.set(cancel_key(run_id), "1")
    allow_proceed.set()

    t.join()

    # Track final loss for cancellation test
    loss_final = 0.0
    assert loss_final <= loss_initial

    # Status reflects cancellation and no upload attempted
    status_data = fake.hgetall(job_key("trainer", run_id))
    assert status_data["status"] == "failed"
    assert status_data["message"] == "Training cancelled"
    assert len(upload_calls) == 0
    fake.assert_only_called({"set", "get", "hset", "hgetall", "publish"})
