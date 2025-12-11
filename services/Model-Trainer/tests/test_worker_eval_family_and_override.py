from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import dump_json_str
from platform_core.trainer_keys import artifact_file_id_key, eval_key
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis as _FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core._test_hooks import ArtifactStoreProto, ServiceContainerProto
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
from model_trainer.core.contracts.queue import EvalJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.unavailable_backend import UNAVAILABLE_CAPABILITIES
from model_trainer.core.services.registries import BackendRegistration, ModelRegistry
from model_trainer.worker.eval_job import process_eval_job

# --- Helper classes extracted from test for C901 complexity reduction ---


# Type alias for manifest dict to avoid line length violations
_ManifestDict = dict[str, str | int | float | bool | None | dict[str, str | int]]


class _FakeStoreForEval:
    """Fake ArtifactStore for eval job tests."""

    _manifest: _ManifestDict

    def __init__(
        self,
        manifest: _ManifestDict,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 600.0,
    ) -> None:
        self._manifest = manifest

    def upload_artifact(
        self,
        dir_path: Path,
        *,
        artifact_name: str,
        request_id: str,
    ) -> FileUploadResponse:
        raise NotImplementedError

    def download_artifact(
        self,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        # Simulate remote archive materialized under expected_root with manifest.json
        out = dest_dir / expected_root
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(dump_json_str(self._manifest), encoding="utf-8")
        return out


class _FakeServiceContainerForEval:
    """Fake ServiceContainer for eval job tests."""

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


def _create_eval_manifest(run_id: str) -> _ManifestDict:
    """Create a manifest dict for eval job tests."""
    return {
        "run_id": run_id,
        "model_family": "char_lstm",
        "model_size": "small",
        "epochs": 1,
        "batch_size": 1,
        "max_seq_len": 8,
        "steps": 0,
        "loss": 0.0,
        "learning_rate": 0.001,
        "tokenizer_id": "tok",
        "corpus_path": "/orig/path",
        "holdout_fraction": 0.05,
        "optimizer": "adamw",
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "device": "cpu",
        "precision": "fp32",
        "early_stopping_patience": 5,
        "test_split_ratio": 0.15,
        "finetune_lr_cap": 5e-5,
        "early_stopped": False,
        "versions": {"torch": "0", "transformers": "0", "tokenizers": "0", "datasets": "0"},
        "system": {"cpu_count": 1, "platform": "x", "platform_release": "y", "machine": "z"},
        "seed": 0,
        "pretrained_run_id": None,
        "git_commit": "g",
    }


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


class _StubBackend(ModelBackend):
    def __init__(self) -> None:
        self.called = False
        self.last_cfg: ModelTrainConfig | None = None

    def name(self) -> str:
        return "char_lstm"

    def capabilities(self) -> BackendCapabilities:
        return UNAVAILABLE_CAPABILITIES

    def prepare(
        self, cfg: ModelTrainConfig, settings: Settings, *, tokenizer: TokenizerHandle
    ) -> PreparedLMModel:
        raise NotImplementedError

    def save(self, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
        raise NotImplementedError

    def load(
        self, artifact_path: str, settings: Settings, *, tokenizer: TokenizerHandle
    ) -> PreparedLMModel:
        raise NotImplementedError

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
        wandb_publisher: WandbPublisher | None = None,
    ) -> TrainOutcome:
        raise NotImplementedError

    def evaluate(self, *, run_id: str, cfg: ModelTrainConfig, settings: Settings) -> EvalOutcome:
        self.called = True
        self.last_cfg = cfg
        return EvalOutcome(loss=1.0, perplexity=2.0)

    def score(
        self, *, prepared: PreparedLMModel, cfg: ScoreConfig, settings: Settings
    ) -> ScoreOutcome:
        raise NotImplementedError

    def generate(
        self, *, prepared: PreparedLMModel, cfg: GenerateConfig, settings: Settings
    ) -> GenerateOutcome:
        raise NotImplementedError


def test_eval_job_uses_manifest_family_and_path_override(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    # Save original hooks
    orig_kv = _test_hooks.kv_store_factory
    orig_load_settings = _test_hooks.load_settings
    orig_artifact_store = _test_hooks.artifact_store_factory
    orig_service_container = _test_hooks.service_container_from_settings

    fake = _FakeRedis()

    def _redis_for_kv(url: str) -> _FakeRedis:
        return fake

    _test_hooks.kv_store_factory = _redis_for_kv

    artifacts = tmp_path / "artifacts"
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

    # Prepare manifest using extracted helper
    run_id = "run-eval-char"
    manifest = _create_eval_manifest(run_id)

    # Create fake artifact store using extracted class
    def _fake_artifact_store_factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return _FakeStoreForEval(manifest, base_url, api_key, timeout_seconds=timeout_seconds)

    _test_hooks.artifact_store_factory = _fake_artifact_store_factory

    # Wire ServiceContainer to return a registry where char_lstm returns our stub backend
    stub_backend = _StubBackend()

    def _fake_service_container_from_settings(settings: Settings) -> ServiceContainerProto:
        model_registry = ModelRegistry(
            registrations={
                "char_lstm": BackendRegistration(
                    factory=lambda _: stub_backend, capabilities=UNAVAILABLE_CAPABILITIES
                )
            },
            dataset_builder=LocalTextDatasetBuilder(),
        )
        return _FakeServiceContainerForEval(settings, fake, model_registry)

    _test_hooks.service_container_from_settings = _fake_service_container_from_settings

    try:
        # Set artifact pointer
        fake.set(artifact_file_id_key(run_id), "fid")

        # Invoke eval with path_override, ensure backend called and override applied
        typed_payload: EvalJobPayload = {
            "run_id": run_id,
            "split": "validation",
            "path_override": str(tmp_path / "override"),
        }
        process_eval_job(typed_payload)  # no exception
        assert stub_backend.called is True
        cfg = stub_backend.last_cfg
        assert cfg is not None and cfg["corpus_path"] == str(tmp_path / "override")
        # Result cached
        raw = fake.get(eval_key(run_id))
        assert isinstance(raw, str) and "completed" in raw
        fake.assert_only_called({"set", "get"})
    finally:
        # Restore original hooks
        _test_hooks.kv_store_factory = orig_kv
        _test_hooks.load_settings = orig_load_settings
        _test_hooks.artifact_store_factory = orig_artifact_store
        _test_hooks.service_container_from_settings = orig_service_container
