from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from platform_core.json_utils import dump_json_str
from platform_core.trainer_keys import artifact_file_id_key, eval_key
from platform_ml.wandb_publisher import WandbPublisher
from platform_workers.testing import FakeRedis as _FakeRedis
from pytest import MonkeyPatch

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
from model_trainer.core.services.model.unavailable_backend import UNAVAILABLE_CAPABILITIES
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
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    fake = _FakeRedis()

    def _redis_for_kv(url: str) -> _FakeRedis:
        return fake

    monkeypatch.setattr("model_trainer.worker.job_utils.redis_for_kv", _redis_for_kv)

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

    monkeypatch.setattr("model_trainer.worker.eval_job.load_settings", _load_settings)

    # Prepare a manifest object; it will be written by the fake store into the download path
    run_id = "run-eval-char"
    manifest = {
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
    # Do not pre-create destination; the worker will create it after download

    # Fake store download as no-op: models dir already present
    class _FakeStore:
        def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
            pass

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
            (out / "manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")
            return out

    monkeypatch.setattr("platform_ml.ArtifactStore", _FakeStore)

    # Wire ServiceContainer to return a registry where char_lstm returns our stub backend
    from model_trainer.core.services.container import ServiceContainer
    from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
    from model_trainer.core.services.registries import BackendRegistration, ModelRegistry

    stub_backend = _StubBackend()

    class _SC(ServiceContainer):
        @classmethod
        def from_settings(cls, s: Settings) -> ServiceContainer:
            r = ServiceContainer.from_settings(s)
            # Replace model registry with stub backend
            r.model_registry = ModelRegistry(
                registrations={
                    "char_lstm": BackendRegistration(
                        factory=lambda _: stub_backend, capabilities=UNAVAILABLE_CAPABILITIES
                    )
                },
                dataset_builder=LocalTextDatasetBuilder(),
            )
            return r

    monkeypatch.setattr("model_trainer.worker.eval_job.ServiceContainer", _SC)

    # Set artifact pointer
    fake.set(artifact_file_id_key(run_id), "fid")

    # Invoke eval with path_override, ensure backend called and override applied
    payload: dict[str, str | None] = {
        "run_id": run_id,
        "split": "validation",
        "path_override": str(tmp_path / "override"),
    }
    from model_trainer.core.contracts.queue import EvalJobPayload

    override_val = payload["path_override"]
    typed_payload: EvalJobPayload = {
        "run_id": str(payload["run_id"]),
        "split": str(payload["split"]),
        "path_override": str(override_val) if override_val is not None else None,
    }
    process_eval_job(typed_payload)  # no exception
    assert stub_backend.called is True
    cfg = stub_backend.last_cfg
    assert cfg is not None and cfg["corpus_path"] == str(tmp_path / "override")
    # Result cached
    raw = fake.get(eval_key(run_id))
    assert isinstance(raw, str) and "completed" in raw
    fake.assert_only_called({"set", "get"})
