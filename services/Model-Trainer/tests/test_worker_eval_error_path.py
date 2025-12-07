from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import dump_json_str
from platform_core.trainer_keys import EVAL_KEY_PREFIX, artifact_file_id_key
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.contracts.queue import EvalJobPayload
from model_trainer.worker.eval_job import process_eval_job


class _Backend:
    def evaluate(
        self: _Backend,
        *,
        run_id: str,
        cfg: dict[str, str | int | float | bool],
        settings: Settings,
    ) -> dict[str, float]:
        raise RuntimeError("boom")


class _ModelRegistry:
    def get(self: _ModelRegistry, name: str) -> _Backend:
        return _Backend()


class _Container:
    def __init__(self: _Container, registry: _ModelRegistry) -> None:
        self.model_registry = registry


def test_worker_eval_backend_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Fake redis (shared instance for reads in assertions)
    fake = FakeRedis()

    def _redis_for_kv(url: str) -> FakeRedis:
        return fake

    monkeypatch.setattr("model_trainer.worker.job_utils.redis_for_kv", _redis_for_kv)

    # Load settings to get artifacts_root
    s = load_settings()
    artifacts = Path(s["app"]["artifacts_root"])
    models_root = artifacts / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    # Manifest will be created by _FakeStore at download time (not pre-created at final location)
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
        "corpus_path": str(tmp_path),
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

    # Set artifact pointer to bypass download flow
    fake.set(artifact_file_id_key("run-err"), "fid-test")

    # Mock ArtifactStore: download creates manifest at temp location (dest_dir / expected_root)
    # which the code then renames to final location (dest_dir / run_id)
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
            # Simulate download: create at temp location (dest_dir / expected_root)
            out = dest_dir / expected_root
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")
            return out

    monkeypatch.setattr("platform_ml.ArtifactStore", _FakeStore)

    # Patch container factory to return backend that raises
    def _from_settings(settings: Settings) -> _Container:
        return _Container(_ModelRegistry())

    monkeypatch.setattr(
        "model_trainer.core.services.container.ServiceContainer.from_settings", _from_settings
    )

    # Now run eval and assert failure is recorded and exception propagated
    payload: EvalJobPayload = {"run_id": "run-err", "split": "validation", "path_override": None}
    with pytest.raises(RuntimeError):
        process_eval_job(payload)
    raw = fake.get(f"{EVAL_KEY_PREFIX}run-err")
    assert isinstance(raw, str) and len(raw) > 0
    fake.assert_only_called({"set", "get"})
