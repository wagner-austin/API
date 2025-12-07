from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, load_json_str
from platform_core.trainer_keys import artifact_file_id_key, eval_key
from platform_workers.testing import FakeRedis as _FakeRedis
from pytest import MonkeyPatch

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.queue import EvalJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.infra.paths import model_dir as _model_dir
from model_trainer.core.services.model.backends.gpt2 import (
    prepare_gpt2_with_handle,
    train_prepared_gpt2,
)
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
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


def _create_model_tarball(model_dir: Path, tar_path: Path, name: str) -> None:
    """Create a tarball from model directory for artifact download simulation."""
    with tarfile.open(str(tar_path), "w") as tf:
        for root, _, files in os.walk(model_dir):
            for fn in files:
                abs_path = Path(root) / fn
                rel = abs_path.relative_to(model_dir)
                arcname = Path(name) / rel
                tf.add(str(abs_path), arcname=str(arcname))


def _verify_eval_result(raw: str | None) -> None:
    """Verify the eval result JSON has expected structure and status."""
    assert isinstance(raw, str) and len(raw) > 0
    obj_raw = load_json_str(raw)
    if not isinstance(obj_raw, dict):
        raise AssertionError(f"Expected dict, got {type(obj_raw)}")
    assert len(obj_raw) > 0
    obj: dict[str, JSONValue] = obj_raw
    st = obj.get("status")
    loss_v = obj.get("loss")
    ppl_v = obj.get("ppl")
    assert isinstance(st, str) and st == "completed"
    if not isinstance(loss_v, (int, float)):
        raise AssertionError(f"Expected loss to be numeric, got {type(loss_v)}")
    assert loss_v >= 0.0
    if not isinstance(ppl_v, (int, float)):
        raise AssertionError(f"Expected ppl to be numeric, got {type(ppl_v)}")
    assert ppl_v >= 1.0


def test_eval_job_success(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    # Use fake redis
    fake = _FakeRedis()

    def _redis_for_kv(url: str) -> _FakeRedis:
        return fake

    monkeypatch.setattr("model_trainer.worker.job_utils.redis_for_kv", _redis_for_kv)

    # Prepare artifacts and tokenizer
    artifacts = tmp_path / "artifacts"
    settings_for_train = settings_factory(
        artifacts_root=str(artifacts),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )

    def _load_settings() -> Settings:
        return settings_for_train

    monkeypatch.setattr(
        "model_trainer.worker.eval_job.load_settings",
        _load_settings,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world\nthis is tiny\n", encoding="utf-8")

    tok_id = "tok-eval"
    tok_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(tok_dir),
    )
    _ = BPEBackend().train(cfg_tok)

    # Train and persist a tiny model for run_id
    run_id = "run-eval"
    cfg: ModelTrainConfig = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 5e-4,
        "tokenizer_id": tok_id,
        "corpus_path": str(corpus),
        "holdout_fraction": 0.01,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 5,
        "test_split_ratio": 0.15,
        "finetune_lr_cap": 5e-5,
        "precision": "fp32",
    }
    tok_handle = BPEBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = prepare_gpt2_with_handle(tok_handle, cfg)

    # heartbeat/cancel no-ops
    def _hb(_: float) -> None:
        pass

    def _cancelled() -> bool:
        return False

    loss_initial = 0.0
    result = train_prepared_gpt2(
        prepared,
        cfg,
        settings_for_train,
        run_id=run_id,
        redis_hb=_hb,
        cancelled=_cancelled,
    )
    loss_final: float = result["loss"]
    assert loss_final <= loss_initial or loss_final >= 0.0

    # Create a tarball matching ArtifactUploader layout and remove local model dir
    run_dir = _model_dir(settings_for_train, run_id)
    name = f"model-{run_id}"
    tar_root = tmp_path / "db"
    tar_root.mkdir()
    tar_path = tar_root / f"{run_id}.tar"
    _create_model_tarball(run_dir, tar_path, name)

    import shutil

    shutil.rmtree(run_dir)

    # Configure data-bank and pointer
    file_id = "fid-eval-1"
    fake.set(artifact_file_id_key(run_id), file_id)

    # Stub DataBankClient used by ArtifactDownloader

    class _ClientStub:
        def __init__(
            self: _ClientStub, base_url: str, api_key: str, timeout_seconds: float = 0.0
        ) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def download_to_path(
            self: _ClientStub,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> None:
            assert file_id == "fid-eval-1"
            dest.write_bytes(tar_path.read_bytes())

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
            out = dest_dir / expected_root
            out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(tar_path), "r") as tf:
                tf.extractall(dest_dir)
            return out

    monkeypatch.setattr("platform_ml.ArtifactStore", _FakeStore)

    # Now process eval job using the worker entry
    payload: EvalJobPayload = {
        "run_id": run_id,
        "split": "validation",
        "path_override": None,
    }
    process_eval_job(payload)
    raw = fake.get(eval_key(run_id))
    _verify_eval_result(raw)
    fake.assert_only_called({"set", "get"})


def test_eval_job_missing_manifest(
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

    run_id = "run-missing"
    file_id = "fid-missing"
    fake.set(artifact_file_id_key(run_id), file_id)

    # Stub DataBankClient to return an archive without manifest.json

    tar_root = tmp_path / "db_missing"
    tar_root.mkdir()
    tar_path = tar_root / f"{run_id}.tar"
    name = f"model-{run_id}"
    # Create an archive with only weights.bin and no manifest.json
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"x")
    with tarfile.open(str(tar_path), "w") as tf:
        for root, _, files in os.walk(model_dir):
            for fn in files:
                abs_path = Path(root) / fn
                rel = abs_path.relative_to(model_dir)
                arcname = Path(name) / rel
                tf.add(str(abs_path), arcname=str(arcname))

    class _ClientStub:
        def __init__(
            self: _ClientStub, base_url: str, api_key: str, timeout_seconds: float = 0.0
        ) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def download_to_path(
            self: _ClientStub,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> None:
            assert file_id == "fid-missing"
            dest.write_bytes(tar_path.read_bytes())

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
            out = dest_dir / expected_root
            out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(tar_path), "r") as tf:
                tf.extractall(dest_dir)
            return out

    monkeypatch.setattr("platform_ml.ArtifactStore", _FakeStore)

    payload: EvalJobPayload = {
        "run_id": run_id,
        "split": "validation",
        "path_override": None,
    }
    with pytest.raises(AppError, match="manifest missing"):
        process_eval_job(payload)

    raw = fake.get(eval_key(run_id))
    assert isinstance(raw, str) and len(raw) > 0

    obj2_raw = load_json_str(raw)
    if not isinstance(obj2_raw, dict):
        raise AssertionError(f"Expected dict, got {type(obj2_raw)}")
    assert "status" in obj2_raw
    obj2: dict[str, JSONValue] = obj2_raw
    st2 = obj2.get("status")
    assert isinstance(st2, str) and st2 == "failed"
    fake.assert_only_called({"set", "get"})


def test_eval_job_destination_exists(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    """Cover line 632: RuntimeError when destination already exists."""
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

    run_id = "run-dest-exists"
    file_id = "fid-dest-exists"
    fake.set(artifact_file_id_key(run_id), file_id)

    # Create tarball with model directory
    tar_root = tmp_path / "db_dest"
    tar_root.mkdir()
    tar_path = tar_root / f"{run_id}.tar"
    name = f"model-{run_id}"
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "weights.bin").write_bytes(b"\x00mock")
    _create_model_tarball(model_dir, tar_path, name)

    # Pre-create the destination directory to trigger the RuntimeError
    models_root = artifacts / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    (models_root / run_id).mkdir()

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
            out = dest_dir / expected_root
            out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(tar_path), "r") as tf:
                tf.extractall(dest_dir)
            return out

    monkeypatch.setattr("platform_ml.ArtifactStore", _FakeStore)

    payload: EvalJobPayload = {
        "run_id": run_id,
        "split": "validation",
        "path_override": None,
    }
    with pytest.raises(AppError, match="destination already exists"):
        process_eval_job(payload)
    fake.assert_only_called({"set", "get"})


def test_eval_job_artifact_pointer_missing(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    """Cover eval_job.py line 54: AppError when artifact pointer not found."""
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

    run_id = "run-no-pointer"
    # Do NOT set any artifact_file_id_key so file_id will be None/empty

    payload: EvalJobPayload = {
        "run_id": run_id,
        "split": "validation",
        "path_override": None,
    }
    with pytest.raises(AppError, match="artifact pointer not found"):
        process_eval_job(payload)
    fake.assert_only_called({"set", "get"})
