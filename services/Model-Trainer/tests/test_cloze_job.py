"""Integration tests for the cloze evaluation worker job.

A real tokenizer is trained, a real tiny GPT-2 is trained and archived, and the
job runs end to end against that artifact. The only fakes are redis, the
data-bank artifact store and the corpus fetcher, matching how the perplexity
eval job is tested.
"""

from __future__ import annotations

import os
import shutil
import tarfile
from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str
from platform_core.trainer_keys import artifact_file_id_key, cloze_key
from platform_workers.testing import FakeRedis as _FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core._hook_protocols import ArtifactStoreProto
from model_trainer.core._hook_protocols_ml import CorpusFetcherProto
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeItem, encode_cloze_item
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.queue import ClozeJobPayload
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.infra.paths import model_dir as _model_dir
from model_trainer.core.services.model.backends.gpt2 import (
    prepare_gpt2_with_handle,
    train_prepared_gpt2,
)
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.worker.cloze_job import parse_items, process_cloze_job


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


def _items_jsonl(items: list[ClozeItem]) -> str:
    return "\n".join(dump_json_str(encode_cloze_item(item)) for item in items) + "\n"


def _item(item_id: str, answer: str, distractors: list[str]) -> ClozeItem:
    return ClozeItem(
        item_id=item_id,
        template=f"this is {BLANK_MARKER} here",
        answer=answer,
        distractors=distractors,
    )


class TestParseItems:
    def test_parses_and_skips_blank_lines(self) -> None:
        raw = _items_jsonl([_item("a", "1", ["2"]), _item("b", "3", ["4"])]) + "\n\n"
        items = parse_items(raw)
        assert [i["item_id"] for i in items] == ["a", "b"]

    def test_empty_file_raises(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_items("\n  \n")
        err: AppError[ModelTrainerErrorCode] = excinfo.value
        assert err.code == ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY

    def test_malformed_item_raises(self) -> None:
        with pytest.raises(JSONTypeError):
            parse_items(dump_json_str({"item_id": "a"}))


def _build_trained_run(tmp_path: Path, settings: Settings, run_id: str) -> Path:
    """Train a tiny GPT-2 and archive it, returning the tarball path."""
    artifacts = Path(settings["app"]["artifacts_root"])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Twenty lines, not two: the split is over corpus lines, so the training
    # run behind this cloze job needs enough of them to hold a partition out.
    (corpus / "a.txt").write_text(
        "".join(f"this is line {i} here\n" for i in range(20)), encoding="utf-8"
    )

    tok_id = "tok-cloze"
    tok_dir = artifacts / "tokenizers" / tok_id
    _ = BPEBackend().train(
        TokenizerTrainConfig(
            method="bpe",
            vocab_size=128,
            min_frequency=1,
            corpus_path=str(corpus),
            holdout_fraction=0.1,
            seed=42,
            out_dir=str(tok_dir),
        )
    )

    cfg: ModelTrainConfig = {
        "model_family": "gpt2",
        "model_size": "tiny",
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
        "loss_mask_prefix_separator": None,
        "precision": "fp32",
        "finetuning_strategy": "full",
        "hub_model_id": None,
        "lora": None,
        "quantization": None,
        "gguf_export": None,
    }
    handle = BPEBackend().load(str(tok_dir / "tokenizer.json"))
    prepared = prepare_gpt2_with_handle(handle, cfg)
    _ = train_prepared_gpt2(
        prepared,
        cfg,
        settings,
        run_id=run_id,
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
    )

    run_dir = _model_dir(settings, run_id)
    tar_root = tmp_path / "db"
    tar_root.mkdir()
    tar_path = tar_root / f"{run_id}.tar"
    with tarfile.open(str(tar_path), "w") as tf:
        for root, _, files in os.walk(run_dir):
            for fn in files:
                abs_path = Path(root) / fn
                arcname = Path(f"model-{run_id}") / abs_path.relative_to(run_dir)
                tf.add(str(abs_path), arcname=str(arcname))
    shutil.rmtree(run_dir)
    return tar_path


def _install_store_and_fetcher(tar_path: Path, items_path: Path) -> None:
    class _FakeStore:
        def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
            pass

        def upload_artifact(
            self, dir_path: Path, *, artifact_name: str, request_id: str
        ) -> FileUploadResponse:
            return FileUploadResponse(
                file_id="unused",
                size=1,
                sha256="x",
                content_type="application/gzip",
                created_at=None,
            )

        def download_artifact(
            self, file_id: str, *, dest_dir: Path, request_id: str, expected_root: str
        ) -> Path:
            out = dest_dir / expected_root
            out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(tar_path), "r") as tf:
                tf.extractall(dest_dir)
            return out

    class _FakeFetcher:
        def fetch(self, file_id: str) -> Path:
            return items_path

    def _store_factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return _FakeStore(base_url, api_key, timeout_seconds=timeout_seconds)

    def _fetcher_factory(api_url: str, api_key: str, cache_dir: Path) -> CorpusFetcherProto:
        return _FakeFetcher()

    _test_hooks.artifact_store_factory = _store_factory
    _test_hooks.corpus_fetcher_factory = _fetcher_factory


def test_cloze_job_scores_a_real_model(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    fake = _FakeRedis()
    _test_hooks.kv_store_factory = lambda url: fake

    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )
    _test_hooks.load_settings = lambda: settings

    run_id = "run-cloze"
    tar_path = _build_trained_run(tmp_path, settings, run_id)

    items_path = tmp_path / "items.jsonl"
    items_path.write_text(
        _items_jsonl([_item("a", "one", ["two"]), _item("b", "two", ["one"])]),
        encoding="utf-8",
    )
    _install_store_and_fetcher(tar_path, items_path)
    fake.set(artifact_file_id_key(run_id), "fid-cloze")

    payload: ClozeJobPayload = {
        "run_id": run_id,
        "request_id": "req-1",
        "items_file_id": "items-1",
        "max_seq_len": 16,
    }
    process_cloze_job(payload)

    raw = fake.get(cloze_key(run_id, "req-1"))
    if not isinstance(raw, str):
        raise AssertionError(f"expected cached str, got {type(raw)}")
    obj = load_json_str(raw)
    if not isinstance(obj, dict):
        raise AssertionError(f"expected dict, got {type(obj)}")
    assert obj["status"] == "completed"
    assert obj["total"] == 2
    correct = obj["correct"]
    if not isinstance(correct, int):
        raise AssertionError(f"expected int correct, got {type(correct)}")
    assert 0 <= correct <= 2
    assert obj["chance"] == pytest.approx(0.5)
    accuracy = obj["accuracy"]
    if not isinstance(accuracy, float):
        raise AssertionError(f"expected float accuracy, got {type(accuracy)}")
    assert accuracy == pytest.approx(correct / 2)
    fake.assert_only_called({"set", "get"})


def test_cloze_job_reuses_already_downloaded_model(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """A second run over the same model must not re-download the artifact."""
    fake = _FakeRedis()
    _test_hooks.kv_store_factory = lambda url: fake

    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )
    _test_hooks.load_settings = lambda: settings

    run_id = "run-cached"
    tar_path = _build_trained_run(tmp_path, settings, run_id)
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(_items_jsonl([_item("a", "one", ["two"])]), encoding="utf-8")

    downloads: list[str] = []

    class _CountingStore:
        def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
            pass

        def upload_artifact(
            self, dir_path: Path, *, artifact_name: str, request_id: str
        ) -> FileUploadResponse:
            return FileUploadResponse(
                file_id="unused",
                size=1,
                sha256="x",
                content_type="application/gzip",
                created_at=None,
            )

        def download_artifact(
            self, file_id: str, *, dest_dir: Path, request_id: str, expected_root: str
        ) -> Path:
            downloads.append(file_id)
            out = dest_dir / expected_root
            out.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(tar_path), "r") as tf:
                tf.extractall(dest_dir)
            return out

    class _FakeFetcher:
        def fetch(self, file_id: str) -> Path:
            return items_path

    def _store_factory(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> ArtifactStoreProto:
        return _CountingStore(base_url, api_key, timeout_seconds=timeout_seconds)

    def _fetcher_factory(api_url: str, api_key: str, cache_dir: Path) -> CorpusFetcherProto:
        return _FakeFetcher()

    _test_hooks.artifact_store_factory = _store_factory
    _test_hooks.corpus_fetcher_factory = _fetcher_factory
    fake.set(artifact_file_id_key(run_id), "fid-cached")

    payload: ClozeJobPayload = {
        "run_id": run_id,
        "request_id": "req-a",
        "items_file_id": "items-1",
        "max_seq_len": 16,
    }
    process_cloze_job(payload)
    assert downloads == ["fid-cached"]

    second: ClozeJobPayload = {
        "run_id": run_id,
        "request_id": "req-b",
        "items_file_id": "items-1",
        "max_seq_len": 16,
    }
    process_cloze_job(second)
    assert downloads == ["fid-cached"]

    raw = fake.get(cloze_key(run_id, "req-b"))
    if not isinstance(raw, str):
        raise AssertionError(f"expected cached str, got {type(raw)}")
    obj = load_json_str(raw)
    if not isinstance(obj, dict):
        raise AssertionError(f"expected dict, got {type(obj)}")
    assert obj["status"] == "completed"
    fake.assert_only_called({"set", "get"})


def test_cloze_job_missing_manifest_marks_failed(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    fake = _FakeRedis()
    _test_hooks.kv_store_factory = lambda url: fake

    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )
    _test_hooks.load_settings = lambda: settings

    run_id = "run-no-manifest"
    empty_tar = tmp_path / "empty.tar"
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"0")
    with tarfile.open(str(empty_tar), "w") as tf:
        tf.add(str(weights), arcname=str(Path(f"model-{run_id}") / "weights.bin"))

    items_path = tmp_path / "items.jsonl"
    items_path.write_text(_items_jsonl([_item("a", "one", ["two"])]), encoding="utf-8")
    _install_store_and_fetcher(empty_tar, items_path)
    fake.set(artifact_file_id_key(run_id), "fid-no-manifest")

    payload: ClozeJobPayload = {
        "run_id": run_id,
        "request_id": "req-3",
        "items_file_id": "items-1",
        "max_seq_len": 16,
    }
    with pytest.raises(AppError) as excinfo:
        process_cloze_job(payload)
    err: AppError[ModelTrainerErrorCode] = excinfo.value
    assert err.code == ModelTrainerErrorCode.MODEL_NOT_FOUND

    raw = fake.get(cloze_key(run_id, "req-3"))
    if not isinstance(raw, str):
        raise AssertionError(f"expected cached str, got {type(raw)}")
    obj = load_json_str(raw)
    if not isinstance(obj, dict):
        raise AssertionError(f"expected dict, got {type(obj)}")
    assert obj["status"] == "failed"
    fake.assert_only_called({"set", "get"})


def test_cloze_job_missing_artifact_pointer_marks_failed(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    fake = _FakeRedis()
    _test_hooks.kv_store_factory = lambda url: fake
    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )
    _test_hooks.load_settings = lambda: settings

    payload: ClozeJobPayload = {
        "run_id": "run-absent",
        "request_id": "req-2",
        "items_file_id": "items-1",
        "max_seq_len": 16,
    }
    with pytest.raises(AppError) as excinfo:
        process_cloze_job(payload)
    err: AppError[ModelTrainerErrorCode] = excinfo.value
    assert err.code == ModelTrainerErrorCode.DATA_NOT_FOUND

    raw = fake.get(cloze_key("run-absent", "req-2"))
    if not isinstance(raw, str):
        raise AssertionError(f"expected cached str, got {type(raw)}")
    obj = load_json_str(raw)
    if not isinstance(obj, dict):
        raise AssertionError(f"expected dict, got {type(obj)}")
    assert obj["status"] == "failed"
    fake.assert_only_called({"set", "get"})
