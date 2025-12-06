from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.json_utils import JSONValue, load_json_str
from typing_extensions import TypedDict

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backends.gpt2 import (
    evaluate_gpt2,
    prepare_gpt2_with_handle,
    train_prepared_gpt2,
)
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend


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


def _validate_versions_dict(versions_raw: dict[str, JSONValue]) -> dict[str, str]:
    """Validate versions dict - all values must be non-empty strings."""
    versions_typed: dict[str, str] = {}
    for k, v in versions_raw.items():
        if not isinstance(v, str):
            raise AssertionError(f"versions[{k}] must be str, got {type(v)}")
        if not v:
            raise AssertionError(f"versions[{k}] must be non-empty")
        versions_typed[k] = v
    return versions_typed


def _validate_system_dict(system_raw: dict[str, JSONValue]) -> dict[str, str | int]:
    """Validate system dict - values must be non-empty str or int."""
    system_typed: dict[str, str | int] = {}
    for k, v in system_raw.items():
        is_str = isinstance(v, str) and len(v) >= 1
        is_int = isinstance(v, int) and not isinstance(v, bool)
        if not (is_str or is_int):
            raise AssertionError(f"system[{k}] must be non-empty str or int")
        if not isinstance(v, (str, int)) or isinstance(v, bool):
            raise AssertionError(f"system[{k}] type check failed")
        system_typed[k] = v
    return system_typed


class _Manifest(TypedDict):
    run_id: str
    epochs: int
    batch_size: int
    max_seq_len: int
    steps: int
    loss: float
    tokenizer_id: str
    corpus_path: str
    optimizer: str
    seed: int
    versions: dict[str, str]
    system: dict[str, str | int]
    git_commit: str | None


def test_training_and_eval_tiny(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Use a larger corpus with repetitive patterns for stable training
    pattern = "hello world this is a test\n" * 50 + "testing the model training\n" * 50
    (corpus / "a.txt").write_text(pattern, encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    settings = settings_factory(
        artifacts_root=str(artifacts),
        data_root=str(tmp_path / "data"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
    )

    # Train tokenizer
    tok_id = "tok-test"
    out_dir = artifacts / "tokenizers" / tok_id
    cfg_tok = TokenizerTrainConfig(
        method="bpe",
        vocab_size=128,
        min_frequency=1,
        corpus_path=str(corpus),
        holdout_fraction=0.1,
        seed=42,
        out_dir=str(out_dir),
    )
    _ = BPEBackend().train(cfg_tok)

    # Prepare model with adequate batch size for stable training
    cfg: ModelTrainConfig = {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 16,
        "num_epochs": 3,  # Multiple epochs for loss reduction test
        "batch_size": 4,
        "learning_rate": 5e-4,
        "tokenizer_id": tok_id,
        "corpus_path": str(corpus),
        "holdout_fraction": 0.1,
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

    def _hb(_: float) -> None:
        pass

    def _cancelled() -> bool:
        return False

    train_losses: list[float] = []

    def track_loss(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
        val_loss: float | None,
        val_ppl: float | None,
    ) -> None:
        train_losses.append(loss)

    builder = LocalTextDatasetBuilder()
    tok_handle = BPEBackend().load(str(out_dir / "tokenizer.json"))
    prepared = prepare_gpt2_with_handle(tok_handle, cfg)
    res = train_prepared_gpt2(
        prepared,
        cfg,
        settings,
        run_id="run-test",
        redis_hb=_hb,
        cancelled=_cancelled,
        progress=track_loss,
    )
    assert res["loss"] >= 0.0
    loss_before = train_losses[0]
    loss_after = train_losses[-1]
    assert loss_after < loss_before

    # Manifest written
    manifest = artifacts / "models" / "run-test" / "manifest.json"
    assert manifest.exists()

    text = manifest.read_text(encoding="utf-8")
    obj_raw = load_json_str(text)
    if not isinstance(obj_raw, dict):
        raise AssertionError("Manifest must be a dict")
    assert "run_id" in obj_raw
    cfg_raw: JSONValue = obj_raw.get("config")
    if not isinstance(cfg_raw, dict):
        raise AssertionError("config must be a dict")
    versions_raw: JSONValue = obj_raw.get("versions")
    system_raw: JSONValue = obj_raw.get("system")
    if not isinstance(versions_raw, dict) or not isinstance(system_raw, dict):
        raise AssertionError("versions and system must be dicts")

    versions_typed = _validate_versions_dict(versions_raw)
    system_typed = _validate_system_dict(system_raw)

    # Validate manifest fields
    assert obj_raw.get("run_id") == "run-test"
    assert obj_raw.get("epochs") == 3
    assert obj_raw.get("batch_size") == 4
    assert obj_raw.get("max_seq_len") == 16
    steps = obj_raw.get("steps", 0)
    assert isinstance(steps, int) and steps > 0
    loss = obj_raw.get("loss", 0.0)
    assert isinstance(loss, (int, float)) and loss >= 0.0
    assert obj_raw.get("tokenizer_id") == tok_id
    corpus_path = obj_raw.get("corpus_path", "")
    assert isinstance(corpus_path, str) and str(corpus) in corpus_path
    assert obj_raw.get("optimizer") == "adamw"
    assert obj_raw.get("seed") == 42
    git_commit = obj_raw.get("git_commit")

    # Extract typed values with narrowing
    run_id_v = obj_raw.get("run_id", "")
    epochs_v = obj_raw.get("epochs", 0)
    batch_size_v = obj_raw.get("batch_size", 0)
    max_seq_len_v = obj_raw.get("max_seq_len", 0)
    tokenizer_id_v = obj_raw.get("tokenizer_id", "")
    optimizer_v = obj_raw.get("optimizer", "")
    seed_v = obj_raw.get("seed", 0)

    m: _Manifest = {
        "run_id": str(run_id_v) if isinstance(run_id_v, str) else "",
        "epochs": epochs_v if isinstance(epochs_v, int) else 0,
        "batch_size": batch_size_v if isinstance(batch_size_v, int) else 0,
        "max_seq_len": max_seq_len_v if isinstance(max_seq_len_v, int) else 0,
        "steps": steps,
        "loss": float(loss),
        "tokenizer_id": str(tokenizer_id_v) if isinstance(tokenizer_id_v, str) else "",
        "corpus_path": corpus_path,
        "optimizer": str(optimizer_v) if isinstance(optimizer_v, str) else "",
        "seed": seed_v if isinstance(seed_v, int) else 0,
        "versions": versions_typed,
        "system": system_typed,
        "git_commit": git_commit if isinstance(git_commit, str) else None,
    }
    assert m["tokenizer_id"] == tok_id

    # Eval metrics
    eval_res = evaluate_gpt2(run_id="run-test", cfg=cfg, settings=settings, dataset_builder=builder)
    assert eval_res.loss >= 0.0
    metrics_path = artifacts / "models" / "run-test" / "eval" / "metrics.json"
    assert metrics_path.exists()
