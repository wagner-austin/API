"""Manifest parsing for training worker."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_str,
)

from model_trainer.infra.persistence.models import (
    TrainingManifest,
    TrainingManifestSystem,
    TrainingManifestVersions,
)


def as_model_family(s: str) -> Literal["gpt2", "llama", "qwen", "char_lstm"]:
    """Convert string to model family literal type."""
    if s == "gpt2":
        return "gpt2"
    if s == "llama":
        return "llama"
    if s == "qwen":
        return "qwen"
    if s == "char_lstm":
        return "char_lstm"
    raise JSONTypeError(f"Invalid model_family: {s}")


def as_optimizer(s: str) -> Literal["adamw", "adam", "sgd"]:
    """Convert string to optimizer literal type."""
    if s == "adamw":
        return "adamw"
    if s == "adam":
        return "adam"
    if s == "sgd":
        return "sgd"
    raise JSONTypeError(f"Invalid optimizer: {s}")


def as_device(s: str) -> Literal["cpu", "cuda"]:
    """Convert string to device literal type."""
    if s == "cpu":
        return "cpu"
    if s == "cuda":
        return "cuda"
    raise JSONTypeError(f"Invalid device: {s}")


def as_precision(s: str) -> Literal["fp32", "fp16", "bf16"]:
    """Convert string to precision literal type."""
    if s == "fp32":
        return "fp32"
    if s == "fp16":
        return "fp16"
    if s == "bf16":
        return "bf16"
    raise JSONTypeError(f"Invalid precision: {s}")


def _decode_manifest_versions(obj: JSONObject) -> TrainingManifestVersions:
    vers = require_dict(obj, "versions")
    return {
        "torch": require_str(vers, "torch"),
        "transformers": require_str(vers, "transformers"),
        "tokenizers": require_str(vers, "tokenizers"),
        "datasets": require_str(vers, "datasets"),
    }


def _decode_manifest_system(obj: JSONObject) -> TrainingManifestSystem:
    sys = require_dict(obj, "system")
    return {
        "cpu_count": require_int(sys, "cpu_count"),
        "platform": require_str(sys, "platform"),
        "platform_release": require_str(sys, "platform_release"),
        "machine": require_str(sys, "machine"),
    }


def _optional_str(obj: JSONObject, key: str) -> str | None:
    """Extract optional string field."""
    val = obj.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise JSONTypeError(f"Field '{key}' must be a string or null, got {type(val).__name__}")
    return val


def _optional_float(obj: JSONObject, key: str) -> float | None:
    """Extract optional float field."""
    val = obj.get(key)
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, int | float):
        raise JSONTypeError(f"Field '{key}' must be a number or null, got {type(val).__name__}")
    return float(val)


class _ManifestFields:
    """Container for decoded manifest fields to avoid long tuples."""

    run_id: str
    model_family: str
    model_size: str
    epochs: int
    batch_size: int
    max_seq_len: int
    steps: int
    loss: float
    learning_rate: float
    holdout_fraction: float
    tokenizer_id: str
    corpus_path: str
    optimizer: str
    freeze_embed: bool
    gradient_clipping: float
    seed: int
    git_commit: str | None
    pretrained_run_id: str | None
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    test_loss: float | None
    test_perplexity: float | None
    best_val_loss: float | None
    early_stopped: bool

    def __init__(
        self: _ManifestFields,
        *,
        run_id: str,
        model_family: str,
        model_size: str,
        epochs: int,
        batch_size: int,
        max_seq_len: int,
        steps: int,
        loss: float,
        learning_rate: float,
        holdout_fraction: float,
        tokenizer_id: str,
        corpus_path: str,
        optimizer: str,
        freeze_embed: bool,
        gradient_clipping: float,
        seed: int,
        git_commit: str | None,
        pretrained_run_id: str | None,
        device: str,
        precision: str,
        early_stopping_patience: int,
        test_split_ratio: float,
        finetune_lr_cap: float,
        test_loss: float | None,
        test_perplexity: float | None,
        best_val_loss: float | None,
        early_stopped: bool,
    ) -> None:
        self.run_id = run_id
        self.model_family = model_family
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.steps = steps
        self.loss = loss
        self.learning_rate = learning_rate
        self.holdout_fraction = holdout_fraction
        self.tokenizer_id = tokenizer_id
        self.corpus_path = corpus_path
        self.optimizer = optimizer
        self.freeze_embed = freeze_embed
        self.gradient_clipping = gradient_clipping
        self.seed = seed
        self.git_commit = git_commit
        self.pretrained_run_id = pretrained_run_id
        self.device = device
        self.precision = precision
        self.early_stopping_patience = early_stopping_patience
        self.test_split_ratio = test_split_ratio
        self.finetune_lr_cap = finetune_lr_cap
        self.test_loss = test_loss
        self.test_perplexity = test_perplexity
        self.best_val_loss = best_val_loss
        self.early_stopped = early_stopped


def _decode_manifest_fields(obj: JSONObject) -> _ManifestFields:
    return _ManifestFields(
        run_id=require_str(obj, "run_id"),
        model_family=require_str(obj, "model_family"),
        model_size=require_str(obj, "model_size"),
        tokenizer_id=require_str(obj, "tokenizer_id"),
        corpus_path=require_str(obj, "corpus_path"),
        optimizer=require_str(obj, "optimizer"),
        device=require_str(obj, "device"),
        precision=require_str(obj, "precision"),
        epochs=require_int(obj, "epochs"),
        batch_size=require_int(obj, "batch_size"),
        max_seq_len=require_int(obj, "max_seq_len"),
        steps=require_int(obj, "steps"),
        seed=require_int(obj, "seed"),
        early_stopping_patience=require_int(obj, "early_stopping_patience"),
        loss=require_float(obj, "loss"),
        learning_rate=require_float(obj, "learning_rate"),
        holdout_fraction=require_float(obj, "holdout_fraction"),
        gradient_clipping=require_float(obj, "gradient_clipping"),
        test_split_ratio=require_float(obj, "test_split_ratio"),
        finetune_lr_cap=require_float(obj, "finetune_lr_cap"),
        freeze_embed=require_bool(obj, "freeze_embed"),
        early_stopped=require_bool(obj, "early_stopped"),
        git_commit=_optional_str(obj, "git_commit"),
        pretrained_run_id=_optional_str(obj, "pretrained_run_id"),
        test_loss=_optional_float(obj, "test_loss"),
        test_perplexity=_optional_float(obj, "test_perplexity"),
        best_val_loss=_optional_float(obj, "best_val_loss"),
    )


def load_manifest_from_text(text: str) -> TrainingManifest:
    """Parse manifest JSON text into typed TrainingManifest.

    Raises:
        JSONTypeError: if the manifest is not a well-formed JSON object.
    """
    obj = narrow_json_to_dict(load_json_str(text))
    versions = _decode_manifest_versions(obj)
    system = _decode_manifest_system(obj)
    fields = _decode_manifest_fields(obj)

    return {
        "run_id": fields.run_id,
        "model_family": fields.model_family,
        "model_size": fields.model_size,
        "epochs": fields.epochs,
        "batch_size": fields.batch_size,
        "max_seq_len": fields.max_seq_len,
        "steps": fields.steps,
        "loss": fields.loss,
        "learning_rate": fields.learning_rate,
        "tokenizer_id": fields.tokenizer_id,
        "corpus_path": fields.corpus_path,
        "holdout_fraction": fields.holdout_fraction,
        "optimizer": fields.optimizer,
        "freeze_embed": fields.freeze_embed,
        "gradient_clipping": fields.gradient_clipping,
        "seed": fields.seed,
        "pretrained_run_id": fields.pretrained_run_id,
        "versions": versions,
        "system": system,
        "git_commit": fields.git_commit,
        "device": fields.device,
        "precision": fields.precision,
        "early_stopping_patience": fields.early_stopping_patience,
        "test_split_ratio": fields.test_split_ratio,
        "finetune_lr_cap": fields.finetune_lr_cap,
        "test_loss": fields.test_loss,
        "test_perplexity": fields.test_perplexity,
        "best_val_loss": fields.best_val_loss,
        "early_stopped": fields.early_stopped,
    }
