"""Manifest parsing for training worker."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONValue, load_json_str

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
    raise ValueError("invalid model_family in manifest")


def as_optimizer(s: str) -> Literal["adamw", "adam", "sgd"]:
    """Convert string to optimizer literal type."""
    if s == "adamw":
        return "adamw"
    if s == "adam":
        return "adam"
    if s == "sgd":
        return "sgd"
    raise ValueError("invalid optimizer in manifest")


def as_device(s: str) -> Literal["cpu", "cuda"]:
    """Convert string to device literal type."""
    if s == "cpu":
        return "cpu"
    if s == "cuda":
        return "cuda"
    raise ValueError("invalid device in manifest")


def as_precision(s: str) -> Literal["fp32", "fp16", "bf16"]:
    """Convert string to precision literal type."""
    if s == "fp32":
        return "fp32"
    if s == "fp16":
        return "fp16"
    if s == "bf16":
        return "bf16"
    raise ValueError("invalid precision in manifest")


def _decode_manifest_versions(vers_o: JSONValue) -> TrainingManifestVersions:
    if not isinstance(vers_o, dict):
        raise ValueError("invalid manifest JSON: versions")
    v_torch: JSONValue = vers_o.get("torch")
    v_transformers: JSONValue = vers_o.get("transformers")
    v_tokenizers: JSONValue = vers_o.get("tokenizers")
    v_datasets: JSONValue = vers_o.get("datasets")
    if not isinstance(v_torch, str):
        raise ValueError("manifest field versions.torch must be str")
    if not isinstance(v_transformers, str):
        raise ValueError("manifest field versions.transformers must be str")
    if not isinstance(v_tokenizers, str):
        raise ValueError("manifest field versions.tokenizers must be str")
    if not isinstance(v_datasets, str):
        raise ValueError("manifest field versions.datasets must be str")
    return {
        "torch": v_torch,
        "transformers": v_transformers,
        "tokenizers": v_tokenizers,
        "datasets": v_datasets,
    }


def _decode_manifest_system(sys_o: JSONValue) -> TrainingManifestSystem:
    if not isinstance(sys_o, dict):
        raise ValueError("invalid manifest JSON: system")
    s_cpu: JSONValue = sys_o.get("cpu_count")
    s_platform: JSONValue = sys_o.get("platform")
    s_release: JSONValue = sys_o.get("platform_release")
    s_machine: JSONValue = sys_o.get("machine")
    if not isinstance(s_cpu, int):
        raise ValueError("manifest field system.cpu_count must be int")
    if not isinstance(s_platform, str):
        raise ValueError("manifest field system.platform must be str")
    if not isinstance(s_release, str):
        raise ValueError("manifest field system.platform_release must be str")
    if not isinstance(s_machine, str):
        raise ValueError("manifest field system.machine must be str")
    return {
        "cpu_count": s_cpu,
        "platform": s_platform,
        "platform_release": s_release,
        "machine": s_machine,
    }


def _decode_manifest_str(obj: dict[str, JSONValue], field: str) -> str:
    val: JSONValue = obj.get(field)
    if not isinstance(val, str):
        raise ValueError(f"manifest field {field} must be str")
    return val


def _decode_manifest_int(obj: dict[str, JSONValue], field: str) -> int:
    val: JSONValue = obj.get(field)
    if not isinstance(val, int):
        raise ValueError(f"manifest field {field} must be int")
    return val


def _decode_manifest_float(obj: dict[str, JSONValue], field: str) -> float:
    val: JSONValue = obj.get(field)
    if not isinstance(val, int | float):
        raise ValueError(f"manifest field {field} must be number")
    return float(val)


def _decode_manifest_str_or_none(obj: dict[str, JSONValue], field: str) -> str | None:
    val: JSONValue = obj.get(field)
    if val is None:
        return None
    if not isinstance(val, str):
        raise ValueError(f"manifest field {field} must be str or null")
    return val


def _decode_manifest_bool(obj: dict[str, JSONValue], field: str) -> bool:
    val: JSONValue = obj.get(field)
    if not isinstance(val, bool):
        raise ValueError(f"manifest field {field} must be bool")
    return val


def _decode_manifest_float_or_none(obj: dict[str, JSONValue], field: str) -> float | None:
    val: JSONValue = obj.get(field)
    if val is None:
        return None
    if not isinstance(val, int | float):
        raise ValueError(f"manifest field {field} must be number or null")
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


def _decode_manifest_fields(obj: dict[str, JSONValue]) -> _ManifestFields:
    run_id = _decode_manifest_str(obj, "run_id")
    model_family = _decode_manifest_str(obj, "model_family")
    model_size = _decode_manifest_str(obj, "model_size")
    tokenizer_id = _decode_manifest_str(obj, "tokenizer_id")
    corpus_path = _decode_manifest_str(obj, "corpus_path")
    optimizer = _decode_manifest_str(obj, "optimizer")
    device = _decode_manifest_str(obj, "device")
    precision = _decode_manifest_str(obj, "precision")

    epochs = _decode_manifest_int(obj, "epochs")
    batch_size = _decode_manifest_int(obj, "batch_size")
    max_seq_len = _decode_manifest_int(obj, "max_seq_len")
    steps = _decode_manifest_int(obj, "steps")
    seed = _decode_manifest_int(obj, "seed")
    early_stopping_patience = _decode_manifest_int(obj, "early_stopping_patience")

    loss = _decode_manifest_float(obj, "loss")
    learning_rate = _decode_manifest_float(obj, "learning_rate")
    holdout_fraction = _decode_manifest_float(obj, "holdout_fraction")
    gradient_clipping = _decode_manifest_float(obj, "gradient_clipping")
    test_split_ratio = _decode_manifest_float(obj, "test_split_ratio")
    finetune_lr_cap = _decode_manifest_float(obj, "finetune_lr_cap")

    freeze_embed = _decode_manifest_bool(obj, "freeze_embed")
    early_stopped = _decode_manifest_bool(obj, "early_stopped")

    git_v: JSONValue = obj.get("git_commit")
    git_commit = git_v if isinstance(git_v, str) else None

    pretrained_run_id = _decode_manifest_str_or_none(obj, "pretrained_run_id")

    test_loss = _decode_manifest_float_or_none(obj, "test_loss")
    test_perplexity = _decode_manifest_float_or_none(obj, "test_perplexity")
    best_val_loss = _decode_manifest_float_or_none(obj, "best_val_loss")

    return _ManifestFields(
        run_id=run_id,
        model_family=model_family,
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        steps=steps,
        loss=loss,
        learning_rate=learning_rate,
        holdout_fraction=holdout_fraction,
        tokenizer_id=tokenizer_id,
        corpus_path=corpus_path,
        optimizer=optimizer,
        freeze_embed=freeze_embed,
        gradient_clipping=gradient_clipping,
        seed=seed,
        git_commit=git_commit,
        pretrained_run_id=pretrained_run_id,
        device=device,
        precision=precision,
        early_stopping_patience=early_stopping_patience,
        test_split_ratio=test_split_ratio,
        finetune_lr_cap=finetune_lr_cap,
        test_loss=test_loss,
        test_perplexity=test_perplexity,
        best_val_loss=best_val_loss,
        early_stopped=early_stopped,
    )


def load_manifest_from_text(text: str) -> TrainingManifest:
    """Parse manifest JSON text into typed TrainingManifest."""
    obj_raw = load_json_str(text)
    if not isinstance(obj_raw, dict):
        raise ValueError("invalid manifest JSON")
    obj: dict[str, JSONValue] = obj_raw

    versions = _decode_manifest_versions(obj.get("versions"))
    system = _decode_manifest_system(obj.get("system"))
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
