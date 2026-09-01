"""Manifest parsing for training worker."""

from __future__ import annotations

from typing import Literal

from platform_core.comparability import (
    RunFingerprint,
    decode_run_fingerprint,
)
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

from model_trainer.core.contracts.dataset import as_corpus_format
from model_trainer.infra.persistence.models import (
    GgufExportManifest,
    TrainingManifest,
    TrainingManifestModelInfo,
    TrainingManifestPerformance,
    TrainingManifestSystem,
    TrainingManifestTiming,
    TrainingManifestVersions,
)


def as_model_family(s: str) -> Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"]:
    """Convert string to model family literal type."""
    if s == "gpt2":
        return "gpt2"
    if s == "llama":
        return "llama"
    if s == "qwen":
        return "qwen"
    if s == "char_lstm":
        return "char_lstm"
    if s == "hf_lm":
        return "hf_lm"
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


def _decode_manifest_timing(obj: JSONObject) -> TrainingManifestTiming:
    """Decode timing section from manifest JSON.

    Args:
        obj: JSON object containing a 'timing' key.

    Returns:
        TrainingManifestTiming with decoded values.

    Raises:
        JSONTypeError: If required fields are missing or have wrong type.
    """
    timing = require_dict(obj, "timing")
    return {
        "training_duration_sec": require_float(timing, "training_duration_sec"),
        "started_at": require_str(timing, "started_at"),
        "completed_at": require_str(timing, "completed_at"),
    }


def _decode_manifest_performance(obj: JSONObject) -> TrainingManifestPerformance:
    """Decode performance section from manifest JSON.

    Args:
        obj: JSON object containing a 'performance' key.

    Returns:
        TrainingManifestPerformance with decoded values.

    Raises:
        JSONTypeError: If required fields are missing or have wrong type.
    """
    perf = require_dict(obj, "performance")
    # peak_gpu_memory_mb is optional (None if CPU training)
    gpu_mem_val = perf.get("peak_gpu_memory_mb")
    peak_gpu_memory_mb: float | None = None
    if gpu_mem_val is not None:
        if isinstance(gpu_mem_val, bool):
            type_name = type(gpu_mem_val).__name__
            raise JSONTypeError(
                f"Field 'peak_gpu_memory_mb' must be a number or null, got {type_name}"
            )
        if isinstance(gpu_mem_val, int | float):
            peak_gpu_memory_mb = float(gpu_mem_val)
        else:
            type_name = type(gpu_mem_val).__name__
            raise JSONTypeError(
                f"Field 'peak_gpu_memory_mb' must be a number or null, got {type_name}"
            )
    return {
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "avg_samples_per_sec": require_float(perf, "avg_samples_per_sec"),
        "total_tokens_processed": require_int(perf, "total_tokens_processed"),
    }


def _decode_manifest_model_info(obj: JSONObject) -> TrainingManifestModelInfo:
    """Decode model_info section from manifest JSON.

    Args:
        obj: JSON object containing a 'model_info' key.

    Returns:
        TrainingManifestModelInfo with decoded values.

    Raises:
        JSONTypeError: If required fields are missing or have wrong type.
    """
    info = require_dict(obj, "model_info")
    return {
        "param_count": require_int(info, "param_count"),
        "model_size_mb": require_float(info, "model_size_mb"),
        "vocab_size": require_int(info, "vocab_size"),
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


def _optional_fingerprint(obj: JSONObject, key: str) -> RunFingerprint | None:
    """Extract an optional run fingerprint.

    Absent means the run predates the field, and None states that. Present
    means it is decoded STRICTLY, by the same decoder the scoring path uses:
    a fingerprint missing an axis would compare equal to another fingerprint
    missing the same axis, reporting two differently-configured runs as
    identical. That is the one failure a comparability record must not have,
    so a partial one is refused rather than repaired.

    Args:
        obj: The manifest object.
        key: The field name.

    Returns:
        The fingerprint, or None when the field is absent or explicitly null.

    Raises:
        JSONTypeError: When the value is present but is not a valid
            fingerprint.
    """
    val = obj.get(key)
    if val is None:
        return None
    return decode_run_fingerprint(val)


def _optional_int(obj: JSONObject, key: str) -> int | None:
    """Extract optional integer field."""
    val = obj.get(key)
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, int):
        raise JSONTypeError(f"Field '{key}' must be an integer or null, got {type(val).__name__}")
    return val


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
    tokenizer_id: str | None  # None for hf_lm (uses HF tokenizer from hub_model_id)
    corpus_path: str
    corpus_format: str
    optimizer: str
    freeze_embed: bool
    gradient_clipping: float
    seed: int
    fingerprint: RunFingerprint | None
    git_commit: str | None
    pretrained_run_id: str | None
    device: str
    precision: str
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float
    # None on runs trained before masking existed, and on every run that did
    # not ask for it. Recorded so an arm's masking setting is auditable from
    # its artifacts rather than only from the script that submitted it.
    loss_mask_prefix_separator: str | None
    test_loss: float | None
    test_perplexity: float | None
    best_val_loss: float | None
    early_stopped: bool
    # None on runs trained before checkpointing existed, and on every run
    # that trained start to finish in one execution.
    resumed_from_epoch: int | None

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
        tokenizer_id: str | None,
        corpus_path: str,
        corpus_format: str,
        optimizer: str,
        freeze_embed: bool,
        gradient_clipping: float,
        seed: int,
        fingerprint: RunFingerprint | None,
        git_commit: str | None,
        pretrained_run_id: str | None,
        device: str,
        precision: str,
        early_stopping_patience: int,
        test_split_ratio: float,
        finetune_lr_cap: float,
        loss_mask_prefix_separator: str | None,
        test_loss: float | None,
        test_perplexity: float | None,
        best_val_loss: float | None,
        early_stopped: bool,
        resumed_from_epoch: int | None,
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
        self.corpus_format = corpus_format
        self.optimizer = optimizer
        self.freeze_embed = freeze_embed
        self.gradient_clipping = gradient_clipping
        self.seed = seed
        self.fingerprint = fingerprint
        self.git_commit = git_commit
        self.pretrained_run_id = pretrained_run_id
        self.device = device
        self.precision = precision
        self.early_stopping_patience = early_stopping_patience
        self.test_split_ratio = test_split_ratio
        self.finetune_lr_cap = finetune_lr_cap
        self.loss_mask_prefix_separator = loss_mask_prefix_separator
        self.test_loss = test_loss
        self.test_perplexity = test_perplexity
        self.best_val_loss = best_val_loss
        self.early_stopped = early_stopped
        self.resumed_from_epoch = resumed_from_epoch


def _decode_manifest_fields(obj: JSONObject) -> _ManifestFields:
    return _ManifestFields(
        run_id=require_str(obj, "run_id"),
        model_family=require_str(obj, "model_family"),
        model_size=require_str(obj, "model_size"),
        tokenizer_id=_optional_str(obj, "tokenizer_id"),
        corpus_path=require_str(obj, "corpus_path"),
        corpus_format=as_corpus_format(require_str(obj, "corpus_format"), "corpus_format"),
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
        # Absent in manifests written before the field existed; absence means
        # the posture was not recorded, which None states exactly. Refusing
        # here would break loading a model trained before 2026-08-25, and
        # loading is not comparing.
        fingerprint=_optional_fingerprint(obj, "fingerprint"),
        git_commit=_optional_str(obj, "git_commit"),
        pretrained_run_id=_optional_str(obj, "pretrained_run_id"),
        loss_mask_prefix_separator=_optional_str(obj, "loss_mask_prefix_separator"),
        test_loss=_optional_float(obj, "test_loss"),
        test_perplexity=_optional_float(obj, "test_perplexity"),
        best_val_loss=_optional_float(obj, "best_val_loss"),
        resumed_from_epoch=_optional_int(obj, "resumed_from_epoch"),
    )


def _decode_optional_gguf_export_manifest(obj: JSONObject) -> GgufExportManifest | None:
    """Decode optional gguf_export section from manifest JSON.

    Args:
        obj: The root manifest JSON object.

    Returns:
        Decoded GgufExportManifest or None if field is missing/null.

    Raises:
        JSONTypeError: If the field exists but has invalid structure.
    """
    raw = obj.get("gguf_export")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise JSONTypeError(
            f"Field 'gguf_export' must be an object or null, got {type(raw).__name__}"
        )
    return {
        "output_type": require_str(raw, "output_type"),
        "output_filename": require_str(raw, "output_filename"),
        "output_size_bytes": require_int(raw, "output_size_bytes"),
    }


def load_manifest_from_text(text: str) -> TrainingManifest:
    """Parse manifest JSON text into typed TrainingManifest.

    Args:
        text: JSON string containing manifest data.

    Returns:
        Fully decoded and validated TrainingManifest.

    Raises:
        JSONTypeError: If the manifest is not a well-formed JSON object
            or required fields are missing/invalid.
    """
    obj = narrow_json_to_dict(load_json_str(text))
    versions = _decode_manifest_versions(obj)
    system = _decode_manifest_system(obj)
    timing = _decode_manifest_timing(obj)
    performance = _decode_manifest_performance(obj)
    model_info = _decode_manifest_model_info(obj)
    fields = _decode_manifest_fields(obj)
    gguf_export = _decode_optional_gguf_export_manifest(obj)

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
        "corpus_format": fields.corpus_format,
        "holdout_fraction": fields.holdout_fraction,
        "optimizer": fields.optimizer,
        "freeze_embed": fields.freeze_embed,
        "gradient_clipping": fields.gradient_clipping,
        "seed": fields.seed,
        "pretrained_run_id": fields.pretrained_run_id,
        "versions": versions,
        "system": system,
        "fingerprint": fields.fingerprint,
        "git_commit": fields.git_commit,
        "device": fields.device,
        "precision": fields.precision,
        "early_stopping_patience": fields.early_stopping_patience,
        "test_split_ratio": fields.test_split_ratio,
        "finetune_lr_cap": fields.finetune_lr_cap,
        "loss_mask_prefix_separator": fields.loss_mask_prefix_separator,
        "test_loss": fields.test_loss,
        "test_perplexity": fields.test_perplexity,
        "best_val_loss": fields.best_val_loss,
        "early_stopped": fields.early_stopped,
        "resumed_from_epoch": fields.resumed_from_epoch,
        "timing": timing,
        "performance": performance,
        "model_info": model_info,
        "gguf_export": gguf_export,
    }
