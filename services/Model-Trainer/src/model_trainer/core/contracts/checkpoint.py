"""Typed contract for training checkpoints.

A checkpoint records everything needed to continue an interrupted training
run from its last completed epoch: progress counters, early-stopping state,
accumulated timing, per-epoch summaries, and the full ``ModelTrainConfig``
the run was started with. The config is the checkpoint's fingerprint: a
resume is only valid when the resubmitted config is identical field for
field, and the decoder-side helpers here are what make that comparison
explicit and reportable.

Tensor state (model weights, optimizer, RNG) travels in the torch payload
beside this metadata and never passes through JSON; this module owns only
the JSON-encodable half.

The encode/decode pair follows the queue_encoding pattern: encode produces
a ``JSONObject``, decode validates every field with ``require_*`` helpers
and raises ``JSONTypeError`` with a named field on any violation.
"""

from __future__ import annotations

from typing import Final, Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_float,
    optional_str,
    require_bool,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from .model import ModelTrainConfig
from .queue_encoding import (
    _decode_optional_gguf_export,
    _decode_optional_lora,
    _decode_optional_quantization,
    _narrow_finetuning_strategy,
    _narrow_model_family,
    _narrow_optimizer,
    encode_gguf_export_config,
    encode_lora_config,
    encode_quantization_config,
)

#: Version stamp written into every checkpoint. A decoder that meets a
#: different version refuses to resume rather than guessing at field
#: semantics; bump this when the meta shape or tensor payload changes.
CHECKPOINT_SCHEMA_VERSION: Final[int] = 1


class EpochSummaryRecord(TypedDict):
    """One completed epoch's metrics, as tracked for the wandb epoch table.

    Attributes:
        epoch: Zero-based epoch index.
        train_loss: Final training loss of the epoch.
        train_ppl: Training perplexity derived from ``train_loss``.
        val_loss: Validation loss measured after the epoch.
        val_ppl: Validation perplexity derived from ``val_loss``.
    """

    epoch: int
    train_loss: float
    train_ppl: float
    val_loss: float
    val_ppl: float


class TrainingCheckpointMeta(TypedDict):
    """JSON-encodable half of a training checkpoint.

    Attributes:
        schema_version: Must equal ``CHECKPOINT_SCHEMA_VERSION`` on load.
        run_id: Run the checkpoint belongs to; a resume for a different
            run id must refuse the file.
        epochs_completed: Fully completed epochs; the resumed loop starts
            at this epoch index.
        global_step: Optimizer steps taken across all completed epochs.
        last_loss: Training loss at the end of the last completed epoch.
        best_val_loss: Best validation loss seen so far, or None when no
            validation pass has run (the in-memory sentinel is +inf,
            which JSON cannot carry).
        epochs_no_improve: Early-stopping counter at the boundary.
        best_saved: Whether a best-validation model has already been
            written to the run's output directory.
        total_samples_processed: Samples consumed across completed epochs.
        total_tokens_processed: Tokens consumed across completed epochs.
        elapsed_seconds: Wall-clock training time consumed by every prior
            execution of this run; the final manifest duration is this
            plus the resumed execution's own time.
        started_at_iso: ISO 8601 timestamp of the run's original start.
        epoch_summaries: Per-epoch metric records for completed epochs
            that ran a validation pass.
        config: The full training config the run started with; the
            resume fingerprint.
    """

    schema_version: int
    run_id: str
    epochs_completed: int
    global_step: int
    last_loss: float
    best_val_loss: float | None
    epochs_no_improve: int
    best_saved: bool
    total_samples_processed: int
    total_tokens_processed: int
    elapsed_seconds: float
    started_at_iso: str
    epoch_summaries: list[EpochSummaryRecord]
    config: ModelTrainConfig


def encode_epoch_summary(record: EpochSummaryRecord) -> JSONObject:
    """Encode an EpochSummaryRecord to a JSONObject.

    Args:
        record: Epoch summary to encode.

    Returns:
        JSON-serializable dictionary with all summary fields.
    """
    return {
        "epoch": record["epoch"],
        "train_loss": record["train_loss"],
        "train_ppl": record["train_ppl"],
        "val_loss": record["val_loss"],
        "val_ppl": record["val_ppl"],
    }


def decode_epoch_summary(obj: JSONObject) -> EpochSummaryRecord:
    """Decode a JSONObject to an EpochSummaryRecord with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated EpochSummaryRecord.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    return {
        "epoch": require_int(obj, "epoch"),
        "train_loss": require_float(obj, "train_loss"),
        "train_ppl": require_float(obj, "train_ppl"),
        "val_loss": require_float(obj, "val_loss"),
        "val_ppl": require_float(obj, "val_ppl"),
    }


def _narrow_resolved_device(raw: str) -> Literal["cpu", "cuda"]:
    """Narrow a resolved device string to its Literal type.

    Args:
        raw: Raw device string.

    Returns:
        The narrowed value, one of ``cpu`` or ``cuda``.

    Raises:
        JSONTypeError: If the value is not a resolved device. ``auto`` is
            a request-time value and never appears in a resolved config.
    """
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda"
    raise JSONTypeError(f"Field 'device' must be 'cpu' or 'cuda', got '{raw}'")


def _narrow_resolved_precision(raw: str) -> Literal["fp32", "fp16", "bf16"]:
    """Narrow a resolved precision string to its Literal type.

    Args:
        raw: Raw precision string.

    Returns:
        The narrowed value, one of ``fp32``, ``fp16`` or ``bf16``.

    Raises:
        JSONTypeError: If the value is not a resolved precision. ``auto``
            is a request-time value and never appears in a resolved
            config.
    """
    if raw == "fp32":
        return "fp32"
    if raw == "fp16":
        return "fp16"
    if raw == "bf16":
        return "bf16"
    raise JSONTypeError(f"Field 'precision' must be 'fp32', 'fp16', or 'bf16', got '{raw}'")


def encode_model_train_config(cfg: ModelTrainConfig) -> JSONObject:
    """Encode a ModelTrainConfig to a JSONObject.

    This is the canonical form used both for checkpoint persistence and
    for the resume fingerprint comparison; two configs are equal exactly
    when their encoded forms are equal.

    Args:
        cfg: Training config to encode.

    Returns:
        JSON-serializable dictionary with every config field.
    """
    lora_encoded: JSONValue = encode_lora_config(cfg["lora"]) if cfg["lora"] is not None else None
    quantization_encoded: JSONValue = (
        encode_quantization_config(cfg["quantization"]) if cfg["quantization"] is not None else None
    )
    gguf_export_encoded: JSONValue = (
        encode_gguf_export_config(cfg["gguf_export"]) if cfg["gguf_export"] is not None else None
    )
    return {
        "model_family": cfg["model_family"],
        "model_size": cfg["model_size"],
        "max_seq_len": cfg["max_seq_len"],
        "num_epochs": cfg["num_epochs"],
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["learning_rate"],
        "tokenizer_id": cfg["tokenizer_id"],
        "corpus_path": cfg["corpus_path"],
        "holdout_fraction": cfg["holdout_fraction"],
        "seed": cfg["seed"],
        "pretrained_run_id": cfg["pretrained_run_id"],
        "freeze_embed": cfg["freeze_embed"],
        "gradient_clipping": cfg["gradient_clipping"],
        "optimizer": cfg["optimizer"],
        "device": cfg["device"],
        "precision": cfg["precision"],
        "data_num_workers": cfg["data_num_workers"],
        "data_pin_memory": cfg["data_pin_memory"],
        "early_stopping_patience": cfg["early_stopping_patience"],
        "test_split_ratio": cfg["test_split_ratio"],
        "finetune_lr_cap": cfg["finetune_lr_cap"],
        "loss_mask_prefix_separator": cfg["loss_mask_prefix_separator"],
        "finetuning_strategy": cfg["finetuning_strategy"],
        "hub_model_id": cfg["hub_model_id"],
        "lora": lora_encoded,
        "quantization": quantization_encoded,
        "gguf_export": gguf_export_encoded,
    }


def decode_model_train_config(obj: JSONObject) -> ModelTrainConfig:
    """Decode a JSONObject to a ModelTrainConfig with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated ModelTrainConfig.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types,
            or if ``device``/``precision`` carry request-time ``auto``
            values that have no place in a resolved config.
    """
    return {
        "model_family": _narrow_model_family(require_str(obj, "model_family")),
        "model_size": require_str(obj, "model_size"),
        "max_seq_len": require_int(obj, "max_seq_len"),
        "num_epochs": require_int(obj, "num_epochs"),
        "batch_size": require_int(obj, "batch_size"),
        "learning_rate": require_float(obj, "learning_rate"),
        "tokenizer_id": optional_str(obj, "tokenizer_id"),
        "corpus_path": require_str(obj, "corpus_path"),
        "holdout_fraction": require_float(obj, "holdout_fraction"),
        "seed": require_int(obj, "seed"),
        "pretrained_run_id": optional_str(obj, "pretrained_run_id"),
        "freeze_embed": require_bool(obj, "freeze_embed"),
        "gradient_clipping": require_float(obj, "gradient_clipping"),
        "optimizer": _narrow_optimizer(require_str(obj, "optimizer")),
        "device": _narrow_resolved_device(require_str(obj, "device")),
        "precision": _narrow_resolved_precision(require_str(obj, "precision")),
        "data_num_workers": require_int(obj, "data_num_workers"),
        "data_pin_memory": require_bool(obj, "data_pin_memory"),
        "early_stopping_patience": require_int(obj, "early_stopping_patience"),
        "test_split_ratio": require_float(obj, "test_split_ratio"),
        "finetune_lr_cap": require_float(obj, "finetune_lr_cap"),
        "loss_mask_prefix_separator": optional_str(obj, "loss_mask_prefix_separator"),
        "finetuning_strategy": _narrow_finetuning_strategy(require_str(obj, "finetuning_strategy")),
        "hub_model_id": optional_str(obj, "hub_model_id"),
        "lora": _decode_optional_lora(obj),
        "quantization": _decode_optional_quantization(obj),
        "gguf_export": _decode_optional_gguf_export(obj),
    }


def model_train_config_mismatches(
    expected: ModelTrainConfig, actual: ModelTrainConfig
) -> list[str]:
    """Report the config fields on which two configs disagree.

    Comparison happens on the canonical encoded forms, so nested configs
    (lora, quantization, gguf_export) compare structurally and tuple
    versus list representation differences cannot produce false
    mismatches.

    Args:
        expected: The config recorded in the checkpoint.
        actual: The config the resume was submitted with.

    Returns:
        Sorted field names whose values differ; empty when the configs
        are identical.
    """
    expected_encoded = encode_model_train_config(expected)
    actual_encoded = encode_model_train_config(actual)
    return sorted(key for key in expected_encoded if expected_encoded[key] != actual_encoded[key])


def encode_training_checkpoint_meta(meta: TrainingCheckpointMeta) -> JSONObject:
    """Encode a TrainingCheckpointMeta to a JSONObject.

    Args:
        meta: Checkpoint metadata to encode.

    Returns:
        JSON-serializable dictionary with every metadata field.
    """
    summaries: list[JSONValue] = [encode_epoch_summary(s) for s in meta["epoch_summaries"]]
    return {
        "schema_version": meta["schema_version"],
        "run_id": meta["run_id"],
        "epochs_completed": meta["epochs_completed"],
        "global_step": meta["global_step"],
        "last_loss": meta["last_loss"],
        "best_val_loss": meta["best_val_loss"],
        "epochs_no_improve": meta["epochs_no_improve"],
        "best_saved": meta["best_saved"],
        "total_samples_processed": meta["total_samples_processed"],
        "total_tokens_processed": meta["total_tokens_processed"],
        "elapsed_seconds": meta["elapsed_seconds"],
        "started_at_iso": meta["started_at_iso"],
        "epoch_summaries": summaries,
        "config": encode_model_train_config(meta["config"]),
    }


def decode_training_checkpoint_meta(obj: JSONObject) -> TrainingCheckpointMeta:
    """Decode a JSONObject to a TrainingCheckpointMeta with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TrainingCheckpointMeta.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    summaries_raw = require_list(obj, "epoch_summaries")
    summaries: list[EpochSummaryRecord] = []
    for index, item in enumerate(summaries_raw):
        if not isinstance(item, dict):
            raise JSONTypeError(
                f"Field 'epoch_summaries[{index}]' must be an object, got {type(item).__name__}"
            )
        summaries.append(decode_epoch_summary(item))

    return {
        "schema_version": require_int(obj, "schema_version"),
        "run_id": require_str(obj, "run_id"),
        "epochs_completed": require_int(obj, "epochs_completed"),
        "global_step": require_int(obj, "global_step"),
        "last_loss": require_float(obj, "last_loss"),
        "best_val_loss": optional_float(obj, "best_val_loss"),
        "epochs_no_improve": require_int(obj, "epochs_no_improve"),
        "best_saved": require_bool(obj, "best_saved"),
        "total_samples_processed": require_int(obj, "total_samples_processed"),
        "total_tokens_processed": require_int(obj, "total_tokens_processed"),
        "elapsed_seconds": require_float(obj, "elapsed_seconds"),
        "started_at_iso": require_str(obj, "started_at_iso"),
        "epoch_summaries": summaries,
        "config": decode_model_train_config(require_dict(obj, "config")),
    }


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "EpochSummaryRecord",
    "TrainingCheckpointMeta",
    "decode_epoch_summary",
    "decode_model_train_config",
    "decode_training_checkpoint_meta",
    "encode_epoch_summary",
    "encode_model_train_config",
    "encode_training_checkpoint_meta",
    "model_train_config_mismatches",
]
