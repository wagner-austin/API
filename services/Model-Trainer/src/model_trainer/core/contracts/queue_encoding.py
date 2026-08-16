"""Encoding and decoding functions for queue payloads.

This module provides type-safe serialization and deserialization of
TrainRequestPayload and its nested TypedDicts (LoraConfig, QuantizationConfig,
UnslothConfig) for RQ job queue communication.

All encode functions convert TypedDicts to JSONObject (dict[str, JSONValue]).
All decode functions validate and convert JSONObject back to TypedDicts using
require_* helpers from platform_core.json_utils.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_int,
    optional_str,
    require_bool,
    require_float,
    require_int,
    require_str,
)

from .model import GgufExportConfig, LoraConfig, QuantizationConfig, UnslothConfig
from .queue import ClozeJobPayload, TrainJobPayload, TrainRequestPayload


def encode_lora_config(config: LoraConfig) -> JSONObject:
    """Encode LoraConfig TypedDict to JSONObject for serialization.

    Args:
        config: LoRA configuration to encode.

    Returns:
        JSON-serializable dictionary with all LoRA fields.
    """
    return {
        "enabled": config["enabled"],
        "r": config["r"],
        "lora_alpha": config["lora_alpha"],
        "lora_dropout": config["lora_dropout"],
        "target_modules": list(config["target_modules"]),
        "bias": config["bias"],
    }


def decode_lora_config(obj: JSONObject) -> LoraConfig:
    """Decode JSONObject to LoraConfig with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated LoraConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    enabled = require_bool(obj, "enabled")
    r = require_int(obj, "r")
    lora_alpha = require_int(obj, "lora_alpha")
    lora_dropout = require_float(obj, "lora_dropout")

    target_modules_raw = obj.get("target_modules")
    if target_modules_raw is None:
        raise JSONTypeError("Missing required field 'target_modules'")
    if not isinstance(target_modules_raw, list):
        raise JSONTypeError(
            f"Field 'target_modules' must be an array, got {type(target_modules_raw).__name__}"
        )
    target_modules: list[str] = []
    for i, item in enumerate(target_modules_raw):
        if not isinstance(item, str):
            raise JSONTypeError(
                f"Field 'target_modules[{i}]' must be a string, got {type(item).__name__}"
            )
        target_modules.append(item)

    bias_raw = require_str(obj, "bias")
    if bias_raw not in ("none", "all", "lora_only"):
        raise JSONTypeError(f"Field 'bias' must be 'none', 'all', or 'lora_only', got '{bias_raw}'")
    bias: Literal["none", "all", "lora_only"]
    if bias_raw == "all":
        bias = "all"
    elif bias_raw == "lora_only":
        bias = "lora_only"
    else:
        bias = "none"

    return {
        "enabled": enabled,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": tuple(target_modules),
        "bias": bias,
    }


def encode_quantization_config(config: QuantizationConfig) -> JSONObject:
    """Encode QuantizationConfig TypedDict to JSONObject for serialization.

    Args:
        config: Quantization configuration to encode.

    Returns:
        JSON-serializable dictionary with all quantization fields.
    """
    return {
        "load_in_4bit": config["load_in_4bit"],
        "load_in_8bit": config["load_in_8bit"],
        "bnb_4bit_compute_dtype": config["bnb_4bit_compute_dtype"],
        "bnb_4bit_quant_type": config["bnb_4bit_quant_type"],
    }


def decode_quantization_config(obj: JSONObject) -> QuantizationConfig:
    """Decode JSONObject to QuantizationConfig with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated QuantizationConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    load_in_4bit = require_bool(obj, "load_in_4bit")
    load_in_8bit = require_bool(obj, "load_in_8bit")

    compute_dtype_raw = require_str(obj, "bnb_4bit_compute_dtype")
    if compute_dtype_raw not in ("float16", "bfloat16", "float32"):
        raise JSONTypeError(
            f"Field 'bnb_4bit_compute_dtype' must be 'float16', 'bfloat16', or 'float32', "
            f"got '{compute_dtype_raw}'"
        )
    compute_dtype: Literal["float16", "bfloat16", "float32"]
    if compute_dtype_raw == "bfloat16":
        compute_dtype = "bfloat16"
    elif compute_dtype_raw == "float32":
        compute_dtype = "float32"
    else:
        compute_dtype = "float16"

    quant_type_raw = require_str(obj, "bnb_4bit_quant_type")
    if quant_type_raw not in ("nf4", "fp4"):
        raise JSONTypeError(
            f"Field 'bnb_4bit_quant_type' must be 'nf4' or 'fp4', got '{quant_type_raw}'"
        )
    quant_type: Literal["nf4", "fp4"] = "fp4" if quant_type_raw == "fp4" else "nf4"

    return {
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "bnb_4bit_compute_dtype": compute_dtype,
        "bnb_4bit_quant_type": quant_type,
    }


def encode_unsloth_config(config: UnslothConfig) -> JSONObject:
    """Encode UnslothConfig TypedDict to JSONObject for serialization.

    Args:
        config: Unsloth configuration to encode.

    Returns:
        JSON-serializable dictionary with all Unsloth fields.
    """
    return {
        "enabled": config["enabled"],
        "max_seq_length": config["max_seq_length"],
        "dtype": config["dtype"],
    }


def decode_unsloth_config(obj: JSONObject) -> UnslothConfig:
    """Decode JSONObject to UnslothConfig with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated UnslothConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    enabled = require_bool(obj, "enabled")
    max_seq_length = require_int(obj, "max_seq_length")

    dtype_raw = obj.get("dtype")
    dtype: Literal["float16", "bfloat16"] | None
    if dtype_raw is None:
        dtype = None
    elif not isinstance(dtype_raw, str):
        raise JSONTypeError(
            f"Field 'dtype' must be a string or null, got {type(dtype_raw).__name__}"
        )
    elif dtype_raw not in ("float16", "bfloat16"):
        raise JSONTypeError(
            f"Field 'dtype' must be 'float16', 'bfloat16', or null, got '{dtype_raw}'"
        )
    else:
        dtype = "bfloat16" if dtype_raw == "bfloat16" else "float16"

    return {
        "enabled": enabled,
        "max_seq_length": max_seq_length,
        "dtype": dtype,
    }


def encode_gguf_export_config(config: GgufExportConfig) -> JSONObject:
    """Encode GgufExportConfig TypedDict to JSONObject for serialization.

    Args:
        config: GGUF export configuration to encode.

    Returns:
        JSON-serializable dictionary with all GGUF export fields.
    """
    return {
        "enabled": config["enabled"],
        "output_type": config["output_type"],
    }


def _narrow_gguf_output_type(raw: str) -> Literal["f32", "f16", "bf16", "q8_0"]:
    """Narrow GGUF output type string to Literal type with validation.

    Args:
        raw: Raw output type string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid output type.
    """
    if raw == "f32":
        return "f32"
    if raw == "f16":
        return "f16"
    if raw == "bf16":
        return "bf16"
    if raw == "q8_0":
        return "q8_0"
    raise JSONTypeError(f"Field 'output_type' must be 'f32', 'f16', 'bf16', or 'q8_0', got '{raw}'")


def decode_gguf_export_config(obj: JSONObject) -> GgufExportConfig:
    """Decode JSONObject to GgufExportConfig with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated GgufExportConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    enabled = require_bool(obj, "enabled")
    output_type = _narrow_gguf_output_type(require_str(obj, "output_type"))

    return {
        "enabled": enabled,
        "output_type": output_type,
    }


def _decode_optional_lora(obj: JSONObject) -> LoraConfig | None:
    """Decode optional lora config from JSONObject.

    Args:
        obj: Parent JSON object containing optional 'lora' field.

    Returns:
        Decoded LoraConfig or None if field is null/missing.

    Raises:
        JSONTypeError: If field has wrong type.
    """
    lora_raw = obj.get("lora")
    if lora_raw is None:
        return None
    if isinstance(lora_raw, dict):
        return decode_lora_config(lora_raw)
    raise JSONTypeError(f"Field 'lora' must be an object or null, got {type(lora_raw).__name__}")


def _decode_optional_quantization(obj: JSONObject) -> QuantizationConfig | None:
    """Decode optional quantization config from JSONObject.

    Args:
        obj: Parent JSON object containing optional 'quantization' field.

    Returns:
        Decoded QuantizationConfig or None if field is null/missing.

    Raises:
        JSONTypeError: If field has wrong type.
    """
    quantization_raw = obj.get("quantization")
    if quantization_raw is None:
        return None
    if isinstance(quantization_raw, dict):
        return decode_quantization_config(quantization_raw)
    raise JSONTypeError(
        f"Field 'quantization' must be an object or null, got {type(quantization_raw).__name__}"
    )


def _decode_optional_unsloth(obj: JSONObject) -> UnslothConfig | None:
    """Decode optional unsloth config from JSONObject.

    Args:
        obj: Parent JSON object containing optional 'unsloth' field.

    Returns:
        Decoded UnslothConfig or None if field is null/missing.

    Raises:
        JSONTypeError: If field has wrong type.
    """
    unsloth_raw = obj.get("unsloth")
    if unsloth_raw is None:
        return None
    if isinstance(unsloth_raw, dict):
        return decode_unsloth_config(unsloth_raw)
    raise JSONTypeError(
        f"Field 'unsloth' must be an object or null, got {type(unsloth_raw).__name__}"
    )


def _decode_optional_gguf_export(obj: JSONObject) -> GgufExportConfig | None:
    """Decode optional gguf_export config from JSONObject.

    Args:
        obj: Parent JSON object containing optional 'gguf_export' field.

    Returns:
        Decoded GgufExportConfig or None if field is null/missing.

    Raises:
        JSONTypeError: If field has wrong type.
    """
    gguf_export_raw = obj.get("gguf_export")
    if gguf_export_raw is None:
        return None
    if isinstance(gguf_export_raw, dict):
        return decode_gguf_export_config(gguf_export_raw)
    raise JSONTypeError(
        f"Field 'gguf_export' must be an object or null, got {type(gguf_export_raw).__name__}"
    )


def encode_train_request_payload(payload: TrainRequestPayload) -> JSONObject:
    """Encode TrainRequestPayload TypedDict to JSONObject for RQ serialization.

    Converts nested LoraConfig, QuantizationConfig, and UnslothConfig TypedDicts
    to plain JSON-serializable dictionaries.

    Args:
        payload: Training request payload to encode.

    Returns:
        JSON-serializable dictionary suitable for RQ job queue.
    """
    lora_encoded: JSONValue = (
        encode_lora_config(payload["lora"]) if payload["lora"] is not None else None
    )
    quantization_encoded: JSONValue = (
        encode_quantization_config(payload["quantization"])
        if payload["quantization"] is not None
        else None
    )
    unsloth_encoded: JSONValue = (
        encode_unsloth_config(payload["unsloth"]) if payload["unsloth"] is not None else None
    )
    gguf_export_encoded: JSONValue = (
        encode_gguf_export_config(payload["gguf_export"])
        if payload["gguf_export"] is not None
        else None
    )

    return {
        "model_family": payload["model_family"],
        "model_size": payload["model_size"],
        "max_seq_len": payload["max_seq_len"],
        "num_epochs": payload["num_epochs"],
        "batch_size": payload["batch_size"],
        "learning_rate": payload["learning_rate"],
        "corpus_file_id": payload["corpus_file_id"],
        "tokenizer_id": payload["tokenizer_id"],
        "holdout_fraction": payload["holdout_fraction"],
        "seed": payload["seed"],
        "pretrained_run_id": payload["pretrained_run_id"],
        "freeze_embed": payload["freeze_embed"],
        "gradient_clipping": payload["gradient_clipping"],
        "optimizer": payload["optimizer"],
        "device": payload["device"],
        "precision": payload["precision"],
        "data_num_workers": payload["data_num_workers"],
        "data_pin_memory": payload["data_pin_memory"],
        "early_stopping_patience": payload["early_stopping_patience"],
        "test_split_ratio": payload["test_split_ratio"],
        "finetune_lr_cap": payload["finetune_lr_cap"],
        "loss_mask_prefix_separator": payload["loss_mask_prefix_separator"],
        "hub_model_id": payload["hub_model_id"],
        "finetuning_strategy": payload["finetuning_strategy"],
        "lora": lora_encoded,
        "quantization": quantization_encoded,
        "unsloth": unsloth_encoded,
        "gguf_export": gguf_export_encoded,
    }


def _narrow_model_family(
    raw: str,
) -> Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"]:
    """Narrow model family string to Literal type with validation.

    Args:
        raw: Raw model family string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid model family.
    """
    if raw == "gpt2":
        return "gpt2"
    if raw == "llama":
        return "llama"
    if raw == "qwen":
        return "qwen"
    if raw == "char_lstm":
        return "char_lstm"
    if raw == "hf_lm":
        return "hf_lm"
    raise JSONTypeError(
        f"Field 'model_family' must be 'gpt2', 'llama', 'qwen', 'char_lstm', or 'hf_lm', "
        f"got '{raw}'"
    )


def _narrow_optimizer(raw: str) -> Literal["adamw", "adam", "sgd"]:
    """Narrow optimizer string to Literal type with validation.

    Args:
        raw: Raw optimizer string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid optimizer.
    """
    if raw == "adamw":
        return "adamw"
    if raw == "adam":
        return "adam"
    if raw == "sgd":
        return "sgd"
    raise JSONTypeError(f"Field 'optimizer' must be 'adamw', 'adam', or 'sgd', got '{raw}'")


def _narrow_device(raw: str) -> Literal["cpu", "cuda", "auto"]:
    """Narrow device string to Literal type with validation.

    Args:
        raw: Raw device string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid device.
    """
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda"
    if raw == "auto":
        return "auto"
    raise JSONTypeError(f"Field 'device' must be 'cpu', 'cuda', or 'auto', got '{raw}'")


def _narrow_precision(raw: str) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Narrow precision string to Literal type with validation.

    Args:
        raw: Raw precision string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid precision.
    """
    if raw == "fp32":
        return "fp32"
    if raw == "fp16":
        return "fp16"
    if raw == "bf16":
        return "bf16"
    if raw == "auto":
        return "auto"
    raise JSONTypeError(f"Field 'precision' must be 'fp32', 'fp16', 'bf16', or 'auto', got '{raw}'")


def _narrow_finetuning_strategy(raw: str) -> Literal["full", "lora", "qlora", "unsloth"]:
    """Narrow finetuning strategy string to Literal type with validation.

    Args:
        raw: Raw finetuning strategy string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid finetuning strategy.
    """
    if raw == "full":
        return "full"
    if raw == "lora":
        return "lora"
    if raw == "qlora":
        return "qlora"
    if raw == "unsloth":
        return "unsloth"
    raise JSONTypeError(
        f"Field 'finetuning_strategy' must be 'full', 'lora', 'qlora', or 'unsloth', got '{raw}'"
    )


def decode_train_request_payload(obj: JSONObject) -> TrainRequestPayload:
    """Decode JSONObject to TrainRequestPayload with full validation.

    Validates all fields and decodes nested LoraConfig, QuantizationConfig,
    and UnslothConfig from JSON objects.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TrainRequestPayload TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    model_family = _narrow_model_family(require_str(obj, "model_family"))
    model_size = require_str(obj, "model_size")
    max_seq_len = require_int(obj, "max_seq_len")
    num_epochs = require_int(obj, "num_epochs")
    batch_size = require_int(obj, "batch_size")
    learning_rate = require_float(obj, "learning_rate")
    corpus_file_id = require_str(obj, "corpus_file_id")
    tokenizer_id = optional_str(obj, "tokenizer_id")
    holdout_fraction = require_float(obj, "holdout_fraction")
    seed = require_int(obj, "seed")
    pretrained_run_id = optional_str(obj, "pretrained_run_id")
    freeze_embed = require_bool(obj, "freeze_embed")
    gradient_clipping = require_float(obj, "gradient_clipping")
    optimizer = _narrow_optimizer(require_str(obj, "optimizer"))
    device = _narrow_device(require_str(obj, "device"))
    precision = _narrow_precision(require_str(obj, "precision"))
    data_num_workers = optional_int(obj, "data_num_workers")
    data_pin_memory_raw = obj.get("data_pin_memory")
    data_pin_memory: bool | None
    if data_pin_memory_raw is None:
        data_pin_memory = None
    elif isinstance(data_pin_memory_raw, bool):
        data_pin_memory = data_pin_memory_raw
    else:
        raise JSONTypeError(
            f"Field 'data_pin_memory' must be a boolean or null, "
            f"got {type(data_pin_memory_raw).__name__}"
        )
    early_stopping_patience = require_int(obj, "early_stopping_patience")
    test_split_ratio = require_float(obj, "test_split_ratio")
    finetune_lr_cap = require_float(obj, "finetune_lr_cap")
    loss_mask_prefix_separator = optional_str(obj, "loss_mask_prefix_separator")
    if loss_mask_prefix_separator == "":
        raise JSONTypeError(
            "Field 'loss_mask_prefix_separator' must not be empty; omit it to disable masking"
        )
    hub_model_id = optional_str(obj, "hub_model_id")
    finetuning_strategy = _narrow_finetuning_strategy(require_str(obj, "finetuning_strategy"))

    # Decode nested configs via helper functions
    lora = _decode_optional_lora(obj)
    quantization = _decode_optional_quantization(obj)
    unsloth = _decode_optional_unsloth(obj)
    gguf_export = _decode_optional_gguf_export(obj)

    return {
        "model_family": model_family,
        "model_size": model_size,
        "max_seq_len": max_seq_len,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "corpus_file_id": corpus_file_id,
        "tokenizer_id": tokenizer_id,
        "holdout_fraction": holdout_fraction,
        "seed": seed,
        "pretrained_run_id": pretrained_run_id,
        "freeze_embed": freeze_embed,
        "gradient_clipping": gradient_clipping,
        "optimizer": optimizer,
        "device": device,
        "precision": precision,
        "data_num_workers": data_num_workers,
        "data_pin_memory": data_pin_memory,
        "early_stopping_patience": early_stopping_patience,
        "test_split_ratio": test_split_ratio,
        "finetune_lr_cap": finetune_lr_cap,
        "loss_mask_prefix_separator": loss_mask_prefix_separator,
        "hub_model_id": hub_model_id,
        "finetuning_strategy": finetuning_strategy,
        "lora": lora,
        "quantization": quantization,
        "unsloth": unsloth,
        "gguf_export": gguf_export,
    }


def encode_cloze_job_payload(payload: ClozeJobPayload) -> JSONObject:
    """Encode ClozeJobPayload TypedDict to JSONObject for RQ serialization.

    Args:
        payload: Cloze evaluation job payload to encode.

    Returns:
        JSON-serializable dictionary suitable for RQ job queue.
    """
    return {
        "run_id": payload["run_id"],
        "request_id": payload["request_id"],
        "items_file_id": payload["items_file_id"],
        "max_seq_len": payload["max_seq_len"],
    }


def decode_cloze_job_payload(obj: JSONObject) -> ClozeJobPayload:
    """Decode JSONObject to ClozeJobPayload with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated ClozeJobPayload TypedDict.

    Raises:
        JSONTypeError: If required fields are missing, have wrong types, or
            ``max_seq_len`` is not positive.
    """
    run_id = require_str(obj, "run_id")
    request_id = require_str(obj, "request_id")
    items_file_id = require_str(obj, "items_file_id")
    max_seq_len = require_int(obj, "max_seq_len")
    if max_seq_len <= 0:
        raise JSONTypeError(f"Field 'max_seq_len' must be positive, got {max_seq_len}")

    return {
        "run_id": run_id,
        "request_id": request_id,
        "items_file_id": items_file_id,
        "max_seq_len": max_seq_len,
    }


def encode_train_job_payload(payload: TrainJobPayload) -> JSONObject:
    """Encode TrainJobPayload TypedDict to JSONObject for RQ serialization.

    Args:
        payload: Training job payload to encode.

    Returns:
        JSON-serializable dictionary suitable for RQ job queue.
    """
    return {
        "run_id": payload["run_id"],
        "user_id": payload["user_id"],
        "request": encode_train_request_payload(payload["request"]),
    }


def decode_train_job_payload(obj: JSONObject) -> TrainJobPayload:
    """Decode JSONObject to TrainJobPayload with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TrainJobPayload TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    run_id = require_str(obj, "run_id")
    user_id = require_int(obj, "user_id")

    request_raw = obj.get("request")
    if request_raw is None:
        raise JSONTypeError("Missing required field 'request'")
    if not isinstance(request_raw, dict):
        raise JSONTypeError(f"Field 'request' must be an object, got {type(request_raw).__name__}")
    request = decode_train_request_payload(request_raw)

    return {
        "run_id": run_id,
        "user_id": user_id,
        "request": request,
    }


__all__ = [
    "decode_cloze_job_payload",
    "decode_gguf_export_config",
    "decode_lora_config",
    "decode_quantization_config",
    "decode_train_job_payload",
    "decode_train_request_payload",
    "decode_unsloth_config",
    "encode_cloze_job_payload",
    "encode_gguf_export_config",
    "encode_lora_config",
    "encode_quantization_config",
    "encode_train_job_payload",
    "encode_train_request_payload",
    "encode_unsloth_config",
]
