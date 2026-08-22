"""Codec for the nested config sections of queue payloads."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_bool,
    require_float,
    require_int,
    require_str,
)

from .model import GgufExportConfig, LoraConfig, QuantizationConfig


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
