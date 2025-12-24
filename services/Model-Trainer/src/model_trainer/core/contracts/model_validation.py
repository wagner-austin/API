"""Validation and encoding functions for model configuration TypedDicts.

Every TypedDict needs encode/decode functions with require_* validation.
Uses JSONObject from platform_core for strict typing.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_bool,
    require_int,
    require_str,
)

from model_trainer.core.contracts.model import (
    LoraConfig,
    QuantizationConfig,
    UnslothConfig,
)

# =============================================================================
# LoraConfig validation and encoding
# =============================================================================


def _decode_lora_config(data: JSONObject) -> LoraConfig:
    """Decode raw JSON object to LoraConfig.

    Args:
        data: Raw JSON dictionary.

    Returns:
        Validated LoraConfig TypedDict.

    Raises:
        TypeError: If field types are incorrect.
        ValueError: If field values are invalid.
    """
    enabled = require_bool(data, "enabled")

    r = require_int(data, "r")
    if r < 1:
        raise ValueError("lora.r must be >= 1")

    lora_alpha = require_int(data, "lora_alpha")
    if lora_alpha < 1:
        raise ValueError("lora.lora_alpha must be >= 1")

    lora_dropout_raw = data.get("lora_dropout")
    if not isinstance(lora_dropout_raw, (int, float)):
        raise TypeError("lora.lora_dropout must be float")
    lora_dropout_float = float(lora_dropout_raw)
    if not 0.0 <= lora_dropout_float <= 1.0:
        raise ValueError("lora.lora_dropout must be between 0.0 and 1.0")

    target_modules_raw = data.get("target_modules")
    if not isinstance(target_modules_raw, list):
        raise TypeError("lora.target_modules must be list")
    if len(target_modules_raw) == 0:
        raise ValueError("lora.target_modules must not be empty")
    target_modules_list: list[str] = []
    for module in target_modules_raw:
        if not isinstance(module, str):
            raise TypeError("lora.target_modules elements must be str")
        target_modules_list.append(module)
    target_modules_tuple = tuple(target_modules_list)

    bias = require_str(data, "bias")
    if bias not in ("none", "all", "lora_only"):
        raise ValueError("lora.bias must be 'none', 'all', or 'lora_only'")
    bias_literal: Literal["none", "all", "lora_only"] = (
        "none" if bias == "none" else ("all" if bias == "all" else "lora_only")
    )

    return LoraConfig(
        enabled=enabled,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout_float,
        target_modules=target_modules_tuple,
        bias=bias_literal,
    )


def encode_lora_config(cfg: LoraConfig) -> JSONObject:
    """Encode LoraConfig to JSON-serializable dict.

    Args:
        cfg: LoraConfig TypedDict.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    result: JSONObject = {
        "enabled": cfg["enabled"],
        "r": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "target_modules": list(cfg["target_modules"]),
        "bias": cfg["bias"],
    }
    return result


# =============================================================================
# QuantizationConfig validation and encoding
# =============================================================================


def _decode_quantization_config(data: JSONObject) -> QuantizationConfig:
    """Decode raw JSON object to QuantizationConfig.

    Args:
        data: Raw JSON dictionary.

    Returns:
        Validated QuantizationConfig TypedDict.

    Raises:
        TypeError: If field types are incorrect.
        ValueError: If field values are invalid.
    """
    load_in_4bit = require_bool(data, "load_in_4bit")
    load_in_8bit = require_bool(data, "load_in_8bit")

    if load_in_4bit and load_in_8bit:
        raise ValueError("quantization cannot have both load_in_4bit and load_in_8bit True")

    compute_dtype = require_str(data, "bnb_4bit_compute_dtype")
    if compute_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(
            "quantization.bnb_4bit_compute_dtype must be 'float16', 'bfloat16', or 'float32'"
        )
    compute_dtype_literal: Literal["float16", "bfloat16", "float32"] = (
        "float16"
        if compute_dtype == "float16"
        else ("bfloat16" if compute_dtype == "bfloat16" else "float32")
    )

    quant_type = require_str(data, "bnb_4bit_quant_type")
    if quant_type not in ("nf4", "fp4"):
        raise ValueError("quantization.bnb_4bit_quant_type must be 'nf4' or 'fp4'")
    quant_type_literal: Literal["nf4", "fp4"] = "nf4" if quant_type == "nf4" else "fp4"

    return QuantizationConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=compute_dtype_literal,
        bnb_4bit_quant_type=quant_type_literal,
    )


def encode_quantization_config(cfg: QuantizationConfig) -> JSONObject:
    """Encode QuantizationConfig to JSON-serializable dict.

    Args:
        cfg: QuantizationConfig TypedDict.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    result: JSONObject = {
        "load_in_4bit": cfg["load_in_4bit"],
        "load_in_8bit": cfg["load_in_8bit"],
        "bnb_4bit_compute_dtype": cfg["bnb_4bit_compute_dtype"],
        "bnb_4bit_quant_type": cfg["bnb_4bit_quant_type"],
    }
    return result


# =============================================================================
# UnslothConfig validation and encoding
# =============================================================================


def _decode_unsloth_config(data: JSONObject) -> UnslothConfig:
    """Decode raw JSON object to UnslothConfig.

    Args:
        data: Raw JSON dictionary.

    Returns:
        Validated UnslothConfig TypedDict.

    Raises:
        TypeError: If field types are incorrect.
        ValueError: If field values are invalid.
    """
    enabled = require_bool(data, "enabled")

    max_seq_length = require_int(data, "max_seq_length")
    if max_seq_length < 1:
        raise ValueError("unsloth.max_seq_length must be >= 1")

    dtype_raw: JSONValue = data.get("dtype")
    if dtype_raw is None:
        dtype_value: Literal["float16", "bfloat16"] | None = None
    elif isinstance(dtype_raw, str):
        if dtype_raw not in ("float16", "bfloat16"):
            raise ValueError("unsloth.dtype must be 'float16', 'bfloat16', or null")
        dtype_value = "float16" if dtype_raw == "float16" else "bfloat16"
    else:
        raise TypeError("unsloth.dtype must be string or null")

    return UnslothConfig(
        enabled=enabled,
        max_seq_length=max_seq_length,
        dtype=dtype_value,
    )


def encode_unsloth_config(cfg: UnslothConfig) -> JSONObject:
    """Encode UnslothConfig to JSON-serializable dict.

    Args:
        cfg: UnslothConfig TypedDict.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    result: JSONObject = {
        "enabled": cfg["enabled"],
        "max_seq_length": cfg["max_seq_length"],
        "dtype": cfg["dtype"],
    }
    return result


# =============================================================================
# Optional config helpers (decode from parent objects)
# =============================================================================


def _decode_optional_lora_config(data: JSONObject) -> LoraConfig | None:
    """Decode optional LoRA config from parent config.

    Args:
        data: Parent config dict that may contain 'lora' key.

    Returns:
        Validated LoraConfig or None if not present.

    Raises:
        TypeError: If lora is present but not a dict.
        ValueError: If lora dict has invalid values.
    """
    lora_raw: JSONValue = data.get("lora")
    if lora_raw is None:
        return None
    if not isinstance(lora_raw, dict):
        raise TypeError("lora must be dict or null")
    return _decode_lora_config(lora_raw)


def _decode_optional_quantization_config(
    data: JSONObject,
) -> QuantizationConfig | None:
    """Decode optional quantization config from parent config.

    Args:
        data: Parent config dict that may contain 'quantization' key.

    Returns:
        Validated QuantizationConfig or None if not present.

    Raises:
        TypeError: If quantization is present but not a dict.
        ValueError: If quantization dict has invalid values.
    """
    quant_raw: JSONValue = data.get("quantization")
    if quant_raw is None:
        return None
    if not isinstance(quant_raw, dict):
        raise TypeError("quantization must be dict or null")
    return _decode_quantization_config(quant_raw)


def _decode_optional_unsloth_config(data: JSONObject) -> UnslothConfig | None:
    """Decode optional Unsloth config from parent config.

    Args:
        data: Parent config dict that may contain 'unsloth' key.

    Returns:
        Validated UnslothConfig or None if not present.

    Raises:
        TypeError: If unsloth is present but not a dict.
        ValueError: If unsloth dict has invalid values.
    """
    unsloth_raw: JSONValue = data.get("unsloth")
    if unsloth_raw is None:
        return None
    if not isinstance(unsloth_raw, dict):
        raise TypeError("unsloth must be dict or null")
    return _decode_unsloth_config(unsloth_raw)


__all__ = [
    "JSONObject",
    "_decode_lora_config",
    "_decode_optional_lora_config",
    "_decode_optional_quantization_config",
    "_decode_optional_unsloth_config",
    "_decode_quantization_config",
    "_decode_unsloth_config",
    "encode_lora_config",
    "encode_quantization_config",
    "encode_unsloth_config",
]
