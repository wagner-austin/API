"""Field, narrowing, and config-section decoders for run requests."""

from __future__ import annotations

from typing import Literal

from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue
from platform_core.validators import (
    validate_float_range,
    validate_int_range,
    validate_optional_literal,
)

from ..schemas.runs import (
    GgufExportConfigRequest,
    LoraConfigRequest,
    QuantizationConfigRequest,
)

_LORA_BIASES: frozenset[str] = frozenset({"none", "all", "lora_only"})

_QUANT_COMPUTE_DTYPES: frozenset[str] = frozenset({"float16", "bfloat16", "float32"})

_QUANT_TYPES: frozenset[str] = frozenset({"nf4", "fp4"})

_GGUF_OUTPUT_TYPES: frozenset[str] = frozenset({"f32", "f16", "bf16", "q8_0"})


def _validate_bool(d: dict[str, JSONValue], field: str, *, default: bool) -> bool:
    """Validate a boolean field with a default value."""
    val = d.get(field)
    if val is None:
        return default
    if not isinstance(val, bool):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=f"{field} must be a boolean",
            http_status=400,
        )
    return val


def _decode_optional_int_ge(d: dict[str, JSONValue], field: str, *, ge: int) -> int | None:
    raw = d.get(field)
    if raw is None:
        return None
    return validate_int_range(raw, field, ge=ge)


def _decode_optional_bool(d: dict[str, JSONValue], field: str) -> bool | None:
    raw = d.get(field)
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=f"{field} must be a boolean",
            http_status=400,
        )
    return bool(raw)


def _decode_loss_mask_prefix_separator(d: dict[str, JSONValue]) -> str | None:
    """Decode the marker separator that splits a masked prefix from the body.

    Args:
        d: Parsed request body.

    Returns:
        The separator, or None when the request does not ask for masking.

    Raises:
        AppError: With ``INVALID_INPUT`` when the value is not a string, or is
            the empty string. Empty would split nothing while the run's
            manifest recorded that masking was requested, so the run would look
            like a masked arm and behave like an unmasked one.
    """
    raw = d.get("loss_mask_prefix_separator")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="loss_mask_prefix_separator must be a string",
            http_status=400,
        )
    if raw == "":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="loss_mask_prefix_separator must not be empty; omit it to disable masking",
            http_status=422,
        )
    return raw


def _narrow_model_family(
    raw: str | None,
) -> Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"]:
    """Narrow model family string to Literal type.

    Args:
        raw: Raw model family string from request.

    Returns:
        Narrowed Literal type for model family.
    """
    val = raw if raw is not None else "gpt2"
    if val == "gpt2":
        return "gpt2"
    if val == "llama":
        return "llama"
    if val == "qwen":
        return "qwen"
    if val == "hf_lm":
        return "hf_lm"
    return "char_lstm"


def _narrow_finetuning_strategy(
    raw: str | None,
) -> Literal["full", "lora", "qlora"]:
    """Narrow finetuning strategy string to Literal type.

    Args:
        raw: Raw finetuning strategy string from request.

    Returns:
        Narrowed Literal type for finetuning strategy.
    """
    val = raw if raw is not None else "full"
    if val == "lora":
        return "lora"
    if val == "qlora":
        return "qlora"
    return "full"


def _narrow_lora_bias(raw: str) -> Literal["none", "all", "lora_only"]:
    """Narrow LoRA bias string to Literal type.

    Args:
        raw: Raw bias string from request.

    Returns:
        Narrowed Literal type for LoRA bias.
    """
    if raw == "all":
        return "all"
    if raw == "lora_only":
        return "lora_only"
    return "none"


def _narrow_quant_compute_dtype(
    raw: str,
) -> Literal["float16", "bfloat16", "float32"]:
    """Narrow quantization compute dtype to Literal type.

    Args:
        raw: Raw compute dtype string from request.

    Returns:
        Narrowed Literal type for compute dtype.
    """
    if raw == "bfloat16":
        return "bfloat16"
    if raw == "float32":
        return "float32"
    return "float16"


def _narrow_quant_type(raw: str) -> Literal["nf4", "fp4"]:
    """Narrow quantization type to Literal type.

    Args:
        raw: Raw quant type string from request.

    Returns:
        Narrowed Literal type for quantization type.
    """
    if raw == "fp4":
        return "fp4"
    return "nf4"


def _narrow_gguf_output_type(raw: str) -> Literal["f32", "f16", "bf16", "q8_0"]:
    """Narrow GGUF output type to Literal type.

    Args:
        raw: Raw output type string from request.

    Returns:
        Narrowed Literal type for GGUF output.
    """
    if raw == "f16":
        return "f16"
    if raw == "bf16":
        return "bf16"
    if raw == "q8_0":
        return "q8_0"
    return "f32"


def _narrow_optimizer(raw: str | None) -> Literal["adamw", "adam", "sgd"]:
    """Narrow optimizer string to Literal type."""
    val = raw if raw is not None else "adamw"
    if val == "adam":
        return "adam"
    if val == "sgd":
        return "sgd"
    return "adamw"


def _narrow_device(raw: str | None) -> Literal["cpu", "cuda", "auto"]:
    """Narrow device string to Literal type."""
    val = raw if raw is not None else "auto"
    if val == "cuda":
        return "cuda"
    if val == "cpu":
        return "cpu"
    return "auto"


def _narrow_precision(raw: str | None) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Narrow precision string to Literal type."""
    val = raw if raw is not None else "auto"
    if val == "fp32":
        return "fp32"
    if val == "fp16":
        return "fp16"
    if val == "bf16":
        return "bf16"
    return "auto"


def _decode_lora_config(d: dict[str, JSONValue]) -> LoraConfigRequest:
    """Decode and validate LoRA configuration from JSON dict.

    Args:
        d: Raw dictionary with LoRA config fields.

    Returns:
        Validated LoraConfigRequest TypedDict.

    Raises:
        AppError: If required fields are missing or invalid.
    """
    enabled_raw = d.get("enabled")
    if enabled_raw is None:
        enabled = True
    elif not isinstance(enabled_raw, bool):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="lora.enabled must be a boolean",
            http_status=400,
        )
    else:
        enabled = enabled_raw

    r_val = validate_int_range(d.get("r"), "lora.r", ge=4, le=128, default=16)
    lora_alpha = validate_int_range(
        d.get("lora_alpha"), "lora.lora_alpha", ge=1, le=256, default=16
    )
    lora_dropout = validate_float_range(
        d.get("lora_dropout"), "lora.lora_dropout", ge=0.0, le=0.5, default=0.1
    )

    target_modules_raw = d.get("target_modules")
    if target_modules_raw is None:
        target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    elif not isinstance(target_modules_raw, list):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="lora.target_modules must be a list of strings",
            http_status=400,
        )
    else:
        modules_list: list[str] = []
        for i, m in enumerate(target_modules_raw):
            if not isinstance(m, str):
                raise AppError(
                    code=ErrorCode.INVALID_INPUT,
                    message=f"lora.target_modules[{i}] must be a string",
                    http_status=400,
                )
            modules_list.append(m)
        target_modules = tuple(modules_list)

    bias_raw = validate_optional_literal(d.get("bias"), "lora.bias", _LORA_BIASES)
    bias = _narrow_lora_bias(bias_raw if bias_raw is not None else "none")

    return {
        "enabled": enabled,
        "r": r_val,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "bias": bias,
    }


def _decode_quantization_config(d: dict[str, JSONValue]) -> QuantizationConfigRequest:
    """Decode and validate quantization configuration from JSON dict.

    Args:
        d: Raw dictionary with quantization config fields.

    Returns:
        Validated QuantizationConfigRequest TypedDict.

    Raises:
        AppError: If required fields are missing or invalid.
    """
    load_in_4bit_raw = d.get("load_in_4bit")
    if load_in_4bit_raw is None:
        load_in_4bit = True
    elif not isinstance(load_in_4bit_raw, bool):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="quantization.load_in_4bit must be a boolean",
            http_status=400,
        )
    else:
        load_in_4bit = load_in_4bit_raw

    load_in_8bit_raw = d.get("load_in_8bit")
    if load_in_8bit_raw is None:
        load_in_8bit = False
    elif not isinstance(load_in_8bit_raw, bool):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="quantization.load_in_8bit must be a boolean",
            http_status=400,
        )
    else:
        load_in_8bit = load_in_8bit_raw

    compute_dtype_raw = validate_optional_literal(
        d.get("bnb_4bit_compute_dtype"),
        "quantization.bnb_4bit_compute_dtype",
        _QUANT_COMPUTE_DTYPES,
    )
    compute_dtype = _narrow_quant_compute_dtype(
        compute_dtype_raw if compute_dtype_raw is not None else "float16"
    )

    quant_type_raw = validate_optional_literal(
        d.get("bnb_4bit_quant_type"),
        "quantization.bnb_4bit_quant_type",
        _QUANT_TYPES,
    )
    quant_type = _narrow_quant_type(quant_type_raw if quant_type_raw is not None else "nf4")

    return {
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "bnb_4bit_compute_dtype": compute_dtype,
        "bnb_4bit_quant_type": quant_type,
    }


def _decode_optional_lora(d: dict[str, JSONValue]) -> LoraConfigRequest | None:
    """Decode optional LoRA config from dict."""
    lora_raw = d.get("lora")
    if lora_raw is None:
        return None
    if not isinstance(lora_raw, dict):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="lora must be an object",
            http_status=400,
        )
    return _decode_lora_config(lora_raw)


def _decode_optional_quantization(d: dict[str, JSONValue]) -> QuantizationConfigRequest | None:
    """Decode optional quantization config from dict."""
    quantization_raw = d.get("quantization")
    if quantization_raw is None:
        return None
    if not isinstance(quantization_raw, dict):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="quantization must be an object",
            http_status=400,
        )
    return _decode_quantization_config(quantization_raw)


def _decode_gguf_export_config(d: dict[str, JSONValue]) -> GgufExportConfigRequest:
    """Decode and validate GGUF export configuration from JSON dict.

    Args:
        d: Raw dictionary with GGUF export config fields.

    Returns:
        Validated GgufExportConfigRequest TypedDict.

    Raises:
        AppError: If required fields are missing or invalid.
    """
    enabled_raw = d.get("enabled")
    if enabled_raw is None:
        enabled = True
    elif not isinstance(enabled_raw, bool):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="gguf_export.enabled must be a boolean",
            http_status=400,
        )
    else:
        enabled = enabled_raw

    output_type_raw = validate_optional_literal(
        d.get("output_type"),
        "gguf_export.output_type",
        _GGUF_OUTPUT_TYPES,
    )
    output_type = _narrow_gguf_output_type(
        output_type_raw if output_type_raw is not None else "f16"
    )

    return {
        "enabled": enabled,
        "output_type": output_type,
    }


def _decode_optional_gguf_export(d: dict[str, JSONValue]) -> GgufExportConfigRequest | None:
    """Decode optional GGUF export config from dict."""
    gguf_export_raw = d.get("gguf_export")
    if gguf_export_raw is None:
        return None
    if not isinstance(gguf_export_raw, dict):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="gguf_export must be an object",
            http_status=400,
        )
    return _decode_gguf_export_config(gguf_export_raw)


def _validate_hf_lm_cross_fields(
    model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"],
    hub_model_id: str | None,
    finetuning_strategy: Literal["full", "lora", "qlora"],
    lora: LoraConfigRequest | None,
    quantization: QuantizationConfigRequest | None,
    gguf_export: GgufExportConfigRequest | None,
) -> None:
    """Validate cross-field requirements for HF LM backend."""
    if model_family == "hf_lm" and hub_model_id is None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="hub_model_id is required when model_family is 'hf_lm'",
            http_status=400,
        )
    if finetuning_strategy in ("lora", "qlora") and lora is None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=f"lora config is required for finetuning_strategy '{finetuning_strategy}'",
            http_status=400,
        )
    if finetuning_strategy == "qlora" and quantization is None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="quantization config is required for finetuning_strategy 'qlora'",
            http_status=400,
        )
    if gguf_export is not None and finetuning_strategy not in ("lora", "qlora"):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="gguf_export requires finetuning_strategy lora or qlora",
            http_status=400,
        )
