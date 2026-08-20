from __future__ import annotations

from typing import Literal

from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue
from platform_core.validators import (
    load_json_dict,
    validate_float_range,
    validate_int_range,
    validate_optional_literal,
    validate_str,
)

from ..schemas.runs import (
    BaselineClozeRequest,
    ChatRequest,
    ClozeRequest,
    EvaluateRequest,
    GenerateRequest,
    GgufExportConfigRequest,
    LoraConfigRequest,
    QuantizationConfigRequest,
    ScoreRequest,
    TrainRequest,
)

_MODEL_FAMILIES: frozenset[str] = frozenset({"gpt2", "llama", "qwen", "char_lstm", "hf_lm"})
_OPTIMIZERS: frozenset[str] = frozenset({"adamw", "adam", "sgd"})
_DEVICES: frozenset[str] = frozenset({"cpu", "cuda", "auto"})
_PRECISIONS: frozenset[str] = frozenset({"fp32", "fp16", "bf16", "auto"})
_SPLITS: frozenset[str] = frozenset({"validation", "test"})
_DETAIL_LEVELS: frozenset[str] = frozenset({"summary", "per_char"})
_FINETUNING_STRATEGIES: frozenset[str] = frozenset({"full", "lora", "qlora"})
_LORA_BIASES: frozenset[str] = frozenset({"none", "all", "lora_only"})
_QUANT_COMPUTE_DTYPES: frozenset[str] = frozenset({"float16", "bfloat16", "float32"})
_QUANT_TYPES: frozenset[str] = frozenset({"nf4", "fp4"})
_GGUF_OUTPUT_TYPES: frozenset[str] = frozenset({"f32", "f16", "bf16", "q8_0"})
_ALLOWED_TRAIN_FIELDS: frozenset[str] = frozenset(
    {
        "model_family",
        "model_size",
        "max_seq_len",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "corpus_file_id",
        "tokenizer_id",
        "holdout_fraction",
        "seed",
        "pretrained_run_id",
        "freeze_embed",
        "gradient_clipping",
        "optimizer",
        "user_id",
        "device",
        "precision",
        "data_num_workers",
        "data_pin_memory",
        "early_stopping_patience",
        "test_split_ratio",
        "finetune_lr_cap",
        "loss_mask_prefix_separator",
        # HuggingFace LM backend fields
        "hub_model_id",
        "finetuning_strategy",
        "lora",
        "quantization",
        "gguf_export",
    }
)


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


class _HfLmFields:
    """Container for decoded HF LM backend fields."""

    hub_model_id: str | None
    finetuning_strategy: Literal["full", "lora", "qlora"]
    lora: LoraConfigRequest | None
    quantization: QuantizationConfigRequest | None
    gguf_export: GgufExportConfigRequest | None

    def __init__(
        self,
        hub_model_id: str | None,
        finetuning_strategy: Literal["full", "lora", "qlora"],
        lora: LoraConfigRequest | None,
        quantization: QuantizationConfigRequest | None,
        gguf_export: GgufExportConfigRequest | None,
    ) -> None:
        self.hub_model_id = hub_model_id
        self.finetuning_strategy = finetuning_strategy
        self.lora = lora
        self.quantization = quantization
        self.gguf_export = gguf_export


def _decode_hf_lm_fields(
    d: dict[str, JSONValue],
    model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"],
) -> _HfLmFields:
    """Decode and validate HuggingFace LM backend fields.

    Args:
        d: Raw request dictionary.
        model_family: Validated model family.

    Returns:
        _HfLmFields container with decoded values.

    Raises:
        AppError: If validation fails.
    """
    hub_model_id_raw = d.get("hub_model_id")
    hub_model_id: str | None = None
    if hub_model_id_raw is not None:
        hub_model_id = validate_str(hub_model_id_raw, "hub_model_id")

    finetuning_strategy_raw = validate_optional_literal(
        d.get("finetuning_strategy"), "finetuning_strategy", _FINETUNING_STRATEGIES
    )
    finetuning_strategy = _narrow_finetuning_strategy(finetuning_strategy_raw)

    lora = _decode_optional_lora(d)
    quantization = _decode_optional_quantization(d)
    gguf_export = _decode_optional_gguf_export(d)

    _validate_hf_lm_cross_fields(
        model_family, hub_model_id, finetuning_strategy, lora, quantization, gguf_export
    )

    return _HfLmFields(
        hub_model_id=hub_model_id,
        finetuning_strategy=finetuning_strategy,
        lora=lora,
        quantization=quantization,
        gguf_export=gguf_export,
    )


def _decode_train_request(obj: JSONValue) -> TrainRequest:
    d = load_json_dict(obj)

    extra_fields = set(d.keys()) - _ALLOWED_TRAIN_FIELDS
    if extra_fields:
        extra_list = ", ".join(sorted(extra_fields))
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Extra fields not allowed: {extra_list}",
            http_status=422,
        )

    model_family_raw = validate_optional_literal(
        d.get("model_family"), "model_family", _MODEL_FAMILIES
    )
    model_family = _narrow_model_family(model_family_raw)

    model_size = validate_str(d.get("model_size"), "model_size", default="small")
    max_seq_len = validate_int_range(d.get("max_seq_len"), "max_seq_len", ge=8, default=512)
    num_epochs = validate_int_range(d.get("num_epochs"), "num_epochs", ge=1, default=1)
    batch_size = validate_int_range(d.get("batch_size"), "batch_size", ge=1, default=4)
    learning_rate = validate_float_range(
        d.get("learning_rate"), "learning_rate", ge=0.0, default=5e-4
    )
    corpus_file_id = validate_str(d.get("corpus_file_id"), "corpus_file_id")

    # tokenizer_id validation depends on model_family:
    # - hf_lm: optional (None) - uses HF tokenizer from hub_model_id
    # - other models: required - must provide a trained tokenizer
    tokenizer_id_raw = d.get("tokenizer_id")
    tokenizer_id: str | None
    if model_family == "hf_lm":
        # For hf_lm, tokenizer_id is optional - accept None or empty string as None
        if tokenizer_id_raw is None or tokenizer_id_raw == "":
            tokenizer_id = None
        elif isinstance(tokenizer_id_raw, str):
            tokenizer_id = tokenizer_id_raw
        else:
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message="tokenizer_id must be a string or null for hf_lm models",
                http_status=400,
            )
    else:
        # For non-hf_lm models, tokenizer_id is required
        if tokenizer_id_raw is None or tokenizer_id_raw == "":
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=f"tokenizer_id is required for {model_family} models",
                http_status=400,
            )
        tokenizer_id = validate_str(tokenizer_id_raw, "tokenizer_id")
    holdout_fraction = validate_float_range(
        d.get("holdout_fraction"), "holdout_fraction", ge=0.0, le=0.5, default=0.01
    )
    seed = validate_int_range(d.get("seed"), "seed", ge=0, default=42)

    pretrained_run_id_raw = d.get("pretrained_run_id")
    pretrained_run_id: str | None = None
    if pretrained_run_id_raw is not None:
        pretrained_run_id = validate_str(pretrained_run_id_raw, "pretrained_run_id")

    freeze_embed = _validate_bool(d, "freeze_embed", default=False)
    gradient_clipping = validate_float_range(
        d.get("gradient_clipping"), "gradient_clipping", ge=0.0, default=1.0
    )

    optimizer_raw = validate_optional_literal(d.get("optimizer"), "optimizer", _OPTIMIZERS)
    optimizer = _narrow_optimizer(optimizer_raw)

    user_id = validate_int_range(d.get("user_id"), "user_id", ge=0, default=0)

    # Device: accept "auto" at API edge and resolve later in worker
    device_raw = validate_optional_literal(d.get("device"), "device", _DEVICES)
    device_api = _narrow_device(device_raw)

    # Precision: accept "auto" at API edge and resolve later in worker
    precision_raw = validate_optional_literal(d.get("precision"), "precision", _PRECISIONS)
    precision_api = _narrow_precision(precision_raw)

    # Early stopping patience validation
    early_stopping_patience = validate_int_range(
        d.get("early_stopping_patience"), "early_stopping_patience", ge=1, default=5
    )

    # Test split ratio validation
    test_split_ratio = validate_float_range(
        d.get("test_split_ratio"), "test_split_ratio", ge=0.0, le=0.5, default=0.15
    )

    # Finetune LR cap validation
    finetune_lr_cap = validate_float_range(
        d.get("finetune_lr_cap"), "finetune_lr_cap", ge=0.0, default=5e-5
    )

    # Marker separator for metadata-conditioned corpora. Absent means every
    # token is a loss target; an empty string would mask nothing while claiming
    # to mask, so it is rejected rather than silently normalised to None.
    loss_mask_prefix_separator = _decode_loss_mask_prefix_separator(d)

    # Optional data loader knobs: accept or leave None; worker resolves defaults by device
    data_num_workers = _decode_optional_int_ge(d, "data_num_workers", ge=0)
    data_pin_memory = _decode_optional_bool(d, "data_pin_memory")

    # HuggingFace LM backend fields (delegated to reduce complexity)
    hf_fields = _decode_hf_lm_fields(d, model_family)

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
        "user_id": user_id,
        "device": device_api,
        "precision": precision_api,
        "data_num_workers": data_num_workers,
        "data_pin_memory": data_pin_memory,
        "early_stopping_patience": early_stopping_patience,
        "test_split_ratio": test_split_ratio,
        "finetune_lr_cap": finetune_lr_cap,
        "loss_mask_prefix_separator": loss_mask_prefix_separator,
        "hub_model_id": hf_fields.hub_model_id,
        "finetuning_strategy": hf_fields.finetuning_strategy,
        "lora": hf_fields.lora,
        "quantization": hf_fields.quantization,
        "gguf_export": hf_fields.gguf_export,
    }


def _decode_evaluate_request(obj: JSONValue) -> EvaluateRequest:
    d = load_json_dict(obj)

    split_raw = validate_optional_literal(d.get("split"), "split", _SPLITS)
    split_str = split_raw if split_raw is not None else "validation"
    split: Literal["validation", "test"] = "validation" if split_str == "validation" else "test"

    path_override_raw = d.get("path_override")
    path_override: str | None = None
    if path_override_raw is not None:
        path_override = validate_str(path_override_raw, "path_override")

    result: EvaluateRequest = {"split": split}
    if path_override is not None:
        result["path_override"] = path_override

    return result


def _decode_cloze_request(obj: JSONValue) -> ClozeRequest:
    """Decode and validate a cloze evaluation request.

    Args:
        obj: Parsed request body.

    Returns:
        Validated ClozeRequest.

    Raises:
        AppError: With ``INVALID_INPUT`` when ``items_file_id`` is missing or
            empty, or ``max_seq_len`` falls outside the supported range.
    """
    d = load_json_dict(obj)

    items_file_id = validate_str(d.get("items_file_id"), "items_file_id")
    if items_file_id.strip() == "":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="items_file_id must not be blank",
            http_status=422,
        )

    max_seq_len = validate_int_range(d.get("max_seq_len"), "max_seq_len", ge=8, default=512)

    return ClozeRequest(items_file_id=items_file_id, max_seq_len=max_seq_len)


def _decode_baseline_cloze_request(obj: JSONValue) -> BaselineClozeRequest:
    """Decode and validate a request to score an untrained model.

    Args:
        obj: Parsed request body.

    Returns:
        Validated BaselineClozeRequest.

    Raises:
        AppError: With ``INVALID_INPUT`` when ``hub_model_id`` or
            ``items_file_id`` is missing or blank, or ``max_seq_len`` falls
            outside the supported range. Both ids are rejected blank here
            because together they form the key the result is stored under, and
            a blank half would produce a record nobody can identify.
    """
    d = load_json_dict(obj)

    hub_model_id = validate_str(d.get("hub_model_id"), "hub_model_id")
    if hub_model_id.strip() == "":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="hub_model_id must not be blank",
            http_status=422,
        )

    items_file_id = validate_str(d.get("items_file_id"), "items_file_id")
    if items_file_id.strip() == "":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="items_file_id must not be blank",
            http_status=422,
        )

    max_seq_len = validate_int_range(d.get("max_seq_len"), "max_seq_len", ge=8, default=512)
    device = validate_str(d.get("device"), "device") if d.get("device") is not None else "cpu"

    return BaselineClozeRequest(
        hub_model_id=hub_model_id,
        items_file_id=items_file_id,
        max_seq_len=max_seq_len,
        device=device,
    )


def _decode_score_request(obj: JSONValue) -> ScoreRequest:
    """Decode and validate a score request."""
    d = load_json_dict(obj)

    # text and path are mutually exclusive
    text_raw = d.get("text")
    path_raw = d.get("path")

    text: str | None = None
    path: str | None = None

    if text_raw is not None and path_raw is not None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="text and path are mutually exclusive",
            http_status=422,
        )
    if text_raw is None and path_raw is None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="either text or path must be provided",
            http_status=422,
        )

    if text_raw is not None:
        text = validate_str(text_raw, "text")
    if path_raw is not None:
        path = validate_str(path_raw, "path")

    detail_level_raw = validate_optional_literal(
        d.get("detail_level"), "detail_level", _DETAIL_LEVELS
    )
    detail_level_str = detail_level_raw if detail_level_raw is not None else "summary"
    detail_level: Literal["summary", "per_char"] = (
        "per_char" if detail_level_str == "per_char" else "summary"
    )

    top_k_raw = d.get("top_k")
    top_k: int | None = None
    if top_k_raw is not None:
        top_k = validate_int_range(top_k_raw, "top_k", ge=1)

    seed_raw = d.get("seed")
    seed: int | None = None
    if seed_raw is not None:
        seed = validate_int_range(seed_raw, "seed", ge=0)

    return {
        "text": text,
        "path": path,
        "detail_level": detail_level,
        "top_k": top_k,
        "seed": seed,
    }


def _validate_stop_on_eos(d: dict[str, JSONValue]) -> bool:
    """Validate stop_on_eos field."""
    stop_on_eos_raw = d.get("stop_on_eos")
    if stop_on_eos_raw is None:
        return True
    if not isinstance(stop_on_eos_raw, bool):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="stop_on_eos must be a boolean",
            http_status=422,
        )
    return stop_on_eos_raw


def _validate_stop_sequences(d: dict[str, JSONValue]) -> list[str]:
    """Validate stop_sequences field."""
    stop_sequences_raw = d.get("stop_sequences")
    if stop_sequences_raw is None:
        return []
    if not isinstance(stop_sequences_raw, list):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="stop_sequences must be a list",
            http_status=422,
        )
    result: list[str] = []
    for i, seq in enumerate(stop_sequences_raw):
        if not isinstance(seq, str):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=f"stop_sequences[{i}] must be a string",
                http_status=422,
            )
        result.append(seq)
    return result


def _decode_generate_request(obj: JSONValue) -> GenerateRequest:
    """Decode and validate a generate request."""
    d = load_json_dict(obj)

    # prompt_text and prompt_path are mutually exclusive
    prompt_text_raw = d.get("prompt_text")
    prompt_path_raw = d.get("prompt_path")

    prompt_text: str | None = None
    prompt_path: str | None = None

    if prompt_text_raw is not None and prompt_path_raw is not None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="prompt_text and prompt_path are mutually exclusive",
            http_status=422,
        )
    if prompt_text_raw is None and prompt_path_raw is None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="either prompt_text or prompt_path must be provided",
            http_status=422,
        )

    if prompt_text_raw is not None:
        prompt_text = validate_str(prompt_text_raw, "prompt_text")
    if prompt_path_raw is not None:
        prompt_path = validate_str(prompt_path_raw, "prompt_path")

    max_new_tokens = validate_int_range(
        d.get("max_new_tokens"), "max_new_tokens", ge=1, le=1024, default=64
    )
    temperature = validate_float_range(
        d.get("temperature"), "temperature", ge=0.0, le=2.0, default=1.0
    )
    top_k = validate_int_range(d.get("top_k"), "top_k", ge=0, default=50)
    top_p = validate_float_range(d.get("top_p"), "top_p", ge=0.0, le=1.0, default=1.0)
    stop_on_eos = _validate_stop_on_eos(d)
    stop_sequences = _validate_stop_sequences(d)

    seed_raw = d.get("seed")
    seed: int | None = None
    if seed_raw is not None:
        seed = validate_int_range(seed_raw, "seed", ge=0)

    num_return_sequences = validate_int_range(
        d.get("num_return_sequences"), "num_return_sequences", ge=1, le=16, default=1
    )

    return {
        "prompt_text": prompt_text,
        "prompt_path": prompt_path,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_on_eos": stop_on_eos,
        "stop_sequences": stop_sequences,
        "seed": seed,
        "num_return_sequences": num_return_sequences,
    }


def _decode_chat_request(obj: JSONValue) -> ChatRequest:
    """Decode and validate a chat request."""
    d = load_json_dict(obj)

    message = validate_str(d.get("message"), "message")

    session_id_raw = d.get("session_id")
    session_id: str | None = None
    if session_id_raw is not None:
        session_id = validate_str(session_id_raw, "session_id")

    max_new_tokens = validate_int_range(
        d.get("max_new_tokens"), "max_new_tokens", ge=1, le=1024, default=128
    )
    temperature = validate_float_range(
        d.get("temperature"), "temperature", ge=0.0, le=2.0, default=0.8
    )
    top_k = validate_int_range(d.get("top_k"), "top_k", ge=0, default=50)
    top_p = validate_float_range(d.get("top_p"), "top_p", ge=0.0, le=1.0, default=0.95)

    return {
        "message": message,
        "session_id": session_id,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }


__all__ = [
    "_decode_baseline_cloze_request",
    "_decode_chat_request",
    "_decode_evaluate_request",
    "_decode_generate_request",
    "_decode_score_request",
    "_decode_train_request",
]
