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

from ..schemas.runs import ChatRequest, EvaluateRequest, GenerateRequest, ScoreRequest, TrainRequest

_MODEL_FAMILIES: frozenset[str] = frozenset({"gpt2", "llama", "qwen", "char_lstm"})
_OPTIMIZERS: frozenset[str] = frozenset({"adamw", "adam", "sgd"})
_DEVICES: frozenset[str] = frozenset({"cpu", "cuda", "auto"})
_PRECISIONS: frozenset[str] = frozenset({"fp32", "fp16", "bf16", "auto"})
_SPLITS: frozenset[str] = frozenset({"validation", "test"})
_DETAIL_LEVELS: frozenset[str] = frozenset({"summary", "per_char"})
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


def _narrow_model_family(
    raw: str | None,
) -> Literal["gpt2", "llama", "qwen", "char_lstm"]:
    """Narrow model family string to Literal type."""
    val = raw if raw is not None else "gpt2"
    if val == "gpt2":
        return "gpt2"
    if val == "llama":
        return "llama"
    if val == "qwen":
        return "qwen"
    return "char_lstm"


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
    tokenizer_id = validate_str(d.get("tokenizer_id"), "tokenizer_id", default="")
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

    # Optional data loader knobs: accept or leave None; worker resolves defaults by device
    data_num_workers = _decode_optional_int_ge(d, "data_num_workers", ge=0)
    data_pin_memory = _decode_optional_bool(d, "data_pin_memory")

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
    "_decode_chat_request",
    "_decode_evaluate_request",
    "_decode_generate_request",
    "_decode_score_request",
    "_decode_train_request",
]
