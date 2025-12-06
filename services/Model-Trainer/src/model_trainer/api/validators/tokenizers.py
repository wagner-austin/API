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

from ..schemas.tokenizers import TokenizerTrainRequest

_METHODS: frozenset[str] = frozenset({"bpe", "sentencepiece", "char"})
_ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        "method",
        "vocab_size",
        "min_frequency",
        "corpus_file_id",
        "holdout_fraction",
        "seed",
    }
)


def _decode_tokenizer_train_request(obj: JSONValue) -> TokenizerTrainRequest:
    d = load_json_dict(obj)

    extra_fields = set(d.keys()) - _ALLOWED_FIELDS
    if extra_fields:
        extra_list = ", ".join(sorted(extra_fields))
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=f"Extra fields not allowed: {extra_list}",
            http_status=422,
        )

    method_raw = validate_optional_literal(d.get("method"), "method", _METHODS)
    method_str = method_raw if method_raw is not None else "bpe"
    method: Literal["bpe", "sentencepiece", "char"]
    if method_str == "bpe":
        method = "bpe"
    elif method_str == "sentencepiece":
        method = "sentencepiece"
    else:
        method = "char"

    vocab_size = validate_int_range(d.get("vocab_size"), "vocab_size", ge=128, default=32000)
    min_frequency = validate_int_range(d.get("min_frequency"), "min_frequency", ge=1, default=2)
    corpus_file_id = validate_str(d.get("corpus_file_id"), "corpus_file_id")
    holdout_fraction = validate_float_range(
        d.get("holdout_fraction"), "holdout_fraction", ge=0.0, le=0.5, default=0.01
    )
    seed = validate_int_range(d.get("seed"), "seed", ge=0, default=42)

    return {
        "method": method,
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "corpus_file_id": corpus_file_id,
        "holdout_fraction": holdout_fraction,
        "seed": seed,
    }


__all__ = ["_decode_tokenizer_train_request"]
