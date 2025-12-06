from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class LoggingExtra(TypedDict, total=False):
    # Core logging context
    category: str
    service: str
    event: str
    run_id: str
    tokenizer_id: str
    error_code: str
    # API and orchestrator fields
    model_family: str
    model_size: str
    split: str
    kind: Literal["tokenizers", "models"]
    item_id: str
    count: int
    path: str
    status: str
    loss: float
    perplexity: float
    steps: int
    tail: int
    method: str
    vocab_size: int
    reason: str
    # Corpus fetcher fields
    file_id: str
    api_url: str
    url: str
    expected_size: int
    actual_size: int
    size: int
    resume_from: int
    elapsed_seconds: float


# Strict list of keys to include when rendering JSON logs.
# Avoids inspecting __annotations__ (dict[str, Any]) to keep typing precise.
LOGGING_EXTRA_FIELDS: tuple[str, ...] = (
    "category",
    "service",
    "event",
    "run_id",
    "tokenizer_id",
    "error_code",
    "model_family",
    "model_size",
    "split",
    "kind",
    "item_id",
    "count",
    "path",
    "status",
    "loss",
    "perplexity",
    "steps",
    "tail",
    "method",
    "vocab_size",
    "reason",
    "file_id",
    "api_url",
    "url",
    "expected_size",
    "actual_size",
    "size",
    "resume_from",
    "elapsed_seconds",
)

__all__ = ["LOGGING_EXTRA_FIELDS", "LoggingExtra"]
