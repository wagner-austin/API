from __future__ import annotations

from typing import Literal, NotRequired

from typing_extensions import TypedDict


class TrainRequest(TypedDict, total=True):
    """Request to start model training."""

    model_family: Literal["gpt2", "llama", "qwen", "char_lstm"]
    model_size: str
    max_seq_len: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    corpus_file_id: str
    tokenizer_id: str
    holdout_fraction: float
    seed: int
    pretrained_run_id: str | None
    freeze_embed: bool
    gradient_clipping: float
    optimizer: Literal["adamw", "adam", "sgd"]
    user_id: int
    device: Literal["cpu", "cuda", "auto"]
    precision: Literal["fp32", "fp16", "bf16", "auto"]
    # Data loading knobs (optional at API layer; resolved in worker)
    data_num_workers: NotRequired[int | None]
    data_pin_memory: NotRequired[bool | None]
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float


class TrainResponse(TypedDict, total=True):
    run_id: str
    job_id: str


class RunStatusResponse(TypedDict, total=True):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    last_heartbeat_ts: float | None
    message: str | None


class EvaluateRequest(TypedDict, total=True):
    split: Literal["validation", "test"]
    path_override: NotRequired[str | None]


class EvaluateResponse(TypedDict, total=True):
    run_id: str
    split: str
    status: Literal["queued", "running", "completed", "failed"]
    loss: float | None
    perplexity: float | None
    artifact_path: str | None


class CancelResponse(TypedDict, total=True):
    status: Literal["cancellation-requested"]


class ArtifactPointerResponse(TypedDict, total=True):
    storage: str
    file_id: str


class ScoreRequest(TypedDict, total=True):
    """Request to score text with a trained model."""

    text: str | None
    path: str | None
    detail_level: Literal["summary", "per_char"]
    top_k: int | None
    seed: int | None


class ScoreResponse(TypedDict, total=True):
    """Response from scoring text."""

    request_id: str
    status: Literal["queued", "running", "completed", "failed"]
    loss: float | None
    perplexity: float | None
    surprisal: list[float] | None
    topk: list[list[tuple[str, float]]] | None
    tokens: list[str] | None


class GenerateRequest(TypedDict, total=True):
    """Request to generate text from a trained model."""

    prompt_text: str | None
    prompt_path: str | None
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    stop_on_eos: bool
    stop_sequences: list[str]
    seed: int | None
    num_return_sequences: int


class GenerateResponse(TypedDict, total=True):
    """Response from text generation."""

    request_id: str
    status: Literal["queued", "running", "completed", "failed"]
    outputs: list[str] | None
    steps: int | None
    eos_terminated: list[bool] | None


class ChatMessage(TypedDict, total=True):
    """A single message in a conversation."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(TypedDict, total=True):
    """Request to send a chat message."""

    message: str
    session_id: str | None
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float


class ChatResponse(TypedDict, total=True):
    """Response from chat endpoint."""

    session_id: str
    status: Literal["queued", "running", "completed", "failed"]
    request_id: str
    response: str | None


class ChatHistoryResponse(TypedDict, total=True):
    """Response containing conversation history."""

    session_id: str
    run_id: str
    messages: list[ChatMessage]
    created_at: str
