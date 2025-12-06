from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class TrainRequestPayload(TypedDict):
    """Payload for training job request."""

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
    device: Literal["cpu", "cuda", "auto"]
    precision: Literal["fp32", "fp16", "bf16", "auto"]
    data_num_workers: int | None
    data_pin_memory: bool | None
    early_stopping_patience: int
    test_split_ratio: float
    finetune_lr_cap: float


class TrainJobPayload(TypedDict):
    run_id: str
    request: TrainRequestPayload
    user_id: int


class EvalJobPayload(TypedDict):
    run_id: str
    split: str
    path_override: str | None


class TokenizerTrainPayload(TypedDict):
    tokenizer_id: str
    method: Literal["bpe", "sentencepiece", "char"]
    vocab_size: int
    min_frequency: int
    corpus_file_id: str
    holdout_fraction: float
    seed: int


class ScoreJobPayload(TypedDict):
    """Payload for score inference job."""

    run_id: str
    request_id: str
    text: str | None
    path: str | None
    detail_level: Literal["summary", "per_char"]
    top_k: int | None
    seed: int | None


class GenerateJobPayload(TypedDict):
    """Payload for generate inference job."""

    run_id: str
    request_id: str
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
