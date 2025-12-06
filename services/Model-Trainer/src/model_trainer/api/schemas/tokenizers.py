from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class TokenizerTrainRequest(TypedDict, total=True):
    method: Literal["bpe", "sentencepiece", "char"]
    vocab_size: int
    min_frequency: int
    corpus_file_id: str
    holdout_fraction: float
    seed: int


class TokenizerTrainResponse(TypedDict, total=True):
    tokenizer_id: str
    artifact_path: str
    coverage: float | None
    oov_rate: float | None


class TokenizerInfoResponse(TypedDict, total=True):
    tokenizer_id: str
    artifact_path: str
    status: str
    coverage: float | None
    oov_rate: float | None
    token_count: int | None
    char_coverage: float | None
