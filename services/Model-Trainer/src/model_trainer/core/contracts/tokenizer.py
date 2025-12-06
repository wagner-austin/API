from __future__ import annotations

from typing import Protocol

from platform_core.json_utils import dump_json_str


class TokenizerTrainConfig:
    method: str
    vocab_size: int
    min_frequency: int
    corpus_path: str
    holdout_fraction: float
    seed: int
    out_dir: str
    sample_max_lines: int | None

    def __init__(
        self: TokenizerTrainConfig,
        *,
        method: str,
        vocab_size: int,
        min_frequency: int,
        corpus_path: str,
        holdout_fraction: float,
        seed: int,
        out_dir: str,
        sample_max_lines: int | None = None,
    ) -> None:
        self.method = method
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.corpus_path = corpus_path
        self.holdout_fraction = holdout_fraction
        self.seed = seed
        self.out_dir = out_dir
        self.sample_max_lines = sample_max_lines


class TokenizerTrainStats:
    coverage: float
    oov_rate: float
    token_count: int
    char_coverage: float

    def __init__(
        self: TokenizerTrainStats,
        *,
        coverage: float,
        oov_rate: float,
        token_count: int,
        char_coverage: float,
    ) -> None:
        self.coverage = coverage
        self.oov_rate = oov_rate
        self.token_count = token_count
        self.char_coverage = char_coverage

    def model_dump_json(self: TokenizerTrainStats) -> str:
        payload: dict[str, float | int] = {
            "coverage": float(self.coverage),
            "oov_rate": float(self.oov_rate),
            "token_count": int(self.token_count),
            "char_coverage": float(self.char_coverage),
        }
        return dump_json_str(payload)


class TokenizerHandle(Protocol):
    def encode(self: TokenizerHandle, text: str) -> list[int]: ...
    def decode(self: TokenizerHandle, ids: list[int]) -> str: ...
    def token_to_id(self: TokenizerHandle, token: str) -> int | None: ...
    def get_vocab_size(self: TokenizerHandle) -> int: ...


class TokenizerBackend(Protocol):
    def name(self: TokenizerBackend) -> str: ...
    def train(self: TokenizerBackend, cfg: TokenizerTrainConfig) -> TokenizerTrainStats: ...
    def load(self: TokenizerBackend, artifact_path: str) -> TokenizerHandle: ...
    def encode(self: TokenizerBackend, handle: TokenizerHandle, text: str) -> list[int]: ...
    def decode(self: TokenizerBackend, handle: TokenizerHandle, ids: list[int]) -> str: ...
    def inspect(self: TokenizerBackend, artifact_path: str) -> TokenizerTrainStats: ...
