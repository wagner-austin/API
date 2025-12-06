from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Protocol

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from ...contracts.tokenizer import (
    TokenizerBackend as _TokenizerBackendProto,
)
from ...contracts.tokenizer import (
    TokenizerHandle as _TokenizerHandle,
)
from ...contracts.tokenizer import (
    TokenizerTrainConfig as _TokenizerTrainConfig,
)
from ...contracts.tokenizer import (
    TokenizerTrainStats as _TokenizerTrainStats,
)
from ..data.corpus import count_lines, list_text_files, sample_lines

DEFAULT_SPECIALS: tuple[str, ...] = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")


class _HFTokenizerProto(Protocol):
    """Protocol for HuggingFace tokenizers.Tokenizer instance."""

    pre_tokenizer: _PreTokenizerProto

    def train_from_iterator(
        self,
        iterator: list[str],
        trainer: _BpeTrainerProto,
    ) -> None: ...

    def save(self, path: str) -> None: ...

    def encode(self, text: str) -> _EncodingProto: ...

    def decode(self, ids: list[int]) -> str: ...

    def token_to_id(self, token: str) -> int | None: ...

    def get_vocab_size(self) -> int: ...


class _TokenizerClassProto(Protocol):
    """Protocol for tokenizers.Tokenizer class (not instance).

    Represents the Tokenizer class itself, which can be called to create
    instances or used to load from file.
    """

    def __call__(self, model: _BpeModelProto) -> _HFTokenizerProto: ...

    def from_file(self, path: str) -> _HFTokenizerProto: ...


class _EncodingProto(Protocol):
    """Protocol for HuggingFace tokenizers.Encoding."""

    @property
    def ids(self) -> list[int]: ...


class _BpeTrainerProto(Protocol):
    """Protocol for HuggingFace tokenizers.trainers.BpeTrainer."""

    pass


class _BpeModelProto(Protocol):
    """Protocol for HuggingFace tokenizers.models.BPE."""

    pass


class _BpeTrainerCtorProto(Protocol):
    """Protocol for tokenizers.trainers.BpeTrainer constructor."""

    def __call__(
        self,
        *,
        vocab_size: int,
        min_frequency: int,
        special_tokens: list[str],
    ) -> _BpeTrainerProto: ...


class _BpeModelCtorProto(Protocol):
    """Protocol for tokenizers.models.BPE constructor."""

    def __call__(self, *, unk_token: str) -> _BpeModelProto: ...


class _PreTokenizerProto(Protocol):
    """Protocol for pre-tokenizer."""

    pass


def _get_tokenizer_class() -> _TokenizerClassProto:
    """Get tokenizers.Tokenizer class via dynamic import."""
    tokenizers_mod = __import__("tokenizers", fromlist=["Tokenizer"])
    cls: _TokenizerClassProto = tokenizers_mod.Tokenizer
    return cls


def _get_bpe_model_ctor() -> _BpeModelCtorProto:
    """Get tokenizers.models.BPE constructor via dynamic import."""
    models_mod = __import__("tokenizers.models", fromlist=["BPE"])
    ctor: _BpeModelCtorProto = models_mod.BPE
    return ctor


def _get_bpe_trainer_ctor() -> _BpeTrainerCtorProto:
    """Get tokenizers.trainers.BpeTrainer constructor via dynamic import."""
    trainers_mod = __import__("tokenizers.trainers", fromlist=["BpeTrainer"])
    ctor: _BpeTrainerCtorProto = trainers_mod.BpeTrainer
    return ctor


def _get_whitespace_pretokenizer() -> _PreTokenizerProto:
    """Get tokenizers.pre_tokenizers.Whitespace instance via dynamic import."""
    pre_tok_mod = __import__("tokenizers.pre_tokenizers", fromlist=["Whitespace"])
    cls: type[_PreTokenizerProto] = pre_tok_mod.Whitespace
    return cls()


def _load_tokenizer_from_file(path: str) -> _HFTokenizerProto:
    """Load tokenizer from file using tokenizers.Tokenizer.from_file()."""
    tokenizer_cls = _get_tokenizer_class()
    return tokenizer_cls.from_file(path)


def _read_corpus_lines(files: list[str]) -> list[str]:
    """Read all lines from corpus files."""
    lines: list[str] = []
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
    return lines


def train_bpe_tokenizer(
    corpus_path: str, out_dir: str, cfg: _TokenizerTrainConfig
) -> _TokenizerTrainStats:
    """Train a real BPE tokenizer using HuggingFace tokenizers library."""
    os.makedirs(out_dir, exist_ok=True)
    files = list_text_files(corpus_path)
    if not files:
        raise AppError(
            ModelTrainerErrorCode.CORPUS_EMPTY,
            f"No text files found under {corpus_path}",
            model_trainer_status_for(ModelTrainerErrorCode.CORPUS_EMPTY),
        )

    # Read corpus
    corpus_lines = _read_corpus_lines(files)

    # Create BPE tokenizer with real algorithm
    tokenizer_cls = _get_tokenizer_class()
    bpe_model_ctor = _get_bpe_model_ctor()
    bpe_trainer_ctor = _get_bpe_trainer_ctor()

    # Initialize tokenizer with BPE model
    tokenizer: _HFTokenizerProto = tokenizer_cls(bpe_model_ctor(unk_token="[UNK]"))

    # Set pre-tokenizer (splits on whitespace before BPE)
    tokenizer.pre_tokenizer = _get_whitespace_pretokenizer()

    # Create trainer with configuration
    trainer: _BpeTrainerProto = bpe_trainer_ctor(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=list(DEFAULT_SPECIALS),
    )

    # Train tokenizer on corpus
    tokenizer.train_from_iterator(corpus_lines, trainer)

    # Save tokenizer
    tokenizer_path = str(Path(out_dir) / "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # Compute stats on holdout
    stats = _compute_stats(cfg, files, tokenizer)

    # Write manifest
    manifest: dict[str, JSONValue] = {
        "created_at": int(time.time()),
        "config": {
            "vocab_size": cfg.vocab_size,
            "min_frequency": cfg.min_frequency,
            "holdout_fraction": cfg.holdout_fraction,
            "seed": cfg.seed,
            "special_tokens": list(DEFAULT_SPECIALS),
        },
        "stats": {
            "coverage": float(stats.coverage),
            "oov_rate": float(stats.oov_rate),
            "token_count": int(stats.token_count),
            "char_coverage": float(stats.char_coverage),
        },
    }
    (Path(out_dir) / "manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")

    return _TokenizerTrainStats(
        coverage=stats.coverage,
        oov_rate=stats.oov_rate,
        token_count=stats.token_count,
        char_coverage=stats.char_coverage,
    )


def _compute_stats(
    cfg: _TokenizerTrainConfig,
    files: list[str],
    tokenizer: _HFTokenizerProto,
) -> _TokenizerTrainStats:
    """Compute tokenization stats on holdout sample."""
    total = count_lines(files)
    holdout_n = max(1, int(total * cfg.holdout_fraction))
    if cfg.sample_max_lines is not None and cfg.sample_max_lines > 0:
        holdout_n = min(holdout_n, int(cfg.sample_max_lines))
    sample = sample_lines(files, holdout_n, seed=cfg.seed)

    unk_id = tokenizer.token_to_id("[UNK]")
    # BPE tokenizers trained by this backend always have [UNK] in DEFAULT_SPECIALS
    if unk_id is None:
        raise AppError(
            ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED,
            "tokenizer missing required [UNK] special token",
            model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED),
        )

    total_tokens = 0
    unk_tokens = 0

    for line in sample:
        encoding = tokenizer.encode(line)
        ids = encoding.ids
        total_tokens += len(ids)
        unk_tokens += ids.count(unk_id)

    coverage = 1.0 if total_tokens == 0 else max(0.0, 1.0 - (unk_tokens / max(1, total_tokens)))
    oov_rate = unk_tokens / max(1, total_tokens)

    # Character coverage: check what fraction of unique chars can be tokenized without UNK
    uniq_chars = set("".join(sample))
    covered_chars = 0
    for ch in uniq_chars:
        encoding = tokenizer.encode(ch)
        ids = encoding.ids
        if ids and all(tid != unk_id for tid in ids):
            covered_chars += 1

    denom = len(uniq_chars)
    char_coverage = 1.0 if denom == 0 else max(0.0, min(1.0, covered_chars / denom))

    return _TokenizerTrainStats(
        coverage=coverage,
        oov_rate=oov_rate,
        token_count=total_tokens,
        char_coverage=char_coverage,
    )


class _TokenizerAdapter:
    """Adapter wrapping HuggingFace tokenizer to match TokenizerHandle protocol."""

    def __init__(self: _TokenizerAdapter, tokenizer: _HFTokenizerProto) -> None:
        self._tokenizer = tokenizer

    def encode(self: _TokenizerAdapter, text: str) -> list[int]:
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self: _TokenizerAdapter, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def token_to_id(self: _TokenizerAdapter, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def get_vocab_size(self: _TokenizerAdapter) -> int:
        return self._tokenizer.get_vocab_size()


class BPEBackend(_TokenizerBackendProto):
    """BPE tokenizer backend using HuggingFace tokenizers library."""

    def name(self: BPEBackend) -> str:
        return "bpe"

    def train(self: BPEBackend, cfg: _TokenizerTrainConfig) -> _TokenizerTrainStats:
        return train_bpe_tokenizer(
            corpus_path=cfg.corpus_path,
            out_dir=cfg.out_dir,
            cfg=cfg,
        )

    def load(self: BPEBackend, artifact_path: str) -> _TokenizerHandle:
        base = Path(artifact_path)
        path = base if base.is_file() else base / "tokenizer.json"
        tokenizer = _load_tokenizer_from_file(str(path))
        return _TokenizerAdapter(tokenizer)

    def encode(self: BPEBackend, handle: _TokenizerHandle, text: str) -> list[int]:
        return handle.encode(text)

    def decode(self: BPEBackend, handle: _TokenizerHandle, ids: list[int]) -> str:
        return handle.decode(ids)

    def inspect(self: BPEBackend, artifact_path: str) -> _TokenizerTrainStats:
        base = Path(artifact_path)
        base_dir = base if base.is_dir() else base.parent
        manifest_path = base_dir / "manifest.json"
        if not manifest_path.exists():
            raise AppError(
                ModelTrainerErrorCode.TOKENIZER_NOT_FOUND,
                f"manifest not found for tokenizer at {base_dir}",
                model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_NOT_FOUND),
            )

        text = manifest_path.read_text(encoding="utf-8")
        obj = load_json_str(text)
        if not isinstance(obj, dict):
            raise AppError(
                ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED,
                "invalid manifest format",
                model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED),
            )

        stats_obj: JSONValue = obj.get("stats")
        if not isinstance(stats_obj, dict):
            raise AppError(
                ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED,
                "invalid stats in manifest",
                model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED),
            )

        cov_v: JSONValue = stats_obj.get("coverage")
        oov_v: JSONValue = stats_obj.get("oov_rate")
        tok_v: JSONValue = stats_obj.get("token_count")
        ch_v: JSONValue = stats_obj.get("char_coverage")

        coverage = float(cov_v) if isinstance(cov_v, int | float) else 0.0
        oov_rate = float(oov_v) if isinstance(oov_v, int | float) else 0.0
        token_count = int(tok_v) if isinstance(tok_v, int) else 0
        char_coverage = float(ch_v) if isinstance(ch_v, int | float) else 0.0

        return _TokenizerTrainStats(
            coverage=coverage,
            oov_rate=oov_rate,
            token_count=token_count,
            char_coverage=char_coverage,
        )
