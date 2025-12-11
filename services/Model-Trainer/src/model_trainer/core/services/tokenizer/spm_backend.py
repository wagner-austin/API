from __future__ import annotations

import os
import time
from pathlib import Path

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.logging import get_logger
from platform_ml import sentencepiece as spm
from typing_extensions import TypedDict

from ... import _test_hooks
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


def _spm_train(files: list[str], *, model_prefix: str, vocab_size: int) -> None:
    """Train a SentencePiece model using the Python API."""
    spm.train(files, model_prefix=model_prefix, vocab_size=vocab_size)


def _spm_encode_ids(model_path: str, text: str) -> list[int]:
    """Encode text to token IDs using the Python API."""
    return spm.encode_ids(model_path, text)


def _spm_decode_ids(model_path: str, ids: list[int]) -> str:
    """Decode token IDs to text using the Python API."""
    return spm.decode_ids(model_path, ids)


class TokenizerStats(TypedDict, total=True):
    coverage: float
    oov_rate: float
    token_count: int
    char_coverage: float


def train_spm_tokenizer(
    corpus_path: str, out_dir: str, cfg: _TokenizerTrainConfig
) -> TokenizerStats:
    spm.require_module()
    os.makedirs(out_dir, exist_ok=True)
    files = list_text_files(corpus_path)
    if not files:
        raise AppError(
            ModelTrainerErrorCode.CORPUS_EMPTY,
            f"No text files found under {corpus_path}",
            model_trainer_status_for(ModelTrainerErrorCode.CORPUS_EMPTY),
        )

    model_prefix = str(Path(out_dir) / "tokenizer")
    _test_hooks.spm_train(files, model_prefix=model_prefix, vocab_size=cfg.vocab_size)

    model_path = f"{model_prefix}.model"
    total = count_lines(files)
    holdout_n = max(1, int(total * cfg.holdout_fraction))
    if cfg.sample_max_lines is not None and cfg.sample_max_lines > 0:
        holdout_n = min(holdout_n, int(cfg.sample_max_lines))
    sample = sample_lines(files, holdout_n, seed=cfg.seed)

    total_tokens = 0
    unk_tokens = 0
    # SentencePiece uses 0 as UNK id by default in CLI models.
    # We assume UNK=0 here based on the generated vocab ordering.
    unk_id = 0
    for s in sample:
        ids = _test_hooks.spm_encode_ids(model_path, s)
        total_tokens += len(ids)
        unk_tokens += ids.count(unk_id)
    coverage = 1.0 if total_tokens == 0 else max(0.0, 1.0 - (unk_tokens / max(1, total_tokens)))

    uniq_chars = set("".join(sample))
    covered_chars = 0
    for ch in uniq_chars:
        ids = _test_hooks.spm_encode_ids(model_path, ch)
        if ids and any(tid != unk_id for tid in ids):
            covered_chars += 1
    char_cov = 1.0 if len(uniq_chars) == 0 else max(0.0, min(1.0, covered_chars / len(uniq_chars)))

    stats: TokenizerStats = {
        "coverage": coverage,
        "oov_rate": (unk_tokens / max(1, total_tokens)),
        "token_count": total_tokens,
        "char_coverage": char_cov,
    }
    manifest: dict[str, JSONValue] = {
        "created_at": int(time.time()),
        "config": {
            "vocab_size": cfg.vocab_size,
            "min_frequency": cfg.min_frequency,
            "holdout_fraction": cfg.holdout_fraction,
            "seed": cfg.seed,
            "special_tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            "method": "sentencepiece",
        },
        "stats": {
            "coverage": float(stats["coverage"]),
            "oov_rate": float(stats["oov_rate"]),
            "token_count": int(stats["token_count"]),
            "char_coverage": float(stats["char_coverage"]),
        },
    }
    Path(out_dir).joinpath("manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")
    return stats


class _SPMAdapter:
    def __init__(self: _SPMAdapter, model_path: str) -> None:
        self._model = model_path
        # Build piece->id map from vocab file
        vocab_path = str(Path(model_path).with_suffix(".vocab"))
        table: dict[str, int] = {}
        try:
            with open(vocab_path, encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    piece = line.split("\t", 1)[0].strip()
                    if piece:
                        table[piece] = i
        except OSError as e:
            get_logger(__name__).warning(
                "Failed to read SentencePiece vocab file %s: %s", vocab_path, e
            )
            table = {}
        self._vocab = table

    def encode(self: _SPMAdapter, text: str) -> list[int]:
        return _test_hooks.spm_encode_ids(self._model, text)

    def decode(self: _SPMAdapter, ids: list[int]) -> str:
        return _test_hooks.spm_decode_ids(self._model, ids)

    def token_to_id(self: _SPMAdapter, token: str) -> int | None:
        return self._vocab.get(token)

    def get_vocab_size(self: _SPMAdapter) -> int:
        return len(self._vocab)


class SentencePieceBackend(_TokenizerBackendProto):
    def name(self: SentencePieceBackend) -> str:
        return "sentencepiece"

    def train(self: SentencePieceBackend, cfg: _TokenizerTrainConfig) -> _TokenizerTrainStats:
        stats = train_spm_tokenizer(
            corpus_path=cfg.corpus_path,
            out_dir=cfg.out_dir,
            cfg=cfg,
        )
        return _TokenizerTrainStats(
            coverage=stats["coverage"],
            oov_rate=stats["oov_rate"],
            token_count=stats["token_count"],
            char_coverage=stats["char_coverage"],
        )

    def load(self: SentencePieceBackend, artifact_path: str) -> _TokenizerHandle:
        return _SPMAdapter(artifact_path)

    def encode(self: SentencePieceBackend, handle: _TokenizerHandle, text: str) -> list[int]:
        return handle.encode(text)

    def decode(self: SentencePieceBackend, handle: _TokenizerHandle, ids: list[int]) -> str:
        return handle.decode(ids)

    def inspect(self: SentencePieceBackend, artifact_path: str) -> _TokenizerTrainStats:
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
        from platform_core.json_utils import load_json_str

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
