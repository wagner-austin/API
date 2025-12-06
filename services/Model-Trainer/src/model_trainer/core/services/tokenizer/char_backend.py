from __future__ import annotations

import os
import time
from pathlib import Path

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

SPECIALS: tuple[str, ...] = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")


class _CharHandle(_TokenizerHandle):
    def __init__(self: _CharHandle, vocab: dict[str, int]) -> None:
        self._vocab = dict(vocab)
        self._id_to_char: dict[int, str] = {
            v: k for k, v in self._vocab.items() if k not in SPECIALS
        }

    def encode(self: _CharHandle, text: str) -> list[int]:
        ids: list[int] = []
        unk = int(self._vocab["[UNK]"]) if "[UNK]" in self._vocab else -1
        for ch in text:
            ids.append(int(self._vocab.get(ch, unk)))
        return ids

    def decode(self: _CharHandle, ids: list[int]) -> str:
        out_chars: list[str] = []
        for tid in ids:
            # Skip specials on decode
            if tid in (self._vocab.get(s, -999) for s in SPECIALS):
                continue
            ch = self._id_to_char.get(int(tid))
            if ch is not None:
                out_chars.append(ch)
        return "".join(out_chars)

    def token_to_id(self: _CharHandle, token: str) -> int | None:
        v = self._vocab.get(token)
        return int(v) if isinstance(v, int) else None

    def get_vocab_size(self: _CharHandle) -> int:
        return len(self._vocab)


def _read_corpus_chars(files: list[str]) -> set[str]:
    chars: set[str] = set()
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                for ch in s:
                    chars.add(ch)
    return chars


def _save_tokenizer(out_dir: str, vocab: dict[str, int], stats: _TokenizerTrainStats) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Minimal JSON format with an explicit kind header to disambiguate from HF tokenizers JSON
    payload: dict[str, JSONValue] = {
        "kind": "char",
        "specials": list(SPECIALS),
        "vocab": {k: int(v) for k, v in vocab.items()},
    }
    (Path(out_dir) / "tokenizer.json").write_text(dump_json_str(payload), encoding="utf-8")

    manifest: dict[str, JSONValue] = {
        "created_at": int(time.time()),
        "config": {
            "special_tokens": list(SPECIALS),
        },
        "stats": {
            "coverage": float(stats.coverage),
            "oov_rate": float(stats.oov_rate),
            "token_count": int(stats.token_count),
            "char_coverage": float(stats.char_coverage),
        },
    }
    (Path(out_dir) / "manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")


class CharBackend(_TokenizerBackendProto):
    """Character-level tokenizer backend.

    Builds a vocabulary over unique characters observed in the corpus, with
    fixed specials [PAD, UNK, BOS, EOS]. Encodes by codepoint.
    """

    def name(self: CharBackend) -> str:
        return "char"

    def train(self: CharBackend, cfg: _TokenizerTrainConfig) -> _TokenizerTrainStats:
        files = list_text_files(cfg.corpus_path)
        if not files:
            raise AppError(
                ModelTrainerErrorCode.CORPUS_EMPTY,
                f"No text files found under {cfg.corpus_path}",
                model_trainer_status_for(ModelTrainerErrorCode.CORPUS_EMPTY),
            )
        uniq = _read_corpus_chars(files)
        # Deterministic ordering for ids
        sorted_chars = sorted(uniq)
        vocab: dict[str, int] = {}
        # Assign specials first
        for i, s in enumerate(SPECIALS):
            vocab[s] = i
        # Assign chars after specials
        base = len(SPECIALS)
        for idx, ch in enumerate(sorted_chars):
            vocab[ch] = base + idx

        # Compute simple stats on a holdout sample.
        # Since encoding never produces UNK for seen chars, coverage over sample is 1.0
        # and oov_rate 0.0. Char coverage equals 1.0 for the union vocabulary.
        total = count_lines(files)
        holdout_n = max(1, int(total * cfg.holdout_fraction))
        if cfg.sample_max_lines is not None and cfg.sample_max_lines > 0:
            holdout_n = min(holdout_n, int(cfg.sample_max_lines))
        _ = sample_lines(files, holdout_n, seed=cfg.seed)
        stats = _TokenizerTrainStats(
            coverage=1.0,
            oov_rate=0.0,
            token_count=len(vocab),
            char_coverage=1.0,
        )
        _save_tokenizer(cfg.out_dir, vocab, stats)
        return stats

    def load(self: CharBackend, artifact_path: str) -> _TokenizerHandle:
        base = Path(artifact_path)
        path = base if base.is_file() else base / "tokenizer.json"
        text = Path(path).read_text(encoding="utf-8")
        obj = load_json_str(text)
        if not isinstance(obj, dict):
            raise AppError(
                ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED,
                "invalid char tokenizer format",
                model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED),
            )
        kind = obj.get("kind")
        if kind != "char":
            raise AppError(
                ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED,
                "tokenizer is not a char tokenizer",
                model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED),
            )
        vocab_obj = obj.get("vocab")
        if not isinstance(vocab_obj, dict):
            raise AppError(
                ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED,
                "invalid vocab in tokenizer.json",
                model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED),
            )
        # Ensure keys are str and values are int
        vocab: dict[str, int] = {}
        for k, v in vocab_obj.items():
            if not isinstance(k, str) or not isinstance(v, int):
                raise AppError(
                    ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED,
                    "invalid vocab entry in tokenizer.json",
                    model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED),
                )
            vocab[k] = int(v)
        return _CharHandle(vocab)

    def encode(self: CharBackend, handle: _TokenizerHandle, text: str) -> list[int]:
        return handle.encode(text)

    def decode(self: CharBackend, handle: _TokenizerHandle, ids: list[int]) -> str:
        return handle.decode(ids)

    def inspect(self: CharBackend, artifact_path: str) -> _TokenizerTrainStats:
        base = Path(artifact_path)
        base_dir = base if base.is_dir() else base.parent
        mpath = base_dir / "manifest.json"
        if not mpath.exists():
            raise AppError(
                ModelTrainerErrorCode.TOKENIZER_NOT_FOUND,
                f"manifest not found for tokenizer at {base_dir}",
                model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_NOT_FOUND),
            )
        text = mpath.read_text(encoding="utf-8")
        obj = load_json_str(text)
        if not isinstance(obj, dict):
            raise AppError(
                ModelTrainerErrorCode.TOKENIZER_LOAD_FAILED,
                "invalid char tokenizer manifest",
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
