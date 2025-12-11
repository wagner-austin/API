from __future__ import annotations

import os
from pathlib import Path

from platform_core.logging import get_logger
from platform_workers.redis import RedisStrProto

from ..core import _test_hooks
from ..core.config.settings import Settings
from ..core.contracts.compute import LocalCPUProvider
from ..core.contracts.queue import TokenizerTrainPayload
from ..core.contracts.tokenizer import TokenizerTrainConfig, TokenizerTrainStats
from ..core.infra.paths import tokenizer_dir
from ..core.services.tokenizer.bpe_backend import BPEBackend
from ..core.services.tokenizer.char_backend import CharBackend

_logger = get_logger(__name__)


def _redis_client(settings: Settings) -> RedisStrProto:
    url = settings["redis"]["url"]
    if url.strip() == "":
        raise RuntimeError("Redis URL must not be empty")
    return _test_hooks.kv_store_factory(url)


def _has_spm_cli() -> bool:
    spm_cmds = ("spm_train", "spm_encode", "spm_decode")
    return all(_test_hooks.shutil_which(x) is not None for x in spm_cmds)


def process_tokenizer_train_job(payload: TokenizerTrainPayload) -> None:
    settings = _test_hooks.load_settings()
    r = _redis_client(settings)
    tok_id = payload["tokenizer_id"]
    r.set(f"tokenizer:{tok_id}:status", "running")
    # Apply local CPU compute environment
    threads_cfg = settings["app"]["threads"]
    threads = threads_cfg if threads_cfg and threads_cfg > 0 else max(1, int(os.cpu_count() or 1))
    env = LocalCPUProvider(threads_count=threads).env()
    for k, v in env.items():
        __import__("os").putenv(k, v)
    # Disable tokenizer internal parallelism for stable CPU usage
    __import__("os").putenv("TOKENIZERS_PARALLELISM", "1")

    out_dir = str(tokenizer_dir(settings, tok_id))
    _logger.info(
        "Tokenizer job started",
        extra={
            "category": "tokenizer",
            "service": "worker",
            "tokenizer_id": tok_id,
            "event": "tokenizer_started",
            "method": payload["method"],
            "vocab_size": int(payload["vocab_size"]),
        },
    )
    # Resolve corpus file id to local cache path
    fid = str(payload["corpus_file_id"]).strip()
    fetcher = _test_hooks.corpus_fetcher_factory(
        api_url=settings["app"]["data_bank_api_url"],
        api_key=settings["app"]["data_bank_api_key"],
        cache_dir=Path(settings["app"]["data_root"]) / "corpus_cache",
    )
    resolved_corpus = str(fetcher.fetch(fid))

    cfg = TokenizerTrainConfig(
        method=payload["method"],
        vocab_size=payload["vocab_size"],
        min_frequency=payload["min_frequency"],
        corpus_path=resolved_corpus,
        holdout_fraction=payload["holdout_fraction"],
        seed=payload["seed"],
        out_dir=out_dir,
        sample_max_lines=settings["app"]["tokenizer_sample_max_lines"],
    )

    def _log_completed(stats: TokenizerTrainStats) -> None:
        _logger.info(
            "Tokenizer training completed",
            extra={
                "category": "tokenizer",
                "service": "worker",
                "tokenizer_id": tok_id,
                "event": "tokenizer_completed",
                "vocab_size": int(payload["vocab_size"]),
                "coverage": float(stats.coverage),
                "oov_rate": float(stats.oov_rate),
                "token_count": int(stats.token_count),
                "char_coverage": float(stats.char_coverage),
            },
        )

    # Select backend by method and finalize per-branch
    if payload["method"] == "bpe":
        backend = BPEBackend()
        stats = backend.train(cfg)
        r.set(f"tokenizer:{tok_id}:status", "completed")
        from platform_core.json_utils import dump_json_str

        stats_payload = {
            "coverage": float(stats.coverage),
            "oov_rate": float(stats.oov_rate),
            "token_count": int(stats.token_count),
            "char_coverage": float(stats.char_coverage),
        }
        r.set(
            f"tokenizer:{tok_id}:stats",
            dump_json_str(stats_payload),
        )
        _log_completed(stats)
        return
    # char
    if payload["method"] == "char":
        backend_char = CharBackend()
        stats3 = backend_char.train(cfg)
        r.set(f"tokenizer:{tok_id}:status", "completed")
        from platform_core.json_utils import dump_json_str as dump_json_str3

        stats_payload3 = {
            "coverage": float(stats3.coverage),
            "oov_rate": float(stats3.oov_rate),
            "token_count": int(stats3.token_count),
            "char_coverage": float(stats3.char_coverage),
        }
        r.set(f"tokenizer:{tok_id}:stats", dump_json_str3(stats_payload3))
        _log_completed(stats3)
        return
    # sentencepiece
    if _has_spm_cli():
        from ..core.services.tokenizer.spm_backend import SentencePieceBackend

        backend_spm = SentencePieceBackend()
        stats = backend_spm.train(cfg)
        r.set(f"tokenizer:{tok_id}:status", "completed")
        from platform_core.json_utils import dump_json_str as dump_json_str2

        stats_payload2 = {
            "coverage": float(stats.coverage),
            "oov_rate": float(stats.oov_rate),
            "token_count": int(stats.token_count),
            "char_coverage": float(stats.char_coverage),
        }
        r.set(
            f"tokenizer:{tok_id}:stats",
            dump_json_str2(stats_payload2),
        )
        _log_completed(stats)
        return
    r.set(f"tokenizer:{tok_id}:status", "failed")
    _logger.error(
        "Unsupported tokenizer method: %s (SentencePiece CLI not available)",
        payload["method"],
    )
    _logger.info(
        "Tokenizer backend unavailable",
        extra={
            "category": "tokenizer",
            "service": "worker",
            "tokenizer_id": tok_id,
            "event": "tokenizer_backend_unavailable",
        },
    )
    return
