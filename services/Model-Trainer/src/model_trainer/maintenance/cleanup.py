from __future__ import annotations

from platform_core.logging import get_logger, setup_logging

from model_trainer.core.config.settings import Settings, load_settings
from model_trainer.core.logging.types import LOGGING_EXTRA_FIELDS
from model_trainer.core.logging.utils import narrow_log_level
from model_trainer.core.services.data.corpus_cache_cleanup import (
    CorpusCacheCleanupResult,
    CorpusCacheCleanupService,
)
from model_trainer.core.services.tokenizer.tokenizer_cleanup import (
    TokenizerCleanupResult,
    TokenizerCleanupService,
)


def _init_logging(service_name: str, settings: Settings) -> None:
    level = narrow_log_level(settings["logging"]["level"])
    setup_logging(
        level=level,
        format_mode="json",
        service_name=service_name,
        instance_id=None,
        extra_fields=list(LOGGING_EXTRA_FIELDS),
    )


def run_corpus_cleanup(settings: Settings | None = None) -> CorpusCacheCleanupResult:
    cfg = settings or load_settings()
    _init_logging("model-trainer-cleanup", cfg)
    logger = get_logger(__name__)
    service = CorpusCacheCleanupService(settings=cfg)
    logger.info(
        "Corpus cache cleanup starting",
        extra={"event": "corpus_cache_cleanup_start"},
    )
    result = service.clean()
    logger.info(
        "Corpus cache cleanup completed",
        extra={
            "event": "corpus_cache_cleanup_completed",
            "deleted_files": result.deleted_files,
            "bytes_freed": result.bytes_freed,
        },
    )
    return result


def run_tokenizer_cleanup(settings: Settings | None = None) -> TokenizerCleanupResult:
    cfg = settings or load_settings()
    _init_logging("model-trainer-cleanup", cfg)
    logger = get_logger(__name__)
    service = TokenizerCleanupService(settings=cfg)
    logger.info(
        "Tokenizer cleanup starting",
        extra={"event": "tokenizer_cleanup_start"},
    )
    result = service.clean()
    logger.info(
        "Tokenizer cleanup completed",
        extra={
            "event": "tokenizer_cleanup_completed",
            "deleted_tokenizers": result.deleted_tokenizers,
            "bytes_freed": result.bytes_freed,
        },
    )
    return result


__all__ = ["run_corpus_cleanup", "run_tokenizer_cleanup"]
