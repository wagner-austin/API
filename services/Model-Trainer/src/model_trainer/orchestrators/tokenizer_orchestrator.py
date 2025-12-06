from __future__ import annotations

import shutil

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger
from platform_workers.redis import RedisStrProto

from ..api.schemas.tokenizers import TokenizerTrainRequest, TokenizerTrainResponse
from ..core.config.settings import Settings
from ..core.contracts.queue import TokenizerTrainPayload
from ..core.services.queue.rq_adapter import RQEnqueuer

_logger = get_logger(__name__)


class TokenizerOrchestrator:
    def __init__(
        self: TokenizerOrchestrator,
        *,
        settings: Settings,
        redis_client: RedisStrProto,
        enqueuer: RQEnqueuer,
    ) -> None:
        self._settings = settings
        self._redis = redis_client
        self._enq = enqueuer

    def enqueue_training(
        self: TokenizerOrchestrator, req: TokenizerTrainRequest
    ) -> TokenizerTrainResponse | None:
        # Early validation of backend availability
        if req["method"] == "sentencepiece" and not all(
            shutil.which(x) is not None for x in ("spm_train", "spm_encode", "spm_decode")
        ):
            _logger.info(
                "tokenizer backend unavailable",
                extra={
                    "category": "orchestrator",
                    "service": "tokenizer",
                    "event": "tokenizer_backend_unavailable",
                    "method": req["method"],
                },
            )
            raise AppError(
                ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
                "sentencepiece backend unavailable",
                model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
            )
        # Pass corpus_file_id through; worker will resolve via Data Bank
        fid = req["corpus_file_id"].strip()
        if fid == "":  # should not occur due to schema min_length
            raise AppError(
                ModelTrainerErrorCode.CORPUS_NOT_FOUND,
                "corpus_file_id must be non-empty",
                model_trainer_status_for(ModelTrainerErrorCode.CORPUS_NOT_FOUND),
            )

        token_hash = abs(hash((req["method"], req["vocab_size"], fid, req["seed"]))) % (10**10)
        tokenizer_id = f"tok-{token_hash:010d}"
        self._redis.set(f"tokenizer:{tokenizer_id}:status", "queued")
        payload: TokenizerTrainPayload = {
            "tokenizer_id": tokenizer_id,
            "method": req["method"],
            "vocab_size": req["vocab_size"],
            "min_frequency": req["min_frequency"],
            "corpus_file_id": fid,
            "holdout_fraction": req["holdout_fraction"],
            "seed": req["seed"],
        }

        _ = self._enq.enqueue_tokenizer(payload)
        artifact_path = f"{self._settings['app']['artifacts_root']}/tokenizers/{tokenizer_id}"
        _logger.info(
            "tokenizer enqueued",
            extra={
                "category": "tokenizer",
                "service": "orchestrator",
                "tokenizer_id": tokenizer_id,
                "event": "enqueued",
            },
        )
        result: TokenizerTrainResponse = {
            "tokenizer_id": tokenizer_id,
            "artifact_path": artifact_path,
            "coverage": None,
            "oov_rate": None,
        }
        return result
