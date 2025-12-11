from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.params import Depends as DependsParamType
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger

from ...core import _test_hooks
from ...core.logging.types import LoggingExtra
from ...core.services.container import ServiceContainer
from ..middleware import api_key_dependency
from ..schemas.tokenizers import (
    TokenizerInfoResponse,
    TokenizerTrainRequest,
    TokenizerTrainResponse,
)
from ..validators.tokenizers import _decode_tokenizer_train_request

_logger = get_logger(__name__)


def build_router(container: ServiceContainer) -> APIRouter:
    api_dep: DependsParamType = Depends(api_key_dependency(container.settings))
    router = APIRouter(dependencies=[api_dep])

    async def start_tokenizer_training(request: Request) -> TokenizerTrainResponse:
        raw_body = await request.body()
        from platform_core.json_utils import load_json_str

        body = load_json_str(raw_body.decode("utf-8"))
        req: TokenizerTrainRequest = _decode_tokenizer_train_request(body)
        orchestrator = container.tokenizer_orchestrator
        extra: LoggingExtra = {
            "category": "api",
            "service": "tokenizers",
            "event": "tokenizers_enqueue",
            "method": req["method"],
            "vocab_size": req["vocab_size"],
        }
        _logger.info("tokenizers enqueue", extra=extra)
        # Allow test hook to override orchestrator behavior
        hook = _test_hooks.tokenizer_enqueue_hook
        out = hook(orchestrator, req) if hook is not None else orchestrator.enqueue_training(req)
        if out is None:
            raise AppError(
                ModelTrainerErrorCode.TOKENIZER_TRAIN_FAILED,
                "tokenizer training enqueue failed",
                model_trainer_status_for(ModelTrainerErrorCode.TOKENIZER_TRAIN_FAILED),
            )
        return out

    def get_tokenizer(tokenizer_id: str) -> TokenizerInfoResponse:
        r = container.redis
        status = r.get(f"tokenizer:{tokenizer_id}:status") or "unknown"
        stats_json = r.get(f"tokenizer:{tokenizer_id}:stats")
        artifact_path = f"{container.settings['app']['artifacts_root']}/tokenizers/{tokenizer_id}"
        extra2: LoggingExtra = {
            "category": "api",
            "service": "tokenizers",
            "tokenizer_id": tokenizer_id,
            "event": "tokenizers_get",
            "status": status,
        }
        _logger.info("tokenizers get", extra=extra2)
        coverage = None
        oov_rate = None
        token_count = None
        char_coverage = None
        if stats_json:
            from platform_core.json_utils import load_json_str

            obj = load_json_str(stats_json)
            if isinstance(obj, dict):
                cov_v = obj.get("coverage")
                oov_v = obj.get("oov_rate")
                tok_v = obj.get("token_count")
                ch_v = obj.get("char_coverage")
                coverage = float(cov_v) if isinstance(cov_v, int | float) else None
                oov_rate = float(oov_v) if isinstance(oov_v, int | float) else None
                token_count = int(tok_v) if isinstance(tok_v, int) else None
                char_coverage = float(ch_v) if isinstance(ch_v, int | float) else None
        return {
            "tokenizer_id": tokenizer_id,
            "artifact_path": artifact_path,
            "status": status,
            "coverage": coverage,
            "oov_rate": oov_rate,
            "token_count": token_count,
            "char_coverage": char_coverage,
        }

    router.add_api_route("/train", start_tokenizer_training, methods=["POST"])
    router.add_api_route("/{tokenizer_id}", get_tokenizer, methods=["GET"])
    return router
