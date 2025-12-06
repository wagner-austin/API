from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter
from platform_core.errors import AppError
from platform_core.logging import get_logger
from platform_core.request_context import request_id_var

from ..events import publish_completed, publish_failed
from ..service import TranscriptService
from ..types import CaptionsPayload, STTPayload, TranscriptOut


def _require_request_id() -> str:
    request_id = request_id_var.get()
    if request_id == "":
        raise RuntimeError("RequestIdMiddleware not installed; request_id missing from context")
    return request_id


def build_captions_handler(
    service: TranscriptService,
) -> Callable[[CaptionsPayload], TranscriptOut]:
    logger = get_logger(__name__)

    def _handler(payload: CaptionsPayload) -> TranscriptOut:
        request_id = _require_request_id()
        try:
            res = service.captions(payload["url"], payload.get("preferred_langs"))
            # Emit completion event if configured
            publish_completed(request_id=request_id, user_id=0, url=res["url"], text=res["text"])
            return {"url": res["url"], "video_id": res["video_id"], "text": res["text"]}
        except AppError as e:
            logger.info("User error in captions: %s", e)
            publish_failed(request_id=request_id, user_id=0, error_kind="user", message=str(e))
            raise

    return _handler


def build_stt_handler(service: TranscriptService) -> Callable[[STTPayload], TranscriptOut]:
    logger = get_logger(__name__)

    def _handler(payload: STTPayload) -> TranscriptOut:
        request_id = _require_request_id()
        try:
            res = service.stt(payload["url"])
            publish_completed(request_id=request_id, user_id=0, url=res["url"], text=res["text"])
            return {"url": res["url"], "video_id": res["video_id"], "text": res["text"]}
        except AppError as e:
            logger.info("User error in stt: %s", e)
            publish_failed(request_id=request_id, user_id=0, error_kind="user", message=str(e))
            raise

    return _handler


def build_router(service: TranscriptService) -> APIRouter:
    router = APIRouter()

    captions: Callable[[CaptionsPayload], TranscriptOut] = build_captions_handler(service)
    stt: Callable[[STTPayload], TranscriptOut] = build_stt_handler(service)

    router.add_api_route("/v1/captions", captions, methods=["POST"])
    router.add_api_route("/v1/stt", stt, methods=["POST"])
    return router


__all__ = ["build_captions_handler", "build_router", "build_stt_handler"]
