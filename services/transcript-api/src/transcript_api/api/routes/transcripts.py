from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter
from platform_core.errors import AppError
from platform_core.logging import get_logger

from ...service import TranscriptService
from ...types import CaptionsPayload, STTPayload, TranscriptOut


def build_captions_handler(
    service: TranscriptService,
) -> Callable[[CaptionsPayload], TranscriptOut]:
    """Build handler for YouTube captions endpoint.

    Args:
        service: TranscriptService instance for fetching captions.

    Returns:
        Handler function for the captions route.
    """
    logger = get_logger(__name__)

    def _handler(payload: CaptionsPayload) -> TranscriptOut:
        try:
            res = service.captions(payload["url"], payload.get("preferred_langs"))
            return {"url": res["url"], "video_id": res["video_id"], "text": res["text"]}
        except AppError as e:
            logger.info("User error in captions: %s", e)
            raise

    return _handler


def build_stt_handler(service: TranscriptService) -> Callable[[STTPayload], TranscriptOut]:
    """Build handler for Whisper STT endpoint.

    Args:
        service: TranscriptService instance for STT transcription.

    Returns:
        Handler function for the STT route.
    """
    logger = get_logger(__name__)

    def _handler(payload: STTPayload) -> TranscriptOut:
        try:
            res = service.stt(payload["url"])
            return {"url": res["url"], "video_id": res["video_id"], "text": res["text"]}
        except AppError as e:
            logger.info("User error in stt: %s", e)
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
