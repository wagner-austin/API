from __future__ import annotations

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.request_context import install_request_id_middleware
from platform_workers.redis import RedisStrProto, redis_for_kv
from typing_extensions import TypedDict

from .provider import YouTubeTranscriptClient
from .routes import health as routes_health
from .routes import jobs as routes_jobs
from .routes import transcripts as routes_transcripts
from .routes.transcripts import build_captions_handler, build_stt_handler
from .service import Clients, Config, TranscriptService
from .stt_provider import ProbeDownloadClient, STTClient
from .types import CaptionsPayload, STTPayload, TranscriptOut


class AppDeps(TypedDict):
    config: Config
    clients: Clients


def create_app(deps: AppDeps) -> FastAPI:
    app = FastAPI(title="transcript-api", version="0.1.0")
    install_request_id_middleware(app)
    service = TranscriptService(deps["config"], deps["clients"])
    install_exception_handlers_fastapi(app, logger_name="transcript-api")

    # Include standardized route modules
    app.include_router(routes_health.build_router())
    app.include_router(routes_transcripts.build_router(service))
    app.include_router(routes_jobs.build_router())

    return app


__all__ = [
    "AppDeps",
    "CaptionsPayload",
    "Clients",
    "Config",
    "ProbeDownloadClient",
    "RedisStrProto",
    "STTClient",
    "STTPayload",
    "TranscriptOut",
    "YouTubeTranscriptClient",
    "build_captions_handler",
    "build_stt_handler",
    "create_app",
    "redis_for_kv",
]
