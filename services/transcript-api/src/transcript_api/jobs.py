"""STT job processing for transcript-api background workers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from platform_core.config import (
    _optional_env_str,
    _parse_bool,
    _parse_float,
    _parse_int,
    _require_env_str,
)
from platform_core.job_events import default_events_channel
from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.logging import get_logger
from platform_core.queues import TRANSCRIPT_QUEUE
from platform_workers.job_context import JobContext, make_job_context
from platform_workers.redis import RedisStrProto, redis_for_kv
from typing_extensions import TypedDict

from .cleaner import clean_segments
from .job_store import TranscriptJobStore
from .stt_provider import ProbeDownloadClient, STTClient, STTTranscriptProvider
from .types import DEFAULT_TRANSCRIPT_LANGS, LoggerProtocol, TranscriptOptions
from .youtube import canonicalize_youtube_url, extract_video_id


class STTJobParams(TypedDict):
    """Parameters for STT job processing."""

    url: str
    user_id: int


class STTJobResult(TypedDict):
    """Result of STT job processing."""

    job_id: str
    status: str
    video_id: str
    text: str


class STTConfig(TypedDict):
    """Configuration for STT processing."""

    max_video_seconds: int
    max_file_mb: int
    enable_chunking: bool
    chunk_threshold_mb: float
    target_chunk_mb: float
    max_chunk_duration_seconds: float
    max_concurrent_chunks: int
    silence_threshold_db: float
    silence_duration_seconds: float
    stt_rtf: float
    dl_mib_per_sec: float


def _load_stt_config() -> STTConfig:
    """Load STT configuration from environment variables."""
    return {
        "max_video_seconds": _parse_int("TRANSCRIPT_MAX_VIDEO_SECONDS", 0),
        "max_file_mb": _parse_int("TRANSCRIPT_MAX_FILE_MB", 0),
        "enable_chunking": _parse_bool("TRANSCRIPT_ENABLE_CHUNKING", False),
        "chunk_threshold_mb": _parse_float("TRANSCRIPT_CHUNK_THRESHOLD_MB", 25.0),
        "target_chunk_mb": _parse_float("TRANSCRIPT_TARGET_CHUNK_MB", 20.0),
        "max_chunk_duration_seconds": _parse_float("TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS", 600.0),
        "max_concurrent_chunks": _parse_int("TRANSCRIPT_MAX_CONCURRENT_CHUNKS", 4),
        "silence_threshold_db": _parse_float("TRANSCRIPT_SILENCE_THRESHOLD_DB", -40.0),
        "silence_duration_seconds": _parse_float("TRANSCRIPT_SILENCE_DURATION_SECONDS", 0.5),
        "stt_rtf": _parse_float("TRANSCRIPT_STT_RTF", 0.5),
        "dl_mib_per_sec": _parse_float("TRANSCRIPT_DL_MIB_PER_SEC", 5.0),
    }


def _decode_stt_params(raw: dict[str, JSONValue]) -> STTJobParams:
    """Parse and validate STT job parameters from queue payload."""
    url_val = raw.get("url")
    if not isinstance(url_val, str) or url_val.strip() == "":
        raise JSONTypeError("url must be a non-empty string")
    user_id_val = raw.get("user_id")
    if not isinstance(user_id_val, int):
        raise JSONTypeError("user_id must be an integer")
    return {"url": url_val.strip(), "user_id": user_id_val}


def process_stt_impl(
    job_id: str,
    params: STTJobParams,
    *,
    redis: RedisStrProto,
    stt_client: STTClient,
    probe_client: ProbeDownloadClient,
    config: STTConfig,
    logger: LoggerProtocol,
) -> STTJobResult:
    """Implementation for STT processing with explicit injected deps."""
    store = TranscriptJobStore(redis)
    created_at = datetime.utcnow()
    user_id = params["user_id"]

    # Parse URL and extract video ID
    canonical_url = canonicalize_youtube_url(params["url"])
    video_id = extract_video_id(canonical_url)

    # Create job context for publishing events
    ctx: JobContext = make_job_context(
        redis=redis,
        domain="transcript",
        events_channel=default_events_channel("transcript"),
        job_id=job_id,
        user_id=user_id,
        queue_name=TRANSCRIPT_QUEUE,
    )

    # Save initial status
    store.save(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": "processing",
            "progress": 0,
            "message": "started",
            "url": canonical_url,
            "video_id": video_id,
            "text": None,
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
        }
    )

    # Publish started event
    ctx.publish_started()

    logger.info("Starting STT job", extra={"job_id": job_id, "video_id": video_id})

    cookies_text = _optional_env_str("TRANSCRIPT_COOKIES_TEXT")
    stt_provider = STTTranscriptProvider(
        stt_client=stt_client,
        probe_client=probe_client,
        max_video_seconds=config["max_video_seconds"],
        max_file_mb=config["max_file_mb"],
        enable_chunking=config["enable_chunking"],
        chunk_threshold_mb=config["chunk_threshold_mb"],
        target_chunk_mb=config["target_chunk_mb"],
        max_chunk_duration=config["max_chunk_duration_seconds"],
        max_concurrent_chunks=config["max_concurrent_chunks"],
        silence_threshold_db=config["silence_threshold_db"],
        silence_duration=config["silence_duration_seconds"],
        stt_rtf=config["stt_rtf"],
        dl_mib_per_sec=config["dl_mib_per_sec"],
        cookies_text=cookies_text,
    )

    # Update progress - downloading
    now = datetime.utcnow()
    store.save(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": "processing",
            "progress": 10,
            "message": "downloading audio",
            "url": canonical_url,
            "video_id": video_id,
            "text": None,
            "created_at": created_at,
            "updated_at": now,
            "error": None,
        }
    )
    ctx.publish_progress(10, "downloading audio")

    # Perform STT
    opts = TranscriptOptions(preferred_langs=DEFAULT_TRANSCRIPT_LANGS)
    segments = stt_provider.fetch(video_id, opts)
    text = clean_segments(segments)
    text_bytes = len(text.encode("utf-8"))

    # Save completed status
    now = datetime.utcnow()
    store.save(
        {
            "job_id": job_id,
            "user_id": user_id,
            "status": "completed",
            "progress": 100,
            "message": "done",
            "url": canonical_url,
            "video_id": video_id,
            "text": text,
            "created_at": created_at,
            "updated_at": now,
            "error": None,
        }
    )
    ctx.publish_completed(video_id, text_bytes)

    logger.info("STT job completed", extra={"job_id": job_id, "video_id": video_id})
    return {"job_id": job_id, "status": "completed", "video_id": video_id, "text": text}


def _get_redis_client(url: str) -> RedisStrProto:
    return redis_for_kv(url)


def _build_stt_client() -> STTClient:
    """Build STT client from environment."""
    from .adapters.openai_client import OpenAISttClient

    api_key = _optional_env_str("OPENAI_API_KEY") or _optional_env_str("OPEN_AI_API_KEY")
    if api_key is None:
        api_key = _require_env_str("OPENAI_API_KEY")
    return OpenAISttClient(api_key=api_key)


def _build_probe_client() -> ProbeDownloadClient:
    """Build probe/download client from environment."""
    from .adapters.yt_dlp_client import YtDlpAdapter

    return YtDlpAdapter()


def _decode_process_stt(job_id: str, params: dict[str, JSONValue]) -> STTJobResult:
    """RQ job entry point. Loads deps from env and delegates to the impl."""
    logger = get_logger(__name__)
    redis_url = _require_env_str("REDIS_URL")
    client = _get_redis_client(redis_url)

    stt_client = _build_stt_client()
    probe_client = _build_probe_client()
    config = _load_stt_config()

    decoded_params = _decode_stt_params(params)

    try:
        return process_stt_impl(
            job_id,
            decoded_params,
            redis=client,
            stt_client=stt_client,
            probe_client=probe_client,
            config=config,
            logger=logger,
        )
    finally:
        client.close()


def process_stt(job_id: str, params: Mapping[str, JSONValue]) -> STTJobResult:
    """Public RQ job entry point for STT processing jobs.

    This is the function that RQ workers call. It delegates to the internal
    implementation after loading settings and decoding the payload.
    """
    return _decode_process_stt(job_id, dict(params))


__all__ = [
    "STTConfig",
    "STTJobParams",
    "STTJobResult",
    "process_stt",
    "process_stt_impl",
]
