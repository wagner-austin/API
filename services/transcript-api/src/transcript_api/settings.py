from __future__ import annotations

from platform_core.config import (
    _optional_env_str,
    _parse_bool,
    _parse_float,
    _parse_int,
    _require_env_str,
)

from .adapters.openai_client import OpenAISttClient
from .adapters.youtube_client import YouTubeTranscriptApiAdapter
from .adapters.yt_dlp_client import YtDlpAdapter
from .service import Clients, Config


def build_config_from_env() -> Config:
    return Config(
        TRANSCRIPT_MAX_VIDEO_SECONDS=_parse_int("TRANSCRIPT_MAX_VIDEO_SECONDS", 0),
        TRANSCRIPT_MAX_FILE_MB=_parse_int("TRANSCRIPT_MAX_FILE_MB", 0),
        TRANSCRIPT_ENABLE_CHUNKING=_parse_bool("TRANSCRIPT_ENABLE_CHUNKING", False),
        TRANSCRIPT_CHUNK_THRESHOLD_MB=_parse_float("TRANSCRIPT_CHUNK_THRESHOLD_MB", 20.0),
        TRANSCRIPT_TARGET_CHUNK_MB=_parse_float("TRANSCRIPT_TARGET_CHUNK_MB", 20.0),
        TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=_parse_float(
            "TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS", 600.0
        ),
        TRANSCRIPT_MAX_CONCURRENT_CHUNKS=_parse_int("TRANSCRIPT_MAX_CONCURRENT_CHUNKS", 3),
        TRANSCRIPT_SILENCE_THRESHOLD_DB=_parse_float("TRANSCRIPT_SILENCE_THRESHOLD_DB", -40.0),
        TRANSCRIPT_SILENCE_DURATION_SECONDS=_parse_float(
            "TRANSCRIPT_SILENCE_DURATION_SECONDS", 0.5
        ),
        TRANSCRIPT_STT_RTF=_parse_float("TRANSCRIPT_STT_RTF", 0.5),
        TRANSCRIPT_DL_MIB_PER_SEC=_parse_float("TRANSCRIPT_DL_MIB_PER_SEC", 4.0),
        TRANSCRIPT_PREFERRED_LANGS=None,
    )


def build_clients_from_env() -> Clients:
    api_key = _optional_env_str("OPENAI_API_KEY") or _optional_env_str("OPEN_AI_API_KEY")
    if api_key is None:
        api_key = _require_env_str("OPENAI_API_KEY")
    return Clients(
        youtube=YouTubeTranscriptApiAdapter(),
        stt=OpenAISttClient(api_key=api_key),
        probe=YtDlpAdapter(),
    )


__all__ = [
    "build_clients_from_env",
    "build_config_from_env",
]
