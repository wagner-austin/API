from __future__ import annotations

from platform_core.config import _optional_env_str
from platform_core.logging import get_logger
from typing_extensions import TypedDict

from .cleaner import clean_segments
from .provider import TranscriptProvider, YouTubeTranscriptClient
from .stt_provider import (
    ProbeDownloadClient,
    STTClient,
    STTTranscriptProvider,
    YtDlpCaptionProvider,
)
from .types import DEFAULT_TRANSCRIPT_LANGS, TranscriptOptions, TranscriptResult
from .youtube import canonicalize_youtube_url, extract_video_id


class Config(TypedDict, total=False):
    TRANSCRIPT_MAX_VIDEO_SECONDS: int
    TRANSCRIPT_MAX_FILE_MB: int
    TRANSCRIPT_ENABLE_CHUNKING: bool
    TRANSCRIPT_CHUNK_THRESHOLD_MB: float
    TRANSCRIPT_TARGET_CHUNK_MB: float
    TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS: float
    TRANSCRIPT_MAX_CONCURRENT_CHUNKS: int
    TRANSCRIPT_SILENCE_THRESHOLD_DB: float
    TRANSCRIPT_SILENCE_DURATION_SECONDS: float
    TRANSCRIPT_STT_RTF: float
    TRANSCRIPT_DL_MIB_PER_SEC: float
    TRANSCRIPT_PREFERRED_LANGS: list[str] | None


class Clients(TypedDict):
    youtube: YouTubeTranscriptClient
    stt: STTClient
    probe: ProbeDownloadClient


class TranscriptService:
    def __init__(self, cfg: Config, clients: Clients) -> None:
        self._cfg = cfg
        self._clients = clients
        self._logger = get_logger(__name__)

    def captions(self, url: str, preferred_langs: list[str] | None) -> TranscriptResult:
        canonical = canonicalize_youtube_url(url)
        vid = extract_video_id(canonical)
        cfg_langs = self._cfg.get("TRANSCRIPT_PREFERRED_LANGS")
        langs = preferred_langs or cfg_langs or DEFAULT_TRANSCRIPT_LANGS
        opts = TranscriptOptions(preferred_langs=langs)
        cookies_text = _optional_env_str("TRANSCRIPT_COOKIES_TEXT")
        provider: TranscriptProvider = YtDlpCaptionProvider(
            probe_client=self._clients["probe"],
            cookies_text=cookies_text,
        )
        segments = provider.fetch(vid, opts)
        text = clean_segments(segments)
        return TranscriptResult(url=canonical, video_id=vid, text=text)

    def stt(self, url: str) -> TranscriptResult:
        canonical = canonicalize_youtube_url(url)
        vid = extract_video_id(canonical)
        cookies_text = _optional_env_str("TRANSCRIPT_COOKIES_TEXT")
        stt = STTTranscriptProvider(
            stt_client=self._clients["stt"],
            probe_client=self._clients["probe"],
            max_video_seconds=int(self._cfg["TRANSCRIPT_MAX_VIDEO_SECONDS"]),
            max_file_mb=int(self._cfg["TRANSCRIPT_MAX_FILE_MB"]),
            enable_chunking=bool(self._cfg["TRANSCRIPT_ENABLE_CHUNKING"]),
            chunk_threshold_mb=float(self._cfg["TRANSCRIPT_CHUNK_THRESHOLD_MB"]),
            target_chunk_mb=float(self._cfg["TRANSCRIPT_TARGET_CHUNK_MB"]),
            max_chunk_duration=float(self._cfg["TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS"]),
            max_concurrent_chunks=int(self._cfg["TRANSCRIPT_MAX_CONCURRENT_CHUNKS"]),
            silence_threshold_db=float(self._cfg["TRANSCRIPT_SILENCE_THRESHOLD_DB"]),
            silence_duration=float(self._cfg["TRANSCRIPT_SILENCE_DURATION_SECONDS"]),
            stt_rtf=float(self._cfg["TRANSCRIPT_STT_RTF"]),
            dl_mib_per_sec=float(self._cfg["TRANSCRIPT_DL_MIB_PER_SEC"]),
            cookies_text=cookies_text,
        )
        segments = stt.fetch(vid, TranscriptOptions(preferred_langs=DEFAULT_TRANSCRIPT_LANGS))
        text = clean_segments(segments)
        return TranscriptResult(url=canonical, video_id=vid, text=text)
