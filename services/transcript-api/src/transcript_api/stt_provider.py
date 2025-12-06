from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, runtime_checkable

from platform_core.errors import AppError, TranscriptErrorCode
from platform_core.logging import get_logger
from typing_extensions import TypedDict

from .chunker import AudioChunker
from .merger import TranscriptMerger
from .parallel import ParallelTranscriber
from .types import (
    AudioChunk,
    SubtitleResultTD,
    TranscriptOptions,
    TranscriptSegment,
    VerboseResponseTD,
    YtInfoTD,
)
from .vtt_parser import parse_vtt_file
from .whisper_parse import convert_verbose_to_segments


@runtime_checkable
class TranscriptProvider(Protocol):
    def fetch(self, video_id: str, opts: TranscriptOptions) -> list[TranscriptSegment]: ...


@runtime_checkable
class STTClient(Protocol):
    """Abstraction over an STT backend (e.g., OpenAI Whisper)."""

    def transcribe_verbose(
        self,
        *,
        file: BinaryIO,
        timeout: float | None,
    ) -> VerboseResponseTD: ...


@runtime_checkable
class ProbeDownloadClient(Protocol):
    """Abstraction over a YouTube probing/downloading backend (e.g., yt_dlp)."""

    def probe(self, url: str) -> YtInfoTD: ...

    def download_audio(self, url: str, *, cookies_path: str | None) -> str: ...

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None: ...


def _as_float(val: int | float | str | None) -> float:
    if isinstance(val, int | float):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if _is_numeric_str(s):
            return float(s)
    return 0.0


def _is_numeric_str(s: str) -> bool:
    if not s:
        return False
    first = s[0]
    rest = s
    if first in "+-":
        rest = s[1:]
        if not rest:
            return False
    dot_seen = False
    digit_seen = False
    for ch in rest:
        if ch == ".":
            if dot_seen:
                return False
            dot_seen = True
        elif ch.isdigit():
            digit_seen = True
        else:
            return False
    return digit_seen


class _FfprobeFormatDurationTD(TypedDict):
    duration: str


class _FfprobeOutputDurationTD(TypedDict):
    format: _FfprobeFormatDurationTD


class STTTranscriptProvider:
    def __init__(
        self,
        stt_client: STTClient,
        probe_client: ProbeDownloadClient,
        max_video_seconds: int,
        max_file_mb: int,
        timeout_seconds: float = 900.0,
        max_retries: int = 2,
        cookies_text: str | None = None,
        cookies_path: str | None = None,
        enable_chunking: bool = False,
        chunk_threshold_mb: float = 20.0,
        target_chunk_mb: float = 20.0,
        max_chunk_duration: float = 600.0,
        max_concurrent_chunks: int = 3,
        silence_threshold_db: float = -40.0,
        silence_duration: float = 0.5,
        stt_rtf: float = 0.5,
        dl_mib_per_sec: float = 4.0,
    ) -> None:
        self.stt_client = stt_client
        self.probe_client = probe_client
        self.max_video_seconds = max_video_seconds
        self.max_file_mb = max_file_mb
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.cookies_text = cookies_text
        self.cookies_path = cookies_path
        self.enable_chunking = enable_chunking
        self.chunk_threshold_mb = chunk_threshold_mb
        self.target_chunk_mb = target_chunk_mb
        self.max_chunk_duration = max_chunk_duration
        self.max_concurrent_chunks = max_concurrent_chunks
        self.silence_threshold_db = silence_threshold_db
        self.silence_duration = silence_duration
        self.stt_rtf = stt_rtf
        self.dl_mib_per_sec = dl_mib_per_sec
        self._logger = get_logger(__name__)
        self._temp_cookies_file: str | None = None
        self._owned_tmp_dirs: set[str] = set()

        if self.cookies_text and not self.cookies_path:
            import base64
            import binascii

            try:
                decoded = base64.b64decode(self.cookies_text).decode("utf-8")
                fd, path = tempfile.mkstemp(prefix="ytcookies_", suffix=".txt", text=True)
                with os.fdopen(fd, "w") as f:
                    f.write(decoded)
                self._temp_cookies_file = path
                self._logger.debug("Using cookies from TEXT (temp file): %s", path)
            except (binascii.Error, UnicodeDecodeError, OSError) as exc:
                self._logger.warning("Failed to use TRANSCRIPT_COOKIES_TEXT: %s", exc)
                self._temp_cookies_file = None

    def __del__(self) -> None:
        path = self._temp_cookies_file
        if isinstance(path, str) and path:
            Path(path).unlink(missing_ok=True)

    def fetch(self, video_id: str, opts: TranscriptOptions) -> list[TranscriptSegment]:
        url = f"https://www.youtube.com/watch?v={video_id}"
        duration = self._probe_or_error(video_id, url)
        self._logger.info("Probe complete: duration=%ss", duration)

        audio_path: str | None = None
        try:
            audio_path, size_bytes = self._download_or_error(url)
            if self._is_over_limit(size_bytes):
                return self._handle_over_limit(audio_path, size_bytes)
            return self._transcribe_with_strategy(audio_path)
        finally:
            if audio_path and self._should_cleanup(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    self._logger.exception("Failed to remove temporary audio file: %s", audio_path)
                    raise

    def _probe_or_error(self, video_id: str, url: str) -> int:
        self._logger.info("Probing video for STT: vid=%s url=%s", video_id, url)
        info = self.probe_client.probe(url)
        raw_dur = info.get("duration", 0)
        duration = int(_as_float(raw_dur))
        if duration <= 0:
            raise AppError(
                TranscriptErrorCode.STT_DURATION_UNKNOWN,
                "Unable to determine video duration for transcription.",
                400,
            )
        if duration > int(self.max_video_seconds):
            raise AppError(
                TranscriptErrorCode.STT_TOO_LONG,
                f"Video is too long for STT (>{self.max_video_seconds} seconds).",
                400,
            )
        return duration

    def _download_or_error(self, url: str) -> tuple[str, int]:
        path = self._download_audio(url)
        try:
            st = os.stat(path)
        except OSError:
            self._logger.warning("Initial stat failed for %s; retrying", path)
            try:
                st = os.stat(path)
            except OSError as exc:
                self._logger.exception("Failed to stat downloaded audio file: %s", exc)
                raise AppError(
                    TranscriptErrorCode.STT_DOWNLOAD_FAILED,
                    "Failed to download audio for transcription",
                    400,
                ) from None
        return path, int(st.st_size)

    def _download_audio(self, url: str) -> str:
        cookies_path = self.cookies_path or self._temp_cookies_file
        filename = self.probe_client.download_audio(url, cookies_path=cookies_path)
        abs_dir = os.path.dirname(os.path.abspath(filename))
        self._owned_tmp_dirs.add(abs_dir)
        return filename

    def _is_over_limit(self, size_bytes: int) -> bool:
        size_mb = float(size_bytes) / (1024 * 1024)
        return size_mb > float(self.max_file_mb)

    def _handle_over_limit(self, audio_path: str, size_bytes: int) -> list[TranscriptSegment]:
        if not self.enable_chunking:
            raise AppError(
                TranscriptErrorCode.STT_CHUNKING_DISABLED,
                "Downloaded audio is too large for STT and chunking is disabled.",
                400,
            )
        return self._transcribe_chunked(audio_path)

    def _transcribe_with_strategy(self, audio_path: str) -> list[TranscriptSegment]:
        if self.enable_chunking and self._should_chunk(audio_path):
            try:
                return self._transcribe_chunked(audio_path)
            except RuntimeError as exc:
                raise AppError(
                    TranscriptErrorCode.STT_CHUNK_FAILED,
                    f"Chunked transcription failed: {exc}",
                    400,
                ) from None
        return self._transcribe(audio_path)

    def _transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        with open(audio_path, "rb") as f:
            resp = self.stt_client.transcribe_verbose(file=f, timeout=self.timeout_seconds)
        return convert_verbose_to_segments(resp)

    def _ffmpeg_available(self) -> bool:
        from shutil import which

        ffmpeg = which("ffmpeg")
        ffprobe = which("ffprobe")
        return bool(ffmpeg and ffprobe)

    def _should_cleanup(self, path: str) -> bool:
        if not isinstance(path, str) or not path:
            self._logger.warning("Invalid path for cleanup: %r", path)
            return False
        abs_path = os.path.abspath(path)
        parent = os.path.dirname(abs_path)
        owned = self._owned_tmp_dirs
        if parent in owned:
            return True
        base = os.path.basename(parent)
        return base.startswith("ytstt_")

    def _should_chunk(self, audio_path: str) -> bool:
        if not self.enable_chunking:
            return False
        try:
            size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        except OSError as exc:
            self._logger.warning("Failed to stat audio for chunking: %s", exc)
            return False
        return size_mb > float(self.chunk_threshold_mb)

    def _get_audio_duration(self, audio_path: str) -> float:
        from .json_util import parse_json_dict

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            audio_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            parsed = parse_json_dict(result.stdout or "{}")
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            self._logger.warning("ffprobe duration query failed: %s", exc)
            return 0.0

        # Validate that json.loads returned a dict (not list/str/null)
        if parsed is None:
            self._logger.warning("ffprobe JSON not a dict")
            return 0.0

        # Validate format field
        fmt = parsed.get("format")
        if not isinstance(fmt, dict):
            return 0.0

        duration_val = fmt.get("duration")
        if not isinstance(duration_val, str):
            return 0.0

        return _as_float(duration_val)

    def _transcribe_chunked(self, audio_path: str) -> list[TranscriptSegment]:
        if not self._ffmpeg_available():
            raise AppError(
                TranscriptErrorCode.STT_FFMPEG_MISSING,
                "ffmpeg/ffprobe not available; cannot chunk audio",
                400,
            )
        duration = self._get_audio_duration(audio_path)
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        chunker = AudioChunker(
            target_chunk_mb=float(self.target_chunk_mb),
            max_chunk_duration_seconds=float(self.max_chunk_duration),
            silence_threshold_db=float(self.silence_threshold_db),
            silence_duration_seconds=float(self.silence_duration),
        )
        chunks = chunker.chunk_audio(audio_path, duration, size_mb)
        if len(chunks) == 1 and os.path.abspath(chunks[0]["path"]) == os.path.abspath(audio_path):
            return self._transcribe(audio_path)

        def _do_transcribe(
            *,
            model: str,
            file: BinaryIO,
            response_format: Literal["verbose_json"],
            timeout: float | None = None,
        ) -> VerboseResponseTD:
            return self.stt_client.transcribe_verbose(file=file, timeout=timeout)

        transcriber = ParallelTranscriber(
            transcribe=_do_transcribe,
            max_concurrent=int(self.max_concurrent_chunks),
            max_retries=int(self.max_retries),
            timeout_seconds=float(self.timeout_seconds),
        )
        try:
            results = transcriber.transcribe_chunks(chunks)
            merger = TranscriptMerger()
            pairs: list[tuple[AudioChunk, list[TranscriptSegment]]] = list(
                zip(chunks, results, strict=False)
            )
            return merger.merge(pairs)
        finally:
            for c in chunks:
                if os.path.abspath(c["path"]) == os.path.abspath(audio_path):
                    continue
                if os.path.exists(c["path"]):
                    os.remove(c["path"])
                else:
                    self._logger.warning("Chunk file missing during cleanup: %s", c["path"])

    def estimate(self, url: str) -> tuple[int, float]:
        info = self.probe_client.probe(url)
        duration = int(_as_float(info.get("duration", 0)))
        approx_mb = 0.0
        formats_val = info.get("formats", [])
        best_abr = 0.0
        for fmt in formats_val:
            vcodec_val = fmt.get("vcodec", "")
            acodec_val = fmt.get("acodec", "")
            if vcodec_val and vcodec_val != "none":
                continue
            if not acodec_val or acodec_val == "none":
                continue
            abr_val = fmt.get("abr", 0.0)
            abr = _as_float(abr_val)
            size_bytes_val = fmt.get("filesize") or fmt.get("filesize_approx")
            size_mb = (
                float(size_bytes_val) / (1024 * 1024)
                if isinstance(size_bytes_val, int | float)
                else 0.0
            )
            if size_mb > approx_mb:
                approx_mb = size_mb
            if abr > best_abr:
                best_abr = abr
        if approx_mb <= 0.0 and duration > 0 and best_abr > 0.0:
            approx_mb = (best_abr * 1000.0 / 8.0) * duration / (1024 * 1024)
        return max(0, duration), max(0.0, approx_mb)

    def estimate_eta_minutes(self, duration_seconds: int, approx_size_mb: float) -> int:
        dur_s = max(0, duration_seconds)
        size_mb = max(0.0, float(approx_size_mb))
        dl_time_min = (size_mb / self.dl_mib_per_sec) if self.dl_mib_per_sec > 0 else 0.0
        will_chunk = (
            self.enable_chunking
            and (size_mb > float(self.chunk_threshold_mb) or size_mb > float(self.max_file_mb))
            and self._ffmpeg_available()
        )
        if not will_chunk or dur_s == 0:
            proc_min = (dur_s * float(self.stt_rtf)) / 60.0
        else:
            n_by_size = math.ceil(max(1e-6, size_mb) / float(self.target_chunk_mb))
            n_by_dur = math.ceil(max(1e-6, dur_s) / float(self.max_chunk_duration))
            n_chunks = max(1, max(n_by_size, n_by_dur))
            parallel = max(1, min(self.max_concurrent_chunks, n_chunks))
            proc_min = ((dur_s / parallel) * float(self.stt_rtf)) / 60.0
        total = proc_min + dl_time_min
        return max(1, int(total + 0.5))


class YtDlpCaptionProvider:
    """Caption provider using yt-dlp to fetch YouTube subtitles/captions."""

    def __init__(
        self,
        probe_client: ProbeDownloadClient,
        cookies_text: str | None = None,
        cookies_path: str | None = None,
    ) -> None:
        self.probe_client = probe_client
        self.cookies_text = cookies_text
        self.cookies_path = cookies_path
        self._logger = get_logger(__name__)
        self._temp_cookies_file: str | None = None

        if self.cookies_text and not self.cookies_path:
            import base64
            import binascii

            try:
                decoded = base64.b64decode(self.cookies_text).decode("utf-8")
                fd, path = tempfile.mkstemp(prefix="ytcookies_", suffix=".txt", text=True)
                with os.fdopen(fd, "w") as f:
                    f.write(decoded)
                self._temp_cookies_file = path
                self._logger.debug("Using cookies from TEXT (temp file): %s", path)
            except (binascii.Error, UnicodeDecodeError, OSError) as exc:
                self._logger.warning("Failed to use TRANSCRIPT_COOKIES_TEXT: %s", exc)
                self._temp_cookies_file = None

    def __del__(self) -> None:
        path = self._temp_cookies_file
        if isinstance(path, str) and path:
            Path(path).unlink(missing_ok=True)

    def fetch(self, video_id: str, opts: TranscriptOptions) -> list[TranscriptSegment]:
        """Fetch captions for a YouTube video using yt-dlp subtitle download."""
        url = f"https://www.youtube.com/watch?v={video_id}"
        preferred_langs = opts.get("preferred_langs", ["en", "en-US", "en-GB"])
        cookies_path = self.cookies_path or self._temp_cookies_file

        self._logger.info(
            "Fetching captions via yt-dlp: vid=%s langs=%s", video_id, preferred_langs
        )

        result = self.probe_client.download_subtitles(
            url,
            cookies_path=cookies_path,
            preferred_langs=preferred_langs,
        )

        if result is None:
            raise AppError(
                TranscriptErrorCode.TRANSCRIPT_UNAVAILABLE,
                f"No captions available for video {video_id}",
                404,
            )

        subtitle_path = result["path"]
        lang = result["lang"]
        is_auto = result["is_auto"]

        self._logger.info(
            "Downloaded subtitle: path=%s lang=%s is_auto=%s",
            subtitle_path,
            lang,
            is_auto,
        )

        raw_items = parse_vtt_file(subtitle_path)

        # Convert RawTranscriptItem to TranscriptSegment
        segments: list[TranscriptSegment] = [
            TranscriptSegment(text=item["text"], start=item["start"], duration=item["duration"])
            for item in raw_items
        ]

        # Clean up the downloaded subtitle file
        try:
            os.remove(subtitle_path)
            # Also remove the temp directory if empty
            parent_dir = os.path.dirname(subtitle_path)
            if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
        except OSError:
            self._logger.warning("Failed to clean up subtitle file: %s", subtitle_path)

        return segments
