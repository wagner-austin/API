"""Audio chunking with silence detection for large file processing.

Uses ffmpeg/ffprobe for silence detection and stream-copy splitting to
avoid re-encoding where possible.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile

from platform_core.json_utils import JSONValue, load_json_str
from platform_core.logging import get_logger
from typing_extensions import TypedDict

from . import _test_hooks
from .types import AudioChunk

_SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<ts>[0-9]+(?:\.[0-9]+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<ts>[0-9]+(?:\.[0-9]+)?)")


class _SplitWindow(TypedDict):
    """Time window for audio splitting."""

    start: float
    end: float


class _FfprobeFormatDict(TypedDict, total=False):
    """Format information from ffprobe output."""

    format_name: str | int


class _FfprobeStreamDict(TypedDict, total=False):
    """Stream information from ffprobe output."""

    codec_type: str | int
    codec_name: str | int


class _FfprobeOutputDict(TypedDict, total=False):
    """Complete ffprobe output structure."""

    format: _FfprobeFormatDict | str
    streams: list[_FfprobeStreamDict] | str


class AudioChunker:
    """Split audio files at optimal points (silence when possible).

    Uses ffmpeg/ffprobe and stream copy to avoid re-encoding for speed.
    Attempts to split at silence points for cleaner audio boundaries.

    Attributes:
        target_chunk_mb: Target size for each chunk in megabytes.
        max_chunk_duration_seconds: Maximum duration for any single chunk.
        silence_threshold_db: Silence detection threshold in decibels.
        silence_duration_seconds: Minimum silence duration to detect.
    """

    __slots__ = (
        "_ffmpeg",
        "_ffprobe",
        "_logger",
        "_max_chunk_dur",
        "_silence_db",
        "_silence_min",
        "_target_chunk_mb",
    )

    def __init__(
        self,
        *,
        target_chunk_mb: float = 20.0,
        max_chunk_duration_seconds: float = 600.0,
        silence_threshold_db: float = -40.0,
        silence_duration_seconds: float = 0.5,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
    ) -> None:
        """Initialize audio chunker.

        Args:
            target_chunk_mb: Target chunk size in MB (default: 20.0).
            max_chunk_duration_seconds: Maximum chunk duration (default: 600.0).
            silence_threshold_db: Silence threshold in dB (default: -40.0).
            silence_duration_seconds: Minimum silence duration (default: 0.5).
            ffmpeg_path: Path to ffmpeg executable (default: "ffmpeg").
            ffprobe_path: Path to ffprobe executable (default: "ffprobe").
        """
        self._target_chunk_mb = max(1.0, float(target_chunk_mb))
        self._max_chunk_dur = max(1.0, float(max_chunk_duration_seconds))
        self._silence_db = float(silence_threshold_db)
        self._silence_min = max(0.1, float(silence_duration_seconds))
        self._logger = get_logger(__name__)
        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffprobe_path

    def chunk_audio(
        self, audio_path: str, total_duration: float, estimated_mb: float
    ) -> list[AudioChunk]:
        """Split audio file into chunks if needed.

        Returns chunk descriptors. If no chunking needed, returns a single
        pass-through chunk pointing to the original file.

        Args:
            audio_path: Path to the audio file.
            total_duration: Total duration of audio in seconds.
            estimated_mb: Estimated file size in megabytes.

        Returns:
            List of AudioChunk descriptors.
        """
        size_mb = self._safe_size_mb(audio_path)
        est_mb = estimated_mb or size_mb
        if est_mb <= self._target_chunk_mb and total_duration <= self._max_chunk_dur:
            return [
                AudioChunk(
                    path=audio_path,
                    start_seconds=0.0,
                    duration_seconds=max(0.0, float(total_duration)),
                    size_bytes=os.path.getsize(audio_path),
                )
            ]

        self._logger.info(
            "Chunking audio: size=%.1fMB duration=%.1fs target=%.1fMB",
            est_mb,
            total_duration,
            self._target_chunk_mb,
        )

        silence_points = self._detect_silence(audio_path, total_duration)
        split_points = self._calculate_split_points(silence_points, total_duration, est_mb)
        return self._split_audio(audio_path, split_points, total_duration)

    def _safe_size_mb(self, audio_path: str) -> float:
        """Get file size in MB, returning 0.0 on error."""
        try:
            return os.path.getsize(audio_path) / (1024 * 1024)
        except OSError as e:
            self._logger.warning("Failed to stat audio file: %s", e)
            return 0.0

    def _detect_silence(self, audio_path: str, duration: float) -> list[float]:
        """Run ffmpeg silencedetect and parse timestamps.

        Prefers silence_end as split points for cleaner audio boundaries.

        Args:
            audio_path: Path to audio file.
            duration: Total audio duration.

        Returns:
            List of silence end timestamps.
        """
        cmd = [
            self._ffmpeg,
            "-i",
            audio_path,
            "-af",
            f"silencedetect=n={self._silence_db}dB:d={self._silence_min}",
            "-f",
            "null",
            "-",
        ]
        self._logger.debug("Running silencedetect: %s", " ".join(cmd))
        try:
            proc = _test_hooks.subprocess_run(cmd, capture_output=True, text=True, timeout=90)
        except (subprocess.TimeoutExpired, OSError) as e:
            self._logger.warning("Silence detection failed to run: %s", e)
            return []
        out = str(proc.stdout or "") + str(proc.stderr or "")
        points: list[float] = []
        for line in out.splitlines():
            m_end = _SILENCE_END_RE.search(line)
            if not m_end:
                continue
            ts = float(m_end.group("ts"))
            points.append(ts)
        self._logger.debug("Detected %d silence points in %.1fs audio", len(points), duration)
        return points

    def _calculate_split_points(
        self, silence_points: list[float], total_duration: float, estimated_mb: float
    ) -> list[float]:
        """Determine optimal split points based on target size and detected silence.

        Returns a monotonically increasing list of split timestamps (seconds)
        within (0, duration).

        Args:
            silence_points: List of silence end timestamps.
            total_duration: Total audio duration.
            estimated_mb: Estimated file size in MB.

        Returns:
            List of split point timestamps.
        """
        num_chunks = max(1, math.ceil(max(1e-6, estimated_mb) / self._target_chunk_mb))
        ideal: list[float] = [(total_duration / num_chunks) * i for i in range(1, num_chunks)]
        if not ideal:
            return []
        if total_duration / num_chunks > self._max_chunk_dur:
            extra_chunks = math.ceil(total_duration / self._max_chunk_dur)
            ideal = [(total_duration / extra_chunks) * i for i in range(1, extra_chunks)]
        if not silence_points:
            return ideal
        tolerance_ratio = 0.30
        out: list[float] = []
        for target in ideal:
            tol = max(1.0, total_duration * tolerance_ratio / max(1, len(ideal)))
            nearest = silence_points[0]
            best_dist = abs(nearest - target)
            for candidate in silence_points[1:]:
                dist = abs(candidate - target)
                if dist < best_dist:
                    nearest = candidate
                    best_dist = dist
            if abs(nearest - target) <= tol:
                out.append(nearest)
                self._logger.debug("Split at %.1fs (silence near ideal %.1fs)", nearest, target)
            else:
                out.append(target)
                self._logger.debug("Split at %.1fs (no nearby silence)", target)
        return sorted({x for x in out if 0.0 < x < total_duration})

    def _split_audio(
        self, audio_path: str, split_points: list[float], total_duration: float
    ) -> list[AudioChunk]:
        """Create audio chunk files at split points.

        Args:
            audio_path: Path to source audio file.
            split_points: List of split timestamps.
            total_duration: Total audio duration.

        Returns:
            List of AudioChunk descriptors for created files.
        """
        container, codec = self._probe_stream_info(audio_path)
        ext = "webm" if codec == "opus" else "m4a"
        if not split_points:
            return [
                AudioChunk(
                    path=audio_path,
                    start_seconds=0.0,
                    duration_seconds=max(0.0, float(total_duration)),
                    size_bytes=os.path.getsize(audio_path),
                )
            ]
        segments: list[_SplitWindow] = []
        last = 0.0
        for s in split_points:
            s_clamped = min(max(0.0, s), total_duration)
            if s_clamped > last:
                segments.append(_SplitWindow(start=last, end=s_clamped))
                last = s_clamped
        if last < total_duration:
            segments.append(_SplitWindow(start=last, end=total_duration))

        self._logger.info(
            "Chunking plan: input_format=%s codec=%s out_ext=.%s parts=%d",
            container or "?",
            codec or "?",
            ext,
            len(segments),
        )

        outdir = tempfile.mkdtemp(prefix="stt_chunks_")
        created: list[AudioChunk] = []
        for idx, seg in enumerate(segments):
            out_path = os.path.join(outdir, f"chunk_{idx:03d}.{ext}")
            copy_cmd = [
                self._ffmpeg,
                "-ss",
                f"{seg['start']:.3f}",
                "-to",
                f"{seg['end']:.3f}",
                "-i",
                audio_path,
                "-c",
                "copy",
                "-y",
                out_path,
            ]
            self._logger.debug("Creating chunk (copy): %s", " ".join(copy_cmd))
            try:
                proc_copy = _test_hooks.subprocess_run(
                    copy_cmd, capture_output=True, text=True, timeout=180, check=True
                )
                _ = proc_copy  # Mark as used
            except subprocess.CalledProcessError:
                reencode_cmd = [
                    self._ffmpeg,
                    "-ss",
                    f"{seg['start']:.3f}",
                    "-to",
                    f"{seg['end']:.3f}",
                    "-i",
                    audio_path,
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    "-y",
                    out_path,
                ]
                self._logger.debug("Creating chunk (reencode): %s", " ".join(reencode_cmd))
                try:
                    proc_reencode = _test_hooks.subprocess_run(
                        reencode_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        check=True,
                    )
                    _ = proc_reencode  # Mark as used
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    self._logger.exception("ffmpeg re-encode split failed: %s", e)
                    self._cleanup_dir(outdir)
                    raise
            except (subprocess.TimeoutExpired, OSError, subprocess.SubprocessError) as e:
                self._logger.exception("ffmpeg split error: %s", e)
                self._cleanup_dir(outdir)
                raise
            if os.path.exists(out_path):
                sz = os.path.getsize(out_path)
            else:
                self._logger.warning("Split segment missing at path: %s", out_path)
                sz = 0
            created.append(
                AudioChunk(
                    path=out_path,
                    start_seconds=seg["start"],
                    duration_seconds=max(0.0, seg["end"] - seg["start"]),
                    size_bytes=sz,
                )
            )
        return created

    def _cleanup_dir(self, path: str) -> None:
        """Clean up temporary directory."""
        if not isinstance(path, str) or not path:
            self._logger.warning("Invalid directory for cleanup: %r", path)
            return
        if not os.path.isdir(path):
            return
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _extract_container_format(raw: _FfprobeOutputDict) -> str:
        """Extract container format from ffprobe output with runtime validation."""
        if "format" in raw:
            fmt = raw["format"]
            if isinstance(fmt, dict):
                fname = fmt.get("format_name")
                if isinstance(fname, str):
                    return fname
        return ""

    @staticmethod
    def _extract_audio_codec(raw: _FfprobeOutputDict) -> str:
        """Extract audio codec from ffprobe streams with runtime validation."""
        streams = raw.get("streams")
        if isinstance(streams, list):
            for s_dict in streams:
                if isinstance(s_dict, dict) and s_dict.get("codec_type") == "audio":
                    cname = s_dict.get("codec_name")
                    if isinstance(cname, str):
                        return cname
        return ""

    @staticmethod
    def _load_ffprobe_json(json_str: str) -> _FfprobeOutputDict | None:
        """Parse and validate ffprobe JSON output into typed structure.

        Returns None if JSON is invalid or not a dict. Does not raise exceptions
        for invalid input - callers should handle None return appropriately.
        """
        # Use platform_core's json parsing which validates structure
        from platform_core.json_utils import InvalidJsonError

        logger = get_logger(__name__)
        parsed_raw: JSONValue
        try:
            parsed_raw = load_json_str(json_str)
        except InvalidJsonError as e:
            # Invalid JSON string from ffprobe - expected at boundary
            logger.debug("ffprobe returned invalid JSON: %s", e)
            return None

        # Check if it's a dict before narrowing
        if not isinstance(parsed_raw, dict):
            return None

        parsed = parsed_raw
        result: _FfprobeOutputDict = {"format": {"format_name": ""}, "streams": []}

        # Validate and extract format
        fmt = parsed.get("format")
        if isinstance(fmt, dict):
            fname = fmt.get("format_name")
            if isinstance(fname, str):
                result["format"] = {"format_name": fname}

        # Validate and extract streams
        streams = parsed.get("streams")
        if isinstance(streams, list):
            validated_streams: list[_FfprobeStreamDict] = []
            for s in streams:
                if isinstance(s, dict):
                    ctype = s.get("codec_type")
                    cname = s.get("codec_name")
                    if isinstance(ctype, str) and isinstance(cname, str):
                        validated_streams.append({"codec_type": ctype, "codec_name": cname})
            result["streams"] = validated_streams

        return result

    def _probe_stream_info(self, audio_path: str) -> tuple[str, str]:
        """Probe audio file for container format and codec.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (container_format, audio_codec).
        """
        cmd = [
            self._ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=format_name",
            "-show_streams",
            "-of",
            "json",
            audio_path,
        ]
        try:
            proc = _test_hooks.subprocess_run(cmd, capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, OSError) as e:
            self._logger.warning("ffprobe failed: %s", e)
            return "", ""
        stdout = str(proc.stdout or "")
        raw = self._load_ffprobe_json(stdout)
        if raw is None:
            self._logger.warning("ffprobe JSON validation failed")
            return "", ""

        container = self._extract_container_format(raw)
        codec = self._extract_audio_codec(raw)
        return container, codec


__all__ = ["AudioChunker"]
