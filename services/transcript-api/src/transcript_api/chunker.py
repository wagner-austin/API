from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from subprocess import CompletedProcess

from platform_core.logging import get_logger
from typing_extensions import TypedDict

from .types import AudioChunk

_SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<ts>[0-9]+(?:\.[0-9]+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<ts>[0-9]+(?:\.[0-9]+)?)")


class _SplitWindow(TypedDict):
    start: float
    end: float


class _FfprobeFormatDict(TypedDict, total=False):
    format_name: str | int


class _FfprobeStreamDict(TypedDict, total=False):
    codec_type: str | int
    codec_name: str | int


class _FfprobeOutputDict(TypedDict, total=False):
    format: _FfprobeFormatDict | str
    streams: list[_FfprobeStreamDict] | str


class AudioChunker:
    """Split audio files at optimal points (silence when possible).

    Uses ffmpeg/ffprobe and stream copy to avoid re-encoding for speed.
    """

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
        """Return chunk descriptors. If no chunking needed, return a single pass-through chunk."""
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
        try:
            return os.path.getsize(audio_path) / (1024 * 1024)
        except OSError as e:
            self._logger.warning("Failed to stat audio file: %s", e)
            return 0.0

    def _detect_silence(self, audio_path: str, duration: float) -> list[float]:
        """Run ffmpeg silencedetect and parse timestamps (prefer silence_end as split)."""
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
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        except (subprocess.TimeoutExpired, OSError) as e:
            self._logger.warning("Silence detection failed to run: %s", e)
            return []
        out = (proc.stdout or "") + (proc.stderr or "")
        points: list[float] = []
        for line in out.splitlines():
            m_end = _SILENCE_END_RE.search(line)
            if not m_end:
                continue
            ts_str = m_end.group("ts")
            try:
                ts = float(ts_str)
            except ValueError:
                continue
            points.append(ts)
        self._logger.debug("Detected %d silence points in %.1fs audio", len(points), duration)
        return points

    def _calculate_split_points(
        self, silence_points: list[float], total_duration: float, estimated_mb: float
    ) -> list[float]:
        """Determine optimal split points based on target size and detected silence.

        Returns a monotonically increasing list of split timestamps (seconds) within (0, duration).
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

        outdir = tempfile.mkdtemp(prefix="ytstt_chunks_")
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
                proc_copy: CompletedProcess[str] = subprocess.run(
                    copy_cmd, check=True, capture_output=True, text=True, timeout=180
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
                    proc_reencode: CompletedProcess[str] = subprocess.run(
                        reencode_cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=300,
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
        """Parse and validate ffprobe JSON output into typed structure."""
        from .json_util import parse_json_dict

        parsed = parse_json_dict(json_str)
        if parsed is None:
            return None

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
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, OSError) as e:
            self._logger.warning("ffprobe failed: %s", e)
            return "", ""
        stdout = proc.stdout or ""
        try:
            raw = self._load_ffprobe_json(stdout)
        except ValueError as e:
            self._logger.warning("ffprobe JSON parse failed: %s", e)
            return "", ""

        if raw is None:
            self._logger.warning("ffprobe JSON validation failed")
            return "", ""

        container = self._extract_container_format(raw)
        codec = self._extract_audio_codec(raw)
        return container, codec
