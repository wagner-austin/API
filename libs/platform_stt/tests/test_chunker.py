"""Tests for platform_stt.chunker module."""

from __future__ import annotations

import subprocess
from pathlib import Path

from platform_stt import _test_hooks
from platform_stt.chunker import AudioChunker, _FfprobeOutputDict
from platform_stt.testing import FakeSubprocessResult, reset_hooks


class FakeSubprocessRunForChunker:
    """Fake subprocess runner for chunker tests."""

    __slots__ = ("_result", "calls")

    def __init__(self, result: FakeSubprocessResult | None = None) -> None:
        self._result = result or FakeSubprocessResult()
        self.calls: list[list[str]] = []

    def __call__(
        self,
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> FakeSubprocessResult:
        """Record call and return configured result."""
        del capture_output, timeout, text, input, cwd, env
        self.calls.append(args)
        if check and self._result.returncode != 0:
            raise subprocess.CalledProcessError(
                self._result.returncode, args, self._result.stdout, self._result.stderr
            )
        return self._result


class TestAudioChunkerInit:
    """Tests for AudioChunker initialization."""

    def test_init_defaults(self) -> None:
        """Initialize with default values."""
        chunker = AudioChunker()
        assert chunker._target_chunk_mb == 20.0
        assert chunker._max_chunk_dur == 600.0
        assert chunker._silence_db == -40.0
        assert chunker._silence_min == 0.5
        assert chunker._ffmpeg == "ffmpeg"
        assert chunker._ffprobe == "ffprobe"

    def test_init_custom_values(self) -> None:
        """Initialize with custom values."""
        chunker = AudioChunker(
            target_chunk_mb=10.0,
            max_chunk_duration_seconds=300.0,
            silence_threshold_db=-35.0,
            silence_duration_seconds=0.3,
            ffmpeg_path="/usr/local/bin/ffmpeg",
            ffprobe_path="/usr/local/bin/ffprobe",
        )
        assert chunker._target_chunk_mb == 10.0
        assert chunker._max_chunk_dur == 300.0
        assert chunker._silence_db == -35.0
        assert chunker._silence_min == 0.3
        assert chunker._ffmpeg == "/usr/local/bin/ffmpeg"
        assert chunker._ffprobe == "/usr/local/bin/ffprobe"

    def test_init_clamps_minimum_values(self) -> None:
        """Clamp values to minimums."""
        chunker = AudioChunker(
            target_chunk_mb=0.5,
            max_chunk_duration_seconds=0.5,
            silence_duration_seconds=0.05,
        )
        assert chunker._target_chunk_mb == 1.0
        assert chunker._max_chunk_dur == 1.0
        assert chunker._silence_min == 0.1


class TestChunkAudio:
    """Tests for chunk_audio method."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_no_chunking_needed_small_file(self, tmp_path: Path) -> None:
        """Return single chunk for small file."""
        audio_file = tmp_path / "small.mp3"
        audio_file.write_bytes(b"x" * 1024)  # 1KB file

        chunker = AudioChunker(target_chunk_mb=20.0, max_chunk_duration_seconds=600.0)
        result = chunker.chunk_audio(
            str(audio_file),
            total_duration=60.0,
            estimated_mb=0.001,
        )

        assert len(result) == 1
        assert result[0]["path"] == str(audio_file)
        assert result[0]["start_seconds"] == 0.0
        assert result[0]["duration_seconds"] == 60.0
        assert result[0]["size_bytes"] == 1024

    def test_no_chunking_at_threshold(self, tmp_path: Path) -> None:
        """Return single chunk when exactly at threshold."""
        audio_file = tmp_path / "threshold.mp3"
        audio_file.write_bytes(b"x" * 1024)

        chunker = AudioChunker(target_chunk_mb=20.0, max_chunk_duration_seconds=600.0)
        result = chunker.chunk_audio(
            str(audio_file),
            total_duration=600.0,
            estimated_mb=20.0,
        )

        assert len(result) == 1

    def test_chunking_needed_large_file(self, tmp_path: Path) -> None:
        """Chunk large file."""
        audio_file = tmp_path / "large.mp3"
        audio_file.write_bytes(b"x" * 1024)

        # Mock ffprobe to return valid format
        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )
        ffprobe_result = FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
        silence_result = FakeSubprocessResult(
            returncode=0,
            stderr="silence_end: 30.0\nsilence_end: 60.0\n",
        )
        split_result = FakeSubprocessResult(returncode=0)

        call_count = 0
        out_files: list[str] = []

        def fake_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            nonlocal call_count
            del capture_output, timeout, text, input, cwd, env
            call_count += 1
            if "ffprobe" in args[0]:
                return ffprobe_result
            if "silencedetect" in str(args):
                return silence_result
            # Split command - create the output file
            if "-c" in args and "copy" in args:
                for i, arg in enumerate(args):
                    if arg == "-y" and i + 1 < len(args):
                        out_path = args[i + 1]
                        out_files.append(out_path)
                        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                        Path(out_path).write_bytes(b"chunk")
                        break
            if check and split_result.returncode != 0:
                raise subprocess.CalledProcessError(
                    split_result.returncode, args, split_result.stdout, split_result.stderr
                )
            return split_result

        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker(target_chunk_mb=20.0, max_chunk_duration_seconds=600.0)
        result = chunker.chunk_audio(
            str(audio_file),
            total_duration=120.0,
            estimated_mb=60.0,  # 3x target
        )

        assert len(result) >= 2
        assert call_count >= 1


class TestDetectSilence:
    """Tests for _detect_silence method."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_detect_silence_parses_output(self, tmp_path: Path) -> None:
        """Parse silence_end timestamps from ffmpeg output."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")

        result = FakeSubprocessResult(
            returncode=0,
            stderr="[silencedetect] silence_end: 10.5\n[silencedetect] silence_end: 25.3\n",
        )
        fake_run = FakeSubprocessRunForChunker(result)
        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker()
        points = chunker._detect_silence(str(audio_file), 60.0)

        assert len(points) == 2
        assert 10.5 in points
        assert 25.3 in points

    def test_detect_silence_handles_timeout(self, tmp_path: Path) -> None:
        """Return empty list on timeout."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")

        def timeout_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del capture_output, check, timeout, text, input, cwd, env
            raise subprocess.TimeoutExpired(args, 90)

        _test_hooks.subprocess_run = timeout_run

        chunker = AudioChunker()
        points = chunker._detect_silence(str(audio_file), 60.0)

        assert points == []

    def test_detect_silence_handles_oserror(self, tmp_path: Path) -> None:
        """Return empty list on OSError."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")

        def error_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del args, capture_output, check, timeout, text, input, cwd, env
            raise OSError("ffmpeg not found")

        _test_hooks.subprocess_run = error_run

        chunker = AudioChunker()
        points = chunker._detect_silence(str(audio_file), 60.0)

        assert points == []


class TestCalculateSplitPoints:
    """Tests for _calculate_split_points method."""

    def test_no_split_points_single_chunk(self) -> None:
        """Return empty list when single chunk is sufficient."""
        chunker = AudioChunker(target_chunk_mb=20.0)
        points = chunker._calculate_split_points(
            silence_points=[],
            total_duration=60.0,
            estimated_mb=15.0,
        )
        assert points == []

    def test_split_points_without_silence(self) -> None:
        """Calculate split points without silence data."""
        chunker = AudioChunker(target_chunk_mb=20.0)
        points = chunker._calculate_split_points(
            silence_points=[],
            total_duration=120.0,
            estimated_mb=60.0,  # Need 3 chunks
        )
        # 60 MB / 20 MB = 3 chunks, so 2 split points at 40s and 80s
        assert len(points) == 2
        assert points[0] == 40.0
        assert points[1] == 80.0

    def test_split_points_with_silence(self) -> None:
        """Use silence points when available."""
        chunker = AudioChunker(target_chunk_mb=20.0)
        points = chunker._calculate_split_points(
            silence_points=[28.0, 32.0, 58.0, 62.0],
            total_duration=120.0,
            estimated_mb=60.0,
        )
        # Should prefer silence points near ideal split (40s, 80s)
        # Closest to 40s is 32.0, closest to 80s is 62.0
        assert len(points) == 2

    def test_split_points_respects_max_duration(self) -> None:
        """Add extra splits when chunks exceed max duration."""
        # Note: Duration check only triggers if size-based splitting is needed first
        chunker = AudioChunker(target_chunk_mb=100.0, max_chunk_duration_seconds=30.0)
        points = chunker._calculate_split_points(
            silence_points=[],
            total_duration=120.0,
            estimated_mb=200.0,  # Above size threshold (needs 2 chunks by size)
        )
        # Size: 200 MB / 100 MB = 2 chunks -> 60s each
        # Duration: 60s > 30s max -> recalculate to 4 chunks (30s each)
        # Result: 3 split points at 30s, 60s, 90s
        assert len(points) == 3
        assert points[0] == 30.0
        assert points[1] == 60.0
        assert points[2] == 90.0


class TestSplitAudio:
    """Tests for _split_audio method."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_split_audio_no_points(self, tmp_path: Path) -> None:
        """Return single chunk when no split points."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 1000)

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )
        ffprobe_result = FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
        _test_hooks.subprocess_run = FakeSubprocessRunForChunker(ffprobe_result)

        chunker = AudioChunker()
        result = chunker._split_audio(str(audio_file), [], 60.0)

        assert len(result) == 1
        assert result[0]["path"] == str(audio_file)

    def test_split_audio_creates_chunks(self, tmp_path: Path) -> None:
        """Create chunk files at split points."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 1000)

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )
        ffprobe_result = FakeSubprocessResult(returncode=0, stdout=ffprobe_json)

        def fake_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del capture_output, timeout, text, input, cwd, env
            if "ffprobe" in args[0]:
                return ffprobe_result
            # Split command - create output file
            for i, arg in enumerate(args):
                if arg == "-y" and i + 1 < len(args):
                    out_path = args[i + 1]
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_bytes(b"chunk data")
                    break
            result = FakeSubprocessResult(returncode=0)
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, args, result.stdout, result.stderr
                )
            return result

        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker()
        result = chunker._split_audio(str(audio_file), [30.0], 60.0)

        assert len(result) == 2
        assert result[0]["start_seconds"] == 0.0
        assert result[0]["duration_seconds"] == 30.0
        assert result[1]["start_seconds"] == 30.0
        assert result[1]["duration_seconds"] == 30.0

    def test_split_audio_duplicate_split_points(self, tmp_path: Path) -> None:
        """Handle duplicate split points (skip duplicates)."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 1000)

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )

        def fake_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del capture_output, timeout, text, input, cwd, env
            if "ffprobe" in args[0]:
                return FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
            # Create output file
            for i, arg in enumerate(args):
                if arg == "-y" and i + 1 < len(args):
                    out_path = args[i + 1]
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_bytes(b"chunk data")
                    break
            result = FakeSubprocessResult(returncode=0)
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, args, result.stdout, result.stderr
                )
            return result

        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker()
        # Split points with duplicates [30, 30, 45] - duplicates should be skipped
        result = chunker._split_audio(str(audio_file), [30.0, 30.0, 45.0], 60.0)

        # Should have 4 segments: 0-30, 30-45, 45-60 (duplicate 30 skipped)
        assert len(result) == 3
        assert result[0]["start_seconds"] == 0.0
        assert result[0]["duration_seconds"] == 30.0
        assert result[1]["start_seconds"] == 30.0
        assert result[1]["duration_seconds"] == 15.0
        assert result[2]["start_seconds"] == 45.0
        assert result[2]["duration_seconds"] == 15.0

    def test_split_audio_last_equals_total(self, tmp_path: Path) -> None:
        """Handle split point equal to total duration (no trailing segment)."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 1000)

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )

        def fake_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del capture_output, timeout, text, input, cwd, env
            if "ffprobe" in args[0]:
                return FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
            # Create output file
            for i, arg in enumerate(args):
                if arg == "-y" and i + 1 < len(args):
                    out_path = args[i + 1]
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_bytes(b"chunk data")
                    break
            result = FakeSubprocessResult(returncode=0)
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, args, result.stdout, result.stderr
                )
            return result

        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker()
        # Split at 60.0 with total_duration 60.0 - should only create one chunk
        result = chunker._split_audio(str(audio_file), [60.0], 60.0)

        # Should have exactly one chunk since split point equals total duration
        assert len(result) == 1
        assert result[0]["start_seconds"] == 0.0
        assert result[0]["duration_seconds"] == 60.0


class TestProbeStreamInfo:
    """Tests for _probe_stream_info method."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_probe_stream_info_mp3(self, tmp_path: Path) -> None:
        """Parse MP3 format info."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )
        result = FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
        _test_hooks.subprocess_run = FakeSubprocessRunForChunker(result)

        chunker = AudioChunker()
        container, codec = chunker._probe_stream_info(str(audio_file))

        assert container == "mp3"
        assert codec == "mp3"

    def test_probe_stream_info_opus(self, tmp_path: Path) -> None:
        """Parse Opus format info."""
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake")

        ffprobe_json = (
            '{"format": {"format_name": "matroska,webm"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "opus"}]}'
        )
        result = FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
        _test_hooks.subprocess_run = FakeSubprocessRunForChunker(result)

        chunker = AudioChunker()
        container, codec = chunker._probe_stream_info(str(audio_file))

        assert container == "matroska,webm"
        assert codec == "opus"

    def test_probe_stream_info_timeout(self, tmp_path: Path) -> None:
        """Return empty strings on timeout."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")

        def timeout_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del capture_output, check, timeout, text, input, cwd, env
            raise subprocess.TimeoutExpired(args, 30)

        _test_hooks.subprocess_run = timeout_run

        chunker = AudioChunker()
        container, codec = chunker._probe_stream_info(str(audio_file))

        assert container == ""
        assert codec == ""

    def test_probe_stream_info_invalid_json(self, tmp_path: Path) -> None:
        """Return empty strings for invalid JSON."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")

        result = FakeSubprocessResult(returncode=0, stdout="not json")
        _test_hooks.subprocess_run = FakeSubprocessRunForChunker(result)

        chunker = AudioChunker()
        container, codec = chunker._probe_stream_info(str(audio_file))

        assert container == ""
        assert codec == ""


class TestSafeSizeMb:
    """Tests for _safe_size_mb method."""

    def test_safe_size_mb_existing_file(self, tmp_path: Path) -> None:
        """Return size for existing file."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"x" * (1024 * 1024))  # 1MB

        chunker = AudioChunker()
        size = chunker._safe_size_mb(str(audio_file))

        assert abs(size - 1.0) < 0.01

    def test_safe_size_mb_missing_file(self) -> None:
        """Return 0.0 for missing file."""
        chunker = AudioChunker()
        size = chunker._safe_size_mb("/nonexistent/file.mp3")

        assert size == 0.0


class TestCleanupDir:
    """Tests for _cleanup_dir method."""

    def test_cleanup_dir_removes_directory(self, tmp_path: Path) -> None:
        """Remove directory and contents."""
        test_dir = tmp_path / "to_clean"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        chunker = AudioChunker()
        chunker._cleanup_dir(str(test_dir))

        assert not test_dir.exists()

    def test_cleanup_dir_handles_nonexistent(self) -> None:
        """Handle nonexistent directory gracefully."""
        chunker = AudioChunker()
        chunker._cleanup_dir("/nonexistent/directory")
        # Should not raise

    def test_cleanup_dir_handles_empty_string(self) -> None:
        """Handle empty string gracefully."""
        chunker = AudioChunker()
        chunker._cleanup_dir("")
        # Should not raise

    def test_cleanup_dir_handles_file_path(self, tmp_path: Path) -> None:
        """Handle file path (not directory) gracefully."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        chunker = AudioChunker()
        chunker._cleanup_dir(str(test_file))
        # Should not raise, file still exists
        assert test_file.exists()


class TestLoadFfprobeJson:
    """Tests for _load_ffprobe_json static method."""

    def test_load_ffprobe_json_valid(self) -> None:
        """Parse valid ffprobe JSON."""
        json_str = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )
        result = AudioChunker._load_ffprobe_json(json_str)

        # Check the format was parsed correctly
        fmt = result.get("format") if result else None
        format_name = fmt.get("format_name") if isinstance(fmt, dict) else None
        assert format_name == "mp3"
        # Check streams were parsed
        streams = result.get("streams") if result else []
        streams_list = streams if isinstance(streams, list) else []
        assert len(streams_list) == 1

    def test_load_ffprobe_json_invalid(self) -> None:
        """Return None for invalid JSON."""
        result = AudioChunker._load_ffprobe_json("not valid json")
        assert result is None

    def test_load_ffprobe_json_not_dict(self) -> None:
        """Return None for non-dict JSON."""
        result = AudioChunker._load_ffprobe_json("[1, 2, 3]")
        assert result is None

    def test_load_ffprobe_json_empty(self) -> None:
        """Handle empty JSON object."""
        result = AudioChunker._load_ffprobe_json("{}")
        # Empty dict is normalized with default format/streams
        assert result == {"format": {"format_name": ""}, "streams": []}

    def test_load_ffprobe_json_format_name_not_string(self) -> None:
        """Handle format_name that is not a string."""
        json_str = '{"format": {"format_name": 123}, "streams": []}'
        result = AudioChunker._load_ffprobe_json(json_str)
        # format_name is not string, so default is used
        assert result == {"format": {"format_name": ""}, "streams": []}

    def test_load_ffprobe_json_stream_not_dict(self) -> None:
        """Handle stream items that are not dicts."""
        json_str = '{"format": {"format_name": "mp3"}, "streams": ["not a dict", 123]}'
        result = AudioChunker._load_ffprobe_json(json_str)
        # Non-dict stream items are skipped - result should have mp3 format but empty streams
        # Access result directly after guard check
        expected: _FfprobeOutputDict = {"format": {"format_name": "mp3"}, "streams": []}
        assert result == expected

    def test_load_ffprobe_json_stream_codec_not_string(self) -> None:
        """Handle stream with non-string codec_type or codec_name."""
        json_str = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": 123, "codec_name": "mp3"}]}'
        )
        result = AudioChunker._load_ffprobe_json(json_str)
        # codec_type is not string, so stream is skipped - verify by checking streams is empty
        expected: _FfprobeOutputDict = {"format": {"format_name": "mp3"}, "streams": []}
        assert result == expected


class TestExtractContainerFormat:
    """Tests for _extract_container_format static method."""

    def test_extract_container_format_valid(self) -> None:
        """Extract format name from valid structure."""
        raw: _FfprobeOutputDict = {"format": {"format_name": "mp3"}, "streams": []}
        result = AudioChunker._extract_container_format(raw)
        assert result == "mp3"

    def test_extract_container_format_missing(self) -> None:
        """Return empty string when format missing."""
        raw: _FfprobeOutputDict = {"streams": []}
        result = AudioChunker._extract_container_format(raw)
        assert result == ""

    def test_extract_container_format_not_dict(self) -> None:
        """Return empty string when format is not dict."""
        raw: _FfprobeOutputDict = {"format": "mp3", "streams": []}
        result = AudioChunker._extract_container_format(raw)
        assert result == ""

    def test_extract_container_format_name_not_string(self) -> None:
        """Return empty string when format_name is not a string."""
        raw: _FfprobeOutputDict = {"format": {"format_name": 123}, "streams": []}
        result = AudioChunker._extract_container_format(raw)
        assert result == ""


class TestExtractAudioCodec:
    """Tests for _extract_audio_codec static method."""

    def test_extract_audio_codec_valid(self) -> None:
        """Extract codec from valid structure."""
        raw: _FfprobeOutputDict = {
            "format": {},
            "streams": [{"codec_type": "audio", "codec_name": "aac"}],
        }
        result = AudioChunker._extract_audio_codec(raw)
        assert result == "aac"

    def test_extract_audio_codec_no_audio_stream(self) -> None:
        """Return empty string when no audio stream."""
        raw: _FfprobeOutputDict = {
            "format": {},
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
        }
        result = AudioChunker._extract_audio_codec(raw)
        assert result == ""

    def test_extract_audio_codec_empty_streams(self) -> None:
        """Return empty string for empty streams."""
        raw: _FfprobeOutputDict = {"format": {}, "streams": []}
        result = AudioChunker._extract_audio_codec(raw)
        assert result == ""

    def test_extract_audio_codec_streams_not_list(self) -> None:
        """Return empty string when streams is not list."""
        raw: _FfprobeOutputDict = {"format": {}, "streams": "invalid"}
        result = AudioChunker._extract_audio_codec(raw)
        assert result == ""

    def test_extract_audio_codec_codec_name_not_string(self) -> None:
        """Return empty string when codec_name is not a string."""
        raw: _FfprobeOutputDict = {
            "format": {},
            "streams": [{"codec_type": "audio", "codec_name": 123}],
        }
        result = AudioChunker._extract_audio_codec(raw)
        assert result == ""


class TestDetectSilenceEdgeCases:
    """Additional tests for _detect_silence edge cases."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_detect_silence_skips_non_matching_lines(self, tmp_path: Path) -> None:
        """Skip output lines that don't match silence pattern."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")

        # Mixed output with non-matching and matching lines
        result = FakeSubprocessResult(
            returncode=0,
            stderr=(
                "Some random ffmpeg output\n"
                "[silencedetect] silence_end: 10.5\n"
                "More noise in output\n"
                "[silencedetect] silence_end: 25.3\n"
                "Final line without match\n"
            ),
        )
        _test_hooks.subprocess_run = FakeSubprocessRunForChunker(result)

        chunker = AudioChunker()
        points = chunker._detect_silence(str(audio_file), 60.0)

        # Should only find the two actual silence points
        assert len(points) == 2
        assert 10.5 in points
        assert 25.3 in points


class TestSplitAudioErrorHandling:
    """Tests for _split_audio error handling."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_split_audio_reencode_on_copy_failure(self, tmp_path: Path) -> None:
        """Re-encode when stream copy fails."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 1000)

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )
        call_count = 0

        def fake_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            nonlocal call_count
            del capture_output, timeout, text, input, cwd, env
            call_count += 1
            if "ffprobe" in args[0]:
                return FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
            # First copy attempt fails, re-encode succeeds
            # Note: copy command uses -c copy (not -c:a)
            if "-c" in args and "copy" in args:
                if check:
                    raise subprocess.CalledProcessError(1, args)
                return FakeSubprocessResult(returncode=1)
            # Re-encode command - create the output file
            for i, arg in enumerate(args):
                if arg == "-y" and i + 1 < len(args):
                    out_path = args[i + 1]
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(out_path).write_bytes(b"reencoded chunk")
                    break
            return FakeSubprocessResult(returncode=0)

        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker()
        result = chunker._split_audio(str(audio_file), [30.0], 60.0)

        assert len(result) == 2
        # Verify re-encoding was attempted (multiple calls beyond ffprobe)
        assert call_count >= 3

    def test_split_audio_handles_missing_output(self, tmp_path: Path) -> None:
        """Handle case when output file doesn't exist after split."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 1000)

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )

        def fake_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del capture_output, check, timeout, text, input, cwd, env
            if "ffprobe" in args[0]:
                return FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
            # Split command succeeds but doesn't create file
            return FakeSubprocessResult(returncode=0)

        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker()
        result = chunker._split_audio(str(audio_file), [30.0], 60.0)

        # Should still return chunks but with size_bytes=0
        assert len(result) == 2
        assert result[0]["size_bytes"] == 0
        assert result[1]["size_bytes"] == 0


class TestProbeStreamInfoErrorHandling:
    """Tests for _probe_stream_info error handling."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_probe_stream_info_json_parse_error(self, tmp_path: Path) -> None:
        """Return empty strings on JSON parse error."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")

        # Return invalid JSON that will raise ValueError during parsing
        result = FakeSubprocessResult(returncode=0, stdout="not valid json {")
        _test_hooks.subprocess_run = FakeSubprocessRunForChunker(result)

        chunker = AudioChunker()
        container, codec = chunker._probe_stream_info(str(audio_file))

        assert container == ""
        assert codec == ""


class TestSplitAudioErrorPaths:
    """Tests for _split_audio error handling paths."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_split_audio_reencode_also_fails(self, tmp_path: Path) -> None:
        """Raise when both copy and re-encode fail."""
        import pytest

        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 1000)

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )

        def fake_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del capture_output, timeout, text, input, cwd, env
            if "ffprobe" in args[0]:
                return FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
            # Copy command fails
            if "-c" in args and "copy" in args:
                if check:
                    raise subprocess.CalledProcessError(1, args)
                return FakeSubprocessResult(returncode=1)
            # Re-encode command ALSO fails
            if check:
                raise subprocess.CalledProcessError(1, args)
            return FakeSubprocessResult(returncode=1)

        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker()
        with pytest.raises(subprocess.CalledProcessError):
            chunker._split_audio(str(audio_file), [30.0], 60.0)

    def test_split_audio_copy_timeout(self, tmp_path: Path) -> None:
        """Raise when copy command times out."""
        import pytest

        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 1000)

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )

        def fake_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del capture_output, check, text, input, cwd, env
            if "ffprobe" in args[0]:
                return FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
            # Copy command times out (not CalledProcessError)
            raise subprocess.TimeoutExpired(args, timeout or 300.0)

        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker()
        with pytest.raises(subprocess.TimeoutExpired):
            chunker._split_audio(str(audio_file), [30.0], 60.0)

    def test_split_audio_copy_oserror(self, tmp_path: Path) -> None:
        """Raise when copy command has OS error."""
        import pytest

        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 1000)

        ffprobe_json = (
            '{"format": {"format_name": "mp3"}, '
            '"streams": [{"codec_type": "audio", "codec_name": "mp3"}]}'
        )

        def fake_run(
            args: list[str],
            *,
            capture_output: bool = False,
            check: bool = False,
            timeout: float | None = None,
            text: bool = False,
            input: bytes | str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> FakeSubprocessResult:
            del capture_output, check, timeout, text, input, cwd, env
            if "ffprobe" in args[0]:
                return FakeSubprocessResult(returncode=0, stdout=ffprobe_json)
            # Copy command has OS error (not CalledProcessError)
            raise OSError("ffmpeg not found")

        _test_hooks.subprocess_run = fake_run

        chunker = AudioChunker()
        with pytest.raises(OSError, match="ffmpeg not found"):
            chunker._split_audio(str(audio_file), [30.0], 60.0)
