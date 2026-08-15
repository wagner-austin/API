"""Tests for chunker: FakeSubprocessRunForChunker."""

from __future__ import annotations

import subprocess
from pathlib import Path

from platform_stt import _test_hooks
from platform_stt.chunker import AudioChunker
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
