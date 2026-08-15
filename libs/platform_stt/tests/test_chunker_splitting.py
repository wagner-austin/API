"""Tests for chunker: CalculateSplitPoints."""

from __future__ import annotations

import subprocess
from pathlib import Path

from platform_stt import _test_hooks
from platform_stt.chunker import AudioChunker
from platform_stt.testing import FakeSubprocessResult, reset_hooks
from tests.test_chunker_chunking import FakeSubprocessRunForChunker


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
