"""Tests for chunker: ProbeStreamInfo."""

from __future__ import annotations

import subprocess
from pathlib import Path

from platform_stt import _test_hooks
from platform_stt.chunker import AudioChunker
from platform_stt.testing import FakeSubprocessResult, reset_hooks
from tests.test_chunker_chunking import FakeSubprocessRunForChunker


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
