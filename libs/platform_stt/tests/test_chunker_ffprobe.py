"""Tests for chunker: LoadFfprobeJson."""

from __future__ import annotations

import subprocess
from pathlib import Path

from platform_stt import _test_hooks
from platform_stt.chunker import AudioChunker, _FfprobeOutputDict
from platform_stt.testing import FakeSubprocessResult, reset_hooks
from tests.test_chunker_chunking import FakeSubprocessRunForChunker


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
