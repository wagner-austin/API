"""Tests for platform_stt._test_hooks module."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from platform_stt import _test_hooks
from platform_stt._test_hooks import (
    AudioChunkerProtocol,
    _default_audio_chunker_factory,
    _default_ffmpeg_available,
    _default_langid_download,
    _default_langid_ensure_model_path,
    _default_mkdtemp,
    _default_os_path_getsize,
    _default_os_remove,
    _default_os_stat,
    _default_subprocess_run,
    _run_subprocess_bytes,
    _run_subprocess_text,
    _SubprocessRunResultImpl,
)
from platform_stt.testing import reset_hooks


class TestSubprocessRunResultImpl:
    """Tests for _SubprocessRunResultImpl class."""

    def test_init(self) -> None:
        """Initialize result implementation."""
        result = _SubprocessRunResultImpl(
            returncode=0,
            stdout=b"output",
            stderr=b"error",
        )
        assert result.returncode == 0
        assert result.stdout == b"output"
        assert result.stderr == b"error"

    def test_init_text(self) -> None:
        """Initialize with text output."""
        result = _SubprocessRunResultImpl(
            returncode=1,
            stdout="text output",
            stderr="text error",
        )
        assert result.returncode == 1
        assert result.stdout == "text output"
        assert result.stderr == "text error"

    def test_init_none(self) -> None:
        """Initialize with None output."""
        result = _SubprocessRunResultImpl(
            returncode=0,
            stdout=None,
            stderr=None,
        )
        assert result.stdout is None
        assert result.stderr is None


class TestRunSubprocessBytes:
    """Tests for _run_subprocess_bytes function."""

    def test_run_echo_bytes(self) -> None:
        """Run simple echo command returning bytes."""
        result = _run_subprocess_bytes(
            ["python", "-c", "print('hello')"],
            capture_output=True,
            check=False,
            timeout=10.0,
            input_data=None,
            cwd=None,
            env=None,
        )
        assert result.returncode == 0
        stdout = result.stdout or b""
        assert b"hello" in stdout

    def test_run_with_input(self) -> None:
        """Run command with stdin input."""
        result = _run_subprocess_bytes(
            ["python", "-c", "import sys; print(sys.stdin.read().strip())"],
            capture_output=True,
            check=False,
            timeout=10.0,
            input_data=b"test input",
            cwd=None,
            env=None,
        )
        assert result.returncode == 0
        stdout = result.stdout or b""
        assert b"test input" in stdout

    def test_run_raises_on_check_failure(self) -> None:
        """Raise CalledProcessError on failure with check=True."""
        with pytest.raises(subprocess.CalledProcessError):
            _run_subprocess_bytes(
                ["python", "-c", "import sys; sys.exit(1)"],
                capture_output=True,
                check=True,
                timeout=10.0,
                input_data=None,
                cwd=None,
                env=None,
            )


class TestRunSubprocessText:
    """Tests for _run_subprocess_text function."""

    def test_run_echo_text(self) -> None:
        """Run simple echo command returning text."""
        result = _run_subprocess_text(
            ["python", "-c", "print('hello')"],
            capture_output=True,
            check=False,
            timeout=10.0,
            input_data=None,
            cwd=None,
            env=None,
        )
        assert result.returncode == 0
        stdout = result.stdout or ""
        assert "hello" in stdout

    def test_run_with_text_input(self) -> None:
        """Run command with text stdin input."""
        result = _run_subprocess_text(
            ["python", "-c", "import sys; print(sys.stdin.read().strip())"],
            capture_output=True,
            check=False,
            timeout=10.0,
            input_data="text input",
            cwd=None,
            env=None,
        )
        assert result.returncode == 0
        stdout = result.stdout or ""
        assert "text input" in stdout

    def test_run_raises_on_check_failure(self) -> None:
        """Raise CalledProcessError on failure with check=True."""
        with pytest.raises(subprocess.CalledProcessError):
            _run_subprocess_text(
                ["python", "-c", "import sys; sys.exit(1)"],
                capture_output=True,
                check=True,
                timeout=10.0,
                input_data=None,
                cwd=None,
                env=None,
            )


class TestDefaultSubprocessRun:
    """Tests for _default_subprocess_run function."""

    def test_run_text_mode(self) -> None:
        """Run in text mode."""
        result = _default_subprocess_run(
            ["python", "-c", "print('hello')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        stdout = result.stdout or ""
        assert "hello" in str(stdout)

    def test_run_bytes_mode(self) -> None:
        """Run in bytes mode."""
        result = _default_subprocess_run(
            ["python", "-c", "print('hello')"],
            capture_output=True,
            text=False,
        )
        assert result.returncode == 0
        stdout = result.stdout or b""
        stdout_bytes = stdout if isinstance(stdout, bytes) else stdout.encode()
        assert b"hello" in stdout_bytes

    def test_run_with_string_input_bytes_mode(self) -> None:
        """String input gets encoded in bytes mode."""
        result = _default_subprocess_run(
            ["python", "-c", "import sys; print(sys.stdin.read().strip())"],
            capture_output=True,
            text=False,
            input="string input",
        )
        assert result.returncode == 0
        stdout = result.stdout or b""
        stdout_bytes = stdout if isinstance(stdout, bytes) else stdout.encode()
        assert b"string input" in stdout_bytes

    def test_run_with_bytes_input(self) -> None:
        """Bytes input works in bytes mode."""
        result = _default_subprocess_run(
            ["python", "-c", "import sys; print(sys.stdin.read().strip())"],
            capture_output=True,
            text=False,
            input=b"bytes input",
        )
        assert result.returncode == 0
        stdout = result.stdout or b""
        stdout_bytes = stdout if isinstance(stdout, bytes) else stdout.encode()
        assert b"bytes input" in stdout_bytes


class TestDefaultOsFunctions:
    """Tests for default OS function hooks."""

    def test_default_os_stat(self, tmp_path: Path) -> None:
        """Test os.stat hook."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = _default_os_stat(str(test_file))
        assert result.st_size > 0

    def test_default_os_path_getsize(self, tmp_path: Path) -> None:
        """Test os.path.getsize hook."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("12345")

        result = _default_os_path_getsize(str(test_file))
        assert result == 5

    def test_default_os_remove(self, tmp_path: Path) -> None:
        """Test os.remove hook."""
        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("delete me")

        _default_os_remove(str(test_file))
        assert not test_file.exists()

    def test_default_mkdtemp(self) -> None:
        """Test tempfile.mkdtemp hook."""
        result = _default_mkdtemp(prefix="test_")
        try:
            assert os.path.isdir(result)
            assert "test_" in result
        finally:
            os.rmdir(result)

    def test_default_mkdtemp_with_dir(self, tmp_path: Path) -> None:
        """Test mkdtemp with custom directory."""
        result = _default_mkdtemp(prefix="sub_", dir=str(tmp_path))
        try:
            assert os.path.isdir(result)
            assert str(tmp_path) in result
        finally:
            os.rmdir(result)


class TestDefaultFfmpegAvailable:
    """Tests for _default_ffmpeg_available function."""

    def test_ffmpeg_available_returns_bool(self) -> None:
        """Returns boolean indicating availability."""
        result = _default_ffmpeg_available()
        assert result is True or result is False


class TestModuleLevelHooks:
    """Tests for module-level hook variables."""

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_subprocess_run_hook(self) -> None:
        """subprocess_run hook is callable."""
        result = _test_hooks.subprocess_run(
            ["python", "-c", "print('test')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_os_stat_hook(self, tmp_path: Path) -> None:
        """os_stat hook is callable."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("x")

        result = _test_hooks.os_stat(str(test_file))
        assert result.st_size == 1

    def test_os_path_getsize_hook(self, tmp_path: Path) -> None:
        """os_path_getsize hook is callable."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("xyz")

        result = _test_hooks.os_path_getsize(str(test_file))
        assert result == 3

    def test_os_remove_hook(self, tmp_path: Path) -> None:
        """os_remove hook is callable."""
        test_file = tmp_path / "to_remove.txt"
        test_file.write_text("x")

        _test_hooks.os_remove(str(test_file))
        assert not test_file.exists()

    def test_mkdtemp_hook(self, tmp_path: Path) -> None:
        """mkdtemp hook is callable."""
        result = _test_hooks.mkdtemp("test_", str(tmp_path))
        try:
            assert os.path.isdir(result)
        finally:
            os.rmdir(result)

    def test_ffmpeg_available_hook(self) -> None:
        """ffmpeg_available hook is callable."""
        result = _test_hooks.ffmpeg_available()
        assert result is True or result is False


class TestHookProtocols:
    """Tests to verify protocol compliance."""

    def test_subprocess_run_protocol(self) -> None:
        """Default subprocess_run matches protocol."""
        # This tests that the function signature matches SubprocessRunProtocol
        fn: _test_hooks.SubprocessRunProtocol = _test_hooks._default_subprocess_run
        result = fn(
            ["python", "-c", "pass"],
            capture_output=True,
            check=False,
            timeout=10.0,
            text=True,
        )
        assert result.returncode == 0


class TestDefaultOpenAIClientFactory:
    """Tests for _default_openai_client_factory function."""

    def test_creates_openai_client(self) -> None:
        """Create OpenAI client with given configuration."""
        from platform_stt._test_hooks import _default_openai_client_factory

        # Create client - this actually imports openai and creates a client
        client = _default_openai_client_factory(
            api_key="test-api-key",
            timeout=30.0,
            max_retries=2,
        )
        # Verify it conforms to OpenAIClientProtocol by accessing properties
        audio = client.audio
        transcriptions = audio.transcriptions
        translations = audio.translations
        # Verify the create method exists on transcriptions (key API method)
        assert callable(transcriptions.create)
        assert callable(translations.create)


class TestDefaultAudioChunkerFactory:
    """Tests for _default_audio_chunker_factory function."""

    def test_creates_audio_chunker(self) -> None:
        """Create AudioChunker with given configuration."""
        from platform_stt.chunker import AudioChunker

        chunker: AudioChunkerProtocol = _default_audio_chunker_factory(
            target_chunk_mb=20.0,
            max_chunk_duration_seconds=600.0,
            silence_threshold_db=-40.0,
            silence_duration_seconds=0.5,
        )
        # Verify correct type returned
        assert type(chunker) is AudioChunker


class TestDefaultLangidDownload:
    """Tests for _default_langid_download function."""

    def test_downloads_file_from_url(self, tmp_path: Path) -> None:
        """Download file from URL to destination path."""
        import http.server
        import socketserver
        import threading

        content = b"test model content"

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, format: str, *args: str) -> None:
                pass

        # Find free port
        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
            port = httpd.server_address[1]

            def serve() -> None:
                httpd.handle_request()

            thread = threading.Thread(target=serve)
            thread.start()

            dest = tmp_path / "models" / "test.bin"
            _default_langid_download(f"http://127.0.0.1:{port}/model", dest)

            thread.join(timeout=5)

        assert dest.exists()
        assert dest.read_bytes() == content


class TestDefaultLangidEnsureModelPath:
    """Tests for _default_langid_ensure_model_path function."""

    def test_returns_existing_model_path(self, tmp_path: Path) -> None:
        """Return path when model already exists."""
        # Create the model directory and file
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "lid218e.bin"
        model_file.write_bytes(b"fake model")

        result = _default_langid_ensure_model_path(str(tmp_path), True)
        assert result == model_file

    def test_returns_lid176_when_preferred(self, tmp_path: Path) -> None:
        """Return lid.176 path when prefer_218e is False."""
        # Create the model directory and file
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "lid.176.bin"
        model_file.write_bytes(b"fake model 176")

        result = _default_langid_ensure_model_path(str(tmp_path), False)
        assert result == model_file


class TestDefaultLangidGetFasttextFactory:
    """Tests for _default_langid_get_fasttext_factory function."""

    def test_returns_fasttext_factory(self, tmp_path: Path) -> None:
        """Return FastText model factory that can load models."""
        from platform_stt._test_hooks import _default_langid_get_fasttext_factory

        # First ensure we have a model to test with (use smaller lid.176)
        model_path = _default_langid_ensure_model_path(str(tmp_path), False)

        # Get the factory
        factory = _default_langid_get_fasttext_factory()
        assert callable(factory)

        # Create a model and verify it works
        model = factory(model_path=str(model_path))
        labels, probs = model.predict("hello world", k=1)
        label: str = labels[0]
        prob: float = probs.item(0)
        assert label.startswith("__label__")
        assert 0.0 <= prob <= 1.0
