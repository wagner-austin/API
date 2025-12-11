"""Tests for _test_hooks default implementations.

These tests cover the production default implementations in _test_hooks.py that
are normally replaced by fakes in other tests.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

from transcript_api._test_hooks import (
    _default_mkdtemp,
    _default_os_path_getsize,
    _default_os_remove,
    _default_os_stat,
    _default_subprocess_run,
    _run_subprocess_bytes,
    _run_subprocess_text,
    _SubprocessRunResultImpl,
)


def test_default_os_stat_returns_real_stat() -> None:
    """Test _default_os_stat calls real os.stat."""
    # Use a file that definitely exists
    result = _default_os_stat(__file__)
    expected = os.stat(__file__)
    assert result.st_size == expected.st_size
    assert result.st_mtime == expected.st_mtime


def test_default_os_path_getsize_returns_real_size() -> None:
    """Test _default_os_path_getsize calls real os.path.getsize."""
    result = _default_os_path_getsize(__file__)
    expected = os.path.getsize(__file__)
    assert result == expected


def test_default_os_remove_deletes_file() -> None:
    """Test _default_os_remove calls real os.remove."""
    # Create a temp file to delete
    fd, path = tempfile.mkstemp()
    os.close(fd)
    assert os.path.exists(path)
    _default_os_remove(path)
    assert not os.path.exists(path)


def test_default_mkdtemp_creates_directory() -> None:
    """Test _default_mkdtemp calls real tempfile.mkdtemp."""
    path = _default_mkdtemp("test_prefix_", None)
    try:
        assert os.path.isdir(path)
        assert "test_prefix_" in path
    finally:
        os.rmdir(path)


def test_subprocess_run_result_impl_stores_values() -> None:
    """Test _SubprocessRunResultImpl stores returncode, stdout, stderr."""
    result = _SubprocessRunResultImpl(0, b"out", b"err")
    assert result.returncode == 0
    assert result.stdout == b"out"
    assert result.stderr == b"err"

    # Test with string values
    result_str = _SubprocessRunResultImpl(1, "stdout", "stderr")
    assert result_str.returncode == 1
    assert result_str.stdout == "stdout"
    assert result_str.stderr == "stderr"

    # Test with None values
    result_none = _SubprocessRunResultImpl(2, None, None)
    assert result_none.returncode == 2
    assert result_none.stdout is None
    assert result_none.stderr is None


def test_run_subprocess_bytes_captures_output() -> None:
    """Test _run_subprocess_bytes runs command and captures bytes output."""
    result = _run_subprocess_bytes(
        ["python", "-c", "print('hello')"],
        capture_output=True,
        check=False,
        timeout=30.0,
        input_data=None,
        cwd=None,
        env=None,
    )
    assert result.returncode == 0
    # Type is bytes; comparison verifies both type and value
    stdout = result.stdout
    assert type(stdout) is bytes and stdout.strip() == b"hello"


def test_run_subprocess_bytes_with_check_raises() -> None:
    """Test _run_subprocess_bytes raises CalledProcessError when check=True."""
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run_subprocess_bytes(
            ["python", "-c", "import sys; sys.exit(1)"],
            capture_output=True,
            check=True,
            timeout=30.0,
            input_data=None,
            cwd=None,
            env=None,
        )
    assert exc_info.value.returncode == 1


def test_run_subprocess_bytes_with_input() -> None:
    """Test _run_subprocess_bytes can pass input data."""
    result = _run_subprocess_bytes(
        ["python", "-c", "import sys; print(sys.stdin.read().strip())"],
        capture_output=True,
        check=False,
        timeout=30.0,
        input_data=b"test_input",
        cwd=None,
        env=None,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert type(stdout) is bytes and stdout.strip() == b"test_input"


def test_run_subprocess_text_captures_output() -> None:
    """Test _run_subprocess_text runs command and captures text output."""
    result = _run_subprocess_text(
        ["python", "-c", "print('hello_text')"],
        capture_output=True,
        check=False,
        timeout=30.0,
        input_data=None,
        cwd=None,
        env=None,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert type(stdout) is str and stdout.strip() == "hello_text"


def test_run_subprocess_text_with_check_raises() -> None:
    """Test _run_subprocess_text raises CalledProcessError when check=True."""
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run_subprocess_text(
            ["python", "-c", "import sys; sys.exit(2)"],
            capture_output=True,
            check=True,
            timeout=30.0,
            input_data=None,
            cwd=None,
            env=None,
        )
    assert exc_info.value.returncode == 2


def test_run_subprocess_text_with_input() -> None:
    """Test _run_subprocess_text can pass input data."""
    result = _run_subprocess_text(
        ["python", "-c", "import sys; print(sys.stdin.read().strip())"],
        capture_output=True,
        check=False,
        timeout=30.0,
        input_data="text_input",
        cwd=None,
        env=None,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert type(stdout) is str and stdout.strip() == "text_input"


def test_default_subprocess_run_bytes_mode() -> None:
    """Test _default_subprocess_run in bytes mode (text=False)."""
    result = _default_subprocess_run(
        ["python", "-c", "print('bytes_mode')"],
        capture_output=True,
        text=False,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert type(stdout) is bytes and stdout.strip() == b"bytes_mode"


def test_default_subprocess_run_text_mode() -> None:
    """Test _default_subprocess_run in text mode (text=True)."""
    result = _default_subprocess_run(
        ["python", "-c", "print('text_mode')"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert type(stdout) is str and stdout.strip() == "text_mode"


def test_default_subprocess_run_bytes_with_str_input() -> None:
    """Test _default_subprocess_run encodes str input when text=False."""
    result = _default_subprocess_run(
        ["python", "-c", "import sys; print(sys.stdin.read().strip())"],
        capture_output=True,
        text=False,
        input="str_as_bytes",
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert type(stdout) is bytes and stdout.strip() == b"str_as_bytes"


def test_default_subprocess_run_bytes_with_bytes_input() -> None:
    """Test _default_subprocess_run passes bytes input directly when text=False."""
    result = _default_subprocess_run(
        ["python", "-c", "import sys; print(sys.stdin.read().strip())"],
        capture_output=True,
        text=False,
        input=b"raw_bytes",
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert type(stdout) is bytes and stdout.strip() == b"raw_bytes"


def test_default_audio_chunker_factory_creates_chunker() -> None:
    """Test _default_audio_chunker_factory creates an AudioChunker instance."""
    from transcript_api._test_hooks import _default_audio_chunker_factory
    from transcript_api.chunker import AudioChunker

    chunker = _default_audio_chunker_factory(
        target_chunk_mb=20.0,
        max_chunk_duration_seconds=600.0,
        silence_threshold_db=-40.0,
        silence_duration_seconds=0.5,
    )
    # Verify correct type returned
    assert type(chunker) is AudioChunker


def test_default_stt_client_builder_creates_client() -> None:
    """Test _default_stt_client_builder creates an OpenAISttClient instance."""
    from transcript_api._test_hooks import _default_stt_client_builder
    from transcript_api.adapters.openai_client import OpenAISttClient

    client = _default_stt_client_builder(api_key="test-api-key")
    # Verify correct type returned
    assert type(client) is OpenAISttClient


def test_default_probe_client_builder_creates_adapter() -> None:
    """Test _default_probe_client_builder creates a YtDlpAdapter instance."""
    from transcript_api._test_hooks import _default_probe_client_builder
    from transcript_api.adapters.yt_dlp_client import YtDlpAdapter

    adapter = _default_probe_client_builder()
    # Verify correct type returned
    assert type(adapter) is YtDlpAdapter


def test_default_stt_provider_factory_creates_provider() -> None:
    """Test _default_stt_provider_factory creates an STTTranscriptProvider instance."""
    from transcript_api._test_hooks import (
        _default_probe_client_builder,
        _default_stt_client_builder,
        _default_stt_provider_factory,
    )
    from transcript_api.stt_provider import STTTranscriptProvider

    stt_client = _default_stt_client_builder(api_key="test-key")
    probe_client = _default_probe_client_builder()

    provider = _default_stt_provider_factory(
        stt_client=stt_client,
        probe_client=probe_client,
        max_video_seconds=600,
        max_file_mb=100,
        enable_chunking=False,
        chunk_threshold_mb=20.0,
        target_chunk_mb=20.0,
        max_chunk_duration=600.0,
        max_concurrent_chunks=3,
        silence_threshold_db=-40.0,
        silence_duration=0.5,
        stt_rtf=0.5,
        dl_mib_per_sec=4.0,
        cookies_text=None,
    )
    # Verify correct type returned
    assert type(provider) is STTTranscriptProvider


def test_default_yt_api_factory_returns_api() -> None:
    """Test _default_yt_api_factory returns the YouTubeTranscriptApi class."""
    from transcript_api._test_hooks import YTApiProto, _default_yt_api_factory

    api: YTApiProto = _default_yt_api_factory()
    # Verify it conforms to YTApiProto by accessing methods - these would fail if missing
    method_ref_1 = api.get_transcript
    method_ref_2 = api.list_transcripts
    assert callable(method_ref_1)
    assert callable(method_ref_2)


def test_default_yt_exceptions_factory_returns_exceptions() -> None:
    """Test _default_yt_exceptions_factory returns exception tuple."""
    from transcript_api._test_hooks import _default_yt_exceptions_factory

    exc_tuple = _default_yt_exceptions_factory()
    # Verify it returns 3 exception classes
    assert len(exc_tuple) == 3
    for exc_cls in exc_tuple:
        assert issubclass(exc_cls, Exception)


def test_default_yt_dlp_factory_creates_youtube_dl() -> None:
    """Test _default_yt_dlp_factory creates a YoutubeDL instance."""
    from transcript_api._test_hooks import _default_yt_dlp_factory
    from transcript_api.types import YtDlpProto

    # Create with quiet options to avoid output
    ydl: YtDlpProto = _default_yt_dlp_factory({"quiet": True, "no_warnings": True})
    # Verify it conforms to YtDlpProto by accessing methods - these would fail if missing
    method_ref_1 = ydl.extract_info
    method_ref_2 = ydl.prepare_filename
    assert callable(method_ref_1)
    assert callable(method_ref_2)


def test_default_redis_for_kv_creates_client() -> None:
    """Test _default_redis_for_kv creates a Redis client via platform_workers."""
    from platform_workers.testing import FakeRedisClient, hooks, make_fake_load_redis_str_module

    from transcript_api._test_hooks import _default_redis_for_kv

    # Save and restore hooks
    original_hook = hooks.load_redis_str_module

    fake_client = FakeRedisClient()
    hook, _module = make_fake_load_redis_str_module(fake_client)
    hooks.load_redis_str_module = hook

    client = _default_redis_for_kv("redis://test")
    assert client.ping()
    assert client.get("missing") is None

    hooks.load_redis_str_module = original_hook
