"""Tests for platform_stt.testing module."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import BinaryIO

import pytest

from platform_stt import _test_hooks
from platform_stt.testing import (
    FakeAudioChunker,
    FakeLangIdModel,
    FakeSTTClient,
    FakeSubprocessResult,
    FakeSubprocessRun,
    make_fake_audio_chunker_factory,
    make_fake_langid_model_factory,
    make_fake_subprocess_run,
    reset_hooks,
    set_production_hooks,
)
from platform_stt.types import AudioChunk, VerboseResponse, VerboseSegment


class TestFakeSTTClient:
    """Tests for FakeSTTClient class."""

    def test_init_default_response(self) -> None:
        """Initialize with default response."""
        client = FakeSTTClient()
        assert client.call_count == 0

    def test_init_custom_response(self) -> None:
        """Initialize with custom response."""
        response = VerboseResponse(
            text="Custom",
            language="en",
            segments=[VerboseSegment(text="Custom", start=0.0, end=1.0)],
        )
        client = FakeSTTClient(response=response)
        file_obj: BinaryIO = io.BytesIO(b"fake")
        result = client.transcribe(file=file_obj)
        assert result["text"] == "Custom"

    def test_transcribe_increments_call_count(self) -> None:
        """Transcribe increments call count."""
        client = FakeSTTClient()
        file_obj: BinaryIO = io.BytesIO(b"fake")
        client.transcribe(file=file_obj)
        client.transcribe(file=file_obj)
        assert client.call_count == 2

    def test_transcribe_stores_language(self) -> None:
        """Transcribe stores language parameter."""
        client = FakeSTTClient()
        file_obj: BinaryIO = io.BytesIO(b"fake")
        client.transcribe(file=file_obj, language="vi")
        assert client._language == "vi"

    def test_translate_returns_translate_response(self) -> None:
        """Translate returns separate translate response."""
        response = VerboseResponse(
            text="Transcribed",
            language="es",
            segments=[VerboseSegment(text="Transcribed", start=0.0, end=1.0)],
        )
        translate_response = VerboseResponse(
            text="Translated",
            language="es",
            segments=[VerboseSegment(text="Translated", start=0.0, end=1.0)],
        )
        client = FakeSTTClient(response=response, translate_response=translate_response)

        file_obj: BinaryIO = io.BytesIO(b"fake")
        result = client.translate(file=file_obj)
        assert result["text"] == "Translated"

    def test_translate_falls_back_to_response(self) -> None:
        """Translate falls back to main response if no translate response."""
        response = VerboseResponse(
            text="Same",
            language="en",
            segments=[VerboseSegment(text="Same", start=0.0, end=1.0)],
        )
        client = FakeSTTClient(response=response)

        file_obj: BinaryIO = io.BytesIO(b"fake")
        result = client.translate(file=file_obj)
        assert result["text"] == "Same"

    def test_process_transcribe(self) -> None:
        """Process with transcribe task."""
        client = FakeSTTClient()
        file_obj: BinaryIO = io.BytesIO(b"fake")
        result = client.process(file=file_obj, task="transcribe", language="en")
        assert result["text"] == "Test transcription"

    def test_process_translate(self) -> None:
        """Process with translate task."""
        translate_response = VerboseResponse(
            text="Translated",
            language="es",
            segments=[VerboseSegment(text="Translated", start=0.0, end=1.0)],
        )
        client = FakeSTTClient(translate_response=translate_response)
        file_obj: BinaryIO = io.BytesIO(b"fake")
        result = client.process(file=file_obj, task="translate")
        assert result["text"] == "Translated"


class TestFakeAudioChunker:
    """Tests for FakeAudioChunker class."""

    def test_init_no_chunks(self) -> None:
        """Initialize without predefined chunks."""
        chunker = FakeAudioChunker()
        assert chunker._chunks is None

    def test_init_with_chunks(self) -> None:
        """Initialize with predefined chunks."""
        chunks = [
            AudioChunk(
                path="/tmp/chunk.mp3",
                start_seconds=0.0,
                duration_seconds=30.0,
                size_bytes=1000,
            )
        ]
        chunker = FakeAudioChunker(chunks=chunks)
        assert chunker._chunks == chunks

    def test_chunk_audio_returns_predefined(self) -> None:
        """Return predefined chunks."""
        chunks = [
            AudioChunk(
                path="/tmp/chunk1.mp3",
                start_seconds=0.0,
                duration_seconds=30.0,
                size_bytes=1000,
            ),
            AudioChunk(
                path="/tmp/chunk2.mp3",
                start_seconds=30.0,
                duration_seconds=30.0,
                size_bytes=1000,
            ),
        ]
        chunker = FakeAudioChunker(chunks=chunks)
        result = chunker.chunk_audio("/any/path.mp3", 60.0, 10.0)
        assert result == chunks

    def test_chunk_audio_returns_passthrough(self, tmp_path: Path) -> None:
        """Return passthrough chunk when no predefined chunks."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"x" * 500)

        chunker = FakeAudioChunker()
        result = chunker.chunk_audio(str(audio_file), 60.0, 10.0)

        assert len(result) == 1
        assert result[0]["path"] == str(audio_file)
        assert result[0]["start_seconds"] == 0.0
        assert result[0]["duration_seconds"] == 60.0
        assert result[0]["size_bytes"] == 500

    def test_chunk_audio_handles_missing_file(self) -> None:
        """Handle missing file gracefully."""
        chunker = FakeAudioChunker()
        result = chunker.chunk_audio("/nonexistent/file.mp3", 60.0, 10.0)

        assert len(result) == 1
        assert result[0]["size_bytes"] == 0


class TestFakeLangIdModel:
    """Tests for FakeLangIdModel class."""

    def test_init_defaults(self) -> None:
        """Initialize with defaults."""
        model = FakeLangIdModel()
        assert model._label == "__label__en"
        assert model._confidence == 0.99

    def test_init_custom(self) -> None:
        """Initialize with custom values."""
        model = FakeLangIdModel(label="__label__vi", confidence=0.95)
        assert model._label == "__label__vi"
        assert model._confidence == 0.95

    def test_predict_returns_configured(self) -> None:
        """Predict returns configured label and confidence."""
        model = FakeLangIdModel(label="__label__vie_Latn", confidence=0.88)
        labels, probs = model.predict("some text", k=1)

        assert labels == ("__label__vie_Latn",)
        assert len(probs) == 1
        # Use item() to get typed scalar from numpy array
        prob_value: float = probs.item(0)
        assert prob_value == 0.88


class TestFakeSubprocessResult:
    """Tests for FakeSubprocessResult class."""

    def test_init_defaults(self) -> None:
        """Initialize with defaults."""
        result = FakeSubprocessResult()
        assert result.returncode == 0
        assert result.stdout is None
        assert result.stderr is None

    def test_init_custom(self) -> None:
        """Initialize with custom values."""
        result = FakeSubprocessResult(
            returncode=1,
            stdout="output",
            stderr="error",
        )
        assert result.returncode == 1
        assert result.stdout == "output"
        assert result.stderr == "error"


class TestFakeSubprocessRun:
    """Tests for FakeSubprocessRun class."""

    def test_init_default_result(self) -> None:
        """Initialize with default result."""
        runner = FakeSubprocessRun()
        assert len(runner.calls) == 0

    def test_init_custom_result(self) -> None:
        """Initialize with custom result."""
        result = FakeSubprocessResult(returncode=2)
        runner = FakeSubprocessRun(result)
        output = runner(["ls"])
        assert output.returncode == 2

    def test_records_calls(self) -> None:
        """Record all calls."""
        runner = FakeSubprocessRun()
        runner(["cmd1", "arg1"])
        runner(["cmd2", "arg2"])

        assert len(runner.calls) == 2
        assert runner.calls[0] == ["cmd1", "arg1"]
        assert runner.calls[1] == ["cmd2", "arg2"]

    def test_raises_on_check_failure(self) -> None:
        """Raise CalledProcessError when check=True and returncode != 0."""
        result = FakeSubprocessResult(returncode=1, stdout="out", stderr="err")
        runner = FakeSubprocessRun(result)

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            runner(["failing_cmd"], check=True)

        assert exc_info.value.returncode == 1

    def test_no_raise_without_check(self) -> None:
        """Don't raise when check=False."""
        result = FakeSubprocessResult(returncode=1)
        runner = FakeSubprocessRun(result)
        output = runner(["failing_cmd"], check=False)
        assert output.returncode == 1


class TestHookManagement:
    """Tests for hook management functions."""

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_reset_hooks_restores_defaults(self) -> None:
        """Reset hooks restores production implementations."""
        # Modify a hook
        fake_runner = FakeSubprocessRun()
        _test_hooks.subprocess_run = fake_runner

        reset_hooks()

        # Should be back to default
        assert _test_hooks.subprocess_run == _test_hooks._default_subprocess_run

    def test_set_production_hooks(self) -> None:
        """Set production hooks sets all defaults."""
        # Modify hooks
        fake_runner = FakeSubprocessRun()
        _test_hooks.subprocess_run = fake_runner
        _test_hooks.ffmpeg_available = lambda: False

        set_production_hooks()

        assert _test_hooks.subprocess_run == _test_hooks._default_subprocess_run
        assert _test_hooks.ffmpeg_available == _test_hooks._default_ffmpeg_available

    def test_make_fake_subprocess_run(self) -> None:
        """Make and install fake subprocess runner."""
        result = FakeSubprocessResult(returncode=42)
        fake = make_fake_subprocess_run(result)

        assert _test_hooks.subprocess_run is fake
        output = _test_hooks.subprocess_run(["test"])
        assert output.returncode == 42

    def test_make_fake_audio_chunker_factory(self) -> None:
        """Make fake audio chunker factory."""
        chunks = [
            AudioChunk(
                path="/tmp/test.mp3",
                start_seconds=0.0,
                duration_seconds=60.0,
                size_bytes=1000,
            )
        ]
        factory = make_fake_audio_chunker_factory(chunks)
        chunker = factory(
            target_chunk_mb=20.0,
            max_chunk_duration_seconds=600.0,
            silence_threshold_db=-40.0,
            silence_duration_seconds=0.5,
        )

        result = chunker.chunk_audio("/any/path.mp3", 60.0, 10.0)
        assert result == chunks

    def test_make_fake_langid_model_factory(self) -> None:
        """Make fake language ID model factory."""
        factory = make_fake_langid_model_factory(
            label="__label__vie_Latn",
            confidence=0.85,
        )
        model = factory(model_path="/any/path.bin")
        labels, probs = model.predict("test")

        assert labels == ("__label__vie_Latn",)
        # Use item() to get typed scalar from numpy array
        prob_value: float = probs.item(0)
        assert prob_value == 0.85
