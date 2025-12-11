"""Tests for _test_hooks default implementations and corpus_download."""

from __future__ import annotations

from pathlib import Path

from data_bank_api import _test_hooks
from data_bank_api.api.config import JobParams
from data_bank_api.core.corpus_download import ensure_corpus_file


def test_default_local_corpus_factory_creates_functional_service(tmp_path: Path) -> None:
    """_default_local_corpus_factory creates a service that streams corpus lines."""
    # Create a corpus file to verify the service actually reads it
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "test_en.txt"
    corpus_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    service = _test_hooks._default_local_corpus_factory(str(tmp_path))
    params: JobParams = {
        "source": "test",
        "language": "en",
        "max_sentences": 0,
        "transliterate": False,
        "confidence_threshold": 0.0,
    }
    # Verify actual behavior - streaming lines from the corpus file
    result = list(service.stream(params))
    assert result == ["line1", "line2", "line3"]


def test_default_data_bank_uploader_factory_creates_functional_client() -> None:
    """_default_data_bank_uploader_factory creates a client that can be used."""
    client = _test_hooks._default_data_bank_uploader_factory(
        "http://test-server:8000", "test-api-key", timeout_seconds=30.0
    )
    # Verify the factory returns something with the expected upload method
    # by checking it's callable
    assert callable(client.upload)


def test_ensure_corpus_file_is_noop() -> None:
    """ensure_corpus_file in core/corpus_download.py is a no-op."""
    # Should complete without error - don't capture return value since it's None
    ensure_corpus_file(
        source="test",
        language="en",
        data_dir="/tmp",
        max_sentences=100,
        transliterate=False,
        confidence_threshold=0.9,
    )
    # If we get here without error, the function completed successfully
