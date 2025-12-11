"""Tests for corpus_download module."""

from __future__ import annotations

import io
import sys
import types
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_core.json_utils import JSONValue

from tests.conftest import make_probs
from turkic_api import _test_hooks
from turkic_api.core.corpus_download import (
    _stream_for_source,
    ensure_corpus_file,
    stream_culturax,
    stream_oscar,
    stream_wikipedia_xml,
)
from turkic_api.core.models import ProcessSpec


def _gen_lines(lines: list[str]) -> Generator[str, None, None]:
    yield from lines


@pytest.fixture(autouse=True)
def _datasets_stub() -> Generator[None, None, None]:
    """Stub datasets module for tests that use HF datasets."""
    # Placeholder to ensure datasets cleanup after test
    yield
    if "datasets" in sys.modules:
        del sys.modules["datasets"]


def test_ensure_corpus_file_writes_and_is_idempotent(tmp_path: Path) -> None:
    # Stub stream_oscar to avoid network
    orig = _test_hooks.stream_oscar_hook

    def _oscar_stub(_lang: str) -> Generator[str, None, None]:
        return _gen_lines(["a", "b", "", "c"])

    _test_hooks.stream_oscar_hook = _oscar_stub
    try:
        spec = ProcessSpec(
            source="oscar",
            language="kk",
            max_sentences=2,
            transliterate=True,
            confidence_threshold=0.0,
        )
        path = ensure_corpus_file(spec, str(tmp_path))
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip().splitlines() == ["a", "b"]
        # Second call should early-return without rewriting
        same = ensure_corpus_file(spec, str(tmp_path))
        assert same == path
    finally:
        _test_hooks.stream_oscar_hook = orig


def test_ensure_corpus_file_zero_written_raises(tmp_path: Path) -> None:
    # Stub wikipedia stream to yield no lines
    orig = _test_hooks.stream_wikipedia_xml_hook

    def _wiki_stub(_lang: str) -> Generator[str, None, None]:
        return _gen_lines([])

    _test_hooks.stream_wikipedia_xml_hook = _wiki_stub
    try:
        spec = ProcessSpec(
            source="wikipedia",
            language="kk",
            max_sentences=10,
            transliterate=True,
            confidence_threshold=0.0,
        )
        with pytest.raises(RuntimeError, match="No sentences"):
            ensure_corpus_file(spec, str(tmp_path))
        out = tmp_path / "corpus" / "wikipedia_kk.txt"
        assert not out.exists()
    finally:
        _test_hooks.stream_wikipedia_xml_hook = orig


def test_ensure_corpus_file_zero_written_logs_unlink_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    orig_wiki = _test_hooks.stream_wikipedia_xml_hook
    orig_unlink = _test_hooks.path_unlink

    # Stub wikipedia stream to yield no lines so ensure_corpus_file triggers unlink path
    def _wiki_stub(_lang: str) -> Generator[str, None, None]:
        return _gen_lines([])

    _test_hooks.stream_wikipedia_xml_hook = _wiki_stub

    # Force path_unlink to raise so we exercise the warning branch
    def _raise_unlink(_path: Path) -> None:
        raise OSError("boom")

    _test_hooks.path_unlink = _raise_unlink

    try:
        spec = ProcessSpec(
            source="wikipedia",
            language="kk",
            max_sentences=10,
            transliterate=True,
            confidence_threshold=0.0,
        )

        with (
            caplog.at_level("WARNING"),
            pytest.raises(RuntimeError, match="No sentences were written"),
        ):
            ensure_corpus_file(spec, str(tmp_path))
        # Validate that the warning branch ran
        assert any("corpus_zero_unlink_failed" in r.getMessage() for r in caplog.records)
    finally:
        _test_hooks.stream_wikipedia_xml_hook = orig_wiki
        _test_hooks.path_unlink = orig_unlink


class _FakeRaw:
    """Fake raw stream that satisfies RawStreamProtocol."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def read(self, n: int, /) -> bytes:
        return self._buf.read(n)

    def seekable(self) -> bool:
        return self._buf.seekable()

    def seek(self, n: int, /) -> int:
        return self._buf.seek(n)


class _FakeWikipediaResponse:
    """Fake Wikipedia response that satisfies WikipediaRequestsResponseProtocol."""

    def __init__(self, compressed: bytes) -> None:
        self._raw = _FakeRaw(compressed)

    @property
    def raw(self) -> _test_hooks.RawStreamProtocol:
        return self._raw

    def raise_for_status(self) -> None:
        return None

    def __enter__(self) -> _test_hooks.WikipediaRequestsResponseProtocol:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        return None


def test_stream_wikipedia_xml_parses_sentences() -> None:
    # Create a minimal XML with <text> content and compress via bz2
    xml = b"<page><revision><text>One. Two! Three?</text></revision></page>"
    import bz2

    compressed = bz2.compress(xml)

    # Patch the request getter via hook
    orig = _test_hooks.wikipedia_requests_get

    def fake_get(
        url: str, *, stream: bool, timeout: int
    ) -> _test_hooks.WikipediaRequestsResponseProtocol:
        return _FakeWikipediaResponse(compressed)

    _test_hooks.wikipedia_requests_get = fake_get
    try:
        out = list(stream_wikipedia_xml("kk"))
        # Empty splits removed; punctuation-split applied
        assert out == ["One", "Two", "Three"]
    finally:
        _test_hooks.wikipedia_requests_get = orig


def test_stream_oscar_uses_datasets() -> None:
    # Provide a dummy datasets module with load_dataset
    class _DS:
        def __iter__(self) -> Generator[dict[str, str | int] | int, None, None]:
            yield {"text": "x"}
            yield {"text": "  y  "}
            yield {"text": ""}
            yield {"text": 99}
            yield 123  # non-dict row to cover branch

    class _Mod(ModuleType):
        @staticmethod
        def load_dataset(*_a: JSONValue, **_k: JSONValue) -> _DS:
            return _DS()

    sys.modules["datasets"] = _Mod("datasets")
    out = list(stream_oscar("kk"))
    assert out == ["x", "y"]


def test_ensure_corpus_file_applies_lang_filter(tmp_path: Path) -> None:
    # Stub stream to emit mixed sentences
    orig = _test_hooks.stream_oscar_hook

    def _oscar_mixed(_lang: str) -> Generator[str, None, None]:
        return _gen_lines(["keep one", "drop x", "keep two"])

    _test_hooks.stream_oscar_hook = _oscar_mixed

    # Provide a dummy LangId model and use the public filter builder
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            label = "__label__kk" if "keep" in text else "__label__eng"
            return ((label,), make_probs(1.0))

    try:
        spec = ProcessSpec(
            source="oscar",
            language="kk",
            max_sentences=10,
            transliterate=True,
            confidence_threshold=0.9,
        )
        path = ensure_corpus_file(spec, str(tmp_path), langid_model=_Model())
        assert path.exists()
        assert path.read_text(encoding="utf-8").splitlines() == ["keep one", "keep two"]
    finally:
        _test_hooks.stream_oscar_hook = orig


def test_ensure_corpus_file_applies_script_filter(tmp_path: Path) -> None:
    # Stub stream to emit mixed sentences
    orig = _test_hooks.stream_oscar_hook

    def _oscar_scripts(_lang: str) -> Generator[str, None, None]:
        return _gen_lines(["CYRL hello", "LATN world", "LATN again"])

    _test_hooks.stream_oscar_hook = _oscar_scripts

    # Keep only 'Latn' script sentences
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            t = text.strip()
            if t.startswith("LATN "):
                return (("__label__kaz_Latn",), make_probs(1.0))
            if t.startswith("CYRL "):
                return (("__label__kaz_Cyrl",), make_probs(1.0))
            return (("__label__eng",), make_probs(1.0))

    try:
        spec = ProcessSpec(
            source="oscar",
            language="kk",
            max_sentences=10,
            transliterate=True,
            confidence_threshold=0.0,
        )
        path = ensure_corpus_file(spec, str(tmp_path), script="Latn", langid_model=_Model())
        assert path.exists()
        assert path.read_text(encoding="utf-8").splitlines() == ["LATN world", "LATN again"]
    finally:
        _test_hooks.stream_oscar_hook = orig


def test_stream_for_source_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported corpus source"):
        _stream_for_source("news", "kk")


def test_stream_culturax_uses_datasets() -> None:
    # Provide a dummy datasets module with load_dataset
    class _DS:
        def __iter__(self) -> Generator[dict[str, str | int] | int, None, None]:
            yield {"text": "culturax1"}
            yield {"text": "  culturax2  "}
            yield {"text": ""}
            yield {"text": 99}
            yield 123  # non-dict row to cover branch

    class _Mod(ModuleType):
        @staticmethod
        def load_dataset(*_a: JSONValue, **_k: JSONValue) -> _DS:
            return _DS()

    sys.modules["datasets"] = _Mod("datasets")
    out = list(stream_culturax("kk"))
    assert out == ["culturax1", "culturax2"]


def test_stream_for_source_culturax() -> None:
    # Stub stream_culturax to avoid network
    orig = _test_hooks.stream_culturax_hook

    def _culturax_stub(_lang: str) -> Generator[str, None, None]:
        return _gen_lines(["a", "b"])

    _test_hooks.stream_culturax_hook = _culturax_stub
    try:
        gen = _stream_for_source("culturax", "kk")
        assert list(gen) == ["a", "b"]
    finally:
        _test_hooks.stream_culturax_hook = orig
