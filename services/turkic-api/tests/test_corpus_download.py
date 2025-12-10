from __future__ import annotations

import io
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.conftest import make_probs

from turkic_api.core.corpus_download import (
    _stream_for_source,
    ensure_corpus_file,
    stream_culturax,
    stream_oscar,
    stream_wikipedia_xml,
)
from turkic_api.core.models import ProcessSpec, UnknownJson


def _gen_lines(lines: list[str]) -> Generator[str, None, None]:
    yield from lines


def test_ensure_corpus_file_writes_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stub stream_oscar to avoid network
    def _oscar_stub(_lang: str) -> Generator[str]:
        return _gen_lines(["a", "b", "", "c"])

    monkeypatch.setattr("turkic_api.core.corpus_download.stream_oscar", _oscar_stub)
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


def test_ensure_corpus_file_zero_written_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stub wikipedia stream to yield no lines
    def _wiki_stub(_lang: str) -> Generator[str]:
        return _gen_lines([])

    monkeypatch.setattr("turkic_api.core.corpus_download.stream_wikipedia_xml", _wiki_stub)
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


def test_ensure_corpus_file_zero_written_logs_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Stub wikipedia stream to yield no lines so ensure_corpus_file triggers unlink path
    def _wiki_stub(_lang: str) -> Generator[str]:
        return _gen_lines([])

    monkeypatch.setattr("turkic_api.core.corpus_download.stream_wikipedia_xml", _wiki_stub)

    # Force Path.unlink to raise so we exercise the warning branch
    def _raise_unlink(self: Path, *args: UnknownJson, **kwargs: UnknownJson) -> None:
        raise OSError("boom")

    monkeypatch.setattr(Path, "unlink", _raise_unlink, raising=True)

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


def test_stream_wikipedia_xml_parses_sentences(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a minimal XML with <text> content and compress via bz2
    xml = b"<page><revision><text>One. Two! Three?</text></revision></page>"
    import bz2

    raw = io.BytesIO(bz2.compress(xml))

    class _Resp:
        def __init__(self) -> None:
            self.raw = raw

        def raise_for_status(self) -> None:
            return None

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_: UnknownJson) -> None:
            return None

    # Patch the request getter used by the module (requests.get in module namespace)
    import turkic_api.core.corpus_download as cd

    class _R:
        @staticmethod
        def get(url: str, *, stream: bool, timeout: int) -> _Resp:
            return _Resp()

    monkeypatch.setattr(cd, "requests", _R())

    out = list(stream_wikipedia_xml("kk"))
    # Empty splits removed; punctuation-split applied
    assert out == ["One", "Two", "Three"]


def test_stream_oscar_uses_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide a dummy datasets module with load_dataset
    class _DS:
        def __iter__(self) -> Generator[dict[str, str | int] | int]:
            yield {"text": "x"}
            yield {"text": "  y  "}
            yield {"text": ""}
            yield {"text": 99}
            yield 123  # non-dict row to cover branch

    class _Mod(ModuleType):
        @staticmethod
        def load_dataset(*_a: UnknownJson, **_k: UnknownJson) -> _DS:
            return _DS()

    sys.modules["datasets"] = _Mod("datasets")
    try:
        out = list(stream_oscar("kk"))
        assert out == ["x", "y"]
    finally:
        del sys.modules["datasets"]


def test_ensure_corpus_file_applies_lang_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stub stream to emit mixed sentences
    def _oscar_mixed(_lang: str) -> Generator[str]:
        return _gen_lines(["keep one", "drop x", "keep two"])

    monkeypatch.setattr("turkic_api.core.corpus_download.stream_oscar", _oscar_mixed)

    # Provide a dummy LangId model and use the public filter builder
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            label = "__label__kk" if "keep" in text else "__label__eng"
            return ((label,), make_probs(1.0))

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


def test_ensure_corpus_file_applies_script_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stub stream to emit mixed sentences
    def _oscar_scripts(_lang: str) -> Generator[str]:
        return _gen_lines(["CYRL hello", "LATN world", "LATN again"])

    monkeypatch.setattr("turkic_api.core.corpus_download.stream_oscar", _oscar_scripts)

    # Keep only 'Latn' script sentences
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            t = text.strip()
            if t.startswith("LATN "):
                return (("__label__kaz_Latn",), make_probs(1.0))
            if t.startswith("CYRL "):
                return (("__label__kaz_Cyrl",), make_probs(1.0))
            return (("__label__eng",), make_probs(1.0))

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


def test_stream_for_source_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported corpus source"):
        _stream_for_source("news", "kk")


def test_stream_culturax_uses_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide a dummy datasets module with load_dataset
    class _DS:
        def __iter__(self) -> Generator[dict[str, str | int] | int]:
            yield {"text": "culturax1"}
            yield {"text": "  culturax2  "}
            yield {"text": ""}
            yield {"text": 99}
            yield 123  # non-dict row to cover branch

    class _Mod(ModuleType):
        @staticmethod
        def load_dataset(*_a: UnknownJson, **_k: UnknownJson) -> _DS:
            return _DS()

    sys.modules["datasets"] = _Mod("datasets")
    try:
        out = list(stream_culturax("kk"))
        assert out == ["culturax1", "culturax2"]
    finally:
        del sys.modules["datasets"]


def test_stream_for_source_culturax(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub stream_culturax to avoid network
    def _culturax_stub(_lang: str) -> Generator[str]:
        return _gen_lines(["a", "b"])

    monkeypatch.setattr("turkic_api.core.corpus_download.stream_culturax", _culturax_stub)
    gen = _stream_for_source("culturax", "kk")
    assert list(gen) == ["a", "b"]
