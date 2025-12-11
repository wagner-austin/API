"""Tests for langid module."""

from __future__ import annotations

import types
from collections.abc import Generator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tests.conftest import make_probs
from turkic_api import _test_hooks
from turkic_api.core import langid as lid


def test_ensure_model_path_218e_downloads_when_missing(tmp_path: Path) -> None:
    orig_download = _test_hooks.langid_download
    calls: list[str] = []

    def _fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"bin")

    _test_hooks.langid_download = _fake_download
    try:
        out = lid.ensure_model_path(str(tmp_path), prefer_218e=True)
        assert out.name == "lid218e.bin"
        assert out.exists()
        assert calls != []
        assert "lid218e" in calls[0]
    finally:
        _test_hooks.langid_download = orig_download


def test_ensure_model_path_218e_existing_no_download(tmp_path: Path) -> None:
    orig_download = _test_hooks.langid_download
    p = tmp_path / "models" / "lid218e.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"bin")

    def _boom(url: str, dest: Path) -> None:
        raise AssertionError("download should not be called")

    _test_hooks.langid_download = _boom
    try:
        out = lid.ensure_model_path(str(tmp_path), prefer_218e=True)
        assert out == p
    finally:
        _test_hooks.langid_download = orig_download


def test_ensure_model_path_176_branch(tmp_path: Path) -> None:
    orig_download = _test_hooks.langid_download
    calls: list[str] = []

    def _fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"bin")

    _test_hooks.langid_download = _fake_download
    try:
        out = lid.ensure_model_path(str(tmp_path), prefer_218e=False)
        assert out.name == "lid.176.bin"
        assert calls != []
        assert "lid.176" in calls[0]

        # Call again after file exists: should not download
        calls.clear()
        out2 = lid.ensure_model_path(str(tmp_path), prefer_218e=False)
        assert out2 == out
        assert calls == []
    finally:
        _test_hooks.langid_download = orig_download


def test_build_lang_filter_with_threshold(tmp_path: Path) -> None:
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            # Return variants to hit mapping logic: __label__kk and kaz_Cyrl
            if "cyrl" in text.lower():
                return (("__label__kaz_Cyrl",), make_probs(0.95))
            return (("__label__kk",), make_probs(0.80))

    model = _Model()
    keep = lid.build_lang_filter(target_lang="kk", threshold=0.90, model=model)
    assert keep("foo cyrl") is True  # kaz_Cyrl maps to kk
    assert keep("bar") is False  # prob below threshold


def test_build_lang_script_filter_match_and_mismatch(tmp_path: Path) -> None:
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            t = text.lower()
            if "latn" in t:
                return (("__label__kaz_Latn",), make_probs(0.99))
            if "cyrl" in t:
                return (("__label__kaz_Cyrl",), make_probs(0.99))
            return (("__label__eng",), make_probs(0.99))

    model = _Model()
    # Script normalized from lower-case
    keep = lid.build_lang_script_filter(target_lang="kk", script="latn", threshold=0.5, model=model)
    assert keep("text latn") is True
    assert keep("text cyrl") is False  # script mismatch -> return False
    # Lang mismatch -> return False
    assert keep("english") is False
    # No script filter
    keep2 = lid.build_lang_script_filter(target_lang="kk", script=None, threshold=0.5, model=model)
    assert keep2("text latn") is True


def test_build_lang_script_filter_blank_script_treated_as_none(tmp_path: Path) -> None:
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__kaz_Latn",), make_probs(0.99))

    model = _Model()
    keep = lid.build_lang_script_filter(target_lang="kk", script="   ", threshold=0.5, model=model)
    # Blank script should be treated as None (no script gating)
    assert keep("anything") is True


def test_get_fasttext_model_factory() -> None:
    """Test that _get_fasttext_model_factory returns the _FastText class via hook."""
    orig_factory = _test_hooks.langid_get_fasttext_factory

    class _FakeModel:
        def __init__(self, *, model_path: str) -> None:
            self._model_path = model_path

        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__en",), make_probs(0.99))

    def _fake_make(*, model_path: str) -> _FakeModel:
        return _FakeModel(model_path=model_path)

    def _fake_factory() -> _test_hooks.LangIdModelFactoryProtocol:
        return _fake_make

    _test_hooks.langid_get_fasttext_factory = _fake_factory
    try:
        factory = lid._get_fasttext_model_factory()
        model = factory(model_path="/fake/path.bin")
        # Verify factory works by calling predict
        labels, _probs = model.predict("test", k=1)
        assert labels == ("__label__en",)
    finally:
        _test_hooks.langid_get_fasttext_factory = orig_factory


def test_extract_prob_empty_array() -> None:
    """Test _extract_prob returns 0.0 for empty arrays."""
    empty_list: list[np.float64] = []
    empty_probs: NDArray[np.float64] = np.array(empty_list, dtype=np.float64)
    assert lid._extract_prob(empty_probs) == 0.0


def test_extract_prob_with_value() -> None:
    """Test _extract_prob returns the first value for non-empty arrays."""
    assert lid._extract_prob(make_probs(0.95, 0.05)) == 0.95


def test_langid_download_writes_nonempty_chunks(tmp_path: Path) -> None:
    """Test the default langid_download implementation writes only non-empty chunks."""
    orig_download = _test_hooks.langid_download
    dest = tmp_path / "models" / "x.bin"

    # Create a fake requests response that yields mixed empty/non-empty chunks
    class _FakeResponse:
        def __init__(self) -> None:
            self._chunks = [b"", b"abc", b"", b"def"]

        def iter_content(self, chunk_size: int = 8192) -> Generator[bytes, None, None]:
            yield from self._chunks

        def raise_for_status(self) -> None:
            return None

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None,
        ) -> None:
            return None

    def _fake_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = _FakeResponse()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    _test_hooks.langid_download = _fake_download
    try:
        _test_hooks.langid_download("http://example/x", dest)
        assert dest.read_bytes() == b"abcdef"
    finally:
        _test_hooks.langid_download = orig_download
